from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import queue
import threading
import time
from uuid import uuid4

from sqlalchemy import select, text, update

from app.db.models import ACTIVE_STATUSES, Area, ProcessingRun, VideoSource
from app.services.persistence import persist_batch, utcnow

OWNERSHIP_KEY = 739_021_418


def _worker(spec, output, stop):
    from app.pipeline.worker import worker_main
    worker_main(spec, output, stop)


@dataclass
class Runtime:
    id: str
    source_id: int
    process: object
    output: object
    stop: object
    condition: threading.Condition = field(default_factory=threading.Condition)
    sequence: int = 0
    jpeg: bytes | None = None
    terminal: bool = False
    viewers: int = 0
    stop_at: float | None = None
    thread: threading.Thread | None = None
    process_lock: threading.Lock = field(default_factory=threading.Lock)
    guardian: threading.Thread | None = None
    snapshot: dict | None = None


class Subscription:
    def __init__(self, manager, runtime):
        self.manager, self.runtime = manager, runtime
        self.seen = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        runtime = self.runtime
        with runtime.condition:
            while not self.closed:
                if runtime.jpeg is not None and runtime.sequence > self.seen:
                    self.seen = runtime.sequence
                    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + runtime.jpeg + b"\r\n"
                if runtime.terminal:
                    break
                runtime.condition.wait(0.5)
        self.close()
        raise StopIteration

    def close(self):
        with self.manager.lock:
            if not self.closed:
                self.closed = True
                self.runtime.viewers -= 1
        with self.runtime.condition:
            self.runtime.condition.notify_all()


class RunManager:
    def __init__(self, config, engine, factory, worker_target=_worker):
        self.config, self.engine, self.factory = config, engine, factory
        self.worker_target = worker_target
        self.owner_id = str(uuid4())
        self.lock = threading.RLock()
        self.runtimes = {}
        self.previewing = set()
        self.connection = None
        self.closing = False
        self.context = mp.get_context("spawn")

    def mutation_guard(self):
        return self.lock

    def begin_preview(self, source_id):
        with self.lock:
            if source_id in self.previewing:
                raise ValueError("A preview is already being decoded")
            self.previewing.add(source_id)

    def end_preview(self, source_id):
        with self.lock:
            self.previewing.discard(source_id)

    def _check_ownership(self):
        connection = self.connection
        if connection is None or connection.invalidated:
            raise ValueError("Processing ownership was lost; restart the application")
        pid = connection.execute(text("SELECT pg_backend_pid()")).scalar()
        if pid != self.ownership_pid:
            raise ValueError("Processing ownership was lost; restart the application")

    def open(self):
        connection = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            if not connection.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": OWNERSHIP_KEY}).scalar():
                raise RuntimeError("Another application owns processing; run exactly one API process")
            self.connection = connection
            self.ownership_pid = connection.execute(text("SELECT pg_backend_pid()")).scalar()
            with self.factory.begin() as db:
                db.execute(update(ProcessingRun).where(ProcessingRun.status.in_(ACTIVE_STATUSES))
                    .values(status="interrupted", ended_at=utcnow(), error_code="application_restarted"))
        except Exception:
            if self.connection is not None:
                try:
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": OWNERSHIP_KEY})
                except Exception:
                    connection.invalidate()
            connection.close()
            self.connection = None
            raise

    def start(self, source_id, settings):
        with self.lock:
            if self.closing or self.connection is None:
                raise ValueError("Processing ownership is unavailable")
            if source_id in self.previewing:
                raise ValueError("Wait for the source preview before starting")
            self._check_ownership()
            with self.factory.begin() as db:
                source = db.get(VideoSource, source_id)
                if source is None or not source.enabled:
                    raise LookupError("Source not found or disabled")
                active = db.execute(select(ProcessingRun).where(ProcessingRun.status.in_(ACTIVE_STATUSES))).scalars().first()
                if active:
                    if active.video_source_id == source_id and active.settings == settings:
                        return active.id
                    raise ValueError("One run is already active; stop it before starting another")
                areas = [{"id": area.id, "name": area.name, "polygon": area.polygon}
                    for area in db.execute(select(Area).where(Area.video_source_id == source_id, Area.active.is_(True)).order_by(Area.id)).scalars()]
                if not areas:
                    raise ValueError("Save an active counting area before starting")
                model_path = self.config.model_dir / settings["model"]
                if not model_path.is_file():
                    raise FileNotFoundError("Selected local model weights are missing")
                run_id = str(uuid4())
                run = ProcessingRun(id=run_id, video_source_id=source_id, kind=source.kind,
                    owner_id=self.owner_id, status="starting", settings=settings, areas=areas)
                db.add(run)
                spec = {"run_id": run_id, "uri": source.uri, "kind": source.kind, "areas": areas,
                        "settings": {**settings, "model_path": str(model_path), "confirmation": 3,
                         "missing_grace_seconds": 2, "open_timeout_seconds": 5, "read_timeout_seconds": 5,
                         "startup_timeout_seconds": 30, "no_frame_timeout_seconds": 15}}
            output, stop = self.context.Queue(maxsize=2), self.context.Event()
            process = self.context.Process(target=self.worker_target, args=(spec, output, stop), daemon=True)
            runtime = Runtime(run_id, source_id, process, output, stop)
            runtime.snapshot = {column.name: getattr(run, column.name) for column in ProcessingRun.__table__.columns if column.name != "owner_id"}
            self.runtimes[run_id] = runtime
            try:
                process.start()
            except Exception:
                self._finish(runtime, "failed", "process_start_failed")
                output.close()
                raise ValueError("Could not start processing")
            runtime.thread = threading.Thread(target=self._monitor, args=(runtime,), daemon=True)
            runtime.thread.start()
            # Retain only a bounded number of final JPEGs in memory.
            for key, old in list(self.runtimes.items()):
                if len(self.runtimes) <= 16:
                    break
                if old.terminal and old.viewers == 0:
                    del self.runtimes[key]
            return run_id

    def stop_run(self, run_id):
        with self.lock:
            runtime = self.runtimes.get(run_id)
            if runtime and not runtime.terminal and runtime.stop_at is None:
                runtime.stop_at = time.monotonic()
                runtime.stop.set()
                if runtime.snapshot:
                    runtime.snapshot["status"] = "stopping"
                runtime.guardian = threading.Thread(target=self._stop_guardian, args=(runtime,), daemon=True)
                runtime.guardian.start()
                threading.Thread(target=self._mark_stopping, args=(runtime,), daemon=True).start()

    def _mark_stopping(self, runtime):
        try:
            with self.factory.begin() as db:
                db.execute(update(ProcessingRun).where(ProcessingRun.id == runtime.id,
                    ProcessingRun.status.in_(ACTIVE_STATUSES)).values(status="stopping"))
        except Exception:
            pass  # Control remains effective even when its status cannot be committed.

    def _reap(self, runtime, grace=0.2):
        with runtime.process_lock:
            runtime.process.join(timeout=grace)
            if runtime.process.is_alive():
                runtime.process.terminate()
                runtime.process.join(timeout=0.5)
            if runtime.process.is_alive():
                runtime.process.kill()
                runtime.process.join(timeout=0.5)

    def _stop_guardian(self, runtime):
        # Independent of monitor/database progress: enforce the accepted Stop deadline.
        self._reap(runtime, grace=self.config.stop_seconds)

    def publish(self, runtime, jpeg):
        with runtime.condition:
            runtime.jpeg = jpeg
            runtime.sequence += 1
            runtime.condition.notify_all()

    def subscribe(self, run_id):
        with self.lock:
            runtime = self.runtimes.get(run_id)
            if runtime is None:
                raise LookupError("Stream is not available for this run")
            if runtime.viewers >= self.config.max_viewers:
                raise ValueError("The viewer limit has been reached")
            runtime.viewers += 1
            return Subscription(self, runtime)

    def latest_source_jpeg(self, source_id):
        with self.lock:
            rows = [r for r in self.runtimes.values() if r.source_id == source_id and not r.terminal]
            return rows[-1].jpeg if rows else None

    def _finish(self, runtime, status, error=None):
        with self.factory.begin() as db:
            run = db.get(ProcessingRun, runtime.id)
            run.status, run.error_code, run.ended_at = status, error, utcnow()
        if runtime.snapshot:
            runtime.snapshot.update(status=status, error_code=error, ended_at=run.ended_at)
        with runtime.condition:
            runtime.terminal = True
            runtime.condition.notify_all()

    def _monitor(self, runtime):
        batch, rows = [], 0
        started = last_frame = last_flush = last_lock_check = time.monotonic()
        status, error = "failed", "worker_exited"
        def flush():
            nonlocal batch, rows, last_flush
            if batch:
                persist_batch(self.factory, runtime.id, runtime.source_id, batch, self.config.write_retries)
                if runtime.snapshot:
                    packet = batch[-1]
                    runtime.snapshot.update(last_frame_index=packet["frame_index"], segment_index=packet["segment_index"],
                        media_time_ms=packet["media_time_ms"], fps=packet["fps"], dropped_frames=packet["dropped_frames"],
                        heartbeat_at=packet["observed_at"], status="stopping" if runtime.stop_at else "running")
                self.publish(runtime, batch[-1]["jpeg"])
                batch, rows = [], 0
            last_flush = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if now - last_lock_check >= 2:
                    # Loss of this dedicated connection also loses the ownership lock.
                    with self.lock:
                        self._check_ownership()
                    last_lock_check = now
                if runtime.stop_at is not None and now - runtime.stop_at >= self.config.stop_seconds:
                    status, error = "stopped", None
                    break
                deadline = 30 if runtime.sequence == 0 else self.config.stale_seconds
                if now - last_frame >= deadline and runtime.stop_at is None:
                    status, error = "failed", "processing_timeout"
                    break
                try:
                    packet = runtime.output.get(timeout=0.1)
                except queue.Empty:
                    if not runtime.process.is_alive():
                        status = "stopped" if runtime.stop_at else "failed"
                        error = None if runtime.stop_at else "worker_exited"
                        break
                    if now - last_flush >= 0.5:
                        flush()
                    continue
                if packet["kind"] == "frame":
                    batch.append(packet)
                    rows += len(packet["detections"])
                    last_frame = time.monotonic()
                    if len(batch) >= 10 or rows >= 1000 or last_frame - last_flush >= 0.5:
                        flush()
                elif packet["status"] in {"completed", "stopped", "failed"}:
                    status, error = packet["status"], packet.get("error_code")
                    if runtime.stop_at:
                        status, error = "stopped", None
                    break
                elif packet["status"] == "reconnecting":
                    flush()
                    with self.factory.begin() as db:
                        run = db.get(ProcessingRun, runtime.id)
                        if run.status != "stopping":
                            run.status = "reconnecting"
                # Running heartbeat comes only from committed observed frames.
            flush()
        except Exception:
            status, error = "failed", "persistence_or_ownership_failed"
            runtime.stop.set()
        finally:
            runtime.stop.set()
            self._reap(runtime)
            try:
                self._finish(runtime, status, error)
            except Exception:
                # On outage, recovery marks the uncommitted terminal state interrupted.
                with runtime.condition:
                    runtime.terminal = True
                    runtime.condition.notify_all()
            runtime.output.close()

    def close(self):
        with self.lock:
            self.closing = True
            active = [r for r in self.runtimes.values() if not r.terminal]
            for runtime in active:
                try:
                    self.stop_run(runtime.id)
                except Exception:
                    runtime.stop_at = time.monotonic()
                    runtime.stop.set()
        for runtime in active:
            if runtime.guardian:
                runtime.guardian.join(timeout=self.config.stop_seconds + 1)
            self._reap(runtime, grace=0)
            runtime.thread.join(timeout=2)
        if self.connection is not None:
            try:
                self.connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": OWNERSHIP_KEY})
            except Exception:
                self.connection.invalidate()
            finally:
                self.connection.close()
                self.connection = None
