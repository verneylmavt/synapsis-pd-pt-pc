from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.core.config import Settings
from app.db.models import Area, Detection, Event, FrameSample, ProcessingRun, VideoSource
from app.services.supervisor import RunManager
from tests.fake_worker import RUN_SETTINGS, blocked_worker, eof_worker, quiet_worker


POLYGON = [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1},
           {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9}]


def source(factory, name="test"):
    with factory.begin() as db:
        row = VideoSource(name=name, uri="fake.mp4", kind="file", enabled=True)
        db.add(row)
        db.flush()
        db.add(Area(video_source_id=row.id, name="zone", polygon=POLYGON, active=True))
        return row.id


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Expected lifecycle transition did not complete")


@pytest.fixture
def manager_factory(db_engine, db_factory, tmp_path):
    (tmp_path / "yolov8n.pt").write_bytes(b"fake local pretrained weights")
    config = replace(Settings.from_env({}, load_file=False), model_dir=tmp_path,
                     stop_seconds=0.3, stale_seconds=5)
    managers = []
    def make(worker=quiet_worker, open_manager=True):
        manager = RunManager(config, db_engine, db_factory, worker_target=worker)
        managers.append(manager)
        if open_manager:
            manager.open()
        return manager
    yield make
    for manager in reversed(managers):
        manager.close()


def test_matching_concurrent_starts_share_one_producer_and_settings_changes_are_rejected(manager_factory, db_factory):
    manager = manager_factory()
    source_id = source(db_factory)
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: manager.start(source_id, dict(RUN_SETTINGS)), range(4)))
    assert len(set(runs)) == 1
    assert len(manager.runtimes) == 1
    with pytest.raises(ValueError, match="already active"):
        manager.start(source_id, {**RUN_SETTINGS, "conf": 0.1})
    with pytest.raises(ValueError, match="already active"):
        manager.start(source(db_factory, "another source"), dict(RUN_SETTINGS))
    with db_factory() as db:
        assert db.scalar(select(func.count()).select_from(ProcessingRun)) == 1


def test_two_viewers_share_durable_quiet_frames_and_disconnect_keeps_producer_alive(manager_factory, db_factory):
    manager = manager_factory()
    run_id = manager.start(source(db_factory), dict(RUN_SETTINGS))
    runtime = manager.runtimes[run_id]
    wait_for(lambda: runtime.sequence > 0)
    first, second = manager.subscribe(run_id), manager.subscribe(run_id)
    try:
        assert b"fake-jpeg" in next(first)
        assert b"fake-jpeg" in next(second)
        assert runtime.viewers == 2
    finally:
        first.close()
        second.close()
    assert runtime.viewers == 0 and runtime.process.is_alive()
    previous = runtime.sequence
    wait_for(lambda: runtime.sequence > previous)
    with db_factory() as db:
        run = db.get(ProcessingRun, run_id)
        assert run.status == "running" and run.heartbeat_at is not None
        assert run.last_frame_index > 0
        assert db.scalar(select(func.count()).select_from(Event)) == 0
        assert db.scalar(select(func.count()).select_from(Detection)) == 0
        samples = select(func.count()).select_from(FrameSample).where(
            FrameSample.run_id == ProcessingRun.id).scalar_subquery()
        progress, sample_count = db.execute(select(ProcessingRun.last_frame_index, samples).where(
            ProcessingRun.id == run_id)).one()
        assert sample_count == progress


def test_eof_flushes_quiet_frames_and_closes_subscriptions(manager_factory, db_factory):
    manager = manager_factory(eof_worker)
    run_id = manager.start(source(db_factory), dict(RUN_SETTINGS))
    runtime = manager.runtimes[run_id]
    wait_for(lambda: runtime.terminal)
    with db_factory() as db:
        run = db.get(ProcessingRun, run_id)
        assert run.status == "completed" and run.ended_at is not None
        assert run.last_frame_index == 3
        assert db.scalar(select(func.count()).select_from(FrameSample)) == 3
    subscription = manager.subscribe(run_id)
    assert b"fake-jpeg" in next(subscription)
    with pytest.raises(StopIteration):
        next(subscription)
    assert runtime.viewers == 0 and not runtime.process.is_alive()


def test_stop_terminates_noncooperative_child_within_bounded_deadline(manager_factory, db_factory):
    manager = manager_factory(blocked_worker)
    run_id = manager.start(source(db_factory), dict(RUN_SETTINGS))
    runtime = manager.runtimes[run_id]
    wait_for(lambda: runtime.sequence > 0)
    started = time.monotonic()
    manager.stop_run(run_id)
    manager.stop_run(run_id)
    wait_for(lambda: runtime.terminal, timeout=3)
    assert time.monotonic() - started < 2.5
    assert not runtime.process.is_alive()
    with db_factory() as db:
        assert db.get(ProcessingRun, run_id).status == "stopped"


def test_ownership_lock_blocks_second_manager_and_restart_marks_orphans_interrupted(manager_factory, db_factory):
    first = manager_factory()
    second = manager_factory(open_manager=False)
    with pytest.raises(RuntimeError, match="Another application owns"):
        second.open()
    first.close()
    source_id = source(db_factory)
    orphan_id = str(uuid4())
    with db_factory.begin() as db:
        db.add(ProcessingRun(id=orphan_id, video_source_id=source_id, kind="file", status="running",
                             owner_id=str(uuid4()), settings=dict(RUN_SETTINGS), areas=[]))
    second.open()
    with db_factory() as db:
        orphan = db.get(ProcessingRun, orphan_id)
        assert orphan.status == "interrupted" and orphan.ended_at is not None
        assert orphan.error_code == "application_restarted"


def test_viewer_limit_does_not_start_extra_producers(manager_factory, db_factory):
    manager = manager_factory()
    run_id = manager.start(source(db_factory), dict(RUN_SETTINGS))
    subscriptions = [manager.subscribe(run_id) for _ in range(manager.config.max_viewers)]
    try:
        with pytest.raises(ValueError, match="viewer limit"):
            manager.subscribe(run_id)
        assert len(manager.runtimes) == 1
    finally:
        for subscription in subscriptions:
            subscription.close()


def test_invalidated_ownership_connection_cannot_start_a_new_producer(manager_factory, db_factory):
    manager = manager_factory()
    source_id = source(db_factory)
    manager.connection.invalidate()
    manager.connection.rollback()
    # SQLAlchemy can reconnect for SELECT 1, but the advisory lock is session-scoped.
    with pytest.raises((ValueError, RuntimeError)):
        manager.start(source_id, dict(RUN_SETTINGS))
    assert manager.runtimes == {}
