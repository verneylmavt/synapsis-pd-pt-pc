from contextlib import contextmanager
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError

from app.db.models import Area, Detection, Event, FrameSample, ProcessingRun, RunAreaStats, VideoSource
from app.services.persistence import persist_batch
from tests.fake_worker import RUN_SETTINGS, frame_packet


@pytest.fixture
def persisted_run(db_factory):
    with db_factory.begin() as db:
        source = VideoSource(name="persistence", uri="fake.mp4", kind="file", enabled=True)
        db.add(source)
        db.flush()
        area = Area(video_source_id=source.id, name="zone", active=True, polygon=[
            {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])
        db.add(area)
        db.flush()
        run_id = str(uuid4())
        db.add(ProcessingRun(id=run_id, video_source_id=source.id, kind="file", status="starting",
                             owner_id=str(uuid4()), settings=dict(RUN_SETTINGS), areas=[]))
        return run_id, source.id, {"run_id": run_id, "areas": [{"id": area.id}]}


def record_counts(factory):
    with factory() as db:
        return {table.__tablename__: db.scalar(select(func.count()).select_from(table))
                for table in (Detection, Event, FrameSample, RunAreaStats)}


def test_replayed_and_older_batches_do_not_duplicate_records_or_regress_absolute_counters(db_factory, persisted_run):
    run_id, source_id, spec = persisted_run
    first = frame_packet(spec, 1, total_in=1, detection=True, event="enter")
    second = frame_packet(spec, 2, total_in=1, total_out=1, detection=True, event="exit")
    persist_batch(db_factory, run_id, source_id, [first, second])
    persist_batch(db_factory, run_id, source_id, [first, second])
    persist_batch(db_factory, run_id, source_id, [first])
    assert record_counts(db_factory) == {"detections": 2, "events": 2, "frame_samples": 2, "run_area_stats": 1}
    with db_factory() as db:
        run = db.get(ProcessingRun, run_id)
        counters = db.get(RunAreaStats, (run_id, spec["areas"][0]["id"]))
        assert run.last_frame_index == 2 and run.status == "running"
        assert run.started_at is not None and run.heartbeat_at is not None
        assert (counters.total_in, counters.total_out) == (1, 1)


def test_invalid_event_rolls_back_detections_stats_and_progress_for_whole_batch(db_factory, persisted_run):
    run_id, source_id, spec = persisted_run
    first = frame_packet(spec, 1, total_in=1, detection=True, event="enter")
    invalid = frame_packet(spec, 2, total_in=2, detection=True, event="invalid")
    with pytest.raises(IntegrityError):
        persist_batch(db_factory, run_id, source_id, [first, invalid], retries=1)
    assert record_counts(db_factory) == {"detections": 0, "events": 0, "frame_samples": 0, "run_area_stats": 0}
    with db_factory() as db:
        assert db.get(ProcessingRun, run_id).last_frame_index == 0


def test_retry_after_uncertain_success_is_idempotent(db_factory, persisted_run, monkeypatch):
    run_id, source_id, spec = persisted_run
    packet = frame_packet(spec, 1, total_in=1, detection=True, event="enter")
    monkeypatch.setattr("app.services.persistence.time.sleep", lambda _: None)
    class UncertainCommit:
        attempts = 0
        @contextmanager
        def begin(self):
            self.attempts += 1
            with db_factory.begin() as db:
                yield db
            if self.attempts == 1:
                raise OperationalError("commit", {}, RuntimeError("reply lost after committed transaction"))
    factory = UncertainCommit()
    persist_batch(factory, run_id, source_id, [packet], retries=3)
    assert factory.attempts == 2
    assert record_counts(db_factory) == {"detections": 1, "events": 1, "frame_samples": 1, "run_area_stats": 1}


def test_database_failure_retries_are_bounded(db_factory, persisted_run, monkeypatch):
    run_id, source_id, spec = persisted_run
    monkeypatch.setattr("app.services.persistence.time.sleep", lambda _: None)
    class FailingFactory:
        attempts = 0
        @contextmanager
        def begin(self):
            self.attempts += 1
            raise OperationalError("connect", {}, RuntimeError("unavailable"))
            yield  # pragma: no cover - makes this a context manager generator
    factory = FailingFactory()
    with pytest.raises(OperationalError):
        persist_batch(factory, run_id, source_id, [frame_packet(spec, 1)], retries=3)
    assert factory.attempts == 3
    assert record_counts(db_factory)["frame_samples"] == 0
