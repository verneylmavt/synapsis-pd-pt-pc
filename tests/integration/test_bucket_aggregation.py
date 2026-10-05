import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError

from app.db import models
from app.services import persistence
from tests.fake_worker import RUN_SETTINGS, frame_packet


def test_packet_deltas_split_half_open_coverage_and_use_event_timeline():
    assert hasattr(persistence, "minute_deltas"), "durable minute aggregation is missing"
    packet = {"coverage_start_seconds": 59, "duration_seconds": 3, "media_time_ms": 62_000,
              "observed_at": "2026-10-05T00:01:00+00:00",
              "stats": [{"area_id": 1, "currently_inside": 4}],
              "events": [{"area_id": 1, "type": "enter"}]}
    rows = persistence.minute_deltas(packet, "run", "file")
    assert [row["window_start"] for row in rows] == [0, 60]
    assert [row["coverage_seconds"] for row in rows] == [1, 2]
    assert [row["occupancy_seconds"] for row in rows] == [4, 8]
    assert [row["sample_count"] for row in rows] == [1, 1]
    assert [row["in_count"] for row in rows] == [0, 1]
    packet.update(coverage_start_seconds=1_791_158_459, duration_seconds=1)
    live = persistence.minute_deltas(packet, "run", "live")
    event_row = next(row for row in live if row["in_count"])
    from datetime import datetime
    assert event_row["window_start"] == datetime.fromisoformat(packet["observed_at"]).timestamp()


@pytest.fixture
def run_scope(db_factory):
    with db_factory.begin() as db:
        source = models.VideoSource(name="bucket test", uri="fake.mp4", kind="file")
        db.add(source)
        db.flush()
        area = models.Area(video_source_id=source.id, name="zone", polygon=[
            {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])
        db.add(area)
        db.flush()
        run_id = str(uuid4())
        db.add(models.ProcessingRun(id=run_id, video_source_id=source.id, kind="file", status="starting",
            owner_id=str(uuid4()), settings=dict(RUN_SETTINGS), areas=[{"id": area.id}]))
        return run_id, source.id, {"run_id": run_id, "areas": [{"id": area.id}]}


def test_aggregate_replay_and_rollback_match_audit_and_absolute_counters(db_factory, run_scope):
    assert hasattr(models, "RunBucket"), "durable minute model is missing"
    run_id, source_id, spec = run_scope
    first = frame_packet(spec, 1, total_in=1, event="enter")
    first.update(coverage_start_seconds=0, duration_seconds=60, media_time_ms=0)
    first["stats"][0]["currently_inside"] = 3
    persistence.persist_batch(db_factory, run_id, source_id, [first])
    persistence.persist_batch(db_factory, run_id, source_id, [first])
    with db_factory() as db:
        bucket = db.get(models.RunBucket, (run_id, spec["areas"][0]["id"], 0))
        assert (bucket.coverage_seconds, bucket.occupancy_seconds, bucket.sample_count) == (60, 180, 1)
        assert (bucket.in_count, bucket.out_count) == (1, 0)
    second = frame_packet(spec, 2, total_in=2, event="enter")
    second.update(coverage_start_seconds=60, duration_seconds=60, media_time_ms=60_000)
    invalid = frame_packet(spec, 3, event="invalid")
    invalid.update(coverage_start_seconds=120, duration_seconds=60, media_time_ms=120_000)
    with pytest.raises(IntegrityError):
        persistence.persist_batch(db_factory, run_id, source_id, [second, invalid], retries=1)
    with db_factory() as db:
        assert len(list(db.scalars(select(models.RunBucket)))) == 1
        assert db.get(models.ProcessingRun, run_id).last_frame_index == 1
        assert len(list(db.scalars(select(models.FrameSample)))) == 1


def test_hardened_run_keys_reject_null_but_legacy_keys_remain_nullable(db_factory, run_scope):
    run_id, source_id, spec = run_scope
    with pytest.raises(IntegrityError):
        with db_factory.begin() as db:
            db.add(models.Event(run_id=run_id, video_source_id=source_id, area_id=spec["areas"][0]["id"],
                                tracker_id="1", type="enter", frame_index=None, segment_index=0))
    with db_factory.begin() as db:
        db.add(models.Event(video_source_id=source_id, area_id=spec["areas"][0]["id"], tracker_id="old", type="enter"))


def test_upgrade_backfills_coverage_and_events_without_touching_legacy(db_engine):
    from alembic import command
    from alembic.config import Config
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.attributes["database_url"] = os.environ["TEST_DATABASE_URL"]
    run_id = str(uuid4())
    try:
        command.downgrade(config, "0003")
        with db_engine.begin() as db:
            db.execute(text("INSERT INTO video_sources(id,name,uri,kind) VALUES(101,'old run','file','file')"))
            db.execute(text("INSERT INTO areas(id,video_source_id,name,polygon) VALUES(201,101,'zone','[]')"))
            db.execute(text("INSERT INTO processing_runs(id,video_source_id,kind,status,owner_id,settings,areas) VALUES(:id,101,'file','completed',:id,'{}','[]')"), {"id": run_id})
            db.execute(text("INSERT INTO frame_samples(run_id,frame_index,time_seconds,duration_seconds,stats) VALUES(:id,1,59,3,CAST(:stats AS JSONB))"),
                       {"id": run_id, "stats": json.dumps([{"area_id": 201, "currently_inside": 4}])})
            db.execute(text("INSERT INTO frame_samples(run_id,frame_index,time_seconds,duration_seconds,stats) VALUES(:id,2,60,1,CAST(:stats AS JSONB))"),
                       {"id": run_id, "stats": json.dumps([{"area_id": 201, "currently_inside": 4}])})
            db.execute(text("INSERT INTO events(video_source_id,area_id,run_id,frame_index,segment_index,media_time_ms,tracker_id,type) VALUES(101,201,:id,1,0,62000,'1','enter')"), {"id": run_id})
            db.execute(text("INSERT INTO events(video_source_id,area_id,tracker_id,type) VALUES(101,201,'legacy','exit')"))
        command.upgrade(config, "head")
        with db_engine.connect() as db:
            rows = db.execute(text("SELECT window_start,coverage_seconds,occupancy_seconds,in_count FROM run_buckets ORDER BY window_start")).all()
            assert rows == [(0, 1, 4, 0), (60, 2, 8, 1)]
            assert db.execute(text("SELECT count(*) FROM events WHERE run_id IS NULL AND tracker_id='legacy'")).scalar_one() == 1
    finally:
        command.upgrade(config, "head")
