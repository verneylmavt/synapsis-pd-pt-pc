from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.db.models import Area, Event, FrameSample, ProcessingRun, RunAreaStats, RunBucket, VideoSource

pytestmark = pytest.mark.integration


def seed(factory, kind="file", status="completed"):
    with factory.begin() as db:
        source = VideoSource(name="test", uri="local", kind=kind)
        db.add(source)
        db.flush()
        area = Area(video_source_id=source.id, name="door", polygon=[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}])
        db.add(area)
        db.flush()
        run_id = str(uuid4())
        db.add(ProcessingRun(id=run_id, video_source_id=source.id, kind=kind, status=status, owner_id=str(uuid4()), settings={},
            areas=[{"id": area.id, "name": area.name, "polygon": area.polygon}], heartbeat_at=datetime.now(timezone.utc)))
        db.flush()
        db.add(RunAreaStats(run_id=run_id, area_id=area.id, total_in=1, total_out=3, currently_inside=2))
        return source.id, area.id, run_id


def test_terminal_occupancy_and_signed_net(api_client):
    source, area, run = seed(api_client.app.state.session_factory)
    result = api_client.get("/api/stats/live", params={"run_id": run, "area_id": area}).json()
    assert result["currently_inside"] is None
    assert result["last_observed_inside"] == 2
    assert result["net_entries"] == -2


def test_file_history_media_time_and_complete_zero_buckets(api_client):
    source, area, run = seed(api_client.app.state.session_factory)
    with api_client.app.state.session_factory.begin() as db:
        for index in range(1200):
            db.add(FrameSample(run_id=run, frame_index=index + 1, time_seconds=index, duration_seconds=1,
                stats=[{"area_id": area, "currently_inside": 2}]))
        for index in range(20):
            db.add(RunBucket(run_id=run, area_id=area, window_start=index * 60, coverage_seconds=60,
                            occupancy_seconds=120, sample_count=60, in_count=0, out_count=0))
    history = api_client.get("/api/stats", params={"run_id": run, "area_id": area}).json()
    assert history["timeline"] == "file"
    assert len(history["buckets"]) == 20
    assert all(b["observed"] and b["in_count"] == 0 for b in history["buckets"])


def test_invalid_timezone_and_run_area_isolation(api_client):
    source, area, run = seed(api_client.app.state.session_factory, kind="live")
    assert api_client.get("/api/stats", params={"run_id": run, "area_id": area, "start": "2026-10-05T00:00:00"}).status_code == 422
    assert api_client.get("/api/stats/live", params={"run_id": run, "area_id": area + 1}).status_code == 404


def test_explicit_legacy_scope_does_not_leak_into_run(api_client):
    source, area, run = seed(api_client.app.state.session_factory)
    with api_client.app.state.session_factory.begin() as db:
        db.add(Event(video_source_id=source, area_id=area, tracker_id="old", type="enter"))
    legacy = api_client.get("/api/stats/live", params={"scope": "legacy", "video_source_id": source}).json()
    assert legacy["total_in"] == 1 and legacy["currently_inside"] is None
    current = api_client.get("/api/stats/live", params={"run_id": run, "area_id": area}).json()
    assert current["total_in"] == 1 and current["total_out"] == 3
