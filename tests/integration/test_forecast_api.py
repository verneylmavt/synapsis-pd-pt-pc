from datetime import datetime, timedelta, timezone

import pytest

from app.db.models import FrameSample, ProcessingRun, RunBucket
from tests.integration.test_stats import seed

pytestmark = pytest.mark.integration


def test_terminal_file_forecast_is_hypothetical_and_zero_traffic_is_observed(api_client):
    _, area, run = seed(api_client.app.state.session_factory)
    with api_client.app.state.session_factory.begin() as db:
        db.add_all([FrameSample(run_id=run, frame_index=1, time_seconds=0, duration_seconds=1, stats=[]),
                    FrameSample(run_id=run, frame_index=1200, time_seconds=1199, duration_seconds=1, stats=[])])
        for minute in range(20):
            db.add(RunBucket(run_id=run, area_id=area, window_start=minute * 60, coverage_seconds=60,
                            occupancy_seconds=120, sample_count=60, in_count=0, out_count=0))
    forecast = api_client.get("/api/forecast", params={"run_id": run, "area_id": area}).json()
    assert forecast["status"] == "ready"
    assert forecast["methods"] == {"in": "last_value", "out": "last_value"}
    assert "Hypothetical" in forecast["label"]


@pytest.mark.parametrize("status,heartbeat", [
    ("completed", datetime.now(timezone.utc)), ("running", None),
    ("running", datetime.now(timezone.utc) - timedelta(minutes=10)),
])
def test_live_forecasts_require_current_observation(api_client, status, heartbeat):
    _, area, run = seed(api_client.app.state.session_factory, kind="live", status=status)
    with api_client.app.state.session_factory.begin() as db:
        db.get(ProcessingRun, run).heartbeat_at = heartbeat
    result = api_client.get("/api/forecast", params={"run_id": run, "area_id": area}).json()
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "live_run_not_current"
    assert result["predictions"] == []
