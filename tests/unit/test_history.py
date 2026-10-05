import importlib.util

import pytest


def build(*args, **kwargs):
    try:
        available = importlib.util.find_spec("app.services.history")
    except ModuleNotFoundError:
        available = None
    assert available, "history bucketing is missing"
    from app.services.history import build_buckets
    return build_buckets(*args, **kwargs)


def sample(start, duration, inside=0, **extra):
    return {"time_seconds": start, "duration_seconds": duration,
            "currently_inside": inside, **extra}


def test_zero_traffic_requires_explicit_coverage_and_gaps_are_null():
    buckets = build([], [sample(0, 60, 3), sample(120, 60, 1)], "file", 0, 180, 60)
    assert len(buckets) == 3
    assert buckets[0]["in_count"] == buckets[0]["out_count"] == 0
    assert buckets[0]["currently_inside"] == 3
    assert buckets[0]["observed"] and buckets[0]["complete"]
    assert buckets[1]["in_count"] is None
    assert buckets[1]["out_count"] is None
    assert not buckets[1]["observed"]
    assert buckets[1]["currently_inside"] is None


def test_half_open_event_buckets_and_partial_tail():
    events = [{"time_seconds": 59.9, "type": "enter"},
              {"time_seconds": 60, "type": "exit"},
              {"time_seconds": 120, "type": "enter"}]
    buckets = build(events, [sample(0, 150)], "file", 0, 150, 60)
    assert buckets[0]["in_count"] == 1
    assert buckets[0]["out_count"] == 0
    assert buckets[1]["in_count"] == 0
    assert buckets[1]["out_count"] == 1
    assert not buckets[2]["complete"]
    assert buckets[2]["in_count"] is None
    assert buckets[2]["window_end"] == 180


def test_fragments_union_and_time_weighted_occupancy():
    buckets = build([], [sample(0, 30, 2), sample(30, 30, 4)], "file", 0, 60, 60)
    assert buckets[0]["observed"]
    assert buckets[0]["currently_inside"] == pytest.approx(3)
    assert buckets[0]["sample_count"] == 2


def test_overlap_does_not_double_weight_occupancy_and_latest_sample_wins():
    buckets = build([], [sample(0, 60, 2), sample(30, 30, 4)], "file", 0, 60, 60)
    assert buckets[0]["currently_inside"] == pytest.approx(3)


def test_utc_epoch_and_file_seconds_use_same_fixed_boundaries():
    timestamp = 1_800_000_000
    buckets = build([], [sample(timestamp, 120)], "live", timestamp, timestamp + 120, 60)
    assert buckets[0]["window_start"] == timestamp
    assert buckets[1]["window_start"] == timestamp + 60
    assert all(bucket["complete"] and bucket["observed"] for bucket in buckets)


def test_unaligned_start_is_incomplete_and_empty_range_has_no_buckets():
    buckets = build([], [sample(30, 90)], "file", 30, 120, 60)
    assert not buckets[0]["complete"]
    assert buckets[0]["in_count"] is None
    assert buckets[1]["complete"]
    assert build([], [], "file", 0, 0, 60) == []


def test_rejects_mixed_runs_instead_of_pooling():
    with pytest.raises(ValueError, match="run"):
        build([], [sample(0, 60, run_id="a"), sample(60, 60, run_id="b")], "file", 0, 120, 60)


@pytest.mark.parametrize("kind,start,end,seconds", [("wallclock", 0, 60, 60),
    ("file", 60, 0, 60), ("file", 0, 60, 0), ("file", 0, float("inf"), 60)])
def test_rejects_invalid_timeline(kind, start, end, seconds):
    with pytest.raises(ValueError):
        build([], [], kind, start, end, seconds)
