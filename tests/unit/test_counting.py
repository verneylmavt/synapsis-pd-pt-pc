import importlib.util

import pytest


AREA = {"id": 1, "name": "zone", "polygon": [
    {"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1},
    {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9}]}


def engine(areas=None, **kwargs):
    assert importlib.util.find_spec("app.logic.counting"), "counting engine is missing"
    from app.logic.counting import CountingEngine
    return CountingEngine(areas or [AREA], **kwargs)


def detection(inside=True, tracker_id="1"):
    center = 0.5 if inside else 0.95
    return {"tracker_id": tracker_id, "x1": center - 0.02, "x2": center + 0.02,
            "y1": 0.4, "y2": 0.6, "conf": 0.9}


def observe(counter, inside, times, tracker_id="1"):
    return [counter.update([detection(inside, tracker_id)], t) for t in times]


def test_initial_inside_is_presence_without_enter_and_exit_can_make_negative_net():
    counter = engine()
    results = observe(counter, True, [0, 0.1, 0.2])
    assert all(result["events"] == [] for result in results)
    assert results[-1]["stats"][0]["currently_inside"] == 1
    final = observe(counter, False, [0.3, 0.4, 0.5])[-1]
    assert final["events"] == [{"area_id": 1, "tracker_id": "1", "type": "exit"}]
    assert final["stats"][0] == {"area_id": 1, "total_in": 0, "total_out": 1,
                                  "currently_inside": 0, "lost_tracks": 0}
    final = observe(counter, True, [0.6, 0.7, 0.8])[-1]
    assert final["events"][0]["type"] == "enter"


def test_confirmed_initial_outside_then_inside_emits_enter():
    counter = engine()
    observe(counter, False, [0, 0.1, 0.2])
    final = observe(counter, True, [0.3, 0.4, 0.5])[-1]
    assert final["events"][0]["type"] == "enter"
    assert final["stats"][0]["total_in"] == 1


def test_missing_frame_resets_pending_streak_but_preserves_latched_presence():
    counter = engine()
    observe(counter, True, [0, 0.1, 0.2])
    observe(counter, False, [0.3, 0.4])
    assert counter.update([], 0.5)["stats"][0]["currently_inside"] == 1
    assert observe(counter, False, [0.6, 0.7])[-1]["events"] == []
    assert observe(counter, False, [0.8])[-1]["events"][0]["type"] == "exit"


def test_expiry_is_strict_and_never_synthesizes_exit():
    counter = engine()
    observe(counter, True, [0, 0.1, 0.2])
    assert counter.update([], 2.2)["stats"][0]["currently_inside"] == 1
    final = counter.update([], 2.21)
    assert final["events"] == []
    assert final["stats"][0]["currently_inside"] == 0
    assert final["stats"][0]["lost_tracks"] == 1
    assert counter.update([], 3)["stats"][0]["lost_tracks"] == 1
    assert observe(counter, True, [3.1, 3.2, 3.3])[-1]["events"] == []


def test_reappearing_after_gap_expires_before_processing_current_detection():
    counter = engine()
    observe(counter, True, [0, 0.1, 0.2])
    final = counter.update([detection(False)], 3)
    assert final["events"] == []
    assert final["stats"][0]["currently_inside"] == 0
    assert final["stats"][0]["lost_tracks"] == 1


def test_overlapping_areas_count_independently_in_sorted_snapshots():
    counter = engine([{**AREA, "id": 2}, AREA])
    observe(counter, False, [0, 0.1, 0.2])
    final = observe(counter, True, [0.3, 0.4, 0.5])[-1]
    assert [row["area_id"] for row in final["stats"]] == [1, 2]
    assert [event["area_id"] for event in final["events"]] == [1, 2]
    assert all(row["currently_inside"] == 1 for row in final["stats"])


def test_untracked_and_duplicate_detections_do_not_accelerate_confirmation():
    counter = engine()
    assert counter.update([detection(tracker_id=None)], 0)["stats"][0]["currently_inside"] == 0
    counter.update([detection(), detection()], 0.1)
    assert counter.update([detection()], 0.2)["stats"][0]["currently_inside"] == 0
    assert counter.update([detection()], 0.3)["stats"][0]["currently_inside"] == 1


def test_time_must_be_finite_and_monotonic():
    counter = engine()
    counter.update([], 1)
    with pytest.raises(ValueError):
        counter.update([], 0)
    with pytest.raises(ValueError):
        counter.update([], float("nan"))


def test_nullable_duplicate_confidence_does_not_crash_or_confirm_twice():
    counter = engine()
    nullable = {**detection(), "conf": None}
    counter.update([nullable, detection()], 0)
    assert counter.update([detection()], 0.1)["stats"][0]["currently_inside"] == 0
    assert counter.update([detection()], 0.2)["stats"][0]["currently_inside"] == 1
