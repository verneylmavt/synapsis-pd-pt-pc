import importlib.util

import pytest


def forecast(buckets, **kwargs):
    assert importlib.util.find_spec("app.services.forecast"), "forecast baseline is missing"
    from app.services.forecast import forecast_counts
    return forecast_counts(buckets, **kwargs)


def history(incoming, outgoing=None):
    outgoing = outgoing or incoming
    return [{"window_start": index * 60, "window_end": (index + 1) * 60,
             "complete": True, "observed": True, "in_count": value,
             "out_count": outgoing[index]} for index, value in enumerate(incoming)]


def test_twenty_contiguous_observed_minutes_gate_and_ties_use_last_value():
    assert forecast(history([2] * 19))["status"] == "insufficient_data"
    result = forecast(history([2] * 20))
    assert result["status"] == "ok"
    assert result["methods"] == {"in": "last_value", "out": "last_value"}
    assert result["backtest_mae"]["in"] == {"naive": 0.0, "ewma": 0.0}
    assert result["predictions"] == [
        {"step": step, "in_count": 2.0, "out_count": 2.0} for step in range(1, 6)]
    assert result["calibrated_intervals"] is False


def test_independent_selection_and_chronological_five_step_mae_without_leakage():
    result = forecast(history(list(range(1, 21)), [0, 10] * 10))
    assert result["methods"] == {"in": "last_value", "out": "ewma"}
    assert result["backtest_mae"]["in"]["naive"] == pytest.approx(3.0)
    expected_ewma_mae = sum(3 + (7 / 3) * (1 - 0.7 ** (n - 1)) for n in range(10, 16)) / 6
    assert result["backtest_mae"]["in"]["ewma"] == pytest.approx(expected_ewma_mae)
    assert result["predictions"][0]["in_count"] == 20.0
    assert 0 < result["predictions"][0]["out_count"] < 10


def test_missing_or_discontinuous_minutes_break_the_training_window():
    buckets = history([1] * 30)
    buckets[20]["observed"] = False
    buckets[20]["in_count"] = None
    assert forecast(buckets)["status"] == "insufficient_data"
    buckets = history([1] * 20)
    buckets[10]["window_start"] += 1
    assert forecast(buckets)["status"] == "insufficient_data"


def test_partial_tail_is_excluded_but_missing_final_complete_bucket_is_not_zero():
    buckets = history([1] * 21)
    buckets[-1].update(complete=False, in_count=None, out_count=None)
    assert forecast(buckets)["status"] == "ok"
    buckets[-2].update(observed=False, in_count=None, out_count=None)
    assert forecast(buckets)["status"] == "insufficient_data"


@pytest.mark.parametrize("invalid", [None, -1, float("nan"), float("inf")])
def test_invalid_counts_do_not_become_zero(invalid):
    buckets = history([1] * 20)
    buckets[-1]["in_count"] = invalid
    assert forecast(buckets)["status"] == "insufficient_data"


def test_hourly_buckets_and_mixed_runs_cannot_be_forecast_as_minutes():
    buckets = history([1] * 20)
    buckets[-1]["window_end"] += 3540
    assert forecast(buckets)["status"] == "insufficient_data"
    buckets = history([1] * 20)
    buckets[0]["run_id"] = "a"
    buckets[-1]["run_id"] = "b"
    assert forecast(buckets)["status"] == "insufficient_data"


def test_predictions_are_nonnegative_and_horizon_is_validated():
    assert all(item["in_count"] == 0 for item in forecast(history([0] * 20))["predictions"])
    with pytest.raises(ValueError):
        forecast(history([1] * 20), horizon=0)
