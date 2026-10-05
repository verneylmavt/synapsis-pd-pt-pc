from __future__ import annotations

from math import isfinite


MIN_HISTORY = 20
TRAINING_MIN = 10
BACKTEST_HORIZON = 5
EWMA_ALPHA = 0.3


def _valid_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def _ewma(values: list[float]) -> float:
    level = values[0]
    for value in values[1:]:
        level = EWMA_ALPHA * value + (1 - EWMA_ALPHA) * level
    return level


def _backtest(values: list[float]) -> dict[str, float]:
    errors = {"naive": 0.0, "ewma": 0.0}
    observations = 0
    levels = [values[0]]
    for value in values[1:]:
        levels.append(EWMA_ALPHA * value + (1 - EWMA_ALPHA) * levels[-1])
    for origin in range(TRAINING_MIN, len(values) - BACKTEST_HORIZON + 1):
        # Every origin fits only on its prefix, then holds a fixed five-step forecast.
        predictions = {"naive": values[origin - 1], "ewma": levels[origin - 1]}
        for actual in values[origin:origin + BACKTEST_HORIZON]:
            observations += 1
            for method, prediction in predictions.items():
                errors[method] += (abs(actual - prediction) - errors[method]) / observations
    return errors


def forecast_counts(buckets: list[dict], horizon: int = 5) -> dict:
    """Compare last-value and EWMA baselines on a single observed minute timeline.

    Predictions assume the recent traffic rate continues. Backtest errors are
    descriptive measurements, not calibrated uncertainty intervals.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive number of minutes")
    base = {"status": "insufficient_data", "reason": "requires_20_observed_contiguous_minutes",
            "horizon_minutes": horizon, "minimum_history_minutes": MIN_HISTORY,
            "predictions": [], "methods": {}, "backtest_mae": {},
            "calibrated_intervals": False, "alpha": EWMA_ALPHA,
            "backtest_horizon_minutes": BACKTEST_HORIZON, "training_min_minutes": TRAINING_MIN}
    run_ids = {bucket["run_id"] for bucket in buckets if bucket.get("run_id") is not None}
    if len(run_ids) > 1:
        return {**base, "reason": "multiple_runs"}
    candidates = list(buckets)
    while candidates and not candidates[-1].get("complete", False):
        candidates.pop()
    trailing = []
    following_start = None
    for bucket in reversed(candidates):
        start, end = bucket.get("window_start"), bucket.get("window_end")
        if (not bucket.get("complete") or not bucket.get("observed")
                or not _valid_number(start) or not _valid_number(end)
                or abs(end - start - 60) > 1e-7
                or (following_start is not None and abs(end - following_start) > 1e-7)
                or any(not _valid_number(bucket.get(key)) or bucket[key] < 0
                       for key in ("in_count", "out_count"))):
            break
        trailing.append(bucket)
        following_start = start
    trailing.reverse()
    if len(trailing) < MIN_HISTORY:
        return {**base, "history_minutes": len(trailing)}
    methods, scores, levels = {}, {}, {}
    for direction, key in (("in", "in_count"), ("out", "out_count")):
        values = [float(bucket[key]) for bucket in trailing]
        scores[direction] = _backtest(values)
        use_ewma = scores[direction]["ewma"] < scores[direction]["naive"]
        methods[direction] = "ewma" if use_ewma else "last_value"
        levels[direction] = max(0.0, _ewma(values) if use_ewma else values[-1])
    return {**base, "status": "ok", "reason": None, "history_minutes": len(trailing),
            "methods": methods, "backtest_mae": scores,
            "predictions": [{"step": step, "in_count": levels["in"], "out_count": levels["out"]}
                            for step in range(1, horizon + 1)]}
