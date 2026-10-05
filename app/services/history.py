from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from math import floor, isfinite


def build_buckets(events: list[dict], samples: list[dict], kind: str,
                  start: float, end: float, bucket_seconds: int) -> list[dict]:
    """Bucket one run/area on media seconds or UTC epoch seconds.

    Samples explicitly cover [time_seconds, time_seconds + duration_seconds).
    Missing coverage is unavailable, including intervals with persisted events.
    Occupancy is time weighted; overlapping intervals use the latest sample.
    """
    if (kind not in {"file", "live"} or not isfinite(start) or not isfinite(end)
            or end < start or (kind == "file" and start < 0)
            or isinstance(bucket_seconds, bool) or not isinstance(bucket_seconds, int)
            or bucket_seconds <= 0):
        raise ValueError("Invalid timeline or bucket size")
    run_ids = {row["run_id"] for row in events + samples if row.get("run_id") is not None}
    if len(run_ids) > 1:
        raise ValueError("History cannot pool multiple runs")
    if start == end:
        return []
    first = floor(start / bucket_seconds) * bucket_seconds
    buckets = []
    left = first
    while left < end:
        right = left + bucket_seconds
        buckets.append({"window_start": left, "window_end": right,
                        "in_count": None, "out_count": None, "observed": False,
                        "complete": left >= start and right <= end,
                        "currently_inside": None, "sample_count": 0})
        left = right

    intervals = []
    points = {start, end}
    points.update(row["window_start"] for row in buckets if row["window_start"] >= start)
    points.update(row["window_end"] for row in buckets if row["window_end"] <= end)
    for index, sample in enumerate(samples):
        sample_start, duration = sample["time_seconds"], sample["duration_seconds"]
        occupancy = sample["currently_inside"]
        if (not isfinite(sample_start) or not isfinite(duration) or duration < 0
                or not isfinite(occupancy) or occupancy < 0):
            raise ValueError("Samples need finite time, nonnegative duration and occupancy")
        sample_end = sample_start + duration
        if not isfinite(sample_end):
            raise ValueError("Sample endpoint must be finite")
        a, b = max(start, sample_start), min(end, sample_end)
        if a >= b:
            continue
        intervals.append((a, b, sample_start, index, occupancy))
        points.update((a, b))
        first_index = int((a - first) // bucket_seconds)
        last_index = min(len(buckets) - 1, int((b - first) // bucket_seconds))
        for bucket_index in range(first_index, last_index + 1):
            if buckets[bucket_index]["window_start"] < b:
                buckets[bucket_index]["sample_count"] += 1

    starts = defaultdict(list)
    for a, b, sample_start, index, occupancy in intervals:
        starts[a].append((-sample_start, -index, b, occupancy))
    active = []
    covered = [0.0] * len(buckets)
    weighted = [0.0] * len(buckets)
    ordered = sorted(points)
    for a, b in zip(ordered, ordered[1:]):
        for interval in starts[a]:
            heappush(active, interval)
        while active and active[0][2] <= a:
            heappop(active)
        if active:
            bucket_index = int((a - first) // bucket_seconds)
            covered[bucket_index] += b - a
            weighted[bucket_index] += (b - a) * active[0][3]
    for index, bucket in enumerate(buckets):
        expected = min(end, bucket["window_end"]) - max(start, bucket["window_start"])
        bucket["observed"] = covered[index] > 0 and abs(covered[index] - expected) <= 1e-7
        if bucket["observed"]:
            bucket["currently_inside"] = weighted[index] / covered[index]
        if bucket["observed"] and bucket["complete"]:
            bucket["in_count"] = bucket["out_count"] = 0
    for event in events:
        timestamp, event_type = event["time_seconds"], event["type"]
        if not isfinite(timestamp) or event_type not in {"enter", "exit"}:
            raise ValueError("Events need finite time and enter/exit type")
        if not start <= timestamp < end:
            continue
        bucket = buckets[int((timestamp - first) // bucket_seconds)]
        if bucket["observed"] and bucket["complete"]:
            key = "in_count" if event_type == "enter" else "out_count"
            bucket[key] += 1
    return buckets
