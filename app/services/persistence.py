from datetime import datetime, timezone
from math import floor, isfinite
import time

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import ACTIVE_STATUSES, Detection, Event, FrameSample, ProcessingRun, RunAreaStats, RunBucket


def utcnow():
    return datetime.now(timezone.utc)


def minute_deltas(packet, run_id, kind):
    rows = {}
    def row(area_id, start):
        key = (area_id, start)
        return rows.setdefault(key, dict(run_id=run_id, area_id=area_id, window_start=start,
            coverage_seconds=0.0, occupancy_seconds=0.0, sample_count=0, in_count=0, out_count=0))
    start, duration = packet["coverage_start_seconds"], packet["duration_seconds"]
    if not isfinite(start) or not isfinite(duration) or duration < 0:
        raise ValueError("Invalid observation coverage")
    end = start + duration
    left = floor(start / 60) * 60
    while left < end - 1e-9:
        overlap = max(0, min(end, left + 60) - max(start, left))
        for stat in packet["stats"]:
            delta = row(stat["area_id"], left)
            delta["coverage_seconds"] += overlap
            delta["occupancy_seconds"] += overlap * stat["currently_inside"]
            delta["sample_count"] += 1
        left += 60
    event_time = packet["media_time_ms"] / 1000 if kind == "file" else datetime.fromisoformat(packet["observed_at"]).timestamp()
    for event in packet["events"]:
        row(event["area_id"], floor(event_time / 60) * 60)["in_count" if event["type"] == "enter" else "out_count"] += 1
    return [rows[key] for key in sorted(rows)]


def persist_batch(factory, run_id, source_id, packets, retries=3):
    """Records, absolute counters and progress commit together; safe to retry."""
    if not packets:
        return
    for attempt in range(retries):
        try:
            with factory.begin() as db:
                run = db.get(ProcessingRun, run_id, with_for_update=True)
                if run is None:
                    raise ValueError("Unknown run")
                if run.status not in ACTIVE_STATUSES:
                    raise ValueError("Run is no longer active")
                deltas = {}
                for packet in packets:
                    # A repeated delivery after an uncertain commit cannot regress counters.
                    if packet["frame_index"] <= run.last_frame_index:
                        continue
                    for delta in minute_deltas(packet, run_id, run.kind):
                        key = (delta["area_id"], delta["window_start"])
                        combined = deltas.setdefault(key, {**delta, **{k: 0 for k in ("coverage_seconds", "occupancy_seconds", "sample_count", "in_count", "out_count")}})
                        for field in ("coverage_seconds", "occupancy_seconds", "sample_count", "in_count", "out_count"):
                            combined[field] += delta[field]
                    common = dict(run_id=run_id, video_source_id=source_id,
                                  frame_index=packet["frame_index"], segment_index=packet["segment_index"],
                                  media_time_ms=packet["media_time_ms"],
                                  timestamp=datetime.fromisoformat(packet["observed_at"]))
                    if packet["detections"]:
                        db.execute(insert(Detection).values([{**common, **row} for row in packet["detections"]])
                                   .on_conflict_do_nothing(constraint="uq_detection_run_frame"))
                    if packet["events"]:
                        db.execute(insert(Event).values([{**common, **row} for row in packet["events"]])
                                   .on_conflict_do_nothing(constraint="uq_event_run_frame"))
                    db.execute(insert(FrameSample).values(run_id=run_id, frame_index=packet["frame_index"],
                        time_seconds=packet["coverage_start_seconds"], duration_seconds=packet["duration_seconds"],
                        stats=packet["stats"]).on_conflict_do_nothing())
                    for stat in packet["stats"]:
                        stmt = insert(RunAreaStats).values(run_id=run_id, **stat)
                        db.execute(stmt.on_conflict_do_update(index_elements=["run_id", "area_id"],
                            set_={key: getattr(stmt.excluded, key) for key in ("total_in", "total_out", "currently_inside", "lost_tracks")}))
                    run.last_frame_index = packet["frame_index"]
                    run.segment_index = packet["segment_index"]
                    run.media_time_ms = packet["media_time_ms"]
                    run.fps = packet["fps"]
                    run.dropped_frames = packet["dropped_frames"]
                    run.heartbeat_at = datetime.fromisoformat(packet["observed_at"])
                    run.started_at = run.started_at or run.heartbeat_at
                    if run.status in {"starting", "reconnecting"}:
                        run.status = "running"
                for delta in deltas.values():
                    stmt = insert(RunBucket).values(**delta)
                    db.execute(stmt.on_conflict_do_update(index_elements=["run_id", "area_id", "window_start"],
                        set_={key: getattr(RunBucket, key) + getattr(stmt.excluded, key)
                              for key in ("coverage_seconds", "occupancy_seconds", "sample_count", "in_count", "out_count")}))
            return
        except SQLAlchemyError:
            if attempt + 1 == retries:
                raise
            time.sleep(0.1 * 2 ** attempt)
