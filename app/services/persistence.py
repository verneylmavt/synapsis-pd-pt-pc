from datetime import datetime, timezone
import time

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Detection, Event, FrameSample, ProcessingRun, RunAreaStats


def utcnow():
    return datetime.now(timezone.utc)


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
                for packet in packets:
                    # A repeated delivery after an uncertain commit cannot regress counters.
                    if packet["frame_index"] <= run.last_frame_index:
                        continue
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
            return
        except SQLAlchemyError:
            if attempt + 1 == retries:
                raise
            time.sleep(0.1 * 2 ** attempt)
