from datetime import datetime, timezone
import math

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.db.models import ACTIVE_STATUSES, Event, FrameSample, ProcessingRun, RunAreaStats, RunBucket
from app.db.session import get_db

router = APIRouter()


def selected_run(db, run_id=None, source_id=None):
    if run_id:
        run = db.get(ProcessingRun, run_id)
    elif source_id:
        run = db.execute(select(ProcessingRun).where(ProcessingRun.video_source_id == source_id)
            .order_by(ProcessingRun.created_at.desc())).scalars().first()
    else:
        raise HTTPException(422, "Select a run or video_source_id")
    if run is None or (source_id and run.video_source_id != source_id):
        raise HTTPException(404, "Run not found for this source")
    return run


def selected_area(run, area_id):
    ids = [row["id"] for row in run.areas]
    area_id = area_id if area_id is not None else (ids[0] if ids else None)
    if area_id not in ids:
        raise HTTPException(404, "Area was not part of this run")
    return area_id


@router.get("/api/stats/live")
def live_stats(request: Request, video_source_id: int | None = None, area_id: int | None = None,
               run_id: str | None = None, scope: str = "run", db: Session = Depends(get_db)):
    if scope == "legacy":
        query = select(Event).where(Event.run_id.is_(None))
        if video_source_id:
            query = query.where(Event.video_source_id == video_source_id)
        if area_id:
            query = query.where(Event.area_id == area_id)
        events = list(db.execute(query).scalars())
        entries, exits = sum(e.type == "enter" for e in events), sum(e.type == "exit" for e in events)
        return {"scope": "legacy", "currently_inside": None, "last_observed_inside": None,
                "total_in": entries, "total_out": exits, "net_entries": entries - exits, "freshness": "unavailable"}
    if scope != "run":
        raise HTTPException(422, "scope must be run or legacy")
    run = selected_run(db, run_id, video_source_id)
    area_id = selected_area(run, area_id)
    stats = db.get(RunAreaStats, (run.id, area_id))
    heartbeat = run.heartbeat_at
    current = (run.status == "running" and heartbeat is not None and
               (datetime.now(timezone.utc) - heartbeat).total_seconds() <= request.app.state.settings.stale_seconds)
    occupancy = stats.currently_inside if stats else 0
    entries, exits = (stats.total_in, stats.total_out) if stats else (0, 0)
    return {"scope": "run", "run_id": run.id, "area_id": area_id, "status": run.status,
            "currently_inside": occupancy if current else None, "last_observed_inside": occupancy if heartbeat else None,
            "total_in": entries, "total_out": exits, "net_entries": entries - exits,
            "lost_tracks": stats.lost_tracks if stats else 0, "ts": heartbeat,
            "freshness": "current" if current else "stale" if run.status in ACTIVE_STATUSES else "last_observed"}


def _bound(value, kind, default):
    if value is None:
        return default
    try:
        if kind == "file":
            result = float(value)
            if not math.isfinite(result) or result < 0:
                raise ValueError()
            return result
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError()
        return date.astimezone(timezone.utc).timestamp()
    except (ValueError, OverflowError) as error:
        raise HTTPException(422, "Use elapsed seconds for files or timezone-aware ISO dates for live sources") from error


def run_history(db, run, area_id, granularity="minute", start=None, end=None):
    seconds = {"minute": 60, "hour": 3600, "day": 86400}.get(granularity)
    if seconds is None:
        raise HTTPException(422, "granularity must be minute, hour or day")
    query = select(FrameSample).where(FrameSample.run_id == run.id)
    first = db.execute(query.order_by(FrameSample.frame_index).limit(1)).scalars().first()
    last = db.execute(query.order_by(FrameSample.frame_index.desc()).limit(1)).scalars().first()
    default_start = 0 if run.kind == "file" else (first.time_seconds if first else datetime.now(timezone.utc).timestamp())
    default_end = last.time_seconds + last.duration_seconds if last else default_start
    if start is None:
        # The dashboard/forecast need a recent bounded view; explicit queries can go further back.
        default_start = max(default_start, math.floor((default_end - 7200) / seconds) * seconds)
    start_seconds, end_seconds = _bound(start, run.kind, default_start), _bound(end, run.kind, default_end)
    if end_seconds < start_seconds or (end_seconds - start_seconds) / seconds > 10000:
        raise HTTPException(422, "Invalid or excessively wide history range")
    if start_seconds == end_seconds:
        return []
    left = math.floor(start_seconds / seconds) * seconds
    window = (func.floor(RunBucket.window_start / seconds) * seconds).label("window_start")
    rows = db.execute(select(window, func.sum(RunBucket.coverage_seconds).label("coverage"),
        func.sum(RunBucket.occupancy_seconds).label("occupancy"), func.sum(RunBucket.sample_count).label("samples"),
        func.sum(RunBucket.in_count).label("entries"), func.sum(RunBucket.out_count).label("exits"))
        .where(RunBucket.run_id == run.id, RunBucket.area_id == area_id,
               RunBucket.window_start >= left, RunBucket.window_start < end_seconds).group_by(window)).all()
    summaries = {float(row.window_start): row for row in rows}
    buckets = []
    while left < end_seconds:
        row = summaries.get(left)
        complete = left >= start_seconds and left + seconds <= end_seconds + 1e-7
        observed = bool(complete and row and row.coverage >= seconds - 1e-5)
        buckets.append({"window_start": left, "window_end": left + seconds, "complete": complete,
            "observed": observed, "in_count": int(row.entries) if observed else None,
            "out_count": int(row.exits) if observed else None,
            "currently_inside": row.occupancy / row.coverage if observed else None,
            "sample_count": int(row.samples) if row else 0})
        left += seconds
    return buckets


@router.get("/api/stats")
def history_stats(video_source_id: int | None = None, area_id: int | None = None, run_id: str | None = None,
                  scope: str = "run", granularity: str = "minute", start: str | None = None, end: str | None = None,
                  db: Session = Depends(get_db)):
    if scope == "legacy":
        seconds = {"minute": 60, "hour": 3600, "day": 86400}.get(granularity)
        if seconds is None:
            raise HTTPException(422, "Invalid granularity")
        query = select(Event).where(Event.run_id.is_(None))
        if video_source_id:
            query = query.where(Event.video_source_id == video_source_id)
        if area_id:
            query = query.where(Event.area_id == area_id)
        lower, upper = _bound(start, "live", -math.inf), _bound(end, "live", math.inf)
        grouped = {}
        for row in db.execute(query).scalars():
            epoch = row.timestamp.timestamp()
            if lower <= epoch < upper:
                left = math.floor(epoch / seconds) * seconds
                bucket = grouped.setdefault(left, {"window_start": datetime.fromtimestamp(left, timezone.utc).isoformat(),
                    "in_count": 0, "out_count": 0, "observed": None, "complete": None, "currently_inside": None})
                bucket["in_count" if row.type == "enter" else "out_count"] += 1
        return {"scope": "legacy", "timeline": "live", "buckets": [grouped[key] for key in sorted(grouped)],
                "summary": {"coverage": "Legacy records have no observation coverage"}}
    if scope != "run":
        raise HTTPException(422, "scope must be run or legacy")
    run = selected_run(db, run_id, video_source_id)
    area_id = selected_area(run, area_id)
    buckets = run_history(db, run, area_id, granularity, start, end)
    if run.kind == "live":
        for bucket in buckets:
            for key in ("window_start", "window_end"):
                bucket[key] = datetime.fromtimestamp(bucket[key], timezone.utc).isoformat()
    return {"scope": "run", "run_id": run.id, "area_id": area_id, "timeline": run.kind, "buckets": buckets,
            "summary": {"complete_observed_buckets": sum(b["complete"] and b["observed"] for b in buckets),
                        "default_recent_minutes": 120 if start is None else None}}
