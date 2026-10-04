import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ACTIVE_STATUSES, Event, ProcessingRun
from app.db.session import get_db

router = APIRouter()


class RunRequest(BaseModel):
    model: Literal["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"] = "yolov8n.pt"
    conf: float = Field(0.25, ge=0.01, le=0.95)
    iou: float = Field(0.45, ge=0.05, le=0.95)
    imgsz: int = Field(640, ge=320, le=1280, multiple_of=32)
    device: str = Field("auto", pattern=r"^(auto|cpu|[0-9]+|cuda:[0-9]+)$")


def run_out(run):
    return {key: getattr(run, key) for key in ("id", "video_source_id", "kind", "status", "settings", "areas",
        "created_at", "started_at", "ended_at", "heartbeat_at", "last_frame_index", "segment_index", "media_time_ms", "fps", "dropped_frames", "error_code")}


@router.post("/api/video-sources/{source_id}/runs", status_code=202)
def start_run(source_id: int, payload: RunRequest, request: Request, db: Session = Depends(get_db)):
    settings = payload.model_dump()
    if settings["device"] == "auto":
        settings["device"] = request.app.state.settings.device
    try:
        run_id = request.app.state.manager.start(source_id, settings)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(422, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    db.expire_all()
    return run_out(db.get(ProcessingRun, run_id))


@router.get("/api/video-sources/{source_id}/runs")
def list_runs(source_id: int, db: Session = Depends(get_db)):
    return [run_out(run) for run in db.execute(select(ProcessingRun).where(ProcessingRun.video_source_id == source_id)
        .order_by(ProcessingRun.created_at.desc()).limit(100)).scalars()]


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(ProcessingRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run_out(run)


@router.post("/api/runs/{run_id}/stop", status_code=202)
def stop_run(run_id: str, request: Request, db: Session = Depends(get_db)):
    manager = request.app.state.manager
    with manager.mutation_guard():
        runtime = manager.runtimes.get(run_id)
        if runtime and runtime.snapshot:
            manager.stop_run(run_id)
            return dict(runtime.snapshot)
    if db.get(ProcessingRun, run_id) is None:
        raise HTTPException(404, "Run not found")
    manager.stop_run(run_id)
    db.expire_all()
    return run_out(db.get(ProcessingRun, run_id))


def _next_frame(subscription):
    return next(subscription, None)


def stream_run(run_id, request):
    try:
        subscription = request.app.state.manager.subscribe(run_id)
    except LookupError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(429, str(error)) from error
    async def images():
        try:
            while not await request.is_disconnected():
                packet = await asyncio.to_thread(_next_frame, subscription)
                if packet is None:
                    break
                yield packet
        finally:
            subscription.close()
    return StreamingResponse(images(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.get("/api/runs/{run_id}/stream")
def run_stream(run_id: str, request: Request, db: Session = Depends(get_db)):
    if db.get(ProcessingRun, run_id) is None:
        raise HTTPException(404, "Run not found")
    return stream_run(run_id, request)


@router.get("/stream/{source_id}")
def source_stream(source_id: int, request: Request, run_id: str | None = None, db: Session = Depends(get_db)):
    query = select(ProcessingRun).where(ProcessingRun.video_source_id == source_id)
    query = query.where(ProcessingRun.id == run_id) if run_id else query.where(ProcessingRun.status.in_(ACTIVE_STATUSES))
    run = db.execute(query.order_by(ProcessingRun.created_at.desc())).scalars().first()
    if run is None:
        raise HTTPException(409, "Start a run explicitly before subscribing")
    return stream_run(run.id, request)


@router.get("/api/runs/{run_id}/events")
def recent_events(run_id: str, area_id: int | None = None, limit: int = 50, db: Session = Depends(get_db)):
    if db.get(ProcessingRun, run_id) is None:
        raise HTTPException(404, "Run not found")
    query = select(Event).where(Event.run_id == run_id)
    if area_id is not None:
        query = query.where(Event.area_id == area_id)
    rows = db.execute(query.order_by(Event.id.desc()).limit(max(1, min(limit, 200)))).scalars()
    return {"items": [{key: getattr(row, key) for key in ("id", "timestamp", "media_time_ms", "tracker_id", "type", "area_id")} for row in rows]}
