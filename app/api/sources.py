from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ACTIVE_STATUSES, Area, ProcessingRun, VideoSource
from app.db.session import get_db
from app.schemas import AreaCreate, AreaOut, AreaUpdate, VideoSourceCreate, VideoSourceOut
from app.services.media import MediaProbeError, preview_jpeg
from app.services.sources import UploadError, save_upload


router = APIRouter(prefix="/api", tags=["sources"])


def _source(db: Session, source_id: int) -> VideoSource:
    source = db.get(VideoSource, source_id)
    if source is None:
        raise HTTPException(404, "video_source_not_found")
    return source


def _active_run(db: Session, source_id: int):
    return db.scalar(select(ProcessingRun.id).where(
        ProcessingRun.video_source_id == source_id, ProcessingRun.status.in_(ACTIVE_STATUSES)).limit(1))


def _commit(db: Session) -> None:
    try:
        db.commit()
    except Exception as error:
        db.rollback()
        raise HTTPException(503, "database_unavailable") from error


@router.get("/video-sources", response_model=list[VideoSourceOut])
def list_sources(db: Session = Depends(get_db)):
    return db.scalars(select(VideoSource).order_by(VideoSource.id)).all()


@router.post("/video-sources", response_model=VideoSourceOut, status_code=201)
def create_source(payload: VideoSourceCreate, db: Session = Depends(get_db)):
    source = VideoSource(name=payload.name, uri=payload.uri, kind="live", enabled=True)
    db.add(source)
    _commit(db)
    db.refresh(source)
    return source


@router.post("/upload", response_model=VideoSourceOut, status_code=201)
@router.post("/upload-video", response_model=VideoSourceOut, status_code=201, include_in_schema=False)
async def upload_video(request: Request, file: UploadFile = File(...), name: str | None = Form(None, max_length=255), db: Session = Depends(get_db)):
    try:
        uploaded = await save_upload(file, request.app.state.settings)
    except UploadError as error:
        raise HTTPException(413 if error.code == "upload_too_large" else 422, error.code) from error
    except MediaProbeError as error:
        raise HTTPException(422, "unreadable_video") from error
    persisted = False
    try:
        source = VideoSource(name=name.strip() if name and name.strip() else uploaded.name, uri=str(uploaded.path), kind="file", enabled=True,
                             size_bytes=uploaded.size_bytes, **uploaded.metadata)
        db.add(source)
        _commit(db)
        persisted = True
        db.refresh(source)
        return source
    finally:
        if not persisted:
            uploaded.path.unlink(missing_ok=True)


@router.get("/preview/{source_id}")
@router.get("/video/first-frame/{source_id}", include_in_schema=False)
def preview_source(source_id: int, request: Request, db: Session = Depends(get_db)):
    manager = request.app.state.manager
    # Reserve this source, while leaving Stop and other sources responsive.
    with manager.mutation_guard():
        source = _source(db, source_id)
        active = _active_run(db, source_id)
        if active:
            image = manager.latest_source_jpeg(source_id)
            if image is None:
                raise HTTPException(409, "preview_not_ready")
            return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        else:
            try:
                manager.begin_preview(source_id)
            except ValueError as error:
                raise HTTPException(409, "preview_in_progress") from error
    try:
        image = preview_jpeg(source.uri)
    except MediaProbeError as error:
        raise HTTPException(422, "preview_unavailable") from error
    finally:
        manager.end_preview(source_id)
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/areas", response_model=list[AreaOut])
def list_areas(video_source_id: int | None = None, db: Session = Depends(get_db)):
    query = select(Area).order_by(Area.id)
    if video_source_id is not None:
        query = query.where(Area.video_source_id == video_source_id)
    return db.scalars(query).all()


@router.post("/areas", response_model=AreaOut, status_code=201)
def create_area(payload: AreaCreate, request: Request, db: Session = Depends(get_db)):
    with request.app.state.manager.mutation_guard():
        _source(db, payload.video_source_id)
        if _active_run(db, payload.video_source_id):
            raise HTTPException(409, "stop_run_before_editing_areas")
        area = Area(video_source_id=payload.video_source_id, name=payload.name,
                    polygon=[point.model_dump() for point in payload.polygon], active=payload.active)
        db.add(area)
        _commit(db)
        db.refresh(area)
        return area


@router.put("/areas/{area_id}", response_model=AreaOut)
def update_area(area_id: int, payload: AreaUpdate, request: Request, db: Session = Depends(get_db)):
    with request.app.state.manager.mutation_guard():
        area = db.get(Area, area_id)
        if area is None:
            raise HTTPException(404, "area_not_found")
        if _active_run(db, area.video_source_id):
            raise HTTPException(409, "stop_run_before_editing_areas")
        for key, value in payload.model_dump(exclude_none=True).items():
            setattr(area, key, value)
        _commit(db)
        db.refresh(area)
        return area
