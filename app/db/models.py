from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, Float, func, Index, UniqueConstraint, CheckConstraint, text
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class VideoSource(Base):
    __tablename__ = "video_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, server_default="file")
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    areas: Mapped[List["Area"]] = relationship("Area", back_populates="video_source", cascade="all, delete-orphan")


class Area(Base):
    __tablename__ = "areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_source_id: Mapped[int] = mapped_column(Integer, ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    polygon: Mapped[dict] = mapped_column(JSONB, nullable=False)  # list of points [{x:float,y:float},...]

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    video_source: Mapped["VideoSource"] = relationship("VideoSource", back_populates="areas")


class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_vs_ts", "video_source_id", "timestamp"),
        UniqueConstraint("run_id", "frame_index", "detection_index", name="uq_detection_run_frame"),
        CheckConstraint("run_id IS NULL OR (frame_index IS NOT NULL AND detection_index IS NOT NULL AND segment_index IS NOT NULL AND frame_index >= 1 AND detection_index >= 0 AND segment_index >= 0)", name="ck_detection_run_frame"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_source_id: Mapped[int] = mapped_column(Integer, ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("processing_runs.id", ondelete="CASCADE"))
    frame_index: Mapped[Optional[int]] = mapped_column(Integer)
    detection_index: Mapped[Optional[int]] = mapped_column(Integer)
    segment_index: Mapped[Optional[int]] = mapped_column(Integer)
    media_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    tracker_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cls: Mapped[str] = mapped_column(String(64), nullable=False)  # "person"
    conf: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)

    inside_area_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("areas.id", ondelete="SET NULL"), nullable=True)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_area_ts", "area_id", "timestamp"),
        UniqueConstraint("run_id", "segment_index", "frame_index", "area_id", "tracker_id", "type", name="uq_event_run_frame"),
        CheckConstraint("run_id IS NULL OR (type IN ('enter', 'exit') AND frame_index IS NOT NULL AND segment_index IS NOT NULL AND frame_index >= 1 AND segment_index >= 0 AND tracker_id IS NOT NULL)", name="ck_event_run_frame"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_source_id: Mapped[int] = mapped_column(Integer, ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("processing_runs.id", ondelete="CASCADE"))
    frame_index: Mapped[Optional[int]] = mapped_column(Integer)
    segment_index: Mapped[Optional[int]] = mapped_column(Integer)
    media_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    area_id: Mapped[int] = mapped_column(Integer, ForeignKey("areas.id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    tracker_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    type: Mapped[str] = mapped_column(String(16), nullable=False)  # "enter" or "exit"


ACTIVE_STATUSES = ("starting", "running", "reconnecting", "stopping")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    __table_args__ = (
        CheckConstraint("status IN ('starting','running','reconnecting','stopping','completed','stopped','failed','interrupted')", name="ck_run_status"),
        Index("uq_active_source_run", "video_source_id", unique=True, postgresql_where=text("status IN ('starting','running','reconnecting','stopping')")),
        Index("ix_runs_source_created", "video_source_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    video_source_id: Mapped[int] = mapped_column(ForeignKey("video_sources.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
    areas: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_frame_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    media_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    dropped_frames: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_code: Mapped[Optional[str]] = mapped_column(String(64))


class RunAreaStats(Base):
    __tablename__ = "run_area_stats"
    run_id: Mapped[str] = mapped_column(ForeignKey("processing_runs.id", ondelete="CASCADE"), primary_key=True)
    area_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    total_in: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_out: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    currently_inside: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    lost_tracks: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class FrameSample(Base):
    __tablename__ = "frame_samples"
    run_id: Mapped[str] = mapped_column(ForeignKey("processing_runs.id", ondelete="CASCADE"), primary_key=True)
    frame_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    stats: Mapped[list] = mapped_column(JSONB, nullable=False)


class RunBucket(Base):
    __tablename__ = "run_buckets"
    run_id: Mapped[str] = mapped_column(ForeignKey("processing_runs.id", ondelete="CASCADE"), primary_key=True)
    area_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    window_start: Mapped[float] = mapped_column(Float, primary_key=True)
    coverage_seconds: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    occupancy_seconds: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    in_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    out_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
