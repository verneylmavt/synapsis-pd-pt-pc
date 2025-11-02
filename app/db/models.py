from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, Float, func
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class VideoSource(Base):
    __tablename__ = "video_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    areas: Mapped[List["Area"]] = relationship("Area", back_populates="video_source", cascade="all, delete-orphan")


class Area(Base):
    __tablename__ = "areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_source_id: Mapped[int] = mapped_column(Integer, ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    polygon: Mapped[dict] = mapped_column(JSONB, nullable=False)  # list of points [{x:float,y:float},...]

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    video_source: Mapped["VideoSource"] = relationship("VideoSource", back_populates="areas")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_source_id: Mapped[int] = mapped_column(Integer, ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False, index=True)
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_source_id: Mapped[int] = mapped_column(Integer, ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    area_id: Mapped[int] = mapped_column(Integer, ForeignKey("areas.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    tracker_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    type: Mapped[str] = mapped_column(String(16), nullable=False)  # "enter" or "exit"