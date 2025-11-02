from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="normalized 0..1")
    y: float = Field(..., ge=0.0, le=1.0, description="normalized 0..1")


class AreaCreate(BaseModel):
    video_source_id: int
    name: str
    polygon: List[Point]

    @field_validator("polygon")
    @classmethod
    def polygon_has_min_vertices(cls, v: List[Point]):
        if len(v) < 3:
            raise ValueError("polygon must have at least 3 points")
        return v


class AreaOut(BaseModel):
    id: int
    video_source_id: int
    name: str
    polygon: List[Point]
    active: bool

    class Config:
        from_attributes = True


class VideoSourceOut(BaseModel):
    id: int
    name: str
    uri: str
    enabled: bool

    class Config:
        from_attributes = True