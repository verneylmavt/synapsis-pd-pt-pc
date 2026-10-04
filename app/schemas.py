from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.security import redact_uri, validate_camera_uri
from app.logic.geometry import validate_polygon


def _name(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 255 or any(ord(char) < 32 for char in value):
        raise ValueError("Name must contain 1 to 255 characters without control characters")
    return value


class Point(BaseModel):
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)


class AreaCreate(BaseModel):
    video_source_id: int = Field(gt=0)
    name: str
    polygon: list[Point]
    active: bool = True

    _clean_name = field_validator("name")(_name)

    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls, value):
        validate_polygon([point.model_dump() for point in value])
        return value


class AreaUpdate(BaseModel):
    name: str | None = None
    polygon: list[Point] | None = None
    active: bool | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        return _name(value) if value is not None else value

    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls, value):
        if value is not None:
            validate_polygon([point.model_dump() for point in value])
        return value

    @model_validator(mode="after")
    def has_changes(self):
        if self.name is None and self.polygon is None and self.active is None:
            raise ValueError("Provide a name, polygon or active flag")
        return self


class AreaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    video_source_id: int
    name: str
    polygon: list[Point]
    active: bool


class VideoSourceCreate(BaseModel):
    name: str
    uri: str

    _clean_name = field_validator("name")(_name)
    _valid_uri = field_validator("uri")(validate_camera_uri)


class VideoSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    uri: str
    enabled: bool
    kind: Literal["file", "live"] = "file"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None

    _redact_uri = field_validator("uri")(redact_uri)
