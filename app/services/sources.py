from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from anyio import to_thread

from app.services.media import probe_video


class UploadError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class UploadedVideo:
    path: Path
    name: str
    size_bytes: int
    metadata: dict


def upload_name(filename: str | None) -> str:
    name = re.split(r"[/\\]", filename or "video")[-1]
    name = "".join(char for char in name if ord(char) >= 32).strip()
    return (name or "video")[:255]


async def save_upload(upload, settings, *, probe=None) -> UploadedVideo:
    """Write bounded chunks and validate decoding before a source is committed."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    name = upload_name(upload.filename)
    extension = Path(name).suffix.lower()
    if extension not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        extension = ".bin"
    path = settings.upload_dir / f"{uuid4().hex}{extension}"
    succeeded = False
    try:
        size = 0
        with path.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadError("upload_too_large")
                output.write(chunk)
        if size == 0:
            raise UploadError("empty_video")
        # Wait for the bounded child on cancellation before removing an open file.
        probing = asyncio.create_task(to_thread.run_sync(probe or probe_video, str(path)))
        try:
            metadata = await asyncio.shield(probing)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(probing)
            finally:
                raise
        result = UploadedVideo(path, name, size, metadata)
        succeeded = True
        return result
    finally:
        if not succeeded:
            path.unlink(missing_ok=True)
        await upload.close()
