import asyncio
import importlib.util
import threading

import pytest
from pydantic import ValidationError

from app import schemas
from app.core.config import Settings


def service():
    assert importlib.util.find_spec("app.services.sources"), "upload service is missing"
    from app.services import sources
    return sources


@pytest.mark.parametrize("uri", ["file:///etc/passwd", "ftp://192.168.1.2/video", "http://169.254.169.254/token", "rtsp://0.0.0.0/video"])
def test_camera_schema_rejects_invalid_destinations(uri):
    assert hasattr(schemas, "VideoSourceCreate"), "camera schema is missing"
    with pytest.raises(ValidationError):
        schemas.VideoSourceCreate(name="camera", uri=uri)


def test_camera_schema_accepts_lan_and_source_output_redacts_all_secrets():
    assert hasattr(schemas, "VideoSourceCreate"), "camera schema is missing"
    source = schemas.VideoSourceCreate(name="  Lobby  ", uri="rtsp://user:secret@192.168.1.2/video?token=secret")
    assert source.name == "Lobby"
    output = schemas.VideoSourceOut(id=1, name=source.name, uri=source.uri, enabled=True, kind="live")
    assert output.uri == "rtsp://192.168.1.2"
    output = schemas.VideoSourceOut(id=2, name="clip", uri="data/uploads/private.mp4", enabled=True)
    assert output.uri == "local video"


def test_area_schema_rejects_bow_tie_and_blank_names():
    with pytest.raises(ValidationError):
        schemas.AreaCreate(video_source_id=1, name="area", polygon=[
            {"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}, {"x": 1, "y": 0}])
    with pytest.raises(ValidationError):
        schemas.AreaCreate(video_source_id=1, name=" ", polygon=[
            {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])


class Upload:
    filename = "../../camera\\secret.mp4"

    def __init__(self, chunks, failure=None):
        self.chunks = list(chunks)
        self.closed = False
        self.failure = failure

    async def read(self, size):
        if self.chunks:
            return self.chunks.pop(0)
        if self.failure:
            raise self.failure
        return b""

    async def close(self):
        self.closed = True


def config(tmp_path, limit=10):
    return Settings(upload_dir=tmp_path, max_upload_bytes=limit)


def test_successful_upload_uses_uuid_and_actual_size(tmp_path):
    upload = Upload([b"123", b"45"])
    metadata = {"width": 640, "height": 360, "fps": 25.0, "duration_seconds": 2.0}
    result = asyncio.run(service().save_upload(upload, config(tmp_path), probe=lambda path: metadata))
    assert result.name == "secret.mp4"
    assert result.size_bytes == 5
    assert result.path.parent == tmp_path
    assert len(result.path.stem) == 32
    assert result.path.read_bytes() == b"12345"
    assert upload.closed


@pytest.mark.parametrize("failure", ["oversize", "empty", "decode", "cancel"])
def test_failed_uploads_remove_partial_files_and_close_input(tmp_path, failure):
    module = service()
    upload = Upload([b"12345", b"678901"] if failure == "oversize" else ([] if failure == "empty" else [b"data"]))
    if failure == "cancel":
        upload.failure = asyncio.CancelledError()
    def probe(path):
        raise ValueError("decode failed")
    expected = asyncio.CancelledError if failure == "cancel" else (module.UploadError if failure != "decode" else ValueError)
    with pytest.raises(expected):
        asyncio.run(module.save_upload(upload, config(tmp_path), probe=probe))
    assert list(tmp_path.iterdir()) == []
    assert upload.closed


def test_filename_name_is_basename_bounded_and_control_free():
    module = service()
    assert module.upload_name("../../secret\\clip.mp4") == "clip.mp4"
    assert len(module.upload_name("a" * 300)) == 255
    assert "\n" not in module.upload_name("bad\nname.mp4")


def test_camera_route_registers_without_network_and_redacts_response():
    try:
        available = importlib.util.find_spec("app.api.sources")
    except ModuleNotFoundError:
        available = None
    assert available, "source routes are missing"
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.sources import router
    from app.db.session import get_db
    class DB:
        def add(self, source):
            self.source = source
        def commit(self):
            self.source.id = 1
            self.source.enabled = True
        def refresh(self, source):
            pass
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: DB()
    with TestClient(app) as client:
        response = client.post("/api/video-sources", json={"name": "Lobby", "uri": "rtsp://user:secret@192.168.1.2/feed?token=secret"})
    assert response.status_code == 201
    assert response.json()["uri"] == "rtsp://192.168.1.2"
    assert response.json()["kind"] == "live"


def test_cancellation_waits_for_decoder_before_removing_file(tmp_path):
    started = threading.Event()
    release = threading.Event()
    probe_saw_file = []
    def probe(path):
        started.set()
        release.wait(timeout=2)
        from pathlib import Path
        probe_saw_file.append(Path(path).exists())
        return {"width": 1, "height": 1, "fps": 1, "duration_seconds": 1}
    async def scenario():
        task = asyncio.create_task(service().save_upload(Upload([b"data"]), config(tmp_path), probe=probe))
        while not started.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0.03)
        present_while_decoding = bool(list(tmp_path.iterdir()))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.03)
        assert present_while_decoding, "upload was removed while decoder still used it"
        assert probe_saw_file == [True]
        assert list(tmp_path.iterdir()) == []
    asyncio.run(scenario())


def test_spawned_probe_decodes_metadata_and_preview_without_loading_model(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from app.services.media import preview_jpeg, probe_video
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    metadata = probe_video(str(path))
    assert metadata == {"width": 64, "height": 48, "fps": 10.0, "duration_seconds": 0.3}
    image = cv2.imdecode(np.frombuffer(preview_jpeg(str(path)), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (48, 64)


def test_spawned_probe_errors_are_sanitized_and_timeout_reaps_child(tmp_path):
    import multiprocessing
    from app.services.media import MediaProbeError, probe_video
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(MediaProbeError, match="unreadable_video"):
        probe_video(str(tmp_path / "private-token.mp4"))
    with pytest.raises(MediaProbeError, match="video_probe_timeout"):
        probe_video(str(tmp_path / "private-token.mp4"), timeout_seconds=0.00001)
    assert {child.pid for child in multiprocessing.active_children()} == before
