from uuid import uuid4

import pytest

from app.db.models import Area, ProcessingRun, VideoSource


pytestmark = pytest.mark.integration


def camera(api_client):
    response = api_client.post("/api/video-sources", json={"name": "Lobby", "uri": "rtsp://user:password@192.168.1.2/feed?token=secret"})
    assert response.status_code == 201
    assert response.json()["uri"] == "rtsp://192.168.1.2"
    return response.json()["id"]


def polygon():
    return [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.9, "y": 0.9}]


def test_register_list_and_validate_camera_without_probe(api_client, monkeypatch):
    from app.services import media
    monkeypatch.setattr(media, "probe_video", lambda *args: pytest.fail("registration must not open the camera"))
    source_id = camera(api_client)
    sources = api_client.get("/api/video-sources").json()
    assert next(source for source in sources if source["id"] == source_id)["kind"] == "live"
    response = api_client.post("/api/video-sources", json={"name": "bad", "uri": "file:///tmp/video"})
    assert response.status_code == 422
    assert "password" not in str(sources) and "token" not in str(sources)


def test_new_disabled_area_and_upload_display_name_are_preserved(api_client, monkeypatch):
    from app.services import sources
    source_id = camera(api_client)
    area = api_client.post("/api/areas", json={"video_source_id": source_id, "name": "next", "polygon": polygon(), "active": False})
    assert area.status_code == 201 and area.json()["active"] is False
    monkeypatch.setattr(sources, "probe_video", lambda path: {"width": 320, "height": 240, "fps": 25, "duration_seconds": 1})
    uploaded = api_client.post("/api/upload", data={"name": "  Lobby clip  "}, files={"file": ("clip.mp4", b"valid", "video/mp4")})
    assert uploaded.status_code == 201 and uploaded.json()["name"] == "Lobby clip"


def test_area_create_edit_disable_and_active_run_guard(api_client, db_factory):
    source_id = camera(api_client)
    response = api_client.post("/api/areas", json={"video_source_id": source_id, "name": "Door", "polygon": polygon()})
    assert response.status_code == 201
    area_id = response.json()["id"]
    response = api_client.put(f"/api/areas/{area_id}", json={"name": "Lobby door", "active": False})
    assert response.status_code == 200 and response.json()["active"] is False
    assert api_client.get(f"/api/areas?video_source_id={source_id}").json()[0]["name"] == "Lobby door"
    with db_factory() as db:
        db.add(ProcessingRun(id=str(uuid4()), video_source_id=source_id, kind="live", status="starting",
                             owner_id=str(uuid4()), settings={}, areas=[]))
        db.commit()
    assert api_client.put(f"/api/areas/{area_id}", json={"active": True}).status_code == 409
    assert api_client.post("/api/areas", json={"video_source_id": source_id, "name": "Other", "polygon": polygon()}).status_code == 409


def test_invalid_polygon_and_missing_sources(api_client):
    source_id = camera(api_client)
    invalid = [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}, {"x": 1, "y": 0}]
    assert api_client.post("/api/areas", json={"video_source_id": source_id, "name": "bad", "polygon": invalid}).status_code == 422
    assert api_client.post("/api/areas", json={"video_source_id": 2147483647, "name": "zone", "polygon": polygon()}).status_code == 404
    assert api_client.get("/api/preview/2147483647").status_code == 404


def test_upload_aliases_persist_metadata_and_bad_decode_cleans(api_client, monkeypatch):
    from app.services import sources
    monkeypatch.setattr(sources, "probe_video", lambda path: {"width": 320, "height": 240, "fps": 25, "duration_seconds": 1})
    for endpoint in ("/api/upload", "/api/upload-video"):
        response = api_client.post(endpoint, files={"file": ("../../clip.mp4", b"video", "video/mp4")})
        assert response.status_code == 201
        assert response.json()["uri"] == "local video"
        assert response.json()["width"] == 320
        assert response.json()["size_bytes"] == 5
        assert response.json()["name"] == "clip.mp4"
    before = set(api_client.app.state.settings.upload_dir.iterdir())
    def invalid(path):
        from app.services.media import MediaProbeError
        raise MediaProbeError("unreadable_video")
    monkeypatch.setattr(sources, "probe_video", invalid)
    assert api_client.post("/api/upload", files={"file": ("bad.mp4", b"bad", "video/mp4")}).status_code == 422
    assert set(api_client.app.state.settings.upload_dir.iterdir()) == before


def test_active_preview_uses_shared_frame_without_opening_source(api_client, db_factory, monkeypatch):
    from app.api import sources
    source_id = camera(api_client)
    with db_factory() as db:
        db.add(ProcessingRun(id=str(uuid4()), video_source_id=source_id, kind="live", status="running",
                             owner_id=str(uuid4()), settings={}, areas=[]))
        db.commit()
    monkeypatch.setattr(api_client.app.state.manager, "latest_source_jpeg", lambda source: b"shared-jpeg")
    monkeypatch.setattr(sources, "preview_jpeg", lambda *args: pytest.fail("active preview cannot open camera"))
    assert api_client.get(f"/api/preview/{source_id}").content == b"shared-jpeg"
    assert api_client.get(f"/api/video/first-frame/{source_id}").content == b"shared-jpeg"


def test_upload_database_failure_cleans_validated_file(api_client, monkeypatch):
    from app.db.session import get_db
    from app.services import sources
    monkeypatch.setattr(sources, "probe_video", lambda path: {"width": 320, "height": 240, "fps": 25, "duration_seconds": 1})
    class BrokenDB:
        def add(self, value):
            pass
        def commit(self):
            raise RuntimeError("database failure")
        def rollback(self):
            pass
    before = set(api_client.app.state.settings.upload_dir.iterdir())
    api_client.app.dependency_overrides[get_db] = lambda: BrokenDB()
    try:
        response = api_client.post("/api/upload", files={"file": ("clip.mp4", b"video", "video/mp4")})
        assert response.status_code == 503
        assert set(api_client.app.state.settings.upload_dir.iterdir()) == before
    finally:
        api_client.app.dependency_overrides.pop(get_db, None)
