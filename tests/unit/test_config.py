from pathlib import Path

from app.core.config import Settings


def test_settings_resolve_resources_from_project_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Settings.from_env({}, load_file=False)
    assert config.upload_dir.is_absolute()
    assert config.model_dir.is_absolute()
    assert config.upload_dir == config.project_root / "data" / "uploads"
    assert config.database_url.startswith("postgresql+psycopg2://")


def test_runtime_limits_reject_invalid_configuration():
    import pytest

    with pytest.raises(ValueError):
        Settings.from_env({"MAX_ACTIVE_RUNS": "0"}, load_file=False)


def test_redaction_hides_camera_password_and_query():
    from app.core.security import redact_uri

    value = redact_uri("rtsp://operator:secret@192.168.1.5:554/path?token=hidden")
    assert value == "rtsp://192.168.1.5:554"
    assert "secret" not in value and "hidden" not in value
