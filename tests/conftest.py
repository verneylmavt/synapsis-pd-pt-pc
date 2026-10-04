import os
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db_engine():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL database")
    from alembic import command
    from alembic.config import Config
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.attributes["database_url"] = url
    command.upgrade(config, "head")
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE frame_samples,run_area_stats,events,detections,processing_runs,areas,video_sources RESTART IDENTITY CASCADE"))
    yield engine
    engine.dispose()


@pytest.fixture
def db_factory(db_engine):
    return sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest.fixture
def api_client(db_engine, tmp_path):
    from fastapi.testclient import TestClient
    from app.application import create_app
    config = replace(Settings.from_env({}, load_file=False), database_url=str(db_engine.url.render_as_string(hide_password=False)), upload_dir=tmp_path)
    app = create_app(config, manage_processing=False)
    with TestClient(app) as client:
        yield client
    app.state.engine.dispose()
