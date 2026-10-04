import os
import json
from pathlib import Path

import pytest
from sqlalchemy import inspect, text


def migration_config():
    from alembic.config import Config
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.attributes["database_url"] = os.environ["TEST_DATABASE_URL"]
    return config


def test_legacy_rows_survive_additive_upgrade_and_downgrade(db_engine):
    """Use only fixture's disposable TEST_DATABASE_URL; restore head even on failure."""
    from alembic import command
    config = migration_config()
    try:
        command.downgrade(config, "0002_create_detections_events")
        with db_engine.begin() as db:
            db.execute(text("INSERT INTO video_sources (id,name,uri) VALUES (101,'legacy upload','data/uploads/old.mp4'),(102,'legacy camera','rtsp://192.168.1.10/live')"))
            db.execute(text("INSERT INTO areas (id,video_source_id,name,polygon) VALUES (201,101,'legacy zone',CAST(:polygon AS JSONB))"),
                       {"polygon": json.dumps([{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])})
            db.execute(text("INSERT INTO detections (video_source_id,tracker_id,cls,x1,y1,x2,y2,inside_area_id) VALUES (101,'7','person',0.1,0.1,0.8,0.8,201)"))
            db.execute(text("INSERT INTO events (video_source_id,area_id,tracker_id,type) VALUES (101,201,'7','enter')"))
        command.upgrade(config, "head")
        with db_engine.connect() as db:
            assert db.execute(text("SELECT kind FROM video_sources WHERE id=101")).scalar_one() == "file"
            assert db.execute(text("SELECT kind FROM video_sources WHERE id=102")).scalar_one() == "live"
            assert db.execute(text("SELECT run_id,tracker_id FROM detections")).one() == (None, "7")
            assert db.execute(text("SELECT run_id,tracker_id,type FROM events")).one() == (None, "7", "enter")
            indexes = {row["name"]: row for row in inspect(db).get_indexes("processing_runs")}
            assert indexes["uq_active_source_run"]["unique"]
        command.downgrade(config, "0002_create_detections_events")
        with db_engine.connect() as db:
            assert db.execute(text("SELECT tracker_id,type FROM events")).one() == ("7", "enter")
            assert db.execute(text("SELECT tracker_id FROM detections")).scalar_one() == "7"
    finally:
        command.upgrade(config, "head")
