from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from typing import Mapping
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Synapsis people analytics"
    database_url: str = "postgresql+psycopg2://synapsis:synapsis@localhost:5432/synapsis"
    project_root: Path = PROJECT_ROOT
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    model_dir: Path = PROJECT_ROOT
    max_upload_bytes: int = 500 * 1024 * 1024
    max_active_runs: int = 1
    max_viewers: int = 8
    stale_seconds: float = 15.0
    stop_seconds: float = 5.0
    write_retries: int = 3
    device: str = "auto"

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None, *, load_file: bool = True):
        env = dict(dotenv_values(PROJECT_ROOT / ".env")) if load_file else {}
        env.update(os.environ if values is None else values)
        url = env.get("DATABASE_URL") or cls.database_url
        url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://")
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        def resource(key, default):
            path = Path(env.get(key) or default)
            return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        config = cls(
            app_name=env.get("APP_NAME") or cls.app_name, database_url=url,
            upload_dir=resource("UPLOAD_DIR", "data/uploads"), model_dir=resource("MODEL_DIR", "."),
            max_upload_bytes=int(env.get("MAX_UPLOAD_BYTES") or cls.max_upload_bytes),
            max_active_runs=int(env.get("MAX_ACTIVE_RUNS") or cls.max_active_runs),
            max_viewers=int(env.get("MAX_VIEWERS") or cls.max_viewers),
            stale_seconds=float(env.get("STALE_SECONDS") or cls.stale_seconds),
            stop_seconds=float(env.get("STOP_SECONDS") or cls.stop_seconds),
            write_retries=int(env.get("WRITE_RETRIES") or cls.write_retries),
            device=env.get("MODEL_DEVICE") or "auto",
        )
        for field in ("max_upload_bytes", "max_active_runs", "max_viewers", "stale_seconds", "stop_seconds", "write_retries"):
            if getattr(config, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if config.max_active_runs != 1:
            raise ValueError("This single-worker application supports MAX_ACTIVE_RUNS=1")
        return config


settings = Settings.from_env()
