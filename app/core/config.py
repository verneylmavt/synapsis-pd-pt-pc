from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "people-counting")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://synapsis:synapsis@localhost:5432/synapsis",
    )


settings = Settings()