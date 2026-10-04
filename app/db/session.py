from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Request

from app.core.config import settings

# Engine & Session factory
def create_session_factory(config=settings):
    engine = create_engine(config.database_url, future=True, pool_pre_ping=True, pool_timeout=5,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=2000 -c lock_timeout=2000",
                      "keepalives": 1, "keepalives_idle": 5, "keepalives_interval": 1, "keepalives_count": 2})
    return engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


engine, SessionLocal = create_session_factory()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# FastAPI dependency
def get_db(request: Request):
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()
