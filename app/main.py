"""ASGI entry point. Run one API process; inference uses a spawned child."""
from app.application import create_app

app = create_app()
