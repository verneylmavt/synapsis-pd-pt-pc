from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import Settings, settings
from app.db.session import create_session_factory


def create_app(config: Settings | None = None, *, manage_processing: bool = True) -> FastAPI:
    config = config or settings
    @asynccontextmanager
    async def lifespan(app):
        if manage_processing:
            app.state.manager.open()
        try:
            yield
        finally:
            if manage_processing:
                app.state.manager.close()
            app.state.engine.dispose()
    app = FastAPI(title=config.app_name, lifespan=lifespan)
    app.state.settings = config
    app.state.engine, app.state.session_factory = create_session_factory(config)
    from app.services.supervisor import RunManager
    app.state.manager = RunManager(config, app.state.engine, app.state.session_factory)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Validation input may contain a camera password; never echo it.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "msg": "Invalid value", "type": item["type"]}
            for item in error.errors()]})

    @app.get("/healthz")
    def health():
        return {"ok": True, "app": config.app_name}

    @app.get("/readyz")
    def ready():
        try:
            with app.state.manager.mutation_guard():
                app.state.manager._check_ownership()
            ready = True
        except Exception:
            ready = False
        return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/dashboard")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return FileResponse(config.project_root / "app" / "templates" / "dashboard.html")

    app.mount("/static", StaticFiles(directory=config.project_root / "app" / "static", check_dir=False), name="static")
    from app.api import sources, runs, stats, forecast
    app.include_router(sources.router)
    app.include_router(runs.router)
    app.include_router(stats.router)
    app.include_router(forecast.router)

    return app
