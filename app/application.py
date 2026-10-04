from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import Settings, settings
from app.db.session import create_session_factory


def create_app(config: Settings | None = None) -> FastAPI:
    config = config or settings
    app = FastAPI(title=config.app_name)
    app.state.settings = config
    app.state.engine, app.state.session_factory = create_session_factory(config)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Validation input may contain a camera password; never echo it.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "msg": "Invalid value", "type": item["type"]}
            for item in error.errors()]})

    @app.get("/healthz")
    def health():
        return {"ok": True, "app": config.app_name}

    return app
