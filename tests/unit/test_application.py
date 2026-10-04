from fastapi.testclient import TestClient

from app.application import create_app
from app.core.config import Settings


def test_factory_can_import_and_serve_health_without_model_or_database():
    app = create_app(Settings.from_env({}, load_file=False))
    with TestClient(app) as client:
        assert client.get("/healthz").json()["ok"] is True


def test_validation_never_echoes_secret_input():
    from pydantic import BaseModel
    class Payload(BaseModel):
        number: int
    app = create_app(Settings.from_env({}, load_file=False))
    @app.post("/validate")
    def validate(payload: Payload):
        return payload
    with TestClient(app) as client:
        response = client.post("/validate", json={"number": "camera-password"})
        assert response.status_code == 422
        assert "camera-password" not in response.text
