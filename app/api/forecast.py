from fastapi import APIRouter, Depends
from datetime import datetime, timezone
from fastapi import Request
from sqlalchemy.orm import Session

from app.api.stats import selected_run, selected_area, run_history
from app.db.session import get_db
from app.services.forecast import forecast_counts

router = APIRouter()


@router.get("/api/forecast")
def forecast(run_id: str, area_id: int, request: Request, db: Session = Depends(get_db)):
    run = selected_run(db, run_id)
    area_id = selected_area(run, area_id)
    if run.kind == "live" and (run.status != "running" or run.heartbeat_at is None or
        (datetime.now(timezone.utc) - run.heartbeat_at).total_seconds() > request.app.state.settings.stale_seconds):
        return {"status": "insufficient_data", "reason": "live_run_not_current", "run_id": run.id,
                "area_id": area_id, "timeline": "live", "label": "Live observation is unavailable",
                "horizon_minutes": 5, "predictions": [], "calibrated_intervals": False}
    result = forecast_counts(run_history(db, run, area_id))
    if result["status"] == "ok":
        result["status"] = "ready"
    result.update(run_id=run.id, area_id=area_id, timeline=run.kind,
        label="Hypothetical video continuation" if run.kind == "file" else "Next five live minutes")
    return result
