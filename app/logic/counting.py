from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from app.logic.geometry import point_in_polygon, validate_polygon
from app.logic.hysteresis import HysteresisState


@dataclass
class _Track:
    last_seen: float
    areas: dict[int, HysteresisState]


class CountingEngine:
    """Pure, segment-scoped observed counting, independent of storage and inference."""

    def __init__(self, areas: list[dict], confirmation: int = 3,
                 missing_grace_seconds: float = 2.0):
        if isinstance(confirmation, bool) or not isinstance(confirmation, int) or confirmation < 1:
            raise ValueError("Confirmation must be a positive integer")
        if not isfinite(missing_grace_seconds) or missing_grace_seconds < 0:
            raise ValueError("Missing grace must be finite and nonnegative")
        self.areas = {area["id"]: validate_polygon(area["polygon"]) for area in areas}
        if len(self.areas) != len(areas):
            raise ValueError("Area IDs must be unique")
        self.confirmation = confirmation
        self.missing_grace_seconds = missing_grace_seconds
        self._tracks: dict[str, _Track] = {}
        self._totals = {area_id: {"total_in": 0, "total_out": 0, "lost_tracks": 0}
                        for area_id in self.areas}
        self._last_time: float | None = None

    def update(self, detections: list[dict], time_seconds: float) -> dict:
        if not isfinite(time_seconds) or (self._last_time is not None and time_seconds < self._last_time):
            raise ValueError("Observation time must be finite and nondecreasing")
        self._last_time = time_seconds
        # Expire before observations so a long discontinuity cannot reuse old state.
        for tracker_id, track in list(self._tracks.items()):
            if time_seconds - track.last_seen > self.missing_grace_seconds:
                for area_id, state in track.areas.items():
                    if state.inside:
                        self._totals[area_id]["lost_tracks"] += 1
                del self._tracks[tracker_id]

        observed: dict[str, dict] = {}
        for detection in detections:
            tracker_id = detection.get("tracker_id")
            if tracker_id is None:
                continue
            tracker_id = str(tracker_id)
            previous = observed.get(tracker_id)
            if previous is None or (detection.get("conf") or 0) > (previous.get("conf") or 0):
                observed[tracker_id] = detection
        for tracker_id, track in self._tracks.items():
            if tracker_id not in observed:
                for state in track.areas.values():
                    state.reset_pending()

        events = []
        for tracker_id, detection in observed.items():
            track = self._tracks.get(tracker_id)
            if track is None:
                track = _Track(time_seconds, {area_id: HysteresisState() for area_id in self.areas})
                self._tracks[tracker_id] = track
            track.last_seen = time_seconds
            center = ((detection["x1"] + detection["x2"]) / 2,
                      (detection["y1"] + detection["y2"]) / 2)
            for area_id in sorted(self.areas):
                event_type = track.areas[area_id].update(
                    point_in_polygon(center, self.areas[area_id]), self.confirmation)
                if event_type:
                    events.append({"area_id": area_id, "tracker_id": tracker_id, "type": event_type})
                    key = "total_in" if event_type == "enter" else "total_out"
                    self._totals[area_id][key] += 1
        stats = [{"area_id": area_id, **self._totals[area_id],
                  "currently_inside": sum(track.areas[area_id].inside for track in self._tracks.values())}
                 for area_id in sorted(self.areas)]
        return {"events": events, "stats": stats}
