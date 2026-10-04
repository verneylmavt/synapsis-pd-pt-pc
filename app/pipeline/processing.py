from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from app.logic.counting import CountingEngine
from app.logic.geometry import point_in_polygon


TRACKER_PATH = Path(__file__).resolve().parents[2] / "configs" / "bytetrack.yaml"


def _numpy(value):
    if value is None:
        return None
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def normalize_results(results, width: int, height: int, areas: list[dict]) -> list[dict]:
    """Normalize original-resolution boxes, including detections without an ID."""
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _numpy(boxes.xyxy)
        if xyxy is None:
            continue
        confs = _numpy(boxes.conf)
        ids = _numpy(boxes.id)
        for index, raw in enumerate(xyxy):
            coords = np.clip(np.asarray(raw, dtype=float) / [width, height, width, height], 0, 1)
            if not np.isfinite(coords).all() or coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            x1, y1, x2, y2 = map(float, coords)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            inside = next((area["id"] for area in sorted(areas, key=lambda area: area["id"])
                           if point_in_polygon(center, area["polygon"])), None)
            confidence = float(confs[index]) if confs is not None else None
            if confidence is not None and not np.isfinite(confidence):
                confidence = None
            detections.append({"detection_index": len(detections),
                               "tracker_id": str(int(ids[index]))
                               if ids is not None and np.isfinite(ids[index]) else None,
                               "cls": "person", "conf": confidence,
                               "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "inside_area_id": inside})
    return detections


def draw_frame(frame, detections: list[dict], areas: list[dict]) -> bytes:
    import cv2
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = [int(detection[key] * scale) for key, scale in
                          [("x1", width), ("y1", height), ("x2", width), ("y2", height)]]
        color = (80, 220, 120) if detection["inside_area_id"] is not None else (220, 180, 70)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"person {detection['tracker_id'] or '-'}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for area in areas:
        polygon = np.array([[round(point["x"] * (width - 1)), round(point["y"] * (height - 1))]
                            for point in area["polygon"]], dtype=np.int32)
        cv2.polylines(annotated, [polygon], True, (80, 220, 120), 2)
        if len(polygon):
            cv2.putText(annotated, area.get("name", str(area["id"])), tuple(polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120), 1, cv2.LINE_AA)
    if width > 1280:
        annotated = cv2.resize(annotated, (1280, round(height * 1280 / width)))
    success, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        raise RuntimeError("jpeg_encode_failed")
    return jpeg.tobytes()


class FrameProcessor:
    """The worker and evaluator use the same detection/counting implementation."""

    def __init__(self, model, areas: list[dict], settings: dict):
        self.model = model
        self.areas = sorted(areas, key=lambda area: area["id"])
        self.settings = settings
        self.offsets = {area["id"]: {"total_in": 0, "total_out": 0, "lost_tracks": 0} for area in areas}
        self.last_stats = []
        self.engine = self._engine()

    def _engine(self):
        return CountingEngine(self.areas, confirmation=int(self.settings.get("confirmation", 3)),
                              missing_grace_seconds=float(self.settings.get("missing_grace_seconds", 2)))

    def reset_segment(self):
        for stat in self.last_stats:
            for key in ("total_in", "total_out", "lost_tracks"):
                self.offsets[stat["area_id"]][key] = stat[key]
        self.engine = self._engine()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()

    def process(self, frame, time_seconds: float, *, encode: bool = True):
        started = perf_counter()
        results = self.model.track(source=frame, stream=False, persist=True, classes=[0],
                                   tracker=str(TRACKER_PATH), imgsz=int(self.settings.get("imgsz", 640)),
                                   conf=float(self.settings.get("conf", 0.25)),
                                   iou=float(self.settings.get("iou", 0.45)),
                                   device=None if self.settings.get("device", "auto") == "auto"
                                   else self.settings["device"], verbose=False)
        height, width = frame.shape[:2]
        detections = normalize_results(results, width, height, self.areas)
        counted = self.engine.update(detections, time_seconds)
        for stat in counted["stats"]:
            for key in ("total_in", "total_out", "lost_tracks"):
                stat[key] += self.offsets[stat["area_id"]][key]
        self.last_stats = counted["stats"]
        jpeg = draw_frame(frame, detections, self.areas) if encode else b""
        return {"detections": detections, **counted, "jpeg": jpeg,
                "latency_seconds": perf_counter() - started}
