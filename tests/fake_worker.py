"""Small DB-free multiprocessing targets; no OpenCV or model dependencies."""
from datetime import datetime, timezone
import queue
import time


RUN_SETTINGS = {"model": "yolov8n.pt", "imgsz": 640, "conf": 0.25, "iou": 0.45, "device": "cpu"}


def frame_packet(spec, index, *, total_in=0, total_out=0, detection=False, event=None):
    area_id = spec["areas"][0]["id"]
    detections = [{"detection_index": 0, "tracker_id": "1", "cls": "person", "conf": 0.9,
                   "x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6, "inside_area_id": area_id}]
    return {"kind": "frame", "run_id": spec["run_id"], "frame_index": index, "segment_index": 0,
            "observed_at": datetime.now(timezone.utc).isoformat(), "media_time_ms": (index - 1) * 100,
            "time_seconds": (index - 1) * 0.1, "coverage_start_seconds": (index - 1) * 0.1,
            "duration_seconds": 0.1, "detections": detections if detection else [],
            "events": [{"area_id": area_id, "tracker_id": "1", "type": event}] if event else [],
            "stats": [{"area_id": area_id, "total_in": total_in, "total_out": total_out,
                       "currently_inside": 0, "lost_tracks": 0}],
            "jpeg": b"fake-jpeg", "fps": 10.0, "dropped_frames": 0}


def _put(output, stop, packet):
    while not stop.is_set():
        try:
            output.put(packet, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def quiet_worker(spec, output, stop):
    _put(output, stop, {"kind": "status", "status": "running", "error_code": None})
    index = 1
    while not stop.is_set():
        if not _put(output, stop, frame_packet(spec, index)):
            break
        index += 1
        stop.wait(0.05)


def eof_worker(spec, output, stop):
    _put(output, stop, {"kind": "status", "status": "running", "error_code": None})
    for index in range(1, 4):
        if not _put(output, stop, frame_packet(spec, index)):
            return
    _put(output, stop, {"kind": "status", "status": "completed", "error_code": None})


def blocked_worker(spec, output, stop):
    _put(output, stop, {"kind": "status", "status": "running", "error_code": None})
    _put(output, stop, frame_packet(spec, 1))
    while True:
        time.sleep(0.1)
