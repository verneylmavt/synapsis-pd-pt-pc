from __future__ import annotations

import math
import multiprocessing
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.pipeline.capture import LatestCapture, OpenCVCapture
from app.pipeline.processing import FrameProcessor


def load_model(settings: dict):
    if not Path(settings["model_path"]).is_file():
        raise ValueError("local model weights missing")
    from ultralytics import YOLO
    return YOLO(settings["model_path"])


def _put(output_queue, packet, stop_event, *, terminal=False):
    deadline = time.monotonic() + 0.5 if terminal and stop_event.is_set() else None
    while terminal or not stop_event.is_set():
        try:
            output_queue.put(packet, timeout=0.1)
            return True
        except queue.Full:
            if terminal and stop_event.is_set():
                if deadline is None:
                    deadline = time.monotonic() + 0.5
                if time.monotonic() >= deadline:
                    return False
    return False


def _status(output_queue, stop_event, status, error_code=None):
    return _put(output_queue, {"kind": "status", "status": status, "error_code": error_code},
                stop_event, terminal=status in {"completed", "stopped", "failed"})


def _watch_parent():
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    def watch():
        while parent.is_alive():
            time.sleep(1)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def worker_main(spec: dict, output_queue, stop_event, *, capture_factory=None, model_factory=None):
    """Spawn entry point: no database imports, network results, or raw error messages."""
    # Native backend diagnostics can otherwise print credential-bearing URLs.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    capture_factory = capture_factory or OpenCVCapture
    model_factory = model_factory or load_model
    _watch_parent()
    settings = spec["settings"]
    if stop_event.is_set():
        _status(output_queue, stop_event, "stopped")
        return
    try:
        processor = FrameProcessor(model_factory(settings), spec["areas"], settings)
    except Exception:
        _status(output_queue, stop_event, "failed", "model_load_failed")
        return
    live = spec["kind"] == "live"
    frame_index = 0
    segment_index = 0
    dropped_total = 0
    failures = 0
    previous_media_ms = -1.0
    media_origin_ms = None
    last_live_utc = None
    covered_live_until = None
    last_file_time = None
    while not stop_event.is_set():
        capture = None
        reader = None
        segment_opened_utc = time.time()
        error_code = "source_open_failed"
        try:
            capture = capture_factory(spec["uri"], settings)
            source_fps = capture.fps
            if not live and (not math.isfinite(source_fps) or source_fps <= 0):
                raise RuntimeError("invalid_video_fps")
            if live:
                reader = LatestCapture(capture, stop_event)
            running_sent = False
            while not stop_event.is_set():
                started = time.monotonic()
                if live:
                    received = reader.next(float(settings.get("no_frame_timeout_seconds", 15)))
                    if received is None:
                        break
                    frame, observed_at, time_seconds = received
                    media_ms = None
                else:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        _status(output_queue, stop_event, "completed")
                        return
                    observed_at = datetime.now(timezone.utc)
                    media_ms = capture.media_time_ms()
                    if math.isfinite(media_ms) and media_ms >= 0:
                        if media_origin_ms is None:
                            media_origin_ms = media_ms
                        media_ms -= media_origin_ms
                    fallback_ms = frame_index * 1000 / source_fps
                    fallback_used = not math.isfinite(media_ms) or media_ms < 0 or media_ms <= previous_media_ms
                    if fallback_used:
                        media_ms = max(fallback_ms, previous_media_ms + 1000 / source_fps)
                    previous_media_ms = media_ms
                    time_seconds = media_ms / 1000
                error_code = "inference_failed"
                processed = processor.process(frame, time_seconds)
                frame_index += 1
                if not running_sent:
                    if not _status(output_queue, stop_event, "running"):
                        break
                    running_sent = True
                    failures = 0
                if live:
                    epoch = observed_at.timestamp()
                    interval = 1 / max(float(settings.get("live_fps", 5)), 0.1)
                    coverage_start = max(segment_opened_utc, epoch - interval)
                    if last_live_utc is not None and 0 <= epoch - last_live_utc <= 2:
                        coverage_start = last_live_utc
                    if covered_live_until is not None:
                        coverage_start = max(coverage_start, covered_live_until)
                    duration = max(0, epoch - coverage_start)
                    covered_live_until = max(covered_live_until or epoch, epoch)
                    last_live_utc = epoch
                else:
                    coverage_start = time_seconds if last_file_time is None else last_file_time
                    duration = max(0, time_seconds - coverage_start)
                    last_file_time = time_seconds
                packet = {"kind": "frame", "run_id": spec["run_id"], "frame_index": frame_index,
                          "segment_index": segment_index, "observed_at": observed_at.isoformat(),
                          "media_time_ms": media_ms, "time_seconds": time_seconds,
                          "media_time_source": None if live else "fps_fallback" if fallback_used else "container",
                          "coverage_start_seconds": coverage_start, "duration_seconds": duration,
                          **processed, "fps": 1 / max(time.monotonic() - started, 1e-9),
                          "dropped_frames": dropped_total + (reader.dropped if reader else 0)}
                if not _put(output_queue, packet, stop_event):
                    break
                error_code = "source_read_failed"
                if live:
                    stop_event.wait(max(0, interval - (time.monotonic() - started)))
        except Exception as error:
            if str(error) in {"source_no_frames", "source_read_failed", "source_open_failed", "invalid_video_fps",
                              "jpeg_encode_failed"}:
                error_code = str(error)
            if stop_event.is_set():
                break
            if not live or error_code in {"inference_failed", "jpeg_encode_failed"}:
                _status(output_queue, stop_event, "failed", error_code)
                return
            failures += 1
            if failures > 3:
                _status(output_queue, stop_event, "failed", "source_reconnect_exhausted")
                return
            _status(output_queue, stop_event, "reconnecting", error_code)
        finally:
            if reader is not None:
                dropped_total += reader.dropped
                if not reader.close():
                    _status(output_queue, stop_event, "stopped" if stop_event.is_set() else "failed",
                            None if stop_event.is_set() else "source_reader_stuck")
                    return
            elif capture is not None:
                capture.release()
        if not live or stop_event.is_set():
            break
        if stop_event.wait((1, 2, 4)[failures - 1]):
            break
        segment_index += 1
        processor.reset_segment()
        last_live_utc = None
    _status(output_queue, stop_event, "stopped")
