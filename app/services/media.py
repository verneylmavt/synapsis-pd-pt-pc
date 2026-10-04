from __future__ import annotations

import multiprocessing
import os
import queue
from math import isfinite


class MediaProbeError(ValueError):
    """Sanitized decode/probe error; camera credentials never become an error message."""


def _probe_child(uri: str, preview: bool, results) -> None:
    capture = None
    try:
        with open(os.devnull, "w") as quiet:
            os.dup2(quiet.fileno(), 2)
        import cv2
        capture = cv2.VideoCapture(uri)
        if not capture.isOpened():
            raise ValueError("unreadable_video")
        ok, frame = capture.read()
        if not ok or frame is None or not frame.size:
            raise ValueError("unreadable_video")
        if preview:
            ok, image = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("preview_encode_failed")
            results.put({"jpeg": image.tobytes()})
        else:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not isfinite(fps) or fps <= 0 or not isfinite(frames) or frames <= 0:
                raise ValueError("invalid_video_timing")
            results.put({"metadata": {"width": int(frame.shape[1]), "height": int(frame.shape[0]),
                                      "fps": fps, "duration_seconds": frames / fps}})
    except Exception:
        results.put({"error": "unreadable_video"})
    finally:
        if capture is not None:
            capture.release()


def _bounded_probe(uri: str, preview: bool, timeout_seconds: float) -> dict:
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Probe timeout must be positive and finite")
    context = multiprocessing.get_context("spawn")
    results = context.Queue(maxsize=1)
    child = context.Process(target=_probe_child, args=(uri, preview, results), daemon=True)
    try:
        child.start()
        try:
            result = results.get(timeout=timeout_seconds)
        except queue.Empty as error:
            raise MediaProbeError("video_probe_timeout") from error
        if "error" in result:
            raise MediaProbeError(result["error"])
        return result
    finally:
        if child.pid is not None:
            child.join(timeout=0.2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=1)
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
            child.close()
        results.close()
        results.cancel_join_thread()


def probe_video(uri: str, timeout_seconds: float = 10.0) -> dict:
    return _bounded_probe(uri, False, timeout_seconds)["metadata"]


def preview_jpeg(uri: str, timeout_seconds: float = 10.0) -> bytes:
    return _bounded_probe(uri, True, timeout_seconds)["jpeg"]
