from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

class OpenCVCapture:
    def __init__(self, uri: str, settings: dict):
        import cv2
        self.cv2 = cv2
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(settings.get("open_timeout_seconds", 5) * 1000),
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(settings.get("read_timeout_seconds", 5) * 1000)]
        self.capture = cv2.VideoCapture(uri, cv2.CAP_FFMPEG, params)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("source_open_failed")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))

    def read(self):
        return self.capture.read()

    def media_time_ms(self):
        return float(self.capture.get(self.cv2.CAP_PROP_POS_MSEC))

    def release(self):
        self.capture.release()


class LatestCapture:
    """A live reader has one replaceable slot; inference consumes the newest frame."""

    def __init__(self, capture, stop_event):
        self.capture = capture
        self.stop_event = stop_event
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.consumed = 0
        self.dropped = 0
        self.failed = False
        self.closed = False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.closed and not self.stop_event.is_set():
                ok, frame = self.capture.read()
                with self.condition:
                    if not ok or frame is None:
                        self.failed = True
                        self.condition.notify_all()
                        return
                    self.sequence += 1
                    self.latest = (self.sequence, frame, datetime.now(timezone.utc), time.monotonic())
                    self.condition.notify_all()
        except Exception:
            with self.condition:
                self.failed = True
                self.condition.notify_all()

    def next(self, timeout_seconds: float):
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            while not self.stop_event.is_set():
                if self.latest is not None and self.latest[0] > self.consumed:
                    sequence, frame, observed_at, monotonic_at = self.latest
                    self.dropped += max(0, sequence - self.consumed - 1)
                    self.consumed = sequence
                    return frame, observed_at, monotonic_at
                if self.failed:
                    raise RuntimeError("source_read_failed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("source_no_frames")
                self.condition.wait(min(remaining, 0.1))
        return None

    def close(self):
        self.closed = True
        with self.condition:
            self.condition.notify_all()
        self.thread.join(0.5)
        # VideoCapture is not safe to release concurrently with a native read.
        # A stuck reader is contained in this child and terminated by its parent.
        if self.thread.is_alive():
            return False
        self.capture.release()
        return True
