import queue
import threading
import time

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fake_jpeg_encoder(monkeypatch):
    # Lifecycle tests inject the codec, as they already inject capture and inference.
    monkeypatch.setattr("app.pipeline.processing.draw_frame", lambda *args: b"test-jpeg")


def worker():
    from app.pipeline.worker import worker_main
    return worker_main


class Capture:
    def __init__(self, frames, fps=10):
        self.frames = list(frames)
        self.fps = fps
        self.index = 0
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        self.index += 1
        return True, self.frames.pop(0)

    def media_time_ms(self):
        return (self.index - 1) * 1000 / self.fps

    def release(self):
        self.released = True


class Model:
    def track(self, **kwargs):
        return []


def spec(**settings):
    return {"run_id": "test", "kind": "file", "uri": "unused", "areas": [],
            "settings": {"model_path": "unused.pt", "confirmation": 3,
                         "missing_grace_seconds": 2, **settings}}


def packets(output):
    result = []
    while not output.empty():
        result.append(output.get_nowait())
    return result


def test_quiet_file_emits_all_frames_and_completes_with_media_coverage():
    capture = Capture([np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)])
    output = queue.Queue()
    worker()(spec(), output, threading.Event(), capture_factory=lambda *args: capture,
             model_factory=lambda *args: Model())
    result = packets(output)
    frames = [item for item in result if item["kind"] == "frame"]
    assert len(frames) == 3
    assert [item["frame_index"] for item in frames] == [1, 2, 3]
    assert [item["media_time_ms"] for item in frames] == [0, 100, 200]
    assert sum(item["duration_seconds"] for item in frames) == pytest.approx(0.2)
    assert all(left["coverage_start_seconds"] + left["duration_seconds"] <= right["coverage_start_seconds"] + 1e-9
               for left, right in zip(frames, frames[1:]))
    assert all(item["detections"] == [] and item["jpeg"] for item in frames)
    assert result[-1] == {"kind": "status", "status": "completed", "error_code": None}
    assert capture.released


def test_stop_interrupts_file_backpressure_and_releases_capture():
    capture = Capture([np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(20)])
    output = queue.Queue(maxsize=2)
    stop = threading.Event()
    thread = threading.Thread(target=worker(), args=(spec(), output, stop),
                              kwargs={"capture_factory": lambda *args: capture,
                                      "model_factory": lambda *args: Model()})
    thread.start()
    deadline = time.monotonic() + 1
    while not output.full() and time.monotonic() < deadline:
        stop.wait(0.01)
    assert output.full() and thread.is_alive()
    stop.set()
    thread.join(2)
    assert not thread.is_alive()
    assert capture.released


def test_completion_status_survives_slow_healthy_consumer_backpressure():
    capture = Capture([np.zeros((20, 30, 3), dtype=np.uint8)])
    output = queue.Queue(maxsize=2)
    stop = threading.Event()
    thread = threading.Thread(target=worker(), args=(spec(), output, stop),
                              kwargs={"capture_factory": lambda *args: capture,
                                      "model_factory": lambda *args: Model()})
    thread.start()
    time.sleep(0.65)
    assert thread.is_alive()
    assert output.get(timeout=1)["status"] == "running"
    assert output.get(timeout=1)["kind"] == "frame"
    assert output.get(timeout=1)["status"] == "completed"
    thread.join(1)
    assert not thread.is_alive()


def test_encode_failure_fails_run_and_releases_capture(monkeypatch):
    capture = Capture([np.zeros((20, 30, 3), dtype=np.uint8)])
    output = queue.Queue()
    def bad_encoder(*args):
        raise RuntimeError("jpeg_encode_failed")
    monkeypatch.setattr("app.pipeline.processing.draw_frame", bad_encoder)
    worker()(spec(), output, threading.Event(), capture_factory=lambda *args: capture,
             model_factory=lambda *args: Model())
    assert packets(output)[-1]["error_code"] == "jpeg_encode_failed"
    assert capture.released


def test_bad_model_failure_is_sanitized_without_uri_credentials():
    output = queue.Queue()
    def bad_model(*args):
        raise RuntimeError("rtsp://secret:password@host/private")
    worker()(spec(), output, threading.Event(), model_factory=bad_model)
    assert packets(output) == [{"kind": "status", "status": "failed",
                                "error_code": "model_load_failed"}]


def test_cancelled_worker_never_loads_model():
    output = queue.Queue()
    stop = threading.Event()
    stop.set()
    def forbidden(*args):
        raise AssertionError("model must not load")
    worker()(spec(), output, stop, model_factory=forbidden)
    assert packets(output) == [{"kind": "status", "status": "stopped", "error_code": None}]


def test_stop_interrupts_live_reconnect_backoff_without_opening_again():
    output = queue.Queue()
    stop = threading.Event()
    attempts = []
    def bad_capture(*args):
        attempts.append(1)
        raise RuntimeError("private URI must not escape")
    configuration = {**spec(), "kind": "live"}
    thread = threading.Thread(target=worker(), args=(configuration, output, stop),
                              kwargs={"capture_factory": bad_capture,
                                      "model_factory": lambda *args: Model()})
    thread.start()
    assert output.get(timeout=1)["status"] == "reconnecting"
    stop.set()
    thread.join(1)
    assert not thread.is_alive() and len(attempts) == 1
    assert output.get(timeout=1)["status"] == "stopped"


def test_normalization_uses_original_dimensions_and_keeps_untracked_detections():
    from app.pipeline.processing import normalize_results
    class Boxes:
        xyxy = np.array([[-1, 0, 50, 100], [10, 20, 30, 40]])
        conf = np.array([0.9, 0.8])
        id = None
    class Result:
        boxes = Boxes()
    result = normalize_results([Result()], width=50, height=100, areas=[])
    assert result[0]["x1"] == 0 and result[0]["x2"] == 1
    assert result[0]["y2"] == 1 and result[0]["tracker_id"] is None
    assert result[1]["detection_index"] == 1


def test_latest_capture_keeps_one_slot_and_accounts_for_skipped_frames():
    from app.pipeline.capture import LatestCapture
    capture = Capture([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(10)])
    reader = LatestCapture(capture, threading.Event())
    reader.thread.join(1)
    assert reader.next(1)[0].shape == (10, 10, 3)
    assert reader.dropped == 9
    with pytest.raises(RuntimeError, match="source_read_failed"):
        reader.next(1)
    reader.close()
    assert capture.released


def test_container_time_fallback_preserves_monotonic_media_timeline():
    capture = Capture([np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)])
    capture.media_time_ms = lambda: 0
    output = queue.Queue()
    worker()(spec(), output, threading.Event(), capture_factory=lambda *args: capture,
             model_factory=lambda *args: Model())
    frames = [item for item in packets(output) if item["kind"] == "frame"]
    assert [item["media_time_ms"] for item in frames] == [0, 100, 200]
    assert [item["media_time_source"] for item in frames] == ["container", "fps_fallback", "fps_fallback"]


def test_broken_container_timestamp_after_valid_jump_cannot_regress_media_time():
    capture = Capture([np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)], fps=10)
    timestamps = iter([0, 500, 0])
    capture.media_time_ms = lambda: next(timestamps)
    output = queue.Queue()
    worker()(spec(), output, threading.Event(), capture_factory=lambda *args: capture,
             model_factory=lambda *args: Model())
    result = packets(output)
    frames = [item for item in result if item["kind"] == "frame"]
    assert [item["media_time_ms"] for item in frames] == [0, 500, 600]
    assert frames[-1]["media_time_source"] == "fps_fallback"
    assert result[-1]["status"] == "completed"


def test_segment_reset_does_not_emit_startup_enter_or_lose_cumulative_counts():
    from app.pipeline.processing import FrameProcessor
    area = {"id": 1, "name": "zone", "polygon": [
        {"x": 0.25, "y": 0.1}, {"x": 0.75, "y": 0.1},
        {"x": 0.75, "y": 0.9}, {"x": 0.25, "y": 0.9}]}
    class MovingModel:
        def track(self, source, **kwargs):
            center = int(source[0, 0, 0])
            class Boxes:
                xyxy = np.array([[center - 2, 40, center + 2, 60]])
                id = np.array([1])
                conf = np.array([0.9])
            class Result:
                boxes = Boxes()
            return [Result()]
    processor = FrameProcessor(MovingModel(), [area], {})
    def observe(center, times):
        frame = np.full((100, 100, 3), center, dtype=np.uint8)
        return [processor.process(frame, moment) for moment in times][-1]
    observe(95, [0, 0.1, 0.2])
    assert observe(50, [0.3, 0.4, 0.5])["stats"][0]["total_in"] == 1
    processor.reset_segment()
    initial = observe(50, [2, 2.1, 2.2])
    assert initial["events"] == [] and initial["stats"][0]["total_in"] == 1
    assert observe(95, [2.3, 2.4, 2.5])["stats"][0]["total_out"] == 1
