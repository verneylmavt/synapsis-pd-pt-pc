from datetime import datetime, timezone
import queue
import threading

import numpy as np


def test_backward_utc_adjustment_never_repeats_previously_observed_coverage(monkeypatch):
    import app.pipeline.worker as worker
    stop = threading.Event()
    class Capture:
        fps = 5
    class Model:
        def track(self, **kwargs):
            return []
    class Reader:
        dropped = 0
        def __init__(self, *args):
            self.observations = iter(zip([1000.2, 1000.4, 1000.35, 1000.5], [1, 2, 3, 4]))
        def next(self, *args):
            try:
                epoch, monotonic_time = next(self.observations)
            except StopIteration:
                stop.set()
                return None
            return (np.zeros((10, 10, 3), dtype=np.uint8),
                    datetime.fromtimestamp(epoch, timezone.utc), monotonic_time)
        def close(self):
            return True
    monkeypatch.setattr(worker, "LatestCapture", Reader)
    monkeypatch.setattr(worker.time, "time", lambda: 1000.0)
    monkeypatch.setattr("app.pipeline.processing.draw_frame", lambda *args: b"fake-jpeg")
    output = queue.Queue()
    worker.worker_main({"run_id": "clock-test", "kind": "live", "uri": "fake", "areas": [],
                        "settings": {"model_path": "unused", "live_fps": 5}},
                       output, stop, capture_factory=lambda *args: Capture(),
                       model_factory=lambda *args: Model())
    packets = []
    while not output.empty():
        packets.append(output.get_nowait())
    frames = [packet for packet in packets if packet["kind"] == "frame"]
    assert len(frames) == 4
    previous_end = float("-inf")
    for frame in frames:
        assert frame["duration_seconds"] >= 0
        assert frame["coverage_start_seconds"] >= previous_end - 1e-9
        previous_end = frame["coverage_start_seconds"] + frame["duration_seconds"]
    assert frames[2]["duration_seconds"] == 0
    assert previous_end == 1000.5
    assert packets[-1]["status"] == "stopped"
