import threading

import pytest

from app.services.supervisor import Runtime, RunManager


def test_two_viewers_share_only_committed_latest_packet():
    manager = RunManager.__new__(RunManager)
    manager.lock = threading.RLock()
    manager.config = type("Config", (), {"max_viewers": 8})()
    runtime = Runtime("run", 1, None, None, None)
    manager.runtimes = {"run": runtime}
    first = manager.subscribe("run")
    second = manager.subscribe("run")
    assert runtime.viewers == 2
    manager.publish(runtime, b"frame")
    assert b"frame" in next(first)
    assert b"frame" in next(second)
    first.close()
    assert runtime.viewers == 1 and not runtime.terminal
    second.close()
    assert runtime.viewers == 0


def test_viewer_limit_rejected_without_starting_any_worker():
    manager = RunManager.__new__(RunManager)
    manager.lock = threading.RLock()
    manager.config = type("Config", (), {"max_viewers": 1})()
    runtime = Runtime("run", 1, None, None, None)
    manager.runtimes = {"run": runtime}
    stream = manager.subscribe("run")
    with pytest.raises(ValueError, match="viewer"):
        manager.subscribe("run")
    stream.close()


def test_preview_reservation_does_not_block_supervisor_control():
    manager = RunManager.__new__(RunManager)
    manager.lock = threading.RLock()
    manager.previewing = set()
    manager.begin_preview(1)
    acquired = []
    def control():
        with manager.mutation_guard():
            acquired.append(True)
    thread = threading.Thread(target=control)
    thread.start()
    thread.join(timeout=0.2)
    assert acquired == [True]
    with pytest.raises(ValueError, match="preview"):
        manager.begin_preview(1)
    manager.end_preview(1)
    manager.begin_preview(1)
    manager.end_preview(1)


def test_stop_signal_precedes_a_blocked_database_write():
    from contextlib import contextmanager
    entered, release = threading.Event(), threading.Event()
    class Factory:
        @contextmanager
        def begin(self):
            entered.set()
            release.wait(3)
            yield type("DB", (), {"get": lambda *args: type("Run", (), {"status": "running"})(), "execute": lambda *args: None})()
    manager = RunManager.__new__(RunManager)
    manager.lock = threading.RLock()
    manager.factory = Factory()
    manager._stop_guardian = lambda runtime: None
    runtime = Runtime("run", 1, None, None, threading.Event())
    manager.runtimes = {"run": runtime}
    caller = threading.Thread(target=manager.stop_run, args=("run",))
    caller.start()
    try:
        assert entered.wait(1)
        assert runtime.stop.is_set()
        caller.join(timeout=0.2)
        assert not caller.is_alive()
    finally:
        release.set()
        caller.join(timeout=1)
