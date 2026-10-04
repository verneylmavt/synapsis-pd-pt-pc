import threading

from tests.fake_worker import blocked_worker
from tests.integration.test_runs import manager_factory, source, RUN_SETTINGS, wait_for


def test_stop_reaps_child_even_when_persistence_is_blocked(manager_factory, db_factory, monkeypatch):
    import app.services.supervisor as supervisor
    blocked, release = threading.Event(), threading.Event()
    real_persist = supervisor.persist_batch
    def slow(*args, **kwargs):
        blocked.set()
        release.wait(10)
        return real_persist(*args, **kwargs)
    monkeypatch.setattr(supervisor, "persist_batch", slow)
    manager = manager_factory(blocked_worker)
    run = manager.start(source(db_factory), dict(RUN_SETTINGS))
    runtime = manager.runtimes[run]
    try:
        assert blocked.wait(3)
        manager.stop_run(run)
        wait_for(lambda: not runtime.process.is_alive(), timeout=1.5)
    finally:
        release.set()
