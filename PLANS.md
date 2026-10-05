# Synapsis holistic improvement implementation

Approved plan: reliable single-host processing for uploads and RTSP/HTTP cameras,
responsive dashboard, run-scoped counting/history, measured pretrained profiles,
and a five-minute entry/exit forecast. Work is authorized on main, including eight
logical commits and a normal push to origin/main. Never force-push.

## Commit checkpoints

- [x] 1. Runtime/configuration, dependencies, environment examples, test foundation (`b30ab1a`).
- [x] 2. Validated sources, polygon editing, durable run schema, additive migration (`4fa45ed`).
- [x] 3. Shared spawned processing, batched idempotent persistence, lifecycle APIs (`9d7d5cd`).
- [x] 4. Initial presence, missing-track expiry, run-scoped history and freshness (`a6a5458`).
- [x] 5. Forecast gate, rolling backtesting, entry/exit baseline forecasts (`2f4ae19`).
- [x] 6. Responsive accessible dashboard and browser verification (`388fd37`).
- [x] 7. Public labeled benchmark, model profiles, reproducible measured report (`6c869c3`).
- [x] 8. Docker/CI, README diagrams/runbook, synchronized instruction files, review.

## Fixed contracts and acceptance

One active producer, eight viewers, queue capacity two. Normalized boxes/polygons;
center membership; boundary outside; three observations; initial presence is not
an entry; missing presence expires after two seconds without an exit. Immutable
run snapshots and segment-scoped tracker IDs. File media time and live UTC stay
separate. Forecast needs twenty complete contiguous minute buckets, compares
last-value and EWMA alpha=.3 with rolling five-step MAE, and never pools runs.

Verify each checkpoint before committing. Final acceptance includes PostgreSQL
migrations/retries, lifecycle and counting tests, browser workflows at three
screen sizes, public-label alignment and evaluation, Docker startup, and an
independent review. Local uploads/models and temporary logs remain ignored.

## Recorded verification

122 Python tests passed with disposable PostgreSQL; 22 browser tests passed.
Clean CPU Docker build/start, dependency checks, and the real upload/polygon/run/
refresh/two-viewer/Stop workflow passed with no browser errors or overflow at
390/768/1440 pixels. Full labeled development and held-out benchmark reports
retain nano confidence .25. Independent review findings were fixed and retested.
See `docs/verification.md` for the measurement boundaries and camera/CI limits.
