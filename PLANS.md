# Synapsis holistic improvement implementation

Approved plan: reliable single-host processing for uploads and RTSP/HTTP cameras,
responsive dashboard, run-scoped counting/history, measured pretrained profiles,
and a five-minute entry/exit forecast. Work is authorized on main, including eight
logical commits and a normal push to origin/main. Never force-push.

## Commit checkpoints

- [ ] 1. Runtime/configuration, dependencies, environment examples, test foundation.
- [ ] 2. Validated sources, polygon editing, durable run schema, additive migration.
- [ ] 3. Shared spawned processing, batched idempotent persistence, lifecycle APIs.
- [ ] 4. Initial presence, missing-track expiry, run-scoped history and freshness.
- [ ] 5. Forecast gate, rolling backtesting, entry/exit baseline forecasts.
- [ ] 6. Responsive accessible dashboard and browser verification.
- [ ] 7. Public labeled benchmark, model profiles, reproducible measured report.
- [ ] 8. Docker/CI, README diagrams/runbook, synchronized instruction files, review.

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
