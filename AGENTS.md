# Synapsis development instructions

## Scope and ownership

- Work from this repository; read nearby code and callers before editing. State goal, context, constraints, and observable verification before non-trivial work.
- Keep changes focused. Do not commit, push, or perform destructive Git operations without user authorization. Keep this file and `CLAUDE.md` identical.
- Preserve challenge assets, uploads, pretrained weights, and existing PostgreSQL data. Never expose local environment values, camera credentials, or token-bearing URLs. Generated scratch files belong in ignored directories.
- This is a trusted single-host FastAPI/PostgreSQL application. Use one Uvicorn process, one active producer, and at most eight viewers. Do not add workers/reload, distributed scheduling, or model training without an explicit scope change.

## Code map and durable contracts

- `app/application.py` builds the app and lifecycle; `app/api/` owns HTTP contracts; `app/core/` resolves settings and validates/redacts camera URLs.
- `app/pipeline/` owns spawn-safe capture/inference/counting result packets. Child code must not import the web app or create database sessions.
- `app/services/` owns supervisor/persistence, bounded media probes/uploads, pure history, and forecasting. The parent commits analytics before publishing a frame to viewers.
- `app/logic/` provides pure geometry, confirmation, and counting. `app/db/` and additive Alembic revisions define durable provenance and idempotency.
- `app/templates/` and `app/static/` contain the responsive dashboard. Browser tests use actual assets with controlled API fixtures.
- New runs snapshot sorted areas and effective settings. Area edits and run startup share a manager mutation guard. Old detections/events retain null run identity; do not invent legacy provenance.
- Results queue capacity is two. Persist detections/events, audit coverage, minute summaries, absolute counters, and progress together in bounded batches. Retries must not double events, coverage, or counters; publish only after commit. Database retry exhaustion fails processing.
- Boxes/polygons are normalized `[0,1]`; membership tests box centers; edges/vertices are outside (`EPS=1e-12`). Reject invalid, degenerate, duplicate, and self-intersecting polygons.
- Three consecutive processed observations confirm initial membership or transitions. Initial inside presence is occupancy without entry. Missing observations reset pending streaks; confirmed inside expiry after more than two source seconds increments loss telemetry without exit.
- Tracker identity is run/segment scoped. Reconnect resets tracker/presence and preserves cumulative totals. Loss, EOF, shutdown, and viewer detach never synthesize exits.
- Current occupancy requires fresh running observations; expose unavailable current occupancy and a last observation for stale/terminal states. Entry minus exit flow is separate and may be negative.
- File history uses elapsed media seconds; live history uses server-observed UTC. Keep coverage non-overlapping, including backward UTC adjustments. Minute summaries drive history; default to 120 recent minutes and cap explicit output at 10,000 buckets. Never pool runs or areas.
- Forecast requires twenty complete contiguous observed minute buckets; last-value versus EWMA alpha 0.3 uses chronological rolling five-step MAE after ten training minutes, independent in/out selection, naive ties, and no calibrated intervals. Live forecasts also require fresh running observations.
- Stop signals capture immediately. Its independent five-second guardian must remain responsive during database stalls. Preview decoding must not hold the global mutation lock.

## Setup and verification

Use a clean Python 3.10 environment and pinned requirements. Install the selected Torch 2.5.1 / torchvision 0.20.1 wheel first; then requirements. The PostgreSQL URL driver is `postgresql+psycopg2`. Upload/model paths resolve from the repository; weights remain local and missing weights must not trigger implicit downloads.

```sh
docker compose up -d db
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m compileall -q app alembic scripts
python -m ruff check --select E9,F63,F7,F82 app alembic scripts tests
python -m pytest -q
npm ci
npx playwright install chromium
npm run test:browser
```

- PostgreSQL integration tests require `TEST_DATABASE_URL` pointing to a disposable database. Fixtures migrate and truncate its tables. Never substitute SQLite or the normal application database.
- For schema changes, preserve applied migrations; inspect SQL and verify upgrade/downgrade, constraints, and retry/idempotency on disposable PostgreSQL.
- Cover lifecycle, gaps/expiry, overlaps, replay/run isolation, stale occupancy, coverage nulls/zeros, and chronological forecast gating. Test cancellation and failure cleanup for uploads/probes.
- Browser checks cover desktop/tablet/phone, polygon editing, run selection/lifecycle, quiet footage, accessibility, and unhandled errors. Browser fixtures do not prove real inference.
- Public evaluation uses `benchmarks/mot15.json` and `benchmarks/README.md`: verify provenance/frame alignment, derive reference crossings from human labels, and select on development data before held-out evaluation. Partial smoke runs are not full accuracy/performance evidence.
- Review the final diff and report actual passed/skipped checks, infrastructure limits, and measured results. Refer to `README.md` and `docs/architecture.md` for public behavior and ownership.
