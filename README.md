# Synapsis people detection, tracking, and counting

FastAPI, pretrained YOLOv8, ByteTrack, and PostgreSQL provide polygon counting for uploaded videos and trusted RTSP/HTTP cameras. One producer processes an explicitly started run; up to eight viewers share its committed MJPEG frames. The responsive dashboard manages sources, edits polygons, controls runs, and displays occupancy, traffic history, recent events, and five-minute baseline forecasts.

The [five-page challenge brief](assets/AI%20Engineer%20-%20Challenge.pdf) covers database design, data collection, detection/tracking, polygon counting, forecasting, API integration, and deployment. Each component has an implementation. Public evaluation and deployment verification need separate evidence; implementing them does not establish an accuracy or throughput result.

| Challenge capability | Implementation |
| --- | --- |
| Video and camera collection | Bounded decoded uploads; direct RTSP/HTTP camera registration and reconnects. |
| Detection and tracking | Local YOLOv8 n/s/m weights with repository-owned ByteTrack settings. |
| Polygon counting | Confirmed observed transitions, separate occupancy, and loss expiry. |
| Database and API | Run snapshots, idempotent analytics, minute history, and lifecycle endpoints. |
| Forecast and visualization | Backtested five-minute baselines, accessible charts/tables, and responsive dashboard. |
| Deployment and evaluation | CPU Compose application, GPU host instructions, CI, and labeled public-sequence reports. |

## Start with Docker (CPU)

Install Docker with Compose, then place `yolov8n.pt` in the repository root. Weights are excluded from the image and mounted read-only under `/models`. Starting a run does not automatically download missing weights.

```sh
docker compose up --build -d
docker compose ps
docker compose logs api
```

Open [the dashboard](http://127.0.0.1:8000/dashboard) or [API documentation](http://127.0.0.1:8000/docs). The API applies additive migrations before startup and waits for PostgreSQL health. Ports bind to loopback. The image uses CPU PyTorch; GPU inference uses the host setup below.

PostgreSQL retains the existing `pgdata` named volume; uploads use `appdata`. `docker compose down` preserves both. Deleting volumes destroys their contents. Optional Adminer runs with `docker compose --profile debug up -d adminer` at port 8080.

Default database credentials are for local development. Set `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` before initializing a new volume. Changing them does not modify accounts in an existing volume. `DOCKER_DATABASE_URL` overrides the container API connection; percent-encode special characters in credentials. The host's `DATABASE_URL` is separate from the container's `db` hostname.

## Host setup (Python 3.10)

Use a clean virtual environment without inherited Conda packages or `--system-site-packages`. On Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# NVIDIA driver supporting CUDA 12.4:
python -m pip install --no-deps torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

On Linux, use `python3.10 -m venv .venv` and `source .venv/bin/activate`. For a CPU host, replace the PyTorch command with:

```sh
python -m pip install --no-deps torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

These wheel pairings follow the [official PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251). Install PyTorch before the pinned project requirements. Verify the interpreter and CUDA availability:

```sh
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"
```

Place `yolov8n.pt` at the project root or set `MODEL_DIR` to its directory. Optional selectable weights are `yolov8s.pt` and `yolov8m.pt`. Keep weights and uploads out of Git. Set the PostgreSQL connection in the untracked `.env`; the installed driver is `postgresql+psycopg2`.

```sh
docker compose up -d db
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run exactly one API process. Do not add `--workers`, a reload supervisor, or a second instance against the same database. A session advisory lock enforces manager ownership; startup marks abandoned active runs interrupted. Upload/model paths resolve relative to the repository, independently of shell location.

[.env.example](.env.example) documents `DATABASE_URL`, `UPLOAD_DIR`, `MODEL_DIR`, `MODEL_DEVICE`, `MAX_UPLOAD_BYTES` (500 MiB), `MAX_ACTIVE_RUNS` (fixed at one), and `MAX_VIEWERS` (eight).

## Dashboard workflow

1. Upload a video or register a direct `rtsp://`, `http://`, or `https://` camera URL. LAN and loopback cameras are permitted. Registration validates URL syntax; preview or processing verifies media decoding. Arbitrary web pages do not become readable video sources by registration.
2. Load a preview and create an area. Click or drag vertices, or use the coordinate editor. Polygons use normalized coordinates in `[0,1]`. Duplicate vertices, crossings, zero area, and invalid coordinates are rejected.
3. Choose model/settings and start a run explicitly. Selections survive reload. Runs snapshot areas/settings; stop before editing areas.
4. Select a run and area for traffic and events. Detaching viewers and quiet footage do not stop processing. Use Stop; file EOF completes its run.

Source responses and errors redact camera credentials, paths, and token queries. This is a trusted single-host application bound to loopback, without public authentication or distributed scheduling.

## API contract

| Endpoint | Behavior |
| --- | --- |
| `GET /healthz` | Application health; startup requires database manager ownership. Does not establish model accuracy or camera readability. |
| `GET /api/video-sources` | Sources with redacted URIs. |
| `POST /api/video-sources` | Register `{name, uri}`; no camera connection during registration. |
| `POST /api/upload` | Multipart `file`; bounded UUID upload and decoded metadata. `/api/upload-video` remains an alias. |
| `GET /api/preview/{source_id}` | Shared active JPEG or bounded isolated preview. `/api/video/first-frame/{source_id}` remains an alias. |
| `GET /api/areas?video_source_id=1` | Source areas, including disabled areas. |
| `POST /api/areas` | Create `{video_source_id, name, polygon}`. |
| `PUT /api/areas/{area_id}` | Edit `name`, `polygon`, or `active`; active runs return 409. |
| `POST /api/video-sources/{source_id}/runs` | Start with settings; 202. Matching active repeats reuse the run; conflicting settings/capacity return 409. |
| `GET /api/video-sources/{source_id}/runs` | Recent runs. |
| `GET /api/runs/{run_id}` | Lifecycle, frame/segment progress, FPS, freshness timestamps, sanitized error code. |
| `POST /api/runs/{run_id}/stop` | Idempotent stop request. |
| `GET /api/runs/{run_id}/stream` | Subscribe to committed MJPEG frames. |
| `GET /stream/{source_id}?run_id=...` | Compatibility subscription endpoint. |
| `GET /api/stats/live?run_id=...&area_id=...` | Occupancy, totals, net flow, loss telemetry, freshness. |
| `GET /api/stats?run_id=...&area_id=...&granularity=minute` | Covered minute/hour/day history. |
| `GET /api/runs/{run_id}/events?area_id=...` | Recent persisted events. |
| `GET /api/forecast?run_id=...&area_id=...` | Five-minute baseline or insufficient-data response. |

**Compatibility change:** the old `GET /stream/{source_id}` started a tracker per viewer. It now subscribes only. Start with POST before opening a stream. Each replay gets a new run; pass `run_id` to avoid attaching to an unintended replay.

After creating source 1 and an area:

```sh
curl -X POST http://127.0.0.1:8000/api/video-sources/1/runs -H "Content-Type: application/json" -d '{"model":"yolov8n.pt","conf":0.25,"iou":0.45,"imgsz":640,"device":"auto"}'
```

Defaults remain YOLOv8n, confidence 0.25, IoU 0.45, image size 640, and automatic device selection. A measured development/evaluation report is required before presenting a changed profile as better.

## Counting, timelines, and forecast

Counting tests the normalized box center. Edges and vertices are outside (`EPS=1e-12`). Areas count independently, including overlaps. Three consecutive processed observations confirm initial membership or a later transition. Initial confirmed inside presence increases occupancy without an entry; a later exit may make `total_in - total_out` negative.

A missing processed frame resets pending confirmation streaks. Confirmed presence remains latched for at most two seconds of source progression. Expiry removes occupancy and increments lost-track telemetry without inventing an exit. Untracked detections remain stored but emit no events. Reconnection starts a fresh tracker segment/presence while preserving run totals. Tracker identity is scoped to run and segment.

`currently_inside` is fresh observed presence in the selected running run. Stopped, completed, reconnecting, or stale runs expose unavailable current occupancy and a separate last observation. `net_entries` is entry minus exit flow; it is not occupancy.

History/forecast never pool clips, replays, runs, or areas. Files use elapsed media seconds; live sources use server-observed UTC, not camera sensor exposure time. File history `start`/`end` take numeric seconds; live history takes timezone-aware ISO dates. Legacy records retain null run identity and are exposed only through explicit `scope=legacy` statistics; they have no reliable coverage or current occupancy.

Persisted frames carry explicit coverage. Transactional minute summaries make history queries independent of detection volume. Fully covered complete buckets with no traffic contain zero. Missing coverage and partial buckets have null traffic. Occupancy is time weighted; camera outage intervals remain unavailable. The default view covers the latest 120 minutes; explicit ranges can return up to 10,000 minute/hour/day buckets.

Forecast requires twenty complete, observed, contiguous minute buckets ending at the latest complete bucket. Missing data breaks this window. Last-value and EWMA (`alpha=0.3`) are compared independently for entries/exits using chronological rolling five-step MAE after ten training minutes. Ties use last-value. Predictions are nonnegative expected counts with method/error metadata; no calibrated confidence intervals are claimed. File forecasts describe a **hypothetical continuation**. Live forecasts require fresh running observations and assume the recent traffic rate continues for five minutes.

## Architecture and public evaluation

[Architecture and ownership](docs/architecture.md) describes the spawned worker, parent-owned database writes, two-packet queue, committed-frame fanout, and durable tables. Analytics, absolute counters, coverage, and progress commit together before frame publication. Idempotency keys protect batch retries.

[Benchmark instructions](benchmarks/README.md) freeze public MOT15 development/held-out sequences, hashes, polygons, label filtering, and alignment checks. Human boxes and track IDs generate reference crossings; detector predictions do not supply ground truth. No model training occurs.

Reports separate event precision/recall/F1, directional count error, occupancy MAE, latency, FPS, and GPU memory. Preview videos may be reencoded; alignment must pass before scoring. These are derived polygon-counting results on selected public training sequences, not official hidden-test leaderboard scores. Short smoke checks do not establish full-sequence accuracy or throughput. Consult recorded benchmark reports for measured outcomes when available.

## Verification

Run from the repository root in the isolated environment:

```sh
python -m compileall -q app alembic scripts
python -m ruff check --select E9,F63,F7,F82 app alembic scripts tests
python -m pytest -q
npm ci
npx playwright install chromium
npm run test:browser
```

Pure tests need neither PostgreSQL nor weights. Real media tests generate a local clip. PostgreSQL integration tests skip without `TEST_DATABASE_URL`; its fixture upgrades migrations and truncates analytics/source tables. Use a **disposable test database**, never the normal application database. After creating `synapsis_test` with development credentials:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg2://synapsis:synapsis@localhost:5432/synapsis_test"
python -m pytest -q
```

Browser tests use actual dashboard assets with controlled API fixtures at desktop, tablet, and phone sizes. They complement PostgreSQL/lifecycle checks and a real model/video smoke run. CI uses CPU dependencies, disposable PostgreSQL, and Chromium; it downloads no datasets or weights.

Preserve challenge assets, local uploads, and weights. Report checks that ran, skips, measured model outcomes, and infrastructure limitations separately.

[Recorded verification](docs/verification.md) summarizes this implementation's checks and remaining evaluation limits.
