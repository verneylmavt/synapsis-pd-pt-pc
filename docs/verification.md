# Implementation verification

## Automated and application checks

- Python: **122 passed**, including PostgreSQL integration tests on a disposable
  database. No database tests were skipped in the recorded full run.
- PostgreSQL: additive upgrade/downgrade, preserved legacy records, ORM/schema
  parity, non-null run keys, idempotent retries/rollback, minute aggregation,
  overlapping backfill coverage, and run/area isolation passed.
- Processing: shared viewers, ownership, quiet footage, EOF, reconnect segments,
  capture deadlines, Stop during blocked persistence, restart recovery,
  file timestamp fallback, and backward UTC coverage regression passed.
- Counting/forecast: initial presence, observed transitions, jitter, missing
  streaks, expiry without exit, overlap handling, missing buckets, zero traffic,
  chronological backtesting, and stale live forecast gates passed.
- Dashboard: **22 browser tests passed** using actual assets with controlled
  API fixtures, including source-switch races, responsive media, keyboard area
  editing, refresh recovery, lifecycle controls, and stale states.
- A real CPU Docker/browser workflow passed upload/decode, polygon save,
  pretrained inference, refresh plus two viewers sharing one producer POST,
  database-backed statistics/history, insufficient forecast, and explicit Stop.
  Screenshots at **390, 768, and 1440 pixels** had no horizontal overflow.
  No unhandled JavaScript errors, console errors, or failed HTTP responses occurred.
- A host CUDA run exercised the real model/video pipeline. The CPU Docker image
  installs isolated pinned dependencies, applies migrations, starts the API,
  and passes health/readiness and dependency checks.
- Python compilation, fatal-error lint checks, and Git whitespace checks passed.
  `AGENTS.md` and `CLAUDE.md` are identical.

## Model evidence and review

Full public-sequence reports cover six PETS09-S2L1 development profiles and the
nano baseline/small candidate on TUD-Stadtmitte. Published frame counts, frame
rates, dimensions, annotation ranges, hashes, and visual annotation alignment
were checked before scoring. Human trajectories define reference crossings.

Reports retain per-frame occupancy, directional events and scores, weight/data
hashes, settings, code hashes, hardware, inference latency, offline pipeline
latency, FPS, and CUDA allocated/reserved memory. The offline pipeline includes
decode through annotated JPEG; database/network transport is outside its timing
boundary. See [benchmark evidence and decision](../benchmarks/README.md).

An independent review covered persistence, migration backfill, history,
forecast freshness, and spawn isolation. Its overlap and backward-clock findings
were fixed and covered by regression tests. The final review found no additional
concrete defects in those areas.

## Limits

Camera failure/reconnect/deadline behavior was tested with injected captures;
no physical RTSP camera was available for this session. CI configuration is
included; its remote run will occur after pushing. Public previews may be
reencoded, and two short sequences with one polygon do not establish general
camera accuracy. Both held-out profiles missed two of five entries. The default
therefore remains nano at confidence 0.25; lower confidence was not promoted.
No retraining or calibrated forecast intervals are claimed.
