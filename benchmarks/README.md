# Public sequence evaluation

`mot15.json` freezes two official MOT15 public training sequences and one polygon
before running any model. PETS09-S2L1 is the development sequence. TUD-Stadtmitte
is held out from profile selection. These are derived polygon-counting scores,
not official MOT leaderboard results. No model training is performed.

Run from the repository root in the configured environment:

```powershell
python scripts/fetch_benchmark.py
python scripts/evaluate.py --sequence PETS09-S2L1 --models yolov8n.pt yolov8s.pt yolov8m.pt --confidences 0.25 0.1 --device 0 --output benchmarks/development.json
# After selecting a profile using development results only:
python scripts/evaluate.py --sequence TUD-Stadtmitte --models yolov8n.pt yolov8s.pt --confidences 0.25 --device 0 --output benchmarks/held-out.json
```

The held-out command compares the default with the small candidate selected on
development data. Use `--device cpu` for CPU
evaluation. Missing local weights fail without triggering a download.
`--max-frames` is only a smoke check and marks the report as partial.

Downloads are bounded to 200 MB and stored under ignored `data/benchmarks/`.
The downloader requests 1 MB HTTP byte ranges, validates every Content-Range,
retries transient failures up to three times, and keeps a hash-verified partial
checkpoint so a failed invocation can resume without repeating finished chunks.
`download-lock.json` records archive, annotation, and video SHA-256 hashes.
Evaluation verifies those hashes and decodes every frame to check the published
count, dimensions, FPS, and annotation range before inference. Official preview
videos may be reencoded; matching metadata cannot prove original pixel identity.
If alignment checks fail, do not score the preview. Obtain the original MOT15
archive separately and preserve corresponding provenance.

Reference crossings come from human trajectory boxes, using the application's
center/boundary, three-observation confirmation, startup-presence-only, and
two-second loss semantics. They do not come from detector output.
MOT [format coordinates](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt)
are one-based; left/top are shifted by one pixel before normalization to align
with OpenCV/YOLO, while width/height are unchanged. Frame indices stay one-based.
Directional events are matched one-to-one within one second, separately for each area.
Reports include precision, recall, F1, signed/absolute directional count error,
frame occupancy MAE, FPS, median/p95 detector and offline pipeline latency, first-frame latency,
and CUDA peak allocated/reserved memory when available. An absent event class
has unavailable precision/recall as appropriate rather than an invented score.

Runtime and evaluation share `FrameProcessor` and `configs/bytetrack.yaml`.
Offline pipeline latency includes decode, inference/tracking/counting, drawing,
and JPEG encoding. Database writes and viewer transport are outside this measurement.
Detector-only timing uses Ultralytics
inference telemetry. FPS includes decoding and cold first inference. Reports
retain hardware, dependency versions, dataset/weight hashes, and code hashes.
Report hardware and exact dependency
versions with every comparison. Select a profile on development results, then
evaluate that selection once on the held-out sequence.

## Measured comparison and profile decision

The complete reports are [development.json](development.json) and
[held-out.json](held-out.json). Both previews passed exact decoded frame count,
FPS, dimension, annotation-range, and recorded SHA-256 checks. Human annotation
overlays also matched visible people at the first, middle, and last frame of
both sequences. PETS has 795 frames at 7 FPS; TUD has 179 frames at 25 FPS.

Measurements used Windows, an NVIDIA RTX 3050 Ti Laptop GPU with 4,095.5 MB,
PyTorch 2.5.1+cu124, CUDA 12.4, OpenCV 4.10.0, and image size 640. All profiles
use the same frozen polygon, tracker, IoU 0.45, and counting rules. Memory below
is the peak PyTorch CUDA **reserved** allocator memory, excluding CUDA context,
driver, and other applications. FPS includes decoding and cold first inference;
latency includes decode, inference/tracking/counting, drawing, and JPEG encoding.

PETS development truth contains 24 entries and 23 exits:

| Model / confidence | Entry / exit F1 | Entry / exit count error | Occupancy MAE | FPS | p95 ms | CUDA reserved MB |
| --- | --- | --- | --- | --- | --- | --- |
| nano / 0.25 | 0.980 / 0.958 | +1 / +2 | 0.151 | 41.0 | 23.2 | 90 |
| nano / 0.10 | 0.980 / 0.958 | +1 / +2 | 0.200 | 49.3 | 22.5 | 102 |
| small / 0.25 | 1.000 / 0.958 | +0 / +2 | 0.176 | 45.4 | 25.3 | 156 |
| small / 0.10 | 1.000 / 0.958 | +0 / +2 | 0.186 | 33.5 | 44.4 | 210 |
| medium / 0.25 | 1.000 / 0.958 | +0 / +2 | 0.184 | 29.2 | 37.0 | 286 |
| medium / 0.10 | 1.000 / 0.958 | +0 / +2 | 0.182 | 28.4 | 41.6 | 396 |

Small / 0.25 was selected on development results for the held-out comparison:
it matched the larger profiles' directional traffic accuracy with the lowest
occupancy error and memory among those profiles. Nano / 0.10 did not improve
traffic counts and worsened occupancy, so there was no development basis for
promoting the lower confidence or testing it to select a held-out winner.

TUD held-out truth contains five entries and two exits:

| Model / confidence | Entry / exit F1 | Entry / exit count error | Occupancy MAE | FPS | p95 ms | CUDA reserved MB |
| --- | --- | --- | --- | --- | --- | --- |
| nano / 0.25 | 0.750 / 1.000 | -2 / +0 | 0.832 | 17.6 | 32.2 | 90 |
| small / 0.25 | 0.750 / 1.000 | -2 / +0 | 0.413 | 45.6 | 24.8 | 140 |

Both models missed **two held-out entries**. Exact exits therefore do not mean
perfect overall accuracy. Small reduced held-out occupancy MAE and slightly
worsened development occupancy. Single-pass
FPS includes cold-start effects and run order; it does not establish that a
larger model is generally faster.

The [profile decision](profile-decision.json) retains **nano / 0.25** as the
default. Small / 0.25 is an optional profile when occupancy matters and measured
hardware throughput is acceptable. These two short public sequences support a
limited comparison, not general camera accuracy, identity/tracking scores,
confidence intervals, or a production reliability claim. No model retraining
or default-profile tuning on the held-out sequence occurred.
