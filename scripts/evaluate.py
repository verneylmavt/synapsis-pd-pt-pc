"""Run pretrained profiles against human MOT15 trajectories; no model training."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_benchmark import DEFAULT_MANIFEST, sha256


def load_mot_truth(path: Path, width: int, height: int):
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if not row:
                continue
            values = list(map(float, row))
            if len(values) < 7 or values[6] != 1:
                continue
            frame, tracker = int(values[0]), int(values[1])
            left, top, box_width, box_height = values[2:6]
            # MOTChallenge coordinates are 1-based; OpenCV/YOLO pixels start at zero.
            left -= 1
            top -= 1
            if frame < 1 or box_width <= 0 or box_height <= 0:
                continue
            frames[frame].append({"tracker_id": str(tracker), "conf": 1.0,
                                  "x1": max(0, min(1, left / width)),
                                  "y1": max(0, min(1, top / height)),
                                  "x2": max(0, min(1, (left + box_width) / width)),
                                  "y2": max(0, min(1, (top + box_height) / height))})
    return frames


def score_events(truth, predicted, tolerance_seconds=1.0):
    """Ordered one-to-one directional matching; identities remain model-independent."""
    report = {}
    areas = {event["area_id"] for event in truth + predicted}
    for direction in ("enter", "exit"):
        matched = expected = actual = 0
        timing_errors = []
        for area in areas:
            reference = sorted(event["time_seconds"] for event in truth
                               if event["area_id"] == area and event["type"] == direction)
            observations = sorted(event["time_seconds"] for event in predicted
                                  if event["area_id"] == area and event["type"] == direction)
            expected += len(reference)
            actual += len(observations)
            left = right = 0
            while left < len(reference) and right < len(observations):
                delta = observations[right] - reference[left]
                if abs(delta) <= tolerance_seconds:
                    matched += 1
                    timing_errors.append(abs(delta))
                    left += 1
                    right += 1
                elif delta < 0:
                    right += 1
                else:
                    left += 1
        precision = matched / actual if actual else None
        recall = matched / expected if expected else None
        f1 = 2 * matched / (actual + expected) if actual + expected else None
        report[direction] = {"tp": matched, "fp": actual - matched, "fn": expected - matched,
                             "precision": precision, "recall": recall, "f1": f1,
                             "truth_count": expected, "predicted_count": actual,
                             "count_error": actual - expected,
                             "absolute_count_error": abs(actual - expected),
                             "matched_timing_mae_seconds": statistics.mean(timing_errors)
                             if timing_errors else None}
    return report


def validate_alignment(video: Path, truth, sequence):
    import cv2
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("benchmark video cannot be decoded")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if (width, height) != (sequence["width"], sequence["height"]):
            raise ValueError("preview dimensions do not match annotation metadata; use original MOT15 archive")
        if not math.isclose(fps, sequence["fps"], rel_tol=0.001):
            raise ValueError("preview frame rate does not match annotation metadata; use original MOT15 archive")
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if frame.shape[:2] != (height, width):
                raise ValueError("decoded frame dimensions changed")
        if decoded != sequence["frames"] or max(truth, default=0) != decoded:
            raise ValueError("preview frame count does not align with human trajectories; use original MOT15 archive")
        return {"decoded_frames": decoded, "fps": fps, "width": width, "height": height,
                "validation": "exact metadata/frame-count check; preview reencoding cannot prove pixel identity"}
    finally:
        capture.release()


def reference_events(truth, sequence, area):
    from app.logic.counting import CountingEngine
    counter = CountingEngine([area], confirmation=3, missing_grace_seconds=2)
    events = []
    occupancy = []
    for frame in range(1, sequence["frames"] + 1):
        elapsed = (frame - 1) / sequence["fps"]
        result = counter.update(truth.get(frame, []), elapsed)
        events.extend({**event, "frame_index": frame, "time_seconds": elapsed} for event in result["events"])
        occupancy.append(result["stats"][0]["currently_inside"])
    return events, occupancy


def evaluate_profile(video: Path, model_path: Path, sequence, area, truth_events, truth_occupancy,
                     confidence=0.25, device="auto", max_frames=None):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from app.pipeline.processing import FrameProcessor
    if not model_path.is_file():
        raise ValueError(f"local pretrained weights missing: {model_path.name}; evaluation never downloads weights")
    cuda = device != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    processor = FrameProcessor(YOLO(str(model_path)), [area], {
        "device": device, "imgsz": 640, "conf": confidence, "iou": 0.45,
        "confirmation": 3, "missing_grace_seconds": 2})
    capture = cv2.VideoCapture(str(video))
    predicted_events, occupancy, latencies, inference_ms = [], [], [], []
    started = time.perf_counter()
    try:
        frame_index = 0
        while max_frames is None or frame_index < max_frames:
            if cuda:
                torch.cuda.synchronize()
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            elapsed = frame_index / sequence["fps"]
            result = processor.process(frame, elapsed, encode=True)
            if cuda:
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - frame_started)
            if result["inference_ms"] is not None:
                inference_ms.append(result["inference_ms"])
            frame_index += 1
            predicted_events.extend({**event, "frame_index": frame_index, "time_seconds": elapsed}
                                    for event in result["events"])
            occupancy.append(result["stats"][0]["currently_inside"])
    finally:
        capture.release()
    wall_seconds = time.perf_counter() - started
    frames = len(occupancy)
    expected_events = [event for event in truth_events if event["frame_index"] <= frames]
    errors = [abs(actual - expected) for actual, expected in zip(occupancy, truth_occupancy)]
    result = {"model": model_path.name, "model_sha256": sha256(model_path), "confidence": confidence,
              "device": device, "frames": frames, "wall_seconds": wall_seconds,
              "fps": frames / wall_seconds if wall_seconds else 0,
              "latency_ms_p50": float(np.percentile(latencies, 50) * 1000) if latencies else None,
              "latency_ms_p95": float(np.percentile(latencies, 95) * 1000) if latencies else None,
              "inference_ms_p50": float(np.percentile(inference_ms, 50)) if inference_ms else None,
              "inference_ms_p95": float(np.percentile(inference_ms, 95)) if inference_ms else None,
              "cold_start_first_frame_ms": latencies[0] * 1000 if latencies else None,
              "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if cuda else None,
              "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2 if cuda else None,
              "occupancy_mae": statistics.mean(errors) if errors else None,
              "occupancy_max_error": max(errors) if errors else None,
              "events": score_events(expected_events, predicted_events),
              "predicted_events": predicted_events, "predicted_occupancy": occupancy}
    del processor
    if cuda:
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "benchmarks" / "mot15")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.25, 0.1])
    parser.add_argument("--sequence", choices=["PETS09-S2L1", "TUD-Stadtmitte"])
    parser.add_argument("--split", choices=["development", "held_out"], default="development")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "evaluation.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lock = json.loads((args.data / "download-lock.json").read_text(encoding="utf-8"))
    if lock["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("benchmark manifest changed after download; fetch again")
    import torch
    import ultralytics
    import psutil
    report = {"dataset": manifest["dataset"], "manifest_sha256": lock["manifest_sha256"],
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "code_sha256": {name: sha256(ROOT / name) for name in
                  ["scripts/evaluate.py", "app/pipeline/processing.py", "app/logic/counting.py",
                   "app/logic/hysteresis.py", "app/logic/geometry.py"]},
              "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                           "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                           "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory
                           if torch.cuda.is_available() else None},
              "latency_scope": "End-to-end offline pipeline: decode, inference, tracking, counting, drawing and JPEG; excludes database and viewer transport. FPS includes cold first inference. Detector inference uses Ultralytics speed telemetry.",
              "reference_semantics": manifest["reference_semantics"], "area": manifest["area"],
              "ground_truth_format_url": manifest["format_reference"],
              "coordinate_conversion": manifest["coordinate_conversion"],
              "labels": lock["labels"],
              "effective_settings": {"imgsz": 640, "iou": 0.45, "confirmation": 3,
                                     "missing_grace_seconds": 2,
                                     "tracker_sha256": sha256(ROOT / "configs" / "bytetrack.yaml")},
              "event_tolerance_seconds": 1, "python": platform.python_version(),
              "torch": torch.__version__, "ultralytics": ultralytics.__version__,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "partial_evaluation": args.max_frames is not None,
              "scope": "Selected public training sequences; derived area-counting scores, not official MOT benchmark scores",
              "sequences": []}
    for sequence in manifest["sequences"]:
        if args.sequence and sequence["name"] != args.sequence:
            continue
        if not args.sequence and sequence["split"] != args.split:
            continue
        directory = args.data / sequence["name"]
        video, gt = directory / "video.mp4", directory / "gt.txt"
        record = next(item for item in lock["sequences"] if item["name"] == sequence["name"])
        if sha256(video) != record["video_sha256"] or sha256(gt) != record["gt_sha256"]:
            raise ValueError("benchmark files differ from download hashes")
        truth = load_mot_truth(gt, sequence["width"], sequence["height"])
        alignment = validate_alignment(video, truth, sequence)
        truth_events, truth_occupancy = reference_events(truth, sequence, manifest["area"])
        item = {"name": sequence["name"], "split": sequence["split"], "alignment": alignment,
                "video_url": sequence["video_url"],
                "video_sha256": record["video_sha256"], "gt_sha256": record["gt_sha256"],
                "reference_events": truth_events, "reference_occupancy": truth_occupancy, "profiles": []}
        report["sequences"].append(item)
        for model in args.models:
            model_path = Path(model).resolve()
            for confidence in args.confidences:
                print(f"Evaluating {sequence['name']} {model_path.name} conf={confidence}", flush=True)
                item["profiles"].append(evaluate_profile(video, model_path, sequence, manifest["area"],
                                                        truth_events, truth_occupancy, confidence,
                                                        args.device, args.max_frames))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
