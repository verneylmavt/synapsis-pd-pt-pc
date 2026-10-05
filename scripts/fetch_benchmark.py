"""Fetch small official videos/labels without downloading a full training archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "mot15.json"
RANGE_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _range(url, start, expected_total, remaining_bytes, retries):
    end = start + RANGE_BYTES - 1
    if expected_total is not None:
        end = min(end, expected_total - 1)
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": "Synapsis benchmark evaluation", "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=30) as response:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or match is None:
                    raise ValueError("server did not supply a valid byte range")
                actual_start, actual_end, total = map(int, match.groups())
                if total > remaining_bytes:
                    raise ValueError("download budget exceeded")
                if (actual_start != start or actual_end != min(end, total - 1)
                        or actual_end < actual_start or (expected_total is not None and total != expected_total)):
                    raise ValueError("server supplied an inconsistent byte range")
                length = actual_end - actual_start + 1
                if int(response.headers.get("Content-Length", length)) != length:
                    raise ValueError("byte range length does not match headers")
                body = bytearray()
                while len(body) < length:
                    chunk = response.read(min(65536, length - len(body)))
                    if not chunk:
                        raise OSError("truncated byte range")
                    body.extend(chunk)
                if response.read(1):
                    raise ValueError("byte range exceeded declared length")
                return bytes(body), total
        except OSError:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def download(url: str, target: Path, remaining_bytes: int, retries: int = 3) -> int:
    if target.exists():
        if target.stat().st_size > remaining_bytes:
            raise ValueError("download budget exceeded")
        return target.stat().st_size
    temporary = target.with_suffix(target.suffix + ".part")
    checkpoint = temporary.with_suffix(temporary.suffix + ".json")
    offset, total = 0, None
    if temporary.exists() and checkpoint.exists():
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (record["url"] != url or temporary.stat().st_size != record["bytes"]
                or sha256(temporary) != record["sha256"]):
            raise ValueError("partial download checkpoint failed integrity check")
        offset, total = record["bytes"], record["total"]
        if total > remaining_bytes:
            raise ValueError("download budget exceeded")
    else:
        # Legacy full-GET partials lack verified boundaries and cannot be resumed.
        temporary.unlink(missing_ok=True)
        checkpoint.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    while total is None or offset < total:
        body, total = _range(url, offset, total, remaining_bytes, retries)
        with temporary.open("ab") as destination:
            destination.write(body)
        offset += len(body)
        checkpoint.write_text(json.dumps({"url": url, "bytes": offset, "total": total,
                                           "sha256": sha256(temporary)}), encoding="utf-8")
        print(f"{target.name}: {offset}/{total} bytes verified", flush=True)
    temporary.replace(target)
    checkpoint.unlink(missing_ok=True)
    return total


def fetch(manifest_path: Path, output: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    remaining = manifest["maximum_download_bytes"]
    prior_lock_path = output / "download-lock.json"
    prior_lock = json.loads(prior_lock_path.read_text(encoding="utf-8")) if prior_lock_path.exists() else None
    labels = output / "MOT15Labels.zip"
    remaining -= download(manifest["labels_url"], labels, remaining)
    if prior_lock and sha256(labels) != prior_lock["labels"]["sha256"]:
        raise ValueError("cached labels differ from recorded hash; remove the corrupt download explicitly")
    record = {"manifest_sha256": sha256(manifest_path), "labels": {
        "url": manifest["labels_url"], "sha256": sha256(labels), "bytes": labels.stat().st_size},
        "sequences": []}
    with zipfile.ZipFile(labels) as archive:
        for sequence in manifest["sequences"]:
            directory = output / sequence["name"]
            directory.mkdir(exist_ok=True)
            video = directory / "video.mp4"
            remaining -= download(sequence["video_url"], video, remaining)
            if prior_lock:
                prior = next((item for item in prior_lock["sequences"] if item["name"] == sequence["name"]), None)
                if prior and sha256(video) != prior["video_sha256"]:
                    raise ValueError("cached video differs from recorded hash; remove the corrupt download explicitly")
            suffix = f"/{sequence['name']}/gt/gt.txt"
            members = [name for name in archive.namelist() if ("/" + name).endswith(suffix)]
            if len(members) != 1:
                raise ValueError(f"expected exactly one human annotation file for {sequence['name']}")
            member = archive.getinfo(members[0])
            if member.file_size > 10_000_000:
                raise ValueError("annotation file is unexpectedly large")
            gt = directory / "gt.txt"
            gt.write_bytes(archive.read(member))
            record["sequences"].append({**sequence, "video_sha256": sha256(video),
                                        "video_bytes": video.stat().st_size, "gt_sha256": sha256(gt),
                                        "gt_member": member.filename})
    lock_path = output / "download-lock.json"
    lock_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return lock_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "benchmarks" / "mot15")
    args = parser.parse_args()
    print(fetch(args.manifest, args.output))


if __name__ == "__main__":
    main()
