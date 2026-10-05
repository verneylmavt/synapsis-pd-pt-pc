import io
import time

import pytest

from scripts import fetch_benchmark as downloader


class Response(io.BytesIO):
    status = 206
    def __init__(self, body, content_range):
        super().__init__(body)
        self.headers = {"Content-Range": content_range, "Content-Length": str(len(body))}


def range_server(monkeypatch, content=b"abcdefghijkl", *, fail_first=False):
    calls = []
    def open_request(request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        calls.append((start, end))
        if fail_first and len(calls) == 1:
            raise TimeoutError("transient read")
        end = min(end, len(content) - 1)
        return Response(content[start:end + 1], f"bytes {start}-{end}/{len(content)}")
    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(downloader, "RANGE_BYTES", 4, raising=False)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return calls


def test_ranges_reassemble_exact_bytes_and_retry_without_duplicating_partial_chunk(tmp_path, monkeypatch):
    calls = range_server(monkeypatch, fail_first=True)
    target = tmp_path / "video.mp4"
    assert downloader.download("https://official.test/video", target, 100) == 12
    assert target.read_bytes() == b"abcdefghijkl"
    assert calls == [(0, 3), (0, 3), (4, 7), (8, 11)]
    assert not target.with_suffix(".mp4.part").exists()


def test_incorrect_content_range_is_rejected_before_committing_target(tmp_path, monkeypatch):
    range_server(monkeypatch)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *args, **kwargs:
                        Response(b"abcd", "bytes 1-4/12"))
    target = tmp_path / "video.mp4"
    with pytest.raises(ValueError, match="range"):
        downloader.download("https://official.test/video", target, 100)
    assert not target.exists()


def test_declared_total_cannot_exceed_remaining_budget(tmp_path, monkeypatch):
    range_server(monkeypatch)
    target = tmp_path / "video.mp4"
    with pytest.raises(ValueError, match="budget"):
        downloader.download("https://official.test/video", target, 10)
    assert not target.exists()


def test_retry_exhaustion_preserves_verified_prefix_for_next_invocation(tmp_path, monkeypatch):
    range_server(monkeypatch)
    original = downloader.urllib.request.urlopen
    def fail_second_range(request, timeout):
        if request.get_header("Range").startswith("bytes=4-"):
            raise TimeoutError("network unavailable")
        return original(request, timeout)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail_second_range)
    target = tmp_path / "video.mp4"
    with pytest.raises(TimeoutError):
        downloader.download("https://official.test/video", target, 100, retries=1)
    assert target.with_suffix(".mp4.part").read_bytes() == b"abcd"
    calls = range_server(monkeypatch)
    assert downloader.download("https://official.test/video", target, 100) == 12
    assert calls == [(4, 7), (8, 11)]
    assert target.read_bytes() == b"abcdefghijkl"


def test_tampered_partial_prefix_is_rejected_instead_of_silently_resumed(tmp_path, monkeypatch):
    range_server(monkeypatch)
    original = downloader.urllib.request.urlopen
    def fail_second_range(request, timeout):
        if request.get_header("Range").startswith("bytes=4-"):
            raise TimeoutError("network unavailable")
        return original(request, timeout)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", fail_second_range)
    target = tmp_path / "video.mp4"
    with pytest.raises(TimeoutError):
        downloader.download("https://official.test/video", target, 100, retries=1)
    target.with_suffix(".mp4.part").write_bytes(b"fake")
    with pytest.raises(ValueError, match="integrity"):
        downloader.download("https://official.test/video", target, 100)
