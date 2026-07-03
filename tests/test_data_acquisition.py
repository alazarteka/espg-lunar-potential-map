"""Tests for the resumable download logic in ``src.data_acquisition``.

These spin up a local Range-capable HTTP server so the real ``requests`` code
path (streaming, ``Range`` headers, size verification) is exercised end to end.
"""

import http.server
import socketserver
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from src.data_acquisition import DataManager

CONTENT = bytes(range(256)) * 4000  # ~1 MB deterministic payload


def _make_server(handler_cls) -> tuple[socketserver.TCPServer, str]:
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/file.bin"


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``CONTENT`` with correct ``Range`` / 206 support."""

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            body = CONTENT[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(CONTENT) - 1}/{len(CONTENT)}"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(CONTENT)))
            self.end_headers()
            self.wfile.write(CONTENT)

    def log_message(self, *args) -> None:  # silence test noise
        pass


class _IgnoreRangeHandler(http.server.BaseHTTPRequestHandler):
    """Always returns the full body with 200, ignoring any ``Range`` header."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(CONTENT)))
        self.end_headers()
        self.wfile.write(CONTENT)

    def log_message(self, *args) -> None:
        pass


class _TruncatedHandler(http.server.BaseHTTPRequestHandler):
    """Advertises the full length but sends a short body (corrupt download)."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(CONTENT)))
        self.end_headers()
        self.wfile.write(CONTENT[:100_000])

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def dm(tmp_path: Path) -> DataManager:
    return DataManager(base_dir=str(tmp_path), base_url="http://unused")


def _serve(handler_cls) -> Iterator[str]:
    httpd, url = _make_server(handler_cls)
    try:
        yield url
    finally:
        httpd.shutdown()


@pytest.fixture
def range_url() -> Iterator[str]:
    yield from _serve(_RangeHandler)


def test_fresh_download(dm: DataManager, range_url: str) -> None:
    dest = dm.base_dir / "file.bin"
    dm.download_file(range_url, dest)
    assert dest.read_bytes() == CONTENT
    assert not dest.with_suffix(".bin.part").exists()


def test_resume_appends_not_truncates(dm: DataManager, range_url: str) -> None:
    dest = dm.base_dir / "file.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(CONTENT[:300_000])  # simulate an interrupted download

    dm.download_file(range_url, dest)

    # Must have appended the remaining bytes, not restarted and truncated.
    assert dest.read_bytes() == CONTENT
    assert not part.exists()


def test_skip_existing(dm: DataManager, range_url: str) -> None:
    dest = dm.base_dir / "file.bin"
    dest.write_bytes(CONTENT)
    mtime = dest.stat().st_mtime_ns
    dm.download_file(range_url, dest)
    assert dest.stat().st_mtime_ns == mtime  # untouched, no re-download


def test_server_ignoring_range_restarts_cleanly(dm: DataManager) -> None:
    for url in _serve(_IgnoreRangeHandler):
        dest = dm.base_dir / "file.bin"
        part = dest.with_suffix(".bin.part")
        part.write_bytes(b"\x00" * 300_000)  # bogus partial

        dm.download_file(url, dest)

        # A 200 (Range ignored) must overwrite the bogus partial, not append.
        assert dest.read_bytes() == CONTENT


def test_failure_keeps_partial_and_no_promotion(dm: DataManager) -> None:
    for url in _serve(_TruncatedHandler):
        dest = dm.base_dir / "file.bin"
        part = dest.with_suffix(".bin.part")

        # Either requests detects the short body, or our own size check does.
        with pytest.raises((requests.exceptions.RequestException, OSError)):
            dm.download_file(url, dest)

        assert part.exists()  # kept for the next run to resume
        assert not dest.exists()  # corrupt data never promoted


def test_list_remote_dirs_filters_navigation(dm: DataManager, monkeypatch) -> None:
    """Absolute breadcrumb links must be dropped by shape, keeping real dirs."""
    html = (
        '<a href="/data/lp-er-calibrated">parent</a>'
        '<a href="../">up</a>'
        '<a href="https://example.com/other/">external</a>'
        '<a href="1998/">1998</a>'
        '<a href="1999/">1999</a>'
        '<a href="file.TAB">a file</a>'
    )

    class _Resp:
        text = html

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        "src.data_acquisition.session.get", lambda *a, **k: _Resp()
    )
    assert dm.list_remote_dirs("") == ["1998", "1999"]
