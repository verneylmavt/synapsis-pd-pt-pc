from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def redact_uri(uri: str) -> str:
    """Expose protocol/host/port only; paths may also contain access tokens."""
    try:
        parts = urlsplit(uri)
        if parts.scheme not in {"rtsp", "http", "https"}:
            return "local video"
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}"
    except ValueError:
        return "camera"


def validate_camera_uri(uri: str) -> str:
    uri = uri.strip()
    if len(uri) > 4096 or any(ord(c) < 32 or c.isspace() for c in uri):
        raise ValueError("Camera URL contains invalid characters")
    try:
        parts = urlsplit(uri)
        if parts.scheme not in {"rtsp", "http", "https"} or not parts.hostname or parts.fragment:
            raise ValueError("Use a direct RTSP, HTTP or HTTPS camera URL")
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError("Invalid camera port")
        try:
            address = ipaddress.ip_address(parts.hostname)
        except ValueError:
            if parts.hostname.lower() == "metadata.google.internal":
                raise ValueError("Metadata endpoints are not camera sources")
        else:
            if address.is_unspecified or address.is_multicast or address.is_link_local:
                raise ValueError("This address is not a camera source")
    except ValueError as error:
        raise ValueError("Invalid direct camera URL") from error
    return uri
