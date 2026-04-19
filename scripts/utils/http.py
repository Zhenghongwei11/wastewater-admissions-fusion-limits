from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests


@dataclass(frozen=True)
class HttpResult:
    url: str
    status_code: int
    elapsed_seconds: float
    bytes_written: int | None = None


def build_session(*, user_agent: str | None, app_token: str | None) -> requests.Session:
    s = requests.Session()
    headers: dict[str, str] = {"Accept": "*/*"}
    if user_agent:
        headers["User-Agent"] = user_agent
    if app_token:
        headers["X-App-Token"] = app_token
    s.headers.update(headers)
    return s


def request_json(
    session: requests.Session,
    *,
    url: str,
    params: Mapping[str, Any] | None = None,
    timeout_seconds: int = 60,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    log_path: Path | None = None,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = session.get(url, params=params, timeout=timeout_seconds)
            r.raise_for_status()
            data = r.json()
            if log_path:
                _append_log(
                    log_path,
                    {
                        "event": "http_json",
                        "url": r.url,
                        "status_code": r.status_code,
                        "elapsed_seconds": round(time.time() - t0, 3),
                    },
                )
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable") from last_err


def download_file(
    session: requests.Session,
    *,
    url: str,
    out_path: Path,
    params: Mapping[str, Any] | None = None,
    timeout_seconds: int = 120,
    retries: int = 2,
    backoff_seconds: float = 2.0,
    chunk_bytes: int = 1024 * 1024,
    log_path: Path | None = None,
) -> HttpResult:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        bytes_written = 0
        try:
            with session.get(url, params=params, timeout=timeout_seconds, stream=True) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_bytes):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)
            tmp_path.replace(out_path)
            if log_path:
                _append_log(
                    log_path,
                    {
                        "event": "http_download",
                        "url": r.url,
                        "status_code": r.status_code,
                        "elapsed_seconds": round(time.time() - t0, 3),
                        "bytes_written": bytes_written,
                        "out_path": str(out_path),
                    },
                )
            return HttpResult(
                url=str(r.url),
                status_code=r.status_code,
                elapsed_seconds=round(time.time() - t0, 3),
                bytes_written=bytes_written,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable") from last_err


def _append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

