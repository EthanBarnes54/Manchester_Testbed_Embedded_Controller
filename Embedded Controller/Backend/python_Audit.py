from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

_LOCK = threading.Lock()

def audit_log_path() -> Path:
    return Path(os.getenv("TESTBED_AUDIT_LOG") or Path(__file__).with_name("audit_log.jsonl"))


def record(event: dict) -> None:
    """Appends one event, stamped with UTC time. Never raises: a full disk must not stop the rig."""

    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **event}

    try:
        with _LOCK, open(audit_log_path(), "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, default=str) + "\n")

    except OSError:
        pass


@functools.lru_cache(maxsize=1)
def backend_revision() -> str:
    """git describe of this checkout, or "unknown" outside one."""

    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty", "--tags"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 16), b""):
            digest.update(chunk)

    return digest.hexdigest()


def frame_sha256(frame) -> str:
    """A content hash of a DataFrame's values, independent of its index."""

    import pandas as pd

    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def provenance(board_version: dict | None = None, **extra) -> dict:
    """What produced a result: the backend revision, the board's VERSION reply, and anything else given."""

    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend_revision": backend_revision(),
        "board": dict(board_version or {}),
        **extra,
    }
