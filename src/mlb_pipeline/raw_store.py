"""Persist raw API responses as gzipped JSON before any transformation.

Invariant: raw responses are saved before normalization so any load can be
replayed without re-hitting the API. Writes are idempotent (same key
overwrites the same path).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def raw_path(raw_dir: Path, kind: str, key) -> Path:
    """Path for a raw payload, e.g. data/raw/schedule/2026-07-03.json.gz."""
    return Path(raw_dir) / kind / f"{key}.json.gz"


def save_raw(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def load_raw(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
