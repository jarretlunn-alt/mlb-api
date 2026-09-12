"""Pipeline paths and settings, overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    site_dir: Path = Path("site")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def marts_dir(self) -> Path:
        return self.data_dir / "marts"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("MLB_DATA_DIR", "data")),
            site_dir=Path(os.environ.get("MLB_SITE_DIR", "site")),
        )
