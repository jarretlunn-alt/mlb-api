import gzip
import json

from mlb_pipeline import raw_store


def test_raw_path_layout(tmp_path):
    path = raw_store.raw_path(tmp_path, "schedule", "2026-07-01")
    assert path == tmp_path / "schedule" / "2026-07-01.json.gz"

    path = raw_store.raw_path(tmp_path, "feed_live", 700001)
    assert path == tmp_path / "feed_live" / "700001.json.gz"


def test_save_and_load_roundtrip(tmp_path):
    payload = {"gamePk": 700001, "nested": {"a": [1, 2, 3]}, "unicode": "Peña"}
    path = raw_store.raw_path(tmp_path, "boxscore", 700001)

    raw_store.save_raw(payload, path)

    assert path.exists()
    assert raw_store.load_raw(path) == payload
    # File really is gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert json.load(f) == payload


def test_save_is_idempotent(tmp_path):
    path = raw_store.raw_path(tmp_path, "schedule", "2026-07-01")
    raw_store.save_raw({"v": 1}, path)
    raw_store.save_raw({"v": 2}, path)

    assert raw_store.load_raw(path) == {"v": 2}
