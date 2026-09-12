import json
from pathlib import Path

import pytest

from mlb_pipeline import db
from mlb_pipeline.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def schedule() -> dict:
    return load_fixture("schedule_2026-07-01.json")


@pytest.fixture
def feed_live() -> dict:
    return load_fixture("feed_live_700001.json")


@pytest.fixture
def boxscore() -> dict:
    return load_fixture("boxscore_700001.json")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", site_dir=tmp_path / "site")


@pytest.fixture
def con(settings):
    connection = db.connect(settings.db_path)
    yield connection
    connection.close()


class FakeClient:
    """Stand-in for MLBApiClient that serves fixtures. No network calls in tests."""

    def __init__(self, schedule: dict, feeds: dict, boxscores: dict):
        self._schedule = schedule
        self._feeds = feeds
        self._boxscores = boxscores
        self.calls = []

    def get_schedule(self, start_date, end_date, sport_id=1):
        self.calls.append(("schedule", start_date, end_date))
        return self._schedule

    def get_feed_live(self, game_pk):
        self.calls.append(("feed_live", game_pk))
        return self._feeds[game_pk]

    def get_boxscore(self, game_pk):
        self.calls.append(("boxscore", game_pk))
        return self._boxscores[game_pk]


@pytest.fixture
def fake_client(schedule, feed_live, boxscore) -> FakeClient:
    return FakeClient(schedule, {700001: feed_live}, {700001: boxscore})
