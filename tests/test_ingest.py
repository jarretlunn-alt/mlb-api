import pytest

from mlb_pipeline import db, ingest, raw_store

EXPECTED_COUNTS = {
    "dim_team": 2,
    "dim_player": 4,
    "fact_game": 1,
    "fact_team_game": 2,
    "fact_player_game_batting": 2,
    "fact_player_game_pitching": 2,
}


def table_counts(con) -> dict:
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in db.TABLES}


def test_date_range():
    assert list(ingest.date_range("2026-07-01", "2026-07-03")) == [
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
    ]
    assert list(ingest.date_range("2026-07-01", "2026-07-01")) == ["2026-07-01"]
    with pytest.raises(ValueError):
        list(ingest.date_range("2026-07-02", "2026-07-01"))


def test_ingest_date_loads_completed_games(fake_client, con, settings):
    result = ingest.ingest_date(fake_client, con, settings, "2026-07-01")

    assert result == {"date": "2026-07-01", "games_loaded": 1}
    assert table_counts(con) == EXPECTED_COUNTS
    # Only the Final game (700001) was fetched, not the Scheduled one (700002)
    assert ("feed_live", 700001) in fake_client.calls
    assert ("feed_live", 700002) not in fake_client.calls


def test_ingest_saves_raw_before_load(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")

    for kind, key in [
        ("schedule", "2026-07-01"),
        ("feed_live", 700001),
        ("boxscore", 700001),
    ]:
        path = raw_store.raw_path(settings.raw_dir, kind, key)
        assert path.exists(), f"missing raw file {path}"

    saved_feed = raw_store.load_raw(raw_store.raw_path(settings.raw_dir, "feed_live", 700001))
    assert saved_feed["gamePk"] == 700001


def test_ingest_is_idempotent(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    first = table_counts(con)

    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    second = table_counts(con)

    assert first == second == EXPECTED_COUNTS
    # Values are replaced, not duplicated
    score = con.execute("SELECT home_score FROM fact_game WHERE game_pk = 700001").fetchone()[0]
    assert score == 5


def test_load_game_updates_on_replay(fake_client, con, settings, feed_live, boxscore):
    ingest.load_game(con, feed_live, boxscore)

    # Simulate a corrected feed: re-load with a changed score
    feed_live["liveData"]["linescore"]["teams"]["home"]["runs"] = 6
    ingest.load_game(con, feed_live, boxscore)

    assert table_counts(con) == EXPECTED_COUNTS
    score = con.execute("SELECT home_score FROM fact_game WHERE game_pk = 700001").fetchone()[0]
    assert score == 6
