from mlb_pipeline import db, ingest, marts


def test_team_records(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    marts.create_views(con)

    records = con.execute(
        "SELECT team, games, wins, losses, runs_scored, runs_allowed FROM mart_team_records"
    ).fetchall()
    by_team = {r[0]: r for r in records}

    assert by_team["Seattle Mariners"] == ("Seattle Mariners", 1, 1, 0, 5, 3)
    assert by_team["Athletics"] == ("Athletics", 1, 0, 1, 3, 5)


def test_top_hitters_and_pitchers(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    marts.create_views(con)

    hitters = con.execute("SELECT player, hits, home_runs, rbi FROM mart_top_hitters").fetchall()
    assert hitters[0] == ("Slugging Catcher", 3, 2, 4)

    pitchers = con.execute(
        "SELECT player, strikeouts, innings_pitched FROM mart_top_pitchers"
    ).fetchall()
    assert pitchers[0] == ("Lefty Starter", 7, "6.0")


def test_export_and_restore_roundtrip(fake_client, con, settings, tmp_path):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    paths = marts.export_parquet(con, settings.marts_dir)

    try:
        con.execute("SELECT 1 FROM fact_prediction LIMIT 1")
        expected_count = len(db.TABLES) + len(marts.MART_VIEWS)
    except Exception:
        expected_count = len(db.TABLES) + len(marts.MART_VIEWS) - 1
    assert len(paths) == expected_count
    for path in paths:
        assert path.exists()

    # Restore into a fresh warehouse and confirm base tables round-trip
    fresh = db.connect(tmp_path / "fresh.duckdb")
    try:
        restored = marts.restore_from_parquet(fresh, settings.marts_dir)
        assert restored["fact_game"] == 1
        assert restored["dim_player"] == 4

        # Restore is idempotent too
        marts.restore_from_parquet(fresh, settings.marts_dir)
        assert fresh.execute("SELECT count(*) FROM fact_game").fetchone()[0] == 1
    finally:
        fresh.close()
