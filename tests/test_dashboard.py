from datetime import date, timedelta

from mlb_pipeline import dashboard, ingest, predict as predict_module
from mlb_pipeline.predict import GamePrediction


def test_render_dashboard(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")

    out_path = dashboard.render_dashboard(con, settings.site_dir)

    assert out_path == settings.site_dir / "index.html"
    html = out_path.read_text(encoding="utf-8")
    assert "Seattle Mariners" in html
    assert "Slugging Catcher" in html
    assert "Lefty Starter" in html
    for heading in ("Games by Date", "Team Records", "Top Hitters", "Top Pitchers"):
        assert heading in html


def test_render_dashboard_empty_warehouse(con, settings):
    out_path = dashboard.render_dashboard(con, settings.site_dir)

    html = out_path.read_text(encoding="utf-8")
    assert "No data ingested yet." in html


def test_render_dashboard_upcoming_predictions(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")

    future_date = date.today() + timedelta(days=3)
    con.execute(
        """INSERT INTO fact_game
        (game_pk, official_date, season, game_type, status, home_team_id, away_team_id)
        VALUES (700099, ?, 2026, 'R', 'Scheduled', 136, 133)""",
        [future_date],
    )
    predict_module.save_predictions(con, [GamePrediction(700099, "ensemble", 0.6, 0.4, None, {})])

    out_path = dashboard.render_dashboard(con, settings.site_dir)
    html = out_path.read_text(encoding="utf-8")

    assert "Upcoming Predictions" in html
    assert "60.0" in html
    assert "40.0" in html


def test_render_dashboard_no_upcoming_predictions_is_graceful(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")

    out_path = dashboard.render_dashboard(con, settings.site_dir)
    html = out_path.read_text(encoding="utf-8")

    assert "Upcoming Predictions" in html
    assert "No upcoming predictions yet" in html
