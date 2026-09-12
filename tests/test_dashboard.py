from mlb_pipeline import dashboard, ingest


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
