"""Render the static HTML dashboard from the warehouse mart views."""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

from . import marts

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Pipeline Dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .generated {{ color: #888; font-size: 0.9rem; margin-bottom: 2rem; }}
  section {{ margin-bottom: 2.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
  th, td {{ border-bottom: 1px solid #ccc3; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #8882; }}
  tr:hover td {{ background: #8881; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .empty {{ color: #888; font-style: italic; }}
</style>
</head>
<body>
<h1>MLB Pipeline Dashboard</h1>
<p class="generated">Generated {generated} UTC &middot; data from the public MLB Stats API</p>
{sections}
</body>
</html>
"""

SECTIONS = [
    ("Upcoming Predictions", "upcoming", "SELECT * FROM mart_upcoming_predictions"),
    ("Predictions", "predictions", "SELECT * FROM mart_predictions"),
    ("Games by Date", "games", "SELECT * FROM mart_games_by_date"),
    ("Team Records &amp; Runs", "records", "SELECT * FROM mart_team_records"),
    ("Top Hitters (by hits, HR, RBI)", "hitters", "SELECT * FROM mart_top_hitters LIMIT 15"),
    ("Top Pitchers (by strikeouts, IP)", "pitchers", "SELECT * FROM mart_top_pitchers LIMIT 15"),
]


def _render_table(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return '<p class="empty">No data ingested yet.</p>'
    numeric = [
        all(isinstance(row[i], (int, float)) or row[i] is None for row in rows)
        for i in range(len(columns))
    ]

    def cell(tag: str, i: int, value) -> str:
        cls = ' class="num"' if numeric[i] else ""
        text = "" if value is None else html.escape(str(value))
        return f"<{tag}{cls}>{text}</{tag}>"

    head = "".join(cell("th", i, c.replace("_", " ")) for i, c in enumerate(columns))
    body = "".join(
        "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard(con, site_dir: Path) -> Path:
    marts.create_views(con)
    sections_html = []
    for section_info in SECTIONS:
        title, section_id, query = section_info

        if section_id == "upcoming":
            table_html = _render_guarded_prediction_section(
                con, "mart_upcoming_predictions",
                "No upcoming predictions — run: python -m mlb_pipeline.cli fetch-schedule, then predict"
            )
        elif section_id == "predictions":
            table_html = _render_predictions_section(con)
        else:
            try:
                result = con.execute(query)
                columns = [d[0] for d in result.description]
                rows = result.fetchall()
                table_html = _render_table(columns, rows)
            except Exception:
                table_html = '<p class="empty">Error loading section.</p>'

        sections_html.append(f"<section><h2>{title}</h2>{table_html}</section>")

    page = PAGE_TEMPLATE.format(
        generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        sections="\n".join(sections_html),
    )
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    out_path = site_dir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


def _render_guarded_prediction_section(con, view_name: str, empty_msg: str) -> str:
    """Render a mart view that depends on fact_prediction, gracefully if missing."""
    try:
        con.execute("SELECT 1 FROM fact_prediction LIMIT 1")
    except Exception:
        return f'<p class="empty">{empty_msg}</p>'
    try:
        result = con.execute(f"SELECT * FROM {view_name}")
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
        if not rows:
            return f'<p class="empty">{empty_msg}</p>'
        return _render_table(columns, rows)
    except Exception:
        return f'<p class="empty">{empty_msg}</p>'


def _render_predictions_section(con) -> str:
    try:
        con.execute("SELECT 1 FROM fact_prediction LIMIT 1")
    except Exception:
        return '<p class="empty">No predictions yet — run: python -m mlb_pipeline.cli predict</p>'

    try:
        result = con.execute("SELECT * FROM mart_predictions")
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
        if not rows:
            return '<p class="empty">No predictions yet — run: python -m mlb_pipeline.cli predict</p>'
        return _render_table(columns, rows)
    except Exception:
        return '<p class="empty">No predictions yet — run: python -m mlb_pipeline.cli predict</p>'
