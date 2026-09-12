"""Command-line entry point: ingest, build-marts, dashboard, restore."""

from __future__ import annotations

import argparse

from . import dashboard, db, ingest, marts
from .api_client import MLBApiClient
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlb-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch and load completed games for a date range")
    p_ingest.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p_ingest.add_argument("--end-date", help="YYYY-MM-DD (defaults to start date)")

    sub.add_parser("build-marts", help="Create mart views and export Parquet")
    sub.add_parser("dashboard", help="Render the static HTML dashboard")
    sub.add_parser("restore", help="Rebuild the warehouse from exported Parquet")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    con = db.connect(settings.db_path)
    try:
        if args.command == "ingest":
            client = MLBApiClient()
            end_date = args.end_date or args.start_date
            results = ingest.ingest_date_range(client, con, settings, args.start_date, end_date)
            for result in results:
                print(f"{result['date']}: loaded {result['games_loaded']} completed game(s)")
        elif args.command == "build-marts":
            paths = marts.export_parquet(con, settings.marts_dir)
            print(f"Exported {len(paths)} Parquet files to {settings.marts_dir}")
        elif args.command == "dashboard":
            out_path = dashboard.render_dashboard(con, settings.site_dir)
            print(f"Dashboard written to {out_path}")
        elif args.command == "restore":
            restored = marts.restore_from_parquet(con, settings.marts_dir)
            if restored:
                for table, count in restored.items():
                    print(f"{table}: {count} rows")
            else:
                print(f"No Parquet exports found in {settings.marts_dir}; nothing restored")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
