"""Command-line entry point: ingest, build-marts, dashboard, restore, train, predict."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from . import dashboard, db, ingest, ingest_odds, ingest_weather, marts, predict as predict_module, schedule
from .api_client import MLBApiClient
from .config import Settings
from .models import ensemble

DEFAULT_TRAIN_SEASONS = [2023, 2024]


def _model_path(settings: Settings) -> Path:
    return settings.data_dir / "models" / "mlb_ensemble.pkl"


def _metadata_path(settings: Settings) -> Path:
    return settings.data_dir / "models" / "metadata.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlb-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch and load completed games for a date range")
    p_ingest.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p_ingest.add_argument("--end-date", help="YYYY-MM-DD (defaults to start date)")

    p_fetch = sub.add_parser(
        "fetch-schedule", help="Fetch upcoming schedule and store in fact_game (no overwrites)"
    )
    p_fetch.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p_fetch.add_argument("--end-date", help="YYYY-MM-DD (defaults to start date)")

    p_weather = sub.add_parser("ingest-weather", help="Fetch pre-game weather from Open-Meteo (free, no key required)")
    p_weather.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p_weather.add_argument("--end-date", help="YYYY-MM-DD (defaults to start-date)")

    p_odds = sub.add_parser("ingest-odds", help="Fetch opening lines from The Odds API (requires ODDS_API_KEY)")
    p_odds.add_argument("--start-date", help="YYYY-MM-DD — backfill from this date (requires paid plan)")
    p_odds.add_argument("--end-date", help="YYYY-MM-DD — backfill end date (defaults to start-date)")

    p_train = sub.add_parser("train", help="Train the XGBoost ensemble model and save it to disk")
    p_train.add_argument("--seasons", type=int, nargs="+", default=None,
                          help=f"Seasons to train on (default: {DEFAULT_TRAIN_SEASONS})")

    p_predict = sub.add_parser("predict", help="Predict win probabilities for a date's games")
    p_predict.add_argument("--date", required=True, help="YYYY-MM-DD")

    sub.add_parser("build-marts", help="Create mart views and export Parquet")
    sub.add_parser("dashboard", help="Render the static HTML dashboard")
    sub.add_parser("restore", help="Rebuild the warehouse from exported Parquet")
    return parser


def run_train(con, settings: Settings, seasons: list[int] | None = None) -> int:
    seasons = seasons or DEFAULT_TRAIN_SEASONS
    model = ensemble.train(con, seasons)
    model_path = _model_path(settings)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    metadata = {
        "seasons": seasons,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "ensemble",
    }
    _metadata_path(settings).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model trained on seasons {seasons}. Saved to {model_path}")
    return 0


def run_predict(con, settings: Settings, target_date: str) -> int:
    """Predict using the saved ensemble model; fall back to Pythagorean if no model."""
    model_path = _model_path(settings)
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        games = con.execute(
            "SELECT game_pk, home_team_id, away_team_id FROM fact_game WHERE official_date = ?",
            [target_date],
        ).fetchall()
        if not games:
            print(f"No games found for {target_date}. Run fetch-schedule first.")
            return 0
        predictions = [
            ensemble.predict(model, con, game_pk, home_id, away_id, target_date)
            for game_pk, home_id, away_id in games
        ]
        predict_module.save_predictions(con, predictions)
        print(f"{'game_pk':>10} | {'home_win_prob':>13} | {'away_win_prob':>13}")
        for pred in predictions:
            print(f"{pred.game_pk:>10} | {pred.home_win_prob:>13.3f} | {pred.away_win_prob:>13.3f}")
        return 0
    else:
        print(
            f"No model found at {model_path}. "
            "Run python -m mlb_pipeline.cli train first."
        )
        return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    con = db.connect(settings.db_path)
    exit_code = 0
    try:
        if args.command == "fetch-schedule":
            client = MLBApiClient()
            end_date = args.end_date or args.start_date
            result = schedule.fetch_schedule(client, con, args.start_date, end_date)
            print(f"Fetched schedule {result['start_date']} – {result['end_date']}: "
                  f"{result['games_scheduled']} upcoming game(s) added")
        elif args.command == "ingest":
            client = MLBApiClient()
            end_date = args.end_date or args.start_date
            results = ingest.ingest_date_range(client, con, settings, args.start_date, end_date)
            for result in results:
                print(f"{result['date']}: loaded {result['games_loaded']} completed game(s)")
        elif args.command == "ingest-weather":
            from .weather_client import WeatherClient
            wc = WeatherClient()
            end_date = args.end_date or args.start_date
            result = ingest_weather.ingest_weather_range(wc, con, args.start_date, end_date)
            print(
                f"Processed {result['days_processed']} day(s), "
                f"loaded weather for {result['games_loaded']} game(s)"
            )
        elif args.command == "ingest-odds":
            from .odds_client import OddsApiClient
            odds_client = OddsApiClient()
            if args.start_date:
                end_date = args.end_date or args.start_date
                result = ingest_odds.ingest_historical_odds(odds_client, con, args.start_date, end_date)
                print(
                    f"Processed {result['days_processed']} day(s), "
                    f"loaded {result['games_loaded']} game(s). "
                    f"API requests remaining: {result['api_remaining']}"
                )
            else:
                result = ingest_odds.ingest_live_odds(odds_client, con)
                print(
                    f"Loaded {result['games_loaded']} game(s). "
                    f"API requests remaining: {result['api_remaining']}"
                )
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
        elif args.command == "train":
            exit_code = run_train(con, settings, args.seasons)
        elif args.command == "predict":
            exit_code = run_predict(con, settings, args.date)
    finally:
        con.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
