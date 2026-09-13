"""Unit tests for the train/predict CLI subcommands.

Uses the real (tmp_path-backed) warehouse fixture but mocks ensemble.train /
ensemble.predict / predict.save_predictions so no real model is fit and no
network calls occur.
"""
import json
import pickle
from datetime import date

import pytest

from mlb_pipeline import cli
from mlb_pipeline.predict import GamePrediction


class _FakeModel:
    """Stand-in for a CalibratedClassifierCV: cheap and picklable."""

    def __init__(self, tag):
        self.tag = tag


def test_build_parser_fetch_schedule():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["fetch-schedule", "--start-date", "2026-09-14", "--end-date", "2026-09-20"]
    )
    assert args.command == "fetch-schedule"
    assert args.start_date == "2026-09-14"
    assert args.end_date == "2026-09-20"


def test_build_parser_fetch_schedule_defaults_end_date():
    parser = cli.build_parser()
    args = parser.parse_args(["fetch-schedule", "--start-date", "2026-09-14"])
    assert args.end_date is None


def test_train_default_seasons(con, settings, monkeypatch, capsys):
    seen = {}

    def fake_train(con_arg, seasons):
        seen["con"] = con_arg
        seen["seasons"] = seasons
        return _FakeModel("trained")

    monkeypatch.setattr(cli.ensemble, "train", fake_train)

    exit_code = cli.run_train(con, settings)

    assert exit_code == 0
    assert seen["con"] is con
    assert seen["seasons"] == cli.DEFAULT_TRAIN_SEASONS

    model_path = settings.data_dir / "models" / "mlb_ensemble.pkl"
    metadata_path = settings.data_dir / "models" / "metadata.json"
    assert model_path.exists()
    assert metadata_path.exists()

    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded.tag == "trained"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["seasons"] == cli.DEFAULT_TRAIN_SEASONS
    assert metadata["model"] == "ensemble"
    assert "trained_at" in metadata

    out = capsys.readouterr().out
    assert "Model trained on seasons" in out
    assert str(model_path) in out


def test_train_custom_seasons(con, settings, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.ensemble, "train",
                         lambda con_arg, seasons: seen.setdefault("seasons", seasons) or _FakeModel("x"))

    exit_code = cli.run_train(con, settings, seasons=[2019, 2020])

    assert exit_code == 0
    assert seen["seasons"] == [2019, 2020]
    metadata = json.loads((settings.data_dir / "models" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seasons"] == [2019, 2020]


def test_predict_missing_model_exits_1(con, settings, capsys):
    exit_code = cli.run_predict(con, settings, "2024-09-01")

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "python -m mlb_pipeline.cli train" in out


def test_predict_no_games_for_date(con, settings, capsys):
    model_path = settings.data_dir / "models" / "mlb_ensemble.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(_FakeModel("noop"), f)

    exit_code = cli.run_predict(con, settings, "2024-09-01")

    assert exit_code == 0
    assert "No games found for 2024-09-01" in capsys.readouterr().out


def test_predict_with_games_saves_and_prints_table(con, settings, monkeypatch, capsys):
    model_path = settings.data_dir / "models" / "mlb_ensemble.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(_FakeModel("ready"), f)

    con.execute("""INSERT INTO fact_game
        (game_pk, official_date, season, status, home_team_id, away_team_id, home_score, away_score)
        VALUES (700001, ?, 2024, 'Scheduled', 10, 20, NULL, NULL)""", [date(2024, 9, 1)])

    def fake_predict(model, con_arg, game_pk, home_id, away_id, as_of_date):
        assert isinstance(model, _FakeModel)
        assert (game_pk, home_id, away_id, as_of_date) == (700001, 10, 20, "2024-09-01")
        return GamePrediction(game_pk, "ensemble", 0.65, 0.35, None, {})

    saved = {}

    def fake_save_predictions(con_arg, preds):
        saved["preds"] = preds
        return len(preds)

    monkeypatch.setattr(cli.ensemble, "predict", fake_predict)
    monkeypatch.setattr(cli.predict_module, "save_predictions", fake_save_predictions)

    exit_code = cli.run_predict(con, settings, "2024-09-01")

    assert exit_code == 0
    assert len(saved["preds"]) == 1
    assert saved["preds"][0].game_pk == 700001

    out = capsys.readouterr().out
    assert "700001" in out
    assert "0.650" in out
    assert "0.350" in out
