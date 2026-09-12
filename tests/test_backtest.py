import math
from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from mlb_pipeline import backtest, db

TEST_SEASON = 2026
TRAIN_SEASONS = [2024, 2025]


@dataclass
class FakePrediction:
    """Stand-in for predict.GamePrediction; backtest must not import a model."""

    game_pk: int
    home_win_prob: float
    model_name: str = "fake"


def _make_games(n: int = 20) -> list[dict]:
    """n synthetic games with known outcomes: even game index -> home wins."""
    start = date(TEST_SEASON, 4, 1)
    games = []
    for i in range(n):
        home_win = i % 2 == 0
        home_id, away_id = 100 + (i % 5), 200 + (i % 7)
        home_score, away_score = (5, 3) if home_win else (2, 6)
        games.append(
            {
                "game_pk": 900000 + i,
                "official_date": start + timedelta(days=i),
                "season": TEST_SEASON,
                "game_type": "R",
                "status": "Final",
                "venue": f"Park {i % 5}",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_score": home_score,
                "away_score": away_score,
                "winning_team_id": home_id if home_win else away_id,
            }
        )
    return games


def _team_game_rows(game: dict) -> list[dict]:
    hs, as_ = game["home_score"], game["away_score"]
    return [
        {
            "game_pk": game["game_pk"],
            "team_id": game["home_team_id"],
            "opponent_team_id": game["away_team_id"],
            "is_home": True,
            "runs_scored": hs,
            "runs_allowed": as_,
            "hits": None,
            "errors": None,
            "win": hs > as_,
        },
        {
            "game_pk": game["game_pk"],
            "team_id": game["away_team_id"],
            "opponent_team_id": game["home_team_id"],
            "is_home": False,
            "runs_scored": as_,
            "runs_allowed": hs,
            "hits": None,
            "errors": None,
            "win": as_ > hs,
        },
    ]


@pytest.fixture
def games() -> list[dict]:
    return _make_games(20)


@pytest.fixture
def mem_con(games):
    """In-memory DuckDB seeded with 20 synthetic final games."""
    con = db.connect(":memory:")
    db.upsert(con, "fact_game", games)
    db.upsert(con, "fact_team_game", [r for g in games for r in _team_game_rows(g)])
    yield con
    con.close()


def _perfect_predictions(games):
    return [
        {
            "game_pk": g["game_pk"],
            "date": g["official_date"].isoformat(),
            "home_id": g["home_team_id"],
            "away_id": g["away_team_id"],
            "home_win_prob": 1.0 if g["winning_team_id"] == g["home_team_id"] else 0.0,
            "actual_home_win": g["winning_team_id"] == g["home_team_id"],
        }
        for g in games
    ]


def _coinflip_predictions(games):
    preds = _perfect_predictions(games)
    for p in preds:
        p["home_win_prob"] = 0.5
    return preds


# --------------------------------------------------------------------------- #
# Scoring metrics
# --------------------------------------------------------------------------- #


def test_brier_score_perfect_predictions_is_zero(games):
    assert backtest.brier_score(_perfect_predictions(games)) == 0.0


def test_brier_score_coinflip_is_quarter(games):
    assert backtest.brier_score(_coinflip_predictions(games)) == pytest.approx(0.25)


def test_brier_score_empty_is_nan():
    assert math.isnan(backtest.brier_score([]))


def test_log_loss_perfect_predictions_is_zero(games):
    assert backtest.log_loss(_perfect_predictions(games)) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_coinflip_is_ln2(games):
    assert backtest.log_loss(_coinflip_predictions(games)) == pytest.approx(math.log(2))


def test_log_loss_confident_wrong_is_finite():
    preds = [{"home_win_prob": 1.0, "actual_home_win": False}]
    loss = backtest.log_loss(preds)
    assert math.isfinite(loss) and loss > 30


def test_calibration_buckets_keys_are_probability_ranges(games):
    buckets = backtest.calibration_buckets(_coinflip_predictions(games), n_buckets=10)

    assert len(buckets) == 10
    assert list(buckets) == [f"{i / 10:.2f}-{(i + 1) / 10:.2f}" for i in range(10)]
    assert all(isinstance(k, str) and "-" in k for k in buckets)

    populated = buckets["0.50-0.60"]
    assert populated == {"predicted_prob": 0.5, "actual_win_rate": 0.5, "n": 20}
    assert buckets["0.00-0.10"] == {"predicted_prob": None, "actual_win_rate": None, "n": 0}
    assert sum(b["n"] for b in buckets.values()) == 20


def test_calibration_buckets_top_edge_lands_in_last_bucket():
    preds = [{"home_win_prob": 1.0, "actual_home_win": True}]
    buckets = backtest.calibration_buckets(preds, n_buckets=4)
    assert buckets["0.75-1.00"]["n"] == 1


# --------------------------------------------------------------------------- #
# ROI simulation
# --------------------------------------------------------------------------- #


def test_simulate_roi_zero_edge_places_no_bets(games):
    preds = _coinflip_predictions(games)
    for p in preds:
        p["market_home_prob"] = p["home_win_prob"]  # model agrees with the market

    result = backtest.simulate_roi(preds, market_vig=0.0)

    assert result["bets_placed"] == 0
    assert result["total_games"] == 20
    assert result["flat_bet_roi"] == 0.0
    assert result["kelly_roi"] == 0.0


def test_simulate_roi_vig_alone_prevents_bets(games):
    # With no explicit market, a 0.5 model never beats a vigged 0.54 market on
    # either side.
    result = backtest.simulate_roi(_coinflip_predictions(games))
    assert result["bets_placed"] == 0


def test_simulate_roi_perfect_model_is_profitable(games):
    preds = _perfect_predictions(games)
    for p in preds:
        p["market_home_prob"] = 0.5

    result = backtest.simulate_roi(preds, market_vig=0.045, min_edge=0.03)

    assert result["bets_placed"] == 20
    # Every bet wins at decimal odds 1 / (0.5 * 1.045)
    expected_roi = 1.0 / (0.5 * 1.045) - 1.0
    assert result["flat_bet_roi"] == pytest.approx(expected_roi)
    assert result["kelly_roi"] == pytest.approx(expected_roi)


def test_simulate_roi_bets_the_better_side():
    # Model strongly likes the away team; the away side has the edge and wins.
    preds = [
        {"home_win_prob": 0.2, "actual_home_win": False, "market_home_prob": 0.5},
        {"home_win_prob": 0.2, "actual_home_win": True, "market_home_prob": 0.5},
    ]
    result = backtest.simulate_roi(preds, market_vig=0.0, min_edge=0.03)

    assert result["bets_placed"] == 2
    # One win at even odds (+1), one loss (-1) -> flat ROI of 0.
    assert result["flat_bet_roi"] == pytest.approx(0.0)


def test_simulate_roi_rejects_bad_kelly_fraction():
    with pytest.raises(ValueError):
        backtest.simulate_roi([], kelly_fraction=0.0)


# --------------------------------------------------------------------------- #
# Walk-forward fold
# --------------------------------------------------------------------------- #


def test_run_backtest_with_constant_predictor(mem_con):
    calls = []

    def predict_fn(con, game_pk, home_id, away_id, as_of_date):
        calls.append((game_pk, home_id, away_id, as_of_date))
        return FakePrediction(game_pk=game_pk, home_win_prob=0.5)

    result = backtest.run_backtest(mem_con, predict_fn, TRAIN_SEASONS, TEST_SEASON)

    assert isinstance(result, backtest.BacktestResult)
    assert result.n_games == 20 and len(result.predictions) == 20
    assert len(calls) == 20
    assert result.brier_score == pytest.approx(0.25)
    assert result.log_loss == pytest.approx(math.log(2))
    assert result.calibration["0.50-0.60"]["n"] == 20
    assert result.roi_flat_bet == 0.0 and result.roi_kelly == 0.0
    assert result.bets_placed == 0
    assert result.test_season == TEST_SEASON
    assert result.train_seasons == TRAIN_SEASONS

    first = result.predictions[0]
    assert set(first) == {
        "game_pk", "date", "home_id", "away_id", "home_win_prob", "actual_home_win"
    }
    assert first["date"] == "2026-04-01"
    assert first["actual_home_win"] is True
    # as_of_date handed to the model is the game's date as an ISO string
    assert calls[0][3] == "2026-04-01"
    # Games arrive in chronological order
    assert [p["date"] for p in result.predictions] == sorted(
        p["date"] for p in result.predictions
    )


def test_run_backtest_perfect_predictor_scores_zero(mem_con, games):
    outcomes = {g["game_pk"]: g["winning_team_id"] == g["home_team_id"] for g in games}

    def oracle(con, game_pk, home_id, away_id, as_of_date):
        return {"home_win_prob": 1.0 if outcomes[game_pk] else 0.0}

    result = backtest.run_backtest(mem_con, oracle, TRAIN_SEASONS, TEST_SEASON)

    assert result.brier_score == 0.0
    assert result.log_loss == pytest.approx(0.0, abs=1e-9)
    assert [p["actual_home_win"] for p in result.predictions] == [
        outcomes[p["game_pk"]] for p in result.predictions
    ]


def test_run_backtest_skips_non_final_and_other_seasons(mem_con, games):
    # Two games in the test season that are not final...
    pending = [
        {**games[0], "game_pk": 999001, "status": "Scheduled", "home_score": None,
         "away_score": None, "winning_team_id": None},
        {**games[1], "game_pk": 999002, "status": "Postponed"},
    ]
    # ...and one final game from a training season.
    prior = {**games[2], "game_pk": 999003, "season": TEST_SEASON - 1}
    db.upsert(mem_con, "fact_game", pending + [prior])

    seen = []

    def predict_fn(con, game_pk, home_id, away_id, as_of_date):
        seen.append(game_pk)
        return 0.5

    result = backtest.run_backtest(mem_con, predict_fn, TRAIN_SEASONS, TEST_SEASON)

    assert result.n_games == 20
    assert result.n_skipped == 2
    assert not {999001, 999002, 999003} & set(seen)


def test_run_backtest_falls_back_to_fact_game_when_team_rows_missing(mem_con, games):
    # Delete the home team's fact_team_game row for one game; the outcome must
    # still resolve from fact_game.winning_team_id.
    target = games[3]
    mem_con.execute(
        "DELETE FROM fact_team_game WHERE game_pk = ? AND team_id = ?",
        [target["game_pk"], target["home_team_id"]],
    )

    result = backtest.run_backtest(mem_con, lambda *a: 0.5, TRAIN_SEASONS, TEST_SEASON)

    by_pk = {p["game_pk"]: p for p in result.predictions}
    assert result.n_games == 20
    assert by_pk[target["game_pk"]]["actual_home_win"] is (
        target["winning_team_id"] == target["home_team_id"]
    )


def test_run_backtest_empty_test_season(mem_con):
    result = backtest.run_backtest(mem_con, lambda *a: 0.5, [2024], 2025)

    assert isinstance(result, backtest.BacktestResult)
    assert result.predictions == []
    assert math.isnan(result.brier_score)
    assert result.roi_flat_bet == 0.0


def test_run_backtest_rejects_leaky_train_seasons(mem_con):
    with pytest.raises(ValueError):
        backtest.run_backtest(mem_con, lambda *a: 0.5, [2025, 2026], 2026)
    with pytest.raises(ValueError):
        backtest.run_backtest(mem_con, lambda *a: 0.5, [2027], 2026)


def test_run_backtest_rejects_out_of_range_probability(mem_con):
    with pytest.raises(ValueError):
        backtest.run_backtest(mem_con, lambda *a: 1.5, TRAIN_SEASONS, TEST_SEASON)


def test_backtest_module_has_no_model_imports():
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module("mlb_pipeline.backtest"))
    for banned in ("elo", "poisson", "models", "predict", "features"):
        assert f"import {banned}" not in source
        assert f"from mlb_pipeline.{banned}" not in source
        assert f"from mlb_pipeline import {banned}" not in source
        assert f"from .{banned}" not in source


def test_backtest_exports_all_six():
    for name in (
        "BacktestResult",
        "brier_score",
        "log_loss",
        "calibration_buckets",
        "simulate_roi",
        "run_backtest",
    ):
        assert hasattr(backtest, name)
