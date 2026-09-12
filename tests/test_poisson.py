import math

import duckdb
import pytest

from conftest import load_fixture
from mlb_pipeline import db, ingest
from mlb_pipeline.models import poisson

# Fixture game 700001: Mariners (136, home) 5 - Athletics (133, away) 3 on 2026-07-01.
GAME_PK = 700001
HOME_ID = 136
AWAY_ID = 133
GAME_DATE = "2026-07-01"
NEXT_DAY = "2026-07-02"


@pytest.fixture
def mem_con():
    """In-memory DuckDB populated from the fixture game. No network."""
    con = duckdb.connect(":memory:")
    db.init_schema(con)
    ingest.load_game(con, load_fixture("feed_live_700001.json"), load_fixture("boxscore_700001.json"))
    yield con
    con.close()


# ---------------------------------------------------------------------------
# poisson_pmf
# ---------------------------------------------------------------------------


def test_pmf_zero_matches_closed_form():
    assert poisson.poisson_pmf(0, 4.5) == pytest.approx(math.exp(-4.5))


def test_pmf_general_matches_closed_form():
    lam, k = 3.7, 4
    expected = math.exp(-lam) * lam**k / math.factorial(k)
    assert poisson.poisson_pmf(k, lam) == pytest.approx(expected)


def test_pmf_negative_k_is_zero():
    assert poisson.poisson_pmf(-1, 4.5) == 0.0


def test_pmf_sums_to_one():
    assert sum(poisson.poisson_pmf(k, 4.5) for k in range(60)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# win_prob_from_lambdas
# ---------------------------------------------------------------------------


def test_stronger_home_lambda_favours_home():
    probs = poisson.win_prob_from_lambdas(5.0, 3.0)
    assert probs["home_win_prob"] > 0.5
    assert probs["home_win_prob"] > probs["away_win_prob"]


def test_equal_lambdas_are_symmetric():
    for lam in (2.0, 4.5, 7.0):
        probs = poisson.win_prob_from_lambdas(lam, lam)
        assert probs["home_win_prob"] == pytest.approx(probs["away_win_prob"])
        assert probs["tie_prob"] > 0


def test_swapping_lambdas_swaps_sides():
    a = poisson.win_prob_from_lambdas(5.0, 3.0)
    b = poisson.win_prob_from_lambdas(3.0, 5.0)
    assert a["home_win_prob"] == pytest.approx(b["away_win_prob"])
    assert a["away_win_prob"] == pytest.approx(b["home_win_prob"])
    assert a["tie_prob"] == pytest.approx(b["tie_prob"])


def test_win_probs_sum_to_one():
    for lam_home, lam_away in ((4.5, 4.5), (5.0, 3.0), (2.0, 7.5)):
        probs = poisson.win_prob_from_lambdas(lam_home, lam_away)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert set(probs) == {"home_win_prob", "away_win_prob", "tie_prob"}
        assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_win_probs_match_brute_force():
    lam_home, lam_away = 4.2, 3.6
    probs = poisson.win_prob_from_lambdas(lam_home, lam_away, max_runs=40)
    home = sum(
        poisson.poisson_pmf(h, lam_home) * poisson.poisson_pmf(a, lam_away)
        for h in range(41)
        for a in range(41)
        if h > a
    )
    assert probs["home_win_prob"] == pytest.approx(home, abs=1e-9)


# ---------------------------------------------------------------------------
# over_prob
# ---------------------------------------------------------------------------


def test_over_prob_even_matchup_near_coin_flip():
    p = poisson.over_prob(4.5, 4.5, 8.5)
    assert 0.40 < p < 0.60


def test_over_prob_half_line_complements_under():
    lam_home, lam_away, line = 4.5, 4.5, 8.5
    p_over = poisson.over_prob(lam_home, lam_away, line)
    # Total ~ Poisson(9); P(T <= 8) is the under.
    p_under = sum(poisson.poisson_pmf(k, lam_home + lam_away) for k in range(9))
    assert p_over + p_under == pytest.approx(1.0)


def test_over_prob_monotone_in_line_and_lambda():
    assert poisson.over_prob(4.5, 4.5, 7.5) > poisson.over_prob(4.5, 4.5, 8.5) > poisson.over_prob(4.5, 4.5, 9.5)
    assert poisson.over_prob(5.5, 5.5, 8.5) > poisson.over_prob(4.5, 4.5, 8.5)


def test_over_prob_whole_number_line_excludes_push():
    # P(T > 8) == P(T > 8.5) for an integer-valued total.
    assert poisson.over_prob(4.5, 4.5, 8) == pytest.approx(poisson.over_prob(4.5, 4.5, 8.5))


# ---------------------------------------------------------------------------
# estimate_lambda
# ---------------------------------------------------------------------------


def test_estimate_lambda_in_reasonable_range():
    lam = poisson.estimate_lambda(5.0, 4.0, 4.5)
    assert 3.5 < lam < 6.0
    assert lam == pytest.approx(5.0 * 4.0 / 4.5)


def test_estimate_lambda_league_average_is_fixed_point():
    assert poisson.estimate_lambda(4.5, 4.5, 4.5) == pytest.approx(4.5)


def test_estimate_lambda_applies_park_and_weather():
    base = poisson.estimate_lambda(5.0, 4.0, 4.5)
    assert poisson.estimate_lambda(5.0, 4.0, 4.5, park_factor=1.1) == pytest.approx(base * 1.1)
    assert poisson.estimate_lambda(5.0, 4.0, 4.5, weather_adjustment=0.9) == pytest.approx(base * 0.9)


def test_estimate_lambda_rejects_bad_league_avg_and_floors_result():
    with pytest.raises(ValueError):
        poisson.estimate_lambda(5.0, 4.0, 0.0)
    assert poisson.estimate_lambda(0.0, 4.0, 4.5) == poisson.MIN_LAMBDA


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_game_prediction_with_total(mem_con):
    pred = poisson.predict(mem_con, GAME_PK, HOME_ID, AWAY_ID, NEXT_DAY)

    assert isinstance(pred, poisson.GamePrediction)
    assert pred.game_pk == GAME_PK
    assert pred.model_name == "poisson"
    assert pred.pred_total is not None
    assert pred.pred_total > 0
    assert pred.pred_total == pytest.approx(pred.features["lam_home"] + pred.features["lam_away"])
    assert pred.home_win_prob + pred.away_win_prob == pytest.approx(1.0)
    assert 0.0 < pred.home_win_prob < 1.0


def test_predict_features_are_json_serialisable(mem_con):
    import json

    pred = poisson.predict(mem_con, GAME_PK, HOME_ID, AWAY_ID, NEXT_DAY, total_line=8.5)
    encoded = json.loads(json.dumps(pred.features))
    assert encoded["total_line"] == 8.5
    assert encoded["over_prob"] + encoded["under_prob"] + encoded["push_prob"] == pytest.approx(1.0)


def test_predict_falls_back_to_league_average_for_thin_history(mem_con):
    # One fixture game is below MIN_GAMES, and a team with no games at all
    # has no history, so both sides project at the league average.
    pred = poisson.predict(mem_con, GAME_PK, HOME_ID, 999, NEXT_DAY)

    assert pred.features["home_fallback"] is True
    assert pred.features["away_fallback"] is True
    assert pred.features["home_runs_per_game"] == poisson.LEAGUE_AVG_RUNS
    assert pred.features["away_runs_per_game"] == poisson.LEAGUE_AVG_RUNS
    park = pred.features["park_run_factor"]
    assert pred.pred_total == pytest.approx(2 * poisson.LEAGUE_AVG_RUNS * park)
    assert pred.home_win_prob == pytest.approx(0.5)


def test_predict_is_deterministic(mem_con):
    a = poisson.predict(mem_con, GAME_PK, HOME_ID, AWAY_ID, NEXT_DAY)
    b = poisson.predict(mem_con, GAME_PK, HOME_ID, AWAY_ID, NEXT_DAY)
    assert a == b


def test_predict_uses_rolling_features_when_history_is_sufficient(mem_con):
    """Requires Task 01a's features module; skipped until it is merged."""
    pytest.importorskip("mlb_pipeline.features")
    import datetime as dt

    # Give the home team MIN_GAMES high-scoring games in the window and the
    # away team the same number of low-scoring ones.
    games, team_games = [], []
    for i in range(poisson.MIN_GAMES):
        pk = 800000 + i
        date = dt.date(2026, 6, 20) + dt.timedelta(days=i)
        games.append(
            {
                "game_pk": pk,
                "official_date": date,
                "season": 2026,
                "game_type": "R",
                "status": "Final",
                "venue": "Test Park",
                "home_team_id": HOME_ID,
                "away_team_id": AWAY_ID,
                "home_score": 8,
                "away_score": 2,
                "winning_team_id": HOME_ID,
            }
        )
        team_games.append(
            {"game_pk": pk, "team_id": HOME_ID, "opponent_team_id": AWAY_ID, "is_home": True,
             "runs_scored": 8, "runs_allowed": 2, "hits": 10, "errors": 0, "win": True}
        )
        team_games.append(
            {"game_pk": pk, "team_id": AWAY_ID, "opponent_team_id": HOME_ID, "is_home": False,
             "runs_scored": 2, "runs_allowed": 8, "hits": 5, "errors": 0, "win": False}
        )
    db.upsert(mem_con, "fact_game", games)
    db.upsert(mem_con, "fact_team_game", team_games)

    pred = poisson.predict(mem_con, GAME_PK, HOME_ID, AWAY_ID, GAME_DATE)

    assert pred.features["home_fallback"] is False
    assert pred.features["away_fallback"] is False
    assert pred.features["lam_home"] > pred.features["lam_away"]
    assert pred.home_win_prob > 0.5
