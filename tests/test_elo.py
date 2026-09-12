import json

import pytest

from mlb_pipeline import db, ingest
from mlb_pipeline.models import elo
from mlb_pipeline.predict import GamePrediction, load_predictions, save_prediction

HOME_ID = 136  # Seattle Mariners (won 5-3 in fixture 700001)
AWAY_ID = 133  # Athletics


def test_expected_score_is_symmetric():
    for a, b in [(1500, 1500), (1600, 1450), (1400, 1700.5)]:
        assert elo.expected_score(a, b) + elo.expected_score(b, a) == pytest.approx(1.0)
    assert elo.expected_score(1500, 1500) == pytest.approx(0.5)
    assert elo.expected_score(1600, 1500) > 0.5


def test_update_ratings_moves_winner_up_and_loser_down():
    start = {HOME_ID: 1500.0, AWAY_ID: 1500.0}

    after_home_win = elo.update_ratings(start, HOME_ID, AWAY_ID, home_won=True)
    assert after_home_win[HOME_ID] > 1500.0
    assert after_home_win[AWAY_ID] < 1500.0

    after_away_win = elo.update_ratings(start, HOME_ID, AWAY_ID, home_won=False)
    assert after_away_win[HOME_ID] < 1500.0
    assert after_away_win[AWAY_ID] > 1500.0

    # Zero-sum and non-mutating
    assert sum(after_home_win.values()) == pytest.approx(sum(start.values()))
    assert start == {HOME_ID: 1500.0, AWAY_ID: 1500.0}
    # Home favourite: an away upset moves ratings more than an expected home win
    assert abs(after_away_win[HOME_ID] - 1500.0) > abs(after_home_win[HOME_ID] - 1500.0)


def test_update_ratings_defaults_unknown_teams():
    new = elo.update_ratings({}, 1, 2, home_won=True)
    assert set(new) == {1, 2}
    assert new[1] > elo.DEFAULT_RATING > new[2]


def test_build_ratings_from_fixture_game(con, feed_live, boxscore):
    ingest.load_game(con, feed_live, boxscore)

    ratings = elo.build_ratings_from_history(con, "2026-07-01")

    assert ratings[HOME_ID] > elo.DEFAULT_RATING
    assert ratings[AWAY_ID] < elo.DEFAULT_RATING
    # through_date is inclusive; a day earlier sees no games
    assert elo.build_ratings_from_history(con, "2026-06-30") == {}


def _game(game_pk, date, season, home_id, away_id, home_score, away_score, status="Final"):
    winner = None
    if home_score is not None and away_score is not None and home_score != away_score:
        winner = home_id if home_score > away_score else away_id
    return {
        "game_pk": game_pk,
        "official_date": date,
        "season": season,
        "game_type": "R",
        "status": status,
        "venue": None,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_score": home_score,
        "away_score": away_score,
        "winning_team_id": winner,
    }


def test_build_ratings_skips_non_final_games(con):
    db.upsert(
        con,
        "fact_game",
        [
            _game(1, "2026-07-01", 2026, 1, 2, None, None, status="Postponed"),
            _game(2, "2026-07-01", 2026, 1, 2, 2, 1, status="In Progress"),
        ],
    )
    assert elo.build_ratings_from_history(con, "2026-07-01") == {}


def test_build_ratings_regresses_at_season_change(con):
    db.upsert(
        con,
        "fact_game",
        [
            _game(1, "2025-09-01", 2025, 1, 2, 4, 1),
            _game(2, "2025-09-02", 2025, 1, 2, 6, 2),
            _game(3, "2026-04-01", 2026, 1, 2, 3, 2),
        ],
    )
    end_2025 = elo.build_ratings_from_history(con, "2025-12-31")
    regressed = elo.regress_to_mean(end_2025)
    expected_2026 = elo.update_ratings(regressed, 1, 2, home_won=True)

    ratings_2026 = elo.build_ratings_from_history(con, "2026-04-01")
    assert ratings_2026 == pytest.approx(expected_2026)

    # Regression actually pulled toward the mean between seasons
    assert abs(regressed[1] - elo.DEFAULT_RATING) == pytest.approx(
        (1 - elo.SEASON_REGRESS) * abs(end_2025[1] - elo.DEFAULT_RATING)
    )


def test_pythagorean_win_pct():
    assert 0.55 < elo.pythagorean_win_pct(800, 600) < 0.65
    assert elo.pythagorean_win_pct(700, 700) == pytest.approx(0.5)
    assert elo.pythagorean_win_pct(0, 0) == 0.5
    assert elo.pythagorean_win_pct(500, 800) < 0.5
    with pytest.raises(ValueError):
        elo.pythagorean_win_pct(-1, 10)


def test_predict_probs_sum_to_one():
    ratings = {HOME_ID: 1550.0, AWAY_ID: 1480.0}
    out = elo.predict(ratings, HOME_ID, AWAY_ID)

    assert out["model"] == "elo"
    assert out["home_win_prob"] + out["away_win_prob"] == pytest.approx(1.0)
    assert 0.0 < out["home_win_prob"] < 1.0
    assert out["home_win_prob"] > 0.5

    # Home advantage tips an even matchup toward the home side
    even = elo.predict({}, HOME_ID, AWAY_ID)
    assert even["home_win_prob"] > 0.5
    neutral = elo.predict({}, HOME_ID, AWAY_ID, home_advantage=0.0)
    assert neutral["home_win_prob"] == pytest.approx(0.5)


def test_predict_game_returns_game_prediction():
    ratings = {HOME_ID: 1550.0, AWAY_ID: 1480.0}
    pred = elo.predict_game(ratings, 700001, HOME_ID, AWAY_ID)

    assert isinstance(pred, GamePrediction)
    assert pred.model_name == "elo"
    assert pred.game_pk == 700001
    assert pred.pred_total is None
    assert pred.home_win_prob + pred.away_win_prob == pytest.approx(1.0)
    assert pred.features["home_rating"] == 1550.0
    assert pred.features["away_rating"] == 1480.0


def test_save_prediction_is_idempotent(con):
    pred = GamePrediction(
        game_pk=700001,
        model_name="elo",
        home_win_prob=0.6,
        away_win_prob=0.4,
        pred_total=None,
        features={"home_rating": 1550.0, "away_rating": 1480.0},
    )
    save_prediction(con, pred)
    save_prediction(con, GamePrediction(700001, "elo", 0.62, 0.38, 8.5, {"k": 1}))

    rows = con.execute(
        "SELECT prediction_id, game_pk, model_name, home_win_prob, away_win_prob, pred_total, "
        "CAST(features_json AS VARCHAR), predicted_at FROM fact_prediction"
    ).fetchall()
    assert len(rows) == 1
    pid, game_pk, model_name, home, away, total, features_json, predicted_at = rows[0]
    assert pid == "elo_700001"
    assert (game_pk, model_name) == (700001, "elo")
    assert (home, away, total) == (pytest.approx(0.62), pytest.approx(0.38), 8.5)
    assert json.loads(features_json) == {"k": 1}
    assert predicted_at is not None

    loaded = load_predictions(con, game_pk=700001)
    assert loaded == [GamePrediction(700001, "elo", 0.62, 0.38, 8.5, {"k": 1})]
    assert load_predictions(con, model_name="poisson") == []
