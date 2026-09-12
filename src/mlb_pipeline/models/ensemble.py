"""Calibrated XGBoost with pregame Elo, Poisson, and engineered inputs.

Temporary neutral fallbacks support the independently dispatched prerequisite
branches. Missing modules are reported explicitly; errors inside installed
modules are never swallowed. No network access or warehouse writes occur.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss


def _optional(name):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        warnings.warn(f"{name} unavailable; using neutral ensemble fallback", RuntimeWarning)
        return None


_prediction = _optional("mlb_pipeline.predict")
if _prediction is not None:
    GamePrediction = _prediction.GamePrediction
else:
    @dataclass
    class GamePrediction:
        game_pk: int
        model_name: str
        home_win_prob: float
        away_win_prob: float
        pred_total: float | None
        features: dict

_features = _optional("mlb_pipeline.features")
_elo = _optional("mlb_pipeline.models.elo")
_poisson = _optional("mlb_pipeline.models.poisson")

FEATURE_NAMES = (
    "elo_home_rating", "elo_away_rating", "poisson_lambda_home",
    "poisson_lambda_away", "home_runs_per_game_15d", "away_runs_allowed_15d",
    "home_park_factor", "is_dome",
)
SEASON_ERROR = "Need at least 2 seasons of data to train. Run make ingest-date for more dates."


def _complete_seasons(con):
    """Closed calendar seasons with decided final games and no scheduled games.

    The warehouse does not track ingestion coverage, so completeness here cannot
    establish that every MLB game has been downloaded.
    """
    return [r[0] for r in con.execute("""
        SELECT season FROM fact_game WHERE season < ? GROUP BY season
        HAVING count(*) FILTER (WHERE status = 'Final' AND home_score IS NOT NULL
            AND away_score IS NOT NULL AND home_score <> away_score) > 0
        AND count(*) FILTER (WHERE status IS NULL OR status NOT IN
            ('Final', 'Completed Early', 'Game Over', 'Cancelled', 'Postponed')) = 0
        ORDER BY season
    """, [date.today().year]).fetchall()]


def _games(con, seasons):
    if not seasons:
        return []
    return con.execute(f"""
        SELECT game_pk, official_date, home_team_id, away_team_id,
               CAST(home_score > away_score AS INTEGER)
        FROM fact_game WHERE season IN ({','.join('?' for _ in seasons)})
          AND status = 'Final' AND official_date IS NOT NULL
          AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL
          AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_score <> away_score
        ORDER BY official_date, game_pk
    """, list(seasons)).fetchall()


def _ratings(con, as_of_date):
    cutoff = date.fromisoformat(str(as_of_date)) - timedelta(days=1)
    return _elo.build_ratings_from_history(con, cutoff.isoformat()) if _elo else {}


def _poisson_prediction(con, game_pk, home_id, away_id, as_of_date):
    if _poisson:
        return _poisson.predict(con, game_pk, home_id, away_id, as_of_date)
    return GamePrediction(game_pk, "poisson", 0.5, 0.5, 9.0,
                          {"lam_home": 4.5, "lam_away": 4.5, "fallback": True})


def _feature_row(con, game_pk, home_id, away_id, as_of_date, ratings=None):
    raw = (_features.build_game_feature_row(con, game_pk, as_of_date)
           if _features else {"home_runs_per_game": 0.0,
                              "away_runs_allowed_per_game": 0.0,
                              "park_run_factor": 1.0, "is_dome": 0.0})
    if raw is None:
        return None
    ratings = _ratings(con, as_of_date) if ratings is None else ratings
    pois = _poisson_prediction(con, game_pk, home_id, away_id, as_of_date)
    # Explicit allowlist excludes score, win, total_runs, and identifier columns.
    values = [ratings.get(home_id, 1500.0), ratings.get(away_id, 1500.0),
              pois.features["lam_home"], pois.features["lam_away"],
              raw.get("home_runs_per_game"), raw.get("away_runs_allowed_per_game"),
              raw.get("park_run_factor", 1.0), raw.get("is_dome", 0.0)]
    if any(v is None for v in values):
        return None
    row = np.asarray(values, dtype=float)
    return row if np.isfinite(row).all() else None


def build_training_set(con, seasons: list[int]) -> tuple:
    """Return ordered float X and binary y, skipping unavailable features."""
    rows, labels = [], []
    rating_cache = {}
    for game_pk, day, home_id, away_id, won in _games(con, seasons):
        as_of = day.isoformat()
        if as_of not in rating_cache:
            rating_cache[as_of] = _ratings(con, as_of)
        row = _feature_row(con, game_pk, home_id, away_id, as_of, rating_cache[as_of])
        if row is not None:
            rows.append(row)
            labels.append(won)
    return np.asarray(rows, dtype=float).reshape(-1, len(FEATURE_NAMES)), np.asarray(labels, dtype=int)


def train(con, seasons: list[int]) -> CalibratedClassifierCV:
    """Fit on earliest 70%, early-stop on next 10%, calibrate on latest 20%.

    Calibration rows are unseen by both tree fitting and early stopping. The
    warehouse must contain two completed seasons; a fold may train on one.
    """
    available = _complete_seasons(con)
    if len(available) < 2:
        raise ValueError(SEASON_ERROR)
    if not seasons or not set(seasons).issubset(available):
        raise ValueError("Training seasons must be completed seasons present in the warehouse.")
    X, y = build_training_set(con, seasons)
    n = len(y)
    n_valid, n_cal = max(2, int(np.ceil(n * 0.1))), max(2, int(np.ceil(n * 0.2)))
    fit_end, valid_end = n - n_valid - n_cal, n - n_cal
    if fit_end < 2 or any(len(np.unique(part)) != 2 for part in
                          (y[:fit_end], y[fit_end:valid_end], y[valid_end:])):
        raise ValueError("Need more usable games with both outcomes in fit, validation, and calibration partitions.")
    estimator = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  objective="binary:logistic", eval_metric="logloss",
                                  early_stopping_rounds=20, random_state=42, n_jobs=1)
    estimator.fit(X[:fit_end], y[:fit_end],
                  eval_set=[(X[fit_end:valid_end], y[fit_end:valid_end])], verbose=False)
    # FrozenEstimator.fit is a no-op. A single explicit split predicts every
    # held-out row without imposing a minimum five-fold class count.
    calibration_indices = np.arange(n_cal)
    model = CalibratedClassifierCV(
        FrozenEstimator(estimator), method="isotonic", ensemble=False,
        cv=[(calibration_indices, calibration_indices)],
    )
    model.fit(X[valid_end:], y[valid_end:])
    model.ensemble_feature_names_ = FEATURE_NAMES
    model.ensemble_estimator_ = estimator
    return model


def predict(model, con, game_pk: int, home_id: int, away_id: int,
            as_of_date: str) -> GamePrediction:
    """Return calibrated home probability; missing inputs use Poisson fallback."""
    row = _feature_row(con, game_pk, home_id, away_id, as_of_date)
    if row is None:
        base = _poisson_prediction(con, game_pk, home_id, away_id, as_of_date)
        return GamePrediction(game_pk, "ensemble", base.home_win_prob, base.away_win_prob,
                              base.pred_total, {**base.features, "ensemble_fallback": "missing features"})
    home = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    return GamePrediction(game_pk, "ensemble", home, 1.0 - home, float(row[2] + row[3]),
                          dict(zip(FEATURE_NAMES, row.tolist())))


def feature_importances(model: CalibratedClassifierCV) -> dict[str, float]:
    """Return raw gain importance, including zero for unused features."""
    scores = model.ensemble_estimator_.get_booster().get_score(importance_type="gain")
    return {name: float(scores.get(name, scores.get(f"f{i}", 0.0)))
            for i, name in enumerate(model.ensemble_feature_names_)}


def _elo_prediction(con, game_pk, home_id, away_id, as_of_date):
    ratings = _ratings(con, as_of_date)
    home = (_elo.predict(ratings, home_id, away_id)["home_win_prob"] if _elo else 0.5)
    return GamePrediction(game_pk, "elo", home, 1.0 - home, None, {})


def _fallback_backtest(con, predict_fn, train_seasons, test_season):
    """Temporary synthetic-market scorer used only until Task 02c is merged."""
    probabilities, outcomes = [], []
    for pk, day, home, away, won in _games(con, [test_season]):
        probabilities.append(predict_fn(con, pk, home, away, day.isoformat()).home_win_prob)
        outcomes.append(won)
    if not outcomes:
        raise ValueError("No decided final games in the test season.")
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    calibration = {}
    for i in range(10):
        mask = np.minimum((p * 10).astype(int), 9) == i
        calibration[f"{i / 10:.2f}-{(i + 1) / 10:.2f}"] = {
            "n": int(mask.sum()), "predicted_prob": float(p[mask].mean()) if mask.any() else None,
            "actual_win_rate": float(y[mask].mean()) if mask.any() else None}
    # Fixed synthetic 50/50 market with proportional 4.5% vig.
    side_prob = np.maximum(p, 1 - p)
    bets = side_prob > 0.5 * 1.045 + 0.03
    payoff = np.where((p >= 0.5) == y, 1 / (0.5 * 1.045) - 1, -1)
    stakes = np.maximum(0, (side_prob - 0.5 * 1.045) / (1 - 0.5 * 1.045)) * 0.25
    return SimpleNamespace(brier_score=float(brier_score_loss(y, p)),
                           log_loss=float(log_loss(y, p, labels=[0, 1])), calibration=calibration,
                           roi_flat_bet=float(payoff[bets].mean()) if bets.any() else 0.0,
                           roi_kelly=float(np.average(payoff[bets], weights=stakes[bets])) if bets.any() else 0.0)


def main():
    """Evaluate the latest completed season against all preceding seasons."""
    import duckdb

    path = Path("data/warehouse.duckdb")
    if not path.exists():
        print(f"Warehouse not found: {path}. Run make ingest-date first.")
        return
    with duckdb.connect(str(path), read_only=True) as con:
        seasons = _complete_seasons(con)
        if len(seasons) < 2:
            print(SEASON_ERROR)
            return
        model = train(con, seasons[:-1])
        backtest = _optional("mlb_pipeline.backtest")
        run = backtest.run_backtest if backtest else _fallback_backtest
        for name, callback in (("ensemble", partial(predict, model)),
                               ("elo", _elo_prediction), ("poisson", _poisson_prediction)):
            result = run(con, callback, seasons[:-1], seasons[-1])
            print(f"{name}: test season {seasons[-1]}, Brier={result.brier_score:.6f}, log-loss={result.log_loss:.6f}")
            print("Calibration table:", result.calibration)
            print(f"ROI simulation (synthetic market): flat={result.roi_flat_bet:.4%}, quarter-Kelly={result.roi_kelly:.4%}")


if __name__ == "__main__":
    main()
