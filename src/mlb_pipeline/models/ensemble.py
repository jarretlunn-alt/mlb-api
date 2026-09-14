"""Calibrated XGBoost with pregame Elo, Poisson, and engineered inputs.

No network access or warehouse writes occur.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from mlb_pipeline import features as _features
from mlb_pipeline.backtest import run_backtest
from mlb_pipeline.models import elo as _elo
from mlb_pipeline.models import poisson as _poisson
from mlb_pipeline.predict import GamePrediction

FEATURE_NAMES = (
    "elo_home_rating", "elo_away_rating", "poisson_lambda_home",
    "poisson_lambda_away", "home_runs_per_game_15d", "away_runs_allowed_15d",
    "home_park_factor", "is_dome",
    "home_sp_fip", "away_sp_fip",
)
SEASON_ERROR = "Need at least 2 seasons of data to train. Run make ingest-date for more dates."

# League-average starter FIP used when a pitcher's stats are missing.
LEAGUE_AVG_FIP = 3.90

# Long-run MLB home-team win rate; used to damp the Poisson fallback.
LEAGUE_AVG_HOME_WIN = 0.54

# Weight blended toward LEAGUE_AVG_HOME_WIN when the full feature row is
# unavailable (early-season games with insufficient rolling history).
# 0.30 preserves Poisson direction while pulling extremes back ~5 pp.
FALLBACK_BLEND = 0.30


def _complete_seasons(con):
    """Closed calendar seasons with decided final games and no scheduled games.

    The warehouse does not track ingestion coverage, so completeness here cannot
    establish that every MLB game has been downloaded.
    """
    return [r[0] for r in con.execute("""
        SELECT season FROM fact_game WHERE season < ? GROUP BY season
        HAVING count(*) FILTER (WHERE status = 'Final' AND home_score IS NOT NULL
            AND away_score IS NOT NULL AND home_score <> away_score) > 0
        AND count(*) FILTER (WHERE status IS NULL OR NOT (
            status LIKE 'Final%' OR status LIKE 'Completed Early%' OR
            status LIKE 'Game Over%' OR status LIKE 'Cancelled%' OR
            status LIKE 'Postponed%')) = 0
        ORDER BY season
    """, [date.today().year]).fetchall()]


def _games(con, seasons):
    if not seasons:
        return []
    return con.execute(f"""
        SELECT game_pk, official_date, home_team_id, away_team_id,
               CAST(home_score > away_score AS INTEGER)
        FROM fact_game WHERE season IN ({','.join('?' for _ in seasons)})
          AND (status = 'Final' OR status LIKE 'Completed Early%') AND official_date IS NOT NULL
          AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL
          AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_score <> away_score
        ORDER BY official_date, game_pk
    """, list(seasons)).fetchall()


def _ratings(con, as_of_date):
    cutoff = date.fromisoformat(str(as_of_date)) - timedelta(days=1)
    return _elo.build_ratings_from_history(con, cutoff.isoformat())


def _poisson_prediction(con, game_pk, home_id, away_id, as_of_date):
    return _poisson.predict(con, game_pk, home_id, away_id, as_of_date)


def _feature_row(con, game_pk, home_id, away_id, as_of_date, ratings=None):
    raw = _features.build_game_feature_row(con, game_pk, as_of_date)
    if raw is None:
        return None
    ratings = _ratings(con, as_of_date) if ratings is None else ratings
    pois = _poisson_prediction(con, game_pk, home_id, away_id, as_of_date)
    # Explicit allowlist excludes score, win, total_runs, and identifier columns.
    # sp_fip: use LEAGUE_AVG_FIP when pitcher stats are unavailable (opener game, etc.)
    home_fip = raw.get("home_sp_fip_last_n")
    away_fip = raw.get("away_sp_fip_last_n")
    values = [ratings.get(home_id, 1500.0), ratings.get(away_id, 1500.0),
              pois.features["lam_home"], pois.features["lam_away"],
              raw.get("home_runs_per_game"), raw.get("away_runs_allowed_per_game"),
              raw.get("park_run_factor", 1.0), raw.get("is_dome", 0.0),
              home_fip if home_fip is not None else LEAGUE_AVG_FIP,
              away_fip if away_fip is not None else LEAGUE_AVG_FIP]
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
    """Return calibrated home probability; missing inputs use blended Poisson fallback."""
    row = _feature_row(con, game_pk, home_id, away_id, as_of_date)
    if row is None:
        base = _poisson_prediction(con, game_pk, home_id, away_id, as_of_date)
        # Blend raw Poisson toward the long-run home-win rate to damp early-season
        # overconfidence when rolling history is insufficient for a full feature row.
        raw_p = base.home_win_prob
        blended_p = raw_p * (1.0 - FALLBACK_BLEND) + LEAGUE_AVG_HOME_WIN * FALLBACK_BLEND
        return GamePrediction(game_pk, "ensemble", blended_p, 1.0 - blended_p,
                              base.pred_total, {**base.features,
                                                "ensemble_fallback": "missing features",
                                                "fallback_blend": FALLBACK_BLEND,
                                                "raw_poisson_home": raw_p})
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
    home = _elo.predict(ratings, home_id, away_id)["home_win_prob"]
    return GamePrediction(game_pk, "elo", home, 1.0 - home, None, {})


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
        for name, callback in (("ensemble", partial(predict, model)),
                               ("elo", _elo_prediction), ("poisson", _poisson_prediction)):
            result = run_backtest(con, callback, seasons[:-1], seasons[-1])
            print(f"{name}: test season {seasons[-1]}, Brier={result.brier_score:.6f}, log-loss={result.log_loss:.6f}")
            print("Calibration table:", result.calibration)
            print(f"ROI simulation (synthetic market): flat={result.roi_flat_bet:.4%}, quarter-Kelly={result.roi_kelly:.4%}")


if __name__ == "__main__":
    main()
