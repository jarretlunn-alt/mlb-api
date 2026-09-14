"""Small offline ensemble integration tests with synthetic warehouse games."""
from datetime import date, timedelta
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from mlb_pipeline import db
from mlb_pipeline.models import ensemble


@pytest.fixture
def warehouse(monkeypatch):
    con = duckdb.connect(":memory:")
    db.init_schema(con)
    seasons = [date.today().year - 3, date.today().year - 2]
    for j, season in enumerate(seasons):
        con.executemany("""INSERT INTO fact_game
            (game_pk, official_date, season, status, home_team_id, away_team_id, home_score, away_score)
            VALUES (?, ?, ?, 'Final', 1, 2, ?, ?)""", [
            (j * 25 + i, date(season, 6, 1) + timedelta(days=i), season,
             5 if i % 2 else 2, 3) for i in range(25)])
    monkeypatch.setattr(ensemble, "_features", SimpleNamespace(build_game_feature_row=
        lambda con, pk, day: {"home_runs_per_game": 4 + (pk % 3),
                             "away_runs_allowed_per_game": 3 + (pk % 2),
                             "park_run_factor": 1.02, "is_dome": True,
                             "home_sp_fip_last_n": 3.5, "away_sp_fip_last_n": 4.0,
                             "home_runs_allowed_per_game": 4.2, "away_runs_per_game": 3.8,
                             "home_bullpen_fip": 3.9, "away_bullpen_fip": 4.1,
                             "home_sp_k_per_9": 8.5, "away_sp_k_per_9": 7.9,
                             "home_score": 99, "home_win": True}))
    monkeypatch.setattr(ensemble, "_elo", SimpleNamespace(
        build_ratings_from_history=lambda con, day: {},
        predict=lambda ratings, home_id, away_id: {"home_win_prob": 0.5, "away_win_prob": 0.5}))
    monkeypatch.setattr(ensemble, "_poisson", SimpleNamespace(predict=lambda *a:
        ensemble.GamePrediction(a[1], "poisson", 0.5, 0.5, 9.0,
                                {"lam_home": 4.5, "lam_away": 4.5})))
    yield con, seasons
    con.close()


def test_training_shape_and_skips(warehouse, monkeypatch):
    con, seasons = warehouse
    X, y = ensemble.build_training_set(con, seasons)
    assert X.shape == (50, len(ensemble.FEATURE_NAMES))
    assert y.shape == (50,)
    assert set(y) == {0, 1}
    assert np.isfinite(X).all()
    assert 99 not in X  # Outcomes are never included in X.
    monkeypatch.setattr(ensemble, "_features", SimpleNamespace(build_game_feature_row=lambda *a: None))
    X, y = ensemble.build_training_set(con, seasons)
    assert X.shape == (0, len(ensemble.FEATURE_NAMES))
    assert y.shape == (0,)


def test_train_predict_importances_and_disjoint_splits(warehouse, monkeypatch):
    con, seasons = warehouse
    seen = {}
    original_fit = ensemble.xgb.XGBClassifier.fit

    def fit(self, X, y, **kwargs):
        seen["fit"] = len(y)
        seen["validation"] = len(kwargs["eval_set"][0][1])
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(ensemble.xgb.XGBClassifier, "fit", fit)
    model = ensemble.train(con, seasons)
    assert isinstance(model, CalibratedClassifierCV)
    assert model.method == "isotonic"
    assert seen == {"fit": 35, "validation": 5}  # remaining 10 are calibration only
    pred = ensemble.predict(model, con, 1, 1, 2, f"{seasons[0]}-06-02")
    assert isinstance(pred, ensemble.GamePrediction)
    assert 0 <= pred.home_win_prob <= 1
    assert pred.home_win_prob + pred.away_win_prob == pytest.approx(1)
    importance = ensemble.feature_importances(model)
    assert set(importance) == set(ensemble.FEATURE_NAMES)
    assert all(v >= 0 for v in importance.values())


def test_insufficient_seasons(warehouse):
    con, seasons = warehouse
    con.execute("DELETE FROM fact_game WHERE season = ?", [seasons[1]])
    with pytest.raises(ValueError, match="Need at least 2 seasons"):
        ensemble.train(con, seasons[:1])


def test_single_class_is_clear_error(warehouse):
    con, seasons = warehouse
    con.execute("UPDATE fact_game SET home_score = 5")
    with pytest.raises(ValueError, match="both outcomes"):
        ensemble.train(con, seasons)


def test_elo_uses_previous_day_and_real_poisson_interface(warehouse, monkeypatch):
    con, seasons = warehouse
    seen = []
    monkeypatch.setattr(ensemble, "_elo", SimpleNamespace(build_ratings_from_history=
        lambda con, day: seen.append(day) or {1: 1600, 2: 1400}))
    monkeypatch.setattr(ensemble, "_poisson", SimpleNamespace(predict=lambda *a:
        ensemble.GamePrediction(a[1], "poisson", 0.6, 0.4, 8,
                                {"lam_home": 5, "lam_away": 3})))
    X, _ = ensemble.build_training_set(con, seasons[:1])
    assert seen[0] == f"{seasons[0]}-05-31"
    np.testing.assert_array_equal(X[0, :4], [1600, 1400, 5, 3])


def test_missing_features_prediction_is_explicit_fallback(warehouse, monkeypatch):
    con, seasons = warehouse
    monkeypatch.setattr(ensemble, "_features", SimpleNamespace(build_game_feature_row=lambda *a: None))
    # Poisson mock returns 0.5; blended toward _FALLBACK_HOME_PRIOR
    pred = ensemble.predict(None, con, 1, 1, 2, f"{seasons[0]}-06-02")
    expected = (1 - ensemble._FALLBACK_BLEND_WEIGHT) * 0.5 + ensemble._FALLBACK_BLEND_WEIGHT * ensemble._FALLBACK_HOME_PRIOR
    assert pred.home_win_prob == pytest.approx(expected)
    assert pred.home_win_prob + pred.away_win_prob == pytest.approx(1)
    assert pred.features["ensemble_fallback"] == "missing features"


def test_main_runs_three_comparisons(warehouse, monkeypatch, tmp_path, capsys):
    con, seasons = warehouse
    data = tmp_path / "data"
    data.mkdir()
    with duckdb.connect(str(data / "warehouse.duckdb")) as target:
        db.init_schema(target)
        cursor = con.execute("SELECT * FROM fact_game")
        columns = [d[0] for d in cursor.description]
        placeholders = ", ".join(["?"] * len(columns))
        target.executemany(f"INSERT INTO fact_game ({', '.join(columns)}) VALUES ({placeholders})",
                           cursor.fetchall())
    monkeypatch.chdir(tmp_path)
    ensemble.main()
    output = capsys.readouterr().out
    for name in ("ensemble", "elo", "poisson"):
        assert f"{name}: test season {seasons[-1]}" in output
    assert output.count("ROI simulation") == 3
