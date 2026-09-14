from mlb_pipeline import features
from mlb_pipeline.models import ensemble


def test_feature_names_iter2():
    names = ensemble.FEATURE_NAMES
    assert names[10:12] == ("home_runs_allowed_per_game", "away_runs_per_game")
    assert set(names[10:12]).issubset(features.feature_row_keys())
