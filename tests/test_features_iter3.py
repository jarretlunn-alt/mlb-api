from mlb_pipeline import features
from mlb_pipeline.models import ensemble


def test_feature_names_iter3():
    names = ensemble.FEATURE_NAMES
    assert names[12:14] == ("home_bullpen_fip", "away_bullpen_fip")
    assert set(names[12:14]).issubset(features.feature_row_keys())
