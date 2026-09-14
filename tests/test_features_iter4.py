from mlb_pipeline import features
from mlb_pipeline.models import ensemble


def test_feature_names_iter4():
    names = ensemble.FEATURE_NAMES
    assert len(names) == 16
    assert names[14:16] == ("home_sp_k_per_9", "away_sp_k_per_9")
    assert set(names[14:16]).issubset(features.feature_row_keys())
