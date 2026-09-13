from mlb_pipeline.models import ensemble


def test_feature_names_iter1():
    names = ensemble.FEATURE_NAMES
    assert len(names) == 10
    assert names[-2:] == ("home_sp_fip_last_n", "away_sp_fip_last_n")
