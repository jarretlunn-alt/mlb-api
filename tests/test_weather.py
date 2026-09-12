import csv
from pathlib import Path

import pytest

from mlb_pipeline import weather
from mlb_pipeline.weather import get_weather_stub, weather_run_adjustment

PARKS_CSV = Path(__file__).resolve().parents[1] / "data" / "parks.csv"


def test_module_exports_both_functions():
    assert callable(weather.get_weather_stub)
    assert callable(weather.weather_run_adjustment)


def test_stub_returns_all_expected_keys():
    result = get_weather_stub(700001, "2026-07-01")
    assert isinstance(result, dict)
    assert set(result) == {"temp_f", "wind_mph", "wind_dir", "precip_in", "humidity_pct"}
    assert result["wind_dir"] == "calm"
    assert result["temp_f"] == 72


def test_stub_weather_is_neutral():
    assert weather_run_adjustment(get_weather_stub(700001, "2026-07-01")) == pytest.approx(1.0)


def test_empty_dict_falls_back_to_neutral():
    assert weather_run_adjustment({}) == pytest.approx(1.0)


def test_cold_calm_suppresses_scoring():
    # 52F is 20 degrees below neutral: 20 * -0.005 = -0.10
    result = weather_run_adjustment({"temp_f": 52, "wind_mph": 0, "wind_dir": "calm"})
    assert result == pytest.approx(0.90)


def test_hot_wind_out_boosts_scoring():
    # 92F is 20 degrees above neutral: 20 * 0.003 = +0.06; wind out adds 5%
    result = weather_run_adjustment({"temp_f": 92, "wind_mph": 15, "wind_dir": "out"})
    assert result == pytest.approx(1.06 * 1.05)


def test_wind_in_reduces_scoring():
    result = weather_run_adjustment({"temp_f": 72, "wind_mph": 12, "wind_dir": "in"})
    assert result == pytest.approx(0.95)


def test_light_wind_is_ignored():
    result = weather_run_adjustment({"temp_f": 72, "wind_mph": 9, "wind_dir": "out"})
    assert result == pytest.approx(1.0)


def test_temperature_effect_is_capped():
    # 32F would be -0.20 uncapped; temperature component caps at -0.10
    assert weather_run_adjustment({"temp_f": 32, "wind_mph": 0, "wind_dir": "calm"}) == pytest.approx(0.90)
    # 120F would be +0.144 uncapped; caps at +0.10
    assert weather_run_adjustment({"temp_f": 120, "wind_mph": 0, "wind_dir": "calm"}) == pytest.approx(1.10)


def test_total_adjustment_capped_at_upper_bound():
    # 1.10 * 1.05 = 1.155 -> clamped to 1.15
    result = weather_run_adjustment({"temp_f": 120, "wind_mph": 30, "wind_dir": "out"})
    assert result == pytest.approx(1.15)


def test_total_adjustment_never_below_lower_bound():
    # 0.90 * 0.95 = 0.855 is the model minimum; the 0.85 clamp is a guard
    result = weather_run_adjustment({"temp_f": -40, "wind_mph": 40, "wind_dir": "in"})
    assert result >= 0.85
    assert result == pytest.approx(0.855)


@pytest.mark.parametrize(
    "w",
    [
        {"temp_f": -100, "wind_mph": 99, "wind_dir": "in"},
        {"temp_f": 200, "wind_mph": 99, "wind_dir": "out"},
        {"temp_f": 72, "wind_mph": 0, "wind_dir": "calm"},
        {"temp_f": 45, "wind_mph": 20, "wind_dir": "cross"},
    ],
)
def test_result_is_float_within_bounds(w):
    result = weather_run_adjustment(w)
    assert isinstance(result, float)
    assert 0.85 <= result <= 1.15


def test_parks_csv_has_30_teams():
    with PARKS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    assert len(rows) == 30
    assert list(rows[0]) == ["park_id", "name", "team_id", "team_name", "run_factor", "hr_factor", "handedness"]
    assert len({r["team_id"] for r in rows}) == 30
    assert len({r["park_id"] for r in rows}) == 30
    for r in rows:
        assert 0.8 <= float(r["run_factor"]) <= 1.2
        assert 0.8 <= float(r["hr_factor"]) <= 1.2
        assert r["handedness"] in {"rhb", "lhb", "neutral"}
