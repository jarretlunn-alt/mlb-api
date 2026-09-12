"""Weather stub for run-scoring adjustments.

A real weather source is future work. Until then, ``get_weather_stub`` returns
neutral defaults and ``weather_run_adjustment`` converts a weather dict into a
run-scoring multiplier centered at 1.0. No network calls happen here.
"""

from __future__ import annotations

NEUTRAL_TEMP_F = 72.0
COLD_PENALTY_PER_DEGREE = 0.005  # below NEUTRAL_TEMP_F
HEAT_BONUS_PER_DEGREE = 0.003  # above NEUTRAL_TEMP_F
TEMP_ADJ_CAP = 0.10

WIND_THRESHOLD_MPH = 10.0
WIND_OUT_BONUS = 0.05
WIND_IN_PENALTY = 0.05
WIND_OUT_DIRS = frozenset({"out", "out_to_cf", "out_to_lf", "out_to_rf"})
WIND_IN_DIRS = frozenset({"in", "in_from_cf", "in_from_lf", "in_from_rf"})

MIN_ADJUSTMENT = 0.85
MAX_ADJUSTMENT = 1.15

WEATHER_KEYS = ("temp_f", "wind_mph", "wind_dir", "precip_in", "humidity_pct")


def get_weather_stub(game_pk: int, official_date: str) -> dict:
    """Returns neutral weather defaults until a real weather source is wired up."""
    return {
        "temp_f": 72,
        "wind_mph": 0,
        "wind_dir": "calm",
        "precip_in": 0.0,
        "humidity_pct": 50,
    }


def _temperature_adjustment(temp_f: float) -> float:
    delta = float(temp_f) - NEUTRAL_TEMP_F
    if delta < 0:
        adj = delta * COLD_PENALTY_PER_DEGREE  # negative
    else:
        adj = delta * HEAT_BONUS_PER_DEGREE
    return max(-TEMP_ADJ_CAP, min(TEMP_ADJ_CAP, adj))


def _wind_adjustment(wind_mph: float, wind_dir: str) -> float:
    if float(wind_mph) < WIND_THRESHOLD_MPH:
        return 0.0
    direction = (wind_dir or "").strip().lower()
    if direction in WIND_OUT_DIRS:
        return WIND_OUT_BONUS
    if direction in WIND_IN_DIRS:
        return -WIND_IN_PENALTY
    return 0.0


def weather_run_adjustment(weather: dict) -> float:
    """Estimated run-scoring multiplier from weather conditions.

    Cold (<50F) suppresses scoring; strong outward wind increases it.
    Returns a multiplier centered at 1.0.

    Linear model:
      - Temperature: -0.005 per degree below 72F, +0.003 per degree above
        72F, capped at +/-0.10.
      - Wind at 10+ mph: blowing out +0.05, blowing in -0.05.
      - Components combine multiplicatively; the result is clamped to
        [0.85, 1.15].
    Missing keys fall back to the neutral stub values.
    """
    defaults = get_weather_stub(0, "")
    temp_f = weather.get("temp_f", defaults["temp_f"])
    wind_mph = weather.get("wind_mph", defaults["wind_mph"])
    wind_dir = weather.get("wind_dir", defaults["wind_dir"])

    multiplier = (1.0 + _temperature_adjustment(temp_f)) * (1.0 + _wind_adjustment(wind_mph, wind_dir))
    return float(max(MIN_ADJUSTMENT, min(MAX_ADJUSTMENT, multiplier)))
