# Task 01b — Park factors seed data + weather schema stub

**Depends on:** nothing (runs in parallel with 01a)  
**Blocks:** nothing directly (02b uses park factors)  
**Files you own:** `data/parks.csv`, `src/mlb_pipeline/weather.py`, `tests/test_weather.py`

---

## What to build

This task runs in parallel with 01a. It does NOT touch `sql/schema.sql` (01a owns that).

### 1. `data/parks.csv` — static park factors for all 30 MLB teams

Create a CSV with columns:
`park_id, name, team_id, team_name, run_factor, hr_factor, handedness`

Use approximate 2022–2024 multi-year park factors (publicly available from FanGraphs or
Baseball Reference). Values should be centered at 1.0 (neutral). Example rows:

```
2392,Fenway Park,111,Boston Red Sox,1.05,0.97,rhb
680,T-Mobile Park,136,Seattle Mariners,0.94,0.88,neutral
19,Globe Life Field,140,Texas Rangers,1.01,1.04,neutral
```

Include all 30 teams. Add a comment row at the top documenting the source year range.

### 2. `src/mlb_pipeline/weather.py`

Stub module — the actual weather API integration is future work. For now:

```python
def get_weather_stub(game_pk: int, official_date: str) -> dict:
    """Returns neutral weather defaults until a real weather source is wired up."""
    return {
        "temp_f": 72,
        "wind_mph": 0,
        "wind_dir": "calm",
        "precip_in": 0.0,
        "humidity_pct": 50,
    }

def weather_run_adjustment(weather: dict) -> float:
    """Estimated run-scoring multiplier from weather conditions.
    Cold (<50F) suppresses scoring; strong outward wind increases it.
    Returns a multiplier centered at 1.0."""
```

Implement `weather_run_adjustment` with a simple linear model:
- Temperature: -0.005 per degree below 72°F, +0.003 per degree above 72°F (capped ±0.10)
- Wind blowing out at 10+ mph: +0.05; blowing in: -0.05
- Combine multiplicatively, cap total adjustment between 0.85 and 1.15

### 3. Tests

`tests/test_weather.py`:
- Test that `get_weather_stub` returns a dict with all expected keys
- Test `weather_run_adjustment` at a few boundary values (cold/calm, hot/wind-out, neutral)
- Test that adjustment is capped at 0.85 / 1.15

---

## Acceptance criteria

- `make test` passes
- `data/parks.csv` has exactly 30 rows (one per team) plus header
- `weather.py` exports both functions
- `weather_run_adjustment` returns a float between 0.85 and 1.15
