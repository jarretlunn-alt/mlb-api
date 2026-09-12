# Task 02b — Poisson run-scoring model

**Depends on:** Task 01a merged  
**Blocks:** 03  
**Files you own:** `src/mlb_pipeline/models/poisson.py`, `tests/test_poisson.py`

---

## What to build

### 1. `src/mlb_pipeline/models/poisson.py`

Model each team's runs scored as an independent Poisson process. Given lambdas for home
and away, compute win/loss/tie probabilities and over/under probabilities analytically.

```python
import math
from scipy.stats import poisson as scipy_poisson
from mlb_pipeline.predict import GamePrediction

def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""

def win_prob_from_lambdas(lam_home: float, lam_away: float, max_runs: int = 30) -> dict:
    """Compute P(home wins), P(away wins), P(tie) by summing over score matrices.
    Returns {'home_win_prob', 'away_win_prob', 'tie_prob'}."""

def over_prob(lam_home: float, lam_away: float, total_line: float) -> float:
    """P(home_runs + away_runs > total_line)."""

def estimate_lambda(
    team_runs_per_game: float,
    opp_runs_allowed_per_game: float,
    league_avg: float,
    park_factor: float = 1.0,
    weather_adjustment: float = 1.0,
) -> float:
    """Log5-style lambda: (off * def / avg) * park * weather."""

def predict(
    con,
    game_pk: int,
    home_id: int,
    away_id: int,
    as_of_date: str,
    total_line: float = 8.5,
) -> GamePrediction:
    """Fetch rolling features from con, estimate lambdas, return GamePrediction.
    Uses features.team_rolling_features and features.game_context_features.
    Falls back to league-average lambda (4.5) for teams with insufficient history."""
```

Use `scipy.stats.poisson` for PMF calculations (add `scipy` to `pyproject.toml` dependencies).

### 2. Tests (`tests/test_poisson.py`)

Use an in-memory DuckDB populated from the fixture files (borrow the pattern from conftest.py).

- `poisson_pmf(0, 4.5)` ≈ `math.exp(-4.5)`
- `win_prob_from_lambdas(5.0, 3.0)['home_win_prob']` > 0.5
- `win_prob_from_lambdas(lam, lam)['home_win_prob']` ≈ `win_prob_from_lambdas(lam, lam)['away_win_prob']` (symmetric)
- Probabilities in `win_prob_from_lambdas` sum to 1.0
- `over_prob(4.5, 4.5, 8.5)` is between 0.40 and 0.60
- `estimate_lambda(5.0, 4.0, 4.5)` is between 3.5 and 6.0

### 3. Add `scipy` to `pyproject.toml`

Add `scipy>=1.12` to the `dependencies` list.

---

## Acceptance criteria

- `make test` passes
- All five public functions exported from `poisson.py`
- `scipy` added to `pyproject.toml`
- `predict()` returns a `GamePrediction` with `pred_total` set
