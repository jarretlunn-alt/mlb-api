# Task 02c — Walk-forward backtest framework

**Depends on:** Task 01a merged  
**Blocks:** 03  
**Files you own:** `src/mlb_pipeline/backtest.py`, `tests/test_backtest.py`

---

## What to build

### 1. `src/mlb_pipeline/backtest.py`

Walk-forward backtesting harness. Trains on seasons 1..N-1, evaluates on season N,
accumulating predictions into a result set for scoring.

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class BacktestResult:
    predictions: list[dict]   # {game_pk, date, home_id, away_id, home_win_prob, actual_home_win}
    brier_score: float         # lower is better; 0.25 = coin-flip baseline
    log_loss: float
    calibration: dict          # {bucket_label: {predicted_prob, actual_win_rate, n}}
    roi_flat_bet: float        # simulated ROI betting $1 on every game where edge > 0.03
    roi_kelly: float           # quarter-Kelly ROI

def brier_score(predictions: list[dict]) -> float:
    """Mean squared error of home_win_prob vs actual_home_win."""

def log_loss(predictions: list[dict]) -> float:
    """Negative mean log likelihood."""

def calibration_buckets(predictions: list[dict], n_buckets: int = 10) -> dict:
    """Bin predictions by predicted probability; compute actual win rate per bucket."""

def simulate_roi(
    predictions: list[dict],
    market_vig: float = 0.045,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> dict:
    """Simulate flat-bet and fractional-Kelly ROI.
    market_vig: the sportsbook take (reduces implied probability).
    min_edge: only bet when model_prob - implied_prob > min_edge.
    Returns {'flat_bet_roi', 'kelly_roi', 'bets_placed', 'total_games'}."""

def run_backtest(
    con,
    predict_fn: Callable,
    train_seasons: list[int],
    test_season: int,
) -> BacktestResult:
    """Run one walk-forward fold.
    - Calls predict_fn(con, game_pk, home_id, away_id, as_of_date) for each test game.
    - predict_fn must be a callable that returns a GamePrediction.
    - Scores results against actual outcomes from fact_game + fact_team_game."""
```

### 2. Tests (`tests/test_backtest.py`)

Use synthetic data: insert 20 fake games into an in-memory DuckDB, with known outcomes.

- `brier_score` with perfect predictions returns 0.0
- `brier_score` with 0.5 predictions always returns 0.25 (coin-flip baseline)
- `log_loss` with perfect predictions returns 0.0
- `calibration_buckets` returns a dict keyed by probability range strings
- `simulate_roi` with edge=0 on all games returns `bets_placed == 0`
- `run_backtest` with a mock predict_fn (always returns 0.5) completes without error and returns a `BacktestResult`

### 3. Notes

- Do not import any model (`elo.py`, `poisson.py`) inside `backtest.py` — keep it model-agnostic.
  Models are passed in as callables.
- `run_backtest` should skip games without a final result (status != 'Final').

---

## Acceptance criteria

- `make test` passes
- All six exports present in `backtest.py`
- No model imports inside `backtest.py`
- `BacktestResult.brier_score` is populated correctly
