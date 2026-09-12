# Task 02a — Elo ratings + Pythagorean win% model

**Depends on:** Task 01a merged  
**Blocks:** 03 (ensemble needs all models)  
**Files you own:** `src/mlb_pipeline/models/elo.py`, `src/mlb_pipeline/predict.py`, `tests/test_elo.py`

---

## What to build

### 1. `src/mlb_pipeline/models/elo.py`

Elo rating system updated after every game. Ratings are stored in memory (dict keyed by
team_id) and persisted to `fact_prediction` only at prediction time.

```python
DEFAULT_RATING = 1500.0
K_FACTOR = 20.0
SEASON_REGRESS = 0.4   # pull 40% toward mean at season start

def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo win probability for team A given ratings."""

def update_ratings(
    ratings: dict[int, float],
    home_id: int,
    away_id: int,
    home_won: bool,
    home_advantage: float = 25.0,
) -> dict[int, float]:
    """Return new ratings dict after one game result."""

def build_ratings_from_history(con, through_date: str) -> dict[int, float]:
    """Replay all fact_game rows up to through_date and return current ratings.
    Apply season-start regression when the season changes.
    Only use games with status='Final'."""

def predict(
    ratings: dict[int, float],
    home_id: int,
    away_id: int,
    home_advantage: float = 25.0,
) -> dict:
    """Return {'home_win_prob': float, 'away_win_prob': float, 'model': 'elo'}."""
```

Also add a `pythagorean_win_pct(runs_scored: float, runs_allowed: float, exponent: float = 1.83) -> float` helper.

### 2. `src/mlb_pipeline/predict.py`

Unified prediction interface. All models must conform to this:

```python
from dataclasses import dataclass

@dataclass
class GamePrediction:
    game_pk: int
    model_name: str
    home_win_prob: float
    away_win_prob: float
    pred_total: float | None
    features: dict

def save_prediction(con, pred: GamePrediction) -> None:
    """Upsert into fact_prediction."""
```

### 3. Tests (`tests/test_elo.py`)

- `expected_score` is symmetric: `expected_score(a, b) + expected_score(b, a) == 1.0`
- `update_ratings` moves winner's rating up and loser's down
- After replaying fixture game (home win 5–3), home team rating > DEFAULT_RATING
- `pythagorean_win_pct(800, 600)` is between 0.55 and 0.65
- `predict` returns probs that sum to 1.0

---

## Acceptance criteria

- `make test` passes
- `elo.py` exports all four functions
- `predict.py` exports `GamePrediction` and `save_prediction`
- No live network calls
