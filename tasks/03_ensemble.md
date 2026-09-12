# Task 03 — XGBoost ensemble + calibration + full backtest

**Depends on:** Tasks 01a, 02a, 02b, 02c all merged  
**Blocks:** nothing (final analytics task)  
**Files you own:** `src/mlb_pipeline/models/ensemble.py`, `tests/test_ensemble.py`

---

## What to build

### 1. `src/mlb_pipeline/models/ensemble.py`

XGBoost classifier trained on features from `features.py`, using Elo and Poisson outputs
as meta-features alongside raw engineered features.

```python
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from mlb_pipeline.predict import GamePrediction

def build_training_set(con, seasons: list[int]) -> tuple:
    """Assemble (X, y) for training.
    X columns (minimum): elo_home_rating, elo_away_rating, poisson_lambda_home,
    poisson_lambda_away, home_runs_per_game_15d, away_runs_allowed_15d,
    home_park_factor, is_dome.
    y: 1 if home team won, 0 otherwise.
    Skip games where features are unavailable."""

def train(con, seasons: list[int]) -> CalibratedClassifierCV:
    """Train an XGBClassifier wrapped in isotonic calibration.
    Use early stopping on a 10% validation split.
    Add xgboost>=2.0 to pyproject.toml."""

def predict(
    model: CalibratedClassifierCV,
    con,
    game_pk: int,
    home_id: int,
    away_id: int,
    as_of_date: str,
) -> GamePrediction:
    """Return calibrated win probability."""

def feature_importances(model: CalibratedClassifierCV) -> dict[str, float]:
    """Return feature name → gain importance from the underlying XGBClassifier."""
```

### 2. Full backtest run

At the bottom of `ensemble.py`, add a `if __name__ == "__main__":` block that:
1. Connects to `data/warehouse.duckdb`
2. Trains on all available seasons except the most recent
3. Runs `backtest.run_backtest` on the most recent season
4. Prints Brier score, log-loss, calibration table, and ROI simulation
5. Also runs the same backtest for the Elo and Poisson models for comparison

### 3. Add `xgboost>=2.0` to `pyproject.toml`

### 4. Tests (`tests/test_ensemble.py`)

Use a minimal in-memory DuckDB with just enough fake rows to test the pipeline.

- `build_training_set` returns arrays of correct shape
- `train` returns a `CalibratedClassifierCV` (don't actually train on real data in tests — mock the DuckDB to return a minimal dataset of ~50 fake games)
- `predict` returns a `GamePrediction` with `home_win_prob` between 0 and 1
- `feature_importances` returns a non-empty dict

### 5. Notes

- If the warehouse has fewer than 2 complete seasons, `train` should raise a clear
  `ValueError("Need at least 2 seasons of data to train. Run make ingest-date for more dates.")`.
- Calibration matters more than raw AUC for this use case — isotonic regression via
  `CalibratedClassifierCV(method='isotonic')` is the right choice.
- The ensemble is not expected to dramatically beat Elo+Poisson on a small dataset;
  the goal is a correct, testable pipeline.

---

## Acceptance criteria

- `make test` passes
- `xgboost` and `scikit-learn` in `pyproject.toml`
- `ensemble.py` exports `build_training_set`, `train`, `predict`, `feature_importances`
- `__main__` block runs to completion when warehouse has data
- No hard-coded season numbers (derive from what's in the DB)
