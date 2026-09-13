# Iteration Task: Implement Proposed Features

**Model:** `codex`
**Role:** You are `iter-implement-$ITER`. You read `tasks/reports/proposal_$ITER.md`
and implement the two proposed features in `features.py` and `ensemble.py`.
Your work is purely additive — do not remove existing features.

**Token budget:** Read ONLY these files (nothing else):
- `tasks/reports/proposal_$ITER.md` (the spec)
- `src/mlb_pipeline/features.py` (to append your new functions)
- `src/mlb_pipeline/models/ensemble.py` (FEATURE_NAMES + _feature_row only)
- `tests/conftest.py` (fixture shapes only — read the first 60 lines)

Do not read the full test suite, raw data files, or Parquet exports.

Read `AGENTS.md` before doing anything.

---

## Inputs

- `tasks/reports/proposal_$ITER.md` — feature specs from gap analysis
- `src/mlb_pipeline/features.py` — existing feature engineering
- `src/mlb_pipeline/models/ensemble.py` — current FEATURE_NAMES + _feature_row()

## What you must NOT do

- Do not remove or rename existing features in FEATURE_NAMES
- Do not modify `sql/schema.sql` unless the proposal explicitly requires a new table
  (the 10 candidate features do not require schema changes — they all compute from
  existing tables)
- Do not modify `tests/conftest.py`
- Do not call the live network anywhere

---

## Step 1 — Read and understand the proposal

```python
from pathlib import Path
import re

ITER = ...  # set from environment or loop_state.json
proposal = Path(f"tasks/reports/proposal_{ITER}.md").read_text(encoding="utf-8")
print(proposal)
```

---

## Step 2 — Implement feature functions in `features.py`

Each feature function must follow this signature:

```python
def feature_<name>(con, team_id: int, as_of_date: str, **kwargs) -> float:
    """One-line description. Returns NaN if insufficient data."""
    ...
```

**Leakage rule:** Only query rows where `official_date < as_of_date` (strict less-than).
Never touch `home_score`, `away_score`, or `winning_team_id` from rows >= as_of_date.

**NULL/NaN handling:** If the query returns 0 rows or all-NULL values, return
`float("nan")` rather than 0.0. The ensemble's `_feature_row()` will replace NaN
with league-average at training/inference time.

**Example skeleton for `starter_era_diff` (F09):**

```python
_STARTER_ERA_SQL = """
    SELECT
        SUM(earned_runs) * 27.0 / NULLIF(SUM(outs), 0) AS era
    FROM fact_pitcher_log pl
    JOIN fact_game g ON g.game_pk = pl.game_pk
    WHERE pl.team_id = ? AND pl.is_starter = TRUE
      AND g.official_date >= ? AND g.official_date < ?
"""

def feature_starter_era_28d(con, team_id: int, as_of_date: str, window_days: int = 28) -> float:
    import datetime as dt
    end = dt.date.fromisoformat(as_of_date)
    start = (end - dt.timedelta(days=window_days)).isoformat()
    row = con.execute(_STARTER_ERA_SQL, [team_id, start, as_of_date]).fetchone()
    if row is None or row[0] is None:
        return float("nan")
    return float(row[0])
```

---

## Step 3 — Register features in `ensemble.py`

### 3a — Add to FEATURE_NAMES tuple

```python
# ensemble.py
FEATURE_NAMES = (
    "elo_home_rating", "elo_away_rating", "poisson_lambda_home",
    "poisson_lambda_away", "home_runs_per_game_15d", "away_runs_allowed_15d",
    "home_park_factor", "is_dome",
    # --- iteration $ITER additions ---
    "home_<feature_1>",
    "away_<feature_1>",   # add both home + away variants unless the feature is symmetric
    "home_<feature_2>",
    "away_<feature_2>",
)
```

Note: Most features need a home variant and an away variant. Symmetric features
(e.g. h2h_win_pct) only need one column since it is already relative.

### 3b — Populate in `_feature_row()`

In `ensemble.py`, after the existing `values = [...]` block:

```python
import math

def _nan_to_mean(v, fallback):
    """Replace NaN with a league-average fallback."""
    return fallback if (v is None or (isinstance(v, float) and math.isnan(v))) else v

# add to values list:
home_feat1 = _features.feature_<name1>(con, home_id, as_of_date)
away_feat1 = _features.feature_<name1>(con, away_id, as_of_date)
home_feat2 = _features.feature_<name2>(con, home_id, as_of_date)
away_feat2 = _features.feature_<name2>(con, away_id, as_of_date)

values.extend([
    _nan_to_mean(home_feat1, <league_avg>),
    _nan_to_mean(away_feat1, <league_avg>),
    _nan_to_mean(home_feat2, <league_avg>),
    _nan_to_mean(away_feat2, <league_avg>),
])
```

---

## Step 4 — Write unit tests

Create `tests/test_features_iter$ITER.py` with at least 3 tests per feature:

1. **Basic correctness** — manually compute expected value from a known in-memory
   DuckDB table, assert within 0.01.
2. **Leakage guard** — verify the function returns the same result when future rows
   are added to the fixture.
3. **Empty data** — verify `float("nan")` returned when no data exists.

Use `tests/conftest.py`'s `con` fixture (in-memory DuckDB) for all tests.
Never call the live MLB API.

```python
# tests/test_features_iter1.py
import math
import pytest

def test_feature_<name>_basic(con):
    # insert known data
    con.execute("INSERT OR REPLACE INTO fact_pitcher_log VALUES (...)")
    result = features.feature_<name>(con, team_id=139, as_of_date="2025-07-15")
    assert abs(result - <expected>) < 0.01

def test_feature_<name>_no_leakage(con):
    # insert pre-date data, get result
    r1 = features.feature_<name>(con, team_id=139, as_of_date="2025-07-15")
    # insert post-date data (should not change result)
    con.execute("INSERT OR REPLACE INTO fact_pitcher_log VALUES (...future row...)")
    r2 = features.feature_<name>(con, team_id=139, as_of_date="2025-07-15")
    assert r1 == pytest.approx(r2)

def test_feature_<name>_empty_returns_nan(con):
    result = features.feature_<name>(con, team_id=9999, as_of_date="2025-07-15")
    assert math.isnan(result)
```

---

## Step 5 — Run tests

```sh
python -m pytest tests/test_features_iter$ITER.py -v
python -m pytest -q   # full suite must pass
```

If ANY test in the full suite fails:
1. Fix the failure. Do not disable or delete tests.
2. If after 2 fix attempts the tests still fail, trigger a gate:

```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "iter-implement-$ITER: Tests still failing after 2 fix attempts. Error: <paste error>. How should we proceed?" \
  --options '["Revert the feature changes and skip this iteration", "I will investigate and fix manually", "Try a simpler fallback implementation"]' \
  --json
```

---

## Step 6 — Signal done

```sh
orca orchestration send \
  --type worker_done \
  --outcome succeeded \
  --task-id $TASK_ID \
  --dispatch-id $DISPATCH_ID \
  --subject "iter-implement-$ITER complete" \
  --body "Added features: <F_ID_1> (<column_name>), <F_ID_2> (<column_name>). All tests pass." \
  --files-modified "src/mlb_pipeline/features.py,src/mlb_pipeline/models/ensemble.py,tests/test_features_iter$ITER.py" \
  --json
```
