# Task 01a — Schema extensions + rolling stats ingestion

**Depends on:** nothing (first task)  
**Blocks:** 02a, 02b, 02c  
**Files you own:** `sql/schema.sql` (additive only), `src/mlb_pipeline/features.py`, `tests/test_features.py`

---

## What to build

### 1. Extend `sql/schema.sql` with four new tables

Add these at the bottom — do not modify existing tables.

```sql
CREATE TABLE IF NOT EXISTS fact_pitcher_log (
    game_pk          BIGINT,
    player_id        INTEGER,
    team_id          INTEGER,
    is_starter       BOOLEAN,
    official_date    DATE,
    rest_days        INTEGER,   -- days since last appearance
    outs             INTEGER,
    hits             INTEGER,
    runs             INTEGER,
    earned_runs      INTEGER,
    walks            INTEGER,
    strikeouts       INTEGER,
    home_runs        INTEGER,
    pitches          INTEGER,
    PRIMARY KEY (game_pk, player_id)
);

CREATE TABLE IF NOT EXISTS fact_team_rolling (
    team_id        INTEGER,
    as_of_date     DATE,
    window_days    INTEGER,   -- 7, 15, or 30
    runs_scored    DOUBLE,    -- per game average
    runs_allowed   DOUBLE,
    ops_vs_rhp     DOUBLE,
    ops_vs_lhp     DOUBLE,
    fip            DOUBLE,    -- team bullpen FIP
    PRIMARY KEY (team_id, as_of_date, window_days)
);

CREATE TABLE IF NOT EXISTS dim_park (
    park_id        INTEGER PRIMARY KEY,
    name           VARCHAR,
    team_id        INTEGER,
    run_factor     DOUBLE,   -- 1.0 = neutral; >1.0 = hitter-friendly
    hr_factor      DOUBLE,
    handedness     VARCHAR   -- 'neutral', 'rhb', 'lhb'
);

CREATE TABLE IF NOT EXISTS fact_prediction (
    prediction_id  VARCHAR PRIMARY KEY,  -- "{model_name}_{game_pk}"
    game_pk        BIGINT,
    model_name     VARCHAR,
    predicted_at   TIMESTAMP,
    home_win_prob  DOUBLE,
    away_win_prob  DOUBLE,
    pred_total     DOUBLE,
    features_json  JSON
);
```

### 2. Update `db.py`

Add the four new tables to the `TABLES` list.

### 3. Create `src/mlb_pipeline/features.py`

This module computes pre-game features from the warehouse for a given (game_pk, date).
It has no I/O side-effects — it only reads from the DuckDB connection passed to it.

Required functions:

```python
def pitcher_features(con, player_id: int, as_of_date: str, n_starts: int = 5) -> dict:
    """Recent performance for a starting pitcher.
    Returns: era_last_n, fip_last_n, k_per_9, bb_per_9, hr_per_9, avg_rest_days.
    Returns None if fewer than 2 starts found."""

def team_rolling_features(con, team_id: int, as_of_date: str, window: int = 15) -> dict:
    """Team offensive/defensive rolling stats.
    Returns: runs_per_game, runs_allowed_per_game, ops_vs_rhp, ops_vs_lhp, bullpen_fip."""

def game_context_features(con, game_pk: int) -> dict:
    """Park factor and home/away flag for a game.
    Returns: park_run_factor, park_hr_factor, is_home (for each side)."""

def build_game_feature_row(con, game_pk: int, as_of_date: str) -> dict | None:
    """Assemble all features for one game into a flat dict suitable for model input.
    Returns None if required features are unavailable (e.g. new team, no starts)."""
```

### 4. Populate `dim_park` with static data

Hard-code a `PARK_FACTORS` dict in `features.py` with approximate 2024-season park factors
for all 30 MLB teams (use publicly available estimates — exact values don't matter for the MVP).
Expose a function `seed_park_factors(con)` that upserts them into `dim_park`.

### 5. Tests

Create `tests/test_features.py` using an in-memory DuckDB populated from
`tests/fixtures/feed_live_700001.json` and `tests/fixtures/boxscore_700001.json`.
Test that `build_game_feature_row` returns a dict with the expected keys.
Test that `pitcher_features` returns None when there are fewer than 2 starts.

---

## Acceptance criteria

- `make test` passes
- `dim_park`, `fact_pitcher_log`, `fact_team_rolling`, `fact_prediction` exist in the schema
- `features.py` exports the four functions above
- `db.TABLES` includes all four new tables
