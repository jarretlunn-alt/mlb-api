# Iteration Task: Feature Gap Analysis

**Model:** alternates — `codex` on odd iterations, `claude` on even iterations
**Role:** You are `iter-gap-analysis-$ITER`. You read the backtest report from
`iter-backtest-$ITER` and decide which 2 features from the candidate pool will
most reduce error, given the specific patterns in this iteration's predictions.

**Token budget:** Read ONLY `tasks/reports/backtest_$ITER.json`,
`tasks/reports/loop_state.json`, and the `FEATURE_NAMES` tuple from
`src/mlb_pipeline/models/ensemble.py`. Do not read the full codebase.
Write `tasks/reports/proposal_$ITER.md` in ≤ 600 words total.

Read `AGENTS.md` before doing anything.

---

## Inputs

- `tasks/reports/backtest_$ITER.json` — metrics, worst predictions, feature importances

## Output

`tasks/reports/proposal_$ITER.md` — a structured feature proposal (schema below).

---

## Feature candidate pool

Each candidate has an ID, a description of what it measures, how to compute it
from the existing warehouse, and which error pattern it addresses.

| ID | Feature | Source tables | Error pattern it addresses |
|----|---------|--------------|---------------------------|
| `F01` | `starter_outs_mean_28d` — home/away starter avg outs per start, last 28 days | `fact_pitcher_log`, `fact_game` | Over-predicts strong teams when ace is on short rest or skipped |
| `F02` | `bullpen_era_15d` — team bullpen ERA last 15 days (non-starters only) | `fact_pitcher_log`, `fact_game` | Under-weights tired/depleted bullpens in close-game predictions |
| `F03` | `h2h_win_pct_3yr` — head-to-head home win% last 3 seasons | `fact_game` | Ignores known matchup advantages (e.g. division rivals) |
| `F04` | `road_trip_game_n` — how many consecutive away games the away team has played | `fact_game` | Undervalues fatigue from long road trips |
| `F05` | `runs_last_10_weighted` — exponentially weighted run differential, last 10 games | `fact_team_game` | Misses hot/cold streaks; calibration gaps in 0.45-0.55 bucket |
| `F06` | `lineup_ops_vs_sp_hand` — home/away lineup OPS split vs opponent starter hand | `fact_player_game_batting`, `fact_pitcher_log` | Under-leverages platoon advantage (RHP-heavy lineups vs LHP starters) |
| `F07` | `days_since_last_game` — rest days for each team | `fact_game` | Misses advantage of rest after off-day, especially for away team |
| `F08` | `park_factor_adj_run_line` — park-factor-adjusted run differential | `dim_park`, `fact_team_game` | Calibration gaps for extreme parks (Coors, T-Mobile) |
| `F09` | `starter_era_diff` — home starter ERA minus away starter ERA (last 28 days) | `fact_pitcher_log` | Weak feature signal from Poisson lam alone; ERA diff is more direct |
| `F10` | `team_ops_last_15d` — team offensive OPS last 15 days | `fact_player_game_batting`, `fact_game` | Slow to react to lineup changes or injured position players |

---

## Algorithm for choosing 2 features

Read the backtest report and run this logic:

### Step 1 — Identify dominant error type

Look at `calibration` buckets:
- If the worst miscalibration is in `0.45-0.65` (near-even games): model is overconfident on favorites → focus on uncertainty/context features (F03, F05, F07)
- If worst miscalibration is in `0.30-0.45` or `0.65-0.80` (moderate favorites): model underweights matchup-specific factors → focus on matchup features (F06, F01, F09)
- If worst miscalibration is outside `0.30-0.80` (extremes): model is poorly calibrated on large favorites → focus on regression-to-mean features (F08, F02)

### Step 2 — Check feature importances

From `feature_importances` in the backtest report:
- If `elo_home_rating` and `elo_away_rating` together > 60% of importance: model over-relies on Elo; try adding non-Elo signals (F03, F05, F10)
- If `poisson_lambda_home`/`away` are near-zero: Poisson is not contributing → try pitcher-level features (F01, F09)
- If `home_park_factor` is lowest importance: park signal is weak → try F08

### Step 3 — Scan worst 20 predictions

For each of the 20 worst-predicted games:
- Are they bunched on specific teams (team_id)? → F03 or F07
- Are they bunched on specific dates (consecutive game stretches)? → F04, F07
- Are they spread randomly? → F05, F10

### Step 4 — Exclude already-implemented features

Read `src/mlb_pipeline/models/ensemble.py` FEATURE_NAMES.
Filter out any candidate whose data is already captured by an existing feature.

### Step 5 — Read `tasks/reports/loop_state.json` if it exists

Exclude features that were tried in a previous iteration (regardless of outcome).
```json
{ "tried_features": ["F01", "F03"], "no_improve_count": 0 }
```

### Step 6 — Choose 2 features

Pick the 2 candidates with highest expected Δbrier based on steps 1-4,
excluding tried features (step 5). If fewer than 2 untried candidates remain,
trigger a plateau gate (see Critical Checks below).

---

## Output format for `tasks/reports/proposal_$ITER.md`

```markdown
# Feature Proposal — Iteration $ITER

## Reasoning
<2-3 sentences explaining which error patterns drove the choice>

## Selected Features

### Feature 1: <ID> — <name>
**SQL/Python sketch:**
```python
# how to compute it from the warehouse
```
**Expected impact:** <one sentence on which error pattern this addresses>
**Column name in FEATURE_NAMES:** `<snake_case_name>`
**Return type:** float (normalized 0-1 preferred but not required)

### Feature 2: <ID> — <name>
... (same structure)

## Implementation notes
<Any edge cases, NULL handling, leakage risk, or test fixture requirements>

## Features NOT chosen this iteration
<Brief note on why each excluded candidate was skipped>
```

---

## Critical checks

**Plateau gate:** If fewer than 2 untried features remain:
```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "Feature pool exhausted after $ITER iterations. All 10 candidate features have been tried. Best Brier achieved: $BEST. Options: extend the feature pool manually, run more historical seasons, or stop the loop." \
  --options '["Stop the loop — results are good enough", "Add new features manually (I will update the pool)", "Try ensembling multiple models differently"]' \
  --json
```

---

## Signal done

```sh
orca orchestration send \
  --type worker_done \
  --outcome succeeded \
  --task-id $TASK_ID \
  --dispatch-id $DISPATCH_ID \
  --subject "iter-gap-analysis-$ITER complete" \
  --body "Selected features: $FEAT1, $FEAT2. Reasoning: <one sentence>. Proposal: tasks/reports/proposal_$ITER.md" \
  --files-modified "tasks/reports/proposal_$ITER.md" \
  --json
```
