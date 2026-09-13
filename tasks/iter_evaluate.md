# Iteration Task: Evaluate & Commit or Revert

**Model:** `claude`
**Role:** You are `iter-evaluate-$ITER`. You retrain the ensemble on all available
complete seasons, run a holdout backtest, compare against the baseline from
`iter-backtest-$ITER`, and decide whether to commit or revert.

**Token budget:** Read ONLY:
- `tasks/reports/backtest_$ITER.json` (baseline metrics — already trimmed to ≤ 50 KB)
- `tasks/reports/loop_state.json`
- `tasks/reports/.current_iter`

Do not re-read source files unless a test fails and you need to diagnose.
Do not read raw data, Parquet files, or previous iterations' reports.

Read `AGENTS.md` before doing anything.

---

## Inputs

- `tasks/reports/backtest_$ITER.json` — baseline metrics (before new features)
- `src/mlb_pipeline/models/ensemble.py` — updated with new FEATURE_NAMES
- `src/mlb_pipeline/features.py` — updated with new feature functions

---

## Decision threshold

| Condition | Action |
|-----------|--------|
| Δbrier ≥ 0.002 AND tests pass AND no NaN predictions | **Commit** — save model, update loop state |
| Δbrier < 0.002 OR any test failure | **Revert** features — restore previous FEATURE_NAMES and feature functions |
| Brier > 0.260 after retraining | **Gate** — critical degradation |
| NaN predictions detected | **Gate** — implementation bug |

---

## Step 1 — Retrain on all complete seasons

```python
import pickle, json, math
from pathlib import Path
from mlb_pipeline.config import Settings
from mlb_pipeline.db import connect
from mlb_pipeline.models import ensemble

settings = Settings.from_env()
con = connect(settings.db_path)

# Use CLI to avoid reimplementing the training pipeline
import subprocess
result = subprocess.run(
    ["python", "-m", "mlb_pipeline.cli", "train"],
    capture_output=True, text=True
)
print(result.stdout, result.stderr)
if result.returncode != 0:
    raise RuntimeError(f"Training failed: {result.stderr}")
```

### 1b — Verify all tests still pass

```sh
python -m pytest -q
```

If tests fail → revert (Step 4), do not commit.

---

## Step 2 — Run holdout backtest

Use the same split as iter-backtest used (last season as test).

```python
from mlb_pipeline.backtest import run_backtest

model_path = settings.data_dir / "models" / "mlb_ensemble.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

seasons_query = con.execute("""
    SELECT season FROM fact_game
    WHERE status IN ('Final','Completed Early','Game Over')
      AND home_score IS NOT NULL AND away_score IS NOT NULL
      AND home_score <> away_score
    GROUP BY season
    HAVING count(*) > 50
    ORDER BY season
""").fetchall()
all_seasons = [s[0] for s in seasons_query]
train_seasons = all_seasons[:-1]
test_season   = all_seasons[-1]

def predict_fn(con, game_pk, home_id, away_id, as_of_date):
    return ensemble.predict(model, con, game_pk, home_id, away_id, as_of_date)

new_result = run_backtest(con, predict_fn, train_seasons, test_season)
```

---

## Step 3 — Compare and decide

```python
ITER = ...  # read from tasks/reports/.current_iter

baseline = json.loads(Path(f"tasks/reports/backtest_{ITER}.json").read_text())
baseline_brier = baseline["brier_score"]
new_brier = new_result.brier_score

delta_brier = baseline_brier - new_brier  # positive = improvement
print(f"Baseline Brier: {baseline_brier:.4f}  New Brier: {new_brier:.4f}  Δ: {delta_brier:+.4f}")

# NaN check
nan_count = sum(1 for p in new_result.predictions if math.isnan(p["home_win_prob"]))

# Write evaluation report
eval_report = {
    "iteration": ITER,
    "baseline_brier": baseline_brier,
    "new_brier": new_brier,
    "delta_brier": delta_brier,
    "new_log_loss": new_result.log_loss,
    "new_roi_kelly": new_result.roi_kelly,
    "n_games": new_result.n_games,
    "nan_predictions": nan_count,
    "decision": None,  # filled below
}

IMPROVE_THRESHOLD = 0.002

if nan_count > 0:
    eval_report["decision"] = "gate_nan"
elif new_brier > 0.260:
    eval_report["decision"] = "gate_degraded"
elif delta_brier >= IMPROVE_THRESHOLD:
    eval_report["decision"] = "commit"
else:
    eval_report["decision"] = "revert"

Path(f"tasks/reports/evaluation_{ITER}.json").write_text(
    json.dumps(eval_report, indent=2), encoding="utf-8"
)
print(f"Decision: {eval_report['decision']}")
```

---

## Step 4 — Execute decision

### If `decision == "commit"`:

```sh
# Commit the feature changes and retrained model metadata
git add src/mlb_pipeline/features.py \
        src/mlb_pipeline/models/ensemble.py \
        "tests/test_features_iter${ITER}.py" \
        "tasks/reports/backtest_${ITER}.json" \
        "tasks/reports/proposal_${ITER}.md" \
        "tasks/reports/evaluation_${ITER}.json"

git commit -m "feat: iter-$ITER feature additions (Δbrier=$DELTA, KellyROI=$KROI)

New features from gap analysis iteration $ITER. Baseline brier=$BASELINE
improved to $NEW_BRIER on $TEST_SEASON holdout.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push origin main
```

Then update loop state:
```python
state_path = Path("tasks/reports/loop_state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.setdefault("tried_features", [])
state.setdefault("no_improve_count", 0)
state["tried_features"].extend([<feat_id_1>, <feat_id_2>])
state["no_improve_count"] = 0
state["best_brier"] = min(state.get("best_brier", 9.9), new_brier)
state["last_iter"] = ITER
# increment for next iteration
Path("tasks/reports/.current_iter").write_text(str(ITER + 1), encoding="utf-8")
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
```

### If `decision == "revert"`:

```sh
# Revert only the feature files, not the test/report files
git checkout HEAD -- src/mlb_pipeline/features.py src/mlb_pipeline/models/ensemble.py
```

Then update loop state:
```python
state_path = Path("tasks/reports/loop_state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.setdefault("tried_features", [])
state.setdefault("no_improve_count", 0)
state["tried_features"].extend([<feat_id_1>, <feat_id_2>])
state["no_improve_count"] = state["no_improve_count"] + 1
state["last_iter"] = ITER
Path("tasks/reports/.current_iter").write_text(str(ITER + 1), encoding="utf-8")
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
```

### Plateau gate — fire when `no_improve_count >= 3`:

```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "Model has not improved for 3 consecutive iterations. Best Brier: $BEST. Tried features: $TRIED. Options?" \
  --options '["Stop the loop — this accuracy is acceptable", "Add more training seasons (load more data)", "Try different model architecture (I will specify)", "Reset and try a different feature combination"]' \
  --json
```

### Critical gate — fire for `gate_nan` or `gate_degraded`:

```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "Critical issue in iter-evaluate-$ITER: $REASON. Brier=$BRIER, NaNs=$NANS. See tasks/reports/evaluation_$ITER.json." \
  --options '["Revert all iter-$ITER changes and stop the loop", "Revert iter-$ITER changes and continue next iteration", "I will fix manually"]' \
  --json
```

---

## Step 5 — Signal done

```sh
orca orchestration send \
  --type worker_done \
  --outcome succeeded \
  --task-id $TASK_ID \
  --dispatch-id $DISPATCH_ID \
  --subject "iter-evaluate-$ITER: $DECISION (Δbrier=$DELTA)" \
  --body "Baseline: $BASELINE_BRIER, New: $NEW_BRIER, Δ=$DELTA. Decision: $DECISION. Loop state updated. Next iteration: $NEXT_ITER." \
  --files-modified "tasks/reports/evaluation_$ITER.json,tasks/reports/loop_state.json,tasks/reports/.current_iter" \
  --json
```
