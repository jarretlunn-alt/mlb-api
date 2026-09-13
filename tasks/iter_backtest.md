# Iteration Task: Backtest & Profile

**Model:** `claude`
**Role:** You are `iter-backtest-$ITER`. You run the walk-forward backtest on the
current ensemble, then produce a structured JSON report that the next worker
(`iter-gap-analysis`) will use to choose new features.

**Token budget:** Keep your context small. Read only the files listed below. Do
not cat the full codebase or any large raw/parquet files. Limit
`worst_predictions` to the top 10 rows so the JSON stays under 50 KB.

Read `AGENTS.md` before doing anything. Read only: `src/mlb_pipeline/backtest.py`,
`src/mlb_pipeline/models/ensemble.py` (FEATURE_NAMES only), `src/mlb_pipeline/config.py`.

---

## What you must produce

`tasks/reports/backtest_$ITER.json` — a JSON object with the schema below.
Write it with `json.dumps(report, indent=2)` using Python.

```json
{
  "iteration": <int>,
  "brier_score": <float>,
  "log_loss":    <float>,
  "roi_flat":    <float>,
  "roi_kelly":   <float>,
  "n_games":     <int>,
  "bets_placed": <int>,
  "train_seasons": [<int>, ...],
  "test_seasons":  [<int>, ...],
  "calibration": {
    "bucket": { "predicted_prob": float, "actual_win_rate": float, "n": int },
    ...
  },
  "calibration_max_error": <float>,
  "worst_predictions": [
    { "game_pk": int, "date": str, "home_id": int, "away_id": int,
      "home_win_prob": float, "actual_home_win": bool, "sq_error": float },
    ...20 rows sorted by sq_error desc...
  ],
  "feature_importances": { "<feature_name>": <float>, ... },
  "previous_brier": <float or null>,
  "delta_brier":    <float or null>,
  "model_path":     "<str>"
}
```

---

## Steps

### 1. Confirm data is available

```python
import duckdb, json
from mlb_pipeline.config import Settings
from mlb_pipeline.db import connect

settings = Settings.from_env()
con = connect(settings.db_path)

seasons = con.execute("""
    SELECT season, count(*) AS n
    FROM fact_game
    WHERE status IN ('Final','Completed Early','Game Over')
      AND home_score IS NOT NULL AND away_score IS NOT NULL
      AND home_score <> away_score
    GROUP BY season ORDER BY season
""").fetchall()
print(seasons)
```

If fewer than 2 complete seasons are available, create a gate:
```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "Fewer than 2 complete seasons in warehouse. Load more data with: make ingest-date DATE=<start> END_DATE=<end>. How many seasons should be targeted?" \
  --options '["2 seasons (2024-2025)", "3 seasons (2023-2025)", "4 seasons (2022-2025)"]' \
  --json
```

### 2. Load the current model

```python
import pickle
from pathlib import Path

model_path = settings.data_dir / "models" / "mlb_ensemble.pkl"
if not model_path.exists():
    raise RuntimeError("No model found — run: python -m mlb_pipeline.cli train")

with open(model_path, "rb") as f:
    model = pickle.load(f)
```

### 3. Run walk-forward backtest

Use the **last two** complete seasons as test (e.g. if seasons=[2023,2024,2025],
train=[2023,2024], test=[2025]). If only 2 seasons, train=[2023], test=[2024].

```python
from mlb_pipeline.backtest import run_backtest
from mlb_pipeline.models import ensemble

all_seasons = [s[0] for s in seasons]
train_seasons = all_seasons[:-1]
test_season   = all_seasons[-1]

def predict_fn(con, game_pk, home_id, away_id, as_of_date):
    return ensemble.predict(model, con, game_pk, home_id, away_id, as_of_date)

result = run_backtest(con, predict_fn, train_seasons, test_season)
```

### 4. Extract feature importances

```python
# model is a CalibratedClassifierCV wrapping XGBoost — get importances
try:
    base_clf = model.estimators_[0].estimator  # calibrated → base XGB
    raw_imp = base_clf.feature_importances_
    importances = dict(zip(ensemble.FEATURE_NAMES, raw_imp.tolist()))
except Exception:
    importances = {}
```

### 5. Find worst predictions (top 20 by squared error)

```python
preds_sorted = sorted(
    result.predictions,
    key=lambda p: (p["home_win_prob"] - (1.0 if p["actual_home_win"] else 0.0))**2,
    reverse=True
)
worst = [
    {**p, "sq_error": (p["home_win_prob"] - (1.0 if p["actual_home_win"] else 0.0))**2}
    for p in preds_sorted[:20]
]
```

### 6. Load previous brier score if available

```python
import glob

prev_brier = None
reports = sorted(glob.glob("tasks/reports/backtest_*.json"))
if reports:
    prev = json.loads(open(reports[-1]).read())
    prev_brier = prev.get("brier_score")
```

### 7. Compute calibration max error

```python
cal_errors = [
    abs(v["predicted_prob"] - v["actual_win_rate"])
    for v in result.calibration.values()
    if v["predicted_prob"] is not None and v["actual_win_rate"] is not None
]
cal_max_error = max(cal_errors) if cal_errors else None
```

### 8. Write the report

```python
ITER = int(open("tasks/reports/.current_iter").read().strip()) if Path("tasks/reports/.current_iter").exists() else 1

report = {
    "iteration": ITER,
    "brier_score": result.brier_score,
    "log_loss": result.log_loss,
    "roi_flat": result.roi_flat_bet,
    "roi_kelly": result.roi_kelly,
    "n_games": result.n_games,
    "bets_placed": result.bets_placed,
    "train_seasons": result.train_seasons,
    "test_seasons": [result.test_season],
    "calibration": result.calibration,
    "calibration_max_error": cal_max_error,
    "worst_predictions": worst,
    "feature_importances": importances,
    "previous_brier": prev_brier,
    "delta_brier": (prev_brier - result.brier_score) if prev_brier else None,
    "model_path": str(model_path),
}

Path(f"tasks/reports/backtest_{ITER}.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8"
)
print(f"Brier: {result.brier_score:.4f}  LogLoss: {result.log_loss:.4f}  "
      f"Kelly ROI: {result.roi_kelly:.3f}  Games: {result.n_games}")
```

### 9. Critical checks — gate if any fail

```python
import math, sys

# Gate 1: NaN predictions
nan_preds = [p for p in result.predictions if math.isnan(p["home_win_prob"])]
if nan_preds:
    # fire gate-create and stop
    sys.exit("NaN predictions detected — see gate")

# Gate 2: Brier score worse than coin-flip
if result.brier_score > 0.260:
    # fire gate-create and stop
    sys.exit("Brier score > 0.260 — model has degraded below coin-flip")
```

For either gate, use:
```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "Critical model issue in iteration $ITER: <describe>. Brier=$BRIER. Inspect tasks/reports/backtest_$ITER.json and choose how to proceed." \
  --options '["Stop the loop and investigate", "Reset model to last good checkpoint", "Continue despite degradation"]' \
  --json
```

### 10. Signal done

```sh
orca orchestration send \
  --type worker_done \
  --outcome succeeded \
  --task-id $TASK_ID \
  --dispatch-id $DISPATCH_ID \
  --subject "iter-backtest-$ITER complete" \
  --body "Brier=$BRIER, LogLoss=$LL, KellyROI=$KROI, n_games=$N. Report: tasks/reports/backtest_$ITER.json" \
  --files-modified "tasks/reports/backtest_$ITER.json" \
  --json
```
