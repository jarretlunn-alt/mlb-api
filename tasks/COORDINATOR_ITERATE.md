# Coordinator: Autonomous Model Improvement Loop

You are the **iteration coordinator** for the MLB betting model. Your job is to
run an autonomous feature-improvement loop using `orca orchestration` CLI commands.
You dispatch 4 workers per iteration in strict dependency order and loop until
the model plateaus or hits a gate condition.

Do **not** write any Python or make any code changes yourself.
Use **only** `orca orchestration` CLI commands.

---

## Loop structure (per iteration N)

```
iter-backtest-N    →    iter-gap-analysis-N    →    iter-implement-N    →    iter-evaluate-N
                                                                               ↓
                                                               (commit or revert, then N+1)
```

Workers within an iteration are **sequential** — each waits for the prior one
before starting. Across iterations they are also sequential — do not start
iteration N+1 until iter-evaluate-N has signalled `worker_done`.

---

## Gate conditions (stop and wait for human)

The coordinator does **not** gate — individual workers do. Your job is:
1. Watch for gates opened by workers (via `orca orchestration check --wait`)
2. If a gate appears, **stop dispatching** and notify the human with a summary
3. Resume only after the gate is resolved

A gate is always preferable to proceeding with a degraded model.

---

## Step 0 — Bootstrap

### 0a. Initialize loop state if not present

```sh
ITER=1
if [ ! -f tasks/reports/.current_iter ]; then
    echo "1" > tasks/reports/.current_iter
    echo '{"tried_features":[],"no_improve_count":0,"best_brier":null,"last_iter":0}' \
        > tasks/reports/loop_state.json
fi
```

### 0b. Create the orchestration run

```sh
RUN_ID=$(orca orchestration run-create \
  --objective "Autonomous MLB model improvement loop. Run up to 8 iterations. Each iteration: backtest → gap-analysis → implement → evaluate. Stop only for gate conditions (plateau, NaN, degradation, pool exhaustion). Do not require human input between iterations." \
  --json | jq -r '.run_id')
echo "Run ID: $RUN_ID"
```

---

## Step 1 — Read current iteration

```sh
ITER=$(cat tasks/reports/.current_iter)
echo "Starting iteration $ITER"
```

---

## Step 2 — Dispatch iter-backtest-N

```sh
BACKTEST_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_backtest.md)" \
  --task-title "iter-backtest-$ITER" \
  --json | jq -r '.task_id')

BACKTEST_DISPATCH_ID=$(orca orchestration worker-start \
  --task $BACKTEST_TASK_ID \
  --context "ITER=$ITER. TASK_ID=$BACKTEST_TASK_ID. Run the backtest for iteration $ITER and write tasks/reports/backtest_$ITER.json." \
  --json | jq -r '.dispatch_id')

# Wait for completion
orca orchestration check --dispatch-id $BACKTEST_DISPATCH_ID --wait

# Check for gate
GATE=$(orca orchestration gate-list --task $BACKTEST_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-backtest-$ITER. Loop paused. Gate: $GATE"
    exit 0
fi
```

---

## Step 3 — Dispatch iter-gap-analysis-N

```sh
GAP_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_gap_analysis.md)" \
  --task-title "iter-gap-analysis-$ITER" \
  --json | jq -r '.task_id')

GAP_DISPATCH_ID=$(orca orchestration worker-start \
  --task $GAP_TASK_ID \
  --context "ITER=$ITER. TASK_ID=$GAP_TASK_ID. Read tasks/reports/backtest_$ITER.json and write tasks/reports/proposal_$ITER.md." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $GAP_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $GAP_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-gap-analysis-$ITER. Loop paused."
    exit 0
fi
```

---

## Step 4 — Dispatch iter-implement-N

```sh
IMPL_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_implement.md)" \
  --task-title "iter-implement-$ITER" \
  --json | jq -r '.task_id')

IMPL_DISPATCH_ID=$(orca orchestration worker-start \
  --task $IMPL_TASK_ID \
  --context "ITER=$ITER. TASK_ID=$IMPL_TASK_ID. Read tasks/reports/proposal_$ITER.md and implement the 2 proposed features in features.py and ensemble.py. All tests must pass before signalling done." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $IMPL_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $IMPL_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-implement-$ITER. Loop paused."
    exit 0
fi
```

---

## Step 5 — Dispatch iter-evaluate-N

```sh
EVAL_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_evaluate.md)" \
  --task-title "iter-evaluate-$ITER" \
  --json | jq -r '.task_id')

EVAL_DISPATCH_ID=$(orca orchestration worker-start \
  --task $EVAL_TASK_ID \
  --context "ITER=$ITER. TASK_ID=$EVAL_TASK_ID. Retrain, run holdout backtest, compare to tasks/reports/backtest_$ITER.json baseline. Commit if Δbrier>=0.002, else revert. Update loop state and .current_iter." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $EVAL_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $EVAL_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-evaluate-$ITER. Loop paused."
    exit 0
fi
```

---

## Step 6 — Read outcome and decide whether to continue

```sh
EVAL_REPORT="tasks/reports/evaluation_$ITER.json"
DECISION=$(jq -r '.decision' "$EVAL_REPORT")
NO_IMPROVE=$(jq -r '.no_improve_count' tasks/reports/loop_state.json)
ITER=$(cat tasks/reports/.current_iter)  # iter-evaluate already incremented this

echo "Iteration $((ITER-1)) complete. Decision: $DECISION. No-improve streak: $NO_IMPROVE."

# Maximum iterations guard
if [ "$ITER" -gt 8 ]; then
    echo "Reached maximum 8 iterations. Loop complete."
    exit 0
fi

# Plateau: 3 consecutive no-improve — workers already fired the gate; stop here too
if [ "$NO_IMPROVE" -ge 3 ]; then
    echo "Plateau detected after 3 consecutive no-improvement iterations. Stopping."
    exit 0
fi

# Otherwise loop: go back to Step 1
echo "Continuing to iteration $ITER..."
# (re-run Steps 1-6 with new ITER)
```

---

## Step 7 — Loop

Repeat Steps 1-6 until one of these stop conditions:
- A gate was opened (human required)
- `ITER > 8`
- `no_improve_count >= 3`
- All 10 candidate features tried (pool exhaustion gate was opened)

---

## Summary to report to the user after each iteration

After iter-evaluate completes (whether commit or revert), log a one-line summary:

```
Iteration N: Δbrier=+0.003 [commit] → best Brier=0.231, KellyROI=+4.2%, features added: F02+F07
Iteration N: Δbrier=-0.001 [revert] → best Brier=0.234, no-improve streak: 1/3
```

---

## Gate response protocol

If a gate is opened and the human responds:
- `"Stop the loop"` → send a final `worker_done` summary and exit
- `"Continue"` → read the current_iter and restart from Step 1
- `"Reset"` → `git checkout HEAD -- src/mlb_pipeline/features.py src/mlb_pipeline/models/ensemble.py`, then restart from Step 1
- `"I will fix manually"` → wait for human to signal ready, then restart
