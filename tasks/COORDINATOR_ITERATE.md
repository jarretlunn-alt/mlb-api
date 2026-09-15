# Coordinator: Autonomous Model Improvement Loop

You are the **iteration coordinator** for the MLB betting model. Your job is to
run an autonomous feature-improvement loop using `orca orchestration` CLI commands.
You dispatch 4 workers per iteration in strict dependency order and loop until
the model plateaus or hits a gate condition.

Do **not** write any Python or make any code changes yourself.
Use **only** `orca orchestration` CLI commands.

---

## Model assignment strategy

Each worker phase is assigned to the LLM best suited to it. Alternating the
gap-analysis worker between Claude and Codex across iterations deliberately
generates different analytical perspectives and feature hypotheses.

| Phase | Model flag | Effort | Reason |
|-------|-----------|--------|--------|
| `iter-backtest` | `--model claude` | high | Heavy SQL + metric reasoning; calibration analysis |
| `iter-gap-analysis` | Odd → `--model codex`, Even → `--model claude` | medium | Rotates perspective; Codex hunts code/pattern signals, Claude reasons statistically |
| `iter-implement` | `--model codex` | medium | Precise Python generation; fewer hallucinated function names |
| `iter-evaluate` | `--model claude` | high | Decision logic, nuanced comparison, commit messages |

`--effort` is supported by Orca but valid level names are not documented in the
CLI help. Check `orca skills get` or Orca's changelog for the accepted values
(likely `low / medium / high` or similar). Add `--effort <level>` to each
`worker-start` call once you confirm the accepted values.

**On local LLMs:** Orca's `--model` flag currently accepts only Claude, Codex,
and Cursor provider IDs — there is no native local-model slot. To route through
a local LLM (Ollama, LM Studio, vLLM):
1. Run a local inference server exposing an OpenAI-compatible API
   (e.g. `ollama serve`, `lms server start`)
2. Use `litellm` as a shim in front of it
3. Set `ANTHROPIC_BASE_URL=http://localhost:4000` before launching Orca so that
   Claude Code's HTTP client hits your proxy instead of Anthropic's servers
4. The `--model claude` flag still tells Orca to use a Claude-style agent, but
   actual inference runs on your GPU

Good local models for each phase (if running this way):
- **iter-backtest / iter-evaluate** (reasoning-heavy): `qwen2.5:32b`, `deepseek-r1:32b`, `llama3.1:70b`
- **iter-gap-analysis** (pattern analysis): `qwen2.5:14b`, `mistral:7b` for fast/cheap
- **iter-implement** (code generation): `qwen2.5-coder:32b` is the strongest local code model as of 2025

Note: local models that can't follow Claude's tool-use format reliably will cause
`worker_done` signal parsing to fail. Test with a simple task before running the
full 8-iteration loop.

---

## Token budget rules (applied by the coordinator)

These are hard limits to prevent workers from exhausting context:

- `backtest_N.json` — must be ≤ 50 KB. Coordinator truncates `worst_predictions`
  to the top 10 rows before signalling gap-analysis (see Step 2c).
- `proposal_N.md` — must be ≤ 600 words. Gap-analysis worker is told this explicitly.
- `iter-implement` context string — include ONLY the two feature IDs + column names
  from the proposal. Do not paste the full proposal text into the context.
- `iter-evaluate` context string — include ONLY the baseline Brier and the model path.

---

## Loop structure (per iteration N)

```
iter-backtest-N  →  iter-gap-analysis-N  →  iter-implement-N  →  iter-evaluate-N
   (claude)            (codex/claude)            (codex)              (claude)
                                                                          ↓
                                                          (commit or revert, then N+1)
```

Workers within an iteration are **sequential** — each waits for the prior one.
Do not start N+1 until iter-evaluate-N has signalled `worker_done`.

---

## Gate conditions (stop and wait for human)

The coordinator does **not** gate — individual workers do. Your job is:
1. Watch for gates opened by workers (`orca orchestration check --wait`)
2. If a gate appears, **stop dispatching** and surface a summary to the human
3. Resume only after the gate is resolved

---

## Step 0 — Bootstrap

### 0a. Initialize loop state if not present

```sh
if [ ! -f tasks/reports/.current_iter ]; then
    echo "1" > tasks/reports/.current_iter
    echo '{"tried_features":[],"no_improve_count":0,"best_brier":null,"last_iter":0}' \
        > tasks/reports/loop_state.json
fi
```

### 0b. Create the orchestration run

```sh
RUN_ID=$(orca orchestration run-create \
  --objective "Autonomous MLB model improvement loop. Up to 8 iterations. Each: backtest(claude) → gap-analysis(codex/claude alternating) → implement(codex) → evaluate(claude). Stop only for gate conditions. No human input needed between iterations." \
  --json | jq -r '.run_id')
echo "Run ID: $RUN_ID"
```

---

## Step 1 — Read current iteration

```sh
ITER=$(cat tasks/reports/.current_iter)
# Determine gap-analysis model: Codex on odd iterations, Claude on even
if [ $((ITER % 2)) -eq 1 ]; then GAP_MODEL="codex"; else GAP_MODEL="claude"; fi
echo "Starting iteration $ITER (gap-analysis model: $GAP_MODEL)"
```

---

## Step 2 — Dispatch iter-backtest-N (model: claude)

```sh
BACKTEST_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_backtest.md)" \
  --task-title "iter-backtest-$ITER" \
  --json | jq -r '.task_id')

BACKTEST_DISPATCH_ID=$(orca orchestration worker-start \
  --task $BACKTEST_TASK_ID \
  --model claude \
  --context "ITER=$ITER. TASK_ID=$BACKTEST_TASK_ID. Write tasks/reports/backtest_$ITER.json. Keep worst_predictions to top 10 rows only to limit file size." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $BACKTEST_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $BACKTEST_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-backtest-$ITER. Loop paused. Gate: $GATE"
    exit 0
fi
```

### 2c — Enforce token budget on the backtest report

After the worker completes, trim the report if it exceeds 50 KB:
```sh
python - <<'PY'
import json, sys
from pathlib import Path
ITER = int(open("tasks/reports/.current_iter").read())
p = Path(f"tasks/reports/backtest_{ITER}.json")
report = json.loads(p.read_text())
report["worst_predictions"] = report.get("worst_predictions", [])[:10]
# Drop full calibration if still large; keep only buckets with n > 5
if p.stat().st_size > 50_000:
    report["calibration"] = {
        k: v for k, v in report.get("calibration", {}).items()
        if v.get("n", 0) > 5
    }
p.write_text(json.dumps(report, indent=2, default=str))
print(f"Backtest report: {p.stat().st_size/1024:.1f} KB")
PY
```

---

## Step 3 — Dispatch iter-gap-analysis-N (model: alternates)

```sh
GAP_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_gap_analysis.md)" \
  --task-title "iter-gap-analysis-$ITER" \
  --json | jq -r '.task_id')

GAP_DISPATCH_ID=$(orca orchestration worker-start \
  --task $GAP_TASK_ID \
  --model $GAP_MODEL \
  --context "ITER=$ITER. TASK_ID=$GAP_TASK_ID. MODEL=$GAP_MODEL. Read tasks/reports/backtest_$ITER.json and tasks/reports/loop_state.json. Write tasks/reports/proposal_$ITER.md (max 600 words). Do NOT read the full codebase — only features.py FEATURE_NAMES and ensemble.py FEATURE_NAMES are needed to check what is already implemented." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $GAP_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $GAP_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-gap-analysis-$ITER. Loop paused."
    exit 0
fi
```

---

## Step 4 — Dispatch iter-implement-N (model: codex)

Extract just the feature IDs and column names from the proposal to keep context small:
```sh
# Pull the two "Column name in FEATURE_NAMES:" lines from the proposal
FEAT_SUMMARY=$(grep "Column name in FEATURE_NAMES" tasks/reports/proposal_$ITER.md \
    | sed 's/.*: //' | tr '\n' ',' | sed 's/,$//')
echo "Features to implement: $FEAT_SUMMARY"
```

```sh
IMPL_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_implement.md)" \
  --task-title "iter-implement-$ITER" \
  --json | jq -r '.task_id')

IMPL_DISPATCH_ID=$(orca orchestration worker-start \
  --task $IMPL_TASK_ID \
  --model codex \
  --context "ITER=$ITER. TASK_ID=$IMPL_TASK_ID. Features to add: $FEAT_SUMMARY. Full spec in tasks/reports/proposal_$ITER.md. Modify ONLY: src/mlb_pipeline/features.py, src/mlb_pipeline/models/ensemble.py. Create: tests/test_features_iter${ITER}.py. Run python -m pytest -q to verify all pass." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $IMPL_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $IMPL_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-implement-$ITER. Loop paused."
    exit 0
fi
```

---

## Step 5 — Dispatch iter-evaluate-N (model: claude)

```sh
BASELINE_BRIER=$(python -c "import json; print(json.load(open(f'tasks/reports/backtest_$ITER.json'))['brier_score'])")

EVAL_TASK_ID=$(orca orchestration task-create \
  --spec "$(cat tasks/iter_evaluate.md)" \
  --task-title "iter-evaluate-$ITER" \
  --json | jq -r '.task_id')

EVAL_DISPATCH_ID=$(orca orchestration worker-start \
  --task $EVAL_TASK_ID \
  --model claude \
  --context "ITER=$ITER. TASK_ID=$EVAL_TASK_ID. Baseline Brier=$BASELINE_BRIER. Retrain, holdout backtest, write tasks/reports/evaluation_$ITER.json. Commit if delta_brier>=0.002. Update tasks/reports/loop_state.json and tasks/reports/.current_iter." \
  --json | jq -r '.dispatch_id')

orca orchestration check --dispatch-id $EVAL_DISPATCH_ID --wait

GATE=$(orca orchestration gate-list --task $EVAL_TASK_ID --status open --json 2>/dev/null | jq -r '.gates[0].gate_id // empty')
if [ -n "$GATE" ]; then
    echo "GATE OPENED by iter-evaluate-$ITER. Loop paused."
    exit 0
fi
```

---

## Step 6 — Read outcome and log summary

```sh
EVAL_REPORT="tasks/reports/evaluation_$ITER.json"
DECISION=$(python -c "import json; print(json.load(open('$EVAL_REPORT'))['decision'])")
NEW_BRIER=$(python -c "import json; print(json.load(open('$EVAL_REPORT'))['new_brier'])")
DELTA=$(python -c "import json; print(json.load(open('$EVAL_REPORT'))['delta_brier'])")
NO_IMPROVE=$(python -c "import json; print(json.load(open('tasks/reports/loop_state.json'))['no_improve_count'])")
NEXT_ITER=$(cat tasks/reports/.current_iter)

echo "=== Iteration $ITER complete ==="
echo "Decision: $DECISION | Δbrier=$DELTA | New Brier=$NEW_BRIER | No-improve streak: $NO_IMPROVE/3"
echo "Gap model this iteration: $GAP_MODEL"

# Maximum iterations guard
if [ "$NEXT_ITER" -gt 8 ]; then
    echo "Reached maximum 8 iterations. Loop complete."
    exit 0
fi

# Plateau guard
if [ "$NO_IMPROVE" -ge 3 ]; then
    echo "Plateau: 3 consecutive no-improvement iterations. Stopping (gate should be open)."
    exit 0
fi

echo "Continuing to iteration $NEXT_ITER..."
# Re-run from Step 1 with ITER=$NEXT_ITER
```

---

## Step 7 — Loop

Repeat Steps 1-6 until one of:
- A gate was opened (human required)
- `NEXT_ITER > 8`
- `no_improve_count >= 3`
- Feature pool exhausted (gap-analysis opens a gate)

---

## Per-iteration log (append after each Step 6)

```
Iter 1 [codex gap]: Δbrier=+0.003 [commit] → Brier=0.231, KellyROI=+4.2%, features: F02+F07
Iter 2 [claude gap]: Δbrier=-0.001 [revert] → Brier=0.231, no-improve: 1/3
Iter 3 [codex gap]: Δbrier=+0.002 [commit] → Brier=0.229, KellyROI=+5.1%, features: F04+F09
```

---

## Gate response protocol

If a gate is opened and the human responds:
- `"Stop the loop"` → send a final summary and exit
- `"Continue"` → read `.current_iter` and restart from Step 1
- `"Reset"` → `git checkout HEAD -- src/mlb_pipeline/features.py src/mlb_pipeline/models/ensemble.py`, then restart from Step 1
- `"I will fix manually"` → wait for human to signal ready, then restart
- `"Use only Claude"` → set `GAP_MODEL=claude` and `IMPL_MODEL=claude` for all remaining iterations
- `"Use only Codex"` → set `GAP_MODEL=codex` and `IMPL_MODEL=codex` for all remaining iterations
