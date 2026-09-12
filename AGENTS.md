# MLB Pipeline — Agent Instructions

All agents working in this repo must read this file before writing any code.

## Project overview

Python data pipeline: MLB Stats API → raw JSON.gz → DuckDB → Parquet → static HTML dashboard.
The analytics layer (in-progress) adds betting odds projection models: moneyline, run line, O/U.

## Stack

- Python 3.12, DuckDB, Parquet, pytest, scikit-learn, XGBoost
- No paid services. No API keys required for MLB public API.
- Tests use `tests/fixtures/` only — no live network calls in tests.

## Repo layout

```
src/mlb_pipeline/
  api_client.py     # ALL network calls go here — do not call requests elsewhere
  config.py         # Settings dataclass (data_dir, site_dir)
  db.py             # DuckDB connect, schema init, upsert()
  raw_store.py      # Save/load raw responses as .json.gz
  normalize.py      # Pure JSON → warehouse row dicts
  ingest.py         # Orchestration: fetch → save raw → normalize → upsert
  marts.py          # Mart views, Parquet export, restore_from_parquet
  dashboard.py      # Static HTML dashboard
  cli.py            # CLI entry point
  models/           # One file per model — created by analytics tasks
  features.py       # Shared feature engineering — created by Task 01a
  backtest.py       # Walk-forward backtesting — created by Task 02c
  predict.py        # Unified prediction interface — created by Task 02a

sql/schema.sql      # Table schemas — source of truth
tests/
  fixtures/         # Static JSON for unit tests
  conftest.py       # Shared fixtures and FakeClient
tasks/              # Per-task specs (read-only for worker agents)
```

## Invariants — never break these

1. All network calls go through `src/mlb_pipeline/api_client.py` only.
2. Raw API responses are saved before transformation (raw_store.py).
3. All DB loads are idempotent (INSERT OR REPLACE on primary key).
4. No live network calls inside unit tests — use tests/fixtures/.
5. Do not modify `sql/schema.sql` unless your task spec explicitly says to.
6. Do not modify `tests/conftest.py` unless your task spec explicitly says to.
7. `make test` must pass before signalling completion.

## Commands

```sh
make setup          # pip install -e ".[dev]"
make test           # pytest
make ingest-date DATE=2026-07-02
make build-marts
make dashboard
make restore        # rebuild warehouse from data/marts/*.parquet
```

On Windows without make:
```powershell
python -m pytest
python -m mlb_pipeline.cli ingest --start-date 2026-07-02
```

## Completing your task

When your task is done and `make test` passes:

```sh
orca orchestration send \
  --type worker_done \
  --outcome succeeded \
  --task-id $TASK_ID \
  --dispatch-id $DISPATCH_ID \
  --subject "Task <title> complete" \
  --body "Brief summary of what was built and any decisions made." \
  --files-modified "<comma-separated list>" \
  --json
```

If you encounter a blocker that requires human input:
```sh
orca orchestration gate-create \
  --task $TASK_ID \
  --question "Describe the decision needed" \
  --options '["Option A", "Option B"]' \
  --json
```

If your task fails unrecoverably:
```sh
orca orchestration send \
  --type worker_done \
  --outcome failed \
  --task-id $TASK_ID \
  --dispatch-id $DISPATCH_ID \
  --subject "Task <title> failed" \
  --body "What went wrong and why." \
  --json
```
