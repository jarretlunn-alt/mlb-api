PYTHON ?= python

.PHONY: setup test ingest-date build-marts dashboard restore

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# Usage: make ingest-date DATE=2026-07-03 [END_DATE=2026-07-05]
ingest-date:
	$(PYTHON) -m mlb_pipeline.cli ingest --start-date $(DATE) --end-date $(if $(END_DATE),$(END_DATE),$(DATE))

build-marts:
	$(PYTHON) -m mlb_pipeline.cli build-marts

dashboard:
	$(PYTHON) -m mlb_pipeline.cli dashboard

# Rebuild the local DuckDB warehouse from committed Parquet exports
restore:
	$(PYTHON) -m mlb_pipeline.cli restore
