"""Unified prediction interface shared by every model in ``mlb_pipeline.models``.

Each model produces a :class:`GamePrediction`; :func:`save_prediction` upserts it
into ``fact_prediction`` keyed on ``prediction_id = "{model_name}_{game_pk}"`` so
re-running a model for the same game replaces the earlier row (idempotent).

No network I/O here; the only side effect is writing to the DuckDB connection
passed in by the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mlb_pipeline import db

# Mirrors the ``fact_prediction`` DDL from Task 01a (sql/schema.sql is the source
# of truth once that task lands). Kept here so prediction code works on a
# warehouse whose schema predates 01a; ``IF NOT EXISTS`` makes it a no-op after.
FACT_PREDICTION_TABLE = "fact_prediction"
FACT_PREDICTION_DDL = """
CREATE TABLE IF NOT EXISTS fact_prediction (
    prediction_id  VARCHAR PRIMARY KEY,
    game_pk        BIGINT,
    model_name     VARCHAR,
    predicted_at   TIMESTAMP,
    home_win_prob  DOUBLE,
    away_win_prob  DOUBLE,
    pred_total     DOUBLE,
    features_json  JSON
);
"""


@dataclass
class GamePrediction:
    """One model's pre-game forecast for a single game."""

    game_pk: int
    model_name: str
    home_win_prob: float
    away_win_prob: float
    pred_total: float | None = None
    features: dict = field(default_factory=dict)

    @property
    def prediction_id(self) -> str:
        return prediction_id(self.model_name, self.game_pk)

    def to_row(self, predicted_at: datetime | None = None) -> dict:
        """Row dict whose keys match the ``fact_prediction`` columns."""
        return {
            "prediction_id": self.prediction_id,
            "game_pk": int(self.game_pk),
            "model_name": self.model_name,
            "predicted_at": predicted_at or _utcnow(),
            "home_win_prob": float(self.home_win_prob),
            "away_win_prob": float(self.away_win_prob),
            "pred_total": None if self.pred_total is None else float(self.pred_total),
            "features_json": json.dumps(self.features or {}, sort_keys=True, default=str),
        }

    @classmethod
    def from_row(cls, row: dict) -> "GamePrediction":
        features = row.get("features_json")
        if isinstance(features, str):
            features = json.loads(features) if features else {}
        return cls(
            game_pk=int(row["game_pk"]),
            model_name=row["model_name"],
            home_win_prob=float(row["home_win_prob"]),
            away_win_prob=float(row["away_win_prob"]),
            pred_total=None if row.get("pred_total") is None else float(row["pred_total"]),
            features=features or {},
        )


def prediction_id(model_name: str, game_pk: int) -> str:
    return f"{model_name}_{game_pk}"


def _utcnow() -> datetime:
    # fact_prediction.predicted_at is a naive TIMESTAMP; store UTC without tzinfo.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ensure_prediction_table(con) -> None:
    """Create ``fact_prediction`` if the connected warehouse predates Task 01a."""
    con.execute(FACT_PREDICTION_DDL)


def save_prediction(con, pred: GamePrediction, predicted_at: datetime | None = None) -> None:
    """Upsert into fact_prediction (INSERT OR REPLACE on prediction_id)."""
    ensure_prediction_table(con)
    db.upsert(con, FACT_PREDICTION_TABLE, [pred.to_row(predicted_at)])


def save_predictions(
    con, preds: list[GamePrediction], predicted_at: datetime | None = None
) -> int:
    """Upsert many predictions in one statement. Returns the number of rows written."""
    if not preds:
        return 0
    ensure_prediction_table(con)
    stamp = predicted_at or _utcnow()
    return db.upsert(con, FACT_PREDICTION_TABLE, [p.to_row(stamp) for p in preds])


def load_predictions(
    con,
    game_pk: int | None = None,
    model_name: str | None = None,
) -> list[GamePrediction]:
    """Read stored predictions, optionally filtered by game and/or model."""
    ensure_prediction_table(con)
    clauses, params = [], []
    if game_pk is not None:
        clauses.append("game_pk = ?")
        params.append(int(game_pk))
    if model_name is not None:
        clauses.append("model_name = ?")
        params.append(model_name)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cur = con.execute(
        f"SELECT game_pk, model_name, home_win_prob, away_win_prob, pred_total, "
        f"CAST(features_json AS VARCHAR) AS features_json "
        f"FROM {FACT_PREDICTION_TABLE} {where} ORDER BY game_pk, model_name",
        params,
    )
    columns = [d[0] for d in cur.description]
    return [GamePrediction.from_row(dict(zip(columns, row))) for row in cur.fetchall()]
