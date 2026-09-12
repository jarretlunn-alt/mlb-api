"""Feature engineering tests. In-memory DuckDB populated from fixtures only."""

import datetime as dt

import duckdb
import pytest

from mlb_pipeline import db, features, ingest

NEW_TABLES = ["fact_pitcher_log", "fact_team_rolling", "dim_park", "fact_prediction"]

GAME_PK = 700001
GAME_DATE = "2026-07-01"
ATHLETICS, MARINERS = 133, 136
LEFTY_STARTER, RIGHTY_RELIEVER = 660002, 660004


@pytest.fixture
def mem():
    con = duckdb.connect(":memory:")
    db.init_schema(con)
    yield con
    con.close()


@pytest.fixture
def loaded(mem, feed_live, boxscore):
    """In-memory warehouse holding the single fixture game (2026-07-01)."""
    ingest.load_game(mem, feed_live, boxscore)
    return mem


def add_history(con, game_pk: int, date: str, home_id: int, away_id: int, home_score: int, away_score: int, pitching: list[dict]):
    """Insert a synthetic prior game so pitchers/teams have history.

    `pitching` rows: dicts with player_id, team_id, outs, earned_runs, walks,
    strikeouts, home_runs, pitches (hits/runs default to 0).
    """
    db.upsert(
        con,
        "fact_game",
        [
            {
                "game_pk": game_pk,
                "official_date": date,
                "season": 2026,
                "game_type": "R",
                "status": "Final",
                "venue": "Synthetic Park",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_score": home_score,
                "away_score": away_score,
                "winning_team_id": home_id if home_score > away_score else away_id,
            }
        ],
    )
    db.upsert(
        con,
        "fact_team_game",
        [
            {
                "game_pk": game_pk,
                "team_id": home_id,
                "opponent_team_id": away_id,
                "is_home": True,
                "runs_scored": home_score,
                "runs_allowed": away_score,
                "hits": 0,
                "errors": 0,
                "win": home_score > away_score,
            },
            {
                "game_pk": game_pk,
                "team_id": away_id,
                "opponent_team_id": home_id,
                "is_home": False,
                "runs_scored": away_score,
                "runs_allowed": home_score,
                "hits": 0,
                "errors": 0,
                "win": away_score > home_score,
            },
        ],
    )
    rows = []
    for p in pitching:
        rows.append(
            {
                "game_pk": game_pk,
                "player_id": p["player_id"],
                "team_id": p["team_id"],
                "innings_pitched": f"{p['outs'] // 3}.{p['outs'] % 3}",
                "outs": p["outs"],
                "hits": p.get("hits", 0),
                "runs": p.get("runs", p["earned_runs"]),
                "earned_runs": p["earned_runs"],
                "walks": p["walks"],
                "strikeouts": p["strikeouts"],
                "home_runs": p["home_runs"],
                "pitches": p["pitches"],
            }
        )
    db.upsert(con, "fact_player_game_pitching", rows)


def start(player_id, team_id, outs=18, er=2, bb=1, k=6, hr=1, pitches=90):
    return {"player_id": player_id, "team_id": team_id, "outs": outs, "earned_runs": er, "walks": bb, "strikeouts": k, "home_runs": hr, "pitches": pitches}


@pytest.fixture
def with_history(loaded):
    """Fixture game plus two earlier games per team so both starters have >= 2 prior starts."""
    # 2026-06-20: ATH @ SEA; Lefty starts for ATH, Righty starts for SEA
    add_history(loaded, 690001, "2026-06-20", MARINERS, ATHLETICS, 4, 2,
                [start(LEFTY_STARTER, ATHLETICS, outs=15, er=3, k=4),
                 start(RIGHTY_RELIEVER, MARINERS, outs=21, er=1, k=8, hr=0),
                 # a reliever so SEA has bullpen innings
                 {"player_id": 660099, "team_id": MARINERS, "outs": 6, "earned_runs": 1, "walks": 1, "strikeouts": 2, "home_runs": 0, "pitches": 25}])
    # 2026-06-25: SEA @ ATH
    add_history(loaded, 690002, "2026-06-25", ATHLETICS, MARINERS, 1, 6,
                [start(LEFTY_STARTER, ATHLETICS, outs=18, er=2, k=7),
                 start(RIGHTY_RELIEVER, MARINERS, outs=18, er=1, k=6, hr=0)])
    return loaded


# --- schema / registry ------------------------------------------------------


def test_schema_has_new_tables(mem):
    tables = {r[0] for r in mem.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    for table in NEW_TABLES:
        assert table in tables


def test_db_tables_registry_includes_new_tables():
    for table in NEW_TABLES:
        assert table in db.TABLES


# --- park factors -------------------------------------------------------------


def test_seed_park_factors_is_idempotent_and_covers_30_teams(mem):
    assert len(features.PARK_FACTORS) == 30
    assert features.seed_park_factors(mem) == 30
    features.seed_park_factors(mem)
    assert mem.execute("SELECT count(*) FROM dim_park").fetchone()[0] == 30
    park = mem.execute("SELECT park_id, name FROM dim_park WHERE team_id = ?", [MARINERS]).fetchone()
    assert park == (680, "T-Mobile Park")


# --- pitcher log / pitcher features -----------------------------------------


def test_refresh_pitcher_log_derives_starters_and_rest(with_history):
    assert features.refresh_pitcher_log(with_history) == 7
    rows = with_history.execute(
        "SELECT game_pk, player_id, is_starter, rest_days FROM fact_pitcher_log ORDER BY official_date, game_pk, player_id"
    ).fetchall()
    by_key = {(g, p): (s, r) for g, p, s, r in rows}
    # First appearance: no rest_days; reliever with fewer outs is not the starter
    assert by_key[(690001, LEFTY_STARTER)] == (True, None)
    assert by_key[(690001, 660099)] == (False, None)
    # 2026-06-20 -> 2026-06-25 -> 2026-07-01
    assert by_key[(690002, LEFTY_STARTER)] == (True, 5)
    assert by_key[(GAME_PK, LEFTY_STARTER)] == (True, 6)
    assert by_key[(GAME_PK, RIGHTY_RELIEVER)] == (True, 6)
    # Re-running does not duplicate
    features.refresh_pitcher_log(with_history)
    assert with_history.execute("SELECT count(*) FROM fact_pitcher_log").fetchone()[0] == 7


def test_pitcher_features_returns_none_with_fewer_than_two_starts(loaded):
    # Only the fixture game exists: one start, and it's on as_of_date (excluded anyway)
    assert features.pitcher_features(loaded, LEFTY_STARTER, GAME_DATE) is None
    assert features.pitcher_features(loaded, LEFTY_STARTER, "2026-07-02") is None
    assert features.pitcher_features(loaded, 999999, "2026-07-02") is None


def test_pitcher_features_excludes_as_of_date_and_aggregates(with_history):
    feats = features.pitcher_features(with_history, LEFTY_STARTER, GAME_DATE)
    assert feats is not None
    assert set(feats) == set(features.PITCHER_FEATURE_KEYS)
    # Two prior starts: 15 + 18 outs = 11 IP, 5 ER, 2 BB, 11 K, 2 HR
    assert feats["n_starts"] == 2
    assert feats["era_last_n"] == pytest.approx(9 * 5 / 11)
    assert feats["k_per_9"] == pytest.approx(9 * 11 / 11)
    assert feats["bb_per_9"] == pytest.approx(9 * 2 / 11)
    assert feats["hr_per_9"] == pytest.approx(9 * 2 / 11)
    assert feats["fip_last_n"] == pytest.approx((13 * 2 + 3 * 2 - 2 * 11) / 11 + features.FIP_CONSTANT)
    assert feats["avg_rest_days"] == 5  # only the second start has a prior appearance

    # Moving as_of forward includes the fixture game as a third start
    later = features.pitcher_features(with_history, LEFTY_STARTER, "2026-07-02")
    assert later["n_starts"] == 3
    assert features.pitcher_features(with_history, LEFTY_STARTER, "2026-07-02", n_starts=2)["n_starts"] == 2


# --- team rolling -------------------------------------------------------------


def test_team_rolling_features_window_is_exclusive_of_as_of_date(loaded):
    assert features.team_rolling_features(loaded, MARINERS, GAME_DATE) is None
    feats = features.team_rolling_features(loaded, MARINERS, "2026-07-02", window=15)
    assert feats is not None
    assert set(feats) == set(features.TEAM_FEATURE_KEYS)
    assert feats["games"] == 1
    assert feats["runs_per_game"] == 5.0
    assert feats["runs_allowed_per_game"] == 3.0
    # Mariners faced the left-handed Athletics starter
    assert feats["ops_vs_lhp"] is not None
    assert feats["ops_vs_rhp"] is None
    # Mariners' only pitcher is their derived starter, so no bullpen innings
    assert feats["bullpen_fip"] is None
    # A window too short to reach the game finds nothing
    assert features.team_rolling_features(loaded, MARINERS, "2026-07-20", window=7) is None


def test_refresh_team_rolling_materializes_rows(with_history):
    n = features.refresh_team_rolling(with_history, GAME_DATE, windows=(7, 15, 30))
    # Both teams have games in every window: 06-25 is 6 days before 07-01, so even the 7-day window hits
    rows = with_history.execute(
        "SELECT team_id, window_days, runs_scored, fip FROM fact_team_rolling ORDER BY team_id, window_days"
    ).fetchall()
    assert n == len(rows) == 6
    by_key = {(t, w): (rs, fip) for t, w, rs, fip in rows}
    assert by_key[(MARINERS, 15)][0] == pytest.approx((4 + 6) / 2)
    assert by_key[(MARINERS, 7)][0] == pytest.approx(6.0)
    assert by_key[(MARINERS, 15)][1] is not None  # reliever 660099 pitched in window
    # Idempotent
    features.refresh_team_rolling(with_history, GAME_DATE, windows=(7, 15, 30))
    assert with_history.execute("SELECT count(*) FROM fact_team_rolling").fetchone()[0] == 6


# --- game context / full row ---------------------------------------------------


def test_game_context_features(loaded):
    features.seed_park_factors(loaded)
    ctx = features.game_context_features(loaded, GAME_PK)
    assert ctx["home_team_id"] == MARINERS
    assert ctx["away_team_id"] == ATHLETICS
    assert ctx["park_id"] == 680
    assert ctx["park_run_factor"] == features.PARK_FACTORS[MARINERS]["run_factor"]
    assert ctx["park_hr_factor"] == features.PARK_FACTORS[MARINERS]["hr_factor"]
    assert ctx["home_is_home"] is True and ctx["away_is_home"] is False
    assert features.game_context_features(loaded, 123) is None


def test_game_context_defaults_to_neutral_park_when_unseeded(loaded):
    ctx = features.game_context_features(loaded, GAME_PK)
    assert ctx["park_run_factor"] == 1.0
    assert ctx["park_hr_factor"] == 1.0
    assert ctx["park_handedness"] == "neutral"


def test_build_game_feature_row_returns_none_without_history(loaded):
    assert features.build_game_feature_row(loaded, GAME_PK, GAME_DATE) is None
    assert features.build_game_feature_row(loaded, 123456, GAME_DATE) is None


def test_build_game_feature_row_has_expected_keys(with_history):
    features.seed_park_factors(with_history)
    row = features.build_game_feature_row(with_history, GAME_PK, GAME_DATE)
    assert row is not None
    expected = set(features.feature_row_keys()) | set(features.LABEL_COLUMNS)
    assert set(row) == expected

    assert row["home_team_id"] == MARINERS
    assert row["away_team_id"] == ATHLETICS
    assert row["home_pitcher_id"] == RIGHTY_RELIEVER
    assert row["away_pitcher_id"] == LEFTY_STARTER
    assert row["as_of_date"] == dt.date(2026, 7, 1)
    assert row["park_run_factor"] == features.PARK_FACTORS[MARINERS]["run_factor"]
    assert row["home_sp_n_starts"] == 2 and row["away_sp_n_starts"] == 2
    assert row["home_games"] == 2 and row["away_games"] == 2
    # Labels come from the game itself, never from the feature window
    assert (row["home_score"], row["away_score"], row["total_runs"], row["home_win"]) == (5, 3, 8, True)


def test_build_game_feature_row_accepts_explicit_probable_pitchers(with_history):
    # Pretend the game is unplayed: pass probable pitchers explicitly, swapped on purpose
    row = features.build_game_feature_row(
        with_history, GAME_PK, GAME_DATE, home_pitcher_id=LEFTY_STARTER, away_pitcher_id=RIGHTY_RELIEVER
    )
    assert row["home_pitcher_id"] == LEFTY_STARTER
    assert row["away_pitcher_id"] == RIGHTY_RELIEVER
