-- Public table schemas for the MLB pipeline warehouse.
-- Invariant: do not change these without updating tests (see CLAUDE.md).
-- All loads are idempotent: every table has a primary key and loads use
-- INSERT OR REPLACE keyed on it.

CREATE TABLE IF NOT EXISTS dim_team (
    team_id       INTEGER PRIMARY KEY,
    name          VARCHAR,
    abbreviation  VARCHAR,
    team_code     VARCHAR,
    league        VARCHAR,
    division      VARCHAR,
    venue         VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_player (
    player_id         INTEGER PRIMARY KEY,
    full_name         VARCHAR,
    primary_position  VARCHAR,
    bat_side          VARCHAR,
    pitch_hand        VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_game (
    game_pk          BIGINT PRIMARY KEY,
    official_date    DATE,
    season           INTEGER,
    game_type        VARCHAR,
    status           VARCHAR,
    venue            VARCHAR,
    home_team_id     INTEGER,
    away_team_id     INTEGER,
    home_score       INTEGER,
    away_score       INTEGER,
    winning_team_id  INTEGER
);

CREATE TABLE IF NOT EXISTS fact_team_game (
    game_pk           BIGINT,
    team_id           INTEGER,
    opponent_team_id  INTEGER,
    is_home           BOOLEAN,
    runs_scored       INTEGER,
    runs_allowed      INTEGER,
    hits              INTEGER,
    errors            INTEGER,
    win               BOOLEAN,
    PRIMARY KEY (game_pk, team_id)
);

CREATE TABLE IF NOT EXISTS fact_player_game_batting (
    game_pk       BIGINT,
    player_id     INTEGER,
    team_id       INTEGER,
    at_bats       INTEGER,
    runs          INTEGER,
    hits          INTEGER,
    doubles       INTEGER,
    triples       INTEGER,
    home_runs     INTEGER,
    rbi           INTEGER,
    walks         INTEGER,
    strikeouts    INTEGER,
    stolen_bases  INTEGER,
    PRIMARY KEY (game_pk, player_id)
);

CREATE TABLE IF NOT EXISTS fact_player_game_pitching (
    game_pk          BIGINT,
    player_id        INTEGER,
    team_id          INTEGER,
    innings_pitched  VARCHAR,
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

-- ---------------------------------------------------------------------------
-- Analytics layer (Task 01a). Additive only: the tables above are unchanged.
-- ---------------------------------------------------------------------------

-- One row per pitcher appearance, derived from fact_player_game_pitching.
-- is_starter / rest_days are computed by features.refresh_pitcher_log().
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

-- Materialized rolling team stats as of a date (games strictly before as_of_date).
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

-- Static park factors, seeded by features.seed_park_factors().
CREATE TABLE IF NOT EXISTS dim_park (
    park_id        INTEGER PRIMARY KEY,
    name           VARCHAR,
    team_id        INTEGER,
    run_factor     DOUBLE,   -- 1.0 = neutral; >1.0 = hitter-friendly
    hr_factor      DOUBLE,
    handedness     VARCHAR   -- 'neutral', 'rhb', 'lhb'
);

-- Model outputs, one row per (model, game).
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
