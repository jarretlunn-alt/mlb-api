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
