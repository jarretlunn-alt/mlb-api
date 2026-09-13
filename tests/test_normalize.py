from mlb_pipeline import normalize


def test_extract_game_pks_final_only(schedule):
    assert normalize.extract_game_pks(schedule) == [700001]


def test_extract_game_pks_all(schedule):
    assert normalize.extract_game_pks(schedule, only_final=False) == [700001, 700002]


def test_normalize_game(feed_live):
    game = normalize.normalize_game(feed_live)

    assert game == {
        "game_pk": 700001,
        "official_date": "2026-07-01",
        "season": 2026,
        "game_type": "R",
        "status": "Final",
        "venue": "T-Mobile Park",
        "home_team_id": 136,
        "away_team_id": 133,
        "home_score": 5,
        "away_score": 3,
        "winning_team_id": 136,
    }


def test_normalize_schedule_games(schedule):
    rows = normalize.normalize_schedule_games(schedule)

    assert rows == [
        {
            "game_pk": 700001,
            "official_date": "2026-07-01",
            "season": 2026,
            "game_type": "R",
            "status": "Scheduled",
            "home_team_id": 136,
            "away_team_id": 133,
        },
        {
            "game_pk": 700002,
            "official_date": "2026-07-01",
            "season": 2026,
            "game_type": "R",
            "status": "Scheduled",
            "home_team_id": 137,
            "away_team_id": 119,
        },
    ]


def test_normalize_schedule_teams(schedule):
    teams = normalize.normalize_schedule_teams(schedule)

    assert {t["team_id"]: t["name"] for t in teams} == {
        133: "Athletics",
        136: "Seattle Mariners",
        119: "Los Angeles Dodgers",
        137: "San Francisco Giants",
    }


def test_normalize_teams(feed_live):
    teams = normalize.normalize_teams(feed_live)

    assert {t["team_id"] for t in teams} == {133, 136}
    home = next(t for t in teams if t["team_id"] == 136)
    assert home["name"] == "Seattle Mariners"
    assert home["abbreviation"] == "SEA"
    assert home["division"] == "American League West"


def test_normalize_players(feed_live):
    players = normalize.normalize_players(feed_live)

    assert len(players) == 4
    catcher = next(p for p in players if p["player_id"] == 660003)
    assert catcher["full_name"] == "Slugging Catcher"
    assert catcher["primary_position"] == "C"
    assert catcher["bat_side"] == "R"


def test_normalize_team_games(boxscore):
    rows = normalize.normalize_team_games(700001, boxscore)

    assert len(rows) == 2
    home = next(r for r in rows if r["is_home"])
    away = next(r for r in rows if not r["is_home"])

    assert home["team_id"] == 136
    assert home["opponent_team_id"] == 133
    assert home["runs_scored"] == 5
    assert home["runs_allowed"] == 3
    assert home["hits"] == 9
    assert home["errors"] == 0
    assert home["win"] is True

    assert away["team_id"] == 133
    assert away["runs_scored"] == 3
    assert away["win"] is False


def test_normalize_batting_skips_non_batters(boxscore):
    rows = normalize.normalize_batting(700001, boxscore)

    # Pitchers with empty batting stats are excluded
    assert {r["player_id"] for r in rows} == {660001, 660003}
    catcher = next(r for r in rows if r["player_id"] == 660003)
    assert catcher["team_id"] == 136
    assert catcher["hits"] == 3
    assert catcher["home_runs"] == 2
    assert catcher["rbi"] == 4
    assert catcher["strikeouts"] == 0


def test_normalize_pitching(boxscore):
    rows = normalize.normalize_pitching(700001, boxscore)

    assert {r["player_id"] for r in rows} == {660002, 660004}
    starter = next(r for r in rows if r["player_id"] == 660002)
    assert starter["team_id"] == 133
    assert starter["innings_pitched"] == "6.0"
    assert starter["outs"] == 18
    assert starter["strikeouts"] == 7
    assert starter["earned_runs"] == 2
    assert starter["pitches"] == 95
