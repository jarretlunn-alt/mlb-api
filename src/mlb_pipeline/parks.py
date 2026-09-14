"""Static park metadata: GPS coordinates, field orientation, timezone.

out_bearing_deg: compass bearing from home plate toward center field.
    Used to compute wind_out_mph = wind_mph * cos(wind_from_deg + 180 - out_bearing_deg).
    Positive wind_out_mph = wind blowing toward the outfield (hitter-friendly).
    Set to None for domes/closed retractable roofs where wind is irrelevant.

timezone: pytz-compatible zone name for the park's local time.
    Used to convert a 19:00 local game time to UTC when calling the weather API.
"""

from __future__ import annotations

# team_id → park metadata (MLB Stats API team IDs)
PARKS: dict[int, dict] = {
    108: {"name": "Angel Stadium",          "lat": 33.800,  "lon": -117.883, "out_bearing_deg": 5,   "timezone": "America/Los_Angeles"},
    109: {"name": "Chase Field",             "lat": 33.445,  "lon": -112.067, "out_bearing_deg": None, "timezone": "America/Phoenix"},       # retractable
    110: {"name": "Camden Yards",            "lat": 39.284,  "lon": -76.622,  "out_bearing_deg": 85,  "timezone": "America/New_York"},
    111: {"name": "Fenway Park",             "lat": 42.347,  "lon": -71.097,  "out_bearing_deg": 95,  "timezone": "America/New_York"},
    112: {"name": "Wrigley Field",           "lat": 41.948,  "lon": -87.655,  "out_bearing_deg": 75,  "timezone": "America/Chicago"},
    113: {"name": "Great American Ball Park","lat": 39.097,  "lon": -84.507,  "out_bearing_deg": 285, "timezone": "America/New_York"},
    114: {"name": "Progressive Field",       "lat": 41.496,  "lon": -81.685,  "out_bearing_deg": 270, "timezone": "America/New_York"},
    115: {"name": "Coors Field",             "lat": 39.756,  "lon": -104.994, "out_bearing_deg": 295, "timezone": "America/Denver"},
    116: {"name": "Comerica Park",           "lat": 42.339,  "lon": -83.049,  "out_bearing_deg": 325, "timezone": "America/Detroit"},
    117: {"name": "Daikin Park",             "lat": 29.757,  "lon": -95.355,  "out_bearing_deg": None, "timezone": "America/Chicago"},       # retractable
    118: {"name": "Kauffman Stadium",        "lat": 39.052,  "lon": -94.480,  "out_bearing_deg": 5,   "timezone": "America/Chicago"},
    119: {"name": "Dodger Stadium",          "lat": 34.074,  "lon": -118.240, "out_bearing_deg": 0,   "timezone": "America/Los_Angeles"},
    120: {"name": "Nationals Park",          "lat": 38.873,  "lon": -77.007,  "out_bearing_deg": 305, "timezone": "America/New_York"},
    121: {"name": "Citi Field",              "lat": 40.757,  "lon": -73.846,  "out_bearing_deg": 5,   "timezone": "America/New_York"},
    133: {"name": "Sutter Health Park",      "lat": 38.583,  "lon": -121.501, "out_bearing_deg": 5,   "timezone": "America/Los_Angeles"},
    134: {"name": "PNC Park",                "lat": 40.447,  "lon": -80.006,  "out_bearing_deg": 310, "timezone": "America/New_York"},
    135: {"name": "Petco Park",              "lat": 32.708,  "lon": -117.157, "out_bearing_deg": 310, "timezone": "America/Los_Angeles"},
    136: {"name": "T-Mobile Park",           "lat": 47.591,  "lon": -122.332, "out_bearing_deg": None, "timezone": "America/Los_Angeles"},   # retractable
    137: {"name": "Oracle Park",             "lat": 37.778,  "lon": -122.389, "out_bearing_deg": 290, "timezone": "America/Los_Angeles"},
    138: {"name": "Busch Stadium",           "lat": 38.623,  "lon": -90.193,  "out_bearing_deg": 300, "timezone": "America/Chicago"},
    139: {"name": "Tropicana Field",         "lat": 27.768,  "lon": -82.653,  "out_bearing_deg": None, "timezone": "America/New_York"},      # dome
    140: {"name": "Globe Life Field",        "lat": 32.747,  "lon": -97.084,  "out_bearing_deg": None, "timezone": "America/Chicago"},       # retractable
    141: {"name": "Rogers Centre",           "lat": 43.641,  "lon": -79.389,  "out_bearing_deg": None, "timezone": "America/Toronto"},       # dome
    142: {"name": "Target Field",            "lat": 44.982,  "lon": -93.278,  "out_bearing_deg": 340, "timezone": "America/Chicago"},
    143: {"name": "Citizens Bank Park",      "lat": 39.906,  "lon": -75.166,  "out_bearing_deg": 300, "timezone": "America/New_York"},
    144: {"name": "Truist Park",             "lat": 33.891,  "lon": -84.468,  "out_bearing_deg": 310, "timezone": "America/New_York"},
    145: {"name": "Rate Field",              "lat": 41.831,  "lon": -87.634,  "out_bearing_deg": 5,   "timezone": "America/Chicago"},
    146: {"name": "loanDepot park",          "lat": 25.778,  "lon": -80.220,  "out_bearing_deg": None, "timezone": "America/New_York"},      # retractable
    147: {"name": "Yankee Stadium",          "lat": 40.829,  "lon": -73.926,  "out_bearing_deg": 325, "timezone": "America/New_York"},
    158: {"name": "American Family Field",   "lat": 43.028,  "lon": -87.971,  "out_bearing_deg": None, "timezone": "America/Chicago"},       # retractable
}


def wind_out_mph(wind_from_deg: float, wind_speed_mph: float, out_bearing_deg: float) -> float:
    """Return the component of wind blowing from HP toward CF.

    Positive = tailwind (hitter-friendly), negative = headwind.
    wind_from_deg is meteorological convention: direction the wind is coming FROM.
    """
    import math
    angle = math.radians(wind_from_deg + 180 - out_bearing_deg)
    return wind_speed_mph * math.cos(angle)
