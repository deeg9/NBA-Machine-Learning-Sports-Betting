"""
Static arena reference data for all 30 NBA teams.

Maps team names (matching TEAM_NAME in the dataset) to arena coordinates,
elevation, and timezone. Includes aliases for historical team names.
"""

# Current team arenas: lat, lon, elevation (feet), IANA timezone
ARENA_DATA = {
    "Atlanta Hawks": {"lat": 33.7573, "lon": -84.3963, "elev_ft": 1050, "tz": "America/New_York"},
    "Boston Celtics": {"lat": 42.3662, "lon": -71.0621, "elev_ft": 20, "tz": "America/New_York"},
    "Brooklyn Nets": {"lat": 40.6826, "lon": -73.9754, "elev_ft": 30, "tz": "America/New_York"},
    "Charlotte Hornets": {"lat": 35.2251, "lon": -80.8392, "elev_ft": 751, "tz": "America/New_York"},
    "Chicago Bulls": {"lat": 41.8807, "lon": -87.6742, "elev_ft": 594, "tz": "America/Chicago"},
    "Cleveland Cavaliers": {"lat": 41.4965, "lon": -81.6882, "elev_ft": 653, "tz": "America/New_York"},
    "Dallas Mavericks": {"lat": 32.7905, "lon": -96.8103, "elev_ft": 430, "tz": "America/Chicago"},
    "Denver Nuggets": {"lat": 39.7487, "lon": -104.9996, "elev_ft": 5280, "tz": "America/Denver"},
    "Detroit Pistons": {"lat": 42.3410, "lon": -83.0551, "elev_ft": 600, "tz": "America/Detroit"},
    "Golden State Warriors": {"lat": 37.7680, "lon": -122.3877, "elev_ft": 5, "tz": "America/Los_Angeles"},
    "Houston Rockets": {"lat": 29.7508, "lon": -95.3621, "elev_ft": 50, "tz": "America/Chicago"},
    "Indiana Pacers": {"lat": 39.7640, "lon": -86.1555, "elev_ft": 715, "tz": "America/Indiana/Indianapolis"},
    "LA Clippers": {"lat": 33.4264, "lon": -118.2617, "elev_ft": 200, "tz": "America/Los_Angeles"},
    "Los Angeles Lakers": {"lat": 34.0430, "lon": -118.2673, "elev_ft": 270, "tz": "America/Los_Angeles"},
    "Memphis Grizzlies": {"lat": 35.1382, "lon": -90.0505, "elev_ft": 337, "tz": "America/Chicago"},
    "Miami Heat": {"lat": 25.7814, "lon": -80.1870, "elev_ft": 7, "tz": "America/New_York"},
    "Milwaukee Bucks": {"lat": 43.0451, "lon": -87.9174, "elev_ft": 617, "tz": "America/Chicago"},
    "Minnesota Timberwolves": {"lat": 44.9795, "lon": -93.2761, "elev_ft": 830, "tz": "America/Chicago"},
    "New Orleans Pelicans": {"lat": 29.9490, "lon": -90.0821, "elev_ft": 3, "tz": "America/Chicago"},
    "New York Knicks": {"lat": 40.7505, "lon": -73.9934, "elev_ft": 33, "tz": "America/New_York"},
    "Oklahoma City Thunder": {"lat": 35.4634, "lon": -97.5151, "elev_ft": 1199, "tz": "America/Chicago"},
    "Orlando Magic": {"lat": 28.5392, "lon": -81.3839, "elev_ft": 82, "tz": "America/New_York"},
    "Philadelphia 76ers": {"lat": 39.9012, "lon": -75.1720, "elev_ft": 39, "tz": "America/New_York"},
    "Phoenix Suns": {"lat": 33.4457, "lon": -112.0712, "elev_ft": 1086, "tz": "America/Phoenix"},
    "Portland Trail Blazers": {"lat": 45.5316, "lon": -122.6668, "elev_ft": 50, "tz": "America/Los_Angeles"},
    "Sacramento Kings": {"lat": 38.5802, "lon": -121.4997, "elev_ft": 30, "tz": "America/Los_Angeles"},
    "San Antonio Spurs": {"lat": 29.4270, "lon": -98.4375, "elev_ft": 650, "tz": "America/Chicago"},
    "Toronto Raptors": {"lat": 43.6435, "lon": -79.3791, "elev_ft": 249, "tz": "America/Toronto"},
    "Utah Jazz": {"lat": 40.7683, "lon": -111.9011, "elev_ft": 4226, "tz": "America/Denver"},
    "Washington Wizards": {"lat": 38.8981, "lon": -77.0209, "elev_ft": 30, "tz": "America/New_York"},
}

# Historical team name aliases → current team name
TEAM_ALIASES = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "Los Angeles Clippers": "LA Clippers",
    "New Jersey Nets": "Brooklyn Nets",
    "Seattle SuperSonics": "Oklahoma City Thunder",
}


def get_arena(team_name: str) -> dict:
    """Look up arena data for a team, resolving historical aliases."""
    if team_name in ARENA_DATA:
        return ARENA_DATA[team_name]
    alias = TEAM_ALIASES.get(team_name)
    if alias and alias in ARENA_DATA:
        return ARENA_DATA[alias]
    return None
