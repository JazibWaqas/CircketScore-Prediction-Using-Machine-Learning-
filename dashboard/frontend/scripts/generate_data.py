"""
Regenerate the bundled reference data and preset predictions.

The frontend ships both files so the page renders a complete, already-predicted
scenario on first paint rather than waiting on a cold API container. Run this
after changing the model, the player database, or the presets in
`src/utils/squad.js`.

    python scripts/generate_data.py

The selection logic below mirrors `src/utils/squad.js`. If that file changes,
change this to match, otherwise the cached predictions will not correspond to
the XIs shown on screen.
"""

import json
import os
import urllib.request

API = os.environ.get("API_BASE", "https://cricket-score-predictor-api.onrender.com/api")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data")

MIN_MATCHES = 20

# Keep in sync with the PRESETS array in src/utils/squad.js.
PRESETS = [
    ("wc-final", "India", "Australia",
     ["Narendra Modi Stadium", "Wankhede", "Eden Gardens"],
     dict(current_score=198, wickets_fallen=4, overs=35, runs_last_10=62)),
    ("pak-ind", "Pakistan", "India",
     ["Dubai International", "Arun Jaitley", "Eden Gardens"],
     dict(current_score=234, wickets_fallen=5, overs=40, runs_last_10=46)),
    ("eng-nz", "England", "New Zealand",
     ["Lord's", "Trent Bridge", "The Oval"],
     dict(current_score=78, wickets_fallen=1, overs=12, runs_last_10=68)),
]


def get(path):
    with urllib.request.urlopen(f"{API}/{path}", timeout=180) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(
        f"{API}/{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def relevance(p):
    matches = p.get("total_matches") or 0
    bat = p.get("batting_avg") or 0
    econ = p.get("bowling_economy") or 0
    experience = min(1, matches / 100)
    batting = min(bat, 60) / 60
    bowling = max(0, min(1, (7.5 - econ) / 4)) if econ > 0 else 0
    return max(batting, bowling * 0.9) * 0.6 + experience * 0.4


def established(p):
    return (p.get("total_matches") or 0) >= MIN_MATCHES and (
        (p.get("batting_avg") or 0) > 0 or (p.get("bowling_economy") or 0) > 0
    )


def build_xi(players, country):
    pool = [p for p in players if (p.get("country") or "").lower() == country.lower()]
    if not pool:
        return []
    ranked = sorted(pool, key=relevance, reverse=True)
    est = [p for p in ranked if established(p)]
    source = est if len(est) >= 11 else ranked

    picked, taken = [], set()
    for role, count in (("Batsman", 5), ("All-rounder", 2), ("Bowler", 4)):
        matching = [
            p for p in source
            if (p.get("player_role") or "All-rounder") == role and p["player_id"] not in taken
        ]
        for p in matching[:count]:
            picked.append(p)
            taken.add(p["player_id"])

    for p in source:
        if len(picked) >= 11:
            break
        if p["player_id"] not in taken:
            picked.append(p)
            taken.add(p["player_id"])

    order = {"Batsman": 0, "All-rounder": 1, "Bowler": 2}
    return sorted(
        picked[:11],
        key=lambda p: (order.get(p.get("player_role"), 1), -(p.get("batting_avg") or 0)),
    )


def resolve_venue(venues, prefs):
    for pref in prefs:
        for v in venues:
            if pref.lower() in (v["venue_name"] or "").lower():
                return v
    return venues[0]


def main():
    print(f"Fetching reference data from {API}")
    players_raw = get("players")["players"]
    venues_raw = get("venues")["venues"]
    teams_raw = get("teams")["teams"]

    player_keys = (
        "player_id", "player_name", "player_role", "country",
        "batting_avg", "bowling_economy", "total_matches",
    )
    players = [
        {k: p.get(k) for k in player_keys}
        for p in players_raw
        if (p.get("total_matches") or 0) >= MIN_MATCHES
    ]
    for p in players:
        for k in ("batting_avg", "bowling_economy"):
            if isinstance(p.get(k), float):
                p[k] = round(p[k], 2)

    venues = [{k: v.get(k) for k in ("venue_name", "avg_score")} for v in venues_raw]
    for v in venues:
        if isinstance(v.get("avg_score"), float):
            v["avg_score"] = round(v["avg_score"], 1)

    teams = [{k: t.get(k) for k in ("team_id", "team_name")} for t in teams_raw]

    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot_path = os.path.join(DATA_DIR, "snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({"teams": teams, "players": players, "venues": venues}, f, separators=(",", ":"))
    print(f"snapshot.json: {len(teams)} teams, {len(players)} players, {len(venues)} venues")

    predictions = {}
    for pid, team_a, team_b, venue_prefs, scenario in PRESETS:
        xi_a = build_xi(players, team_a)
        xi_b = build_xi(players, team_b)
        venue = resolve_venue(venues, venue_prefs)
        result = post("predict", dict(
            batting_team_players=[p["player_name"] for p in xi_a],
            bowling_team_players=[p["player_name"] for p in xi_b],
            venue=venue["venue_name"],
            venue_avg_score=venue["avg_score"],
            current_score=scenario["current_score"],
            wickets_fallen=scenario["wickets_fallen"],
            balls_bowled=scenario["overs"] * 6,
            runs_last_10_overs=scenario["runs_last_10"],
            batsman_1="",
            batsman_2="",
            model="xgboost",
        ))
        predictions[pid] = result
        print(f"  {pid}: {team_a} v {team_b} -> {result['predicted_score']}")

    preset_path = os.path.join(DATA_DIR, "presetPredictions.json")
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, separators=(",", ":"))
    print("presetPredictions.json written")


if __name__ == "__main__":
    main()
