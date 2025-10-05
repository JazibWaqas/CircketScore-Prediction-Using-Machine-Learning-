import json
import requests


def main() -> None:
    payload = {
        "team_a_id": 117,  # Pakistan
        "team_b_id": 64,   # India
        "venue_id": 116,   # Dubai International Cricket Stadium
        "team_a_players": [],
        "team_b_players": [],
        "match_context": {
            "isHomeTeam": True,
            "isFinal": False,
            "isPlayoff": False,
            "isT20WorldCup": True,
            "isBilateral": False,
            "isImportantMatch": True,
            "seasonYear": 2025,
            "seasonMonth": 9,
            "isWinter": False,
            "isSummer": True,
            "isMonsoon": False,
            "tournamentType": "t20_world_cup",
        },
        "model": "random_forest",
    }

    url = "http://localhost:5000/api/predict"
    resp = requests.post(url, json=payload, timeout=20)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()


