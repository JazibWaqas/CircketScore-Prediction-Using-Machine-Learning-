import requests
import json

# Test the API with a simple prediction
url = "http://localhost:5000/api/predict"
data = {
    "team_a_id": 1,
    "team_b_id": 2,
    "venue_id": 1,
    "team_a_players": [1,2,3,4,5,6,7,8,9,10,11],
    "team_b_players": [12,13,14,15,16,17,18,19,20,21,22],
    "match_context": {
        "isHomeTeam": True,
        "isFinal": False,
        "isPlayoff": False,
        "isT20WorldCup": True,
        "isBilateral": False,
        "isImportantMatch": True,
        "seasonYear": 2024,
        "seasonMonth": 6,
        "isWinter": False,
        "isSummer": True,
        "isMonsoon": False
    },
    "model": "random_forest"
}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
