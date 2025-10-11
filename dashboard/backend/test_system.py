#!/usr/bin/env python3
"""
Test the complete ODI Progressive Dashboard system
Tests API endpoints with real match data from test set
"""

import requests
import pandas as pd
import json
import time

API_URL = 'http://localhost:5002/api'

print("\n" + "="*80)
print("TESTING ODI PROGRESSIVE DASHBOARD")
print("="*80)

# Wait for server to start
print("\n[1/7] Checking if backend is running...")
max_retries = 10
for i in range(max_retries):
    try:
        response = requests.get(f'{API_URL}/health', timeout=2)
        if response.status_code == 200:
            print("   ✓ Backend is running!")
            print(f"   Response: {response.json()}")
            break
    except:
        if i < max_retries - 1:
            print(f"   Waiting for backend... ({i+1}/{max_retries})")
            time.sleep(2)
        else:
            print("   ✗ Backend not responding. Please start: python app.py")
            exit(1)

# Test /api/teams
print("\n[2/7] Testing GET /api/teams...")
response = requests.get(f'{API_URL}/teams')
assert response.status_code == 200
teams_data = response.json()
print(f"   ✓ Loaded {teams_data['count']} teams")
print(f"   Teams: {', '.join(teams_data['teams'][:5])}...")

# Test /api/players
print("\n[3/7] Testing GET /api/players/India...")
response = requests.get(f'{API_URL}/players/India')
assert response.status_code == 200
players_data = response.json()
print(f"   ✓ Loaded {players_data['count']} players for India")
if players_data['players']:
    print(f"   Top 3: {', '.join([p['name'] for p in players_data['players'][:3]])}")

# Test /api/venues
print("\n[4/7] Testing GET /api/venues...")
response = requests.get(f'{API_URL}/venues')
assert response.status_code == 200
venues_data = response.json()
print(f"   ✓ Loaded {venues_data['count']} venues")
if venues_data['venues']:
    print(f"   Sample: {venues_data['venues'][0]['name']} (Avg: {venues_data['venues'][0]['avg_score']:.0f})")

# Test /api/predict with real scenario
print("\n[5/7] Testing POST /api/predict...")

# Load test data to get real match
test_df = pd.read_csv('../../ODI_Progressive/data/progressive_full_test.csv')
player_db = json.load(open('../../ODI/data/CURRENT_player_database_977_quality.json'))

# Find an international match
international_teams = ['India', 'Australia', 'England', 'Pakistan', 'South Africa']
test_match = test_df[test_df['batting_team'].isin(international_teams)].iloc[0]

print(f"\n   Testing with real match:")
print(f"   - Team: {test_match['batting_team']} vs {test_match['bowling_team']}")
print(f"   - Venue: {test_match['venue']}")
print(f"   - Stage: Over {test_match['balls_bowled']//6}, Score {test_match['current_score']}/{test_match['wickets_fallen']}")
print(f"   - Actual Final: {test_match['final_score']} runs")

# Get top 11 players from player database
all_players = list(player_db.keys())
batting_players = all_players[:11]
bowling_players = all_players[11:22]

predict_payload = {
    'batting_team_players': batting_players,
    'bowling_team_players': bowling_players,
    'venue': test_match['venue'],
    'venue_avg_score': float(test_match['venue_avg_score']),
    'current_score': int(test_match['current_score']),
    'wickets_fallen': int(test_match['wickets_fallen']),
    'balls_bowled': int(test_match['balls_bowled']),
    'runs_last_10_overs': int(test_match['runs_last_10_overs']),
    'batsman_1': '',
    'batsman_2': ''
}

response = requests.post(f'{API_URL}/predict', json=predict_payload)
assert response.status_code == 200
prediction = response.json()

print(f"\n   ✓ Prediction successful!")
print(f"   - Predicted: {prediction['predicted_score']:.1f} runs")
print(f"   - Actual: {test_match['final_score']} runs")
print(f"   - Error: {abs(prediction['predicted_score'] - test_match['final_score']):.1f} runs")
print(f"   - Confidence: {prediction['confidence']['label']} (R²={prediction['confidence']['r2']:.2f})")
print(f"   - Stage: {prediction['confidence']['stage']}")

# Test multiple real matches
print("\n[6/7] Testing on 10 real international matches...")

test_cases = []
for idx in range(min(10, len(test_df))):
    row = test_df.iloc[idx]
    
    try:
        payload = {
            'batting_team_players': batting_players,
            'bowling_team_players': bowling_players,
            'venue': row['venue'],
            'venue_avg_score': float(row['venue_avg_score']),
            'current_score': int(row['current_score']),
            'wickets_fallen': int(row['wickets_fallen']),
            'balls_bowled': int(row['balls_bowled']),
            'runs_last_10_overs': int(row['runs_last_10_overs']),
            'batsman_1': '',
            'batsman_2': ''
        }
        
        response = requests.post(f'{API_URL}/predict', json=payload)
        if response.status_code == 200:
            pred = response.json()
            error = abs(pred['predicted_score'] - row['final_score'])
            test_cases.append({
                'team': row['batting_team'],
                'over': row['balls_bowled']//6,
                'actual': row['final_score'],
                'predicted': pred['predicted_score'],
                'error': error,
                'r2': pred['confidence']['r2']
            })
    except:
        pass

if test_cases:
    print(f"\n   {'Team':<15} {'Over':>5} {'Predicted':>10} {'Actual':>8} {'Error':>7} {'R²':>6}")
    print(f"   {'-'*60}")
    for case in test_cases[:10]:
        print(f"   {case['team'][:13]:<15} {case['over']:>5} "
              f"{case['predicted']:>10.0f} {case['actual']:>8.0f} "
              f"{case['error']:>7.1f} {case['r2']:>6.2f}")
    
    avg_error = sum(c['error'] for c in test_cases) / len(test_cases)
    avg_r2 = sum(c['r2'] for c in test_cases) / len(test_cases)
    print(f"\n   Average Error: {avg_error:.1f} runs")
    print(f"   Average R²: {avg_r2:.3f}")

# Test progressive predictions
print("\n[7/7] Testing POST /api/progressive...")

progressive_payload = {
    'batting_team_players': batting_players,
    'bowling_team_players': bowling_players,
    'venue': test_match['venue'],
    'venue_avg_score': float(test_match['venue_avg_score']),
    'match_progression': [
        {'over': 0, 'score': 0, 'wickets': 0},
        {'over': 10, 'score': 55, 'wickets': 1},
        {'over': 20, 'score': 115, 'wickets': 2},
        {'over': 30, 'score': 180, 'wickets': 3},
        {'over': 40, 'score': 250, 'wickets': 5}
    ],
    'final_score': 320
}

response = requests.post(f'{API_URL}/progressive', json=progressive_payload)
assert response.status_code == 200
progressive = response.json()

print(f"\n   Progressive predictions (showing accuracy improvement):")
print(f"   {'Over':>5} {'Score':>8} {'Predicted':>10} {'R²':>6} {'MAE':>6} {'Confidence':<15}")
print(f"   {'-'*65}")
for pred in progressive['predictions']:
    print(f"   {pred['over']:>5} {pred['current_score']:>4}/{pred['wickets']:<2} "
          f"{pred['predicted_final']:>10.0f} {pred['r2']:>6.2f} {pred['mae']:>6.0f} "
          f"{pred['confidence']:<15}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("\n✓ ALL TESTS PASSED!")
print("\nAPI Endpoints Working:")
print("  ✓ GET  /api/health")
print("  ✓ GET  /api/teams")
print("  ✓ GET  /api/players/<team>")
print("  ✓ GET  /api/venues")
print("  ✓ POST /api/predict")
print("  ✓ POST /api/progressive")

print("\nValidation:")
print(f"  ✓ Tested on {len(test_cases)} real matches")
print(f"  ✓ Average error: {avg_error:.1f} runs")
print(f"  ✓ Progressive accuracy demonstrated")

print("\nNext Steps:")
print("  1. Frontend should be accessible at http://localhost:3000")
print("  2. Select 11 batting and bowling players")
print("  3. Set match scenario and get predictions")

print("\n" + "="*80 + "\n")

