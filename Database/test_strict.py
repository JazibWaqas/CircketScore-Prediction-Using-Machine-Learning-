#!/usr/bin/env python3
"""
Test the STRICT ML models only system
"""

import requests
import json

def test_strict_system():
    """Test the strict ML system"""
    print('🧪 TESTING STRICT ML SYSTEM')
    print('=' * 50)
    
    # Test data
    match_data = {
        'team_a_id': 117,
        'team_b_id': 64,
        'venue_id': 478,
        'team_a_players': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'team_b_players': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'match_context': {
            'battingFirst': 'team_a',
            'tossWinner': 'team_a',
            'tossDecision': 'bat',
            'isHomeTeam': True,
            'isFinal': False,
            'isT20WorldCup': False,
            'isImportantMatch': True,
            'seasonYear': 2025,
            'seasonMonth': 9,
            'isSummer': True,
            'tournamentType': 'Asia Cup'
        },
        'model': 'xgboost'
    }
    
    try:
        # Test single prediction
        response = requests.post('http://localhost:5000/api/predict', json=match_data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            pred = result['prediction']
            
            print('✅ SUCCESS! STRICT ML system working!')
            print(f'Team A ({pred["team_a"]}): {pred["predicted_score_a"]} runs')
            print(f'Team B ({pred["team_b"]}): {pred["predicted_score_b"]} runs')
            print(f'Predicted Winner: {pred["predicted_winner"]}')
            print(f'Model Used: {pred["model_used"]}')
            print(f'Confidence: {pred["confidence"]:.2f}')
            print(f'Venue: {pred["venue"]}')
            
            # Test health endpoint
            health_resp = requests.get('http://localhost:5000/api/health')
            if health_resp.status_code == 200:
                health = health_resp.json()
                print(f'\\n🔍 System Mode: {health["mode"]}')
                print(f'Models Loaded: {health["models_loaded"]}')
                print(f'Encoders Loaded: {health["encoders_loaded"]}')
                
        else:
            print(f'❌ Error {response.status_code}: {response.text}')
            
    except Exception as e:
        print(f'❌ Connection error: {e}')

if __name__ == "__main__":
    test_strict_system()
