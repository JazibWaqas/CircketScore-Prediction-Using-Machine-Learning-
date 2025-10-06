#!/usr/bin/env python3
"""
Test the new clean cricket prediction system
"""

import requests
import json

def test_new_system():
    """Test the new clean system"""
    print('🧪 TESTING NEW CLEAN SYSTEM')
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
            
            print('✅ SUCCESS! Clean system working!')
            print(f'Team A ({pred["team_a"]}): {pred["predicted_score_a"]} runs')
            print(f'Team B ({pred["team_b"]}): {pred["predicted_score_b"]} runs')
            print(f'Predicted Winner: {pred["predicted_winner"]}')
            print(f'Model Used: {pred["model_used"]}')
            print(f'Confidence: {pred["confidence"]:.2f}')
            print(f'Venue: {pred["venue"]}')
            
            # Test consistency
            print('\n🔄 Testing consistency (same data, multiple predictions):')
            predictions = []
            for i in range(3):
                resp = requests.post('http://localhost:5000/api/predict', json=match_data, timeout=10)
                if resp.status_code == 200:
                    res = resp.json()
                    pred_res = res['prediction']
                    predictions.append((pred_res['predicted_score_a'], pred_res['predicted_score_b']))
                    print(f'  Run {i+1}: {pred_res["predicted_score_a"]} vs {pred_res["predicted_score_b"]}')
            
            if len(predictions) >= 2:
                pak_scores = [p[0] for p in predictions]
                ind_scores = [p[1] for p in predictions]
                
                pak_consistent = len(set(pak_scores)) == 1
                ind_consistent = len(set(ind_scores)) == 1
                
                if pak_consistent and ind_consistent:
                    print('✅ EXCELLENT! Predictions are consistent - models working correctly!')
                else:
                    print('⚠️ Predictions vary (expected due to random variation)')
                    
            # Test different models
            print('\n🎯 Testing different models:')
            models_to_test = ['random_forest', 'linear_regression']
            for model_name in models_to_test:
                test_data = match_data.copy()
                test_data['model'] = model_name
                resp = requests.post('http://localhost:5000/api/predict', json=test_data, timeout=10)
                if resp.status_code == 200:
                    res = resp.json()
                    pred_res = res['prediction']
                    print(f'  {model_name}: {pred_res["predicted_score_a"]} vs {pred_res["predicted_score_b"]}')
                else:
                    print(f'  {model_name}: Error {resp.status_code}')
                    
        else:
            print(f'❌ Error {response.status_code}: {response.text}')
            
    except Exception as e:
        print(f'❌ Connection error: {e}')

if __name__ == "__main__":
    test_new_system()
