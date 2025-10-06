#!/usr/bin/env python3

import requests
import json

def test_venue_prediction(venue_id, venue_name):
    """Test prediction for a specific venue"""
    
    match_data = {
        'team_a_id': 117,  # Pakistan
        'team_b_id': 64,   # India
        'venue_id': venue_id,
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
        response = requests.post('http://localhost:5000/api/predict', json=match_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            pred = result['prediction']
            pak_pred = pred['predicted_score_a']
            ind_pred = pred['predicted_score_b']
            avg_pred = (pak_pred + ind_pred) / 2
            
            return {
                'venue_id': venue_id,
                'venue_name': venue_name,
                'pak_pred': pak_pred,
                'ind_pred': ind_pred,
                'avg_pred': avg_pred
            }
        else:
            print(f"❌ Error for venue {venue_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Connection error for venue {venue_id}: {e}")
        return None

# Test different venues
test_venues = [
    (116, "Dubai International Cricket Stadium"),  # Average venue
    (478, "High-scoring venue (avg 181)"),         # Highest scoring venue
    (131, "Low-scoring venue (avg 88)"),           # Lowest scoring venue
    (174, "High-scoring venue (avg 178)"),         # Another high scorer
    (27, "Low-scoring venue (avg 94)"),            # Another low scorer
]

print('🏏 TESTING VENUE-SPECIFIC PREDICTIONS')
print('=' * 60)
print('Testing Pakistan vs India at different venues...')
print()

results = []
for venue_id, venue_name in test_venues:
    print(f'🤖 Testing venue {venue_id} ({venue_name})...')
    result = test_venue_prediction(venue_id, venue_name)
    if result:
        results.append(result)
        print(f'   Pakistan: {result["pak_pred"]} runs')
        print(f'   India: {result["ind_pred"]} runs')
        print(f'   Average: {result["avg_pred"]:.1f} runs')
    print()

# Analysis
if results:
    print('📊 VENUE PREDICTION ANALYSIS:')
    print('=' * 50)
    
    # Sort by average prediction
    results.sort(key=lambda x: x['avg_pred'], reverse=True)
    
    print(f'{"Venue ID":<10} {"Venue Type":<25} {"Avg Prediction":<15}')
    print('-' * 50)
    
    for result in results:
        venue_type = "High-scoring" if result['avg_pred'] > 150 else "Low-scoring" if result['avg_pred'] < 120 else "Average"
        print(f'{result["venue_id"]:<10} {venue_type:<25} {result["avg_pred"]:.1f} runs')
    
    print()
    print('🎯 VERDICT:')
    highest_avg = max(results, key=lambda x: x['avg_pred'])
    lowest_avg = min(results, key=lambda x: x['avg_pred'])
    
    print(f'✅ Highest prediction: {highest_avg["avg_pred"]:.1f} runs at venue {highest_avg["venue_id"]}')
    print(f'✅ Lowest prediction: {lowest_avg["avg_pred"]:.1f} runs at venue {lowest_avg["venue_id"]}')
    print(f'✅ Prediction range: {highest_avg["avg_pred"] - lowest_avg["avg_pred"]:.1f} runs difference')
    
    if highest_avg["avg_pred"] - lowest_avg["avg_pred"] > 30:
        print('🎉 EXCELLENT! System now properly accounts for venue differences!')
    elif highest_avg["avg_pred"] - lowest_avg["avg_pred"] > 15:
        print('✅ GOOD! System shows venue variation, but could be improved.')
    else:
        print('⚠️ System still not properly accounting for venue differences.')

else:
    print('❌ No successful predictions - check server status')
