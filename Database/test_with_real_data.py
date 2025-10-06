#!/usr/bin/env python3
"""
Test the strict ML system with REAL test data from simple_enhanced_test.csv
"""

import pandas as pd
import requests
import json

def test_with_real_data():
    """Test with actual test data"""
    print('🧪 TESTING WITH REAL TEST DATA')
    print('=' * 60)
    
    # Load the test data
    test_df = pd.read_csv('../data/simple_enhanced_test.csv')
    
    print(f"📊 Test dataset: {test_df.shape}")
    print(f"Score range: {test_df['total_runs'].min()}-{test_df['total_runs'].max()}")
    print(f"Average score: {test_df['total_runs'].mean():.1f}")
    
    # Test with first 5 matches from test data
    print(f"\n🎯 Testing with first 5 matches from test data:")
    print("=" * 60)
    
    results = []
    
    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        
        # Extract data from test row
        team_a_id = int(row['team_id'])
        venue_id = int(row['venue_id'])
        season_year = int(row['season_year'])
        season_month = int(row['season_month'])
        batting_first = bool(row['batting_first'])
        is_home_team = bool(row['is_home_team'])
        is_final = bool(row['is_final'])
        is_t20_world_cup = bool(row['is_t20_world_cup'])
        
        # Get actual score from test data
        actual_score = int(row['total_runs'])
        
        # Create prediction request
        match_data = {
            'team_a_id': team_a_id,
            'team_b_id': team_a_id + 1,  # Use a different team as opposition
            'venue_id': venue_id,
            'team_a_players': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'team_b_players': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'match_context': {
                'battingFirst': 'team_a' if batting_first else 'team_b',
                'tossWinner': 'team_a',
                'tossDecision': 'bat',
                'isHomeTeam': is_home_team,
                'isFinal': is_final,
                'isT20WorldCup': is_t20_world_cup,
                'isImportantMatch': is_final or is_t20_world_cup,
                'seasonYear': season_year,
                'seasonMonth': season_month,
                'isSummer': season_month in [6, 7, 8],
                'tournamentType': 'T20 World Cup' if is_t20_world_cup else 'Bilateral'
            },
            'model': 'xgboost'
        }
        
        try:
            response = requests.post('http://localhost:5000/api/predict', json=match_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                pred = result['prediction']
                
                predicted_score_a = pred['predicted_score_a']
                predicted_score_b = pred['predicted_score_b']
                avg_predicted = (predicted_score_a + predicted_score_b) / 2
                
                error = abs(avg_predicted - actual_score)
                error_percent = (error / actual_score) * 100
                
                results.append({
                    'actual': actual_score,
                    'predicted': avg_predicted,
                    'error': error,
                    'error_percent': error_percent,
                    'team_a': pred['team_a'],
                    'venue': pred['venue']
                })
                
                print(f"Match {i+1}:")
                print(f"  Actual Score: {actual_score}")
                print(f"  Predicted: {predicted_score_a} vs {predicted_score_b} (avg: {avg_predicted:.1f})")
                print(f"  Error: {error:.1f} runs ({error_percent:.1f}%)")
                print(f"  Teams: {pred['team_a']} vs {pred['team_b']}")
                print(f"  Venue: {pred['venue']}")
                print()
                
            else:
                print(f"Match {i+1}: Error {response.status_code}")
                
        except Exception as e:
            print(f"Match {i+1}: Connection error - {e}")
    
    # Calculate overall performance
    if results:
        avg_error = sum(r['error'] for r in results) / len(results)
        avg_error_percent = sum(r['error_percent'] for r in results) / len(results)
        
        print("📊 OVERALL PERFORMANCE:")
        print("=" * 40)
        print(f"Average Error: {avg_error:.1f} runs")
        print(f"Average Error %: {avg_error_percent:.1f}%")
        
        # Count good predictions (within 20 runs)
        good_predictions = sum(1 for r in results if r['error'] <= 20)
        print(f"Good Predictions (≤20 runs error): {good_predictions}/{len(results)} ({good_predictions/len(results)*100:.1f}%)")
        
        # Count excellent predictions (within 10 runs)
        excellent_predictions = sum(1 for r in results if r['error'] <= 10)
        print(f"Excellent Predictions (≤10 runs error): {excellent_predictions}/{len(results)} ({excellent_predictions/len(results)*100:.1f}%)")

if __name__ == "__main__":
    test_with_real_data()
