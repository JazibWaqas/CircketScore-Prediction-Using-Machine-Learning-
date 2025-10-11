#!/usr/bin/env python3
"""
Test the ODI Progressive Dashboard with Real Match Data
=====================================================

This script automatically tests the dashboard system by:
1. Loading real match data from the test dataset
2. Making predictions through the API
3. Comparing with actual scores
4. Showing system performance

Run this to validate your entire system works correctly.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5002/api"
TEST_DATA_PATH = "../ODI_Progressive/data/progressive_full_test.csv"

def test_api_health():
    """Test if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Backend API is running")
            return True
        else:
            print(f"[ERROR] Backend API error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Backend API not running. Start it first:")
        print("   cd dashboard/backend && python app.py")
        return False

def load_test_data():
    """Load real match test data"""
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        print(f"[OK] Loaded {len(df)} test cases from {TEST_DATA_PATH}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Test data not found: {TEST_DATA_PATH}")
        return None

def test_api_endpoints():
    """Test all API endpoints"""
    print("\nTesting API Endpoints...")
    
    # Test teams
    try:
        response = requests.get(f"{API_BASE_URL}/teams")
        if response.status_code == 200:
            teams = response.json()['teams']
            print(f"[OK] Teams endpoint: {len(teams)} teams loaded")
        else:
            print(f"[ERROR] Teams endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Teams endpoint error: {e}")
        return False
    
    # Test players
    try:
        response = requests.get(f"{API_BASE_URL}/players")
        if response.status_code == 200:
            players = response.json()['players']
            print(f"[OK] Players endpoint: {len(players)} players loaded")
        else:
            print(f"[ERROR] Players endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Players endpoint error: {e}")
        return False
    
    # Test venues
    try:
        response = requests.get(f"{API_BASE_URL}/venues")
        if response.status_code == 200:
            venues = response.json()['venues']
            print(f"[OK] Venues endpoint: {len(venues)} venues loaded")
        else:
            print(f"[ERROR] Venues endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Venues endpoint error: {e}")
        return False
    
    return True

def create_test_scenario(row):
    """Create a test scenario from a row of test data"""
    
    # Sample some players for teams (using actual player names from the data)
    batting_team_players = [
        "Virat Kohli", "Rohit Sharma", "KL Rahul", "Shikhar Dhawan", 
        "Hardik Pandya", "Ravindra Jadeja", "MS Dhoni", "Bhuvneshwar Kumar",
        "Jasprit Bumrah", "Yuzvendra Chahal", "Mohammed Shami"
    ]
    
    bowling_team_players = [
        "Steve Smith", "David Warner", "Aaron Finch", "Glenn Maxwell",
        "Marcus Stoinis", "Alex Carey", "Pat Cummins", "Mitchell Starc",
        "Adam Zampa", "Josh Hazlewood", "Nathan Lyon"
    ]
    
    # Current batsmen (optional)
    current_batsmen = ["Virat Kohli", "Rohit Sharma"] if row['wickets_fallen'] < 2 else ["KL Rahul", "Hardik Pandya"]
    
    scenario = {
        "batting_team_players": batting_team_players,
        "bowling_team_players": bowling_team_players,
        "venue": row.get('venue', 'Melbourne Cricket Ground'),
        "venue_avg_score": row.get('venue_avg_score', 250),
        "current_score": int(row['current_score']),
        "wickets_fallen": int(row['wickets_fallen']),
        "balls_bowled": int(row['balls_bowled']),
        "runs_last_10_overs": int(row.get('runs_last_10_overs', 45)),
        "batsman_1": current_batsmen[0],
        "batsman_2": current_batsmen[1]
    }
    
    return scenario

def test_prediction(scenario, actual_score):
    """Test a single prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=scenario,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted_score = result['predicted_score']
            confidence = result.get('confidence', 'Unknown')
            
            error = abs(predicted_score - actual_score)
            
            return {
                'success': True,
                'predicted': predicted_score,
                'actual': actual_score,
                'error': error,
                'confidence': confidence,
                'scenario': scenario
            }
        else:
            return {
                'success': False,
                'error': f"API error: {response.status_code}",
                'response': response.text
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_comprehensive_test():
    """Run comprehensive test on real match data"""
    
    print("ODI Progressive Dashboard - Real Match Testing")
    print("=" * 60)
    
    # Check API health
    if not test_api_health():
        return
    
    # Test endpoints
    if not test_api_endpoints():
        return
    
    # Load test data
    df = load_test_data()
    if df is None:
        return
    
    print(f"\nTesting {min(20, len(df))} real match scenarios...")
    
    results = []
    errors = []
    
    # Test first 20 scenarios (or all if less than 20)
    test_cases = df.head(20)
    
    for i, (idx, row) in enumerate(test_cases.iterrows()):
        print(f"\nTest Case {i+1}/20:")
        print(f"   Match: {row.get('match_id', 'Unknown')}")
        print(f"   Stage: {row['balls_bowled']} balls, {row['current_score']}/{row['wickets_fallen']}")
        print(f"   Actual Final Score: {row['final_score']}")
        
        # Create test scenario
        scenario = create_test_scenario(row)
        
        # Make prediction
        result = test_prediction(scenario, row['final_score'])
        
        if result['success']:
            print(f"   [OK] Predicted: {result['predicted']:.0f} runs")
            print(f"   Error: {result['error']:.0f} runs")
            print(f"   Confidence: {result['confidence']}")
            results.append(result)
        else:
            print(f"   [ERROR] Failed: {result['error']}")
            errors.append(result)
        
        # Small delay to not overwhelm the API
        time.sleep(0.5)
    
    # Calculate statistics
    if results:
        errors_list = [r['error'] for r in results]
        predicted_list = [r['predicted'] for r in results]
        actual_list = [r['actual'] for r in results]
        
        mae = sum(errors_list) / len(errors_list)
        mape = sum(abs(e/a) for e, a in zip(errors_list, actual_list)) / len(errors_list) * 100
        
        # Calculate R²
        actual_mean = sum(actual_list) / len(actual_list)
        ss_tot = sum((a - actual_mean)**2 for a in actual_list)
        ss_res = sum((a - p)**2 for a, p in zip(actual_list, predicted_list))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\nSYSTEM PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"[OK] Successful Predictions: {len(results)}/20")
        print(f"[ERROR] Failed Predictions: {len(errors)}/20")
        print(f"Mean Absolute Error (MAE): {mae:.1f} runs")
        print(f"Mean Absolute Percentage Error: {mape:.1f}%")
        print(f"R² Score: {r2:.3f}")
        
        # Show best and worst predictions
        best = min(results, key=lambda x: x['error'])
        worst = max(results, key=lambda x: x['error'])
        
        print(f"\nBest Prediction:")
        print(f"   Predicted: {best['predicted']:.0f}, Actual: {best['actual']}, Error: {best['error']:.0f}")
        
        print(f"\nWorst Prediction:")
        print(f"   Predicted: {worst['predicted']:.0f}, Actual: {worst['actual']}, Error: {worst['error']:.0f}")
        
        # Show confidence distribution
        confidence_counts = {}
        for r in results:
            conf = r['confidence']
            conf_label = conf['label'] if isinstance(conf, dict) else str(conf)
            confidence_counts[conf_label] = confidence_counts.get(conf_label, 0) + 1
        
        print(f"\nConfidence Distribution:")
        for conf, count in confidence_counts.items():
            print(f"   {conf}: {count} predictions")
    
    else:
        print("\n[ERROR] No successful predictions to analyze")
    
    # Show errors if any
    if errors:
        print(f"\n[ERROR] ERRORS ENCOUNTERED:")
        for i, error in enumerate(errors[:5]):  # Show first 5 errors
            print(f"   Error {i+1}: {error['error']}")
    
    print(f"\nTesting Complete!")
    print(f"   Your system is {'[OK] WORKING WELL' if results and len(results) >= 15 else '[WARNING] NEEDS ATTENTION'}")
    
    if results and len(results) >= 15:
        print(f"   [OK] {len(results)}/20 predictions successful")
        print(f"   [OK] MAE of {mae:.1f} runs is good for ODI predictions")
        print(f"   [OK] R² of {r2:.3f} shows model is learning patterns")
        print(f"\n[SUCCESS] Your dashboard is ready for production use!")
    else:
        print(f"   [WARNING] Only {len(results)}/20 predictions successful")
        print(f"   [WARNING] Check backend logs for errors")
        print(f"   [WARNING] Ensure model and data are loaded correctly")

if __name__ == "__main__":
    run_comprehensive_test()
