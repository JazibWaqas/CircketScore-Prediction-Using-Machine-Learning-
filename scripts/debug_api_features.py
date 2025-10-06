#!/usr/bin/env python3
"""
Debug API Feature Preparation
Compare API-generated features with training data features
"""

import pandas as pd
import numpy as np
import requests
import json

def debug_api_features():
    """Debug what features the API is actually generating"""
    print("🔍 DEBUGGING API FEATURE PREPARATION")
    print("=" * 60)
    
    # Load training data to compare
    train_df = pd.read_csv('data/simple_enhanced_train.csv')
    print(f"Training data shape: {train_df.shape}")
    print(f"Training data columns: {len(train_df.columns)}")
    
    # Get a sample from training data for comparison
    sample_match = train_df.iloc[0]  # Australia vs New Zealand at Eden Park
    print(f"\n📊 SAMPLE MATCH FROM TRAINING DATA:")
    print(f"Match: {sample_match['team']} vs {sample_match['opposition']} at {sample_match['venue']}")
    print(f"Actual Score: {sample_match['total_runs']}")
    print(f"Date: {sample_match['date']}")
    print(f"Season: {sample_match['season']}")
    
    # Show key features from training data
    key_features = [
        'venue_avg_runs', 'venue_runs_std', 'venue_matches', 'venue_high_score', 'venue_low_score',
        'h2h_matches', 'h2h_avg_runs', 'h2h_win_rate', 'team_form_avg_runs', 'team_form_win_rate',
        'venue_difficulty', 'team_form_score', 'h2h_strength', 'team_balance'
    ]
    
    print(f"\n🎯 KEY FEATURES FROM TRAINING DATA:")
    for feature in key_features:
        if feature in sample_match.index:
            print(f"  {feature}: {sample_match[feature]}")
    
    # Test API with same match parameters
    print(f"\n🌐 TESTING API WITH SAME PARAMETERS:")
    
    payload = {
        "team_a_id": 3,  # Australia
        "team_b_id": 108,  # New Zealand
        "venue_id": 119,  # Eden Park
        "team_a_players": [],
        "team_b_players": [],
        "match_context": {
            "isHomeTeam": True,
            "isFinal": False,
            "isPlayoff": False,
            "isT20WorldCup": False,
            "isBilateral": True,
            "isImportantMatch": False,
            "seasonYear": 2005,  # Match was in 2005
            "seasonMonth": 2,   # February
            "isWinter": True,
            "isSummer": False,
            "isMonsoon": False,
            "tournamentType": "bilateral"
        },
        "model": "random_forest"
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/predict", 
            json=payload, 
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction', {})
            
            print(f"✅ API Response received")
            print(f"Predicted Score: {prediction.get('predicted_score_a', 'N/A')}")
            print(f"Confidence: {prediction.get('confidence', 'N/A')}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
    
    # Let's also check what the API is actually doing internally
    print(f"\n🔧 ANALYZING API FEATURE GENERATION LOGIC:")
    
    # Read the API code to understand feature preparation
    try:
        with open('Database/run.py', 'r') as f:
            api_code = f.read()
        
        # Look for the feature preparation function
        lines = api_code.split('\n')
        in_feature_function = False
        feature_lines = []
        
        for i, line in enumerate(lines):
            if 'def prepare_features_for_model(' in line:
                in_feature_function = True
                feature_lines.append(f"Line {i+1}: {line}")
            elif in_feature_function:
                if line.startswith('def ') or line.startswith('class '):
                    break
                feature_lines.append(f"Line {i+1}: {line}")
        
        print(f"Feature preparation function found:")
        for line in feature_lines[:20]:  # Show first 20 lines
            print(f"  {line}")
        
        # Look for specific feature assignments
        print(f"\n🎯 KEY FEATURE ASSIGNMENTS IN API:")
        key_assignments = [
            'venue_avg_runs', 'venue_runs_std', 'venue_matches', 
            'h2h_matches', 'h2h_avg_runs', 'team_form_avg_runs'
        ]
        
        for i, line in enumerate(lines):
            for feature in key_assignments:
                if feature in line and 'features.append' in line:
                    print(f"  Line {i+1}: {line.strip()}")
        
    except Exception as e:
        print(f"Could not analyze API code: {e}")
    
    # Check if there are default values being used
    print(f"\n⚠️ CHECKING FOR DEFAULT VALUES IN API:")
    
    default_patterns = [
        '140.0', '30.0', '50', '200', '80',  # Venue defaults
        '20', '135.0', '0.5', '365',  # H2H defaults
        '135.0', '0.5'  # Team form defaults
    ]
    
    try:
        with open('Database/run.py', 'r') as f:
            api_code = f.read()
        
        for pattern in default_patterns:
            if pattern in api_code:
                print(f"  Found default value: {pattern}")
                
    except Exception as e:
        print(f"Could not check for defaults: {e}")

if __name__ == "__main__":
    debug_api_features()
