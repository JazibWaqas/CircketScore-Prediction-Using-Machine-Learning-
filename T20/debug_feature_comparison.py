#!/usr/bin/env python3
"""
Compare our generated features with actual training data features
"""

import pandas as pd
import numpy as np
import joblib

def compare_features():
    print("🔍 COMPARING GENERATED FEATURES VS TRAINING DATA")
    print("=" * 60)
    
    # Load training data
    df = pd.read_csv('processed_data/cleaned_cricket_dataset.csv')
    feature_names = joblib.load('models/final_trained_feature_names.pkl')
    
    print(f"✅ Loaded training data with {len(df)} samples")
    print(f"✅ Feature names: {len(feature_names)} features")
    
    # Show actual training data feature ranges
    print(f"\nTRAINING DATA FEATURE RANGES (first 10 features):")
    print("Feature Name | Min | Max | Mean | Std")
    print("-" * 50)
    
    for i, feature_name in enumerate(feature_names[:10]):
        values = df[feature_name]
        print(f"{feature_name:12} | {values.min():5.2f} | {values.max():5.2f} | {values.mean():5.2f} | {values.std():5.2f}")
    
    # Show sample training data
    print(f"\nSAMPLE TRAINING DATA (first 3 samples):")
    for i in range(3):
        print(f"\nSample {i+1} (actual score: {df.iloc[i]['total_runs']}):")
        for j, feature_name in enumerate(feature_names[:5]):
            print(f"  {feature_name}: {df.iloc[i][feature_name]:.3f}")
    
    # Generate our features
    print(f"\nOUR GENERATED FEATURES:")
    team_a_name = 'Pakistan'
    team_b_name = 'India'
    venue_name = 'Dubai International Cricket Stadium'
    
    # Generate features using our logic
    features = {}
    
    # 1. team_balance_x
    player_count = 3
    features['team_balance_x'] = (player_count - 11) * 0.2
    
    # 2. h2h_avg_runs
    features['h2h_avg_runs'] = ((hash(f"{team_a_name}_{team_b_name}") % 20) - 10) / 5.0
    
    # 3. pitch_bounce
    features['pitch_bounce'] = ((hash(venue_name) % 10) - 5) / 3.0
    
    # 4. team_form_avg_runs
    features['team_form_avg_runs'] = ((hash(team_a_name) % 15) - 7.5) / 4.0
    
    # 5. venue_avg_runs
    features['venue_avg_runs'] = ((hash(venue_name) % 25) - 12.5) / 6.0
    
    print(f"Our generated features (first 5):")
    for i, feature_name in enumerate(feature_names[:5]):
        value = features.get(feature_name, 0.0)
        print(f"  {feature_name}: {value:.3f}")
    
    # Check if our features are in the training data range
    print(f"\nFEATURE RANGE CHECK:")
    for i, feature_name in enumerate(feature_names[:5]):
        training_min = df[feature_name].min()
        training_max = df[feature_name].max()
        our_value = features.get(feature_name, 0.0)
        
        in_range = training_min <= our_value <= training_max
        print(f"{feature_name}: {our_value:.3f} (training: {training_min:.2f} to {training_max:.2f}) {'✅' if in_range else '❌'}")

if __name__ == "__main__":
    compare_features()
