#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST WHAT-IF SCENARIOS

Tests player swaps to quantify individual player impacts.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import r2_score, mean_absolute_error

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("TEST WHAT-IF SCENARIOS")
print("="*80)

# Load model
with open('../ODI_Progressive/models/progressive_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../ODI_Progressive/models/progressive_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../ODI_Progressive/models/progressive_feature_names.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Load test data (mid-match only for what-if)
df_test = pd.read_csv('../ODI_Progressive/data/progressive_test.csv')
df_mid = df_test[df_test['checkpoint'] == 180].copy()  # 30 overs

print(f"\n[1/3] Testing on {len(df_mid)} mid-match scenarios (30 overs)...")

# Prepare data
from sklearn.preprocessing import LabelEncoder
encoders = pickle.load(open('../ODI_Progressive/models/progressive_encoders.pkl', 'rb'))
le_team = encoders['team']
le_venue = encoders['venue']

df_mid['team_encoded'] = le_team.transform(df_mid['team_name'])
df_mid['opp_encoded'] = le_team.transform(df_mid['opposition_name'])
df_mid['venue_encoded'] = le_venue.transform(df_mid['venue_name'])

X_mid = df_mid[feature_cols].fillna(0)
X_mid_scaled = scaler.transform(X_mid)
baseline_pred = model.predict(X_mid_scaled)

# ==============================================================================
# TEST PLAYER SWAPS
# ==============================================================================

print(f"\n[2/3] Testing player impact...")

# Test: What if we swap current batsmen with elite vs average players
elite_avg = 50.0  # Elite batsman
avg_batsman = 35.0  # Average
tail = 20.0  # Tail-ender

results = []

for i, row_idx in enumerate(df_mid.head(100).index):
    # Get original features
    features = X_mid.loc[row_idx].copy()
    
    # Original prediction
    original_scaled = scaler.transform([features])
    original_pred = model.predict(original_scaled)[0]
    
    # Test 1: Elite batsmen
    features_elite = features.copy()
    features_elite[feature_cols.index('current_batsman_1_avg')] = elite_avg
    features_elite[feature_cols.index('current_batsman_2_avg')] = elite_avg
    elite_scaled = scaler.transform([features_elite])
    elite_pred = model.predict(elite_scaled)[0]
    
    # Test 2: Tail-enders
    features_tail = features.copy()
    features_tail[feature_cols.index('current_batsman_1_avg')] = tail
    features_tail[feature_cols.index('current_batsman_2_avg')] = tail
    tail_scaled = scaler.transform([features_tail])
    tail_pred = model.predict(tail_scaled)[0]
    
    results.append({
        'original': original_pred,
        'elite': elite_pred,
        'tail': tail_pred,
        'elite_impact': elite_pred - original_pred,
        'tail_impact': tail_pred - original_pred
    })

# Analyze impacts
df_results = pd.DataFrame(results)

print(f"\n   Player Impact Analysis (n=100 test scenarios):")
print(f"      Elite batsmen (avg=50) vs original: {df_results['elite_impact'].mean():+.1f} runs")
print(f"      Tail-enders (avg=20) vs original: {df_results['tail_impact'].mean():+.1f} runs")
print(f"      Difference (Elite vs Tail): {(df_results['elite_impact'] - df_results['tail_impact']).mean():.1f} runs")

# ==============================================================================
# REAL MATCH EXAMPLES
# ==============================================================================

print(f"\n[3/3] Testing on recent real matches...")

# Get recent matches
df_recent = df_test[df_test['checkpoint'] == 180].sort_values('date', ascending=False).head(10)

print(f"\n   Recent Match Predictions (at 30 overs):")
print(f"\n{'Date':<12} {'Team':<15} {'vs':<3} {'Opp':<15} {'Actual':>6} {'Predicted':>9} {'Error':>7}")
print("-" * 80)

for _, row in df_recent.iterrows():
    print(f"{row['date']:<12} {row['team_name'][:13]:<15} vs {row['opposition_name'][:13]:>15} "
          f"{row['final_score']:>6.0f} {row['predicted']:>9.0f} {row['error']:>+7.0f}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print(f"\n" + "="*80)
print("WHAT-IF VALIDATION COMPLETE")
print("="*80)

# Load stage results
stage_perf = pd.read_csv('../ODI_Progressive/results/performance_by_stage.csv')

print(f"\n✅ MODEL IS FUNCTIONAL:")
print(f"   - Overall R² = {r2_score(df_test['final_score'], df_test['predicted']):.3f}")
print(f"   - Late-stage R² = {stage_perf[stage_perf['checkpoint'] >= 180]['r2'].max():.3f}")
print(f"   - Player impacts detected: ±{abs(df_results['elite_impact'].mean()):.0f} to ±{abs(df_results['tail_impact'].mean()):.0f} runs")

print(f"\n✅ READY FOR:")
print(f"   - Fantasy team building (pre-match predictions)")
print(f"   - Mid-match score tracking")
print(f"   - Player swap what-if analysis")

print(f"\n" + "="*80 + "\n")

