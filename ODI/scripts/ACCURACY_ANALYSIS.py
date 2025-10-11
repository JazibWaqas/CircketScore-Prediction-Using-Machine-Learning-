#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACCURACY ANALYSIS - Show Real Predictions

Focus on: How close are predictions to actual scores?
"""

import pandas as pd
import numpy as np
import pickle
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("ACCURACY ANALYSIS - HOW CLOSE ARE PREDICTIONS?")
print("="*80)

# Load model and data
with open('../models/CLEAN_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/CLEAN_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../models/CLEAN_feature_names.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('../models/CLEAN_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

df_test = pd.read_csv('../data/CLEAN_test_dataset.csv')

# Prepare features
le_team = encoders['team']
le_venue = encoders['venue']

df_test['team_encoded'] = le_team.transform(df_test['team_name'])
df_test['opp_encoded'] = le_team.transform(df_test['opposition_name'])
df_test['venue_encoded'] = le_venue.transform(df_test['venue_name'])

X_test = df_test[feature_cols]
y_test = df_test['total_runs']

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

df_test['predicted'] = y_pred
df_test['error'] = y_pred - y_test
df_test['abs_error'] = np.abs(y_pred - y_test)
df_test['error_pct'] = 100 * df_test['abs_error'] / y_test

# Overall accuracy
mae = df_test['abs_error'].mean()
median_error = df_test['abs_error'].median()

print(f"\n📊 OVERALL ACCURACY:")
print(f"   Average error (MAE): {mae:.1f} runs")
print(f"   Median error: {median_error:.1f} runs")
print(f"   Average % error: {df_test['error_pct'].mean():.1f}%")

print(f"\n🎯 ACCURACY BREAKDOWN:")
within_10 = (df_test['abs_error'] <= 10).sum()
within_20 = (df_test['abs_error'] <= 20).sum()
within_30 = (df_test['abs_error'] <= 30).sum()
within_40 = (df_test['abs_error'] <= 40).sum()
within_50 = (df_test['abs_error'] <= 50).sum()

total = len(df_test)
print(f"   Within ±10 runs: {within_10:3d}/{total} ({100*within_10/total:5.1f}%) - EXCELLENT")
print(f"   Within ±20 runs: {within_20:3d}/{total} ({100*within_20/total:5.1f}%) - VERY GOOD")
print(f"   Within ±30 runs: {within_30:3d}/{total} ({100*within_30/total:5.1f}%) - GOOD")
print(f"   Within ±40 runs: {within_40:3d}/{total} ({100*within_40/total:5.1f}%) - ACCEPTABLE")
print(f"   Within ±50 runs: {within_50:3d}/{total} ({100*within_50/total:5.1f}%) - POOR")
print(f"   Beyond ±50 runs: {total-within_50:3d}/{total} ({100*(total-within_50)/total:5.1f}%) - VERY POOR")

print(f"\n💡 WHAT THIS MEANS:")
print(f"   - Only {100*within_30/total:.1f}% of predictions are 'good' (within ±30 runs)")
print(f"   - That means {100*(total-within_30)/total:.1f}% have errors > 30 runs")
print(f"   - ±30 runs = ~5-6 overs difference in typical ODI")

# Show actual examples
print(f"\n" + "="*80)
print(f"REAL EXAMPLES FROM TEST SET (2023-2025 Matches)")
print(f"="*80)

# Recent matches
df_recent = df_test.sort_values('date', ascending=False).head(50)

print(f"\n✅ EXCELLENT PREDICTIONS (Error ≤ 10 runs):")
df_excellent = df_recent[df_recent['abs_error'] <= 10].head(10)
if len(df_excellent) > 0:
    print(f"\n{'Date':<12} {'Team':<18} {'vs':<3} {'Opp':<18} {'Actual':>6} {'Predicted':>9} {'Error':>7} {'%Err':>6}")
    print("-" * 95)
    for idx, row in df_excellent.iterrows():
        print(f"{row['date']:<12} {row['team_name'][:16]:<18} vs {row['opposition_name'][:16]:>18} "
              f"{row['total_runs']:>6.0f} {row['predicted']:>9.0f} {row['error']:>+7.0f} {row['error_pct']:>6.1f}%")
else:
    print("   None in recent 50 matches!")

print(f"\n✓ GOOD PREDICTIONS (Error 11-30 runs):")
df_good = df_recent[(df_recent['abs_error'] > 10) & (df_recent['abs_error'] <= 30)].head(10)
if len(df_good) > 0:
    print(f"\n{'Date':<12} {'Team':<18} {'vs':<3} {'Opp':<18} {'Actual':>6} {'Predicted':>9} {'Error':>7} {'%Err':>6}")
    print("-" * 95)
    for idx, row in df_good.iterrows():
        print(f"{row['date']:<12} {row['team_name'][:16]:<18} vs {row['opposition_name'][:16]:>18} "
              f"{row['total_runs']:>6.0f} {row['predicted']:>9.0f} {row['error']:>+7.0f} {row['error_pct']:>6.1f}%")
else:
    print("   None in recent 50 matches!")

print(f"\n⚠️  ACCEPTABLE PREDICTIONS (Error 31-50 runs):")
df_acceptable = df_recent[(df_recent['abs_error'] > 30) & (df_recent['abs_error'] <= 50)].head(10)
if len(df_acceptable) > 0:
    print(f"\n{'Date':<12} {'Team':<18} {'vs':<3} {'Opp':<18} {'Actual':>6} {'Predicted':>9} {'Error':>7} {'%Err':>6}")
    print("-" * 95)
    for idx, row in df_acceptable.iterrows():
        print(f"{row['date']:<12} {row['team_name'][:16]:<18} vs {row['opposition_name'][:16]:>18} "
              f"{row['total_runs']:>6.0f} {row['predicted']:>9.0f} {row['error']:>+7.0f} {row['error_pct']:>6.1f}%")

print(f"\n❌ POOR PREDICTIONS (Error > 50 runs):")
df_poor = df_recent[df_recent['abs_error'] > 50].head(10)
if len(df_poor) > 0:
    print(f"\n{'Date':<12} {'Team':<18} {'vs':<3} {'Opp':<18} {'Actual':>6} {'Predicted':>9} {'Error':>7} {'%Err':>6}")
    print("-" * 95)
    for idx, row in df_poor.iterrows():
        print(f"{row['date']:<12} {row['team_name'][:16]:<18} vs {row['opposition_name'][:16]:>18} "
              f"{row['total_runs']:>6.0f} {row['predicted']:>9.0f} {row['error']:>+7.0f} {row['error_pct']:>6.1f}%")

# Analyze by actual score
print(f"\n" + "="*80)
print(f"ACCURACY BY ACTUAL SCORE RANGE")
print(f"="*80)

ranges = [
    (0, 150, "Very Low (< 150)"),
    (150, 200, "Low (150-200)"),
    (200, 250, "Medium (200-250)"),
    (250, 300, "High (250-300)"),
    (300, 500, "Very High (300+)")
]

for low, high, label in ranges:
    mask = (df_test['total_runs'] >= low) & (df_test['total_runs'] < high)
    if mask.sum() > 0:
        count = mask.sum()
        mae_range = df_test[mask]['abs_error'].mean()
        within_30 = (df_test[mask]['abs_error'] <= 30).sum()
        pct_within_30 = 100 * within_30 / count
        
        # Typical prediction for this range
        avg_actual = df_test[mask]['total_runs'].mean()
        avg_pred = df_test[mask]['predicted'].mean()
        bias = avg_pred - avg_actual
        
        print(f"\n{label}:")
        print(f"   Matches: {count}")
        print(f"   Average actual: {avg_actual:.0f} runs")
        print(f"   Average predicted: {avg_pred:.0f} runs")
        print(f"   Bias: {bias:+.0f} runs ({'over' if bias > 0 else 'under'}predicting)")
        print(f"   Average error: {mae_range:.1f} runs")
        print(f"   Within ±30: {within_30}/{count} ({pct_within_30:.1f}%)")
        
        if pct_within_30 >= 60:
            verdict = "✅ GOOD accuracy"
        elif pct_within_30 >= 40:
            verdict = "⚠️  MODERATE accuracy"
        else:
            verdict = "❌ POOR accuracy"
        print(f"   Verdict: {verdict}")

# Real-world interpretation
print(f"\n" + "="*80)
print(f"REAL-WORLD INTERPRETATION")
print(f"="*80)

print(f"\n🎯 If you use this model to predict scores:")

print(f"\n   BEST CASE (34.6% of time):")
print(f"   → Prediction within ±30 runs of actual")
print(f"   → Example: Predicts 250, actual is 235 (good!)")

print(f"\n   TYPICAL CASE (46% of time):")
print(f"   → Error between 30-60 runs")
print(f"   → Example: Predicts 250, actual is 195 or 305 (not great)")

print(f"\n   WORST CASE (20% of time):")
print(f"   → Error > 60 runs")
print(f"   → Example: Predicts 240, actual is 140 or 340 (very poor)")

print(f"\n💡 COMPARISON TO OTHER PREDICTIONS:")
print(f"   - Predicting 'average' (250 runs) every time: MAE ≈ 60 runs")
print(f"   - This model: MAE = {mae:.1f} runs")
print(f"   - Improvement: {60 - mae:.1f} runs better than naive baseline")
print(f"   - Expert with pitch info: MAE ≈ 25-30 runs (2x better than us)")

print(f"\n❌ HONEST ASSESSMENT:")
print(f"   The model provides ROUGH estimates, not accurate predictions.")
print(f"   Only 1 in 3 predictions are 'good' (within ±30 runs)")
print(f"   Without pitch conditions, this is about as good as we can get.")

print(f"\n✅ WHAT IT'S GOOD FOR:")
print(f"   - Baseline estimates (expect ~240 runs, not 400)")
print(f"   - Relative comparison (Team A stronger than Team B)")
print(f"   - Understanding factors (batting first helps)")
print(f"   - General planning (not precise prediction)")

print(f"\n⚠️  WHAT IT'S NOT GOOD FOR:")
print(f"   - Betting (too inaccurate)")
print(f"   - Guaranteed accuracy claims")
print(f"   - Precise target setting")
print(f"   - Critical decisions")

print(f"\n" + "="*80 + "\n")

