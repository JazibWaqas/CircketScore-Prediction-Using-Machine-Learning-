#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST CLEAN MODEL - Detailed Analysis

Analyzes model performance on test set to understand:
- Where predictions are accurate vs inaccurate
- Feature patterns in good vs bad predictions
- Systematic biases
- Opportunities for improvement
"""

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import r2_score, mean_absolute_error

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("DETAILED MODEL ANALYSIS")
print("="*80)

# Load test data
df_test = pd.read_csv('../data/CLEAN_test_dataset.csv')

# Load model and artifacts
with open('../models/CLEAN_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/CLEAN_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../models/CLEAN_feature_names.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('../models/CLEAN_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare test data
le_team = encoders['team']
le_venue = encoders['venue']

df_test['team_encoded'] = le_team.transform(df_test['team_name'])
df_test['opp_encoded'] = le_team.transform(df_test['opposition_name'])
df_test['venue_encoded'] = le_venue.transform(df_test['venue_name'])

X_test = df_test[feature_cols]
y_test = df_test['total_runs']

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
errors = y_pred - y_test

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   R² = {r2:.4f}")
print(f"   MAE = {mae:.2f} runs")
print(f"   Bias = {errors.mean():+.2f} runs (positive = overpredicting)")
print(f"   Prediction range: {y_pred.min():.0f} - {y_pred.max():.0f} runs")
print(f"   Actual range: {y_test.min():.0f} - {y_test.max():.0f} runs")

# Analyze by score ranges
print(f"\n📈 PERFORMANCE BY SCORE RANGE:")
score_ranges = [
    (0, 150, "Very Low (< 150)"),
    (150, 200, "Low (150-200)"),
    (200, 250, "Medium (200-250)"),
    (250, 300, "High (250-300)"),
    (300, 500, "Very High (300+)")
]

for low, high, label in score_ranges:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        range_bias = errors[mask].mean()
        print(f"   {label:25s} n={mask.sum():3d} | MAE={range_mae:5.1f} | Bias={range_bias:+6.1f}")

# Best and worst predictions
print(f"\n✅ BEST 10 PREDICTIONS:")
df_results = pd.DataFrame({
    'team': df_test['team_name'].values,
    'opposition': df_test['opposition_name'].values,
    'actual': y_test.values,
    'predicted': y_pred,
    'error': np.abs(errors),
    'date': df_test['date'].values
})
df_results = df_results.sort_values('error')

for idx, row in df_results.head(10).iterrows():
    print(f"   {row['team']:20s} vs {row['opposition']:20s} | Actual:{row['actual']:3.0f} Pred:{row['predicted']:3.0f} | Error:{row['error']:4.1f}")

print(f"\n❌ WORST 10 PREDICTIONS:")
for idx, row in df_results.tail(10).iterrows():
    print(f"   {row['team']:20s} vs {row['opposition']:20s} | Actual:{row['actual']:3.0f} Pred:{row['predicted']:3.0f} | Error:{row['error']:4.1f}")

# Feature analysis
print(f"\n🔍 FEATURE ANALYSIS:")
print(f"   Top features actually being used:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    feat = row['feature']
    imp = row['importance']
    # Get correlation with actual scores
    if feat in X_test.columns:
        corr = np.corrcoef(X_test[feat], y_test)[0,1]
        print(f"   {feat:35s} Imp:{imp:.4f} Corr:{corr:+.3f}")

# Check prediction variance issue
print(f"\n⚠️  PREDICTION VARIANCE ISSUE:")
print(f"   Actual scores std: {y_test.std():.1f} runs")
print(f"   Predicted scores std: {y_pred.std():.1f} runs")
print(f"   Ratio: {y_pred.std() / y_test.std():.2f} (should be ~1.0)")
print(f"\n   This means predictions don't vary enough!")
print(f"   Model is too conservative / regularized")

# Analyze by batting_first (most important feature)
print(f"\n🏏 BATTING FIRST ANALYSIS:")
for bf_val in [0, 1]:
    mask = df_test['batting_first'] == bf_val
    label = "Batting First" if bf_val == 1 else "Batting Second"
    if mask.sum() > 0:
        actual_mean = y_test[mask].mean()
        pred_mean = y_pred[mask].mean()
        mae_bf = mean_absolute_error(y_test[mask], y_pred[mask])
        print(f"   {label:15s} n={mask.sum():3d} | Actual:{actual_mean:6.1f} Pred:{pred_mean:6.1f} | MAE:{mae_bf:5.1f}")

# Save detailed results
output_path = '../results/clean_model_test_results.csv'
df_results.to_csv(output_path, index=False)
print(f"\n✓ Saved detailed results to: {output_path}")

# Assessment
print(f"\n" + "="*80)
print(f"DIAGNOSIS:")
print(f"="*80)

if r2 < 0.30:
    print(f"\n❌ MODEL PERFORMANCE IS POOR (R² = {r2:.3f})")
    print(f"\n   LIKELY CAUSES:")
    print(f"   1. Features are too weak without pitch conditions")
    print(f"   2. Team aggregates from career stats may not reflect current form well")
    print(f"   3. Model is over-regularized (predictions too conservative)")
    print(f"   4. ODI scores have high inherent variance (~70 runs std)")
    
    print(f"\n   RECOMMENDATIONS:")
    print(f"   1. Add more predictive features:")
    print(f"      - Recent match scores at THIS venue")
    print(f"      - Team vs team historical scores")
    print(f"      - Season/weather patterns")
    print(f"   2. Reduce regularization:")
    print(f"      - Increase max_depth (try 7-8)")
    print(f"      - Decrease min_child_weight (try 3)")
    print(f"   3. Use ensemble of models")
    print(f"   4. Accept that R²=0.30-0.40 may be realistic limit")
elif r2 < 0.50:
    print(f"\n⚠️  MODEL PERFORMANCE IS ACCEPTABLE BUT COULD BE BETTER (R² = {r2:.3f})")
    print(f"\n   Try feature engineering improvements")
else:
    print(f"\n✅ MODEL PERFORMANCE IS GOOD (R² = {r2:.3f})")

print(f"\n" + "="*80 + "\n")

