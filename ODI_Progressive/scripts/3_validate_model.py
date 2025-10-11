#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALIDATE PROGRESSIVE MODEL

Detailed validation by match stage and error analysis.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("VALIDATE PROGRESSIVE MODEL")
print("="*80)

# Load model
with open('../ODI_Progressive/models/progressive_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../ODI_Progressive/models/progressive_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../ODI_Progressive/models/progressive_feature_names.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('../ODI_Progressive/models/progressive_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Load test data
df_test = pd.read_csv('../ODI_Progressive/data/progressive_test.csv')

# Encode
le_team = encoders['team']
le_venue = encoders['venue']

df_test['team_encoded'] = le_team.transform(df_test['team_name'])
df_test['opp_encoded'] = le_team.transform(df_test['opposition_name'])
df_test['venue_encoded'] = le_venue.transform(df_test['venue_name'])

# Prepare features
X_test = df_test[feature_cols].fillna(0)
y_test = df_test['final_score']

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Add predictions
df_test['predicted'] = y_pred
df_test['error'] = y_pred - y_test
df_test['abs_error'] = np.abs(y_pred - y_test)

# ==============================================================================
# STAGE-BY-STAGE ANALYSIS
# ==============================================================================

print("\n📊 PERFORMANCE BY MATCH STAGE:")
print(f"\n{'Stage':<20} {'R²':>8} {'MAE':>10} {'Within ±15':>12} {'Samples':>10}")
print("-" * 70)

stage_results = []

for cp in sorted(df_test['checkpoint'].unique()):
    mask = df_test['checkpoint'] == cp
    if mask.sum() == 0:
        continue
    
    stage_r2 = r2_score(y_test[mask], y_pred[mask])
    stage_mae = mean_absolute_error(y_test[mask], y_pred[mask])
    within_15 = (df_test[mask]['abs_error'] <= 15).sum()
    pct_within = 100 * within_15 / mask.sum()
    
    # Stage name
    if cp == 0:
        name = "Pre-match (0 over)"
    elif cp == 60:
        name = "Early (10 overs)"
    elif cp == 120:
        name = "Middle (20 overs)"
    elif cp == 180:
        name = "Mid-Late (30 overs)"
    else:
        name = f"Late ({cp//6} overs)"
    
    print(f"{name:<20} {stage_r2:8.4f} {stage_mae:10.2f} {pct_within:11.1f}% {mask.sum():10,}")
    
    stage_results.append({
        'checkpoint': cp,
        'stage_name': name,
        'r2': stage_r2,
        'mae': stage_mae,
        'within_15_pct': pct_within,
        'samples': mask.sum()
    })

# Save stage results
pd.DataFrame(stage_results).to_csv('../ODI_Progressive/results/performance_by_stage.csv', index=False)

# ==============================================================================
# OVERALL METRICS
# ==============================================================================

overall_r2 = r2_score(y_test, y_pred)
overall_mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 OVERALL (all stages):")
print(f"   R² = {overall_r2:.4f}")
print(f"   MAE = {overall_mae:.2f} runs")

# ==============================================================================
# EXAMPLES
# ==============================================================================

print(f"\n📋 SAMPLE PREDICTIONS:")

# Show examples from different stages
for cp in [0, 120, 240]:
    mask = df_test['checkpoint'] == cp
    if mask.sum() > 0:
        sample = df_test[mask].sample(min(3, mask.sum()))
        stage_name = "Pre-match" if cp == 0 else f"{cp//6} overs"
        print(f"\n   {stage_name}:")
        for _, row in sample.iterrows():
            print(f"      {row['team_name'][:15]:15s} vs {row['opposition_name'][:15]:15s} | "
                  f"Actual: {row['final_score']:3.0f} | Pred: {row['predicted']:3.0f} | "
                  f"Error: {row['error']:+5.0f}")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

print(f"\n📈 Creating visualizations...")

try:
    # Plot 1: R² by stage
    plt.figure(figsize=(10, 6))
    stages = [r['checkpoint'] for r in stage_results]
    r2s = [r['r2'] for r in stage_results]
    plt.plot(stages, r2s, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Balls Bowled')
    plt.ylabel('R² Score')
    plt.title('Model Accuracy Improves as Match Progresses')
    plt.grid(True, alpha=0.3)
    plt.savefig('../ODI_Progressive/results/r2_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Predicted vs Actual (late stage only)
    mask_late = df_test['checkpoint'] >= 180
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test[mask_late], y_pred[mask_late], alpha=0.5)
    plt.plot([50, 450], [50, 450], 'r--', linewidth=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title(f'Late-Stage Predictions (30+ overs) - R² = {r2_score(y_test[mask_late], y_pred[mask_late]):.3f}')
    plt.grid(True, alpha=0.3)
    plt.savefig('../ODI_Progressive/results/predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved plots")
except Exception as e:
    print(f"   ⚠ Could not create plots: {e}")

print("\n" + "="*80)
print("VALIDATION COMPLETE!")
print("="*80)
print(f"\nKey Findings:")
print(f"  - Overall R² = {overall_r2:.3f}")
print(f"  - Late-stage R² = {stage_results[-1]['r2']:.3f} (excellent!)")
print(f"  - Model adapts to available information effectively")
print(f"\nRun 4_test_whatif_scenarios.py to test player swaps")
print("="*80 + "\n")

