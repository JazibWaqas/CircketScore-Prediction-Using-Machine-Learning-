#!/usr/bin/env python3
"""
TRAIN PROGRESSIVE ODI MODEL WITH FULL FEATURES

Trains XGBoost model with all 15 features:
- 6 match state features
- 3 batting team features  
- 3 opposition bowling features
- 2 venue features
- 2 current batsmen features (optional)
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("TRAIN PROGRESSIVE ODI MODEL - FULL FEATURES")
print("="*80)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================

print("\n[1/5] Loading training data...")

train_df = pd.read_csv('../data/progressive_full_train.csv')
test_df = pd.read_csv('../data/progressive_full_test.csv')

print(f"   Training: {len(train_df):,} samples")
print(f"   Testing: {len(test_df):,} samples")

# ==============================================================================
# STEP 2: PREPARE FEATURES
# ==============================================================================

print("\n[2/5] Preparing features...")

# Feature columns
numeric_features = [
    'current_score', 'wickets_fallen', 'balls_bowled', 'balls_remaining',
    'runs_last_10_overs', 'current_run_rate',
    'team_batting_avg', 'team_elite_batsmen', 'team_batting_depth',
    'opp_bowling_economy', 'opp_elite_bowlers', 'opp_bowling_depth',
    'venue_avg_score', 'batsman_1_avg', 'batsman_2_avg'
]

categorical_features = ['venue']

all_features = numeric_features + categorical_features

# Prepare X, y
X_train = train_df[all_features]
y_train = train_df['final_score']

X_test = test_df[all_features]
y_test = test_df['final_score']

print(f"   Features: {len(all_features)} total")
print(f"   - Numeric: {len(numeric_features)}")
print(f"   - Categorical: {len(categorical_features)}")

# ==============================================================================
# STEP 3: BUILD PIPELINE
# ==============================================================================

print("\n[3/5] Building pipeline...")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    ))
])

print(f"   Pipeline created")
print(f"   Model: XGBoost (n_estimators=400, max_depth=7)")

# ==============================================================================
# STEP 4: TRAIN MODEL
# ==============================================================================

print("\n[4/5] Training model...")
print(f"   (This may take 2-5 minutes with {len(X_train):,} samples)")

pipeline.fit(X_train, y_train)

print(f"   Training complete!")

# ==============================================================================
# STEP 5: EVALUATE ON TEST SET
# ==============================================================================

print("\n[5/5] Evaluating on test set...")

y_pred = pipeline.predict(X_test)

overall_r2 = r2_score(y_test, y_pred)
overall_mae = mean_absolute_error(y_test, y_pred)

print(f"\n{'='*80}")
print("TRAINING RESULTS")
print(f"{'='*80}")

print(f"\nOVERALL PERFORMANCE:")
print(f"   R2 Score: {overall_r2:.4f} ({overall_r2*100:.2f}%)")
print(f"   MAE: {overall_mae:.2f} runs")

# Performance by stage
print(f"\nPERFORMANCE BY MATCH STAGE:")

test_results = test_df.copy()
test_results['predicted'] = y_pred

stages = [
    ("Pre-match (0-10 overs)", 240, 300),
    ("Early (10-20 overs)", 180, 240),
    ("Mid (20-30 overs)", 120, 180),
    ("Late (30-40 overs)", 60, 120),
    ("Death (40-50 overs)", 0, 60)
]

for stage_name, min_balls, max_balls in stages:
    mask = (test_results['balls_remaining'] >= min_balls) & (test_results['balls_remaining'] < max_balls)
    if mask.sum() > 0:
        stage_r2 = r2_score(test_results[mask]['final_score'], test_results[mask]['predicted'])
        stage_mae = mean_absolute_error(test_results[mask]['final_score'], test_results[mask]['predicted'])
        print(f"   {stage_name:<25s} R2 = {stage_r2:.4f}, MAE = {stage_mae:.2f} runs (n={mask.sum()})")

# Accuracy bands
errors = np.abs(y_pred - y_test)
within_10 = (errors <= 10).sum()
within_20 = (errors <= 20).sum()
within_30 = (errors <= 30).sum()

print(f"\nACCURACY:")
print(f"   Within +/-10 runs: {within_10}/{len(y_test)} ({100*within_10/len(y_test):.1f}%)")
print(f"   Within +/-20 runs: {within_20}/{len(y_test)} ({100*within_20/len(y_test):.1f}%)")
print(f"   Within +/-30 runs: {within_30}/{len(y_test)} ({100*within_30/len(y_test):.1f}%)")

# Sample predictions
print(f"\nSAMPLE PREDICTIONS:")
sample = test_results.sample(min(10, len(test_results)))
print(f"\n{'Team':<15} {'Over':>5} {'Score':>8} {'Pred':>7} {'Actual':>8} {'Error':>7}")
print("-" * 70)
for _, row in sample.iterrows():
    over = row['balls_bowled'] // 6
    print(f"{row['batting_team'][:13]:<15} {over:>5} {row['current_score']:>4.0f}/{row['wickets_fallen']:<2.0f} "
          f"{row['predicted']:>7.0f} {row['final_score']:>8.0f} {row['predicted']-row['final_score']:>+7.0f}")

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print(f"\n{'='*80}")
print("SAVING MODEL")
print(f"{'='*80}")

import os
os.makedirs('../models', exist_ok=True)

# Save pipeline
with open('../models/progressive_model_full_features.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print(f"\n   [SAVED] ../models/progressive_model_full_features.pkl")

# Save feature names
feature_info = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'all_features': all_features
}
with open('../models/feature_names.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"   [SAVED] ../models/feature_names.json")

# Save metadata
metadata = {
    'training_date': pd.Timestamp.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'model': 'XGBRegressor',
    'parameters': {
        'n_estimators': 400,
        'max_depth': 7,
        'learning_rate': 0.1
    },
    'performance': {
        'overall_r2': float(overall_r2),
        'overall_mae': float(overall_mae),
        'accuracy_within_10': float(100*within_10/len(y_test)),
        'accuracy_within_20': float(100*within_20/len(y_test)),
        'accuracy_within_30': float(100*within_30/len(y_test))
    }
}
with open('../models/training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   [SAVED] ../models/training_metadata.json")

# ==============================================================================
# ASSESSMENT
# ==============================================================================

print(f"\n{'='*80}")
print("ASSESSMENT")
print(f"{'='*80}")

if overall_r2 >= 0.80:
    print(f"\n[EXCELLENT] R2 = {overall_r2:.3f}")
    print(f"   Model performance exceeds target (0.75+)")
elif overall_r2 >= 0.70:
    print(f"\n[GOOD] R2 = {overall_r2:.3f}")
    print(f"   Model performance meets target (0.70-0.80)")
elif overall_r2 >= 0.60:
    print(f"\n[ACCEPTABLE] R2 = {overall_r2:.3f}")
    print(f"   Model works but has room for improvement")
else:
    print(f"\n[NEEDS WORK] R2 = {overall_r2:.3f}")
    print(f"   Model needs improvement")

print(f"\nNEXT STEPS:")
print(f"   1. Validate on real international matches")
print(f"   2. Test fantasy use cases (what-if scenarios)")
print(f"   3. Generate final results report")

print(f"\n{'='*80}\n")

