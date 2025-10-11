#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN XGBOOST ON FILTERED DATA (NO OUTLIERS)

Strategy:
1. Remove extreme outlier matches from training (< 120 or > 350 runs)
2. Focus model on "normal" ODI scores (120-350 range)
3. This should improve performance on typical matches
"""

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("TRAIN XGBOOST ON FILTERED DATA (NO OUTLIERS)")
print("="*80)

# ==============================================================================
# LOAD AND FILTER DATA
# ==============================================================================

print("\n[1/5] Loading and filtering data...")

df_train_full = pd.read_csv('../data/CLEAN_train_dataset.csv')
df_test = pd.read_csv('../data/CLEAN_test_dataset.csv')

print(f"   Original training: {len(df_train_full):,} samples")
print(f"      Score range: {df_train_full['total_runs'].min():.0f} - {df_train_full['total_runs'].max():.0f}")
print(f"      Mean: {df_train_full['total_runs'].mean():.1f} ± {df_train_full['total_runs'].std():.1f}")

# Filter outliers from TRAINING ONLY (keep test as-is to evaluate properly)
MIN_SCORE = 120  # Remove collapses
MAX_SCORE = 350  # Remove extreme high scores

df_train = df_train_full[
    (df_train_full['total_runs'] >= MIN_SCORE) & 
    (df_train_full['total_runs'] <= MAX_SCORE)
].copy()

removed = len(df_train_full) - len(df_train)
pct_removed = 100 * removed / len(df_train_full)

print(f"\n   Filtered training (keeping {MIN_SCORE}-{MAX_SCORE} runs):")
print(f"      Kept: {len(df_train):,} samples")
print(f"      Removed: {removed:,} outliers ({pct_removed:.1f}%)")
print(f"      New range: {df_train['total_runs'].min():.0f} - {df_train['total_runs'].max():.0f}")
print(f"      New mean: {df_train['total_runs'].mean():.1f} ± {df_train['total_runs'].std():.1f}")

print(f"\n   Test set (unchanged): {len(df_test):,} samples")
print(f"      Score range: {df_test['total_runs'].min():.0f} - {df_test['total_runs'].max():.0f}")

# ==============================================================================
# ENCODE CATEGORICAL
# ==============================================================================

print("\n[2/5] Encoding categorical features...")

le_team = LabelEncoder()
le_venue = LabelEncoder()

all_teams = pd.concat([df_train['team_name'], df_train['opposition_name'], 
                       df_test['team_name'], df_test['opposition_name']]).unique()
all_venues = pd.concat([df_train['venue_name'], df_test['venue_name']]).unique()

le_team.fit(all_teams)
le_venue.fit(all_venues)

df_train['team_encoded'] = le_team.transform(df_train['team_name'])
df_train['opp_encoded'] = le_team.transform(df_train['opposition_name'])
df_train['venue_encoded'] = le_venue.transform(df_train['venue_name'])

df_test['team_encoded'] = le_team.transform(df_test['team_name'])
df_test['opp_encoded'] = le_team.transform(df_test['opposition_name'])
df_test['venue_encoded'] = le_venue.transform(df_test['venue_name'])

print(f"   ✓ Encoded features")

# ==============================================================================
# PREPARE FEATURES
# ==============================================================================

print("\n[3/5] Preparing features...")

drop_cols = ['match_id', 'date', 'team_name', 'opposition_name', 'venue_name', 'total_runs']
feature_cols = [col for col in df_train.columns if col not in drop_cols]

X_train = df_train[feature_cols]
y_train = df_train['total_runs']

X_test = df_test[feature_cols]
y_test = df_test['total_runs']

print(f"   Features: {len(feature_cols)}")
print(f"   Training samples: {len(X_train):,} (filtered)")
print(f"   Test samples: {len(X_test):,} (all data)")

# Fill NaN and scale
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# TRAIN WITH MODERATE REGULARIZATION
# ==============================================================================

print("\n[4/5] Training XGBoost on filtered data...")

# Balance between conservative and flexible
xgb_params = {
    'n_estimators': 400,
    'max_depth': 6,              # Moderate depth
    'learning_rate': 0.05,
    'min_child_weight': 3,       # Moderate regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.05,               # Some minimum loss reduction
    'reg_alpha': 0.05,           # Moderate L1
    'reg_lambda': 0.7,           # Moderate L2
    'random_state': 42,
    'objective': 'reg:squarederror',
    'n_jobs': -1
}

print(f"\n   Strategy: Train on 'normal' matches, test on ALL matches")
print(f"   This focuses learning on typical scenarios")

# Cross-validation on filtered training data
print(f"\n   5-fold cross-validation on filtered training...")
model_cv = xgb.XGBRegressor(**xgb_params)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(model_cv, X_train_scaled, y_train, cv=kfold, scoring='r2', n_jobs=-1)
cv_mae = -cross_val_score(model_cv, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)

print(f"   Cross-validation R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"   Cross-validation MAE = {cv_mae.mean():.2f} ± {cv_mae.std():.2f} runs")

# Train final model
print(f"\n   Training final model...")
model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train_scaled, y_train, verbose=False)

# ==============================================================================
# EVALUATE ON FULL TEST SET
# ==============================================================================

print("\n[5/5] Evaluating on full test set (including outliers)...")

# Training performance (on filtered data)
y_train_pred = model.predict(X_train_scaled)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test performance (on ALL test data)
y_test_pred = model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

pred_std = y_test_pred.std()
actual_std = y_test.std()

print(f"\n   📊 TRAINING SET (filtered {MIN_SCORE}-{MAX_SCORE} runs):")
print(f"      R² = {train_r2:.4f}")
print(f"      MAE = {train_mae:.2f} runs")

print(f"\n   📊 TEST SET (all scores, including outliers):")
print(f"      R² = {test_r2:.4f}")
print(f"      MAE = {test_mae:.2f} runs")

print(f"\n   📈 VARIANCE:")
print(f"      Actual std: {actual_std:.1f} runs")
print(f"      Predicted std: {pred_std:.1f} runs")
print(f"      Ratio: {pred_std/actual_std:.2f}")

# Test on NORMAL matches only (same range as training)
mask_normal = (y_test >= MIN_SCORE) & (y_test <= MAX_SCORE)
if mask_normal.sum() > 0:
    test_r2_normal = r2_score(y_test[mask_normal], y_test_pred[mask_normal])
    test_mae_normal = mean_absolute_error(y_test[mask_normal], y_test_pred[mask_normal])
    
    print(f"\n   📊 TEST SET (normal matches {MIN_SCORE}-{MAX_SCORE} only):")
    print(f"      Count: {mask_normal.sum()} / {len(y_test)}")
    print(f"      R² = {test_r2_normal:.4f}")
    print(f"      MAE = {test_mae_normal:.2f} runs")

# Accuracy bands
within_20 = np.sum(np.abs(y_test_pred - y_test) <= 20)
within_30 = np.sum(np.abs(y_test_pred - y_test) <= 30)
within_40 = np.sum(np.abs(y_test_pred - y_test) <= 40)

print(f"\n   🎯 ACCURACY BANDS (all test matches):")
print(f"      Within ±20 runs: {within_20}/{len(y_test)} ({100*within_20/len(y_test):.1f}%)")
print(f"      Within ±30 runs: {within_30}/{len(y_test)} ({100*within_30/len(y_test):.1f}%)")
print(f"      Within ±40 runs: {within_40}/{len(y_test)} ({100*within_40/len(y_test):.1f}%)")

# Feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   🏆 TOP 10 FEATURES:")
for idx, (i, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"      {idx:2d}. {row['feature']:35s} {row['importance']:.4f}")

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print(f"\n   Saving model...")

with open('../models/FILTERED_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../models/FILTERED_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../models/FILTERED_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
with open('../models/FILTERED_encoders.pkl', 'wb') as f:
    pickle.dump({'team': le_team, 'venue': le_venue}, f)

feature_importance.to_csv('../results/FILTERED_feature_importance.csv', index=False)

print(f"   ✓ Saved FILTERED_* model files")

# ==============================================================================
# COMPARISON
# ==============================================================================

print(f"\n" + "="*80)
print(f"COMPARISON: CLEAN vs FILTERED")
print(f"="*80)

print(f"\n   Training Data:")
print(f"      CLEAN:    4,772 samples (all scores)")
print(f"      FILTERED: {len(df_train):,} samples ({MIN_SCORE}-{MAX_SCORE} runs only)")

print(f"\n   Test R² (on ALL test data):")
print(f"      CLEAN:    0.178")
print(f"      FILTERED: {test_r2:.3f}")
print(f"      Change:   {test_r2-0.178:+.3f}")

print(f"\n   Test MAE (on ALL test data):")
print(f"      CLEAN:    54.7 runs")
print(f"      FILTERED: {test_mae:.1f} runs")
print(f"      Change:   {test_mae-54.7:+.1f} runs")

if mask_normal.sum() > 0:
    print(f"\n   Test R² (on NORMAL matches only):")
    print(f"      CLEAN:    N/A")
    print(f"      FILTERED: {test_r2_normal:.3f}")
    
    print(f"\n   Test MAE (on NORMAL matches only):")
    print(f"      CLEAN:    N/A")
    print(f"      FILTERED: {test_mae_normal:.1f} runs")

print(f"\n   Prediction Variance:")
print(f"      CLEAN:    32.4 runs (ratio 0.42)")
print(f"      FILTERED: {pred_std:.1f} runs (ratio {pred_std/actual_std:.2f})")

if test_r2 > 0.25 or (mask_normal.sum() > 0 and test_r2_normal > 0.35):
    print(f"\n   ✅ FILTERED MODEL IS BETTER!")
    print(f"   Removing outliers improved learning on typical matches")
else:
    print(f"\n   ⚠️  MIXED RESULTS")
    print(f"   Filtering may help for normal matches but hurts overall R²")

print(f"\n💡 KEY INSIGHT:")
print(f"   By training only on 'normal' matches ({MIN_SCORE}-{MAX_SCORE} runs),")
print(f"   the model focuses on learning patterns for typical scenarios")
print(f"   rather than trying to fit extreme outliers.")

print(f"\n" + "="*80 + "\n")

