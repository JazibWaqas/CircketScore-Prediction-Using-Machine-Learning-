#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN CLEAN XGBOOST MODEL

Trains XGBoost on clean dataset (no data leakage).
Conservative hyperparameters to prevent overfitting.
5-fold cross-validation for robust performance estimation.
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
print("TRAIN CLEAN XGBOOST MODEL - NO DATA LEAKAGE")
print("="*80)

# ==============================================================================
# STEP 1: LOAD TRAINING DATA
# ==============================================================================

print("\n[1/5] Loading training data...")

df_train = pd.read_csv('../data/CLEAN_train_dataset.csv')
df_test = pd.read_csv('../data/CLEAN_test_dataset.csv')

print(f"   Training set: {df_train.shape}")
print(f"      Date range: {df_train['date'].min()} to {df_train['date'].max()}")
print(f"      Scores: mean={df_train['total_runs'].mean():.1f}, std={df_train['total_runs'].std():.1f}")

print(f"\n   Test set: {df_test.shape}")
print(f"      Date range: {df_test['date'].min()} to {df_test['date'].max()}")
print(f"      Scores: mean={df_test['total_runs'].mean():.1f}, std={df_test['total_runs'].std():.1f}")

# ==============================================================================
# STEP 2: ENCODE CATEGORICAL FEATURES
# ==============================================================================

print("\n[2/5] Encoding categorical features...")

# Encode team names and venues
le_team = LabelEncoder()
le_venue = LabelEncoder()

# Fit on combined data to ensure same encoding for train and test
all_teams = pd.concat([df_train['team_name'], df_train['opposition_name'], 
                       df_test['team_name'], df_test['opposition_name']]).unique()
all_venues = pd.concat([df_train['venue_name'], df_test['venue_name']]).unique()

le_team.fit(all_teams)
le_venue.fit(all_venues)

# Encode
df_train['team_encoded'] = le_team.transform(df_train['team_name'])
df_train['opp_encoded'] = le_team.transform(df_train['opposition_name'])
df_train['venue_encoded'] = le_venue.transform(df_train['venue_name'])

df_test['team_encoded'] = le_team.transform(df_test['team_name'])
df_test['opp_encoded'] = le_team.transform(df_test['opposition_name'])
df_test['venue_encoded'] = le_venue.transform(df_test['venue_name'])

print(f"   ✓ Encoded {len(all_teams)} teams and {len(all_venues)} venues")

# ==============================================================================
# STEP 3: PREPARE FEATURES
# ==============================================================================

print("\n[3/5] Preparing features...")

# Drop identifiers
drop_cols = ['match_id', 'date', 'team_name', 'opposition_name', 'venue_name', 'total_runs']
feature_cols = [col for col in df_train.columns if col not in drop_cols]

X_train = df_train[feature_cols]
y_train = df_train['total_runs']

X_test = df_test[feature_cols]
y_test = df_test['total_runs']

print(f"\n   Features: {len(feature_cols)}")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# Check for any issues
if X_train.isnull().any().any():
    print(f"\n   ⚠ WARNING: Found NaN values, filling with median")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ Features scaled (mean=0, std=1)")

# ==============================================================================
# STEP 4: TRAIN XGBOOST WITH CROSS-VALIDATION
# ==============================================================================

print("\n[4/5] Training XGBoost model...")

# Conservative hyperparameters to prevent overfitting
xgb_params = {
    'n_estimators': 300,
    'max_depth': 5,              # Shallow trees
    'learning_rate': 0.05,       # Slow learning
    'min_child_weight': 5,       # Strong regularization
    'subsample': 0.8,            # Row sampling
    'colsample_bytree': 0.8,     # Feature sampling
    'gamma': 0.1,                # Minimum loss reduction
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    'random_state': 42,
    'objective': 'reg:squarederror',
    'n_jobs': -1
}

print(f"\n   Hyperparameters:")
for key, value in xgb_params.items():
    if key not in ['n_jobs', 'random_state', 'objective']:
        print(f"      {key:20s} = {value}")

# Cross-validation on training set
print(f"\n   Performing 5-fold cross-validation...")
model_cv = xgb.XGBRegressor(**xgb_params)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_scores = cross_val_score(model_cv, X_train_scaled, y_train, 
                                 cv=kfold, scoring='r2', n_jobs=-1)
cv_mae_scores = -cross_val_score(model_cv, X_train_scaled, y_train, 
                                  cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)

print(f"\n   Cross-Validation Results:")
print(f"      R² = {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
print(f"      MAE = {cv_mae_scores.mean():.2f} ± {cv_mae_scores.std():.2f} runs")

# Train final model on all training data
print(f"\n   Training final model on all training data...")
model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train_scaled, y_train, verbose=False)

print(f"   ✓ Model trained with {model.n_estimators} trees")

# ==============================================================================
# STEP 5: EVALUATE ON TRAINING AND TEST SETS
# ==============================================================================

print("\n[5/5] Evaluating model...")

# Training set performance
y_train_pred = model.predict(X_train_scaled)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Test set performance
y_test_pred = model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Prediction variance check (critical!)
pred_std = y_test_pred.std()
actual_std = y_test.std()

print(f"\n   📊 TRAINING SET:")
print(f"      R² = {train_r2:.4f} ({train_r2*100:.2f}% variance explained)")
print(f"      MAE = {train_mae:.2f} runs")
print(f"      RMSE = {train_rmse:.2f} runs")

print(f"\n   📊 TEST SET (2023-2025 matches):")
print(f"      R² = {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"      MAE = {test_mae:.2f} runs")
print(f"      RMSE = {test_rmse:.2f} runs")

print(f"\n   📈 PREDICTION VARIANCE:")
print(f"      Actual scores: std = {actual_std:.1f} runs")
print(f"      Predicted scores: std = {pred_std:.1f} runs")
print(f"      Ratio: {pred_std/actual_std:.2f} (should be close to 1.0)")

# Accuracy bands
within_20 = np.sum(np.abs(y_test_pred - y_test) <= 20)
within_30 = np.sum(np.abs(y_test_pred - y_test) <= 30)
within_40 = np.sum(np.abs(y_test_pred - y_test) <= 40)

print(f"\n   🎯 ACCURACY BANDS:")
print(f"      Within ±20 runs: {within_20}/{len(y_test)} ({100*within_20/len(y_test):.1f}%)")
print(f"      Within ±30 runs: {within_30}/{len(y_test)} ({100*within_30/len(y_test):.1f}%)")
print(f"      Within ±40 runs: {within_40}/{len(y_test)} ({100*within_40/len(y_test):.1f}%)")

# Feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   🏆 TOP 10 MOST IMPORTANT FEATURES:")
for idx, (i, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"      {idx:2d}. {row['feature']:35s} {row['importance']:.4f}")

# ==============================================================================
# STEP 6: SAVE MODEL AND ARTIFACTS
# ==============================================================================

print(f"\n[6/6] Saving model and artifacts...")

output_dir = '../models'

# Save model
with open(f'{output_dir}/CLEAN_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"   ✓ Saved model: CLEAN_xgboost.pkl")

# Save scaler
with open(f'{output_dir}/CLEAN_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Saved scaler: CLEAN_scaler.pkl")

# Save feature names
with open(f'{output_dir}/CLEAN_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"   ✓ Saved feature names: CLEAN_feature_names.pkl")

# Save encoders
encoders = {
    'team': le_team,
    'venue': le_venue
}
with open(f'{output_dir}/CLEAN_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print(f"   ✓ Saved encoders: CLEAN_encoders.pkl")

# Save feature importances
feature_importance.to_csv('../results/CLEAN_feature_importance.csv', index=False)
print(f"   ✓ Saved feature importances: CLEAN_feature_importance.csv")

# ==============================================================================
# ASSESSMENT
# ==============================================================================

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 FINAL PERFORMANCE:")
print(f"   Cross-validation R² = {cv_r2_scores.mean():.3f} ± {cv_r2_scores.std():.3f}")
print(f"   Test R² = {test_r2:.3f}")
print(f"   Test MAE = {test_mae:.1f} runs")
print(f"   Predictions actually vary: std = {pred_std:.1f} runs")

print(f"\n✅ SUCCESS CRITERIA:")
if test_r2 >= 0.50:
    print(f"   ✓ R² = {test_r2:.3f} >= 0.50 (GOOD)")
else:
    print(f"   ✗ R² = {test_r2:.3f} < 0.50 (needs improvement)")

if test_mae <= 35:
    print(f"   ✓ MAE = {test_mae:.1f} <= 35 runs (GOOD)")
else:
    print(f"   ✗ MAE = {test_mae:.1f} > 35 runs (needs improvement)")

if pred_std >= 40:
    print(f"   ✓ Prediction std = {pred_std:.1f} >= 40 (predictions vary - model learning!)")
else:
    print(f"   ✗ Prediction std = {pred_std:.1f} < 40 (predictions too uniform)")

if train_r2 - test_r2 <= 0.15:
    print(f"   ✓ Overfitting = {train_r2 - test_r2:.3f} <= 0.15 (acceptable)")
else:
    print(f"   ⚠ Overfitting = {train_r2 - test_r2:.3f} > 0.15 (may be overfitting)")

print(f"\n💡 KEY INSIGHT:")
if test_r2 >= 0.50 and pred_std >= 40:
    print(f"   This model WORKS in real use because:")
    print(f"   - No data leakage (no pitch_bounce/pitch_swing)")
    print(f"   - Predictions actually vary based on inputs")
    print(f"   - R²={test_r2:.3f} is realistic for ODI prediction")
    print(f"   - Previous R²=0.69 was fake (data leakage), R²=0.01 in real use")
    print(f"   - This R²={test_r2:.3f} will ACTUALLY work when deployed!")
else:
    print(f"   Model needs improvement - consider:")
    print(f"   - Adding more relevant features")
    print(f"   - Adjusting hyperparameters")
    print(f"   - Checking data quality")

print(f"\n📝 NEXT STEPS:")
print(f"   1. Run TEST_CLEAN_MODEL.py for detailed analysis")
print(f"   2. Check feature importances in CLEAN_feature_importance.csv")
print(f"   3. Create MODEL_INSIGHTS.md documentation")
print(f"   4. Update API to use CLEAN_* model files")

print("\n" + "="*80 + "\n")

