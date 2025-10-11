#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN PROGRESSIVE ODI MODEL

Trains single unified XGBoost model on data from all match stages.
Model learns to adapt to available information at each stage.
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
print("TRAIN PROGRESSIVE ODI MODEL")
print("="*80)

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n[1/5] Loading data...")

df_train = pd.read_csv('../ODI_Progressive/data/progressive_train.csv')
df_test = pd.read_csv('../ODI_Progressive/data/progressive_test.csv')

print(f"   Training: {df_train.shape}")
print(f"      Date range: {df_train['date'].min()} to {df_train['date'].max()}")
print(f"      Checkpoints: {sorted(df_train['checkpoint'].unique())}")

print(f"\n   Test: {df_test.shape}")
print(f"      Date range: {df_test['date'].min()} to {df_test['date'].max()}")

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

print(f"   ✓ Encoded {len(all_teams)} teams, {len(all_venues)} venues")

# ==============================================================================
# PREPARE FEATURES
# ==============================================================================

print("\n[3/5] Preparing features...")

drop_cols = ['match_id', 'date', 'checkpoint', 'team_name', 'opposition_name', 'venue_name', 'final_score']
feature_cols = [col for col in df_train.columns if col not in drop_cols]

X_train = df_train[feature_cols]
y_train = df_train['final_score']

X_test = df_test[feature_cols]
y_test = df_test['final_score']

print(f"   Features: {len(feature_cols)}")
print(f"   Feature list: {feature_cols[:10]}...")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# Handle NaN
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ Scaled features")

# ==============================================================================
# TRAIN MODEL
# ==============================================================================

print("\n[4/5] Training XGBoost model...")

# Check GPU availability
try:
    tree_method = 'gpu_hist'
    test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1)
    test_model.fit(X_train_scaled[:100], y_train[:100])
    print(f"   ✓ GPU available, using gpu_hist")
except:
    tree_method = 'hist'
    print(f"   ⚠ GPU not available, using CPU")

xgb_params = {
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 0.7,
    'random_state': 42,
    'tree_method': tree_method,
    'n_jobs': -1
}

# Cross-validation
print(f"\n   5-fold cross-validation...")
model_cv = xgb.XGBRegressor(**xgb_params)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(model_cv, X_train_scaled, y_train, cv=kfold, scoring='r2', n_jobs=-1)
cv_mae = -cross_val_score(model_cv, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)

print(f"   CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"   CV MAE = {cv_mae.mean():.2f} ± {cv_mae.std():.2f} runs")

# Train final model
print(f"\n   Training final model...")
model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train_scaled, y_train, verbose=False)

print(f"   ✓ Model trained")

# ==============================================================================
# EVALUATE
# ==============================================================================

print("\n[5/5] Evaluating...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n   📊 OVERALL PERFORMANCE:")
print(f"      Train R² = {train_r2:.4f}")
print(f"      Test R² = {test_r2:.4f}")
print(f"      Test MAE = {test_mae:.2f} runs")

# Performance by checkpoint
print(f"\n   📊 PERFORMANCE BY STAGE:")
for cp in sorted(df_test['checkpoint'].unique()):
    mask = df_test['checkpoint'] == cp
    if mask.sum() > 0:
        stage_r2 = r2_score(y_test[mask], y_test_pred[mask])
        stage_mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
        stage_name = f"Ball {cp:3d}"
        if cp == 0:
            stage_name += " (Pre-match)"
        elif cp >= 240:
            stage_name += " (Late)"
        print(f"      {stage_name}: R² = {stage_r2:.4f}, MAE = {stage_mae:.2f} runs")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   🏆 TOP 10 FEATURES:")
for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"      {idx:2d}. {row['feature']:35s} {row['importance']:.4f}")

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print(f"\n   Saving model...")

with open('../ODI_Progressive/models/progressive_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../ODI_Progressive/models/progressive_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../ODI_Progressive/models/progressive_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
with open('../ODI_Progressive/models/progressive_encoders.pkl', 'wb') as f:
    pickle.dump({'team': le_team, 'venue': le_venue}, f)

feature_importance.to_csv('../ODI_Progressive/results/feature_importance.csv', index=False)

print(f"   ✓ Model saved")

print("\n" + "="*80)
print(f"TRAINING COMPLETE!")
print(f"   Overall R² = {test_r2:.4f}")
print(f"   Run 3_validate_model.py for detailed analysis")
print("="*80 + "\n")

