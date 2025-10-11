#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN IMPROVED XGBOOST MODEL

Improvements:
1. Less regularization (allow model to learn variance)
2. Interaction features (team quality × venue, form × batting_first, etc.)
3. Better handling of extreme scores
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
print("TRAIN IMPROVED XGBOOST MODEL")
print("="*80)

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n[1/6] Loading data...")
df_train = pd.read_csv('../data/CLEAN_train_dataset.csv')
df_test = pd.read_csv('../data/CLEAN_test_dataset.csv')

print(f"   Training: {df_train.shape}")
print(f"   Test: {df_test.shape}")

# ==============================================================================
# CREATE INTERACTION FEATURES
# ==============================================================================

print("\n[2/6] Creating interaction features...")

def add_interaction_features(df):
    """Add feature interactions that might be predictive"""
    
    # Team quality × Venue quality
    df['team_batting_x_venue'] = df['team_team_avg_batting_avg'] * df['venue_avg_score'] / 250.0
    
    # Form × Batting first (teams in good form score more batting first)
    df['form_x_batting_first'] = df['team_recent_avg_score'] * df['batting_first'] / 250.0
    
    # Batting strength advantage
    df['batting_advantage'] = df['team_team_avg_batting_avg'] - df['opp_team_avg_bowling_economy'] * 5
    
    # Bowling threat
    df['bowling_threat'] = df['opp_team_elite_bowlers'] * df['opp_team_avg_bowling_economy']
    
    # Team balance
    df['team_balance'] = df['team_team_avg_strike_rate'] / 80.0 * df['team_team_batting_depth']
    
    # Venue difficulty
    df['venue_difficulty'] = df['venue_avg_score'] - 250.0  # Deviation from ODI average
    
    # Recent form trend × team quality
    df['quality_x_form'] = df['team_team_avg_batting_avg'] * (df['team_recent_avg_score'] / 250.0)
    
    # Match intensity (h2h matches × elite players)
    df['match_intensity'] = np.log1p(df['h2h_matches_played']) * df['team_team_elite_batsmen']
    
    return df

df_train = add_interaction_features(df_train)
df_test = add_interaction_features(df_test)

print(f"   ✓ Added 8 interaction features")
print(f"   New shape: {df_train.shape}")

# ==============================================================================
# ENCODE CATEGORICAL
# ==============================================================================

print("\n[3/6] Encoding categorical features...")

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

print("\n[4/6] Preparing features...")

drop_cols = ['match_id', 'date', 'team_name', 'opposition_name', 'venue_name', 'total_runs']
feature_cols = [col for col in df_train.columns if col not in drop_cols]

X_train = df_train[feature_cols]
y_train = df_train['total_runs']

X_test = df_test[feature_cols]
y_test = df_test['total_runs']

print(f"   Features: {len(feature_cols)}")
print(f"   Training samples: {len(X_train):,}")

# Fill NaN and scale
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# TRAIN WITH REDUCED REGULARIZATION
# ==============================================================================

print("\n[5/6] Training improved XGBoost...")

# LESS regularization to allow variance
xgb_params = {
    'n_estimators': 500,         # More trees
    'max_depth': 7,              # Deeper (was 5)
    'learning_rate': 0.05,       
    'min_child_weight': 2,       # Less regularization (was 5)
    'subsample': 0.85,           # More data per tree
    'colsample_bytree': 0.85,    # More features per tree
    'gamma': 0.0,                # No minimum loss reduction (was 0.1)
    'reg_alpha': 0.01,           # Less L1 (was 0.1)
    'reg_lambda': 0.5,           # Less L2 (was 1.0)
    'random_state': 42,
    'objective': 'reg:squarederror',
    'n_jobs': -1
}

print(f"\n   Key changes from conservative model:")
print(f"      max_depth: 5 → 7 (allow more complex patterns)")
print(f"      min_child_weight: 5 → 2 (less regularization)")
print(f"      gamma: 0.1 → 0.0 (no minimum loss)")
print(f"      reg_alpha: 0.1 → 0.01 (less L1)")
print(f"      reg_lambda: 1.0 → 0.5 (less L2)")

# Cross-validation
print(f"\n   5-fold cross-validation...")
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
# EVALUATE
# ==============================================================================

print("\n[6/6] Evaluating...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

pred_std = y_test_pred.std()
actual_std = y_test.std()

print(f"\n   📊 TRAINING SET:")
print(f"      R² = {train_r2:.4f}")
print(f"      MAE = {train_mae:.2f} runs")

print(f"\n   📊 TEST SET:")
print(f"      R² = {test_r2:.4f}")
print(f"      MAE = {test_mae:.2f} runs")

print(f"\n   📈 VARIANCE:")
print(f"      Actual std: {actual_std:.1f} runs")
print(f"      Predicted std: {pred_std:.1f} runs")
print(f"      Ratio: {pred_std/actual_std:.2f}")

within_30 = np.sum(np.abs(y_test_pred - y_test) <= 30)
print(f"\n   🎯 Within ±30 runs: {within_30}/{len(y_test)} ({100*within_30/len(y_test):.1f}%)")

# Feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   🏆 TOP 10 FEATURES:")
for idx, (i, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"      {idx:2d}. {row['feature']:40s} {row['importance']:.4f}")

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print(f"\n   Saving model...")

with open('../models/IMPROVED_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../models/IMPROVED_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../models/IMPROVED_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
with open('../models/IMPROVED_encoders.pkl', 'wb') as f:
    pickle.dump({'team': le_team, 'venue': le_venue}, f)

feature_importance.to_csv('../results/IMPROVED_feature_importance.csv', index=False)

print(f"   ✓ Saved IMPROVED_* model files")

# ==============================================================================
# COMPARISON
# ==============================================================================

print(f"\n" + "="*80)
print(f"COMPARISON: CLEAN vs IMPROVED")
print(f"="*80)

print(f"\n   Test R²:")
print(f"      CLEAN:    0.178 (17.8% variance)")
print(f"      IMPROVED: {test_r2:.3f} ({test_r2*100:.1f}% variance)")
print(f"      Change:   {test_r2-0.178:+.3f}")

print(f"\n   Test MAE:")
print(f"      CLEAN:    54.7 runs")
print(f"      IMPROVED: {test_mae:.1f} runs")
print(f"      Change:   {test_mae-54.7:+.1f} runs")

print(f"\n   Prediction Std:")
print(f"      CLEAN:    32.4 runs (ratio 0.42)")
print(f"      IMPROVED: {pred_std:.1f} runs (ratio {pred_std/actual_std:.2f})")

if test_r2 > 0.30:
    print(f"\n   ✅ IMPROVED MODEL IS BETTER!")
    print(f"   Use IMPROVED_* files for deployment")
else:
    print(f"\n   ⚠️  STILL NEEDS WORK")
    print(f"   R² = {test_r2:.3f} is below target of 0.50")

print(f"\n" + "="*80 + "\n")

