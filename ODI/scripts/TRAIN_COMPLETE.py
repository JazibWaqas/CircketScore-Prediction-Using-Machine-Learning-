#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train on COMPLETE dataset (career + contextual)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import sys
import joblib

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load
df = pd.read_csv('../data/odi_complete_dataset.csv')
print(f"Loaded COMPLETE dataset: {df.shape}\n")

# Encode
le_team = LabelEncoder()
le_venue = LabelEncoder()

df['team_encoded'] = le_team.fit_transform(df['team'])
df['opposition_encoded'] = le_team.transform(df['opposition'])
df['venue_encoded'] = le_venue.fit_transform(df['venue'])

# One-hot
for col in ['gender', 'match_type']:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)

# Drop text
df = df.drop(columns=['match_id', 'date', 'venue', 'team', 'opposition', 'gender', 'match_type', 'event_name'])

print(f"After encoding: {df.shape}")

# Split
df['score_bin'] = pd.cut(df['total_runs'], bins=[0, 150, 200, 250, 300, 600], labels=[0,1,2,3,4])
df = df[~df['score_bin'].isna()].copy()

X = df.drop(columns=['total_runs', 'score_bin'])
y = df['total_runs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=df['score_bin'], random_state=42)

print(f"\nTrain: {len(X_train):,} rows")
print(f"Test: {len(X_test):,} rows")
print(f"Features: {X.shape[1]}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
print("\n" + "="*70)
print("TRAINING XGBOOST ON COMPLETE DATASET")
print("="*70)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    verbose=0
)

xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

# Evaluate
y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n📊 RESULTS:")
print(f"  Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"  Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"  Test MAE: {test_mae:.2f} runs")
print(f"  Test RMSE: {test_rmse:.2f} runs")
print(f"  Overfitting: {train_r2 - test_r2:.4f}")

print(f"\n📈 COMPARISON:")
print(f"  T20-style only: R² = 0.69, MAE = 28.67")
print(f"  Complete (career+context): R² = {test_r2:.2f}, MAE = {test_mae:.2f}")

if test_r2 >= 0.65:
    print(f"\n✅ SUCCESS! Accuracy maintained with career stats!")
    print(f"   Now we have BOTH:")
    print(f"   - High accuracy (contextual features)")
    print(f"   - Player swap capability (career features)")
elif test_r2 >= 0.55:
    print(f"\n✓ GOOD! Slight drop but still usable")
else:
    print(f"\n⚠️ Accuracy dropped with more features")

# Get feature importance
importances = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n🏆 TOP 15 FEATURES:")
for i, row in feature_importance.head(15).iterrows():
    print(f"  {i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

# Check if career features are important
career_features = [f for f in X.columns if 'team_batting_avg' in f or 'star_players' in f or 'elite_' in f]
career_importance = feature_importance[feature_importance['feature'].isin(career_features)]

print(f"\n📊 Career Feature Importance:")
total_career_importance = career_importance['importance'].sum()
print(f"  Total importance from career features: {total_career_importance:.4f}")
print(f"  Top career features:")
for _, row in career_importance.head(5).iterrows():
    print(f"    - {row['feature']:35s} {row['importance']:.4f}")

# Save
joblib.dump(xgb_model, '../models/xgboost_COMPLETE.pkl')
joblib.dump(scaler, '../models/scaler_COMPLETE.pkl')
joblib.dump(X.columns.tolist(), '../models/feature_names_COMPLETE.pkl')
joblib.dump(le_team, '../models/team_encoder.pkl')
joblib.dump(le_venue, '../models/venue_encoder.pkl')

print(f"\n✓ Saved COMPLETE models")

print("\n" + "="*70)
print("VERDICT:")
if test_r2 >= 0.65:
    print("✅ USE THIS MODEL - Has career stats AND maintains accuracy!")
elif test_r2 >= 0.55:
    print("⚠️ Slight accuracy loss but player swaps work")
else:
    print("❌ Too much accuracy loss, need to choose one approach")
print("="*70 + "\n")

