#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train on T20-Style ODI Dataset (with weather/pitch features)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import sys
import joblib

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("FINAL TRAINING - T20-STYLE ODI DATASET")
print("="*70)

# Load
df = pd.read_csv('../data/odi_t20_style_dataset.csv')
print(f"\nLoaded: {df.shape}\n")

# Encode categoricals
le_team = LabelEncoder()
le_venue = LabelEncoder()

# Create text mapping for venue (keep venue name for later)
venue_names = df['venue'].unique()
venue_to_name = {i: name for i, name in enumerate(venue_names)}

df['team_encoded'] = le_team.fit_transform(df['team'])
df['opposition_encoded'] = le_team.transform(df['opposition'])
df['venue_encoded'] = le_venue.fit_transform(df['venue'])

# One-hot
for col in ['gender']:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)

# Drop text
df = df.drop(columns=['match_id', 'date', 'venue', 'team', 'opposition', 'gender', 'event_name'])

print(f"After encoding: {df.shape}")

# Split (stratified by score ranges)
df['score_bin'] = pd.cut(df['total_runs'], bins=[0, 150, 200, 250, 300, 600], labels=[0,1,2,3,4])

# Remove any NaN in score_bin
df = df[~df['score_bin'].isna()].copy()

X = df.drop(columns=['total_runs', 'score_bin'])
y = df['total_runs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=df['score_bin'], random_state=42)

print(f"\nTrain: {len(X_train):,} rows (mean: {y_train.mean():.2f})")
print(f"Test: {len(X_test):,} rows (mean: {y_test.mean():.2f})")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Random Forest
print("\n1. Random Forest...")
rf = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=10, random_state=42, n_jobs=-1, verbose=0)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print(f"   Train R²: {r2_score(y_train, rf.predict(X_train_scaled)):.4f}")
print(f"   Test R²: {rf_r2:.4f}")
print(f"   Test MAE: {rf_mae:.2f} runs")

# XGBoost  
print("\n2. XGBoost...")
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

y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

train_r2 = r2_score(y_train, xgb_model.predict(X_train_scaled))

print(f"   Train R²: {train_r2:.4f}")
print(f"   Test R²: {xgb_r2:.4f}")
print(f"   Test MAE: {xgb_mae:.2f} runs")
print(f"   Test RMSE: {xgb_rmse:.2f} runs")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\n🏆 BEST MODEL: {'XGBoost' if xgb_r2 > rf_r2 else 'Random Forest'}")
best_r2 = max(xgb_r2, rf_r2)
best_mae = xgb_mae if xgb_r2 > rf_r2 else rf_mae

print(f"   Test R²: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")
print(f"   Test MAE: {best_mae:.2f} runs")

print(f"\n📊 COMPARISON:")
print(f"   T20: R² = 0.70, MAE = 35 runs")
print(f"   ODI: R² = {best_r2:.2f}, MAE = {best_mae:.2f} runs")

if best_r2 >= 0.60:
    print(f"\n✅ SUCCESS! Comparable to T20!")
    verdict = "EXCELLENT"
elif best_r2 >= 0.50:
    print(f"\n✓ GOOD! Close to T20 performance!")
    verdict = "GOOD"
elif best_r2 >= 0.40:
    print(f"\n⚠️ ACCEPTABLE. Lower than T20 but usable.")
    verdict = "ACCEPTABLE"
else:
    print(f"\n❌ STILL STRUGGLING. ODI harder than T20.")
    verdict = "POOR"

# Save best model
if xgb_r2 > rf_r2:
    joblib.dump(xgb_model, '../models/xgboost_FINAL.pkl')
    joblib.dump(scaler, '../models/scaler_FINAL.pkl')
    print(f"\n✓ Saved XGBoost model (FINAL)")
else:
    joblib.dump(rf, '../models/random_forest_FINAL.pkl')
    joblib.dump(scaler, '../models/scaler_FINAL.pkl')
    print(f"\n✓ Saved Random Forest model (FINAL)")

# Save feature names
joblib.dump(X.columns.tolist(), '../models/feature_names_FINAL.pkl')

print("\n" + "="*70)
print(f"VERDICT: {verdict}")
print("="*70 + "\n")

