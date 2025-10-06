#!/usr/bin/env python3
"""
Debug Prediction Consistency
Test if models give consistent predictions for the same inputs
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def debug_prediction_consistency():
    """Debug why models give inconsistent predictions"""
    print("DEBUGGING PREDICTION CONSISTENCY")
    print("=" * 50)
    
    # Load data
    train_df = pd.read_csv('data/simple_enhanced_train.csv')
    test_df = pd.read_csv('data/simple_enhanced_test.csv')
    
    # Prepare features
    feature_cols = [col for col in train_df.columns if col != 'total_runs']
    X_train = train_df[feature_cols]
    y_train = train_df['total_runs']
    X_test = test_df[feature_cols]
    y_test = test_df['total_runs']
    
    # Handle missing values and convert to numeric
    print(f"Original data types:")
    print(X_train.dtypes.value_counts())
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Fill NaN values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"After conversion:")
    print(X_train.dtypes.value_counts())
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Test with a single sample multiple times
    test_sample = X_test_scaled[0:1]  # First test sample
    print(f"\nTesting with first test sample:")
    print(f"Sample features shape: {test_sample.shape}")
    
    # Test Linear Regression (should be deterministic)
    print(f"\n" + "="*40)
    print("TESTING LINEAR REGRESSION CONSISTENCY")
    print("="*40)
    
    predictions_lr = []
    for i in range(5):
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        pred = lr_model.predict(test_sample)[0]
        predictions_lr.append(pred)
        print(f"Run {i+1}: {pred:.6f}")
    
    lr_std = np.std(predictions_lr)
    print(f"Linear Regression - Std Dev: {lr_std:.10f}")
    print(f"Linear Regression - Consistent: {lr_std < 1e-10}")
    
    # Test Random Forest (should be deterministic with random_state)
    print(f"\n" + "="*40)
    print("TESTING RANDOM FOREST CONSISTENCY")
    print("="*40)
    
    predictions_rf = []
    for i in range(5):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        pred = rf_model.predict(test_sample)[0]
        predictions_rf.append(pred)
        print(f"Run {i+1}: {pred:.6f}")
    
    rf_std = np.std(predictions_rf)
    print(f"Random Forest - Std Dev: {rf_std:.10f}")
    print(f"Random Forest - Consistent: {rf_std < 1e-10}")
    
    # Test XGBoost (should be deterministic with random_state)
    if XGBOOST_AVAILABLE:
        print(f"\n" + "="*40)
        print("TESTING XGBOOST CONSISTENCY")
        print("="*40)
        
        predictions_xgb = []
        for i in range(5):
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            pred = xgb_model.predict(test_sample)[0]
            predictions_xgb.append(pred)
            print(f"Run {i+1}: {pred:.6f}")
        
        xgb_std = np.std(predictions_xgb)
        print(f"XGBoost - Std Dev: {xgb_std:.10f}")
        print(f"XGBoost - Consistent: {xgb_std < 1e-10}")
    
    # Test if data is consistent
    print(f"\n" + "="*40)
    print("TESTING DATA CONSISTENCY")
    print("="*40)
    
    # Check if data changes between loads
    train_df2 = pd.read_csv('data/simple_enhanced_train.csv')
    test_df2 = pd.read_csv('data/simple_enhanced_test.csv')
    
    train_identical = train_df.equals(train_df2)
    test_identical = test_df.equals(test_df2)
    
    print(f"Train data identical between loads: {train_identical}")
    print(f"Test data identical between loads: {test_identical}")
    
    # Check for any random elements in the data
    print(f"\nChecking for random elements in data...")
    
    # Look for any columns that might contain random values
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:10]:  # Check first 10 numeric columns
        values = train_df[col].dropna()
        if len(values) > 0:
            # Check if values are integers (might indicate random generation)
            is_integer = np.allclose(values, values.astype(int))
            unique_ratio = len(values.unique()) / len(values)
            print(f"  {col}: {len(values)} values, {unique_ratio:.3f} unique ratio, integer-like: {is_integer}")
    
    # Check for duplicate rows
    train_duplicates = train_df.duplicated().sum()
    test_duplicates = test_df.duplicated().sum()
    
    print(f"\nTrain data duplicates: {train_duplicates}")
    print(f"Test data duplicates: {test_duplicates}")
    
    # Check for NaN values
    train_nan = train_df.isnull().sum().sum()
    test_nan = test_df.isnull().sum().sum()
    
    print(f"Train data NaN values: {train_nan}")
    print(f"Test data NaN values: {test_nan}")
    
    return {
        'linear_regression_consistent': lr_std < 1e-10,
        'random_forest_consistent': rf_std < 1e-10,
        'xgboost_consistent': xgb_std < 1e-10 if XGBOOST_AVAILABLE else None,
        'data_consistent': train_identical and test_identical,
        'linear_regression_std': lr_std,
        'random_forest_std': rf_std,
        'xgboost_std': xgb_std if XGBOOST_AVAILABLE else None
    }

if __name__ == "__main__":
    results = debug_prediction_consistency()
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if results['linear_regression_consistent']:
        print("✅ Linear Regression: CONSISTENT")
    else:
        print("❌ Linear Regression: INCONSISTENT")
    
    if results['random_forest_consistent']:
        print("✅ Random Forest: CONSISTENT")
    else:
        print("❌ Random Forest: INCONSISTENT")
    
    if results['xgboost_consistent'] is not None:
        if results['xgboost_consistent']:
            print("✅ XGBoost: CONSISTENT")
        else:
            print("❌ XGBoost: INCONSISTENT")
    
    if results['data_consistent']:
        print("✅ Data: CONSISTENT")
    else:
        print("❌ Data: INCONSISTENT")
