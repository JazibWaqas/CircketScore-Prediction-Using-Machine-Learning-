"""
Simple script to run machine learning models on the cricket dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def run_ml_models():
    """Run machine learning models on the cricket dataset"""
    print("🏏 CRICKET SCORE PREDICTION - ML MODELS")
    print("="*50)
    
    # Load the ML-ready dataset
    print("Loading ML-ready dataset...")
    df = pd.read_csv('ml_ready_fixed_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'total_runs']
    X = df[feature_columns]
    y = df['total_runs']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Model 1: Linear Regression
    print("\n1. Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print(f"   R² Score: {lr_r2:.4f}")
    print(f"   RMSE: {lr_rmse:.2f}")
    
    # Model 2: Random Forest
    print("\n2. Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"   R² Score: {rf_r2:.4f}")
    print(f"   RMSE: {rf_rmse:.2f}")
    
    # Model 3: XGBoost
    print("\n3. XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    print(f"   R² Score: {xgb_r2:.4f}")
    print(f"   RMSE: {xgb_rmse:.2f}")
    
    # Model comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Linear Regression: R² = {lr_r2:.4f}, RMSE = {lr_rmse:.2f}")
    print(f"Random Forest:     R² = {rf_r2:.4f}, RMSE = {rf_rmse:.2f}")
    print(f"XGBoost:          R² = {xgb_r2:.4f}, RMSE = {xgb_rmse:.2f}")
    
    # Best model
    best_model = "XGBoost" if xgb_r2 > rf_r2 else "Random Forest"
    print(f"\nBest Model: {best_model}")
    
    # Sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    sample_indices = np.random.choice(X_test.index, 5, replace=False)
    for i, idx in enumerate(sample_indices):
        actual = y_test.loc[idx]
        predicted = xgb_pred[X_test.index == idx][0]
        error = abs(predicted - actual)
        print(f"Sample {i+1}: Predicted = {predicted:.1f}, Actual = {actual:.1f}, Error = {error:.1f}")
    
    print("\n✅ ML models completed successfully!")
    return lr, rf, xgb_model

if __name__ == "__main__":
    models = run_ml_models()
