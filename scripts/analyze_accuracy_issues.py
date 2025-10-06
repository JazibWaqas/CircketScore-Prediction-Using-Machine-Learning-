#!/usr/bin/env python3
"""
Analyze Model Accuracy Issues
Identify why models are predicting so far from actual scores
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def analyze_accuracy_issues():
    """Analyze why models are so inaccurate"""
    print("🔍 ANALYZING MODEL ACCURACY ISSUES")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv('data/simple_enhanced_train.csv')
    test_df = pd.read_csv('data/simple_enhanced_test.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Analyze target variable distribution
    print(f"\n📊 TARGET VARIABLE ANALYSIS:")
    print(f"Training total_runs - Mean: {train_df['total_runs'].mean():.1f}, Std: {train_df['total_runs'].std():.1f}")
    print(f"Training total_runs - Min: {train_df['total_runs'].min()}, Max: {train_df['total_runs'].max()}")
    print(f"Training total_runs - 25th percentile: {train_df['total_runs'].quantile(0.25):.1f}")
    print(f"Training total_runs - 75th percentile: {train_df['total_runs'].quantile(0.75):.1f}")
    
    print(f"\nTest total_runs - Mean: {test_df['total_runs'].mean():.1f}, Std: {test_df['total_runs'].std():.1f}")
    print(f"Test total_runs - Min: {test_df['total_runs'].min()}, Max: {test_df['total_runs'].max()}")
    
    # Check for high-scoring matches
    high_scoring_train = train_df[train_df['total_runs'] > 180]
    high_scoring_test = test_df[test_df['total_runs'] > 180]
    
    print(f"\n🎯 HIGH-SCORING MATCHES ANALYSIS:")
    print(f"Training matches > 180 runs: {len(high_scoring_train)} ({len(high_scoring_train)/len(train_df)*100:.1f}%)")
    print(f"Test matches > 180 runs: {len(high_scoring_test)} ({len(high_scoring_test)/len(test_df)*100:.1f}%)")
    
    if len(high_scoring_train) > 0:
        print(f"High-scoring training matches - Mean: {high_scoring_train['total_runs'].mean():.1f}")
        print(f"High-scoring training matches - Max: {high_scoring_train['total_runs'].max()}")
    
    # Analyze feature quality
    print(f"\n🔧 FEATURE QUALITY ANALYSIS:")
    feature_cols = [col for col in train_df.columns if col != 'total_runs']
    
    # Check for constant features
    constant_features = []
    low_variance_features = []
    
    for col in feature_cols:
        if train_df[col].dtype in ['object']:
            # For categorical features, check unique values
            unique_ratio = len(train_df[col].unique()) / len(train_df)
            if unique_ratio < 0.01:  # Less than 1% unique values
                low_variance_features.append(col)
        else:
            # For numeric features, check variance
            if train_df[col].std() < 0.01:
                constant_features.append(col)
            elif train_df[col].std() / train_df[col].mean() < 0.01:
                low_variance_features.append(col)
    
    print(f"Constant features: {len(constant_features)}")
    print(f"Low variance features: {len(low_variance_features)}")
    
    if constant_features:
        print(f"Constant features: {constant_features[:5]}...")
    if low_variance_features:
        print(f"Low variance features: {low_variance_features[:5]}...")
    
    # Check for missing values
    missing_counts = train_df.isnull().sum()
    high_missing = missing_counts[missing_counts > len(train_df) * 0.5]
    print(f"Features with >50% missing values: {len(high_missing)}")
    
    # Test model performance
    print(f"\n🤖 MODEL PERFORMANCE ANALYSIS:")
    
    # Prepare features
    X_train = train_df[feature_cols]
    y_train = train_df['total_runs']
    X_test = test_df[feature_cols]
    y_test = test_df['total_runs']
    
    # Handle missing values and convert to numeric
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test Linear Regression
    print(f"\n📈 LINEAR REGRESSION PERFORMANCE:")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    y_train_pred_lr = lr_model.predict(X_train_scaled)
    y_test_pred_lr = lr_model.predict(X_test_scaled)
    
    train_r2_lr = r2_score(y_train, y_train_pred_lr)
    test_r2_lr = r2_score(y_test, y_test_pred_lr)
    train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
    test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
    
    print(f"Training R²: {train_r2_lr:.4f}")
    print(f"Test R²: {test_r2_lr:.4f}")
    print(f"Training RMSE: {train_rmse_lr:.1f}")
    print(f"Test RMSE: {test_rmse_lr:.1f}")
    print(f"Prediction range: {y_test_pred_lr.min():.1f} to {y_test_pred_lr.max():.1f}")
    
    # Test Random Forest
    print(f"\n🌲 RANDOM FOREST PERFORMANCE:")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    y_train_pred_rf = rf_model.predict(X_train_scaled)
    y_test_pred_rf = rf_model.predict(X_test_scaled)
    
    train_r2_rf = r2_score(y_train, y_train_pred_rf)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)
    train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
    test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    
    print(f"Training R²: {train_r2_rf:.4f}")
    print(f"Test R²: {test_r2_rf:.4f}")
    print(f"Training RMSE: {train_rmse_rf:.1f}")
    print(f"Test RMSE: {test_rmse_rf:.1f}")
    print(f"Prediction range: {y_test_pred_rf.min():.1f} to {y_test_pred_rf.max():.1f}")
    
    # Analyze prediction errors
    print(f"\n❌ PREDICTION ERROR ANALYSIS:")
    
    # High-scoring match errors
    high_scoring_indices = test_df[test_df['total_runs'] > 180].index
    if len(high_scoring_indices) > 0:
        high_scoring_actual = y_test.iloc[high_scoring_indices]
        high_scoring_pred_lr = y_test_pred_lr[high_scoring_indices]
        high_scoring_pred_rf = y_test_pred_rf[high_scoring_indices]
        
        print(f"High-scoring matches (>180 runs) error analysis:")
        print(f"Actual scores: {high_scoring_actual.tolist()[:5]}")
        print(f"LR predictions: {high_scoring_pred_lr[:5]}")
        print(f"RF predictions: {high_scoring_pred_rf[:5]}")
        
        lr_error_high = np.mean(np.abs(high_scoring_actual - high_scoring_pred_lr))
        rf_error_high = np.mean(np.abs(high_scoring_actual - high_scoring_pred_rf))
        
        print(f"LR MAE on high-scoring matches: {lr_error_high:.1f}")
        print(f"RF MAE on high-scoring matches: {rf_error_high:.1f}")
    
    # Feature importance analysis
    print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nBottom 10 least important features:")
    for i, row in feature_importance.tail(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")
    
    # Identify key issues
    print(f"\n🚨 IDENTIFIED ISSUES:")
    
    issues = []
    
    if test_r2_lr < 0:
        issues.append("Linear Regression has negative R² - model is worse than predicting the mean")
    
    if test_r2_rf < 0.3:
        issues.append("Random Forest R² is very low - poor predictive power")
    
    if train_rmse_lr > 50:
        issues.append("Linear Regression RMSE is very high - large prediction errors")
    
    if len(high_scoring_train) < len(train_df) * 0.1:
        issues.append("Very few high-scoring matches in training data - models can't learn high scores")
    
    if len(constant_features) > 10:
        issues.append("Many constant features - no predictive value")
    
    if len(low_variance_features) > 20:
        issues.append("Many low-variance features - little predictive information")
    
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    if not issues:
        print("  No major issues identified - models should be performing better")
    
    return {
        'train_r2_lr': train_r2_lr,
        'test_r2_lr': test_r2_lr,
        'train_r2_rf': train_r2_rf,
        'test_r2_rf': test_r2_rf,
        'issues': issues,
        'high_scoring_ratio': len(high_scoring_train) / len(train_df)
    }

if __name__ == "__main__":
    results = analyze_accuracy_issues()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results['test_r2_lr'] < 0:
        print("❌ Linear Regression: COMPLETELY BROKEN (negative R²)")
    elif results['test_r2_lr'] < 0.3:
        print("⚠️ Linear Regression: POOR PERFORMANCE")
    else:
        print("✅ Linear Regression: ACCEPTABLE")
    
    if results['test_r2_rf'] < 0.3:
        print("⚠️ Random Forest: POOR PERFORMANCE")
    elif results['test_r2_rf'] < 0.6:
        print("🔶 Random Forest: MODERATE PERFORMANCE")
    else:
        print("✅ Random Forest: GOOD PERFORMANCE")
    
    print(f"\nHigh-scoring matches in training: {results['high_scoring_ratio']*100:.1f}%")
    
    if results['high_scoring_ratio'] < 0.1:
        print("🚨 CRITICAL: Too few high-scoring matches for models to learn from!")
