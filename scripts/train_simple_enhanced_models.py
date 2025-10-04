"""
Train Simple Enhanced Models
Train models with clean pre-match features only
"""

import pandas as pd
import numpy as np
import time
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_simple_enhanced_models():
    """Train models with simple enhanced features"""
    print("TRAINING SIMPLE ENHANCED MODELS - CLEAN PRE-MATCH FEATURES")
    print("=" * 70)
    
    # Load enhanced datasets
    train_df = pd.read_csv('data/simple_enhanced_train.csv')
    test_df = pd.read_csv('data/simple_enhanced_test.csv')
    
    print(f"Enhanced training data: {train_df.shape}")
    print(f"Enhanced test data: {test_df.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col != 'total_runs']
    X_train = train_df[feature_cols]
    y_train = train_df['total_runs']
    X_test = test_df[feature_cols]
    y_test = test_df['total_runs']
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")
    
    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized")
    
    # Train models
    models = {}
    results = {}
    
    # 1. Linear Regression
    print(f"\n" + "="*60)
    print("TRAINING LINEAR REGRESSION")
    print("="*60)
    
    start_time = time.time()
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(X_train_scaled, y_train)
    
    y_train_pred = lr_model.predict(X_train_scaled)
    y_test_pred = lr_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')
    train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 10) * 100
    test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 10) * 100
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f} runs")
    print(f"Test RMSE: {test_rmse:.2f} runs")
    print(f"Training MAE: {train_mae:.2f} runs")
    print(f"Test MAE: {test_mae:.2f} runs")
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"Training Accuracy (±10 runs): {train_accuracy:.1f}%")
    print(f"Test Accuracy (±10 runs): {test_accuracy:.1f}%")
    
    models['linear_regression'] = lr_model
    results['linear_regression'] = {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'training_time': training_time
    }
    
    # 2. Random Forest
    print(f"\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    start_time = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
    train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 10) * 100
    test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 10) * 100
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f} runs")
    print(f"Test RMSE: {test_rmse:.2f} runs")
    print(f"Training MAE: {train_mae:.2f} runs")
    print(f"Test MAE: {test_mae:.2f} runs")
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"Training Accuracy (±10 runs): {train_accuracy:.1f}%")
    print(f"Test Accuracy (±10 runs): {test_accuracy:.1f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'training_time': training_time,
        'feature_importance': feature_importance
    }
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        print(f"\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)
        
        start_time = time.time()
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        y_train_pred = xgb_model.predict(X_train_scaled)
        y_test_pred = xgb_model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='r2')
        train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 10) * 100
        test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 10) * 100
        
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.2f} runs")
        print(f"Test RMSE: {test_rmse:.2f} runs")
        print(f"Training MAE: {train_mae:.2f} runs")
        print(f"Test MAE: {test_mae:.2f} runs")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"Training Accuracy (±10 runs): {train_accuracy:.1f}%")
        print(f"Test Accuracy (±10 runs): {test_accuracy:.1f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
            'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
            'training_time': training_time,
            'feature_importance': feature_importance
        }
    
    # Compare models
    print(f"\n" + "="*90)
    print("SIMPLE ENHANCED MODEL COMPARISON - CLEAN PRE-MATCH FEATURES")
    print("="*90)
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Test R²': result['test_r2'],
            'Test RMSE': result['test_rmse'],
            'Test MAE': result['test_mae'],
            'Test Accuracy (±10)': result['test_accuracy'],
            'Training Time (s)': result['training_time'],
            'CV R² Mean': result['cv_r2_mean']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['Test R²'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]
    
    print(f"\nBEST MODEL: {best_model['Model']}")
    print(f"Test R²: {best_model['Test R²']:.4f}")
    print(f"Test RMSE: {best_model['Test RMSE']:.2f} runs")
    print(f"Test MAE: {best_model['Test MAE']:.2f} runs")
    print(f"Test Accuracy (±10 runs): {best_model['Test Accuracy (±10)']:.1f}%")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    for model_name, model in models.items():
        with open(f'models/{model_name}_simple_enhanced.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved: models/{model_name}_simple_enhanced.pkl")
    
    # Save scaler
    with open('models/scaler_simple_enhanced.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: models/scaler_simple_enhanced.pkl")
    
    # Save results
    comparison_df.to_csv('models/simple_enhanced_model_comparison.csv', index=False)
    print(f"Comparison saved: models/simple_enhanced_model_comparison.csv")
    
    # Honest assessment
    print(f"\n" + "="*70)
    print("HONEST ASSESSMENT - CLEAN PRE-MATCH FEATURES")
    print("="*70)
    
    best_accuracy = best_model['Test Accuracy (±10)']
    best_r2 = best_model['Test R²']
    best_rmse = best_model['Test RMSE']
    
    print(f"BEST MODEL PERFORMANCE:")
    print(f"  - Test R²: {best_r2:.4f}")
    print(f"  - Test RMSE: {best_rmse:.2f} runs")
    print(f"  - Test Accuracy (±10 runs): {best_accuracy:.1f}%")
    
    if best_accuracy < 30:
        print(f"\nVERDICT: POOR - Model needs significant improvement")
        print(f"  - Consider more feature engineering")
        print(f"  - Try different algorithms")
        print(f"  - Add more data sources")
    elif best_accuracy < 50:
        print(f"\nVERDICT: FAIR - Getting better, usable for basic predictions")
        print(f"  - Good foundation for improvement")
        print(f"  - Can be used with caution")
    elif best_accuracy < 70:
        print(f"\nVERDICT: GOOD - Solid performance for cricket prediction")
        print(f"  - Reliable for most scenarios")
        print(f"  - Ready for frontend integration")
    else:
        print(f"\nVERDICT: EXCELLENT - Outstanding performance")
        print(f"  - Ready for production use")
        print(f"  - Highly reliable predictions")
    
    if best_r2 < 0.3:
        print(f"  - Low R² indicates model needs more work")
    elif best_r2 < 0.6:
        print(f"  - Moderate R² indicates decent model fit")
    else:
        print(f"  - High R² indicates strong model fit")
    
    return models, results, comparison_df, best_model

if __name__ == "__main__":
    models, results, comparison, best = train_simple_enhanced_models()
