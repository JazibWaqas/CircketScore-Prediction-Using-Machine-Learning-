"""
Train Proper Models with Pre-Match Features Only
No data leakage - realistic cricket score prediction
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

def train_proper_models():
    """Train models with proper pre-match features only"""
    print("TRAINING PROPER MODELS - NO DATA LEAKAGE")
    print("=" * 60)
    
    # Load clean datasets
    train_df = pd.read_csv('data/proper_train_dataset.csv')
    test_df = pd.read_csv('data/proper_test_dataset.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col != 'total_runs']
    X_train = train_df[feature_cols]
    y_train = train_df['total_runs']
    X_test = test_df[feature_cols]
    y_test = test_df['total_runs']
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}...")
    
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
    print(f"\n" + "="*50)
    print("TRAINING LINEAR REGRESSION")
    print("="*50)
    
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
    print(f"\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
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
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'training_time': training_time
    }
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        print(f"\n" + "="*50)
        print("TRAINING XGBOOST")
        print("="*50)
        
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
        
        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
            'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
            'training_time': training_time
        }
    
    # Compare models
    print(f"\n" + "="*80)
    print("MODEL COMPARISON - PROPER PRE-MATCH FEATURES")
    print("="*80)
    
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
        with open(f'models/{model_name}_proper.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved: models/{model_name}_proper.pkl")
    
    # Save scaler
    with open('models/scaler_proper.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: models/scaler_proper.pkl")
    
    # Save results
    comparison_df.to_csv('models/proper_model_comparison.csv', index=False)
    print(f"Comparison saved: models/proper_model_comparison.csv")
    
    # Honest assessment
    print(f"\n" + "="*60)
    print("HONEST ASSESSMENT")
    print("="*60)
    
    best_accuracy = best_model['Test Accuracy (±10)']
    best_r2 = best_model['Test R²']
    best_rmse = best_model['Test RMSE']
    
    print(f"BEST MODEL PERFORMANCE:")
    print(f"  - Test R²: {best_r2:.4f}")
    print(f"  - Test RMSE: {best_rmse:.2f} runs")
    print(f"  - Test Accuracy (±10 runs): {best_accuracy:.1f}%")
    
    if best_accuracy < 30:
        print(f"\nVERDICT: POOR - Model needs significant improvement")
    elif best_accuracy < 50:
        print(f"\nVERDICT: FAIR - Model is usable but needs work")
    elif best_accuracy < 70:
        print(f"\nVERDICT: GOOD - Model is useful for predictions")
    else:
        print(f"\nVERDICT: EXCELLENT - Model performs very well")
    
    if best_r2 < 0.3:
        print(f"  - Low R² indicates poor model fit")
    elif best_r2 < 0.6:
        print(f"  - Moderate R² indicates fair model fit")
    else:
        print(f"  - Good R² indicates strong model fit")
    
    return models, results, comparison_df, best_model

if __name__ == "__main__":
    models, results, comparison, best = train_proper_models()
