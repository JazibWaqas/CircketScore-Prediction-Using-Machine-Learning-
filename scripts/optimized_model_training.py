"""
Optimized Model Training Pipeline
Clean pre-match features, efficient training, GPU acceleration
"""

import pandas as pd
import numpy as np
import time
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost with GPU support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

# Try to detect GPU
def detect_gpu():
    """Detect if GPU is available for XGBoost"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def prepare_clean_data():
    """Prepare clean dataset with only pre-match features"""
    print("Preparing Clean Dataset")
    print("=" * 40)
    
    # Load clean training data
    train_df = pd.read_csv('data/clean_train_dataset.csv')
    test_df = pd.read_csv('data/test_dataset.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Identify features to use (exclude target and string columns)
    exclude_cols = ['total_runs', 'toss_winner', 'toss_decision', 'match_winner', 
                   'player_of_match', 'season', 'event_name', 'gender']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare features and target
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['total_runs']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['total_runs']
    
    # Convert team_player_ids from string to numeric (sum of player IDs)
    if 'team_player_ids' in feature_cols:
        def process_player_ids(ids_str):
            try:
                if isinstance(ids_str, str) and ids_str.startswith('['):
                    # Extract numbers from string representation of list
                    import re
                    numbers = re.findall(r'\d+', ids_str)
                    return sum(int(num) for num in numbers)
                else:
                    return 0
            except:
                return 0
        
        X_train['team_player_ids'] = X_train['team_player_ids'].apply(process_player_ids)
        X_test['team_player_ids'] = X_test['team_player_ids'].apply(process_player_ids)
    
    # Ensure all columns are numeric
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression model"""
    print("\n" + "="*50)
    print("TRAINING LINEAR REGRESSION")
    print("="*50)
    
    start_time = time.time()
    
    # Train model
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')
    
    # Accuracy within ±10 runs
    train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 10) * 100
    test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 10) * 100
    
    training_time = time.time() - start_time
    
    # Display results
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
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/linear_regression_clean.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    return {
        'model': lr_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time
    }

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
    start_time = time.time()
    
    # Train model with optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    
    # Accuracy within ±10 runs
    train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 10) * 100
    test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 10) * 100
    
    training_time = time.time() - start_time
    
    # Display results
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
        'feature': range(len(rf_model.feature_importances_)),
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model
    with open('models/random_forest_clean.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    return {
        'model': rf_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'feature_importance': feature_importance
    }

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model with GPU acceleration if available"""
    print("\n" + "="*50)
    print("TRAINING XGBOOST")
    print("="*50)
    
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping...")
        return None
    
    start_time = time.time()
    
    # Check for GPU
    gpu_available = detect_gpu()
    print(f"GPU available: {gpu_available}")
    
    # Configure XGBoost parameters
    if gpu_available:
        print("Using GPU acceleration for XGBoost")
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'random_state': 42
        }
    else:
        print("Using CPU for XGBoost")
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Train model
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
    
    # Accuracy within ±10 runs
    train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 10) * 100
    test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 10) * 100
    
    training_time = time.time() - start_time
    
    # Display results
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
        'feature': range(len(xgb_model.feature_importances_)),
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model
    with open('models/xgboost_clean.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    return {
        'model': xgb_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'feature_importance': feature_importance,
        'gpu_used': gpu_available
    }

def compare_models(results):
    """Compare all trained models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    comparison_data = []
    for model_name, result in results.items():
        if result is not None:
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
    
    # Save comparison
    comparison_df.to_csv('models/model_comparison_clean.csv', index=False)
    print(f"\nComparison saved: models/model_comparison_clean.csv")
    
    return comparison_df, best_model

def main():
    """Main training pipeline"""
    print("OPTIMIZED CRICKET SCORE PREDICTION TRAINING")
    print("=" * 60)
    print("Features: Clean pre-match features only")
    print("Models: Linear Regression, Random Forest, XGBoost")
    print("Optimization: GPU acceleration, efficient training")
    print("=" * 60)
    
    # Prepare clean data
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_clean_data()
    
    # Train all models
    results = {}
    
    # 1. Linear Regression
    results['Linear Regression'] = train_linear_regression(X_train, X_test, y_train, y_test)
    
    # 2. Random Forest
    results['Random Forest'] = train_random_forest(X_train, X_test, y_train, y_test)
    
    # 3. XGBoost
    results['XGBoost'] = train_xgboost(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df, best_model = compare_models(results)
    
    # Save scaler and feature info
    with open('models/scaler_clean.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    feature_info = {
        'feature_columns': feature_cols,
        'scaler': scaler,
        'results': results,
        'comparison': comparison_df,
        'best_model': best_model
    }
    
    with open('models/training_info_clean.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"\nTraining completed!")
    print(f"Models saved in: models/")
    print(f"Best model: {best_model['Model']}")
    
    return results, comparison_df, best_model

if __name__ == "__main__":
    results, comparison, best = main()
