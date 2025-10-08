#!/usr/bin/env python3
"""
Comprehensive Model Training for Cricket Score Prediction
Trains Linear Regression, Random Forest, and XGBoost models with full evaluation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

def comprehensive_model_training():
    """Train and compare multiple models for cricket score prediction"""
    print("🏏 COMPREHENSIVE MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load and confirm dataset
    print("📊 STEP 1: DATASET CONFIRMATION")
    print("-" * 40)
    
    try:
        df = pd.read_csv('processed_data/cleaned_cricket_dataset.csv')
        print(f"✅ Loaded cleaned dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Confirm target variable
    if 'total_runs' not in df.columns:
        print("❌ Target variable 'total_runs' not found!")
        return
    
    print(f"✅ Target variable confirmed: 'total_runs'")
    print(f"📈 Target statistics:")
    print(f"   Mean: {df['total_runs'].mean():.2f}")
    print(f"   Std: {df['total_runs'].std():.2f}")
    print(f"   Range: {df['total_runs'].min():.0f} - {df['total_runs'].max():.0f}")
    
    # Prepare features and target
    X = df.drop('total_runs', axis=1)
    y = df['total_runs']
    
    # Convert boolean columns to int and handle date column
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        elif X[col].dtype == 'object' and col == 'date':
            # Convert date to numeric (days since epoch)
            X[col] = pd.to_datetime(X[col], errors='coerce').astype(np.int64) / (10**9 * 86400)
        elif X[col].dtype == 'object':
            # Convert any remaining object columns to numeric
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"✅ Features: {X.shape[1]} features")
    print(f"✅ Samples: {X.shape[0]:,} samples")
    
    # 2. Chronological train-test split
    print(f"\n📅 STEP 2: CHRONOLOGICAL TRAIN-TEST SPLIT")
    print("-" * 40)
    
    # Sort by date to ensure chronological order
    if 'date' in df.columns:
        # Convert date to datetime for proper sorting
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        df_sorted = df.sort_values('date_parsed')
        
        # Split chronologically (80% oldest for training, 20% latest for testing)
        split_idx = int(0.8 * len(df_sorted))
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        X_train = train_df.drop('total_runs', axis=1)
        y_train = train_df['total_runs']
        X_test = test_df.drop('total_runs', axis=1)
        y_test = test_df['total_runs']
        
        # Apply the same data type conversions to train/test sets
        for col in X_train.columns:
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                X_test[col] = X_test[col].astype(int)
            elif X_train[col].dtype == 'object' and col == 'date':
                X_train[col] = pd.to_datetime(X_train[col], errors='coerce').astype(np.int64) / (10**9 * 86400)
                X_test[col] = pd.to_datetime(X_test[col], errors='coerce').astype(np.int64) / (10**9 * 86400)
            elif X_train[col].dtype == 'object':
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        print(f"✅ Chronological split completed")
        print(f"   Training set: {X_train.shape[0]:,} samples (oldest 80%)")
        print(f"   Test set: {X_test.shape[0]:,} samples (latest 20%)")
        print(f"   Date range - Train: {train_df['date_parsed'].min().date()} to {train_df['date_parsed'].max().date()}")
        print(f"   Date range - Test: {test_df['date_parsed'].min().date()} to {test_df['date_parsed'].max().date()}")
    else:
        # Fallback to random split if no date column
        print("⚠️ No date column found, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Training set: {X_train.shape[0]:,} samples")
        print(f"   Test set: {X_test.shape[0]:,} samples")
    
    # 3. Model training and evaluation
    print(f"\n🤖 STEP 3: MODEL TRAINING")
    print("-" * 40)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Check for GPU availability for XGBoost
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected, enabling GPU acceleration for XGBoost")
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                gpu_id=0
            )
        else:
            print("⚠️ No GPU detected, using CPU for XGBoost")
    except:
        print("⚠️ Could not check GPU, using CPU for XGBoost")
    
    # Results storage
    results = []
    trained_models = {}
    feature_importance = {}
    
    # 4. Train each model
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Relative error (% difference)
        train_rel_error = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        test_rel_error = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Store results
        model_results = {
            'Model': name,
            'Training_Time': training_time,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Train_Rel_Error': train_rel_error,
            'Test_Rel_Error': test_rel_error
        }
        results.append(model_results)
        
        # Store trained model
        trained_models[name] = model
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Training completed in {training_time:.2f} seconds")
        print(f"   📊 Test R²: {test_r2:.4f}")
        print(f"   📊 Test RMSE: {test_rmse:.2f}")
        print(f"   📊 Test MAE: {test_mae:.2f}")
        print(f"   📊 Test Rel Error: {test_rel_error:.2f}%")
    
    # 5. Cross-validation
    print(f"\n🔄 STEP 4: CROSS-VALIDATION")
    print("-" * 40)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, model in models.items():
        print(f"🔄 Running 5-fold CV for {name}...")
        
        # Cross-validation scores
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(cv_rmse_scores)
        
        cv_results[name] = {
            'CV_R2_Mean': cv_r2_scores.mean(),
            'CV_R2_Std': cv_r2_scores.std(),
            'CV_RMSE_Mean': cv_rmse_scores.mean(),
            'CV_RMSE_Std': cv_rmse_scores.std()
        }
        
        print(f"   📊 CV R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
        print(f"   📊 CV RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")
    
    # 6. Results summary
    print(f"\n📊 STEP 5: RESULTS SUMMARY")
    print("-" * 40)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add cross-validation results
    for i, result in enumerate(results_df['Model']):
        if result in cv_results:
            results_df.loc[i, 'CV_R2_Mean'] = cv_results[result]['CV_R2_Mean']
            results_df.loc[i, 'CV_R2_Std'] = cv_results[result]['CV_R2_Std']
            results_df.loc[i, 'CV_RMSE_Mean'] = cv_results[result]['CV_RMSE_Mean']
            results_df.loc[i, 'CV_RMSE_Std'] = cv_results[result]['CV_RMSE_Std']
    
    # Display results table
    print("🏆 MODEL COMPARISON RESULTS:")
    print("=" * 80)
    
    # Format and display results
    display_cols = ['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Test_Rel_Error', 'CV_R2_Mean', 'Training_Time']
    display_df = results_df[display_cols].copy()
    
    # Round numeric columns
    numeric_cols = ['Test_R2', 'Test_RMSE', 'Test_MAE', 'Test_Rel_Error', 'CV_R2_Mean', 'Training_Time']
    for col in numeric_cols:
        if col in display_df.columns:
            if col == 'Training_Time':
                display_df[col] = display_df[col].round(2)
            else:
                display_df[col] = display_df[col].round(4)
    
    print(display_df.to_string(index=False))
    
    # 7. Feature importance analysis
    print(f"\n🎯 STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    for name, importance_df in feature_importance.items():
        print(f"\n🌲 {name.upper()} - TOP 15 FEATURES:")
        print("-" * 50)
        
        top_features = importance_df.head(15)
        for i, (_, row) in enumerate(top_features.iterrows()):
            print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # 8. Save models and results
    print(f"\n💾 STEP 7: SAVING MODELS AND RESULTS")
    print("-" * 40)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save trained models
    for name, model in trained_models.items():
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"✅ Saved {name} model: {model_filename}")
    
    # Save results
    results_filename = 'results/model_comparison.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"✅ Saved results: {results_filename}")
    
    # Save feature importance
    for name, importance_df in feature_importance.items():
        importance_filename = f"results/{name.lower().replace(' ', '_')}_feature_importance.csv"
        importance_df.to_csv(importance_filename, index=False)
        print(f"✅ Saved {name} feature importance: {importance_filename}")
    
    # 9. Final recommendations
    print(f"\n🏆 STEP 8: FINAL RECOMMENDATIONS")
    print("-" * 40)
    
    # Find best model
    best_model = results_df.loc[results_df['Test_R2'].idxmax()]
    print(f"🥇 BEST MODEL: {best_model['Model']}")
    print(f"   Test R²: {best_model['Test_R2']:.4f}")
    print(f"   Test RMSE: {best_model['Test_RMSE']:.2f}")
    print(f"   Test MAE: {best_model['Test_MAE']:.2f}")
    print(f"   Relative Error: {best_model['Test_Rel_Error']:.2f}%")
    
    # Performance interpretation
    print(f"\n📈 PERFORMANCE INTERPRETATION:")
    print(f"   R² Score: {best_model['Test_R2']:.1%} of variance explained")
    print(f"   RMSE: Average prediction error of {best_model['Test_RMSE']:.1f} runs")
    print(f"   MAE: Median prediction error of {best_model['Test_MAE']:.1f} runs")
    print(f"   Relative Error: {best_model['Test_Rel_Error']:.1f}% average deviation from actual")
    
    # Model comparison
    print(f"\n🔄 MODEL COMPARISON:")
    for _, row in results_df.iterrows():
        status = "🥇" if row['Test_R2'] == best_model['Test_R2'] else "🥈" if row['Test_R2'] >= best_model['Test_R2'] * 0.95 else "🥉"
        print(f"   {status} {row['Model']}: R²={row['Test_R2']:.4f}, RMSE={row['Test_RMSE']:.1f}")
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Models saved in: models/")
    print(f"   Results saved in: results/")
    print(f"   Best model: {best_model['Model']}")
    
    return {
        'best_model': best_model['Model'],
        'best_r2': best_model['Test_R2'],
        'best_rmse': best_model['Test_RMSE'],
        'results_df': results_df,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    results = comprehensive_model_training()
    
    print(f"\n" + "="*60)
    print("COMPREHENSIVE MODEL TRAINING COMPLETE")
    print("="*60)
    
    print(f"🏆 Best Model: {results['best_model']}")
    print(f"📊 Best R²: {results['best_r2']:.4f}")
    print(f"📊 Best RMSE: {results['best_rmse']:.2f}")
    
    print(f"\n🚀 READY FOR PRODUCTION!")
