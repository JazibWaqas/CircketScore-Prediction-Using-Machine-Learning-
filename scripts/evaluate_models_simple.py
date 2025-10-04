"""
Simple Model Evaluation with Dynamic Accuracy Metrics
Evaluate models with realistic cricket prediction metrics
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models_simple():
    """Evaluate models with proper cricket prediction metrics"""
    print("PROPER MODEL EVALUATION - CRICKET PREDICTION METRICS")
    print("=" * 70)
    
    # Load test data
    test_df = pd.read_csv('data/simple_enhanced_test.csv')
    print(f"Test dataset: {test_df.shape}")
    
    # Load models and scalers
    try:
        model = pickle.load(open('models/random_forest_mixed_features.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler_mixed_features.pkl', 'rb'))
        encoders = pickle.load(open('models/label_encoders_mixed_features.pkl', 'rb'))
        print("Models loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return
    
    # Prepare features (same as training)
    feature_cols = [col for col in test_df.columns if col != 'total_runs']
    X_test = test_df[feature_cols]
    y_test = test_df['total_runs']
    
    # Identify feature types
    numerical_features = []
    categorical_features = []
    
    for col in X_test.columns:
        if X_test[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    # Handle categorical features with Label Encoding (same as training)
    X_test_num = X_test[numerical_features]
    X_test_cat = X_test[categorical_features]
    
    for col in categorical_features:
        if col in encoders:
            le = encoders[col]
            test_values = X_test_cat[col].astype(str)
            unseen_mask = ~test_values.isin(le.classes_)
            
            if unseen_mask.any():
                test_values = test_values.copy()
                test_values[unseen_mask] = le.classes_[0]
            
            X_test_cat[col] = le.transform(test_values)
    
    # Combine numerical and categorical features
    X_test_combined = pd.concat([X_test_num, X_test_cat], axis=1)
    
    # Handle missing values and standardize
    X_test_combined = X_test_combined.fillna(0)
    X_test_scaled = scaler.transform(X_test_combined)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    print(f"\nPREDICTION RESULTS:")
    print(f"Actual scores range: {y_test.min():.0f} to {y_test.max():.0f} runs")
    print(f"Predicted scores range: {y_pred.min():.0f} to {y_pred.max():.0f} runs")
    print(f"Average actual score: {y_test.mean():.1f} runs")
    print(f"Average predicted score: {y_pred.mean():.1f} runs")
    
    # Calculate basic metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nBASIC METRICS:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} runs")
    print(f"MAE: {mae:.2f} runs")
    
    # Calculate dynamic accuracy metrics
    print(f"\nDYNAMIC ACCURACY METRICS:")
    
    # Option 1: ±10% Tolerance (Dynamic Accuracy)
    tolerance_10_percent = np.abs(y_pred - y_test) <= (0.10 * y_test)
    accuracy_10_percent = np.mean(tolerance_10_percent) * 100
    
    # Option 2: ±15% Tolerance
    tolerance_15_percent = np.abs(y_pred - y_test) <= (0.15 * y_test)
    accuracy_15_percent = np.mean(tolerance_15_percent) * 100
    
    # Option 3: ±20% Tolerance
    tolerance_20_percent = np.abs(y_pred - y_test) <= (0.20 * y_test)
    accuracy_20_percent = np.mean(tolerance_20_percent) * 100
    
    print(f"±10% Tolerance: {accuracy_10_percent:.1f}% accurate")
    print(f"±15% Tolerance: {accuracy_15_percent:.1f}% accurate")
    print(f"±20% Tolerance: {accuracy_20_percent:.1f}% accurate")
    
    # Fixed tolerance for comparison
    tolerance_10_runs = np.abs(y_pred - y_test) <= 10
    accuracy_10_runs = np.mean(tolerance_10_runs) * 100
    print(f"±10 runs (fixed): {accuracy_10_runs:.1f}% accurate")
    
    # Calculate error distribution
    errors = y_pred - y_test
    abs_errors = np.abs(errors)
    
    print(f"\nERROR DISTRIBUTION:")
    print(f"Mean error: {np.mean(errors):.2f} runs")
    print(f"Median absolute error: {np.median(abs_errors):.2f} runs")
    print(f"75th percentile error: {np.percentile(abs_errors, 75):.2f} runs")
    print(f"90th percentile error: {np.percentile(abs_errors, 90):.2f} runs")
    print(f"95th percentile error: {np.percentile(abs_errors, 95):.2f} runs")
    
    # Score range analysis
    print(f"\nSCORE RANGE ANALYSIS:")
    
    # Low scores (0-100)
    low_scores = y_test <= 100
    if low_scores.any():
        low_mae = mean_absolute_error(y_test[low_scores], y_pred[low_scores])
        low_accuracy = np.mean(np.abs(y_pred[low_scores] - y_test[low_scores]) <= (0.15 * y_test[low_scores])) * 100
        print(f"Low scores (≤100): MAE = {low_mae:.1f} runs, ±15% accuracy = {low_accuracy:.1f}%")
    
    # Medium scores (100-150)
    medium_scores = (y_test > 100) & (y_test <= 150)
    if medium_scores.any():
        medium_mae = mean_absolute_error(y_test[medium_scores], y_pred[medium_scores])
        medium_accuracy = np.mean(np.abs(y_pred[medium_scores] - y_test[medium_scores]) <= (0.15 * y_test[medium_scores])) * 100
        print(f"Medium scores (100-150): MAE = {medium_mae:.1f} runs, ±15% accuracy = {medium_accuracy:.1f}%")
    
    # High scores (150+)
    high_scores = y_test > 150
    if high_scores.any():
        high_mae = mean_absolute_error(y_test[high_scores], y_pred[high_scores])
        high_accuracy = np.mean(np.abs(y_pred[high_scores] - y_test[high_scores]) <= (0.15 * y_test[high_scores])) * 100
        print(f"High scores (150+): MAE = {high_mae:.1f} runs, ±15% accuracy = {high_accuracy:.1f}%")
    
    # Cricket-specific evaluation
    print(f"\nCRICKET-SPECIFIC EVALUATION:")
    
    # Close matches (within 20 runs)
    close_matches = abs_errors <= 20
    close_accuracy = np.mean(close_matches) * 100
    print(f"Close predictions (≤20 runs off): {close_accuracy:.1f}%")
    
    # Good predictions (within 30 runs)
    good_predictions = abs_errors <= 30
    good_accuracy = np.mean(good_predictions) * 100
    print(f"Good predictions (≤30 runs off): {good_accuracy:.1f}%")
    
    # Reasonable predictions (within 40 runs)
    reasonable_predictions = abs_errors <= 40
    reasonable_accuracy = np.mean(reasonable_predictions) * 100
    print(f"Reasonable predictions (≤40 runs off): {reasonable_accuracy:.1f}%")
    
    # Model performance assessment
    print(f"\nMODEL PERFORMANCE ASSESSMENT:")
    
    if r2 >= 0.7:
        print(f"✅ R² = {r2:.3f} - EXCELLENT model fit")
    elif r2 >= 0.5:
        print(f"✅ R² = {r2:.3f} - GOOD model fit")
    elif r2 >= 0.3:
        print(f"⚠️ R² = {r2:.3f} - FAIR model fit")
    else:
        print(f"❌ R² = {r2:.3f} - POOR model fit")
    
    if mae <= 20:
        print(f"✅ MAE = {mae:.1f} runs - EXCELLENT accuracy")
    elif mae <= 30:
        print(f"✅ MAE = {mae:.1f} runs - GOOD accuracy")
    elif mae <= 40:
        print(f"⚠️ MAE = {mae:.1f} runs - FAIR accuracy")
    else:
        print(f"❌ MAE = {mae:.1f} runs - POOR accuracy")
    
    if accuracy_15_percent >= 60:
        print(f"✅ ±15% accuracy = {accuracy_15_percent:.1f}% - EXCELLENT")
    elif accuracy_15_percent >= 50:
        print(f"✅ ±15% accuracy = {accuracy_15_percent:.1f}% - GOOD")
    elif accuracy_15_percent >= 40:
        print(f"⚠️ ±15% accuracy = {accuracy_15_percent:.1f}% - FAIR")
    else:
        print(f"❌ ±15% accuracy = {accuracy_15_percent:.1f}% - POOR")
    
    # Save results
    results = {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'accuracy_10_percent': accuracy_10_percent,
        'accuracy_15_percent': accuracy_15_percent,
        'accuracy_20_percent': accuracy_20_percent,
        'accuracy_10_runs': accuracy_10_runs,
        'close_accuracy': close_accuracy,
        'good_accuracy': good_accuracy,
        'reasonable_accuracy': reasonable_accuracy
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('models/proper_evaluation_results.csv', index=False)
    print(f"\nResults saved: models/proper_evaluation_results.csv")
    
    return results

if __name__ == "__main__":
    results = evaluate_models_simple()
