"""
Test T20 models with real held-out test data
Same approach as ODI testing to verify actual performance
"""
import pandas as pd
import pickle
import joblib
import numpy as np
import os

print("="*100)
print("TESTING T20 MODELS WITH REAL TEST DATA")
print("="*100)

# Paths
t20_dir = r'C:\Users\OMNIBOOK\Documents\GitHub\CircketScore-Prediction-Using-Machine-Learning-\T20'

# Try different test datasets
test_files = [
    'simple_enhanced_test.csv',
    'test_dataset.csv'
]

test_df = None
for test_file in test_files:
    test_path = os.path.join(t20_dir, 'data', test_file)
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"\n[+] Loaded {test_file}: {len(test_df)} matches")
        break

if test_df is None:
    print("[-] No test data found!")
    exit(1)

print(f"    Columns: {len(test_df.columns)}")
print(f"    Target column: {'predicted_score' if 'predicted_score' in test_df.columns else 'score' if 'score' in test_df.columns else 'total_runs'}")

# Determine target column
target_col = None
for col in ['predicted_score', 'score', 'total_runs', 'runs']:
    if col in test_df.columns:
        target_col = col
        break

if target_col is None:
    print(f"[-] No score column found! Available: {test_df.columns.tolist()[:20]}")
    exit(1)

y_test = test_df[target_col]
print(f"    Actual scores: min={y_test.min()}, max={y_test.max()}, mean={y_test.mean():.1f}")

# Load models
models_to_test = {
    'XGBoost': 'final_trained_xgboost.pkl',
    'Random Forest': 'final_trained_random_forest.pkl',
    'Linear Regression': 'final_trained_linear_regression.pkl'
}

# Load feature names and scaler
try:
    feature_names = joblib.load(os.path.join(t20_dir, 'models', 'final_trained_feature_names.pkl'))
    print(f"[+] Loaded feature names: {len(feature_names)} features")
except Exception as e:
    print(f"[-] Could not load feature names: {e}")
    exit(1)

try:
    scaler = joblib.load(os.path.join(t20_dir, 'models', 'final_trained_scaler.pkl'))
    print(f"[+] Loaded scaler")
except Exception as e:
    print(f"[!] Could not load scaler: {e}")
    scaler = None

# Check features
missing_features = [f for f in feature_names if f not in test_df.columns]
if missing_features:
    print(f"\n[!] WARNING: {len(missing_features)} features missing:")
    print(f"    {missing_features[:10]}")
    for feat in missing_features:
        test_df[feat] = 0
    print(f"[+] Filled with zeros")

# Test each model
print(f"\n{'='*100}")
print(f"TESTING MODELS")
print(f"{'='*100}\n")

results = {}

for model_name, model_file in models_to_test.items():
    try:
        # Load model
        model_path = os.path.join(t20_dir, 'models', model_file)
        model = joblib.load(model_path)
        
        # Prepare features
        X_test = test_df[feature_names].copy()
        
        # Handle categorical columns for XGBoost
        categorical_cols = X_test.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"  [!] Converting {len(categorical_cols)} categorical columns to numeric")
            for col in categorical_cols:
                X_test[col] = pd.Categorical(X_test[col]).codes
        
        # Scale if scaler available
        if scaler is not None:
            try:
                X_test_scaled = scaler.transform(X_test)
                predictions = model.predict(X_test_scaled)
            except:
                predictions = model.predict(X_test)
        else:
            predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        bias = np.mean(predictions - y_test)
        
        # R-squared
        ss_res = np.sum((y_test - predictions)**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Accuracy bands
        within_10 = np.sum(np.abs(predictions - y_test) <= 10) / len(test_df) * 100
        within_15 = np.sum(np.abs(predictions - y_test) <= 15) / len(test_df) * 100
        within_20 = np.sum(np.abs(predictions - y_test) <= 20) / len(test_df) * 100
        
        results[model_name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'within_10': within_10,
            'within_15': within_15,
            'within_20': within_20,
            'pred_mean': predictions.mean(),
            'pred_std': predictions.std()
        }
        
        print(f"{model_name}:")
        print(f"  R² Score:           {r2:.4f} ({r2*100:.2f}%)")
        print(f"  MAE:                {mae:.2f} runs")
        print(f"  RMSE:               {rmse:.2f} runs")
        print(f"  Systematic Bias:    {bias:+.2f} runs")
        print(f"  Within ±10 runs:    {within_10:.1f}%")
        print(f"  Within ±15 runs:    {within_15:.1f}%")
        print(f"  Within ±20 runs:    {within_20:.1f}%")
        print(f"  Predicted: mean={predictions.mean():.1f}, std={predictions.std():.1f}")
        print(f"  Actual:    mean={y_test.mean():.1f}, std={y_test.std():.1f}")
        print()
        
    except Exception as e:
        print(f"{model_name}: [FAILED] {e}\n")

# Show comparison
if results:
    print(f"{'='*100}")
    print(f"MODEL COMPARISON")
    print(f"{'='*100}\n")
    
    best_r2 = max(results.values(), key=lambda x: x['r2'])
    best_mae = min(results.values(), key=lambda x: x['mae'])
    
    for model_name, metrics in results.items():
        status = "[BEST R2]" if metrics == best_r2 else "[BEST MAE]" if metrics == best_mae else ""
        print(f"{model_name:20} R²={metrics['r2']:.3f} | MAE={metrics['mae']:.1f} | ±20={metrics['within_20']:.0f}% {status}")
else:
    print(f"\n[-] No models loaded successfully! Pickle compatibility issues.")
    exit(1)

# Show sample predictions
best_model_name = max(results, key=lambda k: results[k]['r2'])
print(f"\n{'='*100}")
print(f"20 SAMPLE PREDICTIONS (Best Model: {best_model_name})")
print(f"{'='*100}")

model_path = os.path.join(t20_dir, 'models', models_to_test[best_model_name])
best_model = joblib.load(model_path)

X_test = test_df[feature_names].copy()
# Convert categorical columns
categorical_cols = X_test.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X_test[col] = pd.Categorical(X_test[col]).codes

if scaler:
    try:
        X_test_scaled = scaler.transform(X_test)
        predictions = best_model.predict(X_test_scaled)
    except:
        predictions = best_model.predict(X_test)
else:
    predictions = best_model.predict(X_test)

print(f"{'Team':<20} {'Venue':<25} {'Actual':<8} {'Predicted':<10} {'Error'}")
print(f"{'-'*80}")

sample_indices = np.random.choice(len(test_df), min(20, len(test_df)), replace=False)
for i in sample_indices:
    team = test_df.iloc[i]['team'][:18] if 'team' in test_df.columns else 'Unknown'
    venue = test_df.iloc[i]['venue'][:23] if 'venue' in test_df.columns else 'Unknown'
    actual = y_test.iloc[i]
    pred = predictions[i]
    error = pred - actual
    status = "[OK]" if abs(error) <= 15 else "[!!]"
    
    print(f"{status} {team:<20} {venue:<25} {actual:<8.0f} {pred:<10.0f} {error:+.0f}")

# Final assessment
print(f"\n{'='*100}")
print(f"ASSESSMENT")
print(f"{'='*100}\n")

best_metrics = results[best_model_name]

if best_metrics['r2'] >= 0.65:
    print(f"[+] R² = {best_metrics['r2']:.3f} - GOOD")
elif best_metrics['r2'] >= 0.50:
    print(f"[~] R² = {best_metrics['r2']:.3f} - ACCEPTABLE")
else:
    print(f"[-] R² = {best_metrics['r2']:.3f} - POOR (same issue as ODI!)")

if best_metrics['mae'] <= 12:
    print(f"[+] MAE = {best_metrics['mae']:.1f} - EXCELLENT (T20 is unpredictable)")
elif best_metrics['mae'] <= 18:
    print(f"[~] MAE = {best_metrics['mae']:.1f} - GOOD")
else:
    print(f"[-] MAE = {best_metrics['mae']:.1f} - POOR")

if best_metrics['within_20'] >= 70:
    print(f"[+] {best_metrics['within_20']:.0f}% within ±20 runs - EXCELLENT")
elif best_metrics['within_20'] >= 55:
    print(f"[~] {best_metrics['within_20']:.0f}% within ±20 runs - GOOD")
else:
    print(f"[-] {best_metrics['within_20']:.0f}% within ±20 runs - POOR")

print(f"\n{'='*100}")
print(f"VERDICT:")
print(f"{'='*100}")

if best_metrics['r2'] >= 0.60 and best_metrics['mae'] <= 18:
    print(f"[+] T20 MODEL IS WORKING!")
    print(f"    Much better than ODI (R²={best_metrics['r2']:.3f} vs 0.01)")
    print(f"    Predictions show good variation (std={best_metrics['pred_std']:.1f})")
elif best_metrics['r2'] >= 0.40:
    print(f"[~] T20 MODEL IS ACCEPTABLE")
    print(f"    Better than ODI but could be improved")
else:
    print(f"[-] T20 MODEL HAS SAME ISSUES AS ODI!")
    print(f"    R² = {best_metrics['r2']:.3f} (too low)")
    print(f"    Model not learning patterns properly")

