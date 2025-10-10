"""
Test the model with ACTUAL features from test dataset
This bypasses the API and directly tests the model
"""
import pandas as pd
import pickle
import numpy as np
import os

print("="*100)
print("TESTING MODEL WITH REAL FEATURES FROM TEST DATASET")
print("="*100)

# Load test dataset (500 matches held out from training)
test_df = pd.read_csv(r'C:\Users\OMNIBOOK\Documents\GitHub\CircketScore-Prediction-Using-Machine-Learning-\ODI\data\odi_test_500.csv')
print(f"\nLoaded {len(test_df)} test matches")

# Load model files
odi_dir = r'C:\Users\OMNIBOOK\Documents\GitHub\CircketScore-Prediction-Using-Machine-Learning-\ODI'

try:
    with open(os.path.join(odi_dir, 'models', 'xgboost_COMPLETE.pkl'), 'rb') as f:
        model = pickle.load(f)
    print(f"[+] Loaded XGBoost model")
except Exception as e:
    print(f"[-] Could not load model: {e}")
    exit(1)

try:
    with open(os.path.join(odi_dir, 'models', 'feature_names_COMPLETE.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    print(f"[+] Model expects {len(feature_names)} features")
except Exception as e:
    print(f"[-] Could not load feature names: {e}")
    exit(1)

# Check which features are in the test dataset
missing_features = [f for f in feature_names if f not in test_df.columns]
if missing_features:
    print(f"\n[!] WARNING: {len(missing_features)} features missing from test data:")
    print(f"    First 10: {missing_features[:10]}")
    
    # Fill missing features with zeros
    for feat in missing_features:
        test_df[feat] = 0
    print(f"[+] Filled missing features with 0")

# Prepare features
X_test = test_df[feature_names]
y_test = test_df['total_runs']

print(f"\n[+] Test data prepared:")
print(f"    Features: {X_test.shape[1]}")
print(f"    Samples: {X_test.shape[0]}")
print(f"    Target (actual scores): min={y_test.min()}, max={y_test.max()}, mean={y_test.mean():.1f}")

# Make predictions (RAW - no scaling needed if model has StandardScaler built in)
try:
    # Try loading scaler
    with open(os.path.join(odi_dir, 'models', 'scaler_COMPLETE.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    print(f"[+] Used StandardScaler before prediction")
except Exception as e:
    print(f"[!] No scaler or error loading: {e}")
    print(f"[+] Predicting without scaling")
    predictions = model.predict(X_test)

print(f"\n{'='*100}")
print(f"RAW MODEL PERFORMANCE (no bias correction):")
print(f"{'='*100}")

# Calculate metrics
mae = np.mean(np.abs(predictions - y_test))
rmse = np.sqrt(np.mean((predictions - y_test)**2))
bias = np.mean(predictions - y_test)

# R-squared
ss_res = np.sum((y_test - predictions)**2)
ss_tot = np.sum((y_test - y_test.mean())**2)
r2 = 1 - (ss_res / ss_tot)

print(f"\nR² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"Mean Absolute Error (MAE): {mae:.2f} runs")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} runs")
print(f"Systematic Bias: {bias:+.2f} runs")
print(f"\nActual scores:    mean={y_test.mean():.1f}, std={y_test.std():.1f}")
print(f"Predicted scores: mean={predictions.mean():.1f}, std={predictions.std():.1f}")

# Accuracy bands
within_20 = np.sum(np.abs(predictions - y_test) <= 20)
within_30 = np.sum(np.abs(predictions - y_test) <= 30)
within_40 = np.sum(np.abs(predictions - y_test) <= 40)

print(f"\n{'='*100}")
print(f"ACCURACY BANDS:")
print(f"{'='*100}")
print(f"Within +/-20 runs: {within_20}/{len(test_df)} ({100*within_20/len(test_df):.1f}%)")
print(f"Within +/-30 runs: {within_30}/{len(test_df)} ({100*within_30/len(test_df):.1f}%)")
print(f"Within +/-40 runs: {within_40}/{len(test_df)} ({100*within_40/len(test_df):.1f}%)")

# Show 20 examples
print(f"\n{'='*100}")
print(f"20 RANDOM EXAMPLES:")
print(f"{'='*100}")
print(f"{'Team':<20} {'Opposition':<20} {'Actual':<8} {'Predicted':<10} {'Error'}")
print(f"{'-'*80}")

sample_indices = np.random.choice(len(test_df), min(20, len(test_df)), replace=False)
for i in sample_indices:
    team = test_df.iloc[i]['team'][:18]
    opp = test_df.iloc[i]['opposition'][:18]
    actual = y_test.iloc[i]
    pred = predictions[i]
    error = pred - actual
    status = "[OK]" if abs(error) <= 30 else "[!!]"
    
    print(f"{status} {team:<20} {opp:<20} {actual:<8.0f} {pred:<10.0f} {error:+.0f}")

# Assessment
print(f"\n{'='*100}")
print(f"ASSESSMENT:")
print(f"{'='*100}")

if r2 >= 0.65:
    print(f"[+] R² = {r2:.3f} - GOOD (model explains {r2*100:.1f}% of variance)")
elif r2 >= 0.50:
    print(f"[~] R² = {r2:.3f} - ACCEPTABLE (model explains {r2*100:.1f}% of variance)")
else:
    print(f"[-] R² = {r2:.3f} - POOR (model only explains {r2*100:.1f}% of variance)")

if mae <= 25:
    print(f"[+] MAE = {mae:.1f} - EXCELLENT")
elif mae <= 35:
    print(f"[~] MAE = {mae:.1f} - GOOD")
else:
    print(f"[-] MAE = {mae:.1f} - NEEDS IMPROVEMENT")

if within_30 / len(test_df) >= 0.60:
    print(f"[+] {100*within_30/len(test_df):.0f}% within +/-30 runs - GOOD")
elif within_30 / len(test_df) >= 0.45:
    print(f"[~] {100*within_30/len(test_df):.0f}% within +/-30 runs - ACCEPTABLE")
else:
    print(f"[-] {100*within_30/len(test_df):.0f}% within +/-30 runs - POOR")

print(f"\n{'='*100}")
print(f"VERDICT:")
print(f"{'='*100}")
if r2 >= 0.65 and mae <= 30:
    print(f"[+] Model is WORKING WELL with proper features!")
    print(f"    The poor API results were due to missing player/context data.")
elif r2 >= 0.50:
    print(f"[~] Model is ACCEPTABLE but could be improved")
else:
    print(f"[-] Model has SERIOUS ISSUES - needs retraining")
    print(f"    Problem: Underfitting - model not learning patterns")
    print(f"    Possible causes:")
    print(f"      1. Insufficient relevant features")
    print(f"      2. Poor feature quality")
    print(f"      3. Over-regularization in hyperparameters")

