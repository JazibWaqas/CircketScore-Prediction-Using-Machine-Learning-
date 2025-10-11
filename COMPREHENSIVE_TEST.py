import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, 'dashboard/backend')
from utils.model_loader import ModelLoader
from utils.predictions import make_prediction

print("="*80)
print("COMPREHENSIVE MODEL TESTING ON ALL TEST MATCHES")
print("="*80)
print()

# Load model and test data
print("[1/4] Loading model and test data...")
ml = ModelLoader()
df = pd.read_csv('ODI_Progressive/data/progressive_full_test.csv')
print(f"Total test samples: {len(df)}")
print(f"Unique matches: {df['match_id'].nunique() if 'match_id' in df.columns else 'N/A'}")
print()

# Prepare features
print("[2/4] Preparing features...")
feature_cols = [
    'current_score', 'wickets_fallen', 'balls_bowled', 'balls_remaining',
    'runs_last_10_overs', 'current_run_rate', 'team_batting_avg',
    'team_elite_batsmen', 'team_batting_depth', 'opp_bowling_economy',
    'opp_elite_bowlers', 'opp_bowling_depth', 'venue_avg_score',
    'batsman_1_avg', 'batsman_2_avg'
]

# Add venue if needed
if 'venue' in df.columns:
    all_features = feature_cols + ['venue']
else:
    all_features = feature_cols

X_test = df[all_features].copy()
y_test = df['final_score'].values

print(f"Features used: {len(all_features)}")
print()

# Make predictions
print("[3/4] Making predictions on all test samples...")
predictions = ml.model.predict(X_test)
print(f"Predictions completed: {len(predictions)}")
print()

# Calculate metrics
print("[4/4] Calculating comprehensive metrics...")
errors = np.abs(predictions - y_test)
mae = np.mean(errors)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
r2 = 1 - (np.sum((y_test - predictions)**2) / np.sum((y_test - np.mean(y_test))**2))

accuracy_10 = (errors <= 10).sum() / len(errors) * 100
accuracy_20 = (errors <= 20).sum() / len(errors) * 100
accuracy_30 = (errors <= 30).sum() / len(errors) * 100
accuracy_50 = (errors <= 50).sum() / len(errors) * 100

print()
print("="*80)
print("OVERALL PERFORMANCE METRICS")
print("="*80)
print(f"R² Score:              {r2:.4f}")
print(f"Mean Absolute Error:   {mae:.2f} runs")
print(f"Root Mean Square Error: {rmse:.2f} runs")
print(f"Accuracy within 10:    {accuracy_10:.2f}%")
print(f"Accuracy within 20:    {accuracy_20:.2f}%")
print(f"Accuracy within 30:    {accuracy_30:.2f}%")
print(f"Accuracy within 50:    {accuracy_50:.2f}%")
print()

# Analyze by match stage
print("="*80)
print("ACCURACY BY MATCH STAGE")
print("="*80)
print()

stages = [
    ('Pre-Match', 0, 60),
    ('Early', 60, 120),
    ('Mid', 120, 180),
    ('Late', 180, 240),
    ('Death', 240, 300)
]

stage_results = []
for stage_name, start_ball, end_ball in stages:
    mask = (df['balls_bowled'] >= start_ball) & (df['balls_bowled'] < end_ball)
    if mask.sum() > 0:
        stage_preds = predictions[mask]
        stage_actual = y_test[mask]
        stage_errors = np.abs(stage_preds - stage_actual)
        stage_mae = np.mean(stage_errors)
        stage_r2 = 1 - (np.sum((stage_actual - stage_preds)**2) / np.sum((stage_actual - np.mean(stage_actual))**2))
        stage_acc = (stage_errors <= 20).sum() / len(stage_errors) * 100
        
        print(f"{stage_name:12} ({start_ball:3}-{end_ball:3} balls) | Samples: {mask.sum():4} | R²: {stage_r2:5.3f} | MAE: {stage_mae:5.1f} | Acc: {stage_acc:5.1f}%")
        stage_results.append({
            'Stage': stage_name,
            'Balls': f'{start_ball}-{end_ball}',
            'Samples': mask.sum(),
            'R2': stage_r2,
            'MAE': stage_mae,
            'Accuracy': stage_acc
        })

print()

# Sample predictions table
print("="*80)
print("SAMPLE PREDICTIONS (First 50 matches)")
print("="*80)
print()

# Group by match_id if available, otherwise just take first 50
if 'match_id' in df.columns:
    sample_df = df.head(50).copy()
else:
    sample_df = df.head(50).copy()

sample_df['predicted'] = predictions[:50]
sample_df['error'] = np.abs(predictions[:50] - y_test[:50])

# Create detailed output file
output_lines = []
output_lines.append("="*100)
output_lines.append("COMPREHENSIVE MODEL TESTING - FULL RESULTS")
output_lines.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output_lines.append("="*100)
output_lines.append("")

output_lines.append("OVERALL PERFORMANCE:")
output_lines.append("-"*100)
output_lines.append(f"Total Test Samples:        {len(predictions):,}")
output_lines.append(f"R² Score:                  {r2:.4f}")
output_lines.append(f"Mean Absolute Error:       {mae:.2f} runs")
output_lines.append(f"Root Mean Square Error:    {rmse:.2f} runs")
output_lines.append(f"Best Prediction Error:     {errors.min():.2f} runs")
output_lines.append(f"Worst Prediction Error:    {errors.max():.2f} runs")
output_lines.append(f"Median Error:              {np.median(errors):.2f} runs")
output_lines.append("")
output_lines.append(f"Accuracy within 10 runs:   {accuracy_10:.2f}%")
output_lines.append(f"Accuracy within 20 runs:   {accuracy_20:.2f}%")
output_lines.append(f"Accuracy within 30 runs:   {accuracy_30:.2f}%")
output_lines.append(f"Accuracy within 50 runs:   {accuracy_50:.2f}%")
output_lines.append("")

output_lines.append("="*100)
output_lines.append("ACCURACY BY MATCH STAGE:")
output_lines.append("="*100)
output_lines.append("")
output_lines.append(f"{'Stage':<12} {'Balls':<12} {'Samples':<10} {'R² Score':<12} {'MAE':<12} {'Accuracy':<12}")
output_lines.append("-"*100)

for result in stage_results:
    output_lines.append(
        f"{result['Stage']:<12} {result['Balls']:<12} {result['Samples']:<10} "
        f"{result['R2']:<12.4f} {result['MAE']:<12.2f} {result['Accuracy']:<12.2f}%"
    )

output_lines.append("")

# Detailed predictions table
output_lines.append("="*100)
output_lines.append("DETAILED PREDICTIONS (All Samples):")
output_lines.append("="*100)
output_lines.append("")
output_lines.append(f"{'Row':<6} {'Balls':<8} {'Score':<10} {'Wickets':<10} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Accuracy':<12}")
output_lines.append("-"*100)

for i in range(min(100, len(df))):
    row = df.iloc[i]
    pred = predictions[i]
    actual = y_test[i]
    error = abs(pred - actual)
    accuracy = max(0, 100 - (error / actual * 100))
    
    output_lines.append(
        f"{i+1:<6} {row['balls_bowled']:<8.0f} {row['current_score']:<10.0f} "
        f"{row['wickets_fallen']:<10.0f} {actual:<12.0f} {pred:<12.1f} "
        f"{error:<12.1f} {accuracy:<12.1f}%"
    )

# Error distribution
output_lines.append("")
output_lines.append("="*100)
output_lines.append("ERROR DISTRIBUTION:")
output_lines.append("="*100)
output_lines.append("")

error_ranges = [
    ('Excellent (0-10 runs)', 0, 10),
    ('Very Good (11-20 runs)', 11, 20),
    ('Good (21-30 runs)', 21, 30),
    ('Acceptable (31-50 runs)', 31, 50),
    ('Poor (51-100 runs)', 51, 100),
    ('Very Poor (>100 runs)', 101, 999)
]

for label, low, high in error_ranges:
    count = ((errors >= low) & (errors <= high)).sum()
    pct = count / len(errors) * 100
    output_lines.append(f"{label:<30} {count:>6} samples ({pct:>5.2f}%)")

output_lines.append("")

# Save to file
with open('CHECKING.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print()
print("="*80)
print("RESULTS SAVED TO: CHECKING.txt")
print("="*80)
print()
print(f"Tested {len(predictions):,} predictions across all match stages")
print(f"Overall R²: {r2:.4f} | MAE: {mae:.2f} runs | Accuracy (±20): {accuracy_20:.2f}%")
print()
print("Check CHECKING.txt for full detailed results!")

