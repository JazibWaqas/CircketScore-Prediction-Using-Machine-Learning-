import pandas as pd
import numpy as np

# Load the training data to understand the exact feature structure
print("=== ANALYZING TRAINING DATA FEATURES ===")
train_df = pd.read_csv('data/enhanced_train_dataset.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")

# Get the last few columns which are likely the ML features
ml_columns = [
    'team_strength_ratio', 'opposition_strength_ratio', 'strength_difference',
    'team_momentum', 'opposition_momentum', 'team_trend', 'opposition_trend',
    'team_experience', 'opposition_experience', 'match_pressure',
    'venue_familiarity', 'form_vs_opposition', 'venue_advantage',
    'h2h_advantage', 'strength_vs_form'
]

print(f"\n=== ML FEATURE COLUMNS ===")
for col in ml_columns:
    if col in train_df.columns:
        print(f"{col}: {train_df[col].dtype}, range: {train_df[col].min():.3f} to {train_df[col].max():.3f}")

# Check the target variable (total_runs)
print(f"\n=== TARGET VARIABLE (total_runs) ===")
print(f"Range: {train_df['total_runs'].min()} to {train_df['total_runs'].max()}")
print(f"Mean: {train_df['total_runs'].mean():.2f}")
print(f"Std: {train_df['total_runs'].std():.2f}")

# Check some sample values
print(f"\n=== SAMPLE VALUES ===")
sample_row = train_df.iloc[0]
for col in ml_columns:
    if col in train_df.columns:
        print(f"{col}: {sample_row[col]}")

# Check if there are any NaN values
print(f"\n=== MISSING VALUES ===")
for col in ml_columns:
    if col in train_df.columns:
        nan_count = train_df[col].isna().sum()
        print(f"{col}: {nan_count} NaN values")
