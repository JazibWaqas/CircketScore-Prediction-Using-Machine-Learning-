#!/usr/bin/env python3
"""
Analyze the actual training data structure to understand what features the models expect
"""

import pandas as pd
import numpy as np

def analyze_training_data():
    """Analyze the training data structure"""
    print("📊 ANALYZING ACTUAL TRAINING DATA STRUCTURE")
    print("=" * 60)
    
    # Load the training data
    train_df = pd.read_csv('../data/simple_enhanced_train.csv')
    
    print(f"Shape: {train_df.shape}")
    print(f"Target variable: total_runs")
    print(f"Target range: {train_df['total_runs'].min()}-{train_df['total_runs'].max()}")
    print(f"Target mean: {train_df['total_runs'].mean():.2f}")
    print()
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col != 'total_runs']
    print(f"Total features: {len(feature_cols)}")
    print()
    
    # Analyze feature types
    numerical_features = []
    categorical_features = []
    
    for col in feature_cols:
        dtype = train_df[col].dtype
        if dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"📋 FEATURE TYPES:")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print()
    
    print(f"🔢 NUMERICAL FEATURES:")
    for i, col in enumerate(numerical_features[:10]):
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        mean_val = train_df[col].mean()
        print(f"  {i+1:2d}. {col:<25} | Range: {min_val:.2f}-{max_val:.2f} | Mean: {mean_val:.2f}")
    
    print(f"\n📝 CATEGORICAL FEATURES:")
    for i, col in enumerate(categorical_features[:10]):
        unique_count = train_df[col].nunique()
        sample_val = train_df[col].iloc[0]
        print(f"  {i+1:2d}. {col:<25} | Unique: {unique_count} | Sample: {sample_val}")
    
    print(f"\n🔍 COMPLETE FEATURE LIST:")
    for i, col in enumerate(feature_cols):
        dtype = train_df[col].dtype
        if dtype in ['int64', 'float64']:
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            print(f"  {i+1:2d}. {col:<25} | {str(dtype):<10} | {min_val:.2f}-{max_val:.2f}")
        else:
            unique_count = train_df[col].nunique()
            print(f"  {i+1:2d}. {col:<25} | {str(dtype):<10} | {unique_count} unique values")
    
    print(f"\n📊 SAMPLE DATA (first row):")
    sample = train_df.iloc[0]
    for i, col in enumerate(feature_cols):
        value = sample[col]
        dtype = train_df[col].dtype
        if dtype in ['int64', 'float64']:
            print(f"  {i+1:2d}. {col:<25} = {value:.2f}")
        else:
            print(f"  {i+1:2d}. {col:<25} = {value}")

if __name__ == "__main__":
    analyze_training_data()
