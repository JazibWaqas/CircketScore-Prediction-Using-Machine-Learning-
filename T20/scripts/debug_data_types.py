#!/usr/bin/env python3
"""
Debug data types in the cleaned dataset
"""

import pandas as pd
import numpy as np

def debug_data_types():
    """Debug data types in the cleaned dataset"""
    print("🔍 DEBUGGING DATA TYPES")
    print("=" * 40)
    
    # Load the cleaned dataset
    df = pd.read_csv('processed_data/cleaned_cricket_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n📊 DATA TYPES:")
    print(df.dtypes)
    
    print(f"\n🔍 PROBLEMATIC COLUMNS:")
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object' or dtype == 'bool' or 'datetime' in str(dtype):
            print(f"  {col}: {dtype}")
            if dtype == 'object':
                print(f"    Sample values: {df[col].head(3).tolist()}")
            elif dtype == 'bool':
                print(f"    Sample values: {df[col].head(3).tolist()}")
    
    # Check for any remaining non-numeric columns
    X = df.drop('total_runs', axis=1)
    print(f"\n🎯 FEATURE COLUMNS ANALYSIS:")
    for col in X.columns:
        print(f"  {col}: {X[col].dtype}")
        if X[col].dtype == 'object':
            print(f"    Unique values: {X[col].nunique()}")
            print(f"    Sample: {X[col].head(3).tolist()}")
        elif X[col].dtype == 'bool':
            print(f"    Unique values: {X[col].nunique()}")
            print(f"    Sample: {X[col].head(3).tolist()}")

if __name__ == "__main__":
    debug_data_types()
