#!/usr/bin/env python3
"""
Create Correct Train/Test Split
500 matches for testing (randomly from 2017-2025), rest for training
"""

import pandas as pd
import numpy as np
import random

def create_correct_train_test_split():
    """Create correct train/test split with 500 test samples"""
    print("🔧 CREATING CORRECT TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Load the final clean dataset
    try:
        df = pd.read_csv('processed_data/final_clean_dataset.csv')
        print(f"✅ Loaded final clean dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n📅 STEP 1: IDENTIFY 2017-2025 ERA MATCHES")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Define 2017-2025 era
    era_start = pd.to_datetime('2017-01-01')
    era_end = pd.to_datetime('2025-12-31')
    
    # Get 2017-2025 matches
    era_matches = df[
        (df['date'] >= era_start) & 
        (df['date'] <= era_end)
    ].copy()
    
    print(f"  Total 2017-2025 matches: {len(era_matches):,}")
    
    # Get pre-2017 matches
    pre_era_matches = df[df['date'] < era_start].copy()
    
    print(f"  Pre-2017 matches: {len(pre_era_matches):,}")
    
    print(f"\n🎯 STEP 2: CREATE TRAIN/TEST SPLIT")
    
    # Randomly select 500 matches from 2017-2025 for testing
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    if len(era_matches) >= 500:
        test_indices = np.random.choice(era_matches.index, size=500, replace=False)
        test_df = era_matches.loc[test_indices].copy()
        
        # Remove test matches from era matches to get remaining era matches
        remaining_era_matches = era_matches.drop(test_indices)
        
        print(f"  ✅ Selected 500 random matches from 2017-2025 for testing")
        print(f"  ✅ Remaining 2017-2025 matches: {len(remaining_era_matches):,}")
    else:
        print(f"  ⚠️ Only {len(era_matches)} matches in 2017-2025, using all for testing")
        test_df = era_matches.copy()
        remaining_era_matches = pd.DataFrame()
    
    # Combine pre-2017 matches with remaining 2017-2025 matches for training
    train_df = pd.concat([pre_era_matches, remaining_era_matches], ignore_index=True)
    
    print(f"\n📊 FINAL SPLIT:")
    print(f"  Training set: {len(train_df):,} matches")
    print(f"  Test set: {len(test_df):,} matches")
    print(f"  Total: {len(train_df) + len(test_df):,} matches")
    
    print(f"\n📈 STEP 3: ANALYZE DISTRIBUTIONS")
    
    # Check target variable distributions
    print(f"  Training set target statistics:")
    print(f"    Mean: {train_df['total_runs'].mean():.1f}")
    print(f"    Std: {train_df['total_runs'].std():.1f}")
    print(f"    Range: {train_df['total_runs'].min()}-{train_df['total_runs'].max()}")
    
    print(f"  Test set target statistics:")
    print(f"    Mean: {test_df['total_runs'].mean():.1f}")
    print(f"    Std: {test_df['total_runs'].std():.1f}")
    print(f"    Range: {test_df['total_runs'].min()}-{test_df['total_runs'].max()}")
    
    # Check for distribution shift
    mean_diff = abs(train_df['total_runs'].mean() - test_df['total_runs'].mean())
    if mean_diff < 10:
        print(f"  ✅ Good distribution consistency (mean diff: {mean_diff:.1f})")
    else:
        print(f"  ⚠️ Potential distribution shift (mean diff: {mean_diff:.1f})")
    
    print(f"\n💾 STEP 4: SAVE DATASETS")
    
    # Save training dataset
    train_file = 'processed_data/final_training_dataset.csv'
    train_df.to_csv(train_file, index=False)
    print(f"  ✅ Saved training dataset: {train_file}")
    print(f"    Shape: {train_df.shape}")
    
    # Save test dataset
    test_file = 'processed_data/final_test_dataset.csv'
    test_df.to_csv(test_file, index=False)
    print(f"  ✅ Saved test dataset: {test_file}")
    print(f"    Shape: {test_df.shape}")
    
    # Create summary
    summary = {
        'total_samples': len(df),
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'training_period': f"2005-2016 + {len(remaining_era_matches)} from 2017-2025",
        'test_period': "500 random matches from 2017-2025",
        'training_mean': float(train_df['total_runs'].mean()),
        'test_mean': float(test_df['total_runs'].mean()),
        'mean_difference': float(mean_diff),
        'total_features': len(df.columns)
    }
    
    import json
    with open('processed_data/final_split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"  Total dataset: {len(df):,} matches")
    print(f"  Training set: {len(train_df):,} matches")
    print(f"  Test set: {len(test_df):,} matches")
    print(f"  Features: {len(df.columns)}")
    print(f"  Test samples: 500 (as requested)")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'summary': summary,
        'train_file': train_file,
        'test_file': test_file
    }

if __name__ == "__main__":
    result = create_correct_train_test_split()
    
    if result:
        print(f"\n" + "="*60)
        print("CORRECT TRAIN/TEST SPLIT READY")
        print("="*60)
        
        print(f"✅ Training dataset: {result['train_file']}")
        print(f"✅ Test dataset: {result['test_file']}")
        print(f"✅ Training samples: {result['summary']['training_samples']:,}")
        print(f"✅ Test samples: {result['summary']['test_samples']:,}")
        print(f"✅ Features: {result['summary']['total_features']}")
        
        print(f"\n🚀 READY FOR MODEL TRAINING!")
    else:
        print(f"❌ Failed to create correct split")
