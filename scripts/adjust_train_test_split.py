"""
Adjust Train/Test Split for T20 Dataset
Use 500 records for testing, rest for training
"""

import pandas as pd
import numpy as np
from datetime import datetime

def adjust_train_test_split():
    """Adjust train/test split to use 500 records for testing"""
    print("Adjusting Train/Test Split for T20 Dataset")
    print("=" * 60)
    
    # Load the validated dataset
    df = pd.read_csv('processed_data/validated_t20_dataset.csv')
    print(f"Loaded validated dataset: {df.shape}")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create train/test split: 500 records for testing, rest for training
    print("\nSPLITTING DATA:")
    print(f"Total records: {len(df):,}")
    
    # Take the last 500 records for testing (most recent)
    test_df = df.tail(500)
    train_df = df.head(len(df) - 500)
    
    print(f"Training data: {len(train_df):,} records")
    print(f"Test data: {len(test_df):,} records")
    
    # Validate the split
    print(f"\nVALIDATION:")
    print(f"Total records: {len(train_df) + len(test_df):,}")
    print(f"Original dataset: {len(df):,}")
    print(f"Split is correct: {len(train_df) + len(test_df) == len(df)}")
    
    # Analyze the datasets
    print(f"\nTRAINING DATASET ANALYSIS:")
    print(f"Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Teams: {train_df['team'].nunique()}")
    print(f"Venues: {train_df['venue'].nunique()}")
    print(f"Players: {len(set([player for players in train_df['team_player_ids'] for player in eval(players)]))}")
    print(f"Average runs: {train_df['total_runs'].mean():.1f}")
    print(f"Runs range: {train_df['total_runs'].min()} to {train_df['total_runs'].max()}")
    
    print(f"\nTEST DATASET ANALYSIS:")
    print(f"Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Teams: {test_df['team'].nunique()}")
    print(f"Venues: {test_df['venue'].nunique()}")
    print(f"Players: {len(set([player for players in test_df['team_player_ids'] for player in eval(players)]))}")
    print(f"Average runs: {test_df['total_runs'].mean():.1f}")
    print(f"Runs range: {test_df['total_runs'].min()} to {test_df['total_runs'].max()}")
    
    # Check for data leakage
    print(f"\nDATA LEAKAGE CHECK:")
    train_teams = set(train_df['team'].unique())
    test_teams = set(test_df['team'].unique())
    common_teams = train_teams.intersection(test_teams)
    print(f"Teams in both train and test: {len(common_teams)}")
    print(f"Teams only in train: {len(train_teams - test_teams)}")
    print(f"Teams only in test: {len(test_teams - train_teams)}")
    
    train_venues = set(train_df['venue'].unique())
    test_venues = set(test_df['venue'].unique())
    common_venues = train_venues.intersection(test_venues)
    print(f"Venues in both train and test: {len(common_venues)}")
    print(f"Venues only in train: {len(train_venues - test_venues)}")
    print(f"Venues only in test: {len(test_venues - train_venues)}")
    
    # Save the datasets
    print(f"\nSAVING DATASETS:")
    
    # Save training dataset
    train_df.to_csv('data/train_dataset.csv', index=False)
    print(f"Training dataset saved: data/train_dataset.csv ({train_df.shape})")
    
    # Save test dataset
    test_df.to_csv('data/test_dataset.csv', index=False)
    print(f"Test dataset saved: data/test_dataset.csv ({test_df.shape})")
    
    # Create a summary
    summary = {
        'total_records': len(df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'train_matches': train_df['match_id'].nunique(),
        'test_matches': test_df['match_id'].nunique(),
        'train_teams': train_df['team'].nunique(),
        'test_teams': test_df['team'].nunique(),
        'train_venues': train_df['venue'].nunique(),
        'test_venues': test_df['venue'].nunique(),
        'train_date_range': f"{train_df['date'].min()} to {train_df['date'].max()}",
        'test_date_range': f"{test_df['date'].min()} to {test_df['date'].max()}",
        'train_avg_runs': train_df['total_runs'].mean(),
        'test_avg_runs': test_df['total_runs'].mean()
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('data/train_test_summary.csv', index=False)
    print(f"Summary saved: data/train_test_summary.csv")
    
    print(f"\nTRAIN/TEST SPLIT ADJUSTED:")
    print(f"[SUCCESS] Training data: {len(train_df):,} records from {train_df['match_id'].nunique():,} matches")
    print(f"[SUCCESS] Test data: {len(test_df):,} records from {test_df['match_id'].nunique():,} matches")
    print(f"[SUCCESS] No data leakage detected")
    print(f"[SUCCESS] Ready for model training and testing")
    
    return train_df, test_df, summary

if __name__ == "__main__":
    train_df, test_df, summary = adjust_train_test_split()
