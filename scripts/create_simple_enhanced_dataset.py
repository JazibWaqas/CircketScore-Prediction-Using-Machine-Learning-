"""
Create Simple Enhanced Dataset
Build on existing data without complex feature engineering
Focus on removing data leakage and adding basic form features
"""

import pandas as pd
import numpy as np

def create_simple_enhanced_dataset():
    """Create simple enhanced dataset with basic form features"""
    print("CREATING SIMPLE ENHANCED DATASET")
    print("=" * 50)
    
    # Load existing datasets
    train_df = pd.read_csv('data/train_dataset.csv')
    test_df = pd.read_csv('data/test_dataset.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Convert dates
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # Sort by date
    train_df = train_df.sort_values('date').reset_index(drop=True)
    test_df = test_df.sort_values('date').reset_index(drop=True)
    
    # Remove data leakage features
    leakage_features = [
        'total_wickets', 'total_overs', 'total_balls', 'run_rate',
        'total_4s', 'total_6s', 'total_boundaries', 'total_extras',
        'powerplay_runs', 'middle_overs_runs', 'death_overs_runs',
        'target_set', 'target_chased', 'win_margin', 'win_type'
    ]
    
    print(f"Removing {len(leakage_features)} data leakage features...")
    
    # Keep only pre-match features
    pre_match_features = [col for col in train_df.columns if col not in leakage_features]
    
    # Create clean datasets
    clean_train = train_df[pre_match_features].copy()
    clean_test = test_df[pre_match_features].copy()
    
    print(f"Clean training data: {clean_train.shape}")
    print(f"Clean test data: {clean_test.shape}")
    
    # Add simple form features for training data
    print("Adding simple form features...")
    
    # Team recent form (last 3 matches)
    clean_train['team_recent_avg'] = 0.0
    clean_train['opposition_recent_avg'] = 0.0
    
    for i, row in clean_train.iterrows():
        team = row['team']
        opposition = row['opposition']
        match_date = row['date']
        
        # Team's last 3 matches
        team_matches = clean_train[
            (clean_train['team'] == team) & 
            (clean_train['date'] < match_date)
        ].tail(3)
        
        if len(team_matches) > 0:
            clean_train.loc[i, 'team_recent_avg'] = team_matches['total_runs'].mean()
        
        # Opposition's last 3 matches
        opp_matches = clean_train[
            (clean_train['team'] == opposition) & 
            (clean_train['date'] < match_date)
        ].tail(3)
        
        if len(opp_matches) > 0:
            clean_train.loc[i, 'opposition_recent_avg'] = opp_matches['total_runs'].mean()
    
    # Add simple form features for test data (using training data for historical context)
    print("Adding form features to test data...")
    
    clean_test['team_recent_avg'] = 0.0
    clean_test['opposition_recent_avg'] = 0.0
    
    for i, row in clean_test.iterrows():
        team = row['team']
        opposition = row['opposition']
        match_date = row['date']
        
        # Team's last 3 matches from training data
        team_matches = clean_train[
            (clean_train['team'] == team) & 
            (clean_train['date'] < match_date)
        ].tail(3)
        
        if len(team_matches) > 0:
            clean_test.loc[i, 'team_recent_avg'] = team_matches['total_runs'].mean()
        
        # Opposition's last 3 matches from training data
        opp_matches = clean_train[
            (clean_train['team'] == opposition) & 
            (clean_train['date'] < match_date)
        ].tail(3)
        
        if len(opp_matches) > 0:
            clean_test.loc[i, 'opposition_recent_avg'] = opp_matches['total_runs'].mean()
    
    # Add match context features
    print("Adding match context features...")
    
    # Training data
    clean_train['is_home_advantage'] = clean_train['is_home_team'].astype(int)
    clean_train['is_important_match'] = (clean_train['is_final'] | clean_train['is_semi_final'] | clean_train['is_playoff']).astype(int)
    clean_train['is_t20_world_cup'] = clean_train['event_name'].str.contains('World Cup', case=False, na=False).astype(int)
    clean_train['is_ipl'] = clean_train['event_name'].str.contains('IPL', case=False, na=False).astype(int)
    
    # Test data
    clean_test['is_home_advantage'] = clean_test['is_home_team'].astype(int)
    clean_test['is_important_match'] = (clean_test['is_final'] | clean_test['is_semi_final'] | clean_test['is_playoff']).astype(int)
    clean_test['is_t20_world_cup'] = clean_test['event_name'].str.contains('World Cup', case=False, na=False).astype(int)
    clean_test['is_ipl'] = clean_test['event_name'].str.contains('IPL', case=False, na=False).astype(int)
    
    # Add season features
    print("Adding season features...")
    
    # Training data
    clean_train['season_year'] = clean_train['date'].dt.year
    clean_train['season_month'] = clean_train['date'].dt.month
    clean_train['is_winter'] = clean_train['season_month'].isin([11, 12, 1, 2]).astype(int)
    clean_train['is_summer'] = clean_train['season_month'].isin([5, 6, 7, 8]).astype(int)
    
    # Test data
    clean_test['season_year'] = clean_test['date'].dt.year
    clean_test['season_month'] = clean_test['date'].dt.month
    clean_test['is_winter'] = clean_test['season_month'].isin([11, 12, 1, 2]).astype(int)
    clean_test['is_summer'] = clean_test['season_month'].isin([5, 6, 7, 8]).astype(int)
    
    # Fill missing values
    clean_train = clean_train.fillna(0)
    clean_test = clean_test.fillna(0)
    
    # Save enhanced datasets
    clean_train.to_csv('data/simple_enhanced_train.csv', index=False)
    clean_test.to_csv('data/simple_enhanced_test.csv', index=False)
    
    print(f"\nEnhanced datasets saved:")
    print(f"Training: data/simple_enhanced_train.csv ({clean_train.shape})")
    print(f"Test: data/simple_enhanced_test.csv ({clean_test.shape})")
    
    # Show feature summary
    print(f"\nFEATURE SUMMARY:")
    print(f"Total features: {len(clean_train.columns)}")
    print(f"Target variable: total_runs")
    print(f"Data leakage removed: {len(leakage_features)} features")
    print(f"New features added: 8")
    
    print(f"\nNew features:")
    new_features = ['team_recent_avg', 'opposition_recent_avg', 'is_home_advantage', 
                   'is_important_match', 'is_t20_world_cup', 'is_ipl', 'is_winter', 'is_summer']
    for feature in new_features:
        print(f"  - {feature}")
    
    return clean_train, clean_test

if __name__ == "__main__":
    train_df, test_df = create_simple_enhanced_dataset()
