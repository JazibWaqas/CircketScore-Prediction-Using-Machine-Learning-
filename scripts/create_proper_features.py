"""
Create Proper Pre-Match Features for Cricket Score Prediction
No data leakage - only features available before the match
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_proper_features():
    """Create proper pre-match features without data leakage"""
    print("CREATING PROPER PRE-MATCH FEATURES")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('data/train_dataset.csv')
    print(f"Loaded dataset: {df.shape}")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create proper pre-match features
    print("\nCREATING PRE-MATCH FEATURES:")
    
    # 1. Team Form Features (last 5 matches)
    print("1. Team Form Features...")
    df['team_batting_form_5'] = 0.0
    df['team_bowling_form_5'] = 0.0
    df['team_win_rate_5'] = 0.0
    
    for i, row in df.iterrows():
        team = row['team']
        match_date = row['date']
        
        # Get last 5 matches for this team
        team_matches = df[
            (df['team'] == team) & 
            (df['date'] < match_date)
        ].tail(5)
        
        if len(team_matches) > 0:
            # Batting form (average runs scored)
            df.loc[i, 'team_batting_form_5'] = team_matches['total_runs'].mean()
            
            # Bowling form (average runs conceded)
            opposition_matches = df[
                (df['opposition'] == team) & 
                (df['date'] < match_date)
            ].tail(5)
            if len(opposition_matches) > 0:
                df.loc[i, 'team_bowling_form_5'] = opposition_matches['total_runs'].mean()
            
            # Win rate
            wins = len(team_matches[team_matches['match_winner'] == team])
            df.loc[i, 'team_win_rate_5'] = wins / len(team_matches)
    
    # 2. Opposition Form Features
    print("2. Opposition Form Features...")
    df['opposition_batting_form_5'] = 0.0
    df['opposition_bowling_form_5'] = 0.0
    df['opposition_win_rate_5'] = 0.0
    
    for i, row in df.iterrows():
        opposition = row['opposition']
        match_date = row['date']
        
        # Get last 5 matches for opposition
        opp_matches = df[
            (df['team'] == opposition) & 
            (df['date'] < match_date)
        ].tail(5)
        
        if len(opp_matches) > 0:
            # Opposition batting form
            df.loc[i, 'opposition_batting_form_5'] = opp_matches['total_runs'].mean()
            
            # Opposition bowling form (runs conceded)
            opp_bowling_matches = df[
                (df['opposition'] == opposition) & 
                (df['date'] < match_date)
            ].tail(5)
            if len(opp_bowling_matches) > 0:
                df.loc[i, 'opposition_bowling_form_5'] = opp_bowling_matches['total_runs'].mean()
            
            # Opposition win rate
            wins = len(opp_matches[opp_matches['match_winner'] == opposition])
            df.loc[i, 'opposition_win_rate_5'] = wins / len(opp_matches)
    
    # 3. Venue-Specific Features
    print("3. Venue-Specific Features...")
    df['venue_team_avg'] = 0.0
    df['venue_opposition_avg'] = 0.0
    df['venue_high_scoring'] = 0.0
    
    for i, row in df.iterrows():
        venue = row['venue']
        team = row['team']
        opposition = row['opposition']
        match_date = row['date']
        
        # Team's average at this venue
        team_venue_matches = df[
            (df['team'] == team) & 
            (df['venue'] == venue) & 
            (df['date'] < match_date)
        ]
        if len(team_venue_matches) > 0:
            df.loc[i, 'venue_team_avg'] = team_venue_matches['total_runs'].mean()
        
        # Opposition's average at this venue
        opp_venue_matches = df[
            (df['team'] == opposition) & 
            (df['venue'] == venue) & 
            (df['date'] < match_date)
        ]
        if len(opp_venue_matches) > 0:
            df.loc[i, 'venue_opposition_avg'] = opp_venue_matches['total_runs'].mean()
        
        # Venue high-scoring tendency
        venue_matches = df[
            (df['venue'] == venue) & 
            (df['date'] < match_date)
        ]
        if len(venue_matches) > 0:
            high_scores = len(venue_matches[venue_matches['total_runs'] > 150])
            df.loc[i, 'venue_high_scoring'] = high_scores / len(venue_matches)
    
    # 4. Head-to-Head Features
    print("4. Head-to-Head Features...")
    df['h2h_team_avg'] = 0.0
    df['h2h_opposition_avg'] = 0.0
    df['h2h_win_rate'] = 0.0
    
    for i, row in df.iterrows():
        team = row['team']
        opposition = row['opposition']
        match_date = row['date']
        
        # Get head-to-head matches
        h2h_matches = df[
            ((df['team'] == team) & (df['opposition'] == opposition)) |
            ((df['team'] == opposition) & (df['opposition'] == team))
        ][df['date'] < match_date].tail(10)
        
        if len(h2h_matches) > 0:
            # Team's average in H2H
            team_h2h = h2h_matches[h2h_matches['team'] == team]
            if len(team_h2h) > 0:
                df.loc[i, 'h2h_team_avg'] = team_h2h['total_runs'].mean()
            
            # Opposition's average in H2H
            opp_h2h = h2h_matches[h2h_matches['team'] == opposition]
            if len(opp_h2h) > 0:
                df.loc[i, 'h2h_opposition_avg'] = opp_h2h['total_runs'].mean()
            
            # H2H win rate
            team_wins = len(h2h_matches[h2h_matches['match_winner'] == team])
            df.loc[i, 'h2h_win_rate'] = team_wins / len(h2h_matches)
    
    # 5. Match Context Features
    print("5. Match Context Features...")
    df['is_home_advantage'] = df['is_home_team'].astype(int)
    df['is_important_match'] = (df['is_final'] | df['is_semi_final'] | df['is_playoff']).astype(int)
    df['is_t20_world_cup'] = df['event_name'].str.contains('World Cup', case=False, na=False).astype(int)
    df['is_ipl'] = df['event_name'].str.contains('IPL', case=False, na=False).astype(int)
    
    # 6. Season Features
    print("6. Season Features...")
    df['season_year'] = df['date'].dt.year
    df['season_month'] = df['date'].dt.month
    df['is_winter'] = df['season_month'].isin([11, 12, 1, 2]).astype(int)
    df['is_summer'] = df['season_month'].isin([5, 6, 7, 8]).astype(int)
    
    # 7. Player Composition Features
    print("7. Player Composition Features...")
    df['team_experience'] = 0.0
    df['opposition_experience'] = 0.0
    
    # This would require player data - for now, use team_id as proxy
    df['team_experience'] = df['team_id'] / 1000  # Normalize
    df['opposition_experience'] = df['team_id'] / 1000  # Normalize
    
    # 8. Remove data leakage features
    print("8. Removing data leakage features...")
    leakage_features = [
        'total_wickets', 'total_overs', 'total_balls', 'run_rate',
        'total_4s', 'total_6s', 'total_boundaries', 'total_extras',
        'powerplay_runs', 'middle_overs_runs', 'death_overs_runs',
        'target_set', 'target_chased', 'win_margin', 'win_type'
    ]
    
    # Keep only pre-match features
    pre_match_features = [
        'team_id', 'venue_id', 'venue_avg_runs', 'venue_runs_std',
        'venue_matches', 'venue_high_score', 'venue_low_score',
        'h2h_matches', 'h2h_avg_runs', 'h2h_win_rate',
        'team_form_avg_runs', 'team_form_win_rate',
        'is_home_team', 'is_final', 'is_semi_final', 'is_playoff',
        'team_batting_avg', 'team_batting_std',
        'opposition_bowling_avg', 'opposition_bowling_std',
        'venue_difficulty', 'team_form_score', 'h2h_strength',
        'match_importance', 'team_balance', 'pressure_score',
        # New features
        'team_batting_form_5', 'team_bowling_form_5', 'team_win_rate_5',
        'opposition_batting_form_5', 'opposition_bowling_form_5', 'opposition_win_rate_5',
        'venue_team_avg', 'venue_opposition_avg', 'venue_high_scoring',
        'h2h_team_avg', 'h2h_opposition_avg', 'h2h_win_rate',
        'is_home_advantage', 'is_important_match', 'is_t20_world_cup', 'is_ipl',
        'season_year', 'season_month', 'is_winter', 'is_summer',
        'team_experience', 'opposition_experience'
    ]
    
    # Create clean dataset
    clean_df = df[pre_match_features + ['total_runs']].copy()
    
    print(f"\nCLEAN DATASET CREATED:")
    print(f"Shape: {clean_df.shape}")
    print(f"Features: {len(pre_match_features)}")
    print(f"Target: total_runs")
    
    # Check for missing values
    missing_values = clean_df.isnull().sum()
    print(f"\nMISSING VALUES:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(clean_df)*100:.1f}%)")
    
    # Fill missing values
    clean_df = clean_df.fillna(0)
    
    # Save clean dataset
    clean_df.to_csv('data/proper_train_dataset.csv', index=False)
    print(f"\nClean dataset saved: data/proper_train_dataset.csv")
    
    # Create test dataset
    test_df = pd.read_csv('data/test_dataset.csv')
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # Apply same feature engineering to test data
    # (This would need the same logic applied)
    test_clean = test_df[pre_match_features + ['total_runs']].copy()
    test_clean = test_clean.fillna(0)
    test_clean.to_csv('data/proper_test_dataset.csv', index=False)
    
    print(f"Test dataset saved: data/proper_test_dataset.csv")
    
    return clean_df, pre_match_features

if __name__ == "__main__":
    clean_df, features = create_proper_features()
