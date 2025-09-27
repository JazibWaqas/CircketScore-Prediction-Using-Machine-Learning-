"""
Create ML-ready dataset from the corrected cricket dataset
This script prepares the dataset specifically for machine learning by handling text columns and creating proper features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_ml_ready_dataset():
    """Create a machine learning ready dataset"""
    print("Creating ML-ready dataset from corrected cricket dataset...")
    
    # Load the corrected dataset
    df = pd.read_csv('corrected_cricket_dataset.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Create a copy for ML processing
    ml_df = df.copy()
    
    # Handle text columns by encoding them
    print("\nHandling text columns...")
    
    # Encode team names
    le_team = LabelEncoder()
    ml_df['team_encoded'] = le_team.fit_transform(ml_df['team'])
    
    # Encode opposition names
    le_opposition = LabelEncoder()
    ml_df['opposition_encoded'] = le_opposition.fit_transform(ml_df['opposition'])
    
    # Encode venue names
    le_venue = LabelEncoder()
    ml_df['venue_encoded'] = le_venue.fit_transform(ml_df['venue'])
    
    # Encode city names
    le_city = LabelEncoder()
    ml_df['city_encoded'] = le_city.fit_transform(ml_df['city'])
    
    # Handle match outcome (winner)
    le_outcome = LabelEncoder()
    ml_df['match_outcome_encoded'] = le_outcome.fit_transform(ml_df['match_outcome'])
    
    # Create binary features
    ml_df['is_first_innings'] = (ml_df['innings_number'] == 1).astype(int)
    ml_df['is_high_scoring_venue'] = ml_df['venue_high_scoring']
    
    # Select features for ML (numerical + encoded categorical)
    ml_features = [
        # Match context
        'innings_number', 'overs_bowled', 'run_rate', 'extras', 'boundaries_total',
        'is_first_innings', 'is_high_scoring_venue',
        
        # Team strength
        'team_batting_avg', 'team_strike_rate', 'team_centuries', 'team_fifties',
        'team_bowling_avg', 'team_economy', 'team_wickets', 'team_maidens',
        'team_batsmen', 'team_bowlers', 'team_allrounders', 'team_wicketkeepers',
        
        # Venue features
        'venue_avg_runs', 'venue_avg_rr', 'venue_avg_boundaries',
        
        # Opposition features
        'opp_avg_runs', 'opp_avg_rr', 'opp_avg_boundaries',
        
        # Encoded categorical features
        'team_encoded', 'opposition_encoded', 'venue_encoded', 'city_encoded', 'match_outcome_encoded',
        
        # Derived features
        'team_balance', 'venue_advantage'
    ]
    
    # Create ML dataset
    ml_dataset = ml_df[ml_features + ['total_runs']].copy()
    
    # Handle missing values
    ml_dataset = ml_dataset.fillna(ml_dataset.mean())
    
    # Remove any remaining NaN values
    ml_dataset = ml_dataset.dropna()
    
    print(f"ML-ready dataset shape: {ml_dataset.shape}")
    print(f"Features: {len(ml_features)}")
    print(f"Target variable (total_runs) statistics:")
    print(ml_dataset['total_runs'].describe())
    
    # Save the ML-ready dataset
    ml_dataset.to_csv('ml_ready_cricket_dataset.csv', index=False)
    print("\nML-ready dataset saved to 'ml_ready_cricket_dataset.csv'")
    
    # Display sample
    print("\nSample of ML-ready dataset:")
    print(ml_dataset[['innings_number', 'team_batting_avg', 'venue_avg_runs', 'total_runs']].head())
    
    return ml_dataset

def main():
    """Main function to create ML-ready dataset"""
    ml_dataset = create_ml_ready_dataset()
    print("\n✅ ML-ready dataset created successfully!")
    print("You can now run machine learning models on this dataset.")
    return ml_dataset

if __name__ == "__main__":
    ml_dataset = main()
