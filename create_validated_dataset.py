"""
Create Validated T20 Dataset with Proper IDs
Clean the comprehensive dataset and add player/match IDs for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_validated_dataset():
    """Create a clean, validated dataset with proper IDs"""
    print("Creating Validated T20 Dataset with IDs")
    print("=" * 60)
    
    # Load the comprehensive dataset
    df = pd.read_csv('comprehensive_t20_dataset.csv')
    print(f"Original dataset: {df.shape}")
    
    # Create a copy for cleaning
    clean_df = df.copy()
    
    print("\nCLEANING DATA:")
    
    # 1. Remove invalid records
    print("1. Removing invalid records...")
    
    # Remove negative runs or extremely high runs
    invalid_runs = clean_df[(clean_df['total_runs'] < 0) | (clean_df['total_runs'] > 300)]
    print(f"   Removing {len(invalid_runs)} records with invalid runs")
    clean_df = clean_df[~((clean_df['total_runs'] < 0) | (clean_df['total_runs'] > 300))]
    
    # Remove invalid overs
    invalid_overs = clean_df[(clean_df['total_overs'] < 0) | (clean_df['total_overs'] > 25)]
    print(f"   Removing {len(invalid_overs)} records with invalid overs")
    clean_df = clean_df[~((clean_df['total_overs'] < 0) | (clean_df['total_overs'] > 25))]
    
    # 2. Filter to 2-team matches only
    print("2. Filtering to 2-team matches only...")
    teams_per_match = clean_df.groupby('match_id').size()
    valid_matches = teams_per_match[teams_per_match == 2].index
    print(f"   Keeping {len(valid_matches)} matches with exactly 2 teams")
    clean_df = clean_df[clean_df['match_id'].isin(valid_matches)]
    
    # 3. Create proper IDs
    print("3. Creating proper IDs...")
    
    # Create match IDs (already exist, but ensure they're clean)
    clean_df['match_id'] = clean_df['match_id'].astype(str)
    
    # Create team IDs
    le_team = LabelEncoder()
    clean_df['team_id'] = le_team.fit_transform(clean_df['team'])
    
    # Create venue IDs
    le_venue = LabelEncoder()
    clean_df['venue_id'] = le_venue.fit_transform(clean_df['venue'])
    
    # Create player IDs (from team_players list)
    print("4. Creating player IDs...")
    
    # Extract all unique players
    all_players = set()
    for players_list in clean_df['team_players']:
        if pd.notna(players_list) and players_list != '[]':
            # Parse the string representation of list
            players = eval(players_list) if isinstance(players_list, str) else players_list
            all_players.update(players)
    
    all_players = list(all_players)
    print(f"   Found {len(all_players)} unique players")
    
    # Create player ID mapping
    le_player = LabelEncoder()
    le_player.fit(all_players)
    
    # Create player IDs for each team
    def get_player_ids(players_list):
        if pd.isna(players_list) or players_list == '[]':
            return []
        players = eval(players_list) if isinstance(players_list, str) else players_list
        return le_player.transform(players).tolist()
    
    clean_df['team_player_ids'] = clean_df['team_players'].apply(get_player_ids)
    
    # 4. Handle missing values
    print("5. Handling missing values...")
    
    # Fill missing values with appropriate defaults
    clean_df['match_winner'] = clean_df['match_winner'].fillna('Unknown')
    clean_df['player_of_match'] = clean_df['player_of_match'].fillna('Unknown')
    clean_df['event_name'] = clean_df['event_name'].fillna('Unknown')
    clean_df['match_number'] = clean_df['match_number'].fillna(1)
    
    # Fill venue statistics with median values
    clean_df['venue_avg_runs'] = clean_df['venue_avg_runs'].fillna(clean_df['total_runs'].median())
    clean_df['venue_runs_std'] = clean_df['venue_runs_std'].fillna(clean_df['total_runs'].std())
    clean_df['venue_matches'] = clean_df['venue_matches'].fillna(1)
    clean_df['venue_high_score'] = clean_df['venue_high_score'].fillna(clean_df['total_runs'].max())
    clean_df['venue_low_score'] = clean_df['venue_low_score'].fillna(clean_df['total_runs'].min())
    
    # Fill head-to-head statistics
    clean_df['h2h_matches'] = clean_df['h2h_matches'].fillna(0)
    clean_df['h2h_avg_runs'] = clean_df['h2h_avg_runs'].fillna(clean_df['total_runs'].mean())
    clean_df['h2h_win_rate'] = clean_df['h2h_win_rate'].fillna(0.5)
    
    # Fill team form statistics
    clean_df['team_form_avg_runs'] = clean_df['team_form_avg_runs'].fillna(clean_df['total_runs'].mean())
    clean_df['team_form_win_rate'] = clean_df['team_form_win_rate'].fillna(0.5)
    
    # Fill target-related fields
    clean_df['target_set'] = clean_df['target_set'].fillna(0)
    clean_df['target_chased'] = clean_df['target_chased'].fillna(0)
    clean_df['win_margin'] = clean_df['win_margin'].fillna(0)
    
    # 5. Improve home team detection
    print("6. Improving home team detection...")
    
    # Simple home team detection based on venue name containing team name
    def is_home_team(row):
        venue = str(row['venue']).lower()
        team = str(row['team']).lower()
        
        # Check if venue contains team name or country
        if team in venue:
            return True
        
        # Check for common home venues
        home_venues = {
            'australia': ['melbourne', 'sydney', 'brisbane', 'perth', 'adelaide'],
            'india': ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad'],
            'england': ['london', 'birmingham', 'manchester', 'leeds', 'southampton'],
            'pakistan': ['karachi', 'lahore', 'islamabad', 'rawalpindi'],
            'south africa': ['johannesburg', 'cape town', 'durban', 'pretoria'],
            'new zealand': ['auckland', 'wellington', 'christchurch', 'hamilton'],
            'sri lanka': ['colombo', 'kandy', 'galle'],
            'bangladesh': ['dhaka', 'chittagong'],
            'west indies': ['barbados', 'jamaica', 'trinidad', 'antigua']
        }
        
        for country, venues in home_venues.items():
            if country in team:
                for venue_name in venues:
                    if venue_name in venue:
                        return True
        
        return False
    
    clean_df['is_home_team'] = clean_df.apply(is_home_team, axis=1)
    
    # 6. Create additional useful features
    print("7. Creating additional features...")
    
    # Team strength based on historical performance
    team_strength = clean_df.groupby('team')['total_runs'].agg(['mean', 'std']).reset_index()
    team_strength.columns = ['team', 'team_batting_avg', 'team_batting_std']
    clean_df = clean_df.merge(team_strength, on='team', how='left')
    
    # Opposition strength
    opposition_strength = clean_df.groupby('opposition')['total_runs'].agg(['mean', 'std']).reset_index()
    opposition_strength.columns = ['opposition', 'opposition_bowling_avg', 'opposition_bowling_std']
    clean_df = clean_df.merge(opposition_strength, on='opposition', how='left')
    
    # Venue difficulty (how hard it is to score at this venue)
    clean_df['venue_difficulty'] = clean_df['venue_avg_runs'] / clean_df['total_runs'].mean()
    
    # Team form score (recent performance)
    clean_df['team_form_score'] = clean_df['team_form_avg_runs'] / clean_df['total_runs'].mean()
    
    # Head-to-head strength
    clean_df['h2h_strength'] = clean_df['h2h_avg_runs'] / clean_df['total_runs'].mean()
    
    # Match importance score
    clean_df['match_importance'] = clean_df['is_final'] * 3 + clean_df['is_semi_final'] * 2 + clean_df['is_playoff'] * 1
    
    # Team balance (boundaries per over)
    clean_df['team_balance'] = clean_df['total_boundaries'] / (clean_df['total_overs'] + 1)
    
    # Pressure score (match importance + home advantage)
    clean_df['pressure_score'] = clean_df['match_importance'] + (clean_df['is_home_team'] * 0.5)
    
    # 7. Create lookup tables
    print("8. Creating lookup tables...")
    
    # Team lookup
    team_lookup = pd.DataFrame({
        'team_id': range(len(le_team.classes_)),
        'team_name': le_team.classes_
    })
    
    # Venue lookup
    venue_lookup = pd.DataFrame({
        'venue_id': range(len(le_venue.classes_)),
        'venue_name': le_venue.classes_
    })
    
    # Player lookup
    player_lookup = pd.DataFrame({
        'player_id': range(len(le_player.classes_)),
        'player_name': le_player.classes_
    })
    
    # 8. Final validation
    print("9. Final validation...")
    
    # Check for remaining issues
    remaining_issues = []
    
    if clean_df['total_runs'].min() < 0:
        remaining_issues.append("Negative runs still present")
    if clean_df['total_overs'].min() < 0:
        remaining_issues.append("Negative overs still present")
    if clean_df.isnull().sum().sum() > 0:
        remaining_issues.append("Missing values still present")
    
    if remaining_issues:
        print(f"   WARNING: {remaining_issues}")
    else:
        print("   SUCCESS: All issues resolved!")
    
    # 9. Save the validated dataset
    print("10. Saving validated dataset...")
    
    # Save main dataset
    clean_df.to_csv('validated_t20_dataset.csv', index=False)
    
    # Save lookup tables
    team_lookup.to_csv('team_lookup.csv', index=False)
    venue_lookup.to_csv('venue_lookup.csv', index=False)
    player_lookup.to_csv('player_lookup.csv', index=False)
    
    # Save encoders for future use
    import joblib
    joblib.dump(le_team, 'team_encoder.pkl')
    joblib.dump(le_venue, 'venue_encoder.pkl')
    joblib.dump(le_player, 'player_encoder.pkl')
    
    print(f"\nVALIDATED DATASET CREATED:")
    print(f"   Main dataset: validated_t20_dataset.csv ({clean_df.shape})")
    print(f"   Team lookup: team_lookup.csv ({len(team_lookup)} teams)")
    print(f"   Venue lookup: venue_lookup.csv ({len(venue_lookup)} venues)")
    print(f"   Player lookup: player_lookup.csv ({len(player_lookup)} players)")
    print(f"   Encoders: team_encoder.pkl, venue_encoder.pkl, player_encoder.pkl")
    
    print(f"\nDATASET STATISTICS:")
    print(f"   Records: {len(clean_df):,}")
    print(f"   Matches: {clean_df['match_id'].nunique():,}")
    print(f"   Teams: {clean_df['team'].nunique()}")
    print(f"   Venues: {clean_df['venue'].nunique()}")
    print(f"   Players: {len(player_lookup)}")
    print(f"   Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    print(f"   Runs range: {clean_df['total_runs'].min()} to {clean_df['total_runs'].max()}")
    
    return clean_df, team_lookup, venue_lookup, player_lookup

if __name__ == "__main__":
    df, teams, venues, players = create_validated_dataset()
