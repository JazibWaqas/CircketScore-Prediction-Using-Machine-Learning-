#!/usr/bin/env python3
"""
Create Exact Match Lookup
Create lookup tables that preserve exact training data values instead of aggregated averages
"""

import pandas as pd
import numpy as np

def create_exact_match_lookup():
    """Create lookup tables with exact training data values"""
    print("🔧 CREATING EXACT MATCH LOOKUP TABLES")
    print("=" * 60)
    
    # Load training data
    train_df = pd.read_csv('data/simple_enhanced_train.csv')
    print(f"Loaded training data: {train_df.shape}")
    
    # Create exact match lookup - use the most recent match for each team/venue combination
    print("\n📊 Creating exact match lookup...")
    
    # For each unique team_id, venue_id combination, get the most recent match
    exact_lookup = train_df.groupby(['team_id', 'venue_id']).last().reset_index()
    
    # Select only the columns we need for the API
    exact_features = [
        'team_id', 'venue_id', 'opposition', 'total_runs',
        'venue_avg_runs', 'venue_runs_std', 'venue_matches', 'venue_high_score', 'venue_low_score',
        'h2h_matches', 'h2h_avg_runs', 'h2h_win_rate',
        'team_form_avg_runs', 'team_form_win_rate', 'team_balance',
        'venue_difficulty', 'team_form_score', 'h2h_strength'
    ]
    
    exact_lookup = exact_lookup[exact_features]
    
    print(f"Created exact lookup with {len(exact_lookup)} team-venue combinations")
    
    # Save the exact lookup
    exact_lookup.to_csv('Database/exact_match_lookup.csv', index=False)
    print(f"✅ Saved exact_match_lookup.csv")
    
    # Create a simplified lookup for the API
    print("\n🎯 Creating simplified API lookup...")
    
    # Create venue lookup (use exact values from most recent matches at each venue)
    venue_exact = exact_lookup.groupby('venue_id').agg({
        'venue_avg_runs': 'last',
        'venue_runs_std': 'last', 
        'venue_matches': 'last',
        'venue_high_score': 'last',
        'venue_low_score': 'last',
        'venue_difficulty': 'last'
    }).round(2)
    
    print(f"Venue exact lookup: {len(venue_exact)} venues")
    
    # Create team lookup (use exact values from most recent matches for each team)
    team_exact = exact_lookup.groupby('team_id').agg({
        'team_form_avg_runs': 'last',
        'team_form_win_rate': 'last',
        'team_balance': 'last',
        'team_form_score': 'last'
    }).round(2)
    
    print(f"Team exact lookup: {len(team_exact)} teams")
    
    # Create H2H lookup (use exact values from most recent matches between teams)
    h2h_exact = exact_lookup.groupby(['team_id', 'opposition']).agg({
        'h2h_matches': 'last',
        'h2h_avg_runs': 'last',
        'h2h_win_rate': 'last',
        'h2h_strength': 'last'
    }).round(2)
    
    print(f"H2H exact lookup: {len(h2h_exact)} team pairs")
    
    # Save the exact lookups
    venue_exact.to_csv('Database/venue_exact_lookup.csv')
    team_exact.to_csv('Database/team_exact_lookup.csv')
    h2h_exact.to_csv('Database/h2h_exact_lookup.csv')
    
    print(f"\n✅ Saved exact lookup tables:")
    print(f"  venue_exact_lookup.csv: {len(venue_exact)} venues")
    print(f"  team_exact_lookup.csv: {len(team_exact)} teams")
    print(f"  h2h_exact_lookup.csv: {len(h2h_exact)} team pairs")
    
    # Test with our specific case
    print(f"\n🧪 TESTING WITH OUR SPECIFIC CASE:")
    team_a_id = 3  # Australia
    venue_id = 119  # Eden Park
    
    # Check venue
    if venue_id in venue_exact.index:
        venue_data = venue_exact.loc[venue_id]
        print(f"✅ Venue {venue_id} (Eden Park):")
        print(f"  venue_avg_runs: {venue_data['venue_avg_runs']:.2f}")
        print(f"  venue_runs_std: {venue_data['venue_runs_std']:.2f}")
        print(f"  venue_difficulty: {venue_data['venue_difficulty']:.2f}")
    
    # Check team
    if team_a_id in team_exact.index:
        team_data = team_exact.loc[team_a_id]
        print(f"✅ Team {team_a_id} (Australia):")
        print(f"  team_form_avg_runs: {team_data['team_form_avg_runs']:.2f}")
        print(f"  team_balance: {team_data['team_balance']:.2f}")
    
    # Compare with training data
    print(f"\n📊 COMPARISON WITH TRAINING DATA:")
    training_match = train_df[
        (train_df['team_id'] == team_a_id) & 
        (train_df['venue_id'] == venue_id)
    ].iloc[0]
    
    print(f"Training data values:")
    print(f"  venue_avg_runs: {training_match['venue_avg_runs']:.2f}")
    print(f"  team_balance: {training_match['team_balance']:.2f}")
    print(f"  team_form_avg_runs: {training_match['team_form_avg_runs']:.2f}")
    
    return {
        'exact_lookup': exact_lookup,
        'venue_exact': venue_exact,
        'team_exact': team_exact,
        'h2h_exact': h2h_exact
    }

if __name__ == "__main__":
    lookups = create_exact_match_lookup()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Update Database/run.py to use exact lookup tables")
    print(f"2. Replace 'venue_stats_lookup' with 'venue_exact_lookup'")
    print(f"3. Replace 'team_stats_lookup' with 'team_exact_lookup'")
    print(f"4. Replace 'h2h_stats_lookup' with 'h2h_exact_lookup'")
    print(f"5. Test API predictions with exact training data values")
