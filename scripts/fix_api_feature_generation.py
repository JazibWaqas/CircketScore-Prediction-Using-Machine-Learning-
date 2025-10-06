#!/usr/bin/env python3
"""
Fix API Feature Generation
Make API use actual training data instead of hardcoded defaults
"""

import pandas as pd
import numpy as np

def create_feature_lookup_tables():
    """Create lookup tables from training data for accurate feature generation"""
    print("🔧 CREATING FEATURE LOOKUP TABLES FROM TRAINING DATA")
    print("=" * 60)
    
    # Load training data
    train_df = pd.read_csv('data/simple_enhanced_train.csv')
    print(f"Loaded training data: {train_df.shape}")
    
    # Create venue statistics lookup
    print("\n📊 Creating venue statistics lookup...")
    venue_stats = train_df.groupby('venue_id').agg({
        'venue_avg_runs': 'mean',
        'venue_runs_std': 'mean', 
        'venue_matches': 'mean',
        'venue_high_score': 'max',
        'venue_low_score': 'min'
    }).round(2)
    
    print(f"Created venue stats for {len(venue_stats)} venues")
    print("Sample venue stats:")
    print(venue_stats.head())
    
    # Create team statistics lookup
    print("\n🏏 Creating team statistics lookup...")
    team_stats = train_df.groupby('team_id').agg({
        'team_batting_avg': 'mean',
        'team_batting_std': 'mean',
        'team_form_avg_runs': 'mean',
        'team_form_win_rate': 'mean',
        'team_balance': 'mean'
    }).round(2)
    
    print(f"Created team stats for {len(team_stats)} teams")
    print("Sample team stats:")
    print(team_stats.head())
    
    # Create head-to-head lookup
    print("\n⚔️ Creating head-to-head lookup...")
    h2h_data = []
    
    for _, row in train_df.iterrows():
        team1 = row['team_id']
        team2 = row['opposition']  # This needs to be converted to team_id
        
        # Create both directions
        h2h_data.append({
            'team_a_id': team1,
            'team_b_id': team2,
            'h2h_matches': row['h2h_matches'],
            'h2h_avg_runs': row['h2h_avg_runs'],
            'h2h_win_rate': row['h2h_win_rate']
        })
    
    h2h_df = pd.DataFrame(h2h_data)
    h2h_lookup = h2h_df.groupby(['team_a_id', 'team_b_id']).agg({
        'h2h_matches': 'mean',
        'h2h_avg_runs': 'mean',
        'h2h_win_rate': 'mean'
    }).round(2)
    
    print(f"Created H2H lookup for {len(h2h_lookup)} team pairs")
    
    # Create opposition lookup (team name to team_id mapping)
    print("\n🔄 Creating team name to ID mapping...")
    team_mapping = train_df[['team', 'team_id']].drop_duplicates()
    opposition_mapping = train_df[['opposition', 'team_id']].drop_duplicates()
    
    # Save all lookup tables
    print("\n💾 Saving lookup tables...")
    
    # Save venue stats
    venue_stats.to_csv('Database/venue_stats_lookup.csv')
    print(f"✅ Saved venue_stats_lookup.csv")
    
    # Save team stats  
    team_stats.to_csv('Database/team_stats_lookup.csv')
    print(f"✅ Saved team_stats_lookup.csv")
    
    # Save H2H lookup
    h2h_lookup.to_csv('Database/h2h_stats_lookup.csv')
    print(f"✅ Saved h2h_stats_lookup.csv")
    
    # Save team mappings
    team_mapping.to_csv('Database/team_name_to_id_mapping.csv', index=False)
    opposition_mapping.to_csv('Database/opposition_name_to_id_mapping.csv', index=False)
    print(f"✅ Saved team name mappings")
    
    # Create a summary of the lookup tables
    summary = {
        'venue_stats_count': len(venue_stats),
        'team_stats_count': len(team_stats),
        'h2h_pairs_count': len(h2h_lookup),
        'training_data_shape': train_df.shape,
        'venue_stats_sample': venue_stats.head(3).to_dict(),
        'team_stats_sample': team_stats.head(3).to_dict()
    }
    
    import json
    with open('Database/lookup_tables_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📋 SUMMARY:")
    print(f"  Venue statistics: {len(venue_stats)} venues")
    print(f"  Team statistics: {len(team_stats)} teams") 
    print(f"  H2H statistics: {len(h2h_lookup)} team pairs")
    print(f"  Training data: {train_df.shape[0]} matches, {train_df.shape[1]} features")
    
    return {
        'venue_stats': venue_stats,
        'team_stats': team_stats,
        'h2h_lookup': h2h_lookup,
        'team_mapping': team_mapping
    }

if __name__ == "__main__":
    lookups = create_feature_lookup_tables()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Update Database/run.py to use these lookup tables")
    print(f"2. Replace hardcoded defaults with actual data")
    print(f"3. Test API predictions with real feature values")
    print(f"4. Compare predictions with training data accuracy")
