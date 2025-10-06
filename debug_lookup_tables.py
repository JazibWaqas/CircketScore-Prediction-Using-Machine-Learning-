#!/usr/bin/env python3
"""
Debug Lookup Tables
Check if the team IDs and venue IDs match between lookup tables and API request
"""

import pandas as pd

def debug_lookup_tables():
    """Debug the lookup tables to see what data is available"""
    print("🔍 DEBUGGING LOOKUP TABLES")
    print("=" * 50)
    
    # Load lookup tables
    try:
        venue_stats = pd.read_csv('Database/venue_stats_lookup.csv', index_col=0)
        team_stats = pd.read_csv('Database/team_stats_lookup.csv', index_col=0)
        h2h_stats = pd.read_csv('Database/h2h_stats_lookup.csv', index_col=[0,1])
        
        print(f"✅ Loaded lookup tables:")
        print(f"  Venue stats: {len(venue_stats)} venues")
        print(f"  Team stats: {len(team_stats)} teams")
        print(f"  H2H stats: {len(h2h_stats)} team pairs")
        
    except Exception as e:
        print(f"❌ Error loading lookup tables: {e}")
        return
    
    # Check specific IDs from our test
    team_a_id = 3  # Australia
    team_b_id = 108  # New Zealand
    venue_id = 119  # Eden Park
    
    print(f"\n🎯 CHECKING SPECIFIC IDs FROM TEST:")
    print(f"Team A ID: {team_a_id} (Australia)")
    print(f"Team B ID: {team_b_id} (New Zealand)")
    print(f"Venue ID: {venue_id} (Eden Park)")
    
    # Check venue stats
    print(f"\n📊 VENUE STATS CHECK:")
    if venue_id in venue_stats.index:
        venue_data = venue_stats.loc[venue_id]
        print(f"✅ Found venue {venue_id} in lookup table:")
        print(f"  venue_avg_runs: {venue_data['venue_avg_runs']:.2f}")
        print(f"  venue_runs_std: {venue_data['venue_runs_std']:.2f}")
        print(f"  venue_matches: {venue_data['venue_matches']:.0f}")
        print(f"  venue_high_score: {venue_data['venue_high_score']:.0f}")
        print(f"  venue_low_score: {venue_data['venue_low_score']:.0f}")
    else:
        print(f"❌ Venue {venue_id} NOT found in lookup table")
        print(f"Available venue IDs: {sorted(venue_stats.index.tolist())[:10]}...")
    
    # Check team stats
    print(f"\n🏏 TEAM STATS CHECK:")
    if team_a_id in team_stats.index:
        team_data = team_stats.loc[team_a_id]
        print(f"✅ Found team {team_a_id} (Australia) in lookup table:")
        print(f"  team_batting_avg: {team_data['team_batting_avg']:.2f}")
        print(f"  team_form_avg_runs: {team_data['team_form_avg_runs']:.2f}")
        print(f"  team_form_win_rate: {team_data['team_form_win_rate']:.2f}")
        print(f"  team_balance: {team_data['team_balance']:.2f}")
    else:
        print(f"❌ Team {team_a_id} (Australia) NOT found in lookup table")
        print(f"Available team IDs: {sorted(team_stats.index.tolist())[:10]}...")
    
    if team_b_id in team_stats.index:
        team_data = team_stats.loc[team_b_id]
        print(f"✅ Found team {team_b_id} (New Zealand) in lookup table:")
        print(f"  team_batting_avg: {team_data['team_batting_avg']:.2f}")
        print(f"  team_form_avg_runs: {team_data['team_form_avg_runs']:.2f}")
        print(f"  team_form_win_rate: {team_data['team_form_win_rate']:.2f}")
        print(f"  team_balance: {team_data['team_balance']:.2f}")
    else:
        print(f"❌ Team {team_b_id} (New Zealand) NOT found in lookup table")
    
    # Check H2H stats
    print(f"\n⚔️ H2H STATS CHECK:")
    h2h_key = (team_a_id, team_b_id)
    if h2h_key in h2h_stats.index:
        h2h_data = h2h_stats.loc[h2h_key]
        print(f"✅ Found H2H stats for teams {team_a_id} vs {team_b_id}:")
        print(f"  h2h_matches: {h2h_data['h2h_matches']:.0f}")
        print(f"  h2h_avg_runs: {h2h_data['h2h_avg_runs']:.2f}")
        print(f"  h2h_win_rate: {h2h_data['h2h_win_rate']:.2f}")
    else:
        print(f"❌ H2H stats for teams {team_a_id} vs {team_b_id} NOT found")
        print(f"Available H2H pairs: {list(h2h_stats.index)[:5]}...")
    
    # Compare with training data
    print(f"\n📚 COMPARING WITH TRAINING DATA:")
    try:
        train_df = pd.read_csv('data/simple_enhanced_train.csv')
        
        # Find the specific match
        match_data = train_df[
            (train_df['team_id'] == team_a_id) & 
            (train_df['venue_id'] == venue_id)
        ]
        
        if len(match_data) > 0:
            match_row = match_data.iloc[0]
            print(f"✅ Found match in training data:")
            print(f"  venue_avg_runs: {match_row['venue_avg_runs']:.2f}")
            print(f"  venue_runs_std: {match_row['venue_runs_std']:.2f}")
            print(f"  team_form_avg_runs: {match_row['team_form_avg_runs']:.2f}")
            print(f"  team_balance: {match_row['team_balance']:.2f}")
            print(f"  h2h_avg_runs: {match_row['h2h_avg_runs']:.2f}")
            print(f"  actual_total_runs: {match_row['total_runs']}")
            
            # Compare with lookup table values
            if venue_id in venue_stats.index:
                venue_lookup = venue_stats.loc[venue_id]
                print(f"\n🔍 COMPARISON - Venue Stats:")
                print(f"  Training venue_avg_runs: {match_row['venue_avg_runs']:.2f}")
                print(f"  Lookup venue_avg_runs: {venue_lookup['venue_avg_runs']:.2f}")
                print(f"  Match: {abs(match_row['venue_avg_runs'] - venue_lookup['venue_avg_runs']) < 0.01}")
            
            if team_a_id in team_stats.index:
                team_lookup = team_stats.loc[team_a_id]
                print(f"\n🔍 COMPARISON - Team Stats:")
                print(f"  Training team_balance: {match_row['team_balance']:.2f}")
                print(f"  Lookup team_balance: {team_lookup['team_balance']:.2f}")
                print(f"  Match: {abs(match_row['team_balance'] - team_lookup['team_balance']) < 0.01}")
        else:
            print(f"❌ Match not found in training data")
            
    except Exception as e:
        print(f"❌ Error loading training data: {e}")

if __name__ == "__main__":
    debug_lookup_tables()
