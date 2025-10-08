#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Lookup Tables for ODI Frontend Integration

Purpose: Generate player, venue, and team lookup CSVs with IDs
These are needed for the frontend dropdown menus and database setup
"""

import pandas as pd
import json
import sys

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def create_player_lookup():
    """Create player_lookup.csv from player_database.json"""
    print("\n" + "="*70)
    print("CREATING PLAYER LOOKUP TABLE")
    print("="*70)
    
    # Load player database
    with open('../data/player_database.json', 'r', encoding='utf-8') as f:
        player_db = json.load(f)
    
    # Create player lookup with IDs
    players = []
    for player_id, (player_name, player_data) in enumerate(player_db.items()):
        # Safely get batting and bowling averages
        batting_data = player_data.get('batting') or {}
        bowling_data = player_data.get('bowling') or {}
        
        players.append({
            'player_id': player_id,
            'player_name': player_name,
            'role': player_data.get('role', 'Unknown'),
            'skill_level': player_data.get('skill_level', 'Unknown'),
            'star_rating': player_data.get('star_rating', 0),
            'teams': ','.join(player_data.get('teams', [])),
            'batting_avg': batting_data.get('average', 0),
            'bowling_avg': bowling_data.get('average', 0),
            'total_matches': player_data.get('total_matches', 0)
        })
    
    player_df = pd.DataFrame(players)
    player_df = player_df.sort_values('player_name')
    
    # Save to CSV
    output_path = '../data/player_lookup.csv'
    player_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created player_lookup.csv")
    print(f"   Total players: {len(player_df):,}")
    print(f"   Elite players: {len(player_df[player_df['skill_level'] == 'Elite']):,}")
    print(f"   Star players: {len(player_df[player_df['skill_level'] == 'Star']):,}")
    print(f"   Saved to: {output_path}")
    
    return player_df

def create_venue_lookup():
    """Create venue_lookup.csv from training dataset"""
    print("\n" + "="*70)
    print("CREATING VENUE LOOKUP TABLE")
    print("="*70)
    
    # Load training dataset
    df = pd.read_csv('../data/odi_training_dataset.csv')
    
    # Get unique venues
    venues = sorted(df['venue'].unique())
    
    # Create venue lookup with IDs
    venue_data = []
    for venue_id, venue_name in enumerate(venues):
        # Extract city/country if possible (venue format is often "Stadium, City")
        parts = venue_name.split(',')
        stadium = parts[0].strip() if len(parts) > 0 else venue_name
        city = parts[1].strip() if len(parts) > 1 else ''
        
        venue_data.append({
            'venue_id': venue_id,
            'venue_name': venue_name,
            'stadium': stadium,
            'city': city
        })
    
    venue_df = pd.DataFrame(venue_data)
    
    # Save to CSV
    output_path = '../data/venue_lookup.csv'
    venue_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created venue_lookup.csv")
    print(f"   Total venues: {len(venue_df):,}")
    print(f"   Saved to: {output_path}")
    
    return venue_df

def create_team_lookup():
    """Create team_lookup.csv from training dataset"""
    print("\n" + "="*70)
    print("CREATING TEAM LOOKUP TABLE")
    print("="*70)
    
    # Load training dataset
    df = pd.read_csv('../data/odi_training_dataset.csv')
    
    # Get unique teams from both 'team' and 'opposition' columns
    teams = sorted(pd.concat([df['team'], df['opposition']]).unique())
    
    # Create team lookup with IDs
    team_data = []
    for team_id, team_name in enumerate(teams):
        # Determine if it's a national team or special team
        special_teams = ['Asia XI', 'Africa XI', 'ICC World XI']
        county_teams = ['Nottinghamshire', 'Yorkshire']
        
        if team_name in special_teams:
            team_type = 'Special XI'
        elif team_name in county_teams:
            team_type = 'County'
        else:
            team_type = 'National'
        
        team_data.append({
            'team_id': team_id,
            'team_name': team_name,
            'team_type': team_type
        })
    
    team_df = pd.DataFrame(team_data)
    
    # Save to CSV
    output_path = '../data/team_lookup.csv'
    team_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created team_lookup.csv")
    print(f"   Total teams: {len(team_df):,}")
    print(f"   National teams: {len(team_df[team_df['team_type'] == 'National']):,}")
    print(f"   Special XIs: {len(team_df[team_df['team_type'] == 'Special XI']):,}")
    print(f"   County teams: {len(team_df[team_df['team_type'] == 'County']):,}")
    print(f"   Saved to: {output_path}")
    
    return team_df

def main():
    """Create all lookup tables"""
    print("\n" + "="*70)
    print("ODI LOOKUP TABLES GENERATOR")
    print("Creating reference data for frontend integration")
    print("="*70)
    
    # Create all lookup tables
    player_df = create_player_lookup()
    venue_df = create_venue_lookup()
    team_df = create_team_lookup()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ALL LOOKUP TABLES CREATED")
    print("="*70)
    print(f"\n📊 Players: {len(player_df):,}")
    print(f"📊 Venues:  {len(venue_df):,}")
    print(f"📊 Teams:   {len(team_df):,}")
    print("\n✅ All lookup tables are ready for database setup!")
    print("\nNext steps:")
    print("  1. Run setup_database.py to create SQLite database")
    print("  2. Use run_odi_api.py to start the backend API")
    print("="*70)

if __name__ == '__main__':
    main()

