#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze the 3,361 missing matches in odi_data"""

import pandas as pd
import json
import os
import sys
from collections import Counter, defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("ANALYZING MISSING MATCHES")
print("=" * 60)

# Load odi_data
df = pd.read_csv('raw_data/odi_data/detailed_player_data.csv')
odi_data_match_ids = set(df['match_id'].astype(str).unique())
print(f"odi_data matches: {len(odi_data_match_ids):,}")

# Get ballbyball matches
json_files = [f for f in os.listdir('raw_data/odis_ballbyBall') if f.endswith('.json')]
ballbyball_match_ids = set([f.replace('.json', '') for f in json_files])
print(f"ballbyball matches: {len(ballbyball_match_ids):,}")

# Find missing matches
missing_matches = ballbyball_match_ids - odi_data_match_ids
print(f"\nMissing from odi_data: {len(missing_matches):,} matches")
print(f"Percentage missing: {len(missing_matches)/len(ballbyball_match_ids)*100:.1f}%")

# Analyze missing matches
print("\nANALYZING MISSING MATCHES")
print("-" * 60)

sample_missing = list(missing_matches)[:50]  # Sample 50 missing matches

missing_info = []
for match_id in sample_missing:
    try:
        with open(f'raw_data/odis_ballbyBall/{match_id}.json', 'r') as f:
            match = json.load(f)
            
        info = match['info']
        missing_info.append({
            'match_id': match_id,
            'date': info['dates'][0] if 'dates' in info else 'Unknown',
            'venue': info.get('venue', 'Unknown'),
            'teams': list(info['players'].keys()),
            'gender': info.get('gender', 'Unknown'),
            'match_type': info.get('match_type', 'Unknown'),
            'event': info.get('event', {}).get('name', 'Unknown')
        })
    except Exception as e:
        print(f"Error reading {match_id}: {e}")

# Convert to DataFrame
missing_df = pd.DataFrame(missing_info)

print(f"\nSample of {len(missing_df)} missing matches:")
print("\nDate Range:")
if len(missing_df) > 0 and 'date' in missing_df.columns:
    dates = pd.to_datetime(missing_df['date'], errors='coerce')
    print(f"  Earliest: {dates.min()}")
    print(f"  Latest: {dates.max()}")

print("\nMatch Types:")
if 'match_type' in missing_df.columns:
    for match_type, count in missing_df['match_type'].value_counts().items():
        print(f"  {match_type}: {count}")

print("\nGender:")
if 'gender' in missing_df.columns:
    for gender, count in missing_df['gender'].value_counts().items():
        print(f"  {gender}: {count}")

print("\nTop Events/Tournaments:")
if 'event' in missing_df.columns:
    for event, count in missing_df['event'].value_counts().head(10).items():
        print(f"  {event}: {count}")

# Now analyze AVAILABLE matches in odi_data
print("\n\nANALYZING AVAILABLE MATCHES (in odi_data)")
print("-" * 60)

print(f"\nDate range in odi_data:")
# Get dates from a sample of available matches
available_sample = list(odi_data_match_ids)[:50]
available_info = []

for match_id in available_sample:
    try:
        with open(f'raw_data/odis_ballbyBall/{match_id}.json', 'r') as f:
            match = json.load(f)
        info = match['info']
        available_info.append({
            'match_id': match_id,
            'date': info['dates'][0] if 'dates' in info else 'Unknown',
            'event': info.get('event', {}).get('name', 'Unknown'),
            'teams': list(info['players'].keys())
        })
    except:
        pass

available_df = pd.DataFrame(available_info)
if len(available_df) > 0:
    dates = pd.to_datetime(available_df['date'], errors='coerce')
    print(f"  Sample earliest: {dates.min()}")
    print(f"  Sample latest: {dates.max()}")

# CRITICAL ANALYSIS
print("\n\nCRITICAL ANALYSIS")
print("=" * 60)

# Check if odi_data has player stats for players in missing matches
print("\nPlayer Coverage Analysis:")
all_players_in_odi_data = set(df['player'].unique())
print(f"Unique players in odi_data: {len(all_players_in_odi_data):,}")

# Get players from a sample of missing matches
missing_players = set()
for match_id in list(missing_matches)[:100]:
    try:
        with open(f'raw_data/odis_ballbyBall/{match_id}.json', 'r') as f:
            match = json.load(f)
        for team, players in match['info']['players'].items():
            missing_players.update(players)
    except:
        pass

print(f"Players in missing matches (sample of 100): {len(missing_players):,}")

# Find overlap
common_players = all_players_in_odi_data & missing_players
print(f"Players who appear in BOTH: {len(common_players):,}")
print(f"Overlap: {len(common_players)/len(missing_players)*100:.1f}%")

print("\nIMPACT ASSESSMENT")
print("-" * 60)

print("\nIF YOU USE ONLY ODI_DATA (2,400 matches):")
print("  - Player stats based on 52K performance records")
print("  - Training dataset: 2,400 matches")
print("  - Missing 3,361 recent matches (58% of data)")

print("\nIF YOU USE BALLBYBALL (5,761 matches) + ODI_DATA (player stats):")
print("  - Player stats from 2,400 matches (odi_data)")
print("  - Training dataset: 5,761 matches (all)")
print("  - For 3,361 missing matches:")
print("    - Can still use players who appear in odi_data")
print("    - May need to calculate stats for new players from ballbyball")

print("\nRECOMMENDATION:")
print("-" * 60)

if len(common_players) > len(missing_players) * 0.5:
    print("GOOD NEWS: Most players in missing matches exist in odi_data")
    print("  - Use odi_data for player career stats")
    print("  - Use ALL 5,761 ballbyball matches for training")
    print("  - For new players: calculate from ballbyball innings data")
else:
    print("CONCERN: Many new players in missing matches")
    print("  - Need to calculate their stats from ballbyball")
    print("  - Or focus on 2,400 matches with complete player data")

print("\nWHAT YOU SHOULD DO:")
print("1. Use odi_data to build player database (fast, clean)")
print("2. Use ALL 5,761 ballbyball matches for training")
print("3. For players not in odi_data:")
print("   - Calculate their stats from ballbyball innings")
print("   - Or use average player stats as fallback")
print("   - Or exclude matches with too many unknown players")

