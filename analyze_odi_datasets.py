#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare ODI_DATA vs ODIS_BALLBYBALL datasets"""

import pandas as pd
import json
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("COMPARING ODI DATASETS")
print("=" * 60)

# 1. Analyze odi_data CSV
print("\nDATASET 1: odi_data/detailed_player_data.csv")
print("-" * 60)
df = pd.read_csv('raw_data/odi_data/detailed_player_data.csv')
print(f"Total records: {len(df):,}")
print(f"Unique matches: {df['match_id'].nunique():,}")
print(f"Unique players: {df['player'].nunique():,}")
print(f"Unique teams: {df['team'].nunique()}")
print(f"Unique venues: {df['venue'].nunique()}")

print(f"\nColumns: {df.columns.tolist()}")

print(f"\nSample record:")
sample = df.iloc[0]
for col in df.columns:
    print(f"  {col}: {sample[col]}")

# Check match_id format
print(f"\nMatch ID examples:")
print(df['match_id'].head(10).tolist())

# 2. Analyze odis_ballbyBall JSON files
print("\n\nDATASET 2: odis_ballbyBall/*.json")
print("-" * 60)
json_files = [f for f in os.listdir('raw_data/odis_ballbyBall') if f.endswith('.json')]
print(f"Total JSON files: {len(json_files):,}")

# Sample a few files
sample_files = json_files[:3]
print(f"\nSample file names: {sample_files}")

# Load one sample
with open(f'raw_data/odis_ballbyBall/{sample_files[0]}', 'r') as f:
    sample_match = json.load(f)

print(f"\nSample match structure:")
print(f"  Match ID (filename): {sample_files[0].replace('.json', '')}")
print(f"  Info keys: {list(sample_match['info'].keys())}")
print(f"  Date: {sample_match['info']['dates']}")
print(f"  Teams: {list(sample_match['info']['players'].keys())}")
print(f"  Venue: {sample_match['info']['venue']}")
print(f"  Winner: {sample_match['info']['outcome'].get('winner', 'N/A')}")

team1, team2 = list(sample_match['info']['players'].keys())
print(f"\n  {team1} players: {len(sample_match['info']['players'][team1])} players")
print(f"    Players: {sample_match['info']['players'][team1][:5]}...")
print(f"\n  {team2} players: {len(sample_match['info']['players'][team2])} players")

print(f"\n  Innings: {len(sample_match['innings'])} innings")

# 3. Cross-reference: Check if match IDs overlap
print("\n\nCROSS-REFERENCING DATASETS")
print("-" * 60)

# Get match IDs from odi_data
odi_data_match_ids = set(df['match_id'].astype(str).unique())
print(f"Match IDs in odi_data: {len(odi_data_match_ids):,}")

# Get match IDs from ballbyball (filenames without .json)
ballbyball_match_ids = set([f.replace('.json', '') for f in json_files])
print(f"Match IDs in ballbyball: {len(ballbyball_match_ids):,}")

# Find overlap
common_matches = odi_data_match_ids & ballbyball_match_ids
print(f"\nCommon matches: {len(common_matches):,}")

only_in_odi_data = odi_data_match_ids - ballbyball_match_ids
only_in_ballbyball = ballbyball_match_ids - odi_data_match_ids

print(f"Only in odi_data: {len(only_in_odi_data):,}")
print(f"Only in ballbyball: {len(only_in_ballbyball):,}")

if len(common_matches) > 0:
    print(f"\nOVERLAP EXISTS! {len(common_matches):,} matches in both datasets")
    print(f"   Coverage: {len(common_matches)/len(ballbyball_match_ids)*100:.1f}% of ballbyball")
else:
    print(f"\nNO OVERLAP! These are completely different matches")

# 4. Summary
print("\n\nSUMMARY & RECOMMENDATIONS")
print("=" * 60)

print("\nDATASET 1 (odi_data):")
print("   - Pre-processed player performance data")
print("   - Already has runs, strike_rate, wickets, economy per player per match")
print("   - Ready to aggregate for career statistics")
print("   - Easier to work with (CSV format)")
print(f"   - Coverage: {df['match_id'].nunique():,} matches")

print("\nDATASET 2 (odis_ballbyBall):")
print("   - Raw ball-by-ball data")
print("   - Complete match context (venue, toss, teams, outcome)")
print("   - Exact playing XI for each match")
print("   - Can calculate ANY statistic we want")
print(f"   - Coverage: {len(json_files):,} matches")

print("\nKEY DIFFERENCES:")
print("   1. odi_data = Player performances (already calculated)")
print("   2. ballbyball = Match context + playing XI (raw data)")

print("\nRECOMMENDED APPROACH:")
if len(common_matches) > 0:
    print("   YES - Use BOTH datasets (they complement each other)")
    print("   - ballbyball for: match context, playing XI, team compositions")
    print("   - odi_data for: quick player career statistics")
    print("   - Merge on match_id where available")
else:
    print("   WARNING - Different match sets - use based on what you need:")
    print("   - ballbyball for: match context, playing XI")
    print("   - odi_data for: player statistics")

