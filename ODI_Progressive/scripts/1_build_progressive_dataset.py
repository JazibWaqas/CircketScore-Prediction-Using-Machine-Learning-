#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUILD PROGRESSIVE ODI DATASET

Creates dataset with samples from multiple match stages (0, 60, 120, 180, 240 balls)
Each row represents match state at that checkpoint with team composition and target final score.

This enables:
- Pre-match prediction (ball 0)
- Progressive prediction (any stage)
- Fantasy team building
- Player what-if analysis
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# Optional progress tracking
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, desc="": x

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("BUILD PROGRESSIVE ODI DATASET")
print("="*80)

# ==============================================================================
# STEP 1: LOAD PLAYER DATABASE
# ==============================================================================

print("\n[1/5] Loading player database...")

player_db_path = '../ODI/data/CURRENT_player_database_977_quality.json'
with open(player_db_path, 'r') as f:
    player_database = json.load(f)

print(f"   ✓ Loaded {len(player_database):,} players")

# ==============================================================================
# STEP 2: CALCULATE TEAM AGGREGATES
# ==============================================================================

def calculate_batting_aggregates(players, player_db):
    """Calculate batting team aggregate features from 11 players"""
    stats = []
    for player in players:
        if player in player_db and 'batting' in player_db[player]:
            stats.append(player_db[player]['batting'])
    
    if len(stats) < 5:
        return None
    
    avgs = [s['average'] for s in stats]
    srs = [s['strike_rate'] for s in stats]
    
    return {
        'team_batting_avg': np.mean(avgs),
        'team_max_batting_avg': np.max(avgs),
        'team_elite_batsmen': sum(1 for a in avgs if a >= 40),
        'team_batting_depth': sum(1 for a in avgs if a >= 30),
        'team_avg_strike_rate': np.mean(srs),
        'team_known_players': len(stats)
    }

def calculate_bowling_aggregates(players, player_db):
    """Calculate bowling team aggregate features from 11 players"""
    stats = []
    for player in players:
        if player in player_db and 'bowling' in player_db[player]:
            bowling = player_db[player]['bowling']
            if bowling.get('economy'):
                stats.append(bowling)
    
    if len(stats) < 3:
        return {
            'opp_bowling_economy': 5.5,
            'opp_elite_bowlers': 0,
            'opp_bowling_depth': 0
        }
    
    econs = [s['economy'] for s in stats]
    
    return {
        'opp_bowling_economy': np.mean(econs),
        'opp_elite_bowlers': sum(1 for e in econs if e < 4.8),
        'opp_bowling_depth': sum(1 for e in econs if e < 5.5)
    }

# ==============================================================================
# STEP 3: PARSE ALL MATCHES
# ==============================================================================

print("\n[2/5] Parsing match files...")

ballbyball_dir = '../raw_data/odis_ballbyBall'
match_files = [f for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"   Found {len(match_files):,} match files")

all_matches = []
parse_errors = 0

for filename in tqdm(match_files, desc="   Parsing"):
    try:
        with open(os.path.join(ballbyball_dir, filename), 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        if 'info' not in match or 'innings' not in match:
            parse_errors += 1
            continue
        
        info = match['info']
        date = info.get('dates', ['2020-01-01'])[0]
        
        all_matches.append({
            'match_id': filename.replace('.json', ''),
            'date': date,
            'match_data': match
        })
    except:
        parse_errors += 1
        continue

all_matches.sort(key=lambda x: x['date'])
print(f"\n   ✓ Parsed {len(all_matches):,} matches")
print(f"   ✗ Skipped {parse_errors} files")

# ==============================================================================
# STEP 4: BUILD PROGRESSIVE DATASET
# ==============================================================================

print("\n[3/5] Building progressive dataset...")
print("   Sampling each match at checkpoints: [0, 60, 120, 180, 240] balls")

# Venue tracking for historical averages
venue_scores = defaultdict(list)

dataset = []
processed = 0
skipped = 0

# Checkpoints to sample at
CHECKPOINTS = [0, 60, 120, 180, 240]

for match_info in tqdm(all_matches, desc="   Processing"):
    try:
        match = match_info['match_data']
        info = match['info']
        match_id = match_info['match_id']
        date = match_info['date']
        
        if 'players' not in info or len(info['players']) != 2:
            skipped += 1
            continue
        
        teams = list(info['players'].keys())
        team_a, team_b = teams[0], teams[1]
        team_a_players = info['players'][team_a]
        team_b_players = info['players'][team_b]
        venue = info.get('venue', 'Unknown')
        
        # Calculate team aggregates
        team_a_batting = calculate_batting_aggregates(team_a_players, player_database)
        team_b_bowling = calculate_bowling_aggregates(team_b_players, player_database)
        
        if not team_a_batting:
            skipped += 1
            continue
        
        # Process Team A's innings
        innings = match.get('innings', [])
        if len(innings) < 1:
            skipped += 1
            continue
        
        team_a_innings = None
        for inning in innings:
            if inning.get('team') == team_a:
                team_a_innings = inning
                break
        
        if not team_a_innings:
            skipped += 1
            continue
        
        # Calculate final score
        final_score = 0
        for over in team_a_innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                final_score += delivery.get('runs', {}).get('total', 0)
        
        if final_score == 0:
            skipped += 1
            continue
        
        # Get venue average (from PAST matches only)
        venue_past = venue_scores[venue]
        venue_avg = np.mean(venue_past) if venue_past else 250.0
        
        # Sample at each checkpoint
        for checkpoint in CHECKPOINTS:
            # Calculate cumulative stats up to this checkpoint
            cumulative_score = 0
            cumulative_wickets = 0
            balls_in_last_10_overs = []
            current_batsman = None
            current_non_striker = None
            
            ball_count = 0
            for over in team_a_innings.get('overs', []):
                if ball_count >= checkpoint:
                    break
                    
                for delivery in over.get('deliveries', []):
                    if ball_count >= checkpoint:
                        break
                    
                    runs = delivery.get('runs', {}).get('total', 0)
                    cumulative_score += runs
                    
                    if 'wickets' in delivery:
                        cumulative_wickets += len(delivery['wickets'])
                    
                    # Track last 60 balls for momentum
                    balls_in_last_10_overs.append(runs)
                    if len(balls_in_last_10_overs) > 60:
                        balls_in_last_10_overs.pop(0)
                    
                    # Track current batsmen (for checkpoints > 0)
                    if checkpoint > 0 and ball_count == checkpoint - 1:
                        current_batsman = delivery.get('batter')
                        current_non_striker = delivery.get('non_striker')
                    
                    ball_count += 1
            
            # Calculate derived features
            balls_bowled = checkpoint
            balls_remaining = 300 - checkpoint
            wickets_remaining = 10 - cumulative_wickets
            runs_last_10 = sum(balls_in_last_10_overs) if balls_in_last_10_overs else 0
            current_rr = (cumulative_score * 6.0 / balls_bowled) if balls_bowled > 0 else 0
            
            # Get current batsman stats (if mid-match)
            batsman_1_avg = 0
            batsman_2_avg = 0
            if checkpoint > 0 and current_batsman and current_batsman in player_database:
                if 'batting' in player_database[current_batsman]:
                    batsman_1_avg = player_database[current_batsman]['batting'].get('average', 0)
            if checkpoint > 0 and current_non_striker and current_non_striker in player_database:
                if 'batting' in player_database[current_non_striker]:
                    batsman_2_avg = player_database[current_non_striker]['batting'].get('average', 0)
            
            # Build row
            row = {
                # Identifiers
                'match_id': match_id,
                'date': date,
                'checkpoint': checkpoint,
                'team_name': team_a,
                'opposition_name': team_b,
                'venue_name': venue,
                
                # Match state
                'current_score': cumulative_score,
                'wickets_fallen': cumulative_wickets,
                'balls_bowled': balls_bowled,
                'balls_remaining': balls_remaining,
                'wickets_remaining': wickets_remaining,
                'runs_last_10_overs': runs_last_10,
                'current_run_rate': current_rr,
                
                # Team aggregates
                **{f'team_{k}': v for k, v in team_a_batting.items()},
                
                # Opposition aggregates
                **team_b_bowling,
                
                # Current batsmen (0 if pre-match)
                'current_batsman_1_avg': batsman_1_avg,
                'current_batsman_2_avg': batsman_2_avg,
                
                # Context
                'venue_avg_score': venue_avg,
                
                # Target
                'final_score': final_score
            }
            
            dataset.append(row)
        
        # Update venue history (for future matches)
        venue_scores[venue].append(final_score)
        
        processed += 1
        
    except Exception as e:
        skipped += 1
        continue

print(f"\n   ✓ Built {len(dataset):,} rows from {processed:,} matches")
print(f"   ✗ Skipped {skipped:,} matches")

# ==============================================================================
# STEP 5: CREATE DATAFRAME AND SAVE
# ==============================================================================

print("\n[4/5] Creating DataFrame...")

df = pd.DataFrame(dataset)

print(f"   Shape: {df.shape}")
print(f"   Matches: {processed}")
print(f"   Rows per match: {len(CHECKPOINTS)}")
print(f"   Features: {len(df.columns) - 7}")  # Minus identifiers and target

# Check distribution by checkpoint
print(f"\n   Rows by checkpoint:")
for cp in CHECKPOINTS:
    count = (df['checkpoint'] == cp).sum()
    print(f"      Ball {cp:3d}: {count:,} rows")

# Split train/test temporally
df = df.sort_values('date')
test_size = 700  # ~350 recent matches
df_train = df.iloc[:-test_size].copy()
df_test = df.iloc[-test_size:].copy()

print(f"\n[5/5] Saving datasets...")

df.to_csv('../ODI_Progressive/data/progressive_full.csv', index=False)
df_train.to_csv('../ODI_Progressive/data/progressive_train.csv', index=False)
df_test.to_csv('../ODI_Progressive/data/progressive_test.csv', index=False)

print(f"\n   ✓ Saved:")
print(f"      Full dataset: {df.shape}")
print(f"      Training: {df_train.shape} ({df_train['date'].min()} to {df_train['date'].max()})")
print(f"      Test: {df_test.shape} ({df_test['date'].min()} to {df_test['date'].max()})")

print("\n" + "="*80)
print("DATASET READY!")
print("="*80)
print(f"\n✓ Features: Match state (7) + Team (6) + Opposition (3) + Context (1) + Batsmen (2)")
print(f"✓ Total rows: {len(df):,}")
print(f"✓ Ready for training!")
print("\n" + "="*80 + "\n")

