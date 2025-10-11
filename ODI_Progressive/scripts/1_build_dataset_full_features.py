#!/usr/bin/env python3
"""
BUILD PROGRESSIVE ODI DATASET WITH FULL FEATURES

Creates dataset with 14+ features as per project specification:
- Match State (6): current_score, wickets_fallen, balls_bowled, balls_remaining, runs_last_10, crr
- Batting Team (3): team_batting_avg, team_elite_batsmen, team_batting_depth
- Opposition (3): opp_bowling_economy, opp_elite_bowlers, opp_bowling_depth
- Venue (2): venue_avg_score, venue (categorical)
- Current Batsmen (2): batsman_1_avg, batsman_2_avg (for mid-match only)

Checkpoints: ball 1, 60, 120, 180, 240 (5 per match)
"""

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict

print("\n" + "="*80)
print("BUILD PROGRESSIVE ODI DATASET - FULL FEATURES")
print("="*80)

# ==============================================================================
# STEP 1: LOAD PLAYER DATABASE
# ==============================================================================

print("\n[1/6] Loading player database...")

player_db_path = '../../ODI/data/CURRENT_player_database_977_quality.json'
with open(player_db_path, 'r') as f:
    player_database = json.load(f)

print(f"   Loaded {len(player_database):,} players")

# ==============================================================================
# STEP 2: HELPER FUNCTIONS
# ==============================================================================

def calculate_batting_aggregates(players, player_db):
    """Calculate batting team aggregate features from 11 players"""
    stats = []
    for player in players:
        if player in player_db and 'batting' in player_db[player]:
            batting = player_db[player]['batting']
            if batting.get('average'):
                stats.append(batting)
    
    if len(stats) < 5:
        return {
            'team_batting_avg': 35.0,
            'team_elite_batsmen': 0,
            'team_batting_depth': 0
        }
    
    avgs = [s['average'] for s in stats]
    
    return {
        'team_batting_avg': np.mean(avgs),
        'team_elite_batsmen': sum(1 for a in avgs if a >= 40),
        'team_batting_depth': sum(1 for a in avgs if a >= 30)
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
        'opp_bowling_depth': len(stats)
    }

def get_batsman_avg(player_name, player_db):
    """Get batting average for a specific player"""
    if player_name in player_db and 'batting' in player_db[player_name]:
        return player_db[player_name]['batting'].get('average', 35.0)
    return 35.0

# ==============================================================================
# STEP 3: PARSE ALL MATCHES AND BUILD DATASET
# ==============================================================================

print("\n[2/6] Parsing ODI ball-by-ball data...")

ballbyball_dir = '../../raw_data/odis_ballbyBall'
all_files = [os.path.join(ballbyball_dir, f) for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"   Found {len(all_files):,} match files")
print(f"   Processing all matches...")

all_samples = []
venue_scores = defaultdict(list)  # For calculating venue averages
match_id = 1

# First pass: collect venue averages
print(f"\n   [Pass 1/2] Collecting venue statistics...")
for file in all_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        if 'innings' not in match or len(match['innings']) == 0:
            continue
        
        innings = match['innings'][0]
        info = match['info']
        venue = info.get('venue', 'Unknown')
        
        # Calculate final score
        final_score = sum(
            delivery.get('runs', {}).get('total', 0)
            for over in innings.get('overs', [])
            for delivery in over.get('deliveries', [])
        )
        
        if final_score > 0:
            venue_scores[venue].append(final_score)
    except:
        continue

# Calculate venue averages
venue_avg_map = {}
global_avg = np.mean([score for scores in venue_scores.values() for score in scores]) if venue_scores else 250.0

for venue, scores in venue_scores.items():
    if len(scores) >= 10:  # Only use venues with 10+ matches
        venue_avg_map[venue] = np.mean(scores)
    else:
        venue_avg_map[venue] = global_avg

print(f"   Calculated averages for {len(venue_avg_map)} venues (global avg: {global_avg:.1f})")

# Second pass: build dataset
print(f"\n   [Pass 2/2] Building dataset with full features...")
processed_count = 0

for file in all_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        if 'innings' not in match or len(match['innings']) == 0:
            continue
        
        innings = match['innings'][0]
        batting_team = innings.get('team', 'Unknown')
        
        # Get match info
        info = match['info']
        venue = info.get('venue', 'Unknown')
        city = info.get('city', venue.split(',')[0] if ',' in venue else venue)
        match_date = info.get('dates', ['Unknown'])[0] if 'dates' in info else 'Unknown'
        
        # Get both teams' players
        teams = list(info.get('players', {}).keys())
        if len(teams) != 2:
            continue
        
        batting_team_players = info['players'].get(batting_team, [])
        bowling_team = [t for t in teams if t != batting_team][0]
        bowling_team_players = info['players'].get(bowling_team, [])
        
        # Calculate team aggregates
        batting_agg = calculate_batting_aggregates(batting_team_players, player_database)
        bowling_agg = calculate_bowling_aggregates(bowling_team_players, player_database)
        
        # Venue average
        venue_avg_score = venue_avg_map.get(venue, global_avg)
        
        # Calculate final score
        final_score = 0
        for over in innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                final_score += delivery.get('runs', {}).get('total', 0)
        
        if final_score == 0:
            continue
        
        # Process ball-by-ball with current batsmen tracking
        cumulative_runs = 0
        cumulative_wickets = 0
        ball_number = 0
        recent_runs = []
        
        # Track batsmen at crease
        batsmen_at_crease = []
        batsman_scores = {}
        
        for over_obj in innings.get('overs', []):
            for delivery in over_obj.get('deliveries', []):
                ball_number += 1
                runs = delivery.get('runs', {}).get('total', 0)
                cumulative_runs += runs
                recent_runs.append(runs)
                
                # Update current batsmen
                batsman = delivery.get('batter', None)
                non_striker = delivery.get('non_striker', None)
                
                if batsman and batsman not in batsman_scores:
                    batsman_scores[batsman] = 0
                if non_striker and non_striker not in batsman_scores:
                    batsman_scores[non_striker] = 0
                
                if batsman:
                    batsman_scores[batsman] += runs
                
                batsmen_at_crease = [batsman, non_striker] if batsman and non_striker else []
                
                # Wickets
                if 'wickets' in delivery:
                    cumulative_wickets += len(delivery['wickets'])
                
                # Sample at checkpoints: ball 1, 60, 120, 180, 240
                if ball_number in [1, 60, 120, 180, 240]:
                    last_10_overs = sum(recent_runs[-60:]) if len(recent_runs) >= 60 else sum(recent_runs)
                    crr = (cumulative_runs * 6.0 / ball_number) if ball_number > 0 else 0
                    
                    # Current batsmen features (only for mid-match)
                    if ball_number > 1 and len(batsmen_at_crease) == 2:
                        batsman_1_avg = get_batsman_avg(batsmen_at_crease[0], player_database)
                        batsman_2_avg = get_batsman_avg(batsmen_at_crease[1], player_database)
                    else:
                        batsman_1_avg = 0  # Pre-match or no batsmen info
                        batsman_2_avg = 0
                    
                    all_samples.append({
                        'match_id': match_id,
                        'match_date': match_date,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'venue': venue,
                        'ball_number': ball_number,
                        # Match State (6)
                        'current_score': cumulative_runs,
                        'wickets_fallen': cumulative_wickets,
                        'balls_bowled': ball_number,
                        'balls_remaining': 300 - ball_number,
                        'runs_last_10_overs': last_10_overs,
                        'current_run_rate': crr,
                        # Batting Team (3)
                        'team_batting_avg': batting_agg['team_batting_avg'],
                        'team_elite_batsmen': batting_agg['team_elite_batsmen'],
                        'team_batting_depth': batting_agg['team_batting_depth'],
                        # Opposition (3)
                        'opp_bowling_economy': bowling_agg['opp_bowling_economy'],
                        'opp_elite_bowlers': bowling_agg['opp_elite_bowlers'],
                        'opp_bowling_depth': bowling_agg['opp_bowling_depth'],
                        # Venue (2)
                        'venue_avg_score': venue_avg_score,
                        # Current Batsmen (2)
                        'batsman_1_avg': batsman_1_avg,
                        'batsman_2_avg': batsman_2_avg,
                        # Target
                        'final_score': final_score
                    })
        
        match_id += 1
        processed_count += 1
        
        if processed_count % 500 == 0:
            print(f"   Processed {processed_count} matches, {len(all_samples):,} samples...")
    
    except Exception as e:
        continue

df = pd.DataFrame(all_samples)

print(f"\n   CREATED: {len(df):,} samples from {processed_count:,} matches")

# ==============================================================================
# STEP 4: CLEAN AND FILTER
# ==============================================================================

print("\n[3/6] Cleaning dataset...")

# Remove nulls
df = df.dropna()
print(f"   After removing nulls: {len(df):,} samples")

# ==============================================================================
# STEP 5: TRAIN/TEST SPLIT (90/10)
# ==============================================================================

print("\n[4/6] Creating 90/10 train/test split...")

from sklearn.model_selection import train_test_split

# Split by match_id to keep all checkpoints together
match_ids = df['match_id'].unique()
train_matches, test_matches = train_test_split(match_ids, test_size=0.10, random_state=42)

train_df = df[df['match_id'].isin(train_matches)]
test_df = df[df['match_id'].isin(test_matches)]

print(f"   Training: {len(train_df):,} samples from {len(train_matches):,} matches")
print(f"   Testing: {len(test_df):,} samples from {len(test_matches):,} matches")

# ==============================================================================
# STEP 6: SAVE DATASETS
# ==============================================================================

print("\n[5/6] Saving datasets...")

os.makedirs('../data', exist_ok=True)

df.to_csv('../data/progressive_full_features_dataset.csv', index=False)
train_df.to_csv('../data/progressive_full_train.csv', index=False)
test_df.to_csv('../data/progressive_full_test.csv', index=False)

print(f"   [SAVED] ../data/progressive_full_features_dataset.csv ({len(df):,} rows)")
print(f"   [SAVED] ../data/progressive_full_train.csv ({len(train_df):,} rows)")
print(f"   [SAVED] ../data/progressive_full_test.csv ({len(test_df):,} rows)")

# ==============================================================================
# STEP 7: SUMMARY
# ==============================================================================

print("\n[6/6] Dataset Summary...")

feature_cols = [
    'current_score', 'wickets_fallen', 'balls_bowled', 'balls_remaining',
    'runs_last_10_overs', 'current_run_rate',
    'team_batting_avg', 'team_elite_batsmen', 'team_batting_depth',
    'opp_bowling_economy', 'opp_elite_bowlers', 'opp_bowling_depth',
    'venue_avg_score', 'batsman_1_avg', 'batsman_2_avg'
]

print(f"\n   Features (15 total):")
for i, feat in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n   Additional columns: venue (categorical for encoding)")
print(f"   Target variable: final_score")

print(f"\n   Checkpoint distribution:")
for ball in [1, 60, 120, 180, 240]:
    count = len(df[df['ball_number'] == ball])
    print(f"   - Ball {ball:>3} (Over {ball//6:>2}): {count:>5} samples")

# Save summary
with open('../data/feature_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PROGRESSIVE ODI DATASET - FULL FEATURES\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total samples: {len(df):,}\n")
    f.write(f"Training: {len(train_df):,} ({len(train_matches):,} matches)\n")
    f.write(f"Testing: {len(test_df):,} ({len(test_matches):,} matches)\n\n")
    f.write(f"Features (15 numeric + 1 categorical):\n")
    for i, feat in enumerate(feature_cols, 1):
        f.write(f"  {i:2d}. {feat}\n")
    f.write(f"  16. venue (categorical)\n\n")
    f.write(f"Target: final_score\n")

print(f"   [SAVED] ../data/feature_summary.txt")

print(f"\n{'='*80}")
print("DATASET BUILD COMPLETE!")
print(f"{'='*80}\n")

