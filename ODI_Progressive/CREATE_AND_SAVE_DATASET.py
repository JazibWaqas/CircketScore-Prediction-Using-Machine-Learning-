#!/usr/bin/env python3
"""
CREATE PROGRESSIVE DATASET AND SAVE IT

This will:
1. Parse ball-by-ball ODI data
2. Sample at multiple checkpoints per match (progressive)
3. SAVE the dataset to CSV so we can inspect it
4. Show statistics about what was created
"""

import numpy as np
import pandas as pd
import json
import os

# Optional progress bar
try:
    from tqdm import tqdm
except:
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable

print("\n" + "="*80)
print("CREATE PROGRESSIVE ODI DATASET")
print("="*80)

# ==============================================================================
# PARSE BALL-BY-BALL DATA
# ==============================================================================

print("\n[1/3] Parsing ball-by-ball ODI data...")

ballbyball_dir = '../raw_data/odis_ballbyBall'
filenames = [os.path.join(ballbyball_dir, f) for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"   Found {len(filenames):,} match files")

# Load player database
player_db = json.load(open('../ODI/data/CURRENT_player_database_977_quality.json'))
print(f"   Loaded {len(player_db):,} players")

all_samples = []
match_id = 1

for file in tqdm(filenames, desc="   Processing matches"):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        if 'innings' not in match or len(match['innings']) == 0:
            continue
        
        # First innings only
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
        
        # Calculate team batting average (OUR INNOVATION FOR FANTASY)
        batting_avgs = []
        for player in batting_team_players:
            if player in player_db and 'batting' in player_db[player]:
                batting_avgs.append(player_db[player]['batting'].get('average', 0))
        team_batting_avg = np.mean(batting_avgs) if batting_avgs else 35.0
        
        # Calculate final score
        final_score = 0
        for over in innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                final_score += delivery.get('runs', {}).get('total', 0)
        
        if final_score == 0:
            continue
        
        # Process ball-by-ball
        cumulative_runs = 0
        cumulative_wickets = 0
        ball_number = 0
        recent_runs = []
        
        for over_obj in innings.get('overs', []):
            for delivery in over_obj.get('deliveries', []):
                ball_number += 1
                runs = delivery.get('runs', {}).get('total', 0)
                cumulative_runs += runs
                recent_runs.append(runs)
                
                # Wickets
                if 'wickets' in delivery:
                    cumulative_wickets += len(delivery['wickets'])
                
                # PROGRESSIVE SAMPLING: Sample at multiple checkpoints
                # Checkpoints: balls 1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280
                if ball_number in [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]:
                    # Last 10 overs = last 60 balls
                    last_10_overs = sum(recent_runs[-60:]) if len(recent_runs) >= 60 else sum(recent_runs)
                    
                    all_samples.append({
                        'match_id': match_id,
                        'match_date': match_date,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'city': city,
                        'venue': venue,
                        'ball_number': ball_number,
                        'over': ball_number // 6,
                        'current_score': cumulative_runs,
                        'balls_left': 300 - ball_number,
                        'wickets_left': 10 - cumulative_wickets,
                        'crr': (cumulative_runs * 6.0 / ball_number) if ball_number > 0 else 0,
                        'last_10_overs': last_10_overs,
                        'team_batting_avg': team_batting_avg,  # OUR FANTASY FEATURE
                        'final_score': final_score
                    })
        
        match_id += 1
        
    except Exception as e:
        continue

df = pd.DataFrame(all_samples)

print(f"\n   [CREATED] Parsed {match_id-1:,} matches")
print(f"   [CREATED] {len(df):,} training samples (progressive checkpoints)")

# ==============================================================================
# CLEAN AND SHOW STATISTICS
# ==============================================================================

print("\n[2/3] Analyzing dataset...")

# Remove nulls
df = df.dropna()
print(f"   After removing nulls: {len(df):,} samples")

# Filter to cities with enough data
eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 200].index.tolist()
df_filtered = df[df['city'].isin(eligible_cities)]

print(f"\n   Dataset Statistics:")
print(f"   - Total samples: {len(df_filtered):,}")
print(f"   - Unique matches: {df_filtered['match_id'].nunique():,}")
print(f"   - Unique teams: {df_filtered['batting_team'].nunique()}")
print(f"   - Unique cities: {len(eligible_cities)}")
print(f"   - Date range: {df_filtered['match_date'].min()} to {df_filtered['match_date'].max()}")

# Show sampling distribution
print(f"\n   Samples by checkpoint:")
checkpoint_counts = df_filtered['ball_number'].value_counts().sort_index()
for ball, count in checkpoint_counts.items():
    over = ball // 6
    print(f"   - Ball {ball:>3} (Over {over:>2}): {count:>5} samples")

# Show some example rows
print(f"\n   Example samples (first 5):")
print(df_filtered.head()[['match_id', 'batting_team', 'over', 'current_score', 'balls_left', 'final_score']])

# Show feature columns
print(f"\n   Feature columns:")
for i, col in enumerate(df_filtered.columns, 1):
    print(f"   {i:>2}. {col}")

# ==============================================================================
# SAVE DATASET
# ==============================================================================

print("\n[3/3] Saving dataset...")

os.makedirs('data', exist_ok=True)

# Save full dataset
df_filtered.to_csv('data/progressive_dataset_full.csv', index=False)
print(f"   [SAVED] data/progressive_dataset_full.csv ({len(df_filtered):,} rows)")

# Save train/test split (80/20) - RANDOM SPLIT
from sklearn.model_selection import train_test_split

# Group by match_id to ensure all checkpoints from same match stay together
match_ids = df_filtered['match_id'].unique()
train_matches, test_matches = train_test_split(match_ids, test_size=0.2, random_state=42)

train_df = df_filtered[df_filtered['match_id'].isin(train_matches)]
test_df = df_filtered[df_filtered['match_id'].isin(test_matches)]

train_df.to_csv('data/progressive_train.csv', index=False)
test_df.to_csv('data/progressive_test.csv', index=False)

print(f"   [SAVED] data/progressive_train.csv ({len(train_df):,} rows, {len(train_matches):,} matches)")
print(f"   [SAVED] data/progressive_test.csv ({len(test_df):,} rows, {len(test_matches):,} matches)")

# Save summary statistics
with open('data/dataset_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PROGRESSIVE ODI DATASET SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Created: {pd.Timestamp.now()}\n\n")
    
    f.write(f"Dataset Size:\n")
    f.write(f"  Total samples: {len(df_filtered):,}\n")
    f.write(f"  Training: {len(train_df):,} ({len(train_matches):,} matches)\n")
    f.write(f"  Testing: {len(test_df):,} ({len(test_matches):,} matches)\n\n")
    
    f.write(f"Match Coverage:\n")
    f.write(f"  Unique matches: {df_filtered['match_id'].nunique():,}\n")
    f.write(f"  Unique teams: {df_filtered['batting_team'].nunique()}\n")
    f.write(f"  Unique cities: {len(eligible_cities)}\n")
    f.write(f"  Date range: {df_filtered['match_date'].min()} to {df_filtered['match_date'].max()}\n\n")
    
    f.write(f"Progressive Checkpoints Per Match:\n")
    for ball, count in checkpoint_counts.items():
        avg_per_match = count / df_filtered['match_id'].nunique()
        f.write(f"  Ball {ball:>3} (Over {ball//6:>2}): {avg_per_match:.1f} avg per match\n")
    
    f.write(f"\nFeatures (8 total):\n")
    for i, col in enumerate(df_filtered.columns, 1):
        f.write(f"  {i:>2}. {col}\n")
    
    f.write(f"\nTarget Variable:\n")
    f.write(f"  final_score (range: {df_filtered['final_score'].min():.0f} - {df_filtered['final_score'].max():.0f})\n")
    f.write(f"  Mean: {df_filtered['final_score'].mean():.1f}\n")
    f.write(f"  Std: {df_filtered['final_score'].std():.1f}\n")

print(f"   [SAVED] data/dataset_summary.txt")

# ==============================================================================
# COMPARISON WITH ODI FOLDER DATA
# ==============================================================================

print(f"\n{'='*80}")
print("COMPARISON: ODI_Progressive vs ODI Folder")
print(f"{'='*80}")

print(f"\nODI Folder (pre-match only):")
print(f"  - One row per match")
print(f"  - Pre-match features only")
print(f"  - ~40+ features (team stats, form, venue, etc.)")
print(f"  - For: Predicting BEFORE match starts")

print(f"\nODI_Progressive (this dataset):")
print(f"  - Multiple rows per match (15 checkpoints)")
print(f"  - Progressive features (current match state)")
print(f"  - 8 features (simpler, includes team_batting_avg)")
print(f"  - For: Predicting at ANY match stage")

print(f"\n[SUCCESS] Dataset created and saved!")
print(f"{'='*80}\n")

