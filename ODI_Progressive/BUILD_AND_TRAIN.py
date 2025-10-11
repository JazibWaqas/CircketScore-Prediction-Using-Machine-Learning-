#!/usr/bin/env python3
"""
COMPLETE ODI PROGRESSIVE PREDICTOR - SIMPLIFIED VERSION
Based on working Cricket-Score-Predictor approach, adapted for our use case

Our Innovation:
1. Progressive prediction (works from ball 0 to 300)
2. Team composition features (player aggregates)
3. Fantasy team building capability
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Optional progress bar
try:
    from tqdm import tqdm
except:
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable

print("\n" + "="*80)
print("ODI PROGRESSIVE SCORE PREDICTOR - COMPLETE BUILD")
print("="*80)

# ==============================================================================
# STEP 1: PARSE BALL-BY-BALL DATA (Like Working Project)
# ==============================================================================

print("\n[1/6] Parsing ODI ball-by-ball data...")

ballbyball_dir = '../raw_data/odis_ballbyBall'
filenames = [os.path.join(ballbyball_dir, f) for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"   Found {len(filenames):,} match files")
print(f"   Processing ALL matches (will take 3-5 minutes)...")

all_balls = []
match_id = 1

# Load player database for team aggregates
player_db = json.load(open('../ODI/data/CURRENT_player_database_977_quality.json'))
print(f"   Loaded {len(player_db):,} players")

for file in tqdm(filenames, desc="   Parsing"):  # ALL matches for proper training
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
        
        # Get both teams' players
        teams = list(info.get('players', {}).keys())
        if len(teams) != 2:
            continue
        
        batting_team_players = info['players'].get(batting_team, [])
        bowling_team = [t for t in teams if t != batting_team][0]
        
        # Calculate team batting average (OUR INNOVATION)
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
                
                # Sample at multiple intervals for better coverage
                # Every 20 balls + key points = ~15 samples per match
                if ball_number in [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]:
                    # Last 10 overs = last 60 balls
                    last_10_overs = sum(recent_runs[-60:]) if len(recent_runs) >= 60 else sum(recent_runs)
                    
                    all_balls.append({
                        'match_id': match_id,
                        'batting_team': batting_team,
                        'city': city,
                        'current_score': cumulative_runs,
                        'balls_left': 300 - ball_number,
                        'wickets_left': 10 - cumulative_wickets,
                        'crr': (cumulative_runs * 6.0 / ball_number) if ball_number > 0 else 0,
                        'last_10_overs': last_10_overs,
                        'team_batting_avg': team_batting_avg,  # OUR ADDITION
                        'final_score': final_score
                    })
        
        match_id += 1
        
    except Exception as e:
        continue

df = pd.DataFrame(all_balls)
print(f"\n   ✓ Parsed {match_id-1:,} matches")
print(f"   ✓ Created {len(df):,} training samples")

# ==============================================================================
# STEP 2: CLEAN DATA (Like Working Project)
# ==============================================================================

print("\n[2/6] Cleaning data...")

# Remove nulls
df = df.dropna()

# Filter to cities with enough data
eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 200].index.tolist()
df = df[df['city'].isin(eligible_cities)]

print(f"   ✓ Dataset: {df.shape}")
print(f"   ✓ Cities: {len(eligible_cities)}")

# ==============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ==============================================================================

print("\n[3/6] Splitting data...")

X = df.drop(columns=['final_score', 'match_id'])
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# ==============================================================================
# STEP 4: BUILD PIPELINE (Like Working Project)
# ==============================================================================

print("\n[4/6] Building pipeline...")

trf = ColumnTransformer([
    ('encoder', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(
        n_estimators=800,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        tree_method='hist'
    ))
])

print(f"   ✓ Pipeline created")

# ==============================================================================
# STEP 5: TRAIN MODEL
# ==============================================================================

print("\n[5/6] Training model...")
print(f"   (This may take 2-5 minutes with {len(X_train):,} samples)")

pipe.fit(X_train, y_train)

print(f"   ✓ Training complete!")

# ==============================================================================
# STEP 6: EVALUATE
# ==============================================================================

print("\n[6/6] Evaluating...")

y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n" + "="*80)
print(f"RESULTS")
print(f"="*80)
print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"   MAE: {mae:.2f} runs")

# By stage
print(f"\n📊 PERFORMANCE BY MATCH STAGE:")

test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['predicted'] = y_pred

for stage, balls_range in [
    ("Pre-match (0-10 overs)", (250, 300)),
    ("Early (10-20 overs)", (180, 240)),
    ("Middle (20-30 overs)", (120, 180)),
    ("Late (30-40 overs)", (60, 120)),
    ("Death (40+ overs)", (0, 60))
]:
    mask = (test_df['balls_left'] >= balls_range[0]) & (test_df['balls_left'] < balls_range[1])
    if mask.sum() > 0:
        stage_r2 = r2_score(test_df[mask]['actual'], test_df[mask]['predicted'])
        stage_mae = mean_absolute_error(test_df[mask]['actual'], test_df[mask]['predicted'])
        print(f"   {stage:<25s} R² = {stage_r2:.4f}, MAE = {stage_mae:.2f} runs")

# Accuracy bands
within_10 = (np.abs(y_pred - y_test) <= 10).sum()
within_20 = (np.abs(y_pred - y_test) <= 20).sum()
within_30 = (np.abs(y_pred - y_test) <= 30).sum()

print(f"\n📊 ACCURACY:")
print(f"   Within ±10 runs: {within_10}/{len(y_test)} ({100*within_10/len(y_test):.1f}%)")
print(f"   Within ±20 runs: {within_20}/{len(y_test)} ({100*within_20/len(y_test):.1f}%)")
print(f"   Within ±30 runs: {within_30}/{len(y_test)} ({100*within_30/len(y_test):.1f}%)")

# Sample predictions
print(f"\n📋 SAMPLE PREDICTIONS:")
sample = test_df.sample(min(10, len(test_df)))
print(f"\n{'Team':<15} {'Current':>7} {'Balls Left':>11} {'Actual':>8} {'Predicted':>10} {'Error':>7}")
print("-" * 75)
for _, row in sample.iterrows():
    print(f"{row['batting_team'][:13]:<15} {row['current_score']:>7.0f} {row['balls_left']:>11.0f} "
          f"{row['actual']:>8.0f} {row['predicted']:>10.0f} {row['predicted']-row['actual']:>+7.0f}")

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print(f"\n💾 Saving model...")

os.makedirs('models', exist_ok=True)

with open('models/odi_progressive_pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print(f"   ✓ Saved to models/odi_progressive_pipe.pkl")

# ==============================================================================
# ASSESSMENT
# ==============================================================================

print(f"\n" + "="*80)
print("ASSESSMENT")
print(f"="*80)

if r2 >= 0.90:
    print(f"\n✅ EXCELLENT! R² = {r2:.3f}")
    print(f"   Model is working very well!")
    print(f"   Ready for production use")
elif r2 >= 0.75:
    print(f"\n✓ GOOD! R² = {r2:.3f}")
    print(f"   Model is functional and useful")
    print(f"   Suitable for course project")
elif r2 >= 0.60:
    print(f"\n⚠ ACCEPTABLE. R² = {r2:.3f}")
    print(f"   Model works but has room for improvement")
else:
    print(f"\n❌ POOR. R² = {r2:.3f}")
    print(f"   Model needs significant improvement")

print(f"\n💡 NEXT STEPS:")
print(f"   1. Test player what-if scenarios")
print(f"   2. Create simple prediction function")
print(f"   3. Build frontend interface")

print(f"\n" + "="*80 + "\n")

# ==============================================================================
# SAVE RESULTS TO FILE
# ==============================================================================

os.makedirs('results', exist_ok=True)

with open('results/training_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ODI PROGRESSIVE PREDICTOR - TRAINING RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset:\n")
    f.write(f"  Total samples: {len(df):,}\n")
    f.write(f"  Training: {len(X_train):,}\n")
    f.write(f"  Test: {len(X_test):,}\n\n")
    
    f.write(f"Overall Performance:\n")
    f.write(f"  R² Score: {r2:.4f} ({r2*100:.2f}%)\n")
    f.write(f"  MAE: {mae:.2f} runs\n\n")
    
    f.write(f"Performance by Stage:\n")
    for stage, balls_range in [
        ("Pre-match (0-10 overs)", (250, 300)),
        ("Early (10-20 overs)", (180, 240)),
        ("Middle (20-30 overs)", (120, 180)),
        ("Late (30-40 overs)", (60, 120)),
        ("Death (40+ overs)", (0, 60))
    ]:
        mask = (test_df['balls_left'] >= balls_range[0]) & (test_df['balls_left'] < balls_range[1])
        if mask.sum() > 0:
            stage_r2 = r2_score(test_df[mask]['actual'], test_df[mask]['predicted'])
            stage_mae = mean_absolute_error(test_df[mask]['actual'], test_df[mask]['predicted'])
            f.write(f"  {stage:<25s} R² = {stage_r2:.4f}, MAE = {stage_mae:.2f} runs\n")
    
    f.write(f"\nAccuracy:\n")
    f.write(f"  Within ±10 runs: {100*within_10/len(y_test):.1f}%\n")
    f.write(f"  Within ±20 runs: {100*within_20/len(y_test):.1f}%\n")
    f.write(f"  Within ±30 runs: {100*within_30/len(y_test):.1f}%\n")

print(f"✓ Results saved to ODI_Progressive/results/training_results.txt")

