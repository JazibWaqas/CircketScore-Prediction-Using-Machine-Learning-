#!/usr/bin/env python3
"""
VALIDATE ON REAL INTERNATIONAL ODI MATCHES ONLY

Tests model on actual international ODI matches (not domestic cricket).
Target: 500-1000 test cases from real international matches.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from collections import defaultdict

print("\n" + "="*80)
print("VALIDATE ON REAL INTERNATIONAL ODI MATCHES")
print("="*80)

# ==============================================================================
# STEP 1: LOAD MODEL
# ==============================================================================

print("\n[1/5] Loading trained model...")

with open('../models/progressive_model_full_features.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/feature_names.json', 'r') as f:
    feature_info = json.load(f)

print(f"   Model loaded successfully")
print(f"   Features: {len(feature_info['all_features'])}")

# ==============================================================================
# STEP 2: LOAD PLAYER DATABASE
# ==============================================================================

print("\n[2/5] Loading player database...")

player_db_path = '../../ODI/data/CURRENT_player_database_977_quality.json'
with open(player_db_path, 'r') as f:
    player_database = json.load(f)

print(f"   Loaded {len(player_database):,} players")

# ==============================================================================
# STEP 3: HELPER FUNCTIONS (same as dataset building)
# ==============================================================================

def calculate_batting_aggregates(players, player_db):
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
    if player_name in player_db and 'batting' in player_db[player_name]:
        return player_db[player_name]['batting'].get('average', 35.0)
    return 35.0

# ==============================================================================
# STEP 4: FIND AND PROCESS INTERNATIONAL MATCHES ONLY
# ==============================================================================

print("\n[3/5] Finding INTERNATIONAL ODI matches...")

# List of international teams
INTERNATIONAL_TEAMS = [
    'India', 'Australia', 'England', 'Pakistan', 'South Africa',
    'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh',
    'Afghanistan', 'Zimbabwe', 'Ireland', 'Scotland', 'Netherlands',
    'Namibia', 'Nepal', 'UAE', 'Oman', 'USA', 'Canada', 'Kenya'
]

ballbyball_dir = '../../raw_data/odis_ballbyBall'
all_files = [os.path.join(ballbyball_dir, f) for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

# Filter for international matches
international_matches = []
for file in all_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        info = match.get('info', {})
        teams = list(info.get('players', {}).keys())
        
        # Both teams must be international teams
        if len(teams) == 2 and all(t in INTERNATIONAL_TEAMS for t in teams):
            international_matches.append(file)
    except:
        continue

print(f"   Found {len(international_matches):,} international ODI matches")
print(f"   Testing on all international matches...")

# Build venue averages (needed for prediction)
venue_scores = defaultdict(list)
for file in all_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        if 'innings' not in match or len(match['innings']) == 0:
            continue
        innings = match['innings'][0]
        venue = match['info'].get('venue', 'Unknown')
        final_score = sum(
            delivery.get('runs', {}).get('total', 0)
            for over in innings.get('overs', [])
            for delivery in over.get('deliveries', [])
        )
        if final_score > 0:
            venue_scores[venue].append(final_score)
    except:
        continue

venue_avg_map = {}
global_avg = np.mean([score for scores in venue_scores.values() for score in scores]) if venue_scores else 250.0
for venue, scores in venue_scores.items():
    venue_avg_map[venue] = np.mean(scores) if len(scores) >= 10 else global_avg

# ==============================================================================
# STEP 5: TEST ON INTERNATIONAL MATCHES
# ==============================================================================

print("\n[4/5] Testing model on international matches...")

validation_results = []
match_count = 0

for file in international_matches:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        if 'innings' not in match or len(match['innings']) == 0:
            continue
        
        innings = match['innings'][0]
        batting_team = innings.get('team', 'Unknown')
        
        info = match['info']
        venue = info.get('venue', 'Unknown')
        match_date = info.get('dates', ['Unknown'])[0] if 'dates' in info else 'Unknown'
        
        teams = list(info.get('players', {}).keys())
        if len(teams) != 2:
            continue
        
        batting_team_players = info['players'].get(batting_team, [])
        bowling_team = [t for t in teams if t != batting_team][0]
        bowling_team_players = info['players'].get(bowling_team, [])
        
        # Calculate aggregates
        batting_agg = calculate_batting_aggregates(batting_team_players, player_database)
        bowling_agg = calculate_bowling_aggregates(bowling_team_players, player_database)
        venue_avg_score = venue_avg_map.get(venue, global_avg)
        
        # Calculate final score
        final_score = sum(
            delivery.get('runs', {}).get('total', 0)
            for over in innings.get('overs', [])
            for delivery in over.get('deliveries', [])
        )
        
        if final_score < 100:
            continue
        
        # Process ball-by-ball
        cumulative_runs = 0
        cumulative_wickets = 0
        ball_number = 0
        recent_runs = []
        batsmen_at_crease = []
        batsman_scores = {}
        
        for over_obj in innings.get('overs', []):
            for delivery in over_obj.get('deliveries', []):
                ball_number += 1
                runs = delivery.get('runs', {}).get('total', 0)
                cumulative_runs += runs
                recent_runs.append(runs)
                
                # Track batsmen
                batsman = delivery.get('batter', None)
                non_striker = delivery.get('non_striker', None)
                
                if batsman and batsman not in batsman_scores:
                    batsman_scores[batsman] = 0
                if non_striker and non_striker not in batsman_scores:
                    batsman_scores[non_striker] = 0
                
                if batsman:
                    batsman_scores[batsman] += runs
                
                batsmen_at_crease = [batsman, non_striker] if batsman and non_striker else []
                
                if 'wickets' in delivery:
                    cumulative_wickets += len(delivery['wickets'])
                
                # Test at checkpoints
                if ball_number in [1, 60, 120, 180, 240]:
                    last_10_overs = sum(recent_runs[-60:]) if len(recent_runs) >= 60 else sum(recent_runs)
                    crr = (cumulative_runs * 6.0 / ball_number) if ball_number > 0 else 0
                    
                    if ball_number > 1 and len(batsmen_at_crease) == 2:
                        batsman_1_avg = get_batsman_avg(batsmen_at_crease[0], player_database)
                        batsman_2_avg = get_batsman_avg(batsmen_at_crease[1], player_database)
                    else:
                        batsman_1_avg = 0
                        batsman_2_avg = 0
                    
                    # Create prediction input
                    test_input = pd.DataFrame([{
                        'current_score': cumulative_runs,
                        'wickets_fallen': cumulative_wickets,
                        'balls_bowled': ball_number,
                        'balls_remaining': 300 - ball_number,
                        'runs_last_10_overs': last_10_overs,
                        'current_run_rate': crr,
                        'team_batting_avg': batting_agg['team_batting_avg'],
                        'team_elite_batsmen': batting_agg['team_elite_batsmen'],
                        'team_batting_depth': batting_agg['team_batting_depth'],
                        'opp_bowling_economy': bowling_agg['opp_bowling_economy'],
                        'opp_elite_bowlers': bowling_agg['opp_elite_bowlers'],
                        'opp_bowling_depth': bowling_agg['opp_bowling_depth'],
                        'venue_avg_score': venue_avg_score,
                        'batsman_1_avg': batsman_1_avg,
                        'batsman_2_avg': batsman_2_avg,
                        'venue': venue
                    }])
                    
                    try:
                        prediction = model.predict(test_input)[0]
                        
                        validation_results.append({
                            'match_id': match_count,
                            'date': match_date,
                            'batting_team': batting_team,
                            'vs': bowling_team,
                            'ball': ball_number,
                            'current_score': cumulative_runs,
                            'wickets': cumulative_wickets,
                            'predicted': prediction,
                            'actual': final_score,
                            'error': prediction - final_score,
                            'abs_error': abs(prediction - final_score)
                        })
                    except Exception as e:
                        pass  # Skip if venue unknown
        
        match_count += 1
        
        if match_count % 100 == 0:
            print(f"   Processed {match_count} matches, {len(validation_results)} predictions...")
    
    except:
        continue

print(f"\n   Total: {match_count:,} international matches")
print(f"   Total predictions: {len(validation_results):,}")

# ==============================================================================
# STEP 6: ANALYZE RESULTS
# ==============================================================================

print("\n[5/5] Analyzing results...")

if len(validation_results) == 0:
    print("\n[ERROR] No predictions made!")
    exit()

results_df = pd.DataFrame(validation_results)

# Overall performance
overall_r2 = r2_score(results_df['actual'], results_df['predicted'])
overall_mae = mean_absolute_error(results_df['actual'], results_df['predicted'])
mean_pct_error = (results_df['abs_error'] / results_df['actual'] * 100).mean()

print(f"\n{'='*80}")
print("VALIDATION RESULTS - INTERNATIONAL ODIs ONLY")
print(f"{'='*80}")

print(f"\nOVERALL PERFORMANCE:")
print(f"   R2 Score: {overall_r2:.4f} ({overall_r2*100:.2f}%)")
print(f"   MAE: {overall_mae:.2f} runs")
print(f"   Mean % Error: {mean_pct_error:.2f}%")

# By stage
print(f"\nPERFORMANCE BY MATCH STAGE:")
stages = [
    ("Pre-match (ball 1)", 1, 1),
    ("Early (ball 60, over 10)", 60, 60),
    ("Mid (ball 120, over 20)", 120, 120),
    ("Late (ball 180, over 30)", 180, 180),
    ("Death (ball 240, over 40)", 240, 240)
]

for stage_name, min_ball, max_ball in stages:
    stage_data = results_df[(results_df['ball'] >= min_ball) & (results_df['ball'] <= max_ball)]
    if len(stage_data) > 0:
        stage_r2 = r2_score(stage_data['actual'], stage_data['predicted'])
        stage_mae = mean_absolute_error(stage_data['actual'], stage_data['predicted'])
        print(f"   {stage_name:<30s} R2 = {stage_r2:.4f}, MAE = {stage_mae:.2f} (n={len(stage_data)})")

# Accuracy
within_10 = (results_df['abs_error'] <= 10).sum()
within_20 = (results_df['abs_error'] <= 20).sum()
within_30 = (results_df['abs_error'] <= 30).sum()

print(f"\nACCURACY:")
print(f"   Within +/-10 runs: {100*within_10/len(results_df):.1f}%")
print(f"   Within +/-20 runs: {100*within_20/len(results_df):.1f}%")
print(f"   Within +/-30 runs: {100*within_30/len(results_df):.1f}%")

# Sample predictions
print(f"\nSAMPLE PREDICTIONS (15 random):")
sample = results_df.sample(min(15, len(results_df)))
print(f"\n{'Team':<15} {'vs':<15} {'Ball':>5} {'Score':>8} {'Pred':>7} {'Actual':>8} {'Error':>7}")
print("-" * 80)
for _, row in sample.iterrows():
    print(f"{row['batting_team'][:13]:<15} {row['vs'][:13]:<15} {row['ball']:>5} "
          f"{row['current_score']:>4.0f}/{row['wickets']:<2.0f} "
          f"{row['predicted']:>7.0f} {row['actual']:>8.0f} {row['error']:>+7.0f}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

os.makedirs('../results', exist_ok=True)

results_df.to_csv('../results/international_validation_results.csv', index=False)
print(f"\n[SAVED] ../results/international_validation_results.csv")

# Save summary
with open('../results/international_validation_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("INTERNATIONAL ODI VALIDATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Test Set:\n")
    f.write(f"  Matches: {match_count:,} international ODIs\n")
    f.write(f"  Predictions: {len(results_df):,}\n\n")
    f.write(f"Overall Performance:\n")
    f.write(f"  R2 Score: {overall_r2:.4f} ({overall_r2*100:.2f}%)\n")
    f.write(f"  MAE: {overall_mae:.2f} runs\n")
    f.write(f"  Mean % Error: {mean_pct_error:.2f}%\n\n")
    f.write(f"Accuracy:\n")
    f.write(f"  Within +/-10 runs: {100*within_10/len(results_df):.1f}%\n")
    f.write(f"  Within +/-20 runs: {100*within_20/len(results_df):.1f}%\n")
    f.write(f"  Within +/-30 runs: {100*within_30/len(results_df):.1f}%\n")

print(f"[SAVED] ../results/international_validation_summary.txt")

# ==============================================================================
# ASSESSMENT
# ==============================================================================

print(f"\n{'='*80}")
print("ASSESSMENT")
print(f"{'='*80}")

print(f"\nTested on {len(validation_results):,} predictions from {match_count:,} REAL international ODI matches")

if overall_r2 >= 0.75:
    print(f"\n[SUCCESS] R2 = {overall_r2:.3f} - Meets target (0.75+)")
elif overall_r2 >= 0.65:
    print(f"\n[GOOD] R2 = {overall_r2:.3f} - Close to target")
else:
    print(f"\n[NEEDS WORK] R2 = {overall_r2:.3f} - Below target (0.75)")

print(f"\nProgressive improvement demonstrated:")
pre_r2 = r2_score(results_df[results_df['ball']==1]['actual'], results_df[results_df['ball']==1]['predicted']) if len(results_df[results_df['ball']==1]) > 0 else 0
late_r2 = r2_score(results_df[results_df['ball']==240]['actual'], results_df[results_df['ball']==240]['predicted']) if len(results_df[results_df['ball']==240]) > 0 else 0
print(f"  Pre-match: R2 = {pre_r2:.3f}")
print(f"  Late-match: R2 = {late_r2:.3f}")
print(f"  Improvement: {(late_r2-pre_r2)/pre_r2*100 if pre_r2 > 0 else 0:.1f}%")

print(f"\n{'='*80}\n")

