#!/usr/bin/env python3
"""
VALIDATE V2 MODELS ON REAL INTERNATIONAL ODI MATCHES

Tests XGBoost v2 and Random Forest v2 on actual international ODI matches.
Target: Compare with original 0.69 R2 baseline.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from collections import defaultdict

print("\n" + "="*80)
print("VALIDATE V2 MODELS ON INTERNATIONAL MATCHES")
print("="*80)

# ==============================================================================
# STEP 1: LOAD MODELS
# ==============================================================================

print("\n[1/5] Loading v2 models...")

try:
    with open('../models/progressive_model_xgboost_v2.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print("   [OK] XGBoost v2 loaded")
except Exception as e:
    print(f"   [ERROR] XGBoost v2 failed: {e}")
    xgb_model = None

try:
    with open('../models/progressive_model_random_forest_v2.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("   [OK] Random Forest v2 loaded")
except Exception as e:
    print(f"   [ERROR] Random Forest v2 failed: {e}")
    rf_model = None

if not xgb_model and not rf_model:
    print("No models loaded!")
    exit()

# ==============================================================================
# STEP 2: LOAD PLAYER DATABASE
# ==============================================================================

print("\n[2/5] Loading player database...")

player_db_path = '../../ODI_Progressive/CURRENT_player_database_977_quality_FIXED.json'
with open(player_db_path, 'r') as f:
    player_database = json.load(f)

print(f"   Loaded {len(player_database):,} players")

# ==============================================================================
# STEP 3: HELPER FUNCTIONS
# ==============================================================================

def get_batsman_avg(player_name, player_db):
    if player_name in player_db and 'batting' in player_db[player_name]:
        batting_data = player_db[player_name]['batting']
        avg = batting_data.get('average')
        if avg is not None and avg > 0:
            return float(avg)
    
    role = player_db.get(player_name, {}).get('role', 'Batsman')
    if 'Bowler' in role: return 18.0
    elif 'All-rounder' in role: return 25.0
    else: return 30.0

def calculate_batting_aggregates(players, player_db):
    avgs = [get_batsman_avg(p, player_db) for p in players]
    return {
        'team_batting_avg': np.mean(avgs),
        'team_elite_batsmen': sum(1 for a in avgs if a >= 40),
        'team_batting_depth': sum(1 for a in avgs if a >= 30)
    }

def calculate_bowling_aggregates(players, player_db):
    economies = []
    for player in players:
        if player in player_db and 'bowling' in player_db[player]:
            economy = player_db[player]['bowling'].get('economy')
            if economy is not None and economy > 0:
                economies.append(float(economy))
            else:
                role = player_db.get(player, {}).get('role', 'Batsman')
                if 'Bowler' in role: economies.append(5.0)
                elif 'All-rounder' in role: economies.append(5.5)
                else: economies.append(6.0)
        else:
            role = player_db.get(player, {}).get('role', 'Batsman')
            if 'Bowler' in role: economies.append(5.0)
            elif 'All-rounder' in role: economies.append(5.5)
            else: economies.append(6.0)
    
    if not economies:
        return {'opp_bowling_economy': 5.5, 'opp_elite_bowlers': 0, 'opp_bowling_depth': 0}
    
    return {
        'opp_bowling_economy': np.mean(economies),
        'opp_elite_bowlers': sum(1 for e in economies if e < 4.8),
        'opp_bowling_depth': len(economies)
    }

# ==============================================================================
# STEP 4: FIND INTERNATIONAL MATCHES
# ==============================================================================

print("\n[3/5] Finding INTERNATIONAL ODI matches...")

INTERNATIONAL_TEAMS = [
    'India', 'Australia', 'England', 'Pakistan', 'South Africa',
    'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh',
    'Afghanistan', 'Zimbabwe', 'Ireland', 'Scotland', 'Netherlands',
    'Namibia', 'Nepal', 'UAE', 'Oman', 'USA', 'Canada', 'Kenya'
]

ballbyball_dir = '../../raw_data/odis_ballbyBall'
all_files = [os.path.join(ballbyball_dir, f) for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

international_matches = []
venue_scores = defaultdict(list)

for file in all_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        info = match.get('info', {})
        teams = list(info.get('players', {}).keys())
        
        # Filter international
        if len(teams) == 2 and all(t in INTERNATIONAL_TEAMS for t in teams):
            international_matches.append(file)
            
            # Collect venue scores
            if 'innings' in match and len(match['innings']) > 0:
                venue = info.get('venue', 'Unknown')
                score = sum(d.get('runs', {}).get('total', 0) 
                          for o in match['innings'][0].get('overs', []) 
                          for d in o.get('deliveries', []))
                if score > 0: venue_scores[venue].append(score)
    except:
        continue

print(f"   Found {len(international_matches):,} international matches")

venue_avg_map = {}
global_avg = np.mean([s for scores in venue_scores.values() for s in scores]) if venue_scores else 250.0
for venue, scores in venue_scores.items():
    venue_avg_map[venue] = np.mean(scores) if len(scores) >= 10 else global_avg

# ==============================================================================
# STEP 5: TEST MODELS
# ==============================================================================

print("\n[4/5] Testing models...")

results = []
match_count = 0

for file in international_matches:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        if 'innings' not in match or len(match['innings']) == 0: continue
        
        innings = match['innings'][0]
        batting_team = innings.get('team', 'Unknown')
        info = match['info']
        venue = info.get('venue', 'Unknown')
        teams = list(info.get('players', {}).keys())
        
        if len(teams) != 2: continue
        
        batting_players = info['players'].get(batting_team, [])
        bowling_team = [t for t in teams if t != batting_team][0]
        bowling_players = info['players'].get(bowling_team, [])
        
        batting_agg = calculate_batting_aggregates(batting_players, player_database)
        bowling_agg = calculate_bowling_aggregates(bowling_players, player_database)
        venue_avg = venue_avg_map.get(venue, global_avg)
        
        final_score = sum(d.get('runs', {}).get('total', 0) 
                        for o in innings.get('overs', []) 
                        for d in o.get('deliveries', []))
        
        if final_score < 100: continue
        
        curr_runs = 0
        curr_wickets = 0
        ball_num = 0
        recent_runs = []
        batsmen_scores = {}
        
        for over in innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                ball_num += 1
                runs = delivery.get('runs', {}).get('total', 0)
                curr_runs += runs
                recent_runs.append(runs)
                
                if 'wickets' in delivery: curr_wickets += len(delivery['wickets'])
                
                batter = delivery.get('batter')
                non_striker = delivery.get('non_striker')
                batsmen = [batter, non_striker] if batter and non_striker else []
                
                if ball_num in [1, 60, 120, 180, 240]:
                    last_10 = sum(recent_runs[-60:]) if len(recent_runs) >= 60 else sum(recent_runs)
                    crr = (curr_runs * 6.0 / ball_num) if ball_num > 0 else 0
                    
                    b1_avg = get_batsman_avg(batsmen[0], player_database) if len(batsmen) > 0 else 0
                    b2_avg = get_batsman_avg(batsmen[1], player_database) if len(batsmen) > 1 else 0
                    
                    row = pd.DataFrame([{
                        'current_score': curr_runs,
                        'wickets_fallen': curr_wickets,
                        'balls_bowled': ball_num,
                        'balls_remaining': 300 - ball_num,
                        'runs_last_10_overs': last_10,
                        'current_run_rate': crr,
                        'team_batting_avg': batting_agg['team_batting_avg'],
                        'team_elite_batsmen': batting_agg['team_elite_batsmen'],
                        'team_batting_depth': batting_agg['team_batting_depth'],
                        'opp_bowling_economy': bowling_agg['opp_bowling_economy'],
                        'opp_elite_bowlers': bowling_agg['opp_elite_bowlers'],
                        'opp_bowling_depth': bowling_agg['opp_bowling_depth'],
                        'venue_avg_score': venue_avg,
                        'batsman_1_avg': b1_avg,
                        'batsman_2_avg': b2_avg,
                        'venue': venue
                    }])
                    
                    res = {
                        'ball': ball_num,
                        'actual': final_score,
                        'xgb_pred': xgb_model.predict(row)[0] if xgb_model else 0,
                        'rf_pred': rf_model.predict(row)[0] if rf_model else 0
                    }
                    results.append(res)
        
        match_count += 1
        if match_count % 100 == 0: print(f"   Processed {match_count} matches...")

    except Exception:
        continue

# ==============================================================================
# STEP 6: ANALYZE
# ==============================================================================

print("\n[5/5] Analyzing results...")

df = pd.DataFrame(results)

def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    acc_30 = (np.abs(y_true - y_pred) <= 30).sum() / len(y_true) * 100
    print(f"\n{name} RESULTS:")
    print(f"   R2 Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   Accuracy (±30): {acc_30:.1f}%")
    return r2, mae

print(f"\n{'='*80}")
print("INTERNATIONAL VALIDATION RESULTS (v2)")
print(f"{'='*80}")

if xgb_model:
    xgb_r2, xgb_mae = print_metrics("XGBoost v2", df['actual'], df['xgb_pred'])

if rf_model:
    rf_r2, rf_mae = print_metrics("Random Forest v2", df['actual'], df['rf_pred'])

print(f"\n{'='*80}")
print("PROGRESSIVE ACCURACY (Death Overs - 240+ balls)")
print(f"{'='*80}")

death_df = df[df['ball'] == 240]

if xgb_model:
    r2 = r2_score(death_df['actual'], death_df['xgb_pred'])
    print(f"XGBoost v2 Death R2: {r2:.4f}")

if rf_model:
    r2 = r2_score(death_df['actual'], death_df['rf_pred'])
    print(f"Random Forest v2 Death R2: {r2:.4f}")

print(f"\n{'='*80}\n")

# Save results to JSON for reliable reading
output_data = {
    "xgboost_v2": {
        "r2": xgb_r2 if xgb_model else 0,
        "mae": xgb_mae if xgb_model else 0,
        "death_r2": r2_score(death_df['actual'], death_df['xgb_pred']) if xgb_model else 0
    },
    "random_forest_v2": {
        "r2": rf_r2 if rf_model else 0,
        "mae": rf_mae if rf_model else 0,
        "death_r2": r2_score(death_df['actual'], death_df['rf_pred']) if rf_model else 0
    }
}

with open('validation_results_v2.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("[SAVED] validation_results_v2.json")
