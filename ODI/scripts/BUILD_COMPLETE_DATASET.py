#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUILD COMPLETE ODI DATASET

Combines:
1. T20-style features (pitch, weather, form, h2h) → For accuracy
2. Career statistics (from player_database.json) → For player swaps

This gives us:
- High accuracy (R² = 0.69 from contextual features)
- Player swap capability (from career stat features)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*70)
print("BUILD COMPLETE ODI DATASET (CONTEXTUAL + CAREER STATS)")
print("="*70)

# Load player database
with open('../data/player_database.json', 'r') as f:
    player_database = json.load(f)

print(f"\n✓ Loaded {len(player_database):,} players with career stats\n")

def estimate_weather(date_str, city):
    """Estimate weather"""
    try:
        date = datetime.strptime(str(date_str), '%Y-%m-%d')
        month = date.month
    except:
        month = 6
    
    humidity = 60.0
    temperature = 25.0
    
    if any(c in city.lower() for c in ['mumbai', 'delhi', 'kolkata', 'chennai', 'bangalore', 'hyderabad']):
        temperature = 28 + 8 * np.sin(2 * np.pi * month / 12)
        humidity = 70 + 15 * np.sin(2 * np.pi * month / 12)
    elif any(c in city.lower() for c in ['london', 'birmingham', 'manchester', 'cardiff']):
        temperature = 15 + 6 * np.sin(2 * np.pi * month / 12)
        humidity = 80 + 10 * np.sin(2 * np.pi * month / 12)
    elif any(c in city.lower() for c in ['sydney', 'melbourne', 'brisbane', 'perth', 'adelaide']):
        temperature = 22 + 10 * np.sin(2 * np.pi * month / 12)
        humidity = 60 + 20 * np.sin(2 * np.pi * month / 12)
    elif any(c in city.lower() for c in ['dubai', 'abu dhabi', 'sharjah']):
        temperature = 35 + 10 * np.sin(2 * np.pi * month / 12)
        humidity = 50 + 20 * np.sin(2 * np.pi * month / 12)
    else:
        temperature = 25 + 8 * np.sin(2 * np.pi * month / 12)
        humidity = 60 + 15 * np.sin(2 * np.pi * month / 12)
    
    temperature += np.random.normal(0, 2)
    humidity += np.random.normal(0, 5)
    
    temperature = max(min(temperature, 45), 5)
    humidity = max(min(humidity, 100), 20)
    
    return temperature, humidity

def calculate_pitch(total_runs, wickets, overs):
    """Calculate pitch characteristics"""
    runs_per_over = total_runs / max(overs, 1)
    wickets_per_over = wickets / max(overs, 1)
    
    pitch_bounce = 0.5 + (runs_per_over / 10.0)
    pitch_swing = 0.3 + (wickets_per_over * 2.0)
    
    pitch_bounce = max(min(pitch_bounce, 2.0), 0.0)
    pitch_swing = max(min(pitch_swing, 2.0), 0.0)
    
    return pitch_bounce, pitch_swing

def calculate_team_career_features(players, player_database):
    """Calculate team features from player career stats"""
    features = {}
    known_players = []
    
    for player in players:
        if player in player_database:
            known_players.append(player_database[player])
    
    if not known_players:
        return None
    
    # BATTING FEATURES
    batting_avgs = []
    strike_rates = []
    total_runs_list = []
    
    for player in known_players:
        if player.get('batting'):
            batting_avgs.append(player['batting']['average'])
            strike_rates.append(player['batting']['strike_rate'])
            total_runs_list.append(player['batting']['total_runs'])
    
    if batting_avgs:
        features['team_batting_avg'] = round(np.mean(batting_avgs), 2)
        features['team_strike_rate'] = round(np.mean(strike_rates), 2)
        features['team_total_runs'] = sum(total_runs_list)
        features['elite_batsmen'] = sum(1 for avg in batting_avgs if avg >= 45)
        features['star_batsmen'] = sum(1 for avg in batting_avgs if 35 <= avg < 45)
        features['power_hitters'] = sum(1 for sr in strike_rates if sr >= 95)
    else:
        features['team_batting_avg'] = 25.0
        features['team_strike_rate'] = 75.0
        features['team_total_runs'] = 0
        features['elite_batsmen'] = 0
        features['star_batsmen'] = 0
        features['power_hitters'] = 0
    
    # BOWLING FEATURES
    bowling_avgs = []
    economies = []
    total_wickets_list = []
    
    for player in known_players:
        if player.get('bowling') and player['bowling'].get('economy'):
            if player['bowling'].get('average'):
                bowling_avgs.append(player['bowling']['average'])
            economies.append(player['bowling']['economy'])
            total_wickets_list.append(player['bowling']['total_wickets'])
    
    if economies:
        features['team_bowling_avg'] = round(np.mean(bowling_avgs), 2) if bowling_avgs else 35.0
        features['team_economy'] = round(np.mean(economies), 2)
        features['team_total_wickets'] = sum(total_wickets_list)
        features['elite_bowlers'] = sum(1 for econ in economies if econ < 4.5)
        features['star_bowlers'] = sum(1 for econ in economies if 4.5 <= econ < 5.0)
    else:
        features['team_bowling_avg'] = 35.0
        features['team_economy'] = 5.5
        features['team_total_wickets'] = 0
        features['elite_bowlers'] = 0
        features['star_bowlers'] = 0
    
    # ROLE FEATURES
    roles = [p['role'] for p in known_players]
    features['all_rounder_count'] = sum(1 for role in roles if role == 'All-rounder')
    features['wicketkeeper_count'] = sum(1 for role in roles if 'Wicketkeeper' in role)
    
    # QUALITY FEATURES
    skill_levels = [p['skill_level'] for p in known_players]
    star_ratings = [p['star_rating'] for p in known_players]
    
    features['elite_players'] = sum(1 for level in skill_levels if level == 'Elite')
    features['star_players'] = sum(1 for level in skill_levels if level == 'Star')
    features['avg_star_rating'] = round(np.mean(star_ratings), 2)
    
    # BALANCE
    if features['team_batting_avg'] > 0 and features['team_bowling_avg'] > 0:
        features['team_balance'] = round(features['team_batting_avg'] / features['team_bowling_avg'], 3)
    else:
        features['team_balance'] = 1.0
    
    features['team_depth'] = sum(1 for avg in batting_avgs if avg >= 25)
    features['known_players_count'] = len(known_players)
    
    return features

# Load and parse all matches
ballbyball_dir = '../../raw_data/odis_ballbyBall'
match_files = [f for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"Found {len(match_files):,} match files\n")
print("Step 1: Parsing matches...")

all_matches = []
for i, filename in enumerate(match_files):
    try:
        with open(os.path.join(ballbyball_dir, filename), 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        info = match['info']
        date = info.get('dates', ['2020-01-01'])[0]
        
        all_matches.append({
            'match_id': filename.replace('.json', ''),
            'date': date,
            'match_data': match
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Parsed {i+1}/{len(match_files)}...")
    except:
        continue

all_matches.sort(key=lambda x: x['date'])
print(f"\n✓ Sorted {len(all_matches):,} matches\n")

print("Step 2: Building COMPLETE dataset...")
print("(Contextual features + Career statistics)\n")

# Tracking
team_history = defaultdict(list)
venue_stats = defaultdict(list)
h2h_history = defaultdict(list)

dataset = []
processed = 0

for match_info in all_matches:
    try:
        match = match_info['match_data']
        info = match['info']
        match_id = match_info['match_id']
        date = match_info['date']
        
        teams = list(info['players'].keys())
        if len(teams) != 2:
            continue
        
        team_a, team_b = teams[0], teams[1]
        team_a_players = info['players'][team_a]
        team_b_players = info['players'][team_b]
        venue = info.get('venue', 'Unknown')
        city = info.get('city', venue.split(',')[0] if ',' in venue else venue)
        
        # Calculate scores
        innings = match.get('innings', [])
        if len(innings) < 2:
            continue
        
        team_a_score = 0
        team_b_score = 0
        team_a_wickets = 0
        team_b_wickets = 0
        team_a_overs = 0
        team_b_overs = 0
        
        for inning in innings:
            inning_team = inning.get('team', '')
            total_runs = 0
            wickets = 0
            overs = len(inning.get('overs', []))
            
            for over in inning.get('overs', []):
                for delivery in over.get('deliveries', []):
                    runs = delivery.get('runs', {}).get('total', 0)
                    total_runs += runs
                    if 'wickets' in delivery:
                        wickets += len(delivery['wickets'])
            
            if inning_team == team_a:
                team_a_score = total_runs
                team_a_wickets = wickets
                team_a_overs = overs
            elif inning_team == team_b:
                team_b_score = total_runs
                team_b_wickets = wickets
                team_b_overs = overs
        
        # CAREER STATISTICS (from player_database)
        team_a_career = calculate_team_career_features(team_a_players, player_database)
        team_b_career = calculate_team_career_features(team_b_players, player_database)
        
        if not team_a_career or not team_b_career:
            continue
        
        # CONTEXTUAL FEATURES
        temperature, humidity = estimate_weather(date, city)
        pitch_bounce_a, pitch_swing_a = calculate_pitch(team_a_score, team_a_wickets, team_a_overs)
        pitch_bounce_b, pitch_swing_b = calculate_pitch(team_b_score, team_b_wickets, team_b_overs)
        pitch_bounce = (pitch_bounce_a + pitch_bounce_b) / 2
        pitch_swing = (pitch_swing_a + pitch_swing_b) / 2
        
        # TEMPORAL FEATURES
        team_a_recent = team_history[team_a][-5:] if team_a in team_history else []
        team_b_recent = team_history[team_b][-5:] if team_b in team_history else []
        
        team_a_recent_avg = np.mean([x[1] for x in team_a_recent]) if team_a_recent else 228.0
        team_b_recent_avg = np.mean([x[1] for x in team_b_recent]) if team_b_recent else 228.0
        
        # VENUE FEATURES
        venue_history = venue_stats[venue] if venue in venue_stats else []
        venue_avg = np.mean(venue_history) if venue_history else 228.0
        venue_high = max(venue_history) if venue_history else 350
        venue_low = min(venue_history) if venue_history else 150
        venue_std = np.std(venue_history) if len(venue_history) > 1 else 50
        
        # H2H FEATURES
        h2h_key_a = (team_a, team_b)
        h2h_a = h2h_history[h2h_key_a] if h2h_key_a in h2h_history else []
        h2h_avg_a = np.mean(h2h_a) if h2h_a else 228.0
        h2h_wins_a = sum(1 for s in h2h_a if s > 228)
        h2h_win_rate_a = h2h_wins_a / max(len(h2h_a), 1)
        
        h2h_key_b = (team_b, team_a)
        h2h_b = h2h_history[h2h_key_b] if h2h_key_b in h2h_history else []
        h2h_avg_b = np.mean(h2h_b) if h2h_b else 228.0
        h2h_wins_b = sum(1 for s in h2h_b if s > 228)
        h2h_win_rate_b = h2h_wins_b / max(len(h2h_b), 1)
        
        # DATE PARSING
        try:
            date_obj = datetime.strptime(str(date), '%Y-%m-%d')
            season_year = date_obj.year
            season_month = date_obj.month
        except:
            season_year = 2020
            season_month = 6
        
        # TOSS & CONTEXT
        toss_winner = info.get('toss', {}).get('winner', '')
        toss_decision = info.get('toss', {}).get('decision', 'bat')
        gender = info.get('gender', 'male')
        match_type = info.get('match_type', 'ODI')
        event = info.get('event', {})
        event_name = event.get('name', '') if isinstance(event, dict) else str(event)
        match_number = event.get('match_number', 0) if isinstance(event, dict) else 0
        
        # ROW 1: Team A
        row_a = {
            'match_id': match_id,
            'date': date,
            'venue': venue,
            'team': team_a,
            'opposition': team_b,
            'season_year': season_year,
            'season_month': season_month,
            'gender': gender,
            'match_type': match_type,
            'event_name': event_name[:50] if event_name else '',
            'match_number': match_number,
            'toss_won': 1 if toss_winner == team_a else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'toss_decision_field': 1 if toss_decision == 'field' else 0,
            # CAREER STATS (Team A)
            **{f'team_{k}': v for k, v in team_a_career.items()},
            # CAREER STATS (Team B as opposition)
            **{f'opp_{k}': v for k, v in team_b_career.items()},
            # RELATIVE FEATURES
            'batting_advantage': team_a_career['team_batting_avg'] - team_b_career['team_bowling_avg'],
            'star_advantage': team_a_career['star_players'] - team_b_career['star_players'],
            'elite_advantage': team_a_career['elite_players'] - team_b_career['elite_players'],
            # TEMPORAL FEATURES
            'team_recent_avg': team_a_recent_avg,
            'team_form_matches': len(team_a_recent),
            'opposition_recent_avg': team_b_recent_avg,
            # H2H FEATURES
            'h2h_avg_runs': h2h_avg_a,
            'h2h_matches': len(h2h_a),
            'h2h_win_rate': h2h_win_rate_a,
            # VENUE FEATURES
            'venue_avg_runs': venue_avg,
            'venue_high_score': venue_high,
            'venue_low_score': venue_low,
            'venue_runs_std': venue_std,
            'venue_matches': len(venue_history),
            # PITCH & WEATHER
            'pitch_bounce': pitch_bounce,
            'pitch_swing': pitch_swing,
            'humidity': humidity,
            'temperature': temperature,
            # TARGET
            'total_runs': team_a_score
        }
        
        # ROW 2: Team B
        row_b = {
            'match_id': match_id,
            'date': date,
            'venue': venue,
            'team': team_b,
            'opposition': team_a,
            'season_year': season_year,
            'season_month': season_month,
            'gender': gender,
            'match_type': match_type,
            'event_name': event_name[:50] if event_name else '',
            'match_number': match_number,
            'toss_won': 1 if toss_winner == team_b else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'toss_decision_field': 1 if toss_decision == 'field' else 0,
            **{f'team_{k}': v for k, v in team_b_career.items()},
            **{f'opp_{k}': v for k, v in team_a_career.items()},
            'batting_advantage': team_b_career['team_batting_avg'] - team_a_career['team_bowling_avg'],
            'star_advantage': team_b_career['star_players'] - team_a_career['star_players'],
            'elite_advantage': team_b_career['elite_players'] - team_a_career['elite_players'],
            'team_recent_avg': team_b_recent_avg,
            'team_form_matches': len(team_b_recent),
            'opposition_recent_avg': team_a_recent_avg,
            'h2h_avg_runs': h2h_avg_b,
            'h2h_matches': len(h2h_b),
            'h2h_win_rate': h2h_win_rate_b,
            'venue_avg_runs': venue_avg,
            'venue_high_score': venue_high,
            'venue_low_score': venue_low,
            'venue_runs_std': venue_std,
            'venue_matches': len(venue_history),
            'pitch_bounce': pitch_bounce,
            'pitch_swing': pitch_swing,
            'humidity': humidity,
            'temperature': temperature,
            'total_runs': team_b_score
        }
        
        dataset.append(row_a)
        dataset.append(row_b)
        
        # UPDATE HISTORIES
        team_history[team_a].append((date, team_a_score))
        team_history[team_b].append((date, team_b_score))
        venue_stats[venue].extend([team_a_score, team_b_score])
        h2h_history[(team_a, team_b)].append(team_a_score)
        h2h_history[(team_b, team_a)].append(team_b_score)
        
        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(all_matches)}... ({len(dataset):,} rows)")
    
    except Exception as e:
        continue

print(f"\n✓ Built {len(dataset):,} rows from {processed:,} matches\n")

# Save
df = pd.DataFrame(dataset)

print("="*70)
print("COMPLETE DATASET")
print("="*70)

print(f"\nShape: {df.shape}")
print(f"\nFeature Groups:")
print(f"  Career Stats: ~38 (team + opp batting/bowling/composition)")
print(f"  Contextual: ~11 (pitch, weather, venue stats)")
print(f"  Temporal: ~9 (recent form, h2h)")
print(f"  Basic: ~8 (toss, season, event)")
print(f"  Total: {len(df.columns)}")

output_path = '../data/odi_complete_dataset.csv'
df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print(f"  Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")

print("\n" + "="*70)
print("COMPLETE DATASET READY!")
print("="*70)
print("\nThis has BOTH:")
print("  ✓ Career stats (for player swaps)")
print("  ✓ Contextual features (for accuracy)")
print("\nNow train and see if accuracy holds!")
print("="*70 + "\n")

