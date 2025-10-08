#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUILD ODI DATASET EXACTLY LIKE T20

Extracts from ball-by-ball JSON:
1. Match context (venue, date, teams, toss)
2. Final scores
3. ESTIMATED weather (humidity, temperature)
4. CALCULATED pitch characteristics (bounce, swing)
5. Team form (recent matches)
6. H2H history
7. Venue statistics

This is EXACTLY what T20 did to achieve R² = 0.70
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
print("BUILD ODI DATASET LIKE T20")
print("="*70)
print("\nProcessing 5,761 ODI matches with T20's exact methodology...\n")

def estimate_weather(date_str, city):
    """Estimate weather (SAME AS T20)"""
    try:
        date = datetime.strptime(str(date_str), '%Y-%m-%d')
        month = date.month
    except:
        month = 6
    
    # Base weather
    humidity = 60.0
    temperature = 25.0
    
    # Location adjustments (SAME AS T20)
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
    
    # Add randomness (SAME AS T20)
    temperature += np.random.normal(0, 2)
    humidity += np.random.normal(0, 5)
    
    # Bounds
    temperature = max(min(temperature, 45), 5)
    humidity = max(min(humidity, 100), 20)
    
    return temperature, humidity

def calculate_pitch_characteristics(total_runs, wickets, overs):
    """Calculate pitch bounce/swing (SAME LOGIC AS T20)"""
    runs_per_over = total_runs / max(overs, 1)
    wickets_per_over = wickets / max(overs, 1)
    
    # Bounce (higher for flat pitches with high scoring)
    pitch_bounce = 0.5 + (runs_per_over / 10.0)
    
    # Swing (higher for green pitches with wickets)
    pitch_swing = 0.3 + (wickets_per_over * 2.0)
    
    # Normalize
    pitch_bounce = max(min(pitch_bounce, 2.0), 0.0)
    pitch_swing = max(min(pitch_swing, 2.0), 0.0)
    
    return pitch_bounce, pitch_swing

# Load all matches
ballbyball_dir = '../../raw_data/odis_ballbyBall'
match_files = [f for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"Found {len(match_files):,} match files\n")
print("Step 1: Parsing matches chronologically...")

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
print(f"\n✓ Sorted {len(all_matches):,} matches chronologically\n")

print("Step 2: Building dataset with ALL features (like T20)...")

# Tracking structures
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
        venue = info.get('venue', 'Unknown')
        city = info.get('city', venue.split(',')[0] if ',' in venue else venue)
        
        # Calculate scores and wickets
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
        
        # ESTIMATE WEATHER (LIKE T20)
        temperature, humidity = estimate_weather(date, city)
        
        # CALCULATE PITCH (LIKE T20)
        pitch_bounce_a, pitch_swing_a = calculate_pitch_characteristics(team_a_score, team_a_wickets, team_a_overs)
        pitch_bounce_b, pitch_swing_b = calculate_pitch_characteristics(team_b_score, team_b_wickets, team_b_overs)
        pitch_bounce = (pitch_bounce_a + pitch_bounce_b) / 2
        pitch_swing = (pitch_swing_a + pitch_swing_b) / 2
        
        # RECENT FORM (LIKE T20)
        team_a_recent = team_history[team_a][-5:] if team_a in team_history else []
        team_b_recent = team_history[team_b][-5:] if team_b in team_history else []
        
        team_a_recent_avg = np.mean([x[1] for x in team_a_recent]) if team_a_recent else 228.0
        team_b_recent_avg = np.mean([x[1] for x in team_b_recent]) if team_b_recent else 228.0
        
        team_a_form_matches = len(team_a_recent)
        team_b_form_matches = len(team_b_recent)
        
        # VENUE STATS (LIKE T20)
        venue_history = venue_stats[venue] if venue in venue_stats else []
        venue_avg = np.mean(venue_history) if venue_history else 228.0
        venue_high = max(venue_history) if venue_history else 350
        venue_low = min(venue_history) if venue_history else 150
        venue_std = np.std(venue_history) if len(venue_history) > 1 else 50
        
        # H2H (LIKE T20)
        h2h_key_a = (team_a, team_b)
        h2h_a = h2h_history[h2h_key_a] if h2h_key_a in h2h_history else []
        h2h_avg_a = np.mean(h2h_a) if h2h_a else 228.0
        h2h_matches_a = len(h2h_a)
        h2h_wins_a = sum(1 for s in h2h_a if s > 228)  # Simple win proxy
        h2h_win_rate_a = h2h_wins_a / max(h2h_matches_a, 1)
        
        h2h_key_b = (team_b, team_a)
        h2h_b = h2h_history[h2h_key_b] if h2h_key_b in h2h_history else []
        h2h_avg_b = np.mean(h2h_b) if h2h_b else 228.0
        h2h_matches_b = len(h2h_b)
        h2h_wins_b = sum(1 for s in h2h_b if s > 228)
        h2h_win_rate_b = h2h_wins_b / max(h2h_matches_b, 1)
        
        # PARSE DATE
        try:
            date_obj = datetime.strptime(str(date), '%Y-%m-%d')
            season_year = date_obj.year
            season_month = date_obj.month
        except:
            season_year = 2020
            season_month = 6
        
        # TOSS
        toss_winner = info.get('toss', {}).get('winner', '')
        toss_decision = info.get('toss', {}).get('decision', 'bat')
        
        # EVENT
        event = info.get('event', {})
        event_name = event.get('name', '') if isinstance(event, dict) else str(event)
        match_number = event.get('match_number', 0) if isinstance(event, dict) else 0
        
        # GENDER
        gender = info.get('gender', 'male')
        
        # CREATE ROWS (2 per match)
        base_features = {
            'venue': venue,
            'venue_avg_runs': venue_avg,
            'venue_high_score': venue_high,
            'venue_low_score': venue_low,
            'venue_runs_std': venue_std,
            'venue_matches': len(venue_history),
            'pitch_bounce': pitch_bounce,
            'pitch_swing': pitch_swing,
            'humidity': humidity,
            'temperature': temperature,
            'season_year': season_year,
            'season_month': season_month,
            'event_name': event_name[:50] if event_name else '',
            'match_number': match_number,
            'date': date
        }
        
        # ROW 1: Team A
        row_a = {
            'match_id': match_id,
            'team': team_a,
            'opposition': team_b,
            'toss_won': 1 if toss_winner == team_a else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'toss_decision_field': 1 if toss_decision == 'field' else 0,
            'team_recent_avg': team_a_recent_avg,
            'team_form_matches': team_a_form_matches,
            'opposition_recent_avg': team_b_recent_avg,
            'h2h_avg_runs': h2h_avg_a,
            'h2h_matches': h2h_matches_a,
            'h2h_win_rate': h2h_win_rate_a,
            'gender': gender,
            **base_features,
            'total_runs': team_a_score
        }
        
        # ROW 2: Team B
        row_b = {
            'match_id': match_id,
            'team': team_b,
            'opposition': team_a,
            'toss_won': 1 if toss_winner == team_b else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'toss_decision_field': 1 if toss_decision == 'field' else 0,
            'team_recent_avg': team_b_recent_avg,
            'team_form_matches': team_b_form_matches,
            'opposition_recent_avg': team_a_recent_avg,
            'h2h_avg_runs': h2h_avg_b,
            'h2h_matches': h2h_matches_b,
            'h2h_win_rate': h2h_win_rate_b,
            'gender': gender,
            **base_features,
            'total_runs': team_b_score
        }
        
        dataset.append(row_a)
        dataset.append(row_b)
        
        # UPDATE HISTORIES (AFTER creating rows)
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
print("SAVING T20-STYLE ODI DATASET")
print("="*70)

print(f"\nDataset: {df.shape}")
print(f"\nFeatures: {len(df.columns)}")
print(df.columns.tolist())

output_path = '../data/odi_t20_style_dataset.csv'
df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print(f"  Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")

print("\n" + "="*70)
print("T20-STYLE DATASET COMPLETE!")
print("="*70)
print("\nThis dataset has the SAME features as T20:")
print("  ✓ pitch_bounce, pitch_swing (calculated)")
print("  ✓ humidity, temperature (estimated)")
print("  ✓ venue stats (high, low, avg, std)")
print("  ✓ team recent form")
print("  ✓ h2h history with win rates")
print("  ✓ match context (event, toss, season)")
print("\nExpected: R² = 0.60-0.70 (like T20)")
print("="*70 + "\n")

