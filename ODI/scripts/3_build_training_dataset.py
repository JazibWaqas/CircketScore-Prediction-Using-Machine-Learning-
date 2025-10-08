#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Build Training Dataset with Real Player Features

Purpose: Combine player database + high-quality matches → Training CSV
Strategy: Calculate REAL team features from actual player career statistics
Output: Training dataset ready for Random Forest & XGBoost

This is where the magic happens - player impact becomes measurable!
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_resources():
    """Load player database and high-quality match list"""
    print("\n" + "="*70)
    print("PHASE 3: BUILD TRAINING DATASET")
    print("="*70)
    print("\nStep 1: Loading resources...")
    print("-"*70)
    
    # Load player database
    with open('../data/player_database.json', 'r') as f:
        player_database = json.load(f)
    print(f"Player database loaded: {len(player_database):,} players")
    
    # Load high-quality match IDs
    with open('../data/high_quality_match_ids.json', 'r') as f:
        match_ids = json.load(f)
    print(f"High-quality matches: {len(match_ids):,} matches")
    
    # Load match quality scores for additional info
    df_matches = pd.read_csv('../processed_data/high_quality_matches.csv')
    print(f"Match metadata loaded: {len(df_matches):,} records")
    
    return player_database, match_ids, df_matches

def calculate_team_features(players, player_database):
    """Calculate team-level features from individual player stats"""
    
    features = {}
    known_players = []
    
    # Get known players
    for player in players:
        if player in player_database:
            known_players.append(player_database[player])
    
    if not known_players:
        return None  # Can't calculate features without known players
    
    # === BATTING FEATURES ===
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
        features['team_batting_avg'] = 25.0  # Default
        features['team_strike_rate'] = 75.0
        features['team_total_runs'] = 0
        features['elite_batsmen'] = 0
        features['star_batsmen'] = 0
        features['power_hitters'] = 0
    
    # === BOWLING FEATURES ===
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
        features['team_bowling_avg'] = 35.0  # Default
        features['team_economy'] = 5.5
        features['team_total_wickets'] = 0
        features['elite_bowlers'] = 0
        features['star_bowlers'] = 0
    
    # === ROLE FEATURES ===
    roles = [p['role'] for p in known_players]
    features['all_rounder_count'] = sum(1 for role in roles if role == 'All-rounder')
    features['wicketkeeper_count'] = sum(1 for role in roles if 'Wicketkeeper' in role)
    
    # === QUALITY FEATURES ===
    skill_levels = [p['skill_level'] for p in known_players]
    star_ratings = [p['star_rating'] for p in known_players]
    
    features['elite_players'] = sum(1 for level in skill_levels if level == 'Elite')
    features['star_players'] = sum(1 for level in skill_levels if level == 'Star')
    features['avg_star_rating'] = round(np.mean(star_ratings), 2)
    
    # === BALANCE FEATURES ===
    if features['team_batting_avg'] > 0 and features['team_bowling_avg'] > 0:
        features['team_balance'] = round(features['team_batting_avg'] / features['team_bowling_avg'], 3)
    else:
        features['team_balance'] = 1.0
    
    features['team_depth'] = sum(1 for avg in batting_avgs if avg >= 25)
    features['known_players_count'] = len(known_players)
    
    return features

def process_match(match_file, player_database):
    """Process a single match and create training rows"""
    
    try:
        with open(match_file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        info = match['info']
        match_id = os.path.basename(match_file).replace('.json', '')
        
        # Extract match context
        teams = list(info['players'].keys())
        if len(teams) != 2:
            return None
        
        team_a, team_b = teams[0], teams[1]
        team_a_players = info['players'][team_a]
        team_b_players = info['players'][team_b]
        
        # Calculate team features
        team_a_features = calculate_team_features(team_a_players, player_database)
        team_b_features = calculate_team_features(team_b_players, player_database)
        
        if not team_a_features or not team_b_features:
            return None
        
        # Extract match outcome (scores)
        innings = match.get('innings', [])
        if len(innings) < 2:
            return None
        
        # Calculate scores from innings
        team_a_score = 0
        team_b_score = 0
        
        for inning in innings:
            inning_team = inning.get('team', '')
            total_runs = 0
            
            for over in inning.get('overs', []):
                for delivery in over.get('deliveries', []):
                    runs = delivery.get('runs', {}).get('total', 0)
                    total_runs += runs
            
            if inning_team == team_a:
                team_a_score = total_runs
            elif inning_team == team_b:
                team_b_score = total_runs
        
        # Match context
        venue = info.get('venue', 'Unknown')
        toss_winner = info.get('toss', {}).get('winner', '')
        toss_decision = info.get('toss', {}).get('decision', 'bat')
        date = info.get('dates', ['2020-01-01'])[0]
        gender = info.get('gender', 'male')
        match_type = info.get('match_type', 'ODI')
        
        # Parse date
        try:
            date_obj = datetime.strptime(str(date), '%Y-%m-%d')
            season_year = date_obj.year
            season_month = date_obj.month
        except:
            season_year = 2020
            season_month = 6
        
        # Create Row 1: Team A batting
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
            'toss_won': 1 if toss_winner == team_a else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'toss_decision_field': 1 if toss_decision == 'field' else 0,
        }
        
        # Add Team A features (prefixed with 'team_')
        for key, value in team_a_features.items():
            row_a[f'team_{key}'] = value
        
        # Add Team B features as opposition (prefixed with 'opp_')
        for key, value in team_b_features.items():
            row_a[f'opp_{key}'] = value
        
        # Relative features
        row_a['batting_advantage'] = round(team_a_features['team_batting_avg'] - team_b_features['team_bowling_avg'], 2)
        row_a['star_advantage'] = team_a_features['star_players'] - team_b_features['star_players']
        row_a['elite_advantage'] = team_a_features['elite_players'] - team_b_features['elite_players']
        
        # Target
        row_a['total_runs'] = team_a_score
        
        # Create Row 2: Team B batting
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
            'toss_won': 1 if toss_winner == team_b else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'toss_decision_field': 1 if toss_decision == 'field' else 0,
        }
        
        # Add Team B features
        for key, value in team_b_features.items():
            row_b[f'team_{key}'] = value
        
        # Add Team A features as opposition
        for key, value in team_a_features.items():
            row_b[f'opp_{key}'] = value
        
        # Relative features
        row_b['batting_advantage'] = round(team_b_features['team_batting_avg'] - team_a_features['team_bowling_avg'], 2)
        row_b['star_advantage'] = team_b_features['star_players'] - team_a_features['star_players']
        row_b['elite_advantage'] = team_b_features['elite_players'] - team_a_features['elite_players']
        
        # Target
        row_b['total_runs'] = team_b_score
        
        return [row_a, row_b]
        
    except Exception as e:
        print(f"Error processing match: {e}")
        return None

def build_dataset(player_database, match_ids):
    """Build complete training dataset"""
    print("\nStep 2: Building training dataset...")
    print("-"*70)
    print(f"Processing {len(match_ids):,} high-quality matches...")
    print("This may take several minutes...\n")
    
    ballbyball_dir = '../../raw_data/odis_ballbyBall'
    all_rows = []
    processed = 0
    skipped = 0
    
    for match_id in match_ids:
        match_file = os.path.join(ballbyball_dir, f'{match_id}.json')
        
        if not os.path.exists(match_file):
            skipped += 1
            continue
        
        rows = process_match(match_file, player_database)
        
        if rows:
            all_rows.extend(rows)
        else:
            skipped += 1
        
        processed += 1
        if processed % 200 == 0:
            print(f"  Processed: {processed:,}/{len(match_ids):,} ({processed/len(match_ids)*100:.1f}%) | Rows: {len(all_rows):,}")
    
    print(f"\nProcessing complete!")
    print(f"  Matches processed: {processed:,}")
    print(f"  Matches skipped: {skipped:,}")
    print(f"  Training rows created: {len(all_rows):,}")
    
    df = pd.DataFrame(all_rows)
    return df

def analyze_dataset(df):
    """Analyze the created dataset"""
    print("\nStep 3: Analyzing dataset...")
    print("-"*70)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(df.columns)}")
    
    print(f"\nFeature Categories:")
    team_features = [col for col in df.columns if col.startswith('team_')]
    opp_features = [col for col in df.columns if col.startswith('opp_')]
    context_features = [col for col in df.columns if col not in team_features + opp_features and col != 'total_runs']
    
    print(f"  Team features: {len(team_features)}")
    print(f"  Opposition features: {len(opp_features)}")
    print(f"  Context features: {len(context_features)}")
    print(f"  Target: total_runs")
    
    print(f"\nTarget Variable (total_runs):")
    print(f"  Mean: {df['total_runs'].mean():.1f}")
    print(f"  Median: {df['total_runs'].median():.1f}")
    print(f"  Std: {df['total_runs'].std():.1f}")
    print(f"  Min: {df['total_runs'].min()}")
    print(f"  Max: {df['total_runs'].max()}")
    
    print(f"\nKey Team Features:")
    if 'team_team_batting_avg' in df.columns:
        print(f"  team_batting_avg: {df['team_team_batting_avg'].mean():.2f} (mean)")
        print(f"  team_star_players: {df['team_star_players'].mean():.2f} (mean)")
        print(f"  team_elite_players: {df['team_elite_players'].mean():.2f} (mean)")
    
    print(f"\nDate Range:")
    print(f"  Earliest: {df['date'].min()}")
    print(f"  Latest: {df['date'].max()}")
    
    print(f"\nGender Distribution:")
    print(df['gender'].value_counts())

def save_dataset(df):
    """Save the training dataset"""
    print("\nStep 4: Saving training dataset...")
    print("-"*70)
    
    os.makedirs('../data', exist_ok=True)
    
    output_path = '../data/odi_training_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Save feature list
    feature_list = {
        'total_features': len(df.columns),
        'target': 'total_runs',
        'features': df.columns.tolist()
    }
    
    feature_path = '../data/feature_list.json'
    with open(feature_path, 'w') as f:
        json.dump(feature_list, f, indent=2)
    print(f"Saved: {feature_path}")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ODI TRAINING DATASET BUILDER")
    print("="*70)
    print("\nObjective: Build training dataset with REAL player features")
    print("Strategy: Calculate team features from actual player career stats")
    print("Output: Ready-to-train CSV for Random Forest & XGBoost")
    
    # Load resources
    player_database, match_ids, df_matches = load_resources()
    
    # Build dataset
    df = build_dataset(player_database, match_ids)
    
    # Analyze
    analyze_dataset(df)
    
    # Save
    save_dataset(df)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING DATASET BUILD COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Training dataset: {len(df):,} rows × {len(df.columns)} features")
    print(f"  Quality players used: {len(player_database):,}")
    print(f"  High-quality matches: {len(match_ids):,}")
    print(f"\nOutput files:")
    print(f"  ODI/data/odi_training_dataset.csv")
    print(f"  ODI/data/feature_list.json")
    print(f"\nNext step: Train models (Random Forest, XGBoost)")
    print(f"\nKey Achievement:")
    print(f"  - Team features calculated from REAL player career stats")
    print(f"  - NOT hash-based pseudo-values")
    print(f"  - Player impact is now MEASURABLE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

