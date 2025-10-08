#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Build Quality Player Database

Purpose: Create reliable player statistics from detailed_player_data.csv
Strategy: Quality over quantity - only players with sufficient data
Output: Clean player database with career statistics

Key Decisions:
- Minimum 10 matches for reliability
- Minimum 100 runs OR 5 wickets for impact
- Calculate real career averages, not hash-based values
- Classify players by role and skill level
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_raw_data():
    """Load raw player performance data"""
    print("\n" + "="*70)
    print("PHASE 1: BUILD QUALITY PLAYER DATABASE")
    print("="*70)
    print("\nStep 1: Loading raw player data...")
    print("-"*70)
    
    df = pd.read_csv('../../raw_data/odi_data/detailed_player_data.csv')
    
    print(f"Total records loaded: {len(df):,}")
    print(f"Unique players: {df['player'].nunique():,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")
    print(f"Date range: {df['match_id'].min()} to {df['match_id'].max()}")
    
    return df

def filter_quality_players(df):
    """Filter to keep only players with sufficient data"""
    print("\nStep 2: Filtering for quality players...")
    print("-"*70)
    print("\nQuality Criteria:")
    print("  - Minimum 10 matches")
    print("  - Minimum 100 runs OR 5 wickets")
    print("  - Real statistical contribution\n")
    
    quality_players = []
    player_stats = {}
    
    for player in df['player'].unique():
        player_data = df[df['player'] == player]
        
        # Basic counts
        total_matches = len(player_data)
        total_runs = player_data['runs'].sum()
        total_wickets = player_data['wickets'].sum()
        
        # Batting innings (where they actually batted)
        batting_innings = len(player_data[player_data['balls_faced'] > 0])
        
        # Bowling performances (where they actually bowled)
        bowling_performances = len(player_data[player_data['overs_bowled'] > 0])
        
        # Quality check
        has_enough_matches = total_matches >= 10
        has_batting_contribution = (batting_innings >= 5 and total_runs >= 100)
        has_bowling_contribution = (bowling_performances >= 5 and total_wickets >= 5)
        
        if has_enough_matches and (has_batting_contribution or has_bowling_contribution):
            quality_players.append(player)
            player_stats[player] = {
                'total_matches': total_matches,
                'batting_innings': batting_innings,
                'bowling_performances': bowling_performances,
                'total_runs': total_runs,
                'total_wickets': total_wickets
            }
    
    print(f"Original players: {df['player'].nunique():,}")
    print(f"Quality players: {len(quality_players):,}")
    print(f"Filtered out: {df['player'].nunique() - len(quality_players):,}")
    print(f"Retention rate: {len(quality_players)/df['player'].nunique()*100:.1f}%")
    
    # Filter dataframe
    df_quality = df[df['player'].isin(quality_players)].copy()
    
    print(f"\nRecords after filtering:")
    print(f"  Original: {len(df):,}")
    print(f"  Quality: {len(df_quality):,}")
    print(f"  Retention: {len(df_quality)/len(df)*100:.1f}%")
    
    return df_quality, quality_players, player_stats

def calculate_career_statistics(df, quality_players):
    """Calculate reliable career statistics for each player"""
    print("\nStep 3: Calculating career statistics...")
    print("-"*70)
    
    player_database = {}
    
    for player in quality_players:
        player_data = df[df['player'] == player]
        
        # ===== BATTING STATISTICS =====
        batting_records = player_data[player_data['balls_faced'] > 0]
        
        if len(batting_records) > 0:
            batting_stats = {
                'matches': len(batting_records),
                'innings': len(batting_records),
                'total_runs': int(batting_records['runs'].sum()),
                'total_balls': int(batting_records['balls_faced'].sum()),
                'average': round(batting_records['runs'].mean(), 2),
                'strike_rate': round(batting_records['strike_rate'].mean(), 2),
                'fours': int(batting_records['fours'].sum()),
                'sixes': int(batting_records['sixes'].sum()),
                'highest_score': int(batting_records['runs'].max()),
                'consistency': round(batting_records['runs'].std(), 2) if len(batting_records) > 1 else 0
            }
        else:
            batting_stats = None
        
        # ===== BOWLING STATISTICS =====
        bowling_records = player_data[player_data['overs_bowled'] > 0]
        
        if len(bowling_records) > 0:
            total_wickets = bowling_records['wickets'].sum()
            total_runs_conceded = bowling_records['runs_conceded'].sum()
            total_overs = bowling_records['overs_bowled'].sum()
            
            bowling_stats = {
                'matches': len(bowling_records),
                'total_wickets': int(total_wickets),
                'total_overs': round(total_overs, 1),
                'total_runs_conceded': int(total_runs_conceded),
                'average': round(total_runs_conceded / total_wickets, 2) if total_wickets > 0 else None,
                'economy': round(total_runs_conceded / total_overs, 2) if total_overs > 0 else None,
                'wickets_per_match': round(total_wickets / len(bowling_records), 2),
                'maidens': int(bowling_records['maiden'].sum()),
                'best_figures': int(bowling_records['wickets'].max())
            }
        else:
            bowling_stats = None
        
        # ===== FIELDING STATISTICS =====
        fielding_stats = {
            'catches': int(player_data['catches'].sum()),
            'run_outs': int(player_data['run_outs'].sum()),
            'stumpings': int(player_data['stumps'].sum())
        }
        
        # ===== CLASSIFY PLAYER ROLE =====
        has_batting = batting_stats and batting_stats['innings'] >= 5
        has_bowling = bowling_stats and bowling_stats['total_wickets'] >= 5
        
        if has_batting and has_bowling:
            role = 'All-rounder'
        elif has_batting:
            if fielding_stats['stumpings'] > 2:
                role = 'Wicketkeeper-Batsman'
            else:
                role = 'Batsman'
        elif has_bowling:
            role = 'Bowler'
        else:
            role = 'Unknown'
        
        # ===== CLASSIFY SKILL LEVEL =====
        if role in ['Batsman', 'Wicketkeeper-Batsman']:
            if batting_stats and batting_stats['average'] >= 45:
                skill_level = 'Elite'
            elif batting_stats and batting_stats['average'] >= 35:
                skill_level = 'Star'
            elif batting_stats and batting_stats['average'] >= 25:
                skill_level = 'Good'
            else:
                skill_level = 'Average'
        
        elif role == 'Bowler':
            if bowling_stats and bowling_stats['economy'] and bowling_stats['economy'] < 4.5:
                skill_level = 'Elite'
            elif bowling_stats and bowling_stats['economy'] and bowling_stats['economy'] < 5.0:
                skill_level = 'Star'
            elif bowling_stats and bowling_stats['economy'] and bowling_stats['economy'] < 5.5:
                skill_level = 'Good'
            else:
                skill_level = 'Average'
        
        elif role == 'All-rounder':
            bat_avg = batting_stats['average'] if batting_stats else 0
            bowl_econ = bowling_stats['economy'] if bowling_stats and bowling_stats['economy'] else 999
            
            if bat_avg >= 35 and bowl_econ < 5.0:
                skill_level = 'Elite'
            elif bat_avg >= 28 and bowl_econ < 5.5:
                skill_level = 'Star'
            elif bat_avg >= 22:
                skill_level = 'Good'
            else:
                skill_level = 'Average'
        else:
            skill_level = 'Average'
        
        # ===== CALCULATE STAR RATING (1-10) =====
        if skill_level == 'Elite':
            star_rating = 8.5
        elif skill_level == 'Star':
            star_rating = 7.0
        elif skill_level == 'Good':
            star_rating = 5.5
        else:
            star_rating = 4.0
        
        # Bonus for exceptional performance
        if batting_stats:
            if batting_stats['average'] > 50:
                star_rating += 1.0
            if batting_stats['strike_rate'] > 95:
                star_rating += 0.5
        
        if bowling_stats and bowling_stats['economy']:
            if bowling_stats['economy'] < 4.5:
                star_rating += 1.0
        
        # Cap at 10.0
        star_rating = min(star_rating, 10.0)
        
        # ===== GET TEAMS =====
        teams = player_data['team'].unique().tolist()
        
        # ===== BUILD PLAYER PROFILE =====
        player_database[player] = {
            'name': player,
            'role': role,
            'skill_level': skill_level,
            'star_rating': round(star_rating, 1),
            'teams': teams,
            'total_matches': len(player_data),
            'batting': batting_stats,
            'bowling': bowling_stats,
            'fielding': fielding_stats
        }
    
    print(f"Career statistics calculated for {len(player_database):,} players")
    
    # Statistics
    roles = [p['role'] for p in player_database.values()]
    print(f"\nRole Distribution:")
    for role in ['Batsman', 'Bowler', 'All-rounder', 'Wicketkeeper-Batsman']:
        count = roles.count(role)
        print(f"  {role:25s}: {count:4d} ({count/len(roles)*100:5.1f}%)")
    
    skill_levels = [p['skill_level'] for p in player_database.values()]
    print(f"\nSkill Level Distribution:")
    for level in ['Elite', 'Star', 'Good', 'Average']:
        count = skill_levels.count(level)
        print(f"  {level:25s}: {count:4d} ({count/len(skill_levels)*100:5.1f}%)")
    
    return player_database

def save_player_database(player_database, df_quality):
    """Save player database to files"""
    print("\nStep 4: Saving player database...")
    print("-"*70)
    
    # Create directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../processed_data', exist_ok=True)
    
    # 1. Save complete player database as JSON
    # Convert numpy types to Python types for JSON serialization
    player_database_json = json.loads(json.dumps(player_database, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, (np.floating, np.float64)) else x))
    
    json_path = '../data/player_database.json'
    with open(json_path, 'w') as f:
        json.dump(player_database_json, f, indent=2)
    print(f"Saved: {json_path}")
    print(f"  Players: {len(player_database):,}")
    
    # 2. Save simplified CSV for easy viewing
    player_list = []
    for player, data in player_database.items():
        row = {
            'player': player,
            'role': data['role'],
            'skill_level': data['skill_level'],
            'star_rating': data['star_rating'],
            'total_matches': data['total_matches'],
            'teams': ', '.join(data['teams'])
        }
        
        if data['batting']:
            row['batting_avg'] = data['batting']['average']
            row['strike_rate'] = data['batting']['strike_rate']
            row['total_runs'] = data['batting']['total_runs']
        
        if data['bowling']:
            row['bowling_avg'] = data['bowling']['average']
            row['economy'] = data['bowling']['economy']
            row['total_wickets'] = data['bowling']['total_wickets']
        
        player_list.append(row)
    
    df_players = pd.DataFrame(player_list)
    csv_path = '../processed_data/player_career_statistics.csv'
    df_players.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # 3. Save quality player records (for reference)
    records_path = '../processed_data/quality_player_records.csv'
    df_quality.to_csv(records_path, index=False)
    print(f"Saved: {records_path}")
    print(f"  Records: {len(df_quality):,}")
    
    # 4. Save elite players list
    elite_players = {k: v for k, v in player_database.items() if v['star_rating'] >= 7.5}
    elite_players_json = json.loads(json.dumps(elite_players, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, (np.floating, np.float64)) else x))
    elite_path = '../data/elite_players.json'
    with open(elite_path, 'w') as f:
        json.dump(elite_players_json, f, indent=2)
    print(f"Saved: {elite_path}")
    print(f"  Elite players: {len(elite_players)}")

def display_sample_players(player_database):
    """Display sample elite players"""
    print("\nStep 5: Sample Elite Players")
    print("-"*70)
    
    # Get top 15 players by star rating
    top_players = sorted(player_database.items(), 
                        key=lambda x: (x[1]['star_rating'], x[1]['total_matches']), 
                        reverse=True)[:15]
    
    print(f"\nTop 15 Players by Rating:\n")
    for i, (player, data) in enumerate(top_players, 1):
        print(f"{i:2d}. {data['name']:30s} | {data['role']:20s} | Rating: {data['star_rating']}/10")
        
        if data['batting']:
            print(f"     Batting: Avg {data['batting']['average']:5.1f}, SR {data['batting']['strike_rate']:5.1f}, Runs {data['batting']['total_runs']:,}")
        
        if data['bowling']:
            bowl_avg = data['bowling']['average'] if data['bowling']['average'] else 'N/A'
            print(f"     Bowling: Avg {bowl_avg}, Econ {data['bowling']['economy']:.2f}, Wickets {data['bowling']['total_wickets']}")
        
        print()

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ODI PLAYER DATABASE BUILDER")
    print("="*70)
    print("\nObjective: Build high-quality player database")
    print("Strategy: Quality over quantity")
    print("Criteria: >= 10 matches, >= 100 runs OR >= 5 wickets")
    
    # Load data
    df = load_raw_data()
    
    # Filter quality players
    df_quality, quality_players, player_stats = filter_quality_players(df)
    
    # Calculate career statistics
    player_database = calculate_career_statistics(df_quality, quality_players)
    
    # Save database
    save_player_database(player_database, df_quality)
    
    # Display samples
    display_sample_players(player_database)
    
    # Final summary
    print("\n" + "="*70)
    print("PLAYER DATABASE BUILD COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Quality players: {len(player_database):,}")
    print(f"  Total matches covered: {df_quality['match_id'].nunique():,}")
    print(f"  Total records: {len(df_quality):,}")
    print(f"\nOutput files:")
    print(f"  ODI/data/player_database.json")
    print(f"  ODI/data/elite_players.json")
    print(f"  ODI/processed_data/player_career_statistics.csv")
    print(f"  ODI/processed_data/quality_player_records.csv")
    print(f"\nNext step: Run '2_score_match_quality.py'")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
