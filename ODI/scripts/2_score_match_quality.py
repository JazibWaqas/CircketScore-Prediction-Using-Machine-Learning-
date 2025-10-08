#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Score Match Quality Based on Player Coverage

Purpose: Evaluate all 5,761 ballbyball matches for player coverage
Strategy: Keep matches where we know most players (>= 70% coverage)
Output: Match quality scores and filtered match list

Key Decisions:
- Tier 1 (Excellent): >= 80% player coverage
- Tier 2 (Good): 65-80% player coverage
- Tier 3 (Fair): 50-65% player coverage
- Tier 4 (Poor): < 50% player coverage (exclude)
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

def load_player_database():
    """Load the quality player database from Phase 1"""
    print("\n" + "="*70)
    print("PHASE 2: SCORE MATCH QUALITY")
    print("="*70)
    print("\nStep 1: Loading player database...")
    print("-"*70)
    
    with open('../data/player_database.json', 'r') as f:
        player_database = json.load(f)
    
    print(f"Quality players loaded: {len(player_database):,}")
    
    # Get player names for quick lookup
    known_players = set(player_database.keys())
    
    return player_database, known_players

def score_single_match(match_file, known_players):
    """Score a single match for player coverage"""
    try:
        with open(match_file, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        match_id = os.path.basename(match_file).replace('.json', '')
        info = match['info']
        
        # Extract teams and players
        if 'players' not in info:
            return None
        
        players_dict = info['players']
        teams = list(players_dict.keys())
        
        if len(teams) != 2:
            return None
        
        team_a, team_b = teams[0], teams[1]
        team_a_players = players_dict[team_a]
        team_b_players = players_dict[team_b]
        
        # Count known vs unknown players
        team_a_known = [p for p in team_a_players if p in known_players]
        team_b_known = [p for p in team_b_players if p in known_players]
        
        team_a_coverage = len(team_a_known) / len(team_a_players) if team_a_players else 0
        team_b_coverage = len(team_b_known) / len(team_b_players) if team_b_players else 0
        total_coverage = (team_a_coverage + team_b_coverage) / 2
        
        # Determine tier
        if total_coverage >= 0.80:
            tier = 'Tier 1 - Excellent'
            tier_num = 1
        elif total_coverage >= 0.65:
            tier = 'Tier 2 - Good'
            tier_num = 2
        elif total_coverage >= 0.50:
            tier = 'Tier 3 - Fair'
            tier_num = 3
        else:
            tier = 'Tier 4 - Poor'
            tier_num = 4
        
        # Extract match context
        match_score = {
            'match_id': match_id,
            'date': info.get('dates', ['Unknown'])[0],
            'venue': info.get('venue', 'Unknown'),
            'team_a': team_a,
            'team_b': team_b,
            'team_a_players_total': len(team_a_players),
            'team_a_players_known': len(team_a_known),
            'team_a_coverage': round(team_a_coverage * 100, 1),
            'team_b_players_total': len(team_b_players),
            'team_b_players_known': len(team_b_known),
            'team_b_coverage': round(team_b_coverage * 100, 1),
            'total_players': len(team_a_players) + len(team_b_players),
            'known_players': len(team_a_known) + len(team_b_known),
            'total_coverage': round(total_coverage * 100, 1),
            'tier': tier,
            'tier_num': tier_num,
            'match_type': info.get('match_type', 'Unknown'),
            'gender': info.get('gender', 'Unknown'),
            'event': info.get('event', {}).get('name', 'Unknown'),
            'has_outcome': 'outcome' in info
        }
        
        return match_score
        
    except Exception as e:
        return None

def score_all_matches(known_players):
    """Score all ballbyball matches"""
    print("\nStep 2: Scoring all ballbyball matches...")
    print("-"*70)
    
    ballbyball_dir = '../../raw_data/odis_ballbyBall'
    json_files = [f for f in os.listdir(ballbyball_dir) if f.endswith('.json')]
    
    print(f"Total matches to score: {len(json_files):,}")
    print("Processing... (this may take a few minutes)")
    
    match_scores = []
    processed = 0
    
    for json_file in json_files:
        match_file = os.path.join(ballbyball_dir, json_file)
        score = score_single_match(match_file, known_players)
        
        if score:
            match_scores.append(score)
        
        processed += 1
        if processed % 500 == 0:
            print(f"  Processed: {processed:,}/{len(json_files):,} ({processed/len(json_files)*100:.1f}%)")
    
    print(f"\nSuccessfully scored: {len(match_scores):,} matches")
    print(f"Failed/Invalid: {len(json_files) - len(match_scores):,} matches")
    
    return pd.DataFrame(match_scores)

def analyze_match_quality(df):
    """Analyze and display match quality distribution"""
    print("\nStep 3: Analyzing match quality distribution...")
    print("-"*70)
    
    print(f"\nMatch Quality Tiers:")
    tier_counts = df['tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        percentage = count / len(df) * 100
        print(f"  {tier:25s}: {count:5,} matches ({percentage:5.1f}%)")
    
    print(f"\nCoverage Statistics:")
    print(f"  Mean coverage: {df['total_coverage'].mean():.1f}%")
    print(f"  Median coverage: {df['total_coverage'].median():.1f}%")
    print(f"  Min coverage: {df['total_coverage'].min():.1f}%")
    print(f"  Max coverage: {df['total_coverage'].max():.1f}%")
    
    print(f"\nMatch Type Distribution:")
    for match_type, count in df['match_type'].value_counts().head(5).items():
        print(f"  {match_type}: {count:,}")
    
    print(f"\nGender Distribution:")
    for gender, count in df['gender'].value_counts().items():
        print(f"  {gender}: {count:,}")

def filter_quality_matches(df):
    """Filter to keep only high-quality matches"""
    print("\nStep 4: Filtering for high-quality matches...")
    print("-"*70)
    
    # Keep Tier 1 and Tier 2 (>= 65% coverage)
    high_quality = df[df['tier_num'] <= 2].copy()
    
    # Also filter for completed matches
    high_quality = high_quality[high_quality['has_outcome'] == True].copy()
    
    print(f"\nFiltering Criteria:")
    print(f"  - Tier 1 or Tier 2 (>= 65% player coverage)")
    print(f"  - Match has outcome (completed)")
    
    print(f"\nResults:")
    print(f"  Original matches: {len(df):,}")
    print(f"  High-quality matches: {len(high_quality):,}")
    print(f"  Filtered out: {len(df) - len(high_quality):,}")
    print(f"  Retention rate: {len(high_quality)/len(df)*100:.1f}%")
    
    # Additional statistics
    tier1_count = len(high_quality[high_quality['tier_num'] == 1])
    tier2_count = len(high_quality[high_quality['tier_num'] == 2])
    
    print(f"\nHigh-Quality Breakdown:")
    print(f"  Tier 1 (Excellent): {tier1_count:,} matches")
    print(f"  Tier 2 (Good): {tier2_count:,} matches")
    
    avg_coverage = high_quality['total_coverage'].mean()
    print(f"\nAverage coverage in high-quality matches: {avg_coverage:.1f}%")
    
    return high_quality

def save_match_scores(df_all, df_quality):
    """Save match quality scores"""
    print("\nStep 5: Saving match quality scores...")
    print("-"*70)
    
    os.makedirs('../processed_data', exist_ok=True)
    os.makedirs('../data', exist_ok=True)
    
    # 1. Save all match scores
    all_scores_path = '../processed_data/all_match_quality_scores.csv'
    df_all.to_csv(all_scores_path, index=False)
    print(f"Saved: {all_scores_path}")
    print(f"  All matches: {len(df_all):,}")
    
    # 2. Save high-quality matches only
    quality_scores_path = '../processed_data/high_quality_matches.csv'
    df_quality.to_csv(quality_scores_path, index=False)
    print(f"Saved: {quality_scores_path}")
    print(f"  High-quality matches: {len(df_quality):,}")
    
    # 3. Save match ID list for easy filtering
    match_ids = df_quality['match_id'].tolist()
    match_ids_path = '../data/high_quality_match_ids.json'
    with open(match_ids_path, 'w') as f:
        json.dump(match_ids, f, indent=2)
    print(f"Saved: {match_ids_path}")
    print(f"  Match IDs: {len(match_ids):,}")
    
    # 4. Save summary statistics
    summary = {
        'total_matches_processed': len(df_all),
        'high_quality_matches': len(df_quality),
        'retention_rate': round(len(df_quality) / len(df_all) * 100, 1),
        'tier_1_matches': len(df_quality[df_quality['tier_num'] == 1]),
        'tier_2_matches': len(df_quality[df_quality['tier_num'] == 2]),
        'average_coverage': round(df_quality['total_coverage'].mean(), 1),
        'date_range': {
            'earliest': str(df_quality['date'].min()),
            'latest': str(df_quality['date'].max())
        }
    }
    
    summary_path = '../data/match_quality_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

def display_sample_matches(df_quality):
    """Display sample high-quality matches"""
    print("\nStep 6: Sample High-Quality Matches")
    print("-"*70)
    
    # Get diverse samples
    tier1_samples = df_quality[df_quality['tier_num'] == 1].head(5)
    tier2_samples = df_quality[df_quality['tier_num'] == 2].head(5)
    
    print(f"\nTier 1 Matches (Excellent Coverage):\n")
    for _, match in tier1_samples.iterrows():
        print(f"Match {match['match_id']}: {match['team_a']} vs {match['team_b']}")
        print(f"  Coverage: {match['total_coverage']}% | Venue: {match['venue']}")
        print(f"  Known players: {match['known_players']}/{match['total_players']}")
        print()
    
    print(f"Tier 2 Matches (Good Coverage):\n")
    for _, match in tier2_samples.iterrows():
        print(f"Match {match['match_id']}: {match['team_a']} vs {match['team_b']}")
        print(f"  Coverage: {match['total_coverage']}% | Venue: {match['venue']}")
        print(f"  Known players: {match['known_players']}/{match['total_players']}")
        print()

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ODI MATCH QUALITY SCORER")
    print("="*70)
    print("\nObjective: Score all matches for player coverage")
    print("Strategy: Keep matches with >= 65% known players")
    print("Goal: Build training dataset from high-quality matches")
    
    # Load player database
    player_database, known_players = load_player_database()
    
    # Score all matches
    df_all = score_all_matches(known_players)
    
    # Analyze distribution
    analyze_match_quality(df_all)
    
    # Filter for quality
    df_quality = filter_quality_matches(df_all)
    
    # Save results
    save_match_scores(df_all, df_quality)
    
    # Display samples
    display_sample_matches(df_quality)
    
    # Final summary
    print("\n" + "="*70)
    print("MATCH QUALITY SCORING COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total matches scored: {len(df_all):,}")
    print(f"  High-quality matches: {len(df_quality):,}")
    print(f"  Average coverage: {df_quality['total_coverage'].mean():.1f}%")
    print(f"\nOutput files:")
    print(f"  ODI/processed_data/all_match_quality_scores.csv")
    print(f"  ODI/processed_data/high_quality_matches.csv")
    print(f"  ODI/data/high_quality_match_ids.json")
    print(f"  ODI/data/match_quality_summary.json")
    print(f"\nNext step: Run '3_build_training_dataset.py'")
    print(f"Expected training rows: ~{len(df_quality) * 2:,} (2 per match)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

