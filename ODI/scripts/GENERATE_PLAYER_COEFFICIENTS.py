#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERATE PLAYER IMPACT COEFFICIENTS

Calculates player impact coefficients from actual career statistics.
Data-driven, not hardcoded!

Formula:
- Batting impact: Based on batting avg vs ODI average (32), strike rate, experience
- Bowling impact: Based on economy vs ODI average (5.2), wicket-taking ability
"""

import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("GENERATE PLAYER IMPACT COEFFICIENTS")
print("="*80)

# ODI Cricket Averages (Statistical Baseline)
ODI_AVG_BATTING = 32.0
ODI_AVG_SR = 80.0
ODI_AVG_ECONOMY = 5.2
ODI_AVG_BOWLING_AVG = 35.0

def calculate_batting_impact(player_avg, player_sr, matches_played):
    """
    Calculate batting impact coefficient
    
    Formula:
    - Base impact from batting average difference
    - Strike rate adjustment
    - Experience reliability weight
    """
    if player_avg is None or player_avg <= 0:
        return 0.0
    
    # Base impact from batting average
    # Assumption: Top order batsman faces ~50-60 balls on average in ODI
    # Higher average = more runs scored
    avg_impact = (player_avg - ODI_AVG_BATTING) * 0.65  # 65% of difference translates to match impact
    
    # Strike rate bonus/penalty (smaller effect)
    sr_impact = (player_sr - ODI_AVG_SR) * 0.08 if player_sr else 0
    
    # Experience reliability weight
    # More matches = more reliable, cap at 1.5x for 150+ matches
    reliability = min(matches_played / 150, 1.5)
    reliability = max(reliability, 0.5)  # Minimum 0.5x for inexperienced players
    
    # Combine
    total_impact = (avg_impact + sr_impact) * reliability
    
    # Reasonable bounds for batting impact
    # Elite players can add up to 20 runs, weak players can reduce by up to 10
    return round(max(min(total_impact, 20), -10), 1)

def calculate_bowling_impact(player_econ, player_bowling_avg, total_wickets):
    """
    Calculate bowling impact coefficient
    
    Better bowler = opposition scores LESS (negative impact)
    """
    if player_econ is None or player_econ <= 0:
        return 0.0
    
    # Economy difference (lower economy = better bowler)
    # Each 0.1 economy difference = ~6 runs per match
    econ_impact = (ODI_AVG_ECONOMY - player_econ) * 6.0
    
    # Wicket-taking ability bonus
    # Bowlers with many wickets are more impactful
    wicket_bonus = min(total_wickets / 250, 1.2) * 2.5 if total_wickets else 0
    
    # Bowling average factor (lower is better)
    avg_factor = (ODI_AVG_BOWLING_AVG - player_bowling_avg) * 0.15 if player_bowling_avg and player_bowling_avg > 0 else 0
    
    # Combine (negative because good bowling reduces opposition score)
    total_impact = -(econ_impact + wicket_bonus + avg_factor)
    
    # Bounds: Elite bowlers reduce opp score by up to 15 runs
    return round(max(min(total_impact, 5), -15), 1)

def determine_tier(batting_impact, bowling_impact):
    """Determine player tier based on impact"""
    max_impact = max(abs(batting_impact), abs(bowling_impact))
    
    if max_impact >= 12:
        return 'elite'
    elif max_impact >= 7:
        return 'star'
    elif max_impact >= 3:
        return 'good'
    else:
        return 'regular'

# Load player database
print("\nLoading player database...")
with open('../data/player_database.json', 'r', encoding='utf-8') as f:
    player_db = json.load(f)

print(f"✓ Loaded {len(player_db):,} players\n")

# Calculate coefficients for all players
print("Calculating impact coefficients...\n")

player_coefficients = {}
stats_summary = {
    'elite': 0,
    'star': 0,
    'good': 0,
    'regular': 0,
    'batting_only': 0,
    'bowling_only': 0,
    'all_rounder': 0
}

for i, (player_name, stats) in enumerate(player_db.items()):
    try:
        coefficient = {
            'name': player_name,
            'role': stats.get('role', 'Unknown')
        }
        
        batting_impact = 0.0
        bowling_impact = 0.0
        
        # Batting coefficient
        if stats.get('batting'):
            batting_avg = stats['batting'].get('average', 0)
            strike_rate = stats['batting'].get('strike_rate', ODI_AVG_SR)
            matches = stats.get('total_matches', 0)
            
            batting_impact = calculate_batting_impact(batting_avg, strike_rate, matches)
            
            coefficient['batting_impact'] = batting_impact
            coefficient['batting_avg'] = batting_avg
            coefficient['strike_rate'] = strike_rate
        
        # Bowling coefficient
        if stats.get('bowling') and stats['bowling'].get('economy'):
            bowling_econ = stats['bowling']['economy']
            bowling_avg = stats['bowling'].get('average', ODI_AVG_BOWLING_AVG)
            total_wickets = stats['bowling'].get('total_wickets', 0)
            
            bowling_impact = calculate_bowling_impact(bowling_econ, bowling_avg, total_wickets)
            
            coefficient['bowling_impact'] = bowling_impact
            coefficient['bowling_economy'] = bowling_econ
            coefficient['bowling_avg'] = bowling_avg
        
        # Determine tier
        tier = determine_tier(batting_impact, bowling_impact)
        coefficient['tier'] = tier
        
        # Overall impact (for sorting)
        coefficient['overall_impact'] = max(abs(batting_impact), abs(bowling_impact))
        
        # Statistics
        stats_summary[tier] += 1
        if batting_impact != 0 and bowling_impact != 0:
            stats_summary['all_rounder'] += 1
        elif batting_impact != 0:
            stats_summary['batting_only'] += 1
        elif bowling_impact != 0:
            stats_summary['bowling_only'] += 1
        
        player_coefficients[player_name] = coefficient
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(player_db)} players...")
    
    except Exception as e:
        print(f"  ⚠ Error processing {player_name}: {e}")
        continue

print(f"\n✓ Calculated coefficients for {len(player_coefficients):,} players\n")

# Save coefficients
output_path = '../data/player_impact_coefficients.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(player_coefficients, f, indent=2)

print(f"✓ Saved: {output_path}\n")

# Summary statistics
print("="*80)
print("COEFFICIENT SUMMARY")
print("="*80)

print(f"\nPlayer Tiers:")
print(f"  Elite (±12+ runs):   {stats_summary['elite']:,} players")
print(f"  Star (±7-12 runs):   {stats_summary['star']:,} players")
print(f"  Good (±3-7 runs):    {stats_summary['good']:,} players")
print(f"  Regular (<±3 runs):  {stats_summary['regular']:,} players")

print(f"\nPlayer Types:")
print(f"  Batting specialists: {stats_summary['batting_only']:,}")
print(f"  Bowling specialists: {stats_summary['bowling_only']:,}")
print(f"  All-rounders:        {stats_summary['all_rounder']:,}")

# Show top players
print(f"\n🏆 TOP 10 BATSMEN (by impact):")
batsmen = [(name, coef) for name, coef in player_coefficients.items() 
           if coef.get('batting_impact', 0) > 0]
batsmen.sort(key=lambda x: x[1]['batting_impact'], reverse=True)

for i, (name, coef) in enumerate(batsmen[:10], 1):
    print(f"  {i:2d}. {name:30s} +{coef['batting_impact']:5.1f} runs (avg: {coef.get('batting_avg', 0):.1f})")

print(f"\n🏆 TOP 10 BOWLERS (by impact):")
bowlers = [(name, coef) for name, coef in player_coefficients.items() 
           if coef.get('bowling_impact', 0) < 0]
bowlers.sort(key=lambda x: x[1]['bowling_impact'])

for i, (name, coef) in enumerate(bowlers[:10], 1):
    print(f"  {i:2d}. {name:30s} {coef['bowling_impact']:5.1f} runs (econ: {coef.get('bowling_economy', 0):.2f})")

print("\n" + "="*80)
print("COEFFICIENT GENERATION COMPLETE!")
print("="*80)
print(f"\nNext steps:")
print(f"  1. Use player_impact_coefficients.json for predictions")
print(f"  2. Create prediction API with impact overlay")
print(f"  3. Test with real match scenarios")
print("="*80 + "\n")

