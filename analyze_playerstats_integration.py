"""
Analyze whether to integrate PlayerStats dataset with our current dataset
Compare current features vs what PlayerStats could add
"""

import pandas as pd
import numpy as np

def analyze_playerstats_integration():
    """Analyze if we need PlayerStats dataset for better predictions"""
    print("ANALYZING PLAYERSTATS INTEGRATION NEED")
    print("=" * 60)
    
    # Load current datasets
    train_df = pd.read_csv('train_dataset.csv')
    player_batting = pd.read_csv('PlayerStats/t20_batting.csv')
    player_bowling = pd.read_csv('PlayerStats/t20_bowling.csv')
    player_fielding = pd.read_csv('PlayerStats/fielding.csv')
    all_players = pd.read_csv('PlayerStats/all_players.csv')
    
    print("\n1. CURRENT DATASET FEATURES:")
    print("-" * 40)
    print("Team-level features we already have:")
    team_features = [
        'team_batting_avg', 'team_batting_std', 'team_form_score',
        'team_balance', 'pressure_score', 'team_player_ids'
    ]
    for feature in team_features:
        if feature in train_df.columns:
            print(f"  [YES] {feature}")
    
    print("\nVenue features we already have:")
    venue_features = [
        'venue_difficulty', 'venue_avg_runs', 'venue_runs_std',
        'venue_matches', 'venue_high_score', 'venue_low_score'
    ]
    for feature in venue_features:
        if feature in train_df.columns:
            print(f"  [YES] {feature}")
    
    print("\nMatch context features we already have:")
    context_features = [
        'toss_decision', 'batting_first', 'is_home_team',
        'is_final', 'is_semi_final', 'is_playoff'
    ]
    for feature in context_features:
        if feature in train_df.columns:
            print(f"  [YES] {feature}")
    
    print("\n2. PLAYERSTATS DATASET ANALYSIS:")
    print("-" * 40)
    print(f"Player batting stats: {len(player_batting)} records")
    print(f"Player bowling stats: {len(player_bowling)} records")
    print(f"Player fielding stats: {len(player_fielding)} records")
    print(f"All players: {len(all_players)} records")
    
    print("\nPlayer batting features available:")
    batting_features = [
        'matches', 'innings', 'runs', 'high_score', 'average_score',
        'strike_rate', '100s', '50', '0s', '4s', '6s'
    ]
    for feature in batting_features:
        if feature in player_batting.columns:
            print(f"  - {feature}")
    
    print("\nPlayer bowling features available:")
    bowling_features = [
        'ov', 'wk', 'pr', 'bwe', 'bwsr', 'md', 'mt'
    ]
    for feature in bowling_features:
        if feature in player_bowling.columns:
            print(f"  - {feature}")
    
    print("\nPlayer fielding features available:")
    fielding_features = [
        'cf', 'ck', 'ct', 'ds', 'dspi', 'st'
    ]
    for feature in fielding_features:
        if feature in player_fielding.columns:
            print(f"  - {feature}")
    
    print("\n3. CURRENT DATASET LIMITATIONS:")
    print("-" * 40)
    print("What we're missing:")
    print("  [NO] Individual player batting averages")
    print("  [NO] Individual player strike rates")
    print("  [NO] Individual player bowling averages")
    print("  [NO] Individual player economy rates")
    print("  [NO] Individual player fielding stats")
    print("  [NO] Player-specific venue performance")
    print("  [NO] Player-specific opposition performance")
    print("  [NO] Player form and recent performance")
    
    print("\n4. WHAT PLAYERSTATS COULD ADD:")
    print("-" * 40)
    print("Enhanced team composition analysis:")
    print("  [YES] Individual player batting averages")
    print("  [YES] Individual player strike rates")
    print("  [YES] Individual player bowling averages")
    print("  [YES] Individual player economy rates")
    print("  [YES] Individual player fielding stats")
    print("  [YES] Player-specific strengths/weaknesses")
    
    print("\n5. INTEGRATION BENEFITS:")
    print("-" * 40)
    print("Better team strength calculation:")
    print("  - Weighted batting average based on individual players")
    print("  - Weighted bowling average based on individual players")
    print("  - Team balance based on player roles")
    print("  - Player-specific venue performance")
    print("  - Player-specific opposition performance")
    
    print("\n6. INTEGRATION CHALLENGES:")
    print("-" * 40)
    print("Data linking issues:")
    print("  - Player IDs need to match between datasets")
    print("  - Player names might have variations")
    print("  - Some players might not have stats")
    print("  - Stats might be outdated or incomplete")
    
    print("\n7. RECOMMENDATION:")
    print("-" * 40)
    print("CURRENT DATASET IS SUFFICIENT FOR:")
    print("  [YES] Basic team vs team predictions")
    print("  [YES] Venue-based predictions")
    print("  [YES] Context-based predictions")
    print("  [YES] Historical performance predictions")
    
    print("\nPLAYERSTATS WOULD ENHANCE:")
    print("  [YES] Individual player impact analysis")
    print("  [YES] Player-specific predictions")
    print("  [YES] More accurate team strength calculation")
    print("  [YES] Player substitution impact")
    print("  [YES] Player-specific venue performance")
    
    print("\n8. DECISION FACTORS:")
    print("-" * 40)
    print("Use current dataset if:")
    print("  - You want to start with basic predictions")
    print("  - You want to test the core concept")
    print("  - You want to avoid data integration complexity")
    print("  - You want to focus on team-level analysis")
    
    print("\nIntegrate PlayerStats if:")
    print("  - You want individual player impact analysis")
    print("  - You want more accurate predictions")
    print("  - You want player-specific insights")
    print("  - You want to handle player substitutions")
    print("  - You want to analyze player combinations")
    
    print("\n9. HYBRID APPROACH:")
    print("-" * 40)
    print("Start with current dataset:")
    print("  1. Train models on current features")
    print("  2. Test model performance")
    print("  3. Identify prediction gaps")
    print("  4. Integrate PlayerStats for specific improvements")
    
    print("\n10. FINAL RECOMMENDATION:")
    print("-" * 40)
    print("CURRENT DATASET IS ENOUGH TO START")
    print("  - You have 14,014 records with rich features")
    print("  - You have team-level performance data")
    print("  - You have venue and context data")
    print("  - You can build working models")
    
    print("\nINTEGRATE PLAYERSTATS LATER")
    print("  - After you have working models")
    print("  - When you want individual player analysis")
    print("  - When you need more accurate predictions")
    print("  - When you want player-specific features")
    
    return train_df, player_batting, player_bowling, player_fielding, all_players

if __name__ == "__main__":
    train_df, player_batting, player_bowling, player_fielding, all_players = analyze_playerstats_integration()
