#!/usr/bin/env python3
"""
Build Player Impact Dataset
Extract individual player performance metrics and create player impact features
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PlayerImpactDatasetBuilder:
    def __init__(self):
        self.ball_by_ball_dir = "raw_data/t20 matches ball by ball"
        self.player_stats_dir = "raw_data/PlayerStats"
        self.output_dir = "processed_data"
        
        # Load player stats
        self.batting_stats = pd.read_csv(f"{self.player_stats_dir}/t20_batting.csv")
        self.bowling_stats = pd.read_csv(f"{self.player_stats_dir}/t20_bowling.csv")
        self.all_round_stats = pd.read_csv(f"{self.player_stats_dir}/t20_all_round.csv")
        
        # Load lookup tables
        self.team_lookup = pd.read_csv("data/team_lookup.csv")
        self.venue_lookup = pd.read_csv("data/venue_lookup.csv")
        self.player_lookup = pd.read_csv("data/player_lookup.csv")
        
        print("Loaded player stats and lookup tables")
        
    def get_player_career_stats(self, player_id):
        """Get comprehensive career stats for a player"""
        player_id_str = str(player_id)
        
        # Get batting stats
        batting = self.batting_stats[self.batting_stats['id'].astype(str) == player_id_str]
        bowling = self.bowling_stats[self.bowling_stats['id'].astype(str) == player_id_str]
        all_round = self.all_round_stats[self.all_round_stats['id'].astype(str) == player_id_str]
        
        stats = {
            'player_id': player_id,
            'batting_avg': batting['average_score'].iloc[0] if not batting.empty and not pd.isna(batting['average_score'].iloc[0]) else 20.0,
            'batting_sr': batting['strike_rate'].iloc[0] if not batting.empty and not pd.isna(batting['strike_rate'].iloc[0]) else 120.0,
            'batting_matches': batting['matches'].iloc[0] if not batting.empty and not pd.isna(batting['matches'].iloc[0]) else 0,
            'batting_runs': batting['runs'].iloc[0] if not batting.empty and not pd.isna(batting['runs'].iloc[0]) else 0,
            'batting_50s': batting['50'].iloc[0] if not batting.empty and not pd.isna(batting['50'].iloc[0]) else 0,
            'batting_100s': batting['100s'].iloc[0] if not batting.empty and not pd.isna(batting['100s'].iloc[0]) else 0,
            'batting_4s': batting['4s'].iloc[0] if not batting.empty and not pd.isna(batting['4s'].iloc[0]) else 0,
            'batting_6s': batting['6s'].iloc[0] if not batting.empty and not pd.isna(batting['6s'].iloc[0]) else 0,
            
            'bowling_avg': bowling['bwa'].iloc[0] if not bowling.empty and not pd.isna(bowling['bwa'].iloc[0]) else 30.0,
            'bowling_econ': bowling['bwe'].iloc[0] if not bowling.empty and not pd.isna(bowling['bwe'].iloc[0]) else 8.0,
            'bowling_sr': bowling['bwsr'].iloc[0] if not bowling.empty and not pd.isna(bowling['bwsr'].iloc[0]) else 30.0,
            'bowling_matches': bowling['mt'].iloc[0] if not bowling.empty and not pd.isna(bowling['mt'].iloc[0]) else 0,
            'bowling_wickets': bowling['wk'].iloc[0] if not bowling.empty and not pd.isna(bowling['wk'].iloc[0]) else 0,
            'bowling_overs': bowling['ov'].iloc[0] if not bowling.empty and not pd.isna(bowling['ov'].iloc[0]) else 0,
            
            'all_round_matches': all_round['ct'].iloc[0] if not all_round.empty and not pd.isna(all_round['ct'].iloc[0]) else 0,
            'all_round_runs': all_round['rn'].iloc[0] if not all_round.empty and not pd.isna(all_round['rn'].iloc[0]) else 0,
            'all_round_wickets': all_round['wk'].iloc[0] if not all_round.empty and not pd.isna(all_round['wk'].iloc[0]) else 0,
        }
        
        # Calculate player impact score
        stats['player_impact_score'] = self.calculate_player_impact_score(stats)
        
        # Determine player role
        stats['player_role'] = self.determine_player_role(stats)
        
        return stats
    
    def calculate_player_impact_score(self, stats):
        """Calculate overall player impact score"""
        # Weighted combination of batting and bowling performance
        batting_score = (
            stats['batting_avg'] * 0.4 +
            stats['batting_sr'] * 0.3 +
            stats['batting_50s'] * 2 +
            stats['batting_100s'] * 5 +
            stats['batting_4s'] * 0.1 +
            stats['batting_6s'] * 0.5
        )
        
        bowling_score = (
            (50 - stats['bowling_avg']) * 0.4 +  # Lower avg is better
            (10 - stats['bowling_econ']) * 0.3 +  # Lower econ is better
            stats['bowling_wickets'] * 2 +
            (50 - stats['bowling_sr']) * 0.1  # Lower SR is better
        )
        
        # Experience factor
        experience_factor = min(stats['batting_matches'] + stats['bowling_matches'], 100) / 100
        
        # Combined impact score
        impact_score = (batting_score * 0.6 + bowling_score * 0.4) * experience_factor
        
        return max(impact_score, 0)
    
    def determine_player_role(self, stats):
        """Determine player role based on stats"""
        batting_impact = stats['batting_avg'] * stats['batting_sr'] / 100
        bowling_impact = (50 - stats['bowling_avg']) * (10 - stats['bowling_econ'])
        
        if batting_impact > 30 and bowling_impact > 20:
            return 'all_rounder'
        elif batting_impact > 30:
            return 'batsman'
        elif bowling_impact > 20:
            return 'bowler'
        else:
            return 'utility'
    
    def extract_match_player_performance(self, match_file):
        """Extract player performance from a single match"""
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)
        except:
            return None
        
        match_id = os.path.basename(match_file).replace('.json', '')
        info = match_data.get('info', {})
        innings = match_data.get('innings', [])
        
        # Get match basic info
        match_info = {
            'match_id': match_id,
            'date': info.get('dates', [None])[0],
            'venue': info.get('venue', 'Unknown'),
            'teams': info.get('teams', []),
            'toss_winner': info.get('toss', {}).get('winner'),
            'toss_decision': info.get('toss', {}).get('decision'),
            'match_winner': info.get('outcome', {}).get('winner'),
        }
        
        # Extract player performance for each team
        team_performances = {}
        
        for i, inning in enumerate(innings):
            team = inning.get('team', f'Team_{i+1}')
            overs = inning.get('overs', [])
            
            # Track player performance
            player_performance = {}
            
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                for delivery in deliveries:
                    batter = delivery.get('batter')
                    bowler = delivery.get('bowler')
                    runs = delivery.get('runs', {})
                    
                    # Track batting performance
                    if batter:
                        if batter not in player_performance:
                            player_performance[batter] = {
                                'runs': 0, 'balls': 0, 'wickets': 0, '4s': 0, '6s': 0
                            }
                        player_performance[batter]['runs'] += runs.get('batter', 0)
                        player_performance[batter]['balls'] += 1
                        if runs.get('batter', 0) == 4:
                            player_performance[batter]['4s'] += 1
                        elif runs.get('batter', 0) == 6:
                            player_performance[batter]['6s'] += 1
                    
                    # Track bowling performance
                    if bowler:
                        if bowler not in player_performance:
                            player_performance[bowler] = {
                                'runs': 0, 'balls': 0, 'wickets': 0, '4s': 0, '6s': 0
                            }
                        player_performance[bowler]['runs'] += runs.get('total', 0)
                        player_performance[bowler]['balls'] += 1
                        
                        if 'wickets' in delivery:
                            player_performance[bowler]['wickets'] += len(delivery['wickets'])
            
            team_performances[team] = player_performance
        
        return match_info, team_performances
    
    def build_player_impact_dataset(self):
        """Build the complete player impact dataset"""
        print("Building Player Impact Dataset...")
        print("=" * 50)
        
        # Get all match files
        json_files = glob.glob(f"{self.ball_by_ball_dir}/*.json")
        print(f"Found {len(json_files)} match files")
        
        player_impact_data = []
        processed_count = 0
        
        for i, match_file in enumerate(json_files):
            if i % 100 == 0:
                print(f"Processing {i}/{len(json_files)} matches...")
            
            try:
                match_info, team_performances = self.extract_match_player_performance(match_file)
                if not match_info or not team_performances:
                    continue
                
                # Process each team's performance
                for team, players in team_performances.items():
                    for player_name, performance in players.items():
                        # Get player ID from lookup
                        player_id = self.get_player_id_from_name(player_name)
                        if not player_id:
                            continue
                        
                        # Get career stats
                        career_stats = self.get_player_career_stats(player_id)
                        
                        # Create player impact record
                        impact_record = {
                            'match_id': match_info['match_id'],
                            'date': match_info['date'],
                            'venue': match_info['venue'],
                            'team': team,
                            'player_id': player_id,
                            'player_name': player_name,
                            
                            # Match performance
                            'match_runs': performance['runs'],
                            'match_balls': performance['balls'],
                            'match_wickets': performance['wickets'],
                            'match_4s': performance['4s'],
                            'match_6s': performance['6s'],
                            'match_strike_rate': (performance['runs'] / performance['balls'] * 100) if performance['balls'] > 0 else 0,
                            
                            # Career stats
                            'career_batting_avg': career_stats['batting_avg'],
                            'career_batting_sr': career_stats['batting_sr'],
                            'career_bowling_avg': career_stats['bowling_avg'],
                            'career_bowling_econ': career_stats['bowling_econ'],
                            'career_matches': career_stats['batting_matches'] + career_stats['bowling_matches'],
                            'career_runs': career_stats['batting_runs'],
                            'career_wickets': career_stats['bowling_wickets'],
                            
                            # Impact metrics
                            'player_impact_score': career_stats['player_impact_score'],
                            'player_role': career_stats['player_role'],
                            
                            # Match context
                            'toss_winner': match_info['toss_winner'],
                            'toss_decision': match_info['toss_decision'],
                            'match_winner': match_info['match_winner'],
                        }
                        
                        player_impact_data.append(impact_record)
                        processed_count += 1
                        
            except Exception as e:
                print(f"Error processing {match_file}: {e}")
                continue
        
        print(f"Processed {processed_count} player performances")
        
        # Create DataFrame
        df = pd.DataFrame(player_impact_data)
        
        if df.empty:
            print("No data processed!")
            return None
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Save dataset
        output_file = f"{self.output_dir}/player_impact_dataset.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Player Impact Dataset created: {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    
    def get_player_id_from_name(self, player_name):
        """Get player ID from player name"""
        # Try to find in player lookup
        player_match = self.player_lookup[
            self.player_lookup['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if not player_match.empty:
            return player_match['player_id'].iloc[0]
        
        # Try to find in batting stats
        batting_match = self.batting_stats[
            self.batting_stats['id'].astype(str).str.contains(player_name, case=False, na=False)
        ]
        
        if not batting_match.empty:
            return batting_match['id'].iloc[0]
        
        return None
    
    def add_derived_features(self, df):
        """Add derived features to the dataset"""
        print("Adding derived features...")
        
        # Performance ratios
        df['performance_ratio'] = df['match_runs'] / (df['career_batting_avg'] + 1)
        df['strike_rate_ratio'] = df['match_strike_rate'] / (df['career_batting_sr'] + 1)
        
        # Experience levels
        df['experience_level'] = pd.cut(df['career_matches'], 
                                       bins=[0, 10, 50, 100, 1000], 
                                       labels=['rookie', 'developing', 'experienced', 'veteran'])
        
        # Performance categories
        df['high_performer'] = (df['match_runs'] > df['career_batting_avg']).astype(int)
        df['consistent_performer'] = (df['performance_ratio'] > 0.8).astype(int)
        
        # Role-based features
        df['is_batsman'] = (df['player_role'] == 'batsman').astype(int)
        df['is_bowler'] = (df['player_role'] == 'bowler').astype(int)
        df['is_all_rounder'] = (df['player_role'] == 'all_rounder').astype(int)
        
        return df

def main():
    """Main function to build player impact dataset"""
    print("Building Player Impact Dataset")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Initialize builder
    builder = PlayerImpactDatasetBuilder()
    
    # Build dataset
    df = builder.build_player_impact_dataset()
    
    if df is not None:
        print("\nPlayer Impact Dataset created successfully!")
        print(f"Dataset contains {len(df)} player performances")
        print(f"Dataset has {len(df.columns)} features")
        print(f"Saved to: processed_data/player_impact_dataset.csv")
        
        # Show sample
        print("\nSample data:")
        print(df.head(3).to_string())
    else:
        print("Failed to create player impact dataset")

if __name__ == "__main__":
    main()
