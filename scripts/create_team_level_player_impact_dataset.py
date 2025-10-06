#!/usr/bin/env python3
"""
Create Team-Level Player Impact Dataset
Convert individual player performances to team-level features for ML training
Fix match_id issues and create proper numerical features
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TeamLevelPlayerImpactBuilder:
    def __init__(self):
        self.ball_by_ball_dir = "raw_data/t20 matches ball by ball"
        self.output_dir = "processed_data"
        
        # Load enhanced player impact dataset
        self.player_impact_df = pd.read_csv("processed_data/enhanced_player_impact_dataset.csv")
        print(f"Loaded player impact dataset: {len(self.player_impact_df)} records")
        
        # Load combined player lookup for additional info
        self.combined_lookup = pd.read_csv("data/combined_player_lookup.csv")
        
        # Initialize storage
        self.team_level_data = []
        self.match_info_cache = {}
        
        print("Team-Level Player Impact Builder initialized")
        
    def extract_match_info_from_json(self, match_id):
        """Extract match information from JSON files to fix match_id issues"""
        if match_id in self.match_info_cache:
            return self.match_info_cache[match_id]
        
        # Try to find the match file
        match_files = glob.glob(f"{self.ball_by_ball_dir}/*{match_id}*.json")
        if not match_files:
            return None
        
        try:
            with open(match_files[0], 'r') as f:
                match_data = json.load(f)
            
            info = match_data.get('info', {})
            
            match_info = {
                'match_id': match_id,
                'date': info.get('dates', [''])[0] if info.get('dates') else '',
                'venue': info.get('venue', 'Unknown'),
                'teams': info.get('teams', []),
                'gender': info.get('gender', 'male'),
                'match_type': info.get('match_type', 'T20'),
                'toss_winner': info.get('toss', {}).get('winner', ''),
                'toss_decision': info.get('toss', {}).get('decision', ''),
                'player_of_match': info.get('player_of_match', ''),
                'season': info.get('season', ''),
                'event_name': info.get('event', {}).get('name', '') if info.get('event') else ''
            }
            
            # Get match outcome
            outcome = info.get('outcome', {})
            match_info['match_winner'] = outcome.get('winner', '')
            match_info['win_margin'] = outcome.get('by', {}).get('runs', 0) if outcome.get('by', {}).get('runs') else 0
            match_info['win_type'] = 'runs' if outcome.get('by', {}).get('runs') else 'wickets'
            
            self.match_info_cache[match_id] = match_info
            return match_info
            
        except Exception as e:
            print(f"Error processing match {match_id}: {str(e)}")
            return None
    
    def create_team_features(self, team_performances, match_info, team_name, opposition):
        """Create team-level features from individual player performances"""
        
        if team_performances.empty:
            return None
        
        # Basic team statistics
        total_players = len(team_performances)
        total_runs = team_performances['match_runs'].sum()
        total_balls = team_performances['match_balls'].sum()
        total_wickets = team_performances['match_wickets'].sum()
        total_4s = team_performances['match_4s'].sum()
        total_6s = team_performances['match_6s'].sum()
        
        # Calculate team strike rate
        team_strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
        
        # Star players analysis
        star_players = team_performances[team_performances['has_meaningful_career'] == True]
        regular_players = team_performances[team_performances['has_meaningful_career'] == False]
        
        star_count = len(star_players)
        regular_count = len(regular_players)
        star_ratio = star_count / total_players if total_players > 0 else 0
        
        # Star player performance
        star_runs = star_players['match_runs'].sum()
        star_wickets = star_players['match_wickets'].sum()
        star_impact = star_players['player_impact_score'].sum()
        
        # Team composition analysis
        batsmen_count = team_performances['is_batsman'].sum()
        bowlers_count = team_performances['is_bowler'].sum()
        all_rounders_count = team_performances['is_all_rounder'].sum()
        
        batting_strength = batsmen_count / total_players if total_players > 0 else 0
        bowling_strength = bowlers_count / total_players if total_players > 0 else 0
        all_rounder_ratio = all_rounders_count / total_players if total_players > 0 else 0
        
        # Impact score statistics
        impact_mean = team_performances['player_impact_score'].mean()
        impact_std = team_performances['player_impact_score'].std()
        impact_max = team_performances['player_impact_score'].max()
        impact_min = team_performances['player_impact_score'].min()
        
        # Performance consistency
        high_performers = team_performances['high_performer'].sum()
        consistent_performers = team_performances['consistent_performer'].sum()
        
        # Career experience (for star players)
        avg_career_matches = star_players['career_matches'].mean() if len(star_players) > 0 else 0
        total_career_matches = star_players['career_matches'].sum()
        
        # Batting performance ratios (for star players)
        avg_batting_performance_ratio = star_players['batting_performance_ratio'].mean() if len(star_players) > 0 else 1.0
        avg_strike_rate_ratio = star_players['strike_rate_ratio'].mean() if len(star_players) > 0 else 1.0
        
        # Create team-level record
        team_record = {
            # Match identification
            'match_id': match_info.get('match_id', 'unknown'),
            'date': match_info.get('date', ''),
            'venue': match_info.get('venue', ''),
            'team': team_name,
            'opposition': opposition,
            'total_runs': total_runs,
            
            # Match context
            'toss_winner': match_info.get('toss_winner', ''),
            'toss_decision': match_info.get('toss_decision', ''),
            'match_winner': match_info.get('match_winner', ''),
            'win_margin': match_info.get('win_margin', 0),
            'win_type': match_info.get('win_type', ''),
            'player_of_match': match_info.get('player_of_match', ''),
            'season': match_info.get('season', ''),
            'event_name': match_info.get('event_name', ''),
            
            # Team composition (numerical features)
            'team_size': total_players,
            'star_players_count': star_count,
            'regular_players_count': regular_count,
            'star_ratio': star_ratio,
            'batting_strength': batting_strength,
            'bowling_strength': bowling_strength,
            'all_rounder_ratio': all_rounder_ratio,
            
            # Performance metrics
            'total_balls': total_balls,
            'total_wickets': total_wickets,
            'total_4s': total_4s,
            'total_6s': total_6s,
            'team_strike_rate': team_strike_rate,
            
            # Star player impact
            'star_runs': star_runs,
            'star_wickets': star_wickets,
            'star_impact_total': star_impact,
            'star_impact_ratio': star_impact / total_runs if total_runs > 0 else 0,
            
            # Team impact statistics
            'impact_mean': impact_mean,
            'impact_std': impact_std,
            'impact_max': impact_max,
            'impact_min': impact_min,
            'impact_range': impact_max - impact_min,
            
            # Performance indicators
            'high_performers_count': high_performers,
            'consistent_performers_count': consistent_performers,
            'performance_consistency': consistent_performers / total_players if total_players > 0 else 0,
            
            # Career experience
            'avg_career_matches': avg_career_matches,
            'total_career_matches': total_career_matches,
            'experience_depth': total_career_matches / total_players if total_players > 0 else 0,
            
            # Star player performance ratios
            'avg_batting_performance_ratio': avg_batting_performance_ratio,
            'avg_strike_rate_ratio': avg_strike_rate_ratio,
            
            # Derived features for ML
            'is_strong_team': 1 if star_ratio > 0.3 and avg_career_matches > 50 else 0,
            'is_balanced_team': 1 if 0.3 <= batting_strength <= 0.7 and 0.3 <= bowling_strength <= 0.7 else 0,
            'is_high_impact_team': 1 if impact_mean > 15 and impact_std < 10 else 0,
            'has_star_batsman': 1 if star_runs > 50 else 0,
            'has_star_bowler': 1 if star_wickets > 2 else 0,
        }
        
        return team_record
    
    def build_team_level_dataset(self):
        """Build team-level dataset from individual player performances"""
        print("Building Team-Level Player Impact Dataset...")
        print("=" * 50)
        
        # Group by match and team
        grouped = self.player_impact_df.groupby(['match_id', 'team'])
        
        print(f"Processing {len(grouped)} team performances...")
        
        processed_matches = 0
        
        for (match_id, team), team_performances in grouped:
            if processed_matches % 1000 == 0:
                print(f"Processed {processed_matches}/{len(grouped)} team performances...")
            
            # Get opposition team
            match_teams = team_performances['opposition'].unique()
            if len(match_teams) > 0:
                opposition = match_teams[0]
            else:
                opposition = 'Unknown'
            
            # Extract match info from JSON files
            match_info = self.extract_match_info_from_json(match_id)
            if not match_info:
                # Create basic match info if JSON not found
                match_info = {
                    'match_id': match_id,
                    'date': team_performances['date'].iloc[0] if len(team_performances) > 0 else '',
                    'venue': team_performances['venue'].iloc[0] if len(team_performances) > 0 else '',
                    'teams': [team, opposition],
                    'toss_winner': '',
                    'toss_decision': '',
                    'match_winner': '',
                    'win_margin': 0,
                    'win_type': '',
                    'player_of_match': '',
                    'season': '',
                    'event_name': ''
                }
            
            # Create team-level features
            team_record = self.create_team_features(team_performances, match_info, team, opposition)
            
            if team_record:
                self.team_level_data.append(team_record)
            
            processed_matches += 1
        
        print(f"Processing completed!")
        
        # Create DataFrame
        df = pd.DataFrame(self.team_level_data)
        
        if df.empty:
            print("No data generated!")
            return None
        
        print(f"Generated {len(df)} team-level records")
        
        # Save dataset
        output_path = f"{self.output_dir}/team_level_player_impact_dataset.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved team-level dataset: {output_path}")
        
        # Save summary statistics
        self.save_summary_statistics(df)
        
        return df
    
    def save_summary_statistics(self, df):
        """Save summary statistics"""
        summary = {
            'total_records': len(df),
            'unique_matches': df['match_id'].nunique(),
            'unique_teams': df['team'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'average_runs': df['total_runs'].mean(),
            'average_team_size': df['team_size'].mean(),
            'average_star_ratio': df['star_ratio'].mean(),
            'strong_teams_count': df['is_strong_team'].sum(),
            'balanced_teams_count': df['is_balanced_team'].sum(),
            'high_impact_teams_count': df['is_high_impact_team'].sum(),
            'feature_columns': list(df.columns),
            'numerical_features': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(df.select_dtypes(include=['object']).columns)
        }
        
        # Save as JSON
        import json
        with open(f"{self.output_dir}/team_level_player_impact_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary statistics saved: {self.output_dir}/team_level_player_impact_summary.json")
        
        # Print key statistics
        print(f"\n=== TEAM-LEVEL PLAYER IMPACT DATASET SUMMARY ===")
        print(f"Total team records: {summary['total_records']:,}")
        print(f"Unique matches: {summary['unique_matches']:,}")
        print(f"Unique teams: {summary['unique_teams']:,}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Average runs per team: {summary['average_runs']:.1f}")
        print(f"Average team size: {summary['average_team_size']:.1f}")
        print(f"Average star player ratio: {summary['average_star_ratio']:.2f}")
        print(f"Strong teams: {summary['strong_teams_count']:,}")
        print(f"Balanced teams: {summary['balanced_teams_count']:,}")
        print(f"High impact teams: {summary['high_impact_teams_count']:,}")
        print(f"Numerical features: {len(summary['numerical_features'])}")
        print(f"Categorical features: {len(summary['categorical_features'])}")

def main():
    print("Team-Level Player Impact Dataset Builder")
    print("Converting individual player performances to team-level ML features")
    print("=" * 70)
    
    builder = TeamLevelPlayerImpactBuilder()
    dataset = builder.build_team_level_dataset()
    
    if dataset is not None:
        print(f"\nTeam-level player impact dataset created successfully!")
        print(f"Ready for ML model training with proper numerical features!")
        print(f"Dataset can now be used to train XGBoost, Random Forest, and DNN models!")
    else:
        print(f"Failed to create dataset!")

if __name__ == "__main__":
    main()
