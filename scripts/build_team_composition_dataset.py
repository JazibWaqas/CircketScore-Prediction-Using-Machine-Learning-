#!/usr/bin/env python3
"""
Build Team Composition Dataset
Extract team composition, chemistry, and strategic features
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TeamCompositionDatasetBuilder:
    def __init__(self):
        self.ball_by_ball_dir = "raw_data/t20 matches ball by ball"
        self.output_dir = "processed_data"
        
        # Load lookup tables
        self.team_lookup = pd.read_csv("data/team_lookup.csv")
        self.player_lookup = pd.read_csv("data/player_lookup.csv")
        
        print("Loaded team and player lookup tables")
        
    def analyze_team_composition(self, team_players, match_data):
        """Analyze team composition and chemistry"""
        if not team_players or not match_data:
            return self.get_default_composition()
        
        # Get player roles and stats
        player_roles = []
        player_impacts = []
        batting_strength = 0
        bowling_strength = 0
        all_rounders = 0
        
        for player in team_players:
            # Get player role (simplified)
            role = self.get_player_role(player)
            player_roles.append(role)
            
            # Get player impact
            impact = self.get_player_impact(player)
            player_impacts.append(impact)
            
            # Categorize by role
            if role in ['opener', 'batsman', 'wicket_keeper']:
                batting_strength += impact
            elif role in ['bowler']:
                bowling_strength += impact
            else:
                all_rounders += 1
        
        # Calculate composition metrics
        total_players = len(team_players)
        batting_ratio = batting_strength / (batting_strength + bowling_strength + 1)
        bowling_ratio = bowling_strength / (batting_strength + bowling_strength + 1)
        all_rounder_ratio = all_rounders / total_players
        
        # Team balance
        team_balance = 1 - abs(batting_ratio - bowling_ratio)
        
        # Team depth
        team_depth = np.mean(player_impacts) if player_impacts else 0
        
        # Team variety
        role_variety = len(set(player_roles)) / total_players if total_players > 0 else 0
        
        return {
            'team_size': total_players,
            'batting_strength': batting_strength,
            'bowling_strength': bowling_strength,
            'all_rounders': all_rounders,
            'batting_ratio': batting_ratio,
            'bowling_ratio': bowling_ratio,
            'all_rounder_ratio': all_rounder_ratio,
            'team_balance': team_balance,
            'team_depth': team_depth,
            'role_variety': role_variety,
            'player_roles': player_roles,
            'player_impacts': player_impacts
        }
    
    def get_player_role(self, player_name):
        """Get player role based on name (simplified)"""
        name_lower = player_name.lower()
        
        # Wicket-keepers
        if any(term in name_lower for term in ['dhoni', 'pant', 'buttler', 'carey', 'rizwan', 'klaasen', 'de kock']):
            return 'wicket_keeper'
        
        # Bowlers
        if any(term in name_lower for term in ['bumrah', 'shami', 'starc', 'cummins', 'archer', 'rabada', 'afridi', 'rauf', 'ali', 'shah', 'lyon', 'rashid', 'chahal', 'jadeja', 'ashwin']):
            return 'bowler'
        
        # All-rounders
        if any(term in name_lower for term in ['jadeja', 'pandya', 'stokes', 'maxwell', 'shadab', 'nawaz', 'jansen', 'curran', 'woakes', 'ali']):
            return 'all_rounder'
        
        # Openers
        if any(term in name_lower for term in ['rohit', 'sharma', 'warner', 'smith', 'root', 'babar', 'azam', 'fakhar', 'zaman']):
            return 'opener'
        
        # Default
        return 'batsman'
    
    def get_player_impact(self, player_name):
        """Get player impact score (simplified)"""
        name_lower = player_name.lower()
        
        # Star players get higher impact
        if any(term in name_lower for term in ['kohli', 'rohit', 'sharma', 'dhoni', 'bumrah', 'pandya', 'jadeja', 'pant', 'warner', 'smith', 'root', 'babar', 'azam', 'stokes', 'maxwell']):
            return 8.0
        elif any(term in name_lower for term in ['buttler', 'carey', 'rizwan', 'klaasen', 'de kock', 'shami', 'starc', 'cummins', 'archer', 'rabada', 'afridi']):
            return 7.0
        else:
            return 5.0
    
    def get_default_composition(self):
        """Get default team composition"""
        return {
            'team_size': 11,
            'batting_strength': 5.0,
            'bowling_strength': 5.0,
            'all_rounders': 2,
            'batting_ratio': 0.5,
            'bowling_ratio': 0.5,
            'all_rounder_ratio': 0.2,
            'team_balance': 0.5,
            'team_depth': 5.0,
            'role_variety': 0.5,
            'player_roles': [],
            'player_impacts': []
        }
    
    def analyze_match_strategy(self, match_data):
        """Analyze match strategy and context"""
        info = match_data.get('info', {})
        innings = match_data.get('innings', [])
        
        # Toss analysis
        toss_winner = info.get('toss', {}).get('winner')
        toss_decision = info.get('toss', {}).get('decision')
        match_winner = info.get('outcome', {}).get('winner')
        
        # Calculate toss impact
        toss_impact = 0
        if toss_winner and match_winner:
            toss_impact = 1 if toss_winner == match_winner else -1
        
        # Analyze scoring patterns
        total_runs = 0
        total_overs = 0
        powerplay_runs = 0
        death_overs_runs = 0
        
        for inning in innings:
            overs = inning.get('overs', [])
            total_overs += len(overs)
            
            for i, over_data in enumerate(overs):
                over_runs = sum(delivery.get('runs', {}).get('total', 0) for delivery in over_data.get('deliveries', []))
                total_runs += over_runs
                
                # Categorize overs
                if i < 6:  # Powerplay
                    powerplay_runs += over_runs
                elif i >= 16:  # Death overs
                    death_overs_runs += over_runs
        
        # Calculate strategy metrics
        run_rate = total_runs / total_overs if total_overs > 0 else 7.5
        powerplay_ratio = powerplay_runs / (total_runs + 1)
        death_overs_ratio = death_overs_runs / (total_runs + 1)
        
        return {
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'toss_impact': toss_impact,
            'total_runs': total_runs,
            'total_overs': total_overs,
            'run_rate': run_rate,
            'powerplay_ratio': powerplay_ratio,
            'death_overs_ratio': death_overs_ratio,
            'match_winner': match_winner
        }
    
    def extract_team_composition(self, match_file):
        """Extract team composition from a single match"""
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)
        except:
            return None
        
        match_id = os.path.basename(match_file).replace('.json', '')
        info = match_data.get('info', {})
        innings = match_data.get('innings', [])
        
        # Get teams and players
        teams = info.get('teams', [])
        players = info.get('players', {})
        
        team_compositions = []
        
        # Analyze each team
        for team_name, team_players in players.items():
            if not team_players:
                continue
            
            # Analyze team composition
            composition = self.analyze_team_composition(team_players, match_data)
            
            # Analyze match strategy
            strategy = self.analyze_match_strategy(match_data)
            
            # Create team composition record
            composition_record = {
                'match_id': match_id,
                'date': info.get('dates', [None])[0],
                'venue': info.get('venue', 'Unknown'),
                'team': team_name,
                'opposition': [t for t in teams if t != team_name][0] if len(teams) > 1 else 'Unknown',
                
                # Team composition
                'team_size': composition['team_size'],
                'batting_strength': composition['batting_strength'],
                'bowling_strength': composition['bowling_strength'],
                'all_rounders': composition['all_rounders'],
                'batting_ratio': composition['batting_ratio'],
                'bowling_ratio': composition['bowling_ratio'],
                'all_rounder_ratio': composition['all_rounder_ratio'],
                'team_balance': composition['team_balance'],
                'team_depth': composition['team_depth'],
                'role_variety': composition['role_variety'],
                
                # Match strategy
                'toss_winner': strategy['toss_winner'],
                'toss_decision': strategy['toss_decision'],
                'toss_impact': strategy['toss_impact'],
                'total_runs': strategy['total_runs'],
                'total_overs': strategy['total_overs'],
                'run_rate': strategy['run_rate'],
                'powerplay_ratio': strategy['powerplay_ratio'],
                'death_overs_ratio': strategy['death_overs_ratio'],
                'match_winner': strategy['match_winner'],
                
                # Derived features
                'is_batting_first': team_name == strategy['toss_winner'] and strategy['toss_decision'] == 'bat',
                'is_chasing': team_name != strategy['toss_winner'] and strategy['toss_decision'] == 'bat',
                'team_chemistry': composition['team_balance'] * composition['team_depth'],
                'strategic_advantage': strategy['toss_impact'] * composition['team_balance'],
            }
            
            team_compositions.append(composition_record)
        
        return team_compositions
    
    def build_team_composition_dataset(self):
        """Build the complete team composition dataset"""
        print("Building Team Composition Dataset...")
        print("=" * 50)
        
        # Get all match files
        json_files = glob.glob(f"{self.ball_by_ball_dir}/*.json")
        print(f"Found {len(json_files)} match files")
        
        team_composition_data = []
        processed_count = 0
        
        for i, match_file in enumerate(json_files):
            if i % 100 == 0:
                print(f"Processing {i}/{len(json_files)} matches...")
            
            try:
                compositions = self.extract_team_composition(match_file)
                if compositions:
                    team_composition_data.extend(compositions)
                    processed_count += len(compositions)
                    
            except Exception as e:
                print(f"Error processing {match_file}: {e}")
                continue
        
        print(f"Processed {processed_count} team compositions")
        
        # Create DataFrame
        df = pd.DataFrame(team_composition_data)
        
        if df.empty:
            print("No data processed!")
            return None
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Save dataset
        output_file = f"{self.output_dir}/team_composition_dataset.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Team Composition Dataset created: {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    
    def add_derived_features(self, df):
        """Add derived features to the dataset"""
        print("Adding derived features...")
        
        # Team strength categories
        df['team_strength'] = pd.cut(df['team_depth'], 
                                    bins=[0, 3, 5, 7, 10], 
                                    labels=['weak', 'average', 'strong', 'elite'])
        
        # Balance categories
        df['balance_category'] = pd.cut(df['team_balance'], 
                                      bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                      labels=['unbalanced', 'poor', 'good', 'excellent'])
        
        # Strategy effectiveness
        df['strategy_effectiveness'] = (
            df['toss_impact'] * 0.3 +
            df['team_balance'] * 0.4 +
            df['team_depth'] * 0.3
        )
        
        # Match outcome prediction
        df['predicted_winner'] = (
            df['team_balance'] * 0.3 +
            df['team_depth'] * 0.3 +
            df['toss_impact'] * 0.2 +
            df['strategic_advantage'] * 0.2
        )
        
        return df

def main():
    """Main function to build team composition dataset"""
    print("Building Team Composition Dataset")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Initialize builder
    builder = TeamCompositionDatasetBuilder()
    
    # Build dataset
    df = builder.build_team_composition_dataset()
    
    if df is not None:
        print("\nTeam Composition Dataset created successfully!")
        print(f"Dataset contains {len(df)} team compositions")
        print(f"Dataset has {len(df.columns)} features")
        print(f"Saved to: processed_data/team_composition_dataset.csv")
        
        # Show sample
        print("\nSample data:")
        print(df.head(3).to_string())
    else:
        print("Failed to create team composition dataset")

if __name__ == "__main__":
    main()
