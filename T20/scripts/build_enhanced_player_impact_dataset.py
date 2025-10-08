#!/usr/bin/env python3
"""
Enhanced Player Impact Dataset Builder
Strategic approach focusing on star players with real career stats
No placeholder data - only meaningful impact calculations
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedPlayerImpactBuilder:
    def __init__(self):
        self.ball_by_ball_dir = "raw_data/t20 matches ball by ball"
        self.output_dir = "processed_data"
        
        # Load combined player lookup
        self.combined_lookup = pd.read_csv("data/combined_player_lookup.csv")
        print(f"Loaded combined player lookup: {len(self.combined_lookup)} players")
        
        # Load existing lookup tables
        self.team_lookup = pd.read_csv("data/team_lookup.csv")
        self.venue_lookup = pd.read_csv("data/venue_lookup.csv")
        
        # Initialize storage
        self.player_performance_data = []
        self.star_players_count = 0
        self.regular_players_count = 0
        
        print("Enhanced Player Impact Builder initialized")
        
    def is_star_player(self, player_id):
        """Determine if player has meaningful career stats (star player)"""
        player_info = self.combined_lookup[self.combined_lookup['ball_by_ball_id'] == player_id]
        
        if player_info.empty:
            return False, {}
        
        player = player_info.iloc[0]
        
        # Star player criteria: Has career stats AND meaningful performance
        has_career_stats = player['has_career_stats']
        
        if not has_career_stats:
            return False, {}
        
        # Check for meaningful career performance
        batting_runs = player.get('batting_runs', 0)
        batting_matches = player.get('batting_matches', 0)
        bowling_wickets = player.get('bowling_wickets', 0)
        bowling_matches = player.get('bowling_matches', 0)
        
        # Star player if they have significant batting OR bowling career
        is_star = (
            (batting_matches >= 20 and batting_runs >= 500) or  # Significant batting career
            (bowling_matches >= 20 and bowling_wickets >= 20) or  # Significant bowling career
            (batting_matches >= 10 and batting_runs >= 200 and batting_runs/batting_matches >= 20)  # High-impact batsman
        )
        
        return is_star, player.to_dict() if is_star else {}
    
    def calculate_star_player_impact(self, match_data, player_id, player_name, player_career_data):
        """Calculate impact score for star players based on career stats"""
        
        # Extract match performance
        match_runs = match_data.get('runs', 0)
        match_balls = match_data.get('balls', 0)
        match_wickets = match_data.get('wickets', 0)
        match_4s = match_data.get('4s', 0)
        match_6s = match_data.get('6s', 0)
        
        # Calculate match strike rate
        match_strike_rate = (match_runs / match_balls * 100) if match_balls > 0 else 0
        
        # Get career benchmarks
        career_batting_avg = player_career_data.get('batting_avg')
        career_batting_sr = player_career_data.get('batting_sr')
        career_bowling_avg = player_career_data.get('bowling_avg')
        career_bowling_econ = player_career_data.get('bowling_econ')
        career_matches = max(
            player_career_data.get('batting_matches', 0),
            player_career_data.get('bowling_matches', 0)
        )
        
        # Calculate performance ratios vs career norms
        batting_performance_ratio = 1.0
        strike_rate_ratio = 1.0
        
        if career_batting_avg and career_batting_avg > 0 and match_balls > 0:
            # Calculate expected runs based on career average
            expected_runs = (match_balls / 100) * career_batting_avg
            batting_performance_ratio = match_runs / expected_runs if expected_runs > 0 else 1.0
        
        if career_batting_sr and career_batting_sr > 0:
            strike_rate_ratio = match_strike_rate / career_batting_sr
        
        # Calculate player impact score (weighted combination)
        impact_score = 0
        
        # Batting impact
        if match_runs > 0:
            batting_impact = (
                (match_runs * 0.4) +  # Base runs value
                (match_4s * 2 * 0.2) +  # 4s bonus
                (match_6s * 4 * 0.2) +  # 6s bonus
                (batting_performance_ratio * 10 * 0.2)  # Performance vs career
            )
            impact_score += batting_impact
        
        # Bowling impact
        if match_wickets > 0:
            bowling_impact = (
                (match_wickets * 15) +  # Base wickets value
                (match_wickets * (1 / max(career_bowling_avg, 20)) * 10)  # Economy bonus
            )
            impact_score += bowling_impact
        
        # Experience and consistency factors
        experience_factor = min(career_matches / 50, 1.0)  # Max bonus at 50 matches
        consistency_bonus = min(impact_score * 0.1 * experience_factor, 5)  # Max 5 point bonus
        
        final_impact_score = impact_score + consistency_bonus
        
        return {
            'player_impact_score': final_impact_score,
            'batting_performance_ratio': batting_performance_ratio,
            'strike_rate_ratio': strike_rate_ratio,
            'experience_level': 'star',
            'high_performer': batting_performance_ratio > 1.5 or strike_rate_ratio > 1.3,
            'consistent_performer': batting_performance_ratio > 0.8 and batting_performance_ratio < 1.5,
            'career_batting_avg': career_batting_avg,
            'career_batting_sr': career_batting_sr,
            'career_bowling_avg': career_bowling_avg,
            'career_bowling_econ': career_bowling_econ,
            'career_matches': career_matches,
            'has_meaningful_career': True
        }
    
    def calculate_regular_player_impact(self, match_data, player_id, player_name):
        """Calculate impact for regular players based on recent performance only"""
        
        # Extract match performance
        match_runs = match_data.get('runs', 0)
        match_balls = match_data.get('balls', 0)
        match_wickets = match_data.get('wickets', 0)
        match_4s = match_data.get('4s', 0)
        match_6s = match_data.get('6s', 0)
        
        # Calculate match strike rate
        match_strike_rate = (match_runs / match_balls * 100) if match_balls > 0 else 0
        
        # For regular players, use T20 norms as benchmarks
        t20_avg_strike_rate = 130  # Typical T20 strike rate
        t20_avg_runs_per_match = 15  # Typical runs per match
        
        # Calculate impact based on T20 standards
        impact_score = 0
        
        # Batting impact (vs T20 norms)
        if match_runs > 0:
            expected_runs = (match_balls / 100) * t20_avg_runs_per_match
            performance_ratio = match_runs / expected_runs if expected_runs > 0 else 1.0
            
            batting_impact = (
                (match_runs * 0.4) +  # Base runs value
                (match_4s * 2 * 0.2) +  # 4s bonus
                (match_6s * 4 * 0.2) +  # 6s bonus
                (performance_ratio * 5 * 0.2)  # Performance vs T20 norm
            )
            impact_score += batting_impact
        
        # Bowling impact
        if match_wickets > 0:
            bowling_impact = match_wickets * 12  # Lower base value for regular players
            impact_score += bowling_impact
        
        # Strike rate performance vs T20 norm
        strike_rate_ratio = match_strike_rate / t20_avg_strike_rate if t20_avg_strike_rate > 0 else 1.0
        
        return {
            'player_impact_score': impact_score,
            'batting_performance_ratio': performance_ratio if match_runs > 0 else 1.0,
            'strike_rate_ratio': strike_rate_ratio,
            'experience_level': 'regular',
            'high_performer': match_runs > 30 or match_wickets > 2,
            'consistent_performer': False,  # Can't assess consistency without career data
            'career_batting_avg': None,  # No career data
            'career_batting_sr': None,
            'career_bowling_avg': None,
            'career_bowling_econ': None,
            'career_matches': 0,
            'has_meaningful_career': False
        }
    
    def determine_player_role(self, player_career_data, match_data):
        """Determine player role based on career stats and match performance"""
        
        if not player_career_data or not player_career_data.get('has_meaningful_career'):
            # For regular players, determine role from match performance
            match_runs = match_data.get('runs', 0)
            match_wickets = match_data.get('wickets', 0)
            match_balls = match_data.get('balls', 0)
            
            if match_runs > 20 and match_balls > 10:
                if match_wickets > 0:
                    return 'all_rounder', True, True, True
                else:
                    return 'batsman', True, False, False
            elif match_wickets > 0:
                return 'bowler', False, True, False
            else:
                return 'batsman', True, False, False
        
        # For star players, use career data
        batting_matches = player_career_data.get('batting_matches', 0)
        bowling_matches = player_career_data.get('bowling_matches', 0)
        batting_runs = player_career_data.get('batting_runs', 0)
        bowling_wickets = player_career_data.get('bowling_wickets', 0)
        
        # Determine primary role
        if batting_matches >= 10 and bowling_matches >= 10:
            if batting_runs > 500 and bowling_wickets > 20:
                return 'all_rounder', True, True, True
            elif batting_runs > 500:
                return 'batsman', True, False, False
            else:
                return 'bowler', False, True, False
        elif batting_matches >= 10 and batting_runs > 200:
            return 'batsman', True, False, False
        elif bowling_matches >= 10 and bowling_wickets > 10:
            return 'bowler', False, True, False
        else:
            # Fallback based on match performance
            match_runs = match_data.get('runs', 0)
            match_wickets = match_data.get('wickets', 0)
            if match_runs > match_wickets * 10:
                return 'batsman', True, False, False
            else:
                return 'bowler', False, True, False
    
    def process_match_file(self, file_path):
        """Process a single match file"""
        try:
            with open(file_path, 'r') as f:
                match_data = json.load(f)
            
            match_id = match_data.get('info', {}).get('id', 'unknown')
            match_date = match_data.get('info', {}).get('dates', [''])[0]
            venue = match_data.get('info', {}).get('venue', 'Unknown')
            
            # Extract team information
            teams = match_data.get('info', {}).get('teams', [])
            if len(teams) != 2:
                return
            
            team1, team2 = teams[0], teams[1]
            
            # Process both innings
            for innings_num, innings in enumerate(match_data.get('innings', []), 1):
                team = innings.get('team', '')
                if team not in [team1, team2]:
                    continue
                
                # Track player performances
                player_performances = {}
                
                # Process deliveries
                for over in innings.get('overs', []):
                    for delivery in over.get('deliveries', []):
                        batsman = delivery.get('batter', '')
                        bowler = delivery.get('bowler', '')
                        
                        # Track batting performance
                        if batsman not in player_performances:
                            player_performances[batsman] = {
                                'runs': 0, 'balls': 0, '4s': 0, '6s': 0, 'wickets': 0
                            }
                        
                        runs = delivery.get('runs', {}).get('batter', 0)
                        player_performances[batsman]['runs'] += runs
                        player_performances[batsman]['balls'] += 1
                        
                        if runs == 4:
                            player_performances[batsman]['4s'] += 1
                        elif runs == 6:
                            player_performances[batsman]['6s'] += 1
                        
                        # Track bowling performance
                        if bowler not in player_performances:
                            player_performances[bowler] = {
                                'runs': 0, 'balls': 0, '4s': 0, '6s': 0, 'wickets': 0
                            }
                        
                        # Count wickets
                        if 'wickets' in delivery:
                            for wicket in delivery['wickets']:
                                if wicket.get('kind') != 'run out':
                                    player_performances[bowler]['wickets'] += 1
                
                # Create player impact records
                for player_name, performance in player_performances.items():
                    if player_name == '':
                        continue
                    
                    # Find player ID
                    player_info = self.combined_lookup[
                        self.combined_lookup['ball_by_ball_name'] == player_name
                    ]
                    
                    if player_info.empty:
                        continue
                    
                    player_id = player_info.iloc[0]['ball_by_ball_id']
                    
                    # Determine if star player
                    is_star, career_data = self.is_star_player(player_id)
                    
                    if is_star:
                        self.star_players_count += 1
                        impact_data = self.calculate_star_player_impact(
                            performance, player_id, player_name, career_data
                        )
                    else:
                        self.regular_players_count += 1
                        impact_data = self.calculate_regular_player_impact(
                            performance, player_id, player_name
                        )
                    
                    # Determine player role
                    player_role, is_batsman, is_bowler, is_all_rounder = self.determine_player_role(
                        career_data, performance
                    )
                    
                    # Create comprehensive record
                    record = {
                        'match_id': match_id,
                        'date': match_date,
                        'venue': venue,
                        'team': team,
                        'opposition': team2 if team == team1 else team1,
                        'player_id': player_id,
                        'player_name': player_name,
                        'match_runs': performance['runs'],
                        'match_balls': performance['balls'],
                        'match_wickets': performance['wickets'],
                        'match_4s': performance['4s'],
                        'match_6s': performance['6s'],
                        'match_strike_rate': (performance['runs'] / performance['balls'] * 100) if performance['balls'] > 0 else 0,
                        'player_role': player_role,
                        'is_batsman': is_batsman,
                        'is_bowler': is_bowler,
                        'is_all_rounder': is_all_rounder,
                        **impact_data
                    }
                    
                    self.player_performance_data.append(record)
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    def build_dataset(self):
        """Build the enhanced player impact dataset"""
        print("Building Enhanced Player Impact Dataset...")
        print("=" * 50)
        
        # Get all match files
        match_files = glob.glob(f"{self.ball_by_ball_dir}/*.json")
        print(f"Processing {len(match_files)} match files...")
        
        # Process each match
        for i, file_path in enumerate(match_files):
            if i % 500 == 0:
                print(f"Processed {i}/{len(match_files)} matches...")
            
            self.process_match_file(file_path)
        
        print(f"Processing completed!")
        print(f"Star players: {self.star_players_count}")
        print(f"Regular players: {self.regular_players_count}")
        
        # Create DataFrame
        df = pd.DataFrame(self.player_performance_data)
        
        if df.empty:
            print("No data generated!")
            return None
        
        print(f"Generated {len(df)} player performance records")
        
        # Save dataset
        output_path = f"{self.output_dir}/enhanced_player_impact_dataset.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved enhanced player impact dataset: {output_path}")
        
        # Save summary statistics
        self.save_summary_statistics(df)
        
        return df
    
    def save_summary_statistics(self, df):
        """Save summary statistics"""
        summary = {
            'total_records': len(df),
            'unique_players': df['player_id'].nunique(),
            'star_players': len(df[df['has_meaningful_career'] == True]),
            'regular_players': len(df[df['has_meaningful_career'] == False]),
            'unique_matches': df['match_id'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'player_roles': df['player_role'].value_counts().to_dict(),
            'average_impact_score': df['player_impact_score'].mean(),
            'top_10_impact_scores': df.nlargest(10, 'player_impact_score')[['player_name', 'player_impact_score', 'match_runs', 'match_wickets']].to_dict('records')
        }
        
        # Save as JSON
        import json
        with open(f"{self.output_dir}/enhanced_player_impact_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary statistics saved: {self.output_dir}/enhanced_player_impact_summary.json")
        
        # Print key statistics
        print(f"\n=== ENHANCED PLAYER IMPACT DATASET SUMMARY ===")
        print(f"Total records: {summary['total_records']:,}")
        print(f"Unique players: {summary['unique_players']:,}")
        print(f"Star players (with career stats): {summary['star_players']:,}")
        print(f"Regular players (recent performance only): {summary['regular_players']:,}")
        print(f"Unique matches: {summary['unique_matches']:,}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Average impact score: {summary['average_impact_score']:.2f}")
        print(f"\nPlayer roles distribution:")
        for role, count in summary['player_roles'].items():
            print(f"  {role}: {count:,}")

def main():
    print("Enhanced Player Impact Dataset Builder")
    print("Strategic approach focusing on star players with real career stats")
    print("=" * 70)
    
    builder = EnhancedPlayerImpactBuilder()
    dataset = builder.build_dataset()
    
    if dataset is not None:
        print(f"\nEnhanced player impact dataset created successfully!")
        print(f"Ready for model training with real player intelligence!")
    else:
        print(f"Failed to create dataset!")

if __name__ == "__main__":
    main()
