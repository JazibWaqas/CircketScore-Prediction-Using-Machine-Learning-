#!/usr/bin/env python3
"""
Comprehensive Enhanced Dataset Creator
Extract maximum value from ball-by-ball data for cricket score prediction

This script creates a much more comprehensive dataset by extracting:
1. Individual player performance metrics
2. Venue-specific conditions and characteristics
3. Match context and pressure situations
4. Weather and pitch conditions
5. Toss impact analysis
6. Player roles and specializations
7. Team composition analysis
8. Historical performance patterns
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDatasetCreator:
    def __init__(self):
        self.ball_by_ball_dir = "raw_data/t20 matches ball by ball"
        self.player_stats_dir = "raw_data/PlayerStats"
        self.output_dir = "processed_data"
        
        # Initialize data storage
        self.matches_data = []
        self.player_performance = {}
        self.venue_characteristics = {}
        self.team_compositions = {}
        
        # Load existing lookup tables
        self.load_lookup_tables()
        
    def load_lookup_tables(self):
        """Load existing lookup tables"""
        print("Loading lookup tables...")
        
        self.team_lookup = pd.read_csv("data/team_lookup.csv")
        self.venue_lookup = pd.read_csv("data/venue_lookup.csv") 
        self.player_lookup = pd.read_csv("data/player_lookup.csv")
        
        # Load player stats
        self.batting_stats = pd.read_csv(f"{self.player_stats_dir}/t20_batting.csv")
        self.bowling_stats = pd.read_csv(f"{self.player_stats_dir}/t20_bowling.csv")
        self.all_round_stats = pd.read_csv(f"{self.player_stats_dir}/t20_all_round.csv")
        
        print(f"Loaded {len(self.team_lookup)} teams, {len(self.venue_lookup)} venues, {len(self.player_lookup)} players")
        
    def extract_ball_by_ball_features(self, match_file):
        """Extract comprehensive features from ball-by-ball JSON data"""
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)
        except:
            return None
            
        match_id = os.path.basename(match_file).replace('.json', '')
        info = match_data.get('info', {})
        innings = match_data.get('innings', [])
        
        # Basic match info
        match_features = {
            'match_id': match_id,
            'date': info.get('dates', [None])[0],
            'venue': info.get('venue', 'Unknown'),
            'city': info.get('city', 'Unknown'),
            'teams': info.get('teams', []),
            'toss_winner': info.get('toss', {}).get('winner'),
            'toss_decision': info.get('toss', {}).get('decision'),
            'match_winner': info.get('outcome', {}).get('winner'),
            'player_of_match': info.get('player_of_match', [None])[0],
            'season': info.get('season', 'Unknown'),
            'match_type': info.get('match_type', 'T20'),
            'gender': info.get('gender', 'male'),
            'balls_per_over': info.get('balls_per_over', 6),
            'overs': info.get('overs', 20)
        }
        
        # Extract detailed innings data
        innings_features = self.extract_innings_features(innings, match_features)
        
        # Extract player performance metrics
        player_features = self.extract_player_performance_features(innings, match_features)
        
        # Extract venue characteristics
        venue_features = self.extract_venue_characteristics(match_features, innings)
        
        # Extract match context features
        context_features = self.extract_match_context_features(match_features, innings)
        
        # Combine all features
        comprehensive_features = {**match_features, **innings_features, **player_features, **venue_features, **context_features}
        
        return comprehensive_features
        
    def extract_innings_features(self, innings, match_info):
        """Extract detailed innings-level features"""
        features = {}
        
        for i, inning in enumerate(innings):
            team = inning.get('team', f'Team_{i+1}')
            overs = inning.get('overs', [])
            
            # Basic innings stats
            total_runs = 0
            total_balls = 0
            wickets = 0
            extras = {'wides': 0, 'noballs': 0, 'byes': 0, 'legbyes': 0}
            
            # Over-by-over analysis
            over_scores = []
            powerplay_runs = 0
            death_overs_runs = 0
            middle_overs_runs = 0
            
            # Player performance tracking
            player_runs = {}
            player_balls = {}
            player_wickets = {}
            
            for over_data in overs:
                over_num = over_data.get('over', 0)
                deliveries = over_data.get('deliveries', [])
                over_runs = 0
                
                for delivery in deliveries:
                    runs = delivery.get('runs', {})
                    total_runs += runs.get('total', 0)
                    total_balls += 1
                    over_runs += runs.get('total', 0)
                    
                    # Track extras
                    extras_data = delivery.get('extras', {})
                    for extra_type, count in extras_data.items():
                        if extra_type in extras:
                            extras[extra_type] += count
                    
                    # Track wickets
                    if 'wickets' in delivery:
                        wickets += len(delivery['wickets'])
                    
                    # Track player performance
                    batter = delivery.get('batter')
                    bowler = delivery.get('bowler')
                    
                    if batter:
                        player_runs[batter] = player_runs.get(batter, 0) + runs.get('batter', 0)
                        player_balls[batter] = player_balls.get(batter, 0) + 1
                    
                    if bowler and 'wickets' in delivery:
                        player_wickets[bowler] = player_wickets.get(bowler, 0) + len(delivery['wickets'])
                
                over_scores.append(over_runs)
                
                # Categorize overs
                if over_num < 6:  # Powerplay
                    powerplay_runs += over_runs
                elif over_num >= 16:  # Death overs
                    death_overs_runs += over_runs
                else:  # Middle overs
                    middle_overs_runs += over_runs
            
            # Calculate derived metrics
            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
            run_rate = total_runs / (total_balls / 6) if total_balls > 0 else 0
            
            # Store features for this team
            prefix = f'team_{i+1}_' if i < 2 else f'team_{i+1}_'
            features.update({
                f'{prefix}total_runs': total_runs,
                f'{prefix}total_balls': total_balls,
                f'{prefix}wickets': wickets,
                f'{prefix}strike_rate': strike_rate,
                f'{prefix}run_rate': run_rate,
                f'{prefix}powerplay_runs': powerplay_runs,
                f'{prefix}middle_overs_runs': middle_overs_runs,
                f'{prefix}death_overs_runs': death_overs_runs,
                f'{prefix}wides': extras['wides'],
                f'{prefix}noballs': extras['noballs'],
                f'{prefix}byes': extras['byes'],
                f'{prefix}legbyes': extras['legbyes'],
                f'{prefix}over_variance': np.var(over_scores) if over_scores else 0,
                f'{prefix}max_over_score': max(over_scores) if over_scores else 0,
                f'{prefix}min_over_score': min(over_scores) if over_scores else 0
            })
            
            # Store player performance
            features[f'{prefix}top_scorer'] = max(player_runs.items(), key=lambda x: x[1])[0] if player_runs else None
            features[f'{prefix}top_scorer_runs'] = max(player_runs.values()) if player_runs else 0
            features[f'{prefix}top_bowler'] = max(player_wickets.items(), key=lambda x: x[1])[0] if player_wickets else None
            features[f'{prefix}top_bowler_wickets'] = max(player_wickets.values()) if player_wickets else 0
            
            # Store individual player stats
            for player, runs in player_runs.items():
                features[f'{prefix}player_{player}_runs'] = runs
                features[f'{prefix}player_{player}_balls'] = player_balls.get(player, 0)
                features[f'{prefix}player_{player}_strike_rate'] = (runs / player_balls.get(player, 1) * 100) if player_balls.get(player, 0) > 0 else 0
        
        return features
    
    def extract_player_performance_features(self, innings, match_info):
        """Extract individual player performance metrics"""
        features = {}
        
        # Get all players in the match
        all_players = set()
        for inning in innings:
            overs = inning.get('overs', [])
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                for delivery in deliveries:
                    if 'batter' in delivery:
                        all_players.add(delivery['batter'])
                    if 'bowler' in delivery:
                        all_players.add(delivery['bowler'])
        
        # Calculate player impact scores
        player_impacts = {}
        for player in all_players:
            # Get player stats from lookup tables
            player_stats = self.get_player_stats(player)
            
            # Calculate match-specific performance
            match_performance = self.calculate_player_match_performance(player, innings)
            
            # Combine for impact score
            impact_score = self.calculate_player_impact_score(player_stats, match_performance)
            player_impacts[player] = impact_score
        
        # Store top performers
        sorted_players = sorted(player_impacts.items(), key=lambda x: x[1], reverse=True)
        for i, (player, impact) in enumerate(sorted_players[:5]):  # Top 5 players
            features[f'top_player_{i+1}'] = player
            features[f'top_player_{i+1}_impact'] = impact
        
        return features
    
    def get_player_stats(self, player_name):
        """Get player statistics from lookup tables"""
        # Try to find player in batting stats
        batting_stats = self.batting_stats[self.batting_stats['id'].astype(str).str.contains(player_name, case=False, na=False)]
        bowling_stats = self.bowling_stats[self.bowling_stats['id'].astype(str).str.contains(player_name, case=False, na=False)]
        
        return {
            'batting_avg': batting_stats['average_score'].iloc[0] if not batting_stats.empty else 20.0,
            'batting_sr': batting_stats['strike_rate'].iloc[0] if not batting_stats.empty else 120.0,
            'bowling_avg': bowling_stats['average_score'].iloc[0] if not bowling_stats.empty else 30.0,
            'bowling_econ': bowling_stats['strike_rate'].iloc[0] if not bowling_stats.empty else 8.0
        }
    
    def calculate_player_match_performance(self, player, innings):
        """Calculate player's performance in this specific match"""
        runs = 0
        balls = 0
        wickets = 0
        
        for inning in innings:
            overs = inning.get('overs', [])
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                for delivery in deliveries:
                    if delivery.get('batter') == player:
                        runs += delivery.get('runs', {}).get('batter', 0)
                        balls += 1
                    if delivery.get('bowler') == player and 'wickets' in delivery:
                        wickets += len(delivery['wickets'])
        
        return {
            'runs': runs,
            'balls': balls,
            'wickets': wickets,
            'strike_rate': (runs / balls * 100) if balls > 0 else 0
        }
    
    def calculate_player_impact_score(self, player_stats, match_performance):
        """Calculate overall player impact score"""
        # Weighted combination of historical stats and match performance
        historical_weight = 0.3
        match_weight = 0.7
        
        historical_score = (
            player_stats['batting_avg'] * 0.4 +
            player_stats['batting_sr'] * 0.3 +
            (50 - player_stats['bowling_avg']) * 0.2 +  # Lower bowling avg is better
            (10 - player_stats['bowling_econ']) * 0.1   # Lower economy is better
        )
        
        match_score = (
            match_performance['runs'] * 0.5 +
            match_performance['strike_rate'] * 0.3 +
            match_performance['wickets'] * 20 * 0.2
        )
        
        return historical_weight * historical_score + match_weight * match_score
    
    def extract_venue_characteristics(self, match_info, innings):
        """Extract venue-specific characteristics"""
        features = {}
        
        venue = match_info.get('venue', 'Unknown')
        city = match_info.get('city', 'Unknown')
        
        # Venue difficulty based on historical data
        venue_difficulty = self.calculate_venue_difficulty(venue, innings)
        
        # Weather conditions (simulated based on season and location)
        weather_conditions = self.estimate_weather_conditions(match_info)
        
        # Pitch characteristics
        pitch_characteristics = self.analyze_pitch_characteristics(innings)
        
        features.update({
            'venue_difficulty': venue_difficulty,
            'weather_temperature': weather_conditions['temperature'],
            'weather_humidity': weather_conditions['humidity'],
            'weather_wind_speed': weather_conditions['wind_speed'],
            'pitch_bounce': pitch_characteristics['bounce'],
            'pitch_pace': pitch_characteristics['pace'],
            'pitch_turn': pitch_characteristics['turn'],
            'pitch_swing': pitch_characteristics['swing']
        })
        
        return features
    
    def calculate_venue_difficulty(self, venue, innings):
        """Calculate venue difficulty based on scoring patterns"""
        if not innings:
            return 1.0
        
        # Calculate average runs per over at this venue
        total_runs = sum(
            sum(delivery.get('runs', {}).get('total', 0) for delivery in over.get('deliveries', []))
            for inning in innings for over in inning.get('overs', [])
        )
        
        total_overs = sum(len(inning.get('overs', [])) for inning in innings)
        avg_rpo = total_runs / total_overs if total_overs > 0 else 7.5
        
        # Normalize difficulty (higher RPO = easier venue)
        return min(max(avg_rpo / 7.5, 0.5), 2.0)
    
    def estimate_weather_conditions(self, match_info):
        """Estimate weather conditions based on date and location"""
        date_str = match_info.get('date')
        city = match_info.get('city', 'Unknown')
        
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                month = date.month
                
                # Simple weather estimation based on month and location
                if city.lower() in ['mumbai', 'delhi', 'kolkata', 'chennai']:  # Indian cities
                    temp = 25 + 10 * np.sin(2 * np.pi * month / 12)
                    humidity = 70 + 20 * np.sin(2 * np.pi * month / 12)
                elif city.lower() in ['london', 'birmingham', 'manchester']:  # UK cities
                    temp = 15 + 8 * np.sin(2 * np.pi * month / 12)
                    humidity = 80 + 10 * np.sin(2 * np.pi * month / 12)
                else:  # Default
                    temp = 20 + 8 * np.sin(2 * np.pi * month / 12)
                    humidity = 60 + 15 * np.sin(2 * np.pi * month / 12)
                
                return {
                    'temperature': temp,
                    'humidity': humidity,
                    'wind_speed': 5 + np.random.normal(0, 2)
                }
            except:
                pass
        
        # Default conditions
        return {
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 5.0
        }
    
    def analyze_pitch_characteristics(self, innings):
        """Analyze pitch characteristics based on match data"""
        # Analyze scoring patterns to determine pitch type
        total_runs = 0
        total_balls = 0
        boundaries = 0
        
        for inning in innings:
            overs = inning.get('overs', [])
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                for delivery in deliveries:
                    runs = delivery.get('runs', {})
                    total_runs += runs.get('total', 0)
                    total_balls += 1
                    
                    # Count boundaries
                    if runs.get('batter', 0) >= 4:
                        boundaries += 1
        
        # Calculate pitch characteristics
        run_rate = total_runs / (total_balls / 6) if total_balls > 0 else 7.5
        boundary_rate = boundaries / total_balls if total_balls > 0 else 0.1
        
        return {
            'bounce': min(max(run_rate / 8, 0.5), 1.5),
            'pace': min(max(boundary_rate * 10, 0.5), 1.5),
            'turn': min(max((8 - run_rate) / 8, 0.5), 1.5),
            'swing': min(max((8 - run_rate) / 8, 0.5), 1.5)
        }
    
    def extract_match_context_features(self, match_info, innings):
        """Extract match context and pressure features"""
        features = {}
        
        # Toss impact analysis
        toss_winner = match_info.get('toss_winner')
        toss_decision = match_info.get('toss_decision')
        match_winner = match_info.get('match_winner')
        
        toss_impact = 0
        if toss_winner and match_winner:
            toss_impact = 1 if toss_winner == match_winner else -1
        
        # Pressure situations
        pressure_score = self.calculate_pressure_score(match_info, innings)
        
        # Match importance
        importance_score = self.calculate_match_importance(match_info)
        
        # Team balance analysis
        team_balance = self.analyze_team_balance(innings)
        
        features.update({
            'toss_impact': toss_impact,
            'pressure_score': pressure_score,
            'importance_score': importance_score,
            'team_balance': team_balance,
            'is_final': 1 if 'final' in match_info.get('season', '').lower() else 0,
            'is_semi_final': 1 if 'semi' in match_info.get('season', '').lower() else 0,
            'is_world_cup': 1 if 'world' in match_info.get('season', '').lower() else 0,
            'is_ipl': 1 if 'ipl' in match_info.get('season', '').lower() else 0
        })
        
        return features
    
    def calculate_pressure_score(self, match_info, innings):
        """Calculate pressure score based on match situation"""
        # Analyze run rate progression and pressure situations
        if not innings:
            return 0.5
        
        # Calculate pressure based on required run rate vs current run rate
        total_runs = sum(
            sum(delivery.get('runs', {}).get('total', 0) for delivery in over.get('deliveries', []))
            for inning in innings for over in inning.get('overs', [])
        )
        
        total_overs = sum(len(inning.get('overs', [])) for inning in innings)
        if total_overs == 0:
            return 0.5
        
        current_rr = total_runs / total_overs
        target_rr = 8.0  # Typical T20 target
        
        pressure = min(max((target_rr - current_rr) / target_rr, 0), 1)
        return pressure
    
    def calculate_match_importance(self, match_info):
        """Calculate match importance score"""
        season = match_info.get('season', '').lower()
        match_type = match_info.get('match_type', 'T20')
        
        importance = 0.5  # Base importance
        
        # Increase importance for major tournaments
        if 'world' in season:
            importance += 0.3
        if 'ipl' in season:
            importance += 0.2
        if 'final' in season:
            importance += 0.4
        if 'semi' in season:
            importance += 0.3
        
        return min(importance, 1.0)
    
    def analyze_team_balance(self, innings):
        """Analyze team balance and composition"""
        if not innings:
            return 0.5
        
        # Analyze batting vs bowling strength
        batting_strength = 0
        bowling_strength = 0
        
        for inning in innings:
            overs = inning.get('overs', [])
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                for delivery in deliveries:
                    runs = delivery.get('runs', {})
                    batting_strength += runs.get('batter', 0)
                    
                    if 'wickets' in delivery:
                        bowling_strength += len(delivery['wickets']) * 10
        
        # Calculate balance (higher is more balanced)
        if batting_strength + bowling_strength > 0:
            balance = (batting_strength + bowling_strength) / (batting_strength + bowling_strength + 100)
        else:
            balance = 0.5
        
        return min(max(balance, 0), 1)
    
    def process_all_matches(self):
        """Process all ball-by-ball match files"""
        print("Processing all ball-by-ball matches...")
        
        # Get all JSON files
        json_files = glob.glob(f"{self.ball_by_ball_dir}/*.json")
        print(f"Found {len(json_files)} match files")
        
        processed_count = 0
        for i, match_file in enumerate(json_files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(json_files)} matches...")
            
            try:
                features = self.extract_ball_by_ball_features(match_file)
                if features:
                    self.matches_data.append(features)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing {match_file}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} matches")
        return self.matches_data
    
    def create_enhanced_dataset(self):
        """Create the final enhanced dataset"""
        print("Creating comprehensive enhanced dataset...")
        
        # Process all matches
        matches_data = self.process_all_matches()
        
        if not matches_data:
            print("No match data processed!")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(matches_data)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Clean and validate data
        df = self.clean_dataset(df)
        
        # Save dataset
        output_file = f"{self.output_dir}/comprehensive_enhanced_dataset.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Enhanced dataset created: {output_file}")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    
    def add_derived_features(self, df):
        """Add derived features to the dataset"""
        print("Adding derived features...")
        
        # Date-based features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_year'] = df['date'].dt.dayofyear
            df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Team performance features
        for team_num in [1, 2]:
            if f'team_{team_num}_total_runs' in df.columns:
                df[f'team_{team_num}_batting_efficiency'] = (
                    df[f'team_{team_num}_total_runs'] / 
                    df[f'team_{team_num}_total_balls'] * 6
                )
                
                df[f'team_{team_num}_boundary_rate'] = (
                    df[f'team_{team_num}_4s'] / 
                    df[f'team_{team_num}_total_balls']
                )
        
        # Match outcome features
        if 'team_1_total_runs' in df.columns and 'team_2_total_runs' in df.columns:
            df['total_match_runs'] = df['team_1_total_runs'] + df['team_2_total_runs']
            df['run_difference'] = abs(df['team_1_total_runs'] - df['team_2_total_runs'])
            df['is_high_scoring'] = (df['total_match_runs'] > 300).astype(int)
            df['is_close_match'] = (df['run_difference'] < 20).astype(int)
        
        return df
    
    def clean_dataset(self, df):
        """Clean and validate the dataset"""
        print("Cleaning dataset...")
        
        # Remove rows with missing critical data
        critical_columns = ['match_id', 'date', 'venue', 'teams']
        df = df.dropna(subset=critical_columns)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Remove duplicate matches
        df = df.drop_duplicates(subset=['match_id'])
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        columns_to_drop = df.columns[df.isnull().mean() > missing_threshold]
        df = df.drop(columns=columns_to_drop)
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df

def main():
    """Main function to create comprehensive enhanced dataset"""
    print("Creating Comprehensive Enhanced Cricket Dataset")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Initialize creator
    creator = ComprehensiveDatasetCreator()
    
    # Create enhanced dataset
    enhanced_df = creator.create_enhanced_dataset()
    
    if enhanced_df is not None:
        print("\nEnhanced dataset created successfully!")
        print(f"Dataset contains {len(enhanced_df)} matches")
        print(f"Dataset has {len(enhanced_df.columns)} features")
        print(f"Saved to: processed_data/comprehensive_enhanced_dataset.csv")
        
        # Show sample of features
        print("\nSample features:")
        for col in enhanced_df.columns[:20]:
            print(f"  - {col}")
        if len(enhanced_df.columns) > 20:
            print(f"  ... and {len(enhanced_df.columns) - 20} more features")
    else:
        print("Failed to create enhanced dataset")

if __name__ == "__main__":
    main()
