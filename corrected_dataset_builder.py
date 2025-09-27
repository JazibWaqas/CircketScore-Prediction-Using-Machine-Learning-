"""
Cricket Score Prediction - Corrected Dataset Builder
This script creates a dataset that preserves team names, match IDs, and other identifying information.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CorrectedDatasetBuilder:
    def __init__(self):
        self.player_stats = {}
        self.match_data = []
        self.team_innings_data = []
        self.final_dataset = None
        
    def load_and_clean_player_stats(self):
        """Load and clean player statistics"""
        print("Loading player statistics...")
        
        # Load all player data
        self.player_stats = {
            'all_players': pd.read_csv('PlayerStats/all_players.csv'),
            't20_batting': pd.read_csv('PlayerStats/t20_batting.csv'),
            't20_bowling': pd.read_csv('PlayerStats/t20_bowling.csv'),
            'fielding': pd.read_csv('PlayerStats/fielding.csv'),
            'country': pd.read_csv('PlayerStats/country.csv'),
            't20_all_round': pd.read_csv('PlayerStats/t20_all_round.csv')
        }
        
        # Clean country names
        self.player_stats['country']['country'] = self.player_stats['country']['country'].str.replace('Newziland', 'New Zealand')
        
        # Create country mapping
        self.country_mapping = dict(zip(
            self.player_stats['country']['id'], 
            self.player_stats['country']['country']
        ))
        
        # Add country names to all_players
        self.player_stats['all_players']['country_name'] = self.player_stats['all_players']['country_id'].map(self.country_mapping)
        
        # Create comprehensive player database
        self.player_database = self.player_stats['all_players'].copy()
        
        # Merge with T20 stats
        self.player_database = self.player_database.merge(
            self.player_stats['t20_batting'], on='id', how='left', suffixes=('', '_batting')
        )
        self.player_database = self.player_database.merge(
            self.player_stats['t20_bowling'], on='id', how='left', suffixes=('', '_bowling')
        )
        self.player_database = self.player_database.merge(
            self.player_stats['fielding'], on='id', how='left', suffixes=('', '_fielding')
        )
        
        print(f"Loaded {len(self.player_database)} players from {len(self.country_mapping)} countries")
        
    def create_team_mapping(self):
        """Create mapping between match team names and country names"""
        print("Creating team mapping...")
        
        # Common team name variations
        self.team_mapping = {
            # Direct matches
            'India': 'India',
            'Australia': 'Australia', 
            'England': 'England',
            'Pakistan': 'Pakistan',
            'South Africa': 'South Africa',
            'Sri Lanka': 'Sri Lanka',
            'Bangladesh': 'Bangladesh',
            'New Zealand': 'New Zealand',
            'West Indies': 'West Indies',
            'Zimbabwe': 'Zimbabwe',
            'Afghanistan': 'Afghanistan',
            'Ireland': 'Ireland',
            
            # Variations and common names
            'Newziland': 'New Zealand',
            'NZ': 'New Zealand',
            'WI': 'West Indies',
            'SA': 'South Africa',
            'SL': 'Sri Lanka',
            'BD': 'Bangladesh',
            'AFG': 'Afghanistan',
            'IRE': 'Ireland',
            'ZIM': 'Zimbabwe'
        }
        
        print(f"Created team mapping for {len(self.team_mapping)} teams")
        
    def load_t20_matches(self, max_matches=500):
        """Load and process T20 match data"""
        print(f"Loading T20 match data (max {max_matches} matches)...")
        
        match_files = [f for f in os.listdir('t20 matches ball by ball') if f.endswith('.json')]
        processed_matches = []
        
        for i, file in enumerate(match_files[:max_matches]):
            try:
                with open(f't20 matches ball by ball/{file}', 'r') as f:
                    match_data = json.load(f)
                
                # Extract match information
                match_info = self.extract_match_info(match_data, file)
                if match_info:
                    processed_matches.append(match_info)
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
                
        self.match_data = processed_matches
        print(f"Processed {len(processed_matches)} matches")
        
    def extract_match_info(self, match_data, filename):
        """Extract relevant information from a match JSON file"""
        try:
            info = match_data.get('info', {})
            innings = match_data.get('innings', [])
            
            if not info or not innings:
                return None
                
            # Extract basic match info
            match_info = {
                'match_id': filename.replace('.json', ''),
                'date': info.get('dates', [None])[0] if info.get('dates') else None,
                'venue': info.get('venue', 'Unknown'),
                'city': info.get('city', 'Unknown'),
                'teams': info.get('teams', []),
                'outcome': info.get('outcome', {}),
                'overs': info.get('overs', 20),
                'gender': info.get('gender', 'unknown'),
                'match_type': info.get('match_type', 'T20'),
                'player_of_match': info.get('player_of_match', []),
                'players': info.get('players', {})  # Keep player information
            }
            
            # Extract innings data with proper scoring
            innings_data = []
            for i, inning in enumerate(innings):
                team = inning.get('team', f'Team_{i+1}')
                
                # Extract scoring from overs
                total_runs = 0
                total_wickets = 0
                balls_bowled = 0
                extras = 0
                boundaries_4s = 0
                boundaries_6s = 0
                
                overs_data = inning.get('overs', [])
                for over_data in overs_data:
                    deliveries = over_data.get('deliveries', [])
                    for delivery in deliveries:
                        if isinstance(delivery, dict):
                            runs_data = delivery.get('runs', {})
                            total_runs += runs_data.get('total', 0)
                            balls_bowled += 1
                            
                            # Count boundaries
                            batter_runs = runs_data.get('batter', 0)
                            if batter_runs == 4:
                                boundaries_4s += 1
                            elif batter_runs == 6:
                                boundaries_6s += 1
                            
                            # Count extras
                            extras += runs_data.get('extras', 0)
                            
                            # Count wickets
                            if 'wicket' in delivery:
                                total_wickets += 1
                
                innings_info = {
                    'match_id': match_info['match_id'],
                    'team': team,
                    'innings_number': i + 1,
                    'total_runs': total_runs,
                    'total_wickets': total_wickets,
                    'balls_bowled': balls_bowled,
                    'overs_bowled': balls_bowled / 6.0,
                    'run_rate': total_runs / (balls_bowled / 6.0) if balls_bowled > 0 else 0,
                    'extras': extras,
                    'boundaries_4s': boundaries_4s,
                    'boundaries_6s': boundaries_6s,
                    'boundaries_total': boundaries_4s + boundaries_6s
                }
                
                innings_data.append(innings_info)
            
            match_info['innings'] = innings_data
            return match_info
            
        except Exception as e:
            print(f"Error extracting match info: {e}")
            return None
    
    def create_team_innings_dataset(self):
        """Create dataset with each row representing a team innings"""
        print("Creating team innings dataset...")
        
        team_innings = []
        
        for match in self.match_data:
            if 'innings' not in match:
                continue
                
            for inning in match['innings']:
                # Create features for this team innings
                team_innings_row = {
                    'match_id': inning['match_id'],
                    'team': inning['team'],
                    'innings_number': inning['innings_number'],
                    'total_runs': inning['total_runs'],
                    'total_wickets': inning['total_wickets'],
                    'balls_bowled': inning['balls_bowled'],
                    'overs_bowled': inning['overs_bowled'],
                    'run_rate': inning['run_rate'],
                    'extras': inning['extras'],
                    'boundaries_4s': inning['boundaries_4s'],
                    'boundaries_6s': inning['boundaries_6s'],
                    'boundaries_total': inning['boundaries_total'],
                    'venue': match.get('venue', 'Unknown'),
                    'city': match.get('city', 'Unknown'),
                    'date': match.get('date'),
                    'opposition': self.get_opposition_team(match, inning['team']),
                    'match_outcome': match.get('outcome', {}).get('winner', 'Unknown'),
                    'player_of_match': match.get('player_of_match', []),
                    'teams': match.get('teams', [])
                }
                
                team_innings.append(team_innings_row)
        
        self.team_innings_df = pd.DataFrame(team_innings)
        print(f"Created team innings dataset with {len(self.team_innings_df)} records")
        
        # Display sample with identifying information
        print("\nSample of dataset with identifying information:")
        print(self.team_innings_df[['match_id', 'team', 'opposition', 'venue', 'total_runs']].head(10))
        
    def get_opposition_team(self, match, current_team):
        """Get the opposition team for a given team"""
        teams = match.get('teams', [])
        if len(teams) >= 2:
            return teams[1] if teams[0] == current_team else teams[0]
        return 'Unknown'
    
    def create_team_strength_features(self):
        """Create team strength features by aggregating player stats"""
        print("Creating team strength features...")
        
        team_features = []
        
        for _, row in self.team_innings_df.iterrows():
            team = row['team']
            
            # Map team name to country
            country = self.team_mapping.get(team, team)
            
            # Get players from this country
            country_players = self.player_database[
                self.player_database['country_name'] == country
            ]
            
            if len(country_players) > 0:
                # Calculate team batting strength
                batting_stats = {
                    'team_batting_avg': country_players['average_score'].mean() if 'average_score' in country_players.columns else 25.0,
                    'team_strike_rate': country_players['strike_rate'].mean() if 'strike_rate' in country_players.columns else 120.0,
                    'team_centuries': country_players['100s'].sum() if '100s' in country_players.columns else 0,
                    'team_fifties': country_players['50'].sum() if '50' in country_players.columns else 0,
                    'team_boundaries_4s': country_players['4s'].sum() if '4s' in country_players.columns else 0,
                    'team_boundaries_6s': country_players['6s'].sum() if '6s' in country_players.columns else 0,
                    'team_players_count': len(country_players)
                }
                
                # Calculate team bowling strength
                bowling_stats = {
                    'team_bowling_avg': country_players['bwa'].mean() if 'bwa' in country_players.columns else 30.0,
                    'team_economy': country_players['bwe'].mean() if 'bwe' in country_players.columns else 7.0,
                    'team_wickets': country_players['tw'].sum() if 'tw' in country_players.columns else 0,
                    'team_maidens': country_players['md'].sum() if 'md' in country_players.columns else 0
                }
                
                # Calculate team composition
                role_counts = country_players['playing_role'].value_counts()
                composition_stats = {
                    'team_batsmen': role_counts.get('top-order batter', 0) + role_counts.get('middle-order batter', 0) + role_counts.get('opening batter', 0),
                    'team_bowlers': role_counts.get('bowler', 0),
                    'team_allrounders': role_counts.get('allrounder', 0) + role_counts.get('batting allrounder', 0) + role_counts.get('bowling allrounder', 0),
                    'team_wicketkeepers': role_counts.get('wicketkeeper batter', 0) + role_counts.get('wicketkeeper', 0)
                }
                
            else:
                # Default values if no players found
                batting_stats = {
                    'team_batting_avg': 25.0, 'team_strike_rate': 120.0, 'team_centuries': 0,
                    'team_fifties': 0, 'team_boundaries_4s': 0, 'team_boundaries_6s': 0, 'team_players_count': 0
                }
                bowling_stats = {
                    'team_bowling_avg': 30.0, 'team_economy': 7.0, 'team_wickets': 0, 'team_maidens': 0
                }
                composition_stats = {
                    'team_batsmen': 6, 'team_bowlers': 4, 'team_allrounders': 1, 'team_wicketkeepers': 1
                }
            
            # Combine all stats
            team_stats = {**batting_stats, **bowling_stats, **composition_stats}
            
            # Add team stats to the row
            for key, value in team_stats.items():
                row[key] = value
                
            team_features.append(row.to_dict())
        
        self.team_innings_df = pd.DataFrame(team_features)
        print(f"Added team strength features to {len(self.team_innings_df)} records")
    
    def create_venue_features(self):
        """Create venue-based features"""
        print("Creating venue features...")
        
        # Calculate venue statistics
        venue_stats = self.team_innings_df.groupby('venue').agg({
            'total_runs': ['mean', 'std', 'count', 'min', 'max'],
            'run_rate': ['mean', 'std'],
            'boundaries_total': ['mean', 'sum']
        }).round(2)
        
        venue_stats.columns = [
            'venue_avg_runs', 'venue_runs_std', 'venue_matches', 'venue_min_runs', 'venue_max_runs',
            'venue_avg_rr', 'venue_rr_std', 'venue_avg_boundaries', 'venue_total_boundaries'
        ]
        venue_stats = venue_stats.reset_index()
        
        # Add venue scoring tendency
        venue_stats['venue_high_scoring'] = (venue_stats['venue_avg_runs'] > venue_stats['venue_avg_runs'].median()).astype(int)
        
        # Merge venue stats back to main dataset
        self.team_innings_df = self.team_innings_df.merge(venue_stats, on='venue', how='left')
        
        # Fill missing venue stats with overall averages
        for col in ['venue_avg_runs', 'venue_avg_rr', 'venue_avg_boundaries']:
            self.team_innings_df[col] = self.team_innings_df[col].fillna(self.team_innings_df[col].mean())
        
        print("Added venue-based features")
    
    def create_opposition_features(self):
        """Create opposition-based features"""
        print("Creating opposition features...")
        
        # Calculate opposition statistics
        opposition_stats = self.team_innings_df.groupby('opposition').agg({
            'total_runs': ['mean', 'std', 'count'],
            'run_rate': ['mean', 'std'],
            'boundaries_total': ['mean', 'sum']
        }).round(2)
        
        opposition_stats.columns = [
            'opp_avg_runs', 'opp_runs_std', 'opp_matches', 'opp_avg_rr', 'opp_rr_std',
            'opp_avg_boundaries', 'opp_total_boundaries'
        ]
        opposition_stats = opposition_stats.reset_index()
        
        # Merge opposition stats
        self.team_innings_df = self.team_innings_df.merge(opposition_stats, on='opposition', how='left')
        
        # Fill missing opposition stats
        for col in ['opp_avg_runs', 'opp_avg_rr', 'opp_avg_boundaries']:
            self.team_innings_df[col] = self.team_innings_df[col].fillna(self.team_innings_df[col].mean())
        
        print("Added opposition-based features")
    
    def create_final_dataset(self):
        """Create final dataset that preserves identifying information"""
        print("Creating final dataset with identifying information...")
        
        # Keep ALL columns including identifying information
        self.final_dataset = self.team_innings_df.copy()
        
        # Handle missing values for numerical columns only
        numerical_columns = self.final_dataset.select_dtypes(include=[np.number]).columns
        self.final_dataset[numerical_columns] = self.final_dataset[numerical_columns].fillna(self.final_dataset[numerical_columns].mean())
        
        # Add derived features
        self.final_dataset['team_balance'] = (
            self.final_dataset['team_batsmen'] / 
            (self.final_dataset['team_batsmen'] + self.final_dataset['team_bowlers'])
        )
        
        self.final_dataset['venue_advantage'] = (
            self.final_dataset['venue_avg_runs'] - self.final_dataset['venue_avg_runs'].mean()
        )
        
        print(f"Final dataset shape: {self.final_dataset.shape}")
        print(f"Columns: {list(self.final_dataset.columns)}")
        
        return self.final_dataset
    
    def save_dataset(self, filename='corrected_cricket_dataset.csv'):
        """Save the final dataset"""
        self.final_dataset.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        
        # Display summary statistics
        print("\nDataset Summary:")
        print(f"Shape: {self.final_dataset.shape}")
        print(f"Target variable (total_runs) statistics:")
        print(self.final_dataset['total_runs'].describe())
        
        # Show sample with identifying information
        print("\nSample records with identifying information:")
        print(self.final_dataset[['match_id', 'team', 'opposition', 'venue', 'total_runs']].head(10))
        
        return self.final_dataset
    
    def run_full_pipeline(self, max_matches=500):
        """Run the complete dataset building pipeline"""
        print("Starting corrected dataset building pipeline...")
        
        # Load and process data
        self.load_and_clean_player_stats()
        self.create_team_mapping()
        self.load_t20_matches(max_matches)
        
        # Create datasets
        self.create_team_innings_dataset()
        self.create_team_strength_features()
        self.create_venue_features()
        self.create_opposition_features()
        
        # Finalize dataset
        final_dataset = self.create_final_dataset()
        
        print("Dataset building completed successfully!")
        return final_dataset

def main():
    """Main function to build the corrected dataset"""
    builder = CorrectedDatasetBuilder()
    final_dataset = builder.run_full_pipeline(max_matches=500)
    
    # Save the dataset
    builder.save_dataset('corrected_cricket_dataset.csv')
    
    return final_dataset

if __name__ == "__main__":
    dataset = main()
