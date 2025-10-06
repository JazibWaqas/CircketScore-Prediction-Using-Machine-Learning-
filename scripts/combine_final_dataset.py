#!/usr/bin/env python3
"""
Combine Final Comprehensive Dataset
Combine all individual datasets into one comprehensive dataset for XGBoost
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalDatasetCombiner:
    def __init__(self):
        self.output_dir = "processed_data"
        self.data_dir = "data"
        
        print("Initializing Final Dataset Combiner...")
        
    def load_individual_datasets(self):
        """Load all individual datasets"""
        print("Loading individual datasets...")
        
        datasets = {}
        
        # Load player impact dataset
        try:
            player_impact = pd.read_csv(f"{self.output_dir}/player_impact_dataset.csv")
            datasets['player_impact'] = player_impact
            print(f"Loaded player impact dataset: {player_impact.shape}")
        except FileNotFoundError:
            print("Player impact dataset not found. Run build_player_impact_dataset.py first.")
            return None
        
        # Load venue conditions dataset
        try:
            venue_conditions = pd.read_csv(f"{self.output_dir}/venue_conditions_dataset.csv")
            datasets['venue_conditions'] = venue_conditions
            print(f"Loaded venue conditions dataset: {venue_conditions.shape}")
        except FileNotFoundError:
            print("Venue conditions dataset not found. Run build_venue_conditions_dataset.py first.")
            return None
        
        # Load team composition dataset
        try:
            team_composition = pd.read_csv(f"{self.output_dir}/team_composition_dataset.csv")
            datasets['team_composition'] = team_composition
            print(f"Loaded team composition dataset: {team_composition.shape}")
        except FileNotFoundError:
            print("Team composition dataset not found. Run build_team_composition_dataset.py first.")
            return None
        
        # Load original simple enhanced dataset
        try:
            simple_enhanced = pd.read_csv(f"{self.data_dir}/simple_enhanced_train.csv")
            datasets['simple_enhanced'] = simple_enhanced
            print(f"Loaded simple enhanced dataset: {simple_enhanced.shape}")
        except FileNotFoundError:
            print("Simple enhanced dataset not found.")
            return None
        
        return datasets
    
    def combine_datasets(self, datasets):
        """Combine all datasets into one comprehensive dataset"""
        print("Combining datasets...")
        
        # Start with simple enhanced dataset as base
        base_df = datasets['simple_enhanced'].copy()
        print(f"Base dataset shape: {base_df.shape}")
        
        # Add player impact features
        player_impact = datasets['player_impact']
        
        # Aggregate player impact by match and team
        player_agg = player_impact.groupby(['match_id', 'team']).agg({
            'player_impact_score': ['mean', 'max', 'sum', 'std'],
            'career_batting_avg': 'mean',
            'career_batting_sr': 'mean',
            'career_bowling_avg': 'mean',
            'career_bowling_econ': 'mean',
            'career_matches': 'mean',
            'player_role': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        }).reset_index()
        
        # Flatten column names
        player_agg.columns = ['match_id', 'team', 'team_player_impact_mean', 'team_player_impact_max', 
                             'team_player_impact_sum', 'team_player_impact_std', 'team_batting_avg', 
                             'team_batting_sr', 'team_bowling_avg', 'team_bowling_econ', 
                             'team_experience', 'team_dominant_role']
        
        # Merge player impact features
        base_df = base_df.merge(player_agg, on=['match_id', 'team'], how='left')
        
        # Add venue conditions features
        venue_conditions = datasets['venue_conditions']
        venue_agg = venue_conditions.groupby('match_id').first().reset_index()
        
        # Merge venue conditions
        base_df = base_df.merge(venue_agg[['match_id', 'temperature', 'humidity', 'wind_speed', 
                                          'pitch_type', 'pitch_bounce', 'pitch_pace', 'pitch_turn', 
                                          'pitch_swing', 'weather_impact', 'pitch_difficulty']], 
                               on='match_id', how='left')
        
        # Add team composition features
        team_composition = datasets['team_composition']
        team_comp_agg = team_composition.groupby(['match_id', 'team']).first().reset_index()
        
        # Merge team composition
        team_comp_cols = ['match_id', 'team', 'team_balance', 'team_depth', 
                         'batting_ratio', 'bowling_ratio', 'all_rounder_ratio', 
                         'role_variety', 'team_chemistry', 'strategic_advantage']
        
        # Check which columns exist
        available_cols = [col for col in team_comp_cols if col in team_comp_agg.columns]
        base_df = base_df.merge(team_comp_agg[available_cols], 
                               on=['match_id', 'team'], how='left')
        
        print(f"Combined dataset shape: {base_df.shape}")
        return base_df
    
    def add_comprehensive_features(self, df):
        """Add comprehensive derived features"""
        print("Adding comprehensive derived features...")
        
        # Player impact features
        df['player_impact_ratio'] = df['team_player_impact_mean'] / (df['team_player_impact_mean'].mean() + 1)
        df['team_star_power'] = df['team_player_impact_max'] / (df['team_player_impact_max'].mean() + 1)
        df['team_consistency'] = 1 / (df['team_player_impact_std'] + 1)
        
        # Venue-player interaction
        df['venue_player_fit'] = df['pitch_type'].map({
            'flat': 1.2, 'spinning': 0.8, 'seaming': 1.0, 'balanced': 1.0
        }).fillna(1.0) * df['team_player_impact_mean']
        
        # Weather impact on performance
        temperature = df['temperature'] if 'temperature' in df.columns else 25
        humidity = df['humidity'] if 'humidity' in df.columns else 60
        wind_speed = df['wind_speed'] if 'wind_speed' in df.columns else 5
        team_impact = df['team_player_impact_mean'] if 'team_player_impact_mean' in df.columns else 0
        
        df['weather_performance_impact'] = (
            temperature * 0.1 +
            humidity * 0.05 +
            wind_speed * 0.2
        ) * team_impact
        
        # Team composition effectiveness
        team_balance = df['team_balance'] if 'team_balance' in df.columns else 0.5
        team_depth = df['team_depth'] if 'team_depth' in df.columns else 0.5
        role_variety = df['role_variety'] if 'role_variety' in df.columns else 0.5
        
        df['composition_effectiveness'] = (
            team_balance * 0.4 +
            team_depth * 0.3 +
            role_variety * 0.3
        )
        
        # Match context features
        if 'venue_avg_runs' in df.columns:
            df['is_high_scoring_venue'] = (df['venue_avg_runs'] > df['venue_avg_runs'].quantile(0.7)).astype(int)
            df['is_difficult_venue'] = (df['venue_avg_runs'] < df['venue_avg_runs'].quantile(0.3)).astype(int)
            df['is_balanced_venue'] = (df['venue_avg_runs'] >= df['venue_avg_runs'].quantile(0.3)) & \
                                     (df['venue_avg_runs'] <= df['venue_avg_runs'].quantile(0.7))
        else:
            df['is_high_scoring_venue'] = 0
            df['is_difficult_venue'] = 0
            df['is_balanced_venue'] = 1
        
        # Player experience impact
        team_experience = df['team_experience'] if 'team_experience' in df.columns else 0
        team_impact = df['team_player_impact_mean'] if 'team_player_impact_mean' in df.columns else 0
        df['experience_impact'] = team_experience * team_impact
        
        # Strategic advantage
        toss_impact = df['toss_impact'] if 'toss_impact' in df.columns else 0
        team_balance = df['team_balance'] if 'team_balance' in df.columns else 0.5
        team_impact = df['team_player_impact_mean'] if 'team_player_impact_mean' in df.columns else 0
        venue_fit = df['venue_player_fit'] if 'venue_player_fit' in df.columns else 0
        
        df['strategic_advantage_score'] = (
            toss_impact * 0.3 +
            team_balance * 0.3 +
            team_impact * 0.2 +
            venue_fit * 0.2
        )
        
        # Match outcome prediction features
        team_impact = df['team_player_impact_mean'] if 'team_player_impact_mean' in df.columns else 0
        venue_avg = df['venue_avg_runs'] if 'venue_avg_runs' in df.columns else 140
        team_balance = df['team_balance'] if 'team_balance' in df.columns else 0.5
        weather_impact = df['weather_impact'] if 'weather_impact' in df.columns else 0
        pitch_difficulty = df['pitch_difficulty'] if 'pitch_difficulty' in df.columns else 1
        team_experience = df['team_experience'] if 'team_experience' in df.columns else 0
        
        df['predicted_score'] = (
            team_impact * 0.3 +
            venue_avg * 0.2 +
            team_balance * 0.2 +
            weather_impact * 0.1 +
            pitch_difficulty * 0.1 +
            team_experience * 0.1
        )
        
        return df
    
    def clean_and_validate_dataset(self, df):
        """Clean and validate the final dataset"""
        print("Cleaning and validating dataset...")
        
        # Remove rows with missing critical data
        critical_columns = ['match_id', 'team', 'total_runs']
        df = df.dropna(subset=critical_columns)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Fill categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('unknown')
        
        # Remove duplicate records
        df = df.drop_duplicates(subset=['match_id', 'team'])
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        columns_to_drop = df.columns[df.isnull().mean() > missing_threshold]
        df = df.drop(columns=columns_to_drop)
        
        # Ensure target variable is present
        if 'total_runs' not in df.columns:
            print("ERROR: Target variable 'total_runs' not found!")
            return None
        
        print(f"Cleaned dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def create_final_dataset(self):
        """Create the final comprehensive dataset"""
        print("Creating Final Comprehensive Dataset")
        print("=" * 50)
        
        # Load individual datasets
        datasets = self.load_individual_datasets()
        if not datasets:
            print("Failed to load individual datasets!")
            return None
        
        # Combine datasets
        combined_df = self.combine_datasets(datasets)
        if combined_df is None:
            print("Failed to combine datasets!")
            return None
        
        # Add comprehensive features
        enhanced_df = self.add_comprehensive_features(combined_df)
        
        # Clean and validate
        final_df = self.clean_and_validate_dataset(enhanced_df)
        if final_df is None:
            print("Failed to clean dataset!")
            return None
        
        # Save final dataset
        output_file = f"{self.output_dir}/final_comprehensive_dataset.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"Final Comprehensive Dataset created: {output_file}")
        print(f"Final shape: {final_df.shape}")
        print(f"Features: {len(final_df.columns)}")
        
        # Show feature importance
        print("\nTop 20 features by correlation with total_runs:")
        correlations = final_df.select_dtypes(include=[np.number]).corrwith(final_df['total_runs']).abs().sort_values(ascending=False)
        for i, (feature, corr) in enumerate(correlations.head(20).items()):
            if feature != 'total_runs':
                print(f"  {i+1}. {feature}: {corr:.3f}")
        
        return final_df

def main():
    """Main function to create final comprehensive dataset"""
    print("Creating Final Comprehensive Dataset")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Initialize combiner
    combiner = FinalDatasetCombiner()
    
    # Create final dataset
    final_df = combiner.create_final_dataset()
    
    if final_df is not None:
        print("\nFinal Comprehensive Dataset created successfully!")
        print(f"Dataset contains {len(final_df)} records")
        print(f"Dataset has {len(final_df.columns)} features")
        print(f"Saved to: processed_data/final_comprehensive_dataset.csv")
        
        # Show sample
        print("\nSample data:")
        print(final_df.head(3).to_string())
        
        # Show target variable stats
        print(f"\nTarget variable (total_runs) stats:")
        print(f"  Mean: {final_df['total_runs'].mean():.2f}")
        print(f"  Std: {final_df['total_runs'].std():.2f}")
        print(f"  Min: {final_df['total_runs'].min()}")
        print(f"  Max: {final_df['total_runs'].max()}")
        
    else:
        print("Failed to create final comprehensive dataset")

if __name__ == "__main__":
    main()
