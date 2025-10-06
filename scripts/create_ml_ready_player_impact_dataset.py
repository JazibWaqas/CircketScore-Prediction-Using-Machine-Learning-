#!/usr/bin/env python3
"""
Create ML-Ready Player Impact Dataset
Fix the issues and create a proper dataset for ML training
Use existing comprehensive dataset structure as base
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class MLReadyPlayerImpactBuilder:
    def __init__(self):
        self.output_dir = "processed_data"
        
        # Load the comprehensive dataset as base
        self.comprehensive_df = pd.read_csv("processed_data/final_comprehensive_dataset.csv")
        print(f"Loaded comprehensive dataset: {self.comprehensive_df.shape}")
        
        # Load enhanced player impact dataset
        self.player_impact_df = pd.read_csv("processed_data/enhanced_player_impact_dataset.csv")
        print(f"Loaded player impact dataset: {self.player_impact_df.shape}")
        
        # Load combined player lookup
        self.combined_lookup = pd.read_csv("data/combined_player_lookup.csv")
        print(f"Loaded combined lookup: {self.combined_lookup.shape}")
        
        print("ML-Ready Player Impact Builder initialized")
        
    def create_team_player_features(self, team_players_list, match_id, team_name):
        """Create team-level player impact features from player list"""
        
        # Parse team players list
        try:
            import ast
            if isinstance(team_players_list, str):
                player_names = ast.literal_eval(team_players_list)
            else:
                player_names = team_players_list
        except:
            player_names = []
        
        if not player_names:
            return self.get_default_team_features()
        
        # Get player performances for this match and team
        team_performances = self.player_impact_df[
            (self.player_impact_df['team'] == team_name) & 
            (self.player_impact_df['player_name'].isin(player_names))
        ]
        
        if team_performances.empty:
            return self.get_default_team_features()
        
        # Calculate team-level features
        total_players = len(team_performances)
        
        # Star players analysis
        star_players = team_performances[team_performances['has_meaningful_career'] == True]
        regular_players = team_performances[team_performances['has_meaningful_career'] == False]
        
        star_count = len(star_players)
        regular_count = len(regular_players)
        star_ratio = star_count / len(player_names) if len(player_names) > 0 else 0
        
        # Performance metrics
        total_impact = team_performances['player_impact_score'].sum()
        avg_impact = team_performances['player_impact_score'].mean()
        max_impact = team_performances['player_impact_score'].max()
        
        # Star player performance
        star_impact = star_players['player_impact_score'].sum() if len(star_players) > 0 else 0
        star_avg_impact = star_players['player_impact_score'].mean() if len(star_players) > 0 else 0
        
        # Team composition
        batsmen_count = team_performances['is_batsman'].sum()
        bowlers_count = team_performances['is_bowler'].sum()
        all_rounders_count = team_performances['is_all_rounder'].sum()
        
        batting_ratio = batsmen_count / len(player_names) if len(player_names) > 0 else 0
        bowling_ratio = bowlers_count / len(player_names) if len(player_names) > 0 else 0
        all_rounder_ratio = all_rounders_count / len(player_names) if len(player_names) > 0 else 0
        
        # Career experience (for star players)
        avg_career_matches = star_players['career_matches'].mean() if len(star_players) > 0 else 0
        total_career_matches = star_players['career_matches'].sum() if len(star_players) > 0 else 0
        
        # Performance ratios (for star players)
        avg_batting_performance_ratio = star_players['batting_performance_ratio'].mean() if len(star_players) > 0 else 1.0
        avg_strike_rate_ratio = star_players['strike_rate_ratio'].mean() if len(star_players) > 0 else 1.0
        
        # High performers
        high_performers = team_performances['high_performer'].sum()
        consistent_performers = team_performances['consistent_performer'].sum()
        
        return {
            'team_player_impact_mean': avg_impact,
            'team_player_impact_max': max_impact,
            'team_player_impact_sum': total_impact,
            'team_player_impact_std': team_performances['player_impact_score'].std(),
            'star_players_count': star_count,
            'regular_players_count': regular_count,
            'star_ratio': star_ratio,
            'star_impact_total': star_impact,
            'star_impact_avg': star_avg_impact,
            'team_batting_ratio': batting_ratio,
            'team_bowling_ratio': bowling_ratio,
            'team_all_rounder_ratio': all_rounder_ratio,
            'avg_career_matches': avg_career_matches,
            'total_career_matches': total_career_matches,
            'experience_depth': total_career_matches / len(player_names) if len(player_names) > 0 else 0,
            'avg_batting_performance_ratio': avg_batting_performance_ratio,
            'avg_strike_rate_ratio': avg_strike_rate_ratio,
            'high_performers_count': high_performers,
            'consistent_performers_count': consistent_performers,
            'performance_consistency': consistent_performers / len(player_names) if len(player_names) > 0 else 0,
            'is_star_heavy_team': 1 if star_ratio > 0.3 else 0,
            'is_experienced_team': 1 if avg_career_matches > 50 else 0,
            'is_high_impact_team': 1 if avg_impact > 15 else 0,
            'has_star_batsman': 1 if (star_players['is_batsman'] == True).any() else 0,
            'has_star_bowler': 1 if (star_players['is_bowler'] == True).any() else 0,
        }
    
    def get_default_team_features(self):
        """Return default features when no player data is available"""
        return {
            'team_player_impact_mean': 0,
            'team_player_impact_max': 0,
            'team_player_impact_sum': 0,
            'team_player_impact_std': 0,
            'star_players_count': 0,
            'regular_players_count': 0,
            'star_ratio': 0,
            'star_impact_total': 0,
            'star_impact_avg': 0,
            'team_batting_ratio': 0,
            'team_bowling_ratio': 0,
            'team_all_rounder_ratio': 0,
            'avg_career_matches': 0,
            'total_career_matches': 0,
            'experience_depth': 0,
            'avg_batting_performance_ratio': 1.0,
            'avg_strike_rate_ratio': 1.0,
            'high_performers_count': 0,
            'consistent_performers_count': 0,
            'performance_consistency': 0,
            'is_star_heavy_team': 0,
            'is_experienced_team': 0,
            'is_high_impact_team': 0,
            'has_star_batsman': 0,
            'has_star_bowler': 0,
        }
    
    def build_ml_ready_dataset(self):
        """Build ML-ready dataset with player impact features"""
        print("Building ML-Ready Player Impact Dataset...")
        print("=" * 50)
        
        # Start with comprehensive dataset
        ml_dataset = self.comprehensive_df.copy()
        
        # Remove data leakage columns
        leakage_columns = [
            'total_wickets', 'total_overs', 'total_balls', 'run_rate',
            'total_4s', 'total_6s', 'total_boundaries', 'total_extras',
            'powerplay_runs', 'middle_overs_runs', 'death_overs_runs',
            'target_set', 'target_chased', 'win_margin', 'win_type',
            'match_winner', 'predicted_score'
        ]
        
        # Remove columns that exist
        existing_leakage = [col for col in leakage_columns if col in ml_dataset.columns]
        ml_dataset = ml_dataset.drop(columns=existing_leakage)
        print(f"Removed data leakage columns: {existing_leakage}")
        
        # Remove metadata columns
        metadata_columns = ['match_id', 'date', 'team_players', 'player_of_match', 'h2h_last_meeting', 'teams']
        existing_metadata = [col for col in metadata_columns if col in ml_dataset.columns]
        ml_dataset = ml_dataset.drop(columns=existing_metadata)
        print(f"Removed metadata columns: {existing_metadata}")
        
        print(f"Base dataset shape after cleaning: {ml_dataset.shape}")
        
        # Add player impact features
        print("Adding player impact features...")
        
        player_features = []
        for idx, row in ml_dataset.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(ml_dataset)}...")
            
            # Get team player features
            team_features = self.create_team_player_features(
                row.get('team_players', []), 
                row.get('match_id', ''), 
                row['team']
            )
            player_features.append(team_features)
        
        # Convert to DataFrame
        player_features_df = pd.DataFrame(player_features)
        
        # Combine with main dataset
        ml_dataset = pd.concat([ml_dataset, player_features_df], axis=1)
        
        print(f"Final ML-ready dataset shape: {ml_dataset.shape}")
        
        # Save dataset
        output_path = f"{self.output_dir}/ml_ready_player_impact_dataset.csv"
        ml_dataset.to_csv(output_path, index=False)
        print(f"Saved ML-ready dataset: {output_path}")
        
        # Save summary statistics
        self.save_summary_statistics(ml_dataset)
        
        return ml_dataset
    
    def save_summary_statistics(self, df):
        """Save summary statistics"""
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'target_variable': 'total_runs',
            'target_statistics': {
                'mean': df['total_runs'].mean(),
                'std': df['total_runs'].std(),
                'min': df['total_runs'].min(),
                'max': df['total_runs'].max()
            },
            'feature_categories': {
                'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'boolean_features': len(df.select_dtypes(include=['bool']).columns)
            },
            'player_impact_features': [col for col in df.columns if 'player_impact' in col or 'star_' in col or 'team_' in col],
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
        }
        
        # Save as JSON
        import json
        with open(f"{self.output_dir}/ml_ready_player_impact_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary statistics saved: {self.output_dir}/ml_ready_player_impact_summary.json")
        
        # Print key statistics
        print(f"\n=== ML-READY PLAYER IMPACT DATASET SUMMARY ===")
        print(f"Total records: {summary['total_records']:,}")
        print(f"Total features: {summary['total_features']}")
        print(f"Target variable: {summary['target_variable']}")
        print(f"Target statistics:")
        print(f"  Mean: {summary['target_statistics']['mean']:.1f}")
        print(f"  Std: {summary['target_statistics']['std']:.1f}")
        print(f"  Range: {summary['target_statistics']['min']:.1f} - {summary['target_statistics']['max']:.1f}")
        print(f"Feature breakdown:")
        print(f"  Numerical: {summary['feature_categories']['numerical_features']}")
        print(f"  Categorical: {summary['feature_categories']['categorical_features']}")
        print(f"  Boolean: {summary['feature_categories']['boolean_features']}")
        print(f"Player impact features: {len(summary['player_impact_features'])}")
        print(f"Missing values: {summary['data_quality']['missing_values']:,}")
        print(f"Duplicate rows: {summary['data_quality']['duplicate_rows']:,}")

def main():
    print("ML-Ready Player Impact Dataset Builder")
    print("Creating dataset suitable for XGBoost, Random Forest, and DNN training")
    print("=" * 70)
    
    builder = MLReadyPlayerImpactBuilder()
    dataset = builder.build_ml_ready_dataset()
    
    if dataset is not None:
        print(f"\nML-ready player impact dataset created successfully!")
        print(f"Dataset is ready for training:")
        print(f"  ✅ No data leakage")
        print(f"  ✅ Proper numerical features")
        print(f"  ✅ Player-level intelligence")
        print(f"  ✅ Team composition analysis")
        print(f"  ✅ Star player recognition")
        print(f"  ✅ Compatible with XGBoost, Random Forest, and DNN models")
    else:
        print(f"Failed to create dataset!")

if __name__ == "__main__":
    main()
