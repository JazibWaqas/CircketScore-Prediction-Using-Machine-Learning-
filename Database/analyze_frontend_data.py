#!/usr/bin/env python3
"""
Analyze what the frontend currently sends vs what the models expect
"""

import pandas as pd

def analyze_frontend_vs_model_data():
    """Compare frontend data structure with model requirements"""
    print("🔍 FRONTEND DATA vs MODEL REQUIREMENTS ANALYSIS")
    print("=" * 60)
    
    # Load training data to see what models expect
    train_df = pd.read_csv('../data/simple_enhanced_train.csv')
    feature_cols = [col for col in train_df.columns if col != 'total_runs']
    
    print("📊 MODEL EXPECTS (from simple_enhanced_train.csv):")
    print(f"Total features: {len(feature_cols)}")
    print()
    
    # Categorize features by type
    categorical_features = []
    numerical_features = []
    boolean_features = []
    
    for col in feature_cols:
        dtype = train_df[col].dtype
        if dtype == 'bool':
            boolean_features.append(col)
        elif dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"🔢 NUMERICAL FEATURES ({len(numerical_features)}):")
    for i, col in enumerate(numerical_features[:10]):
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        print(f"  {i+1:2d}. {col:<25} | Range: {min_val:.2f}-{max_val:.2f}")
    
    print(f"\n📝 CATEGORICAL FEATURES ({len(categorical_features)}):")
    for i, col in enumerate(categorical_features[:10]):
        unique_count = train_df[col].nunique()
        print(f"  {i+1:2d}. {col:<25} | {unique_count} unique values")
    
    print(f"\n✅ BOOLEAN FEATURES ({len(boolean_features)}):")
    for i, col in enumerate(boolean_features):
        unique_vals = train_df[col].unique()
        print(f"  {i+1:2d}. {col:<25} | Values: {unique_vals}")
    
    print(f"\n🎯 FRONTEND CURRENTLY SENDS:")
    print("=" * 40)
    
    # Simulate frontend data
    frontend_data = {
        'team_a_id': 117,
        'team_b_id': 64,
        'venue_id': 478,
        'team_a_players': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'team_b_players': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'match_context': {
            'battingFirst': 'team_a',
            'tossWinner': 'team_a',
            'tossDecision': 'bat',
            'isHomeTeam': True,
            'isFinal': False,
            'isT20WorldCup': False,
            'isImportantMatch': True,
            'seasonYear': 2025,
            'seasonMonth': 9,
            'isSummer': True,
            'tournamentType': 'Asia Cup'
        },
        'model': 'xgboost'
    }
    
    print("Frontend sends:")
    print(f"  team_a_id: {frontend_data['team_a_id']} (int)")
    print(f"  team_b_id: {frontend_data['team_b_id']} (int)")
    print(f"  venue_id: {frontend_data['venue_id']} (int)")
    print(f"  team_a_players: {frontend_data['team_a_players']} (list of ints)")
    print(f"  team_b_players: {frontend_data['team_b_players']} (list of ints)")
    print(f"  match_context: {frontend_data['match_context']}")
    print(f"  model: {frontend_data['model']} (string)")
    
    print(f"\n❌ GAP ANALYSIS:")
    print("=" * 40)
    
    # Check what we can map directly
    mappable_features = []
    missing_features = []
    
    # Direct mappings from frontend
    direct_mappings = {
        'venue_id': 'venue_id from frontend',
        'team_id': 'team_a_id from frontend',
        'season_year': 'seasonYear from match_context',
        'season_month': 'seasonMonth from match_context',
        'batting_first': 'battingFirst from match_context',
        'toss_decision': 'tossDecision from match_context',
        'is_home_team': 'isHomeTeam from match_context',
        'is_final': 'isFinal from match_context',
        'is_t20_world_cup': 'isT20WorldCup from match_context',
        'is_summer': 'isSummer from match_context',
        'is_winter': 'can calculate from seasonMonth',
    }
    
    for feature in feature_cols:
        if feature in direct_mappings:
            mappable_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"✅ MAPPABLE FEATURES ({len(mappable_features)}):")
    for feature in mappable_features[:10]:
        mapping = direct_mappings.get(feature, 'unknown')
        print(f"  {feature:<25} <- {mapping}")
    
    print(f"\n❌ MISSING FEATURES ({len(missing_features)}):")
    for feature in missing_features[:15]:
        print(f"  {feature}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 40)
    print("1. Use venue_statistics.pkl for venue-related features")
    print("2. Use database to look up team names from IDs")
    print("3. Calculate missing features from available data")
    print("4. Use encoders to convert strings to numbers")
    print("5. Apply proper feature scaling")

if __name__ == "__main__":
    analyze_frontend_vs_model_data()
