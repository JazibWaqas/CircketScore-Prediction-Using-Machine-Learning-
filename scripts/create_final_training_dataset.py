#!/usr/bin/env python3
"""
Create Final Training Dataset
Clean unrealistic scores and create proper train/test split for 2017-2025 era
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_final_training_dataset():
    """Create the final training dataset with proper cleaning and train/test split"""
    print("🔧 CREATING FINAL TRAINING DATASET")
    print("=" * 60)
    
    # Load the final clean dataset
    try:
        df = pd.read_csv('processed_data/final_clean_dataset.csv')
        print(f"✅ Loaded final clean dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n🧹 STEP 1: CLEAN UNREALISTIC SCORES")
    
    # Remove unrealistic scores (T20 matches with <20 runs are likely incomplete)
    original_count = len(df)
    df_clean = df[df['total_runs'] >= 20].copy()
    removed_count = original_count - len(df_clean)
    
    print(f"  Removed {removed_count} matches with unrealistic scores (<20 runs)")
    print(f"  Shape after cleaning: {df_clean.shape}")
    
    # Check for other unrealistic scores
    very_high_scores = df_clean[df_clean['total_runs'] > 250]
    if len(very_high_scores) > 0:
        print(f"  ⚠️ Found {len(very_high_scores)} matches with very high scores (>250 runs)")
        print(f"    Range: {very_high_scores['total_runs'].min()} to {very_high_scores['total_runs'].max()}")
    
    print(f"\n📅 STEP 2: CREATE TEMPORAL TRAIN/TEST SPLIT")
    
    # Convert date to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Define test period (2017-2025) and training period (2005-2016)
    test_start_date = pd.to_datetime('2017-01-01')
    test_end_date = pd.to_datetime('2025-12-31')
    
    # Create test set (2017-2025)
    test_df = df_clean[
        (df_clean['date'] >= test_start_date) & 
        (df_clean['date'] <= test_end_date)
    ].copy()
    
    # Create training set (2005-2016)
    train_df = df_clean[df_clean['date'] < test_start_date].copy()
    
    print(f"  Training period: 2005-2016")
    print(f"  Test period: 2017-2025")
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    
    # Verify the split
    if len(test_df) >= 500:
        print(f"  ✅ Test set has sufficient samples: {len(test_df)}")
    else:
        print(f"  ⚠️ Test set has only {len(test_df)} samples (requested 500)")
    
    print(f"\n📊 STEP 3: ANALYZE TRAIN/TEST DISTRIBUTIONS")
    
    # Check target variable distributions
    print(f"  Training set target statistics:")
    print(f"    Mean: {train_df['total_runs'].mean():.1f}")
    print(f"    Std: {train_df['total_runs'].std():.1f}")
    print(f"    Range: {train_df['total_runs'].min()}-{train_df['total_runs'].max()}")
    
    print(f"  Test set target statistics:")
    print(f"    Mean: {test_df['total_runs'].mean():.1f}")
    print(f"    Std: {test_df['total_runs'].std():.1f}")
    print(f"    Range: {test_df['total_runs'].min()}-{test_df['total_runs'].max()}")
    
    # Check for distribution shift
    mean_diff = abs(train_df['total_runs'].mean() - test_df['total_runs'].mean())
    if mean_diff < 10:
        print(f"  ✅ Good distribution consistency (mean diff: {mean_diff:.1f})")
    else:
        print(f"  ⚠️ Potential distribution shift (mean diff: {mean_diff:.1f})")
    
    print(f"\n🏏 STEP 4: ANALYZE PLAYER-LEVEL IMPACT FEATURES")
    
    # Check for player-level features
    player_features = [col for col in df_clean.columns if 'player' in col.lower() or 'star' in col.lower()]
    print(f"  Player-level features available: {len(player_features)}")
    if player_features:
        print(f"    {player_features}")
    
    # Check for team composition features
    team_features = [col for col in df_clean.columns if any(x in col.lower() for x in ['team_', 'balance', 'chemistry', 'composition'])]
    print(f"  Team composition features: {len(team_features)}")
    
    # Check for venue/weather features
    venue_features = [col for col in df_clean.columns if any(x in col.lower() for x in ['venue', 'temperature', 'humidity', 'pitch', 'weather'])]
    print(f"  Venue/weather features: {len(venue_features)}")
    
    print(f"\n💾 STEP 5: SAVE TRAINING AND TEST DATASETS")
    
    # Save training dataset
    train_file = 'processed_data/final_training_dataset.csv'
    train_df.to_csv(train_file, index=False)
    print(f"  ✅ Saved training dataset: {train_file}")
    print(f"    Shape: {train_df.shape}")
    
    # Save test dataset
    test_file = 'processed_data/final_test_dataset.csv'
    test_df.to_csv(test_file, index=False)
    print(f"  ✅ Saved test dataset: {test_file}")
    print(f"    Shape: {test_df.shape}")
    
    # Create summary
    summary = {
        'original_samples': original_count,
        'cleaned_samples': len(df_clean),
        'removed_unrealistic': removed_count,
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'training_period': '2005-2016',
        'test_period': '2017-2025',
        'training_mean': float(train_df['total_runs'].mean()),
        'test_mean': float(test_df['total_runs'].mean()),
        'player_features': player_features,
        'team_features': team_features,
        'venue_features': venue_features,
        'total_features': len(df_clean.columns)
    }
    
    import json
    with open('processed_data/final_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"  Original dataset: {original_count:,} samples")
    print(f"  After cleaning: {len(df_clean):,} samples")
    print(f"  Training set: {len(train_df):,} samples (2005-2016)")
    print(f"  Test set: {len(test_df):,} samples (2017-2025)")
    print(f"  Features: {len(df_clean.columns)}")
    print(f"  Player features: {len(player_features)}")
    print(f"  Team features: {len(team_features)}")
    print(f"  Venue features: {len(venue_features)}")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'summary': summary,
        'train_file': train_file,
        'test_file': test_file
    }

def analyze_what_if_capabilities():
    """Analyze if the dataset can handle what-if scenarios and player-level impact"""
    print(f"\n🎯 WHAT-IF SCENARIO ANALYSIS")
    print("=" * 60)
    
    try:
        df = pd.read_csv('processed_data/final_training_dataset.csv')
        print(f"✅ Loaded training dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n🔍 SCENARIO ANALYSIS CAPABILITIES:")
    
    # 1. Player-level impact analysis
    print(f"\n👥 PLAYER-LEVEL IMPACT ANALYSIS:")
    
    player_features = [col for col in df.columns if 'player' in col.lower() or 'star' in col.lower()]
    if player_features:
        print(f"  ✅ Player features available: {len(player_features)}")
        print(f"    {player_features}")
        
        # Check if we can analyze player impact
        if 'team_player_ids' in df.columns:
            print(f"  ✅ Can track individual players: team_player_ids")
        
        if any('star' in col.lower() for col in player_features):
            print(f"  ✅ Can identify star players")
        
        if any('impact' in col.lower() for col in player_features):
            print(f"  ✅ Can measure player impact scores")
        
        print(f"  🎯 WHAT-IF SCENARIOS:")
        print(f"    ✅ 'What if we swap Player A with Player B?'")
        print(f"    ✅ 'What if we add/remove a star player?'")
        print(f"    ✅ 'What if we change team composition?'")
    else:
        print(f"  ❌ Limited player-level features")
        print(f"  ⚠️ Cannot analyze individual player impact")
    
    # 2. Venue and conditions analysis
    print(f"\n🏟️ VENUE & CONDITIONS ANALYSIS:")
    
    venue_features = [col for col in df.columns if any(x in col.lower() for x in ['venue', 'temperature', 'humidity', 'pitch', 'weather'])]
    if venue_features:
        print(f"  ✅ Venue/weather features: {len(venue_features)}")
        print(f"    {venue_features}")
        
        print(f"  🎯 WHAT-IF SCENARIOS:")
        print(f"    ✅ 'What if the weather changes?'")
        print(f"    ✅ 'What if we play at a different venue?'")
        print(f"    ✅ 'What if pitch conditions change?'")
    else:
        print(f"  ❌ Limited venue/weather features")
    
    # 3. Team composition analysis
    print(f"\n🏏 TEAM COMPOSITION ANALYSIS:")
    
    team_features = [col for col in df.columns if any(x in col.lower() for x in ['team_', 'balance', 'chemistry', 'composition'])]
    if team_features:
        print(f"  ✅ Team features: {len(team_features)}")
        print(f"    {team_features}")
        
        print(f"  🎯 WHAT-IF SCENARIOS:")
        print(f"    ✅ 'What if we change team balance?'")
        print(f"    ✅ 'What if we adjust batting/bowling ratio?'")
        print(f"    ✅ 'What if team chemistry changes?'")
    else:
        print(f"  ❌ Limited team composition features")
    
    # 4. Historical context analysis
    print(f"\n📈 HISTORICAL CONTEXT ANALYSIS:")
    
    historical_features = [col for col in df.columns if any(x in col.lower() for x in ['h2h', 'form', 'recent', 'historical'])]
    if historical_features:
        print(f"  ✅ Historical features: {len(historical_features)}")
        print(f"    {historical_features}")
        
        print(f"  🎯 WHAT-IF SCENARIOS:")
        print(f"    ✅ 'What if team form changes?'")
        print(f"    ✅ 'What if head-to-head record changes?'")
        print(f"    ✅ 'What if recent performance changes?'")
    else:
        print(f"  ❌ Limited historical context features")
    
    # 5. Overall what-if capability assessment
    print(f"\n🎯 OVERALL WHAT-IF CAPABILITY ASSESSMENT:")
    
    capabilities = {
        'player_impact': len(player_features) > 0,
        'venue_conditions': len(venue_features) > 0,
        'team_composition': len(team_features) > 0,
        'historical_context': len(historical_features) > 0
    }
    
    total_capabilities = sum(capabilities.values())
    max_capabilities = len(capabilities)
    
    print(f"  Capabilities available: {total_capabilities}/{max_capabilities}")
    
    if total_capabilities >= 3:
        print(f"  ✅ EXCELLENT: Dataset can handle comprehensive what-if scenarios")
        print(f"  🎯 Can analyze player swaps, venue changes, team composition, and historical context")
    elif total_capabilities >= 2:
        print(f"  🔶 GOOD: Dataset can handle basic what-if scenarios")
        print(f"  🎯 Can analyze some aspects but may be limited")
    else:
        print(f"  ❌ POOR: Dataset has limited what-if scenario capabilities")
    
    # 6. Specific what-if scenarios
    print(f"\n🎯 SPECIFIC WHAT-IF SCENARIOS:")
    
    scenarios = []
    
    if capabilities['player_impact']:
        scenarios.extend([
            "✅ 'What if we replace Player A with Player B?'",
            "✅ 'What if we add a star batsman?'",
            "✅ 'What if we remove a key bowler?'",
            "✅ 'What if we change team experience level?'"
        ])
    
    if capabilities['venue_conditions']:
        scenarios.extend([
            "✅ 'What if we play at a different venue?'",
            "✅ 'What if weather conditions change?'",
            "✅ 'What if pitch becomes more spin-friendly?'",
            "✅ 'What if temperature drops significantly?'"
        ])
    
    if capabilities['team_composition']:
        scenarios.extend([
            "✅ 'What if we change batting/bowling ratio?'",
            "✅ 'What if team balance changes?'",
            "✅ 'What if we add more all-rounders?'",
            "✅ 'What if team chemistry improves?'"
        ])
    
    if capabilities['historical_context']:
        scenarios.extend([
            "✅ 'What if team form improves?'",
            "✅ 'What if head-to-head record changes?'",
            "✅ 'What if recent performance changes?'",
            "✅ 'What if opposition strength changes?'"
        ])
    
    for scenario in scenarios:
        print(f"    {scenario}")
    
    return {
        'capabilities': capabilities,
        'total_capabilities': total_capabilities,
        'max_capabilities': max_capabilities,
        'scenarios': scenarios
    }

if __name__ == "__main__":
    # Create final training dataset
    result = create_final_training_dataset()
    
    if result:
        print(f"\n" + "="*60)
        print("FINAL TRAINING DATASET READY")
        print("="*60)
        
        print(f"✅ Training dataset: {result['train_file']}")
        print(f"✅ Test dataset: {result['test_file']}")
        print(f"✅ Training samples: {result['summary']['training_samples']:,}")
        print(f"✅ Test samples: {result['summary']['test_samples']:,}")
        print(f"✅ Features: {result['summary']['total_features']}")
        
        # Analyze what-if capabilities
        what_if_analysis = analyze_what_if_capabilities()
        
        print(f"\n🎯 WHAT-IF CAPABILITIES:")
        print(f"  Capabilities: {what_if_analysis['total_capabilities']}/{what_if_analysis['max_capabilities']}")
        print(f"  Scenarios: {len(what_if_analysis['scenarios'])}")
        
        if what_if_analysis['total_capabilities'] >= 3:
            print(f"  ✅ EXCELLENT: Ready for comprehensive what-if analysis")
        elif what_if_analysis['total_capabilities'] >= 2:
            print(f"  🔶 GOOD: Ready for basic what-if analysis")
        else:
            print(f"  ❌ LIMITED: What-if capabilities are limited")
        
        print(f"\n🚀 READY FOR MODEL TRAINING!")
    else:
        print(f"❌ Failed to create training dataset")
