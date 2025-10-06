#!/usr/bin/env python3
"""
Analyze Enhanced Datasets
Evaluate the quality and usefulness of the processed datasets
"""

import pandas as pd
import numpy as np
import json

def analyze_enhanced_datasets():
    """Analyze the enhanced datasets created in processed_data folder"""
    print("🔬 ANALYZING ENHANCED DATASETS")
    print("=" * 70)
    
    # 1. Analyze ML-Ready Player Impact Dataset
    print(f"\n📊 ML-READY PLAYER IMPACT DATASET:")
    try:
        ml_ready_df = pd.read_csv('processed_data/ml_ready_player_impact_dataset.csv')
        print(f"  ✅ Shape: {ml_ready_df.shape}")
        print(f"  ✅ Features: {len(ml_ready_df.columns)}")
        print(f"  ✅ Target mean: {ml_ready_df['total_runs'].mean():.1f}")
        print(f"  ✅ Target std: {ml_ready_df['total_runs'].std():.1f}")
        
        # Check for player impact features
        player_features = [col for col in ml_ready_df.columns if 'player' in col.lower() or 'star' in col.lower()]
        print(f"  🎯 Player impact features: {len(player_features)}")
        print(f"    {player_features[:5]}...")
        
        # Check for venue conditions
        venue_features = [col for col in ml_ready_df.columns if any(x in col.lower() for x in ['temperature', 'humidity', 'pitch', 'weather'])]
        print(f"  🌤️ Venue condition features: {len(venue_features)}")
        print(f"    {venue_features[:5]}...")
        
        # Check for team composition
        team_features = [col for col in ml_ready_df.columns if any(x in col.lower() for x in ['team_', 'balance', 'chemistry', 'composition'])]
        print(f"  🏏 Team composition features: {len(team_features)}")
        print(f"    {team_features[:5]}...")
        
    except Exception as e:
        print(f"  ❌ Error loading ML-ready dataset: {e}")
    
    # 2. Analyze Team Composition Dataset
    print(f"\n🏏 TEAM COMPOSITION DATASET:")
    try:
        team_df = pd.read_csv('processed_data/team_composition_dataset.csv')
        print(f"  ✅ Shape: {team_df.shape}")
        print(f"  ✅ Features: {len(team_df.columns)}")
        
        # Check key team features
        key_features = ['team_balance', 'team_chemistry', 'strategic_advantage', 'batting_strength', 'bowling_strength']
        available_features = [f for f in key_features if f in team_df.columns]
        print(f"  🎯 Key team features available: {len(available_features)}/{len(key_features)}")
        print(f"    {available_features}")
        
        # Check for data quality
        missing_values = team_df.isnull().sum().sum()
        print(f"  📊 Missing values: {missing_values}")
        
        # Check target variable
        if 'total_runs' in team_df.columns:
            print(f"  🎯 Target mean: {team_df['total_runs'].mean():.1f}")
            print(f"  🎯 Target std: {team_df['total_runs'].std():.1f}")
        
    except Exception as e:
        print(f"  ❌ Error loading team composition dataset: {e}")
    
    # 3. Analyze Venue Conditions Dataset
    print(f"\n🌤️ VENUE CONDITIONS DATASET:")
    try:
        venue_df = pd.read_csv('processed_data/venue_conditions_dataset.csv')
        print(f"  ✅ Shape: {venue_df.shape}")
        print(f"  ✅ Features: {len(venue_df.columns)}")
        
        # Check weather features
        weather_features = [col for col in venue_df.columns if any(x in col.lower() for x in ['temperature', 'humidity', 'wind', 'precipitation', 'dew'])]
        print(f"  🌤️ Weather features: {len(weather_features)}")
        print(f"    {weather_features}")
        
        # Check pitch features
        pitch_features = [col for col in venue_df.columns if 'pitch' in col.lower()]
        print(f"  🏟️ Pitch features: {len(pitch_features)}")
        print(f"    {pitch_features}")
        
        # Check for realistic values
        if 'temperature' in venue_df.columns:
            temp_range = venue_df['temperature'].max() - venue_df['temperature'].min()
            print(f"  🌡️ Temperature range: {venue_df['temperature'].min():.1f}°C to {venue_df['temperature'].max():.1f}°C")
        
    except Exception as e:
        print(f"  ❌ Error loading venue conditions dataset: {e}")
    
    # 4. Analyze Final Comprehensive Dataset
    print(f"\n🎯 FINAL COMPREHENSIVE DATASET:")
    try:
        final_df = pd.read_csv('processed_data/final_comprehensive_dataset.csv')
        print(f"  ✅ Shape: {final_df.shape}")
        print(f"  ✅ Features: {len(final_df.columns)}")
        
        # Check for data leakage
        leakage_features = ['match_winner', 'player_of_match', 'is_final', 'is_semi_final']
        found_leakage = [f for f in leakage_features if f in final_df.columns]
        if found_leakage:
            print(f"  🚨 Data leakage features found: {found_leakage}")
        else:
            print(f"  ✅ No data leakage features found")
        
        # Check target variable
        if 'total_runs' in final_df.columns:
            print(f"  🎯 Target mean: {final_df['total_runs'].mean():.1f}")
            print(f"  🎯 Target std: {final_df['total_runs'].std():.1f}")
            print(f"  🎯 Target range: {final_df['total_runs'].min()} to {final_df['total_runs'].max()}")
        
        # Check feature categories
        numeric_features = final_df.select_dtypes(include=[np.number]).columns
        categorical_features = final_df.select_dtypes(include=['object']).columns
        print(f"  📊 Numeric features: {len(numeric_features)}")
        print(f"  📊 Categorical features: {len(categorical_features)}")
        
    except Exception as e:
        print(f"  ❌ Error loading final comprehensive dataset: {e}")
    
    # 5. Compare with original dataset
    print(f"\n📈 COMPARISON WITH ORIGINAL DATASET:")
    try:
        original_df = pd.read_csv('data/simple_enhanced_train_cleaned.csv')
        print(f"  Original dataset: {original_df.shape}")
        print(f"  Enhanced dataset: {final_df.shape}")
        print(f"  Feature increase: {len(final_df.columns) - len(original_df.columns)} features")
        print(f"  Feature ratio: {len(final_df.columns) / len(original_df.columns):.1f}x more features")
        
    except Exception as e:
        print(f"  ❌ Error comparing datasets: {e}")
    
    # 6. Feature Quality Analysis
    print(f"\n🔍 FEATURE QUALITY ANALYSIS:")
    
    try:
        # Check for constant features
        constant_features = []
        for col in final_df.select_dtypes(include=[np.number]).columns:
            if col != 'total_runs' and final_df[col].nunique() <= 2:
                constant_features.append(col)
        
        print(f"  Constant features: {len(constant_features)}")
        if constant_features:
            print(f"    {constant_features[:5]}...")
        
        # Check for missing values
        missing_data = final_df.isnull().sum()
        high_missing = missing_data[missing_data > len(final_df) * 0.1]
        print(f"  Features with >10% missing: {len(high_missing)}")
        
        # Check correlations with target
        if 'total_runs' in final_df.columns:
            numeric_features = final_df.select_dtypes(include=[np.number]).columns
            correlations = final_df[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
            
            strong_features = correlations[correlations > 0.3].drop('total_runs')
            print(f"  Strong correlations (>0.3): {len(strong_features)}")
            print(f"    Top 5: {strong_features.head().index.tolist()}")
            
    except Exception as e:
        print(f"  ❌ Error in feature quality analysis: {e}")
    
    # 7. Overall Assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    improvements = []
    
    # Check if player features are present
    if 'processed_data/ml_ready_player_impact_dataset.csv' in str(ml_ready_df.columns):
        improvements.append("✅ Player-level features added")
    else:
        improvements.append("❌ Player-level features missing")
    
    # Check if venue conditions are present
    if any('temperature' in col.lower() for col in final_df.columns):
        improvements.append("✅ Weather and pitch conditions added")
    else:
        improvements.append("❌ Weather and pitch conditions missing")
    
    # Check if team composition is present
    if any('team_balance' in col.lower() for col in final_df.columns):
        improvements.append("✅ Team composition features added")
    else:
        improvements.append("❌ Team composition features missing")
    
    # Check for data leakage
    if not any(f in final_df.columns for f in ['match_winner', 'player_of_match']):
        improvements.append("✅ Data leakage removed")
    else:
        improvements.append("❌ Data leakage still present")
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    # 8. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    recommendations = []
    
    # Check if we should use the enhanced dataset
    if len(final_df.columns) > 80:
        recommendations.append("✅ Use enhanced dataset for model training")
    else:
        recommendations.append("⚠️ Enhanced dataset may need more features")
    
    # Check for feature engineering opportunities
    if len(constant_features) > 0:
        recommendations.append(f"🔧 Remove {len(constant_features)} constant features")
    
    if len(high_missing) > 0:
        recommendations.append(f"🔧 Address {len(high_missing)} features with high missing values")
    
    # Check if we have enough strong features
    if len(strong_features) >= 10:
        recommendations.append("✅ Sufficient strong features for good model performance")
    else:
        recommendations.append("⚠️ May need more strong predictive features")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return {
        'ml_ready_shape': ml_ready_df.shape if 'ml_ready_df' in locals() else None,
        'team_shape': team_df.shape if 'team_df' in locals() else None,
        'venue_shape': venue_df.shape if 'venue_df' in locals() else None,
        'final_shape': final_df.shape if 'final_df' in locals() else None,
        'improvements': improvements,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_enhanced_datasets()
    
    print(f"\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if results['final_shape'] and results['final_shape'][1] > 80:
        print(f"✅ ENHANCED DATASET IS READY FOR TRAINING")
        print(f"   Features: {results['final_shape'][1]}")
        print(f"   Records: {results['final_shape'][0]:,}")
        print(f"   Improvements: {len([i for i in results['improvements'] if '✅' in i])}")
    else:
        print(f"⚠️ ENHANCED DATASET NEEDS REVIEW")
        print(f"   Current features: {results['final_shape'][1] if results['final_shape'] else 'Unknown'}")
        print(f"   Issues to address: {len([i for i in results['improvements'] if '❌' in i])}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Use enhanced dataset for model training")
    print(f"2. Test model performance with new features")
    print(f"3. Compare accuracy with original dataset")
    print(f"4. Fine-tune feature selection if needed")
