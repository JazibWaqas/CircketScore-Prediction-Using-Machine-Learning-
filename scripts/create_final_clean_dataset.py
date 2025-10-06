#!/usr/bin/env python3
"""
Create Final Clean Dataset
Fix data leakage, remove constant features, and create the ultimate training dataset
"""

import pandas as pd
import numpy as np

def create_final_clean_dataset():
    """Create the final, clean dataset for model training"""
    print("🔧 CREATING FINAL CLEAN DATASET")
    print("=" * 60)
    
    # Load the enhanced dataset
    try:
        df = pd.read_csv('processed_data/final_comprehensive_dataset.csv')
        print(f"✅ Loaded enhanced dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n🧹 STEP 1: REMOVE DATA LEAKAGE")
    
    # Remove data leakage features
    data_leakage_features = [
        'match_winner', 'player_of_match', 'is_final', 'is_semi_final'
    ]
    
    existing_leakage = [f for f in data_leakage_features if f in df.columns]
    df_clean = df.drop(columns=existing_leakage)
    
    print(f"  Removed {len(existing_leakage)} data leakage features: {existing_leakage}")
    print(f"  Shape after removal: {df_clean.shape}")
    
    print(f"\n🔧 STEP 2: REMOVE CONSTANT FEATURES")
    
    # Identify constant features
    constant_features = []
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col != 'total_runs' and df_clean[col].nunique() <= 2:
            constant_features.append(col)
    
    # Also check for features with very low variance
    low_variance_features = []
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col != 'total_runs' and df_clean[col].std() < 0.01:
            low_variance_features.append(col)
    
    # Remove constant and low-variance features
    features_to_remove = list(set(constant_features + low_variance_features))
    df_clean = df_clean.drop(columns=features_to_remove)
    
    print(f"  Removed {len(features_to_remove)} constant/low-variance features")
    print(f"  Shape after removal: {df_clean.shape}")
    
    print(f"\n📊 STEP 3: FEATURE QUALITY ANALYSIS")
    
    # Check for missing values
    missing_data = df_clean.isnull().sum()
    high_missing = missing_data[missing_data > len(df_clean) * 0.05]  # >5% missing
    
    if len(high_missing) > 0:
        print(f"  ⚠️ Features with >5% missing values: {len(high_missing)}")
        for feature, count in high_missing.items():
            percentage = count / len(df_clean) * 100
            print(f"    {feature}: {count} ({percentage:.1f}%)")
    else:
        print(f"  ✅ No features with high missing values")
    
    # Check correlations with target
    if 'total_runs' in df_clean.columns:
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns
        correlations = df_clean[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
        
        strong_features = correlations[correlations > 0.3]
        if 'total_runs' in strong_features.index:
            strong_features = strong_features.drop('total_runs')
        
        moderate_features = correlations[(correlations > 0.1) & (correlations <= 0.3)]
        if 'total_runs' in moderate_features.index:
            moderate_features = moderate_features.drop('total_runs')
        
        weak_features = correlations[correlations <= 0.1]
        if 'total_runs' in weak_features.index:
            weak_features = weak_features.drop('total_runs')
        
        print(f"  Strong correlations (>0.3): {len(strong_features)}")
        print(f"  Moderate correlations (0.1-0.3): {len(moderate_features)}")
        print(f"  Weak correlations (<0.1): {len(weak_features)}")
        
        print(f"  Top 10 strongest features:")
        for feature, corr in strong_features.head(10).items():
            print(f"    {feature}: {corr:.3f}")
    
    print(f"\n🎯 STEP 4: FEATURE CATEGORIZATION")
    
    # Categorize features by type
    feature_categories = {
        'VENUE': [],
        'TEAM': [],
        'PLAYER': [],
        'WEATHER': [],
        'PITCH': [],
        'HISTORICAL': [],
        'MATCH_CONTEXT': [],
        'DERIVED': []
    }
    
    for col in df_clean.columns:
        if col == 'total_runs':
            continue
        elif any(x in col.lower() for x in ['venue', 'stadium']):
            feature_categories['VENUE'].append(col)
        elif any(x in col.lower() for x in ['team', 'batting', 'bowling']):
            feature_categories['TEAM'].append(col)
        elif any(x in col.lower() for x in ['player', 'star']):
            feature_categories['PLAYER'].append(col)
        elif any(x in col.lower() for x in ['temperature', 'humidity', 'wind', 'weather']):
            feature_categories['WEATHER'].append(col)
        elif 'pitch' in col.lower():
            feature_categories['PITCH'].append(col)
        elif any(x in col.lower() for x in ['h2h', 'form', 'recent']):
            feature_categories['HISTORICAL'].append(col)
        elif any(x in col.lower() for x in ['toss', 'first', 'important']):
            feature_categories['MATCH_CONTEXT'].append(col)
        else:
            feature_categories['DERIVED'].append(col)
    
    for category, features in feature_categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
            print(f"    {features[:3]}...")
    
    print(f"\n💾 STEP 5: SAVE CLEAN DATASET")
    
    # Save the clean dataset
    output_file = 'processed_data/final_clean_dataset.csv'
    df_clean.to_csv(output_file, index=False)
    
    print(f"  ✅ Saved clean dataset: {output_file}")
    print(f"  Final shape: {df_clean.shape}")
    print(f"  Features removed: {len(existing_leakage) + len(features_to_remove)}")
    print(f"  Final features: {len(df_clean.columns)}")
    
    # Create a summary
    summary = {
        'original_shape': df.shape,
        'final_shape': df_clean.shape,
        'features_removed': len(existing_leakage) + len(features_to_remove),
        'data_leakage_removed': existing_leakage,
        'constant_features_removed': features_to_remove,
        'strong_correlations': len(strong_features) if 'strong_features' in locals() else 0,
        'feature_categories': {k: len(v) for k, v in feature_categories.items() if v}
    }
    
    import json
    with open('processed_data/final_clean_dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📋 CLEANING SUMMARY:")
    print(f"  Original features: {df.shape[1]}")
    print(f"  Final features: {df_clean.shape[1]}")
    print(f"  Features removed: {summary['features_removed']}")
    print(f"  Strong correlations: {summary['strong_correlations']}")
    print(f"  Data leakage removed: {len(existing_leakage)}")
    print(f"  Constant features removed: {len(features_to_remove)}")
    
    return {
        'clean_df': df_clean,
        'summary': summary,
        'output_file': output_file
    }

if __name__ == "__main__":
    result = create_final_clean_dataset()
    
    print(f"\n" + "="*60)
    print("FINAL CLEAN DATASET READY")
    print("="*60)
    
    if result:
        print(f"✅ Clean dataset created: {result['output_file']}")
        print(f"   Shape: {result['clean_df'].shape}")
        print(f"   Features: {result['clean_df'].shape[1]}")
        print(f"   Ready for model training!")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Train models with clean dataset")
        print(f"2. Test model performance")
        print(f"3. Compare with original dataset")
        print(f"4. Deploy improved models")
    else:
        print(f"❌ Failed to create clean dataset")
