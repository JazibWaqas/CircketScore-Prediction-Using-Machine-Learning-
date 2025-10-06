#!/usr/bin/env python3
"""
Fix Data Leakage in Training Dataset
Remove post-match features that cause data leakage
"""

import pandas as pd
import numpy as np

def fix_data_leakage():
    """Remove data leakage features from training dataset"""
    print("🔧 FIXING DATA LEAKAGE IN TRAINING DATASET")
    print("=" * 60)
    
    # Load the original training dataset
    try:
        train_df = pd.read_csv('data/simple_enhanced_train.csv')
        print(f"✅ Loaded original dataset: {train_df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Identify data leakage features (post-match information)
    data_leakage_features = [
        'match_winner',      # Who won the match
        'player_of_match',   # Who was player of the match
        'is_final',          # Whether it was a final (could be derived from context)
        'is_semi_final',     # Whether it was a semi-final
    ]
    
    # Check which features actually exist in the dataset
    existing_leakage_features = [col for col in data_leakage_features if col in train_df.columns]
    
    print(f"\n🚨 DATA LEAKAGE FEATURES FOUND:")
    for feature in existing_leakage_features:
        print(f"  ❌ {feature}")
        if feature == 'match_winner':
            print(f"      Values: {train_df[feature].value_counts().head()}")
    
    # Create cleaned dataset
    print(f"\n🧹 CREATING CLEANED DATASET:")
    
    # Remove data leakage features
    cleaned_df = train_df.drop(columns=existing_leakage_features)
    
    print(f"  Removed {len(existing_leakage_features)} data leakage features")
    print(f"  Original shape: {train_df.shape}")
    print(f"  Cleaned shape: {cleaned_df.shape}")
    
    # Check for any other potentially suspicious features
    print(f"\n🔍 CHECKING FOR OTHER POTENTIAL ISSUES:")
    
    # Features that might contain future information
    suspicious_patterns = ['result', 'outcome', 'final_score', 'actual_score']
    suspicious_features = []
    
    for col in cleaned_df.columns:
        if any(pattern in col.lower() for pattern in suspicious_patterns):
            suspicious_features.append(col)
    
    if suspicious_features:
        print(f"  ⚠️ Potentially suspicious features: {suspicious_features}")
    else:
        print(f"  ✅ No other suspicious features found")
    
    # Verify the target variable is still present
    if 'total_runs' in cleaned_df.columns:
        print(f"  ✅ Target variable 'total_runs' preserved")
        print(f"      Mean: {cleaned_df['total_runs'].mean():.1f}")
        print(f"      Std: {cleaned_df['total_runs'].std():.1f}")
    else:
        print(f"  ❌ ERROR: Target variable 'total_runs' missing!")
        return
    
    # Save the cleaned dataset
    output_file = 'data/simple_enhanced_train_cleaned.csv'
    cleaned_df.to_csv(output_file, index=False)
    print(f"\n💾 SAVED CLEANED DATASET:")
    print(f"  File: {output_file}")
    print(f"  Shape: {cleaned_df.shape}")
    
    # Create a summary of what was removed
    removal_summary = {
        'original_features': len(train_df.columns),
        'cleaned_features': len(cleaned_df.columns),
        'removed_features': len(existing_leakage_features),
        'removed_feature_names': existing_leakage_features,
        'target_variable_preserved': 'total_runs' in cleaned_df.columns,
        'rows_preserved': len(cleaned_df)
    }
    
    print(f"\n📋 REMOVAL SUMMARY:")
    print(f"  Original features: {removal_summary['original_features']}")
    print(f"  Cleaned features: {removal_summary['cleaned_features']}")
    print(f"  Removed features: {removal_summary['removed_features']}")
    print(f"  Rows preserved: {removal_summary['rows_preserved']}")
    
    # Check feature correlations with target after cleaning
    print(f"\n📈 FEATURE-TARGET CORRELATIONS (AFTER CLEANING):")
    
    numeric_features = cleaned_df.select_dtypes(include=[np.number]).columns
    correlations = cleaned_df[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
    
    print(f"  Top 10 features correlated with total_runs:")
    for feature, corr in correlations.head(11).items():
        if feature != 'total_runs':
            print(f"    {feature}: {corr:.3f}")
    
    # Compare with original correlations
    print(f"\n🔍 CORRELATION COMPARISON:")
    original_correlations = train_df[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
    
    print(f"  Original top 3 correlations:")
    for feature, corr in original_correlations.head(4).items():
        if feature != 'total_runs':
            print(f"    {feature}: {corr:.3f}")
    
    print(f"  Cleaned top 3 correlations:")
    for feature, corr in correlations.head(4).items():
        if feature != 'total_runs':
            print(f"    {feature}: {corr:.3f}")
    
    return {
        'cleaned_df': cleaned_df,
        'removal_summary': removal_summary,
        'output_file': output_file
    }

if __name__ == "__main__":
    result = fix_data_leakage()
    
    print(f"\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"1. ✅ Data leakage removed from training dataset")
    print(f"2. 🔄 Retrain models with cleaned dataset")
    print(f"3. 🧪 Test model accuracy on cleaned data")
    print(f"4. 📊 Compare performance before/after cleaning")
    print(f"5. 🎯 Use cleaned models for real predictions")
    
    if result:
        print(f"\n📁 Cleaned dataset saved to: {result['output_file']}")
        print(f"   Ready for retraining models without data leakage!")
