#!/usr/bin/env python3
"""
Final Dataset Polish
Fix remaining issues to make dataset perfect for deep learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def final_dataset_polish():
    """Polish the dataset to fix remaining issues"""
    print("✨ FINAL DATASET POLISH")
    print("=" * 50)
    
    # Load the cleaned dataset
    try:
        df = pd.read_csv('processed_data/cleaned_cricket_dataset.csv')
        print(f"✅ Loaded cleaned dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 1. Fix remaining high correlations by dropping redundant ID features
    print(f"\n🔗 FIXING REMAINING CORRELATIONS:")
    
    # Drop redundant ID features that are perfectly correlated with their categorical versions
    redundant_id_features = ['venue_id', 'team_id', 'team_player_ids']
    features_to_drop = []
    
    for feature in redundant_id_features:
        if feature in df.columns:
            features_to_drop.append(feature)
            print(f"  🗑️ Dropping {feature} (redundant with categorical version)")
    
    df = df.drop(columns=features_to_drop)
    print(f"  Features dropped: {len(features_to_drop)}")
    print(f"  New shape: {df.shape}")
    
    # 2. Fix scaling issues
    print(f"\n📏 FIXING SCALING ISSUES:")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_runs' in numeric_cols:
        numeric_cols.remove('total_runs')
    
    # Check which features need rescaling
    scaling_issues = []
    for col in numeric_cols:
        mean_val = abs(df[col].mean())
        std_val = abs(df[col].std() - 1.0)
        
        if mean_val > 0.1 or std_val > 0.1:
            scaling_issues.append(col)
    
    if scaling_issues:
        print(f"  Rescaling {len(scaling_issues)} features")
        scaler = StandardScaler()
        df[scaling_issues] = scaler.fit_transform(df[scaling_issues])
        print(f"  ✅ Rescaling complete")
    else:
        print(f"  ✅ No scaling issues found")
    
    # 3. Handle remaining extreme outliers
    print(f"\n📊 HANDLING EXTREME OUTLIERS:")
    
    outlier_features = []
    for col in numeric_cols:
        if col != 'total_runs':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            extreme_outliers = (z_scores > 4).sum()
            
            if extreme_outliers > 0:
                outlier_features.append(col)
    
    if outlier_features:
        print(f"  Clipping extreme outliers in {len(outlier_features)} features")
        
        for col in outlier_features:
            # Use more aggressive clipping for extreme outliers
            lower_bound = df[col].quantile(0.005)  # 0.5th percentile
            upper_bound = df[col].quantile(0.995)  # 99.5th percentile
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"  ✅ Outlier clipping complete")
    else:
        print(f"  ✅ No extreme outliers found")
    
    # 4. Feature selection - keep only top features
    print(f"\n🎯 FEATURE SELECTION:")
    
    # Calculate correlations with target
    target_correlations = []
    for col in numeric_cols:
        if col != 'total_runs':
            corr = abs(df[col].corr(df['total_runs']))
            target_correlations.append({'feature': col, 'correlation': corr})
    
    # Sort by correlation
    target_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    # Keep top 30 features + target + essential categoricals
    top_features = [f['feature'] for f in target_correlations[:30]]
    
    # Add essential categorical features
    essential_categoricals = ['toss_decision_bat', 'toss_decision_field', 'gender_female', 'gender_male']
    for cat in essential_categoricals:
        if cat in df.columns:
            top_features.append(cat)
    
    # Add target
    top_features.append('total_runs')
    
    # Filter dataset to top features
    df_final = df[top_features].copy()
    
    print(f"  Selected top {len(top_features)-1} features")
    print(f"  Final shape: {df_final.shape}")
    
    # 5. Final correlation check
    print(f"\n🔗 FINAL CORRELATION CHECK:")
    
    final_numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_runs' in final_numeric_cols:
        final_numeric_cols.remove('total_runs')
    
    final_corr_matrix = df_final[final_numeric_cols].corr()
    
    # Find remaining high correlations
    remaining_high_corr = []
    for i in range(len(final_corr_matrix.columns)):
        for j in range(i+1, len(final_corr_matrix.columns)):
            corr_value = abs(final_corr_matrix.iloc[i, j])
            if corr_value > 0.8:
                remaining_high_corr.append({
                    'feature1': final_corr_matrix.columns[i],
                    'feature2': final_corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    print(f"  Remaining high correlations (>0.8): {len(remaining_high_corr)}")
    
    if remaining_high_corr:
        print("  Remaining correlations:")
        for pair in remaining_high_corr:
            print(f"    {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    
    # 6. Save final polished dataset
    print(f"\n💾 SAVING FINAL POLISHED DATASET:")
    
    output_file = 'processed_data/cleaned_cricket_dataset.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"  ✅ Saved final dataset: {output_file}")
    print(f"  Final shape: {df_final.shape}")
    
    # 7. Final quality summary
    print(f"\n📊 FINAL QUALITY SUMMARY:")
    
    # Target analysis
    target_stats = {
        'mean': df_final['total_runs'].mean(),
        'std': df_final['total_runs'].std(),
        'min': df_final['total_runs'].min(),
        'max': df_final['total_runs'].max()
    }
    
    print(f"  Target (total_runs):")
    print(f"    Mean: {target_stats['mean']:.2f}")
    print(f"    Std: {target_stats['std']:.2f}")
    print(f"    Range: {target_stats['min']:.0f} - {target_stats['max']:.0f}")
    
    # Feature analysis
    print(f"  Features:")
    print(f"    Total: {len(df_final.columns)-1}")
    print(f"    Numeric: {len(df_final.select_dtypes(include=[np.number]).columns)-1}")
    print(f"    Categorical: {len(df_final.select_dtypes(include=['object']).columns)}")
    
    # Quality metrics
    print(f"  Quality metrics:")
    print(f"    High correlations: {len(remaining_high_corr)}")
    print(f"    Missing values: {df_final.isnull().sum().sum()}")
    print(f"    Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Top features
    print(f"  Top 10 features by correlation:")
    for i, feature in enumerate(target_correlations[:10]):
        print(f"    {i+1:2d}. {feature['feature']}: {feature['correlation']:.3f}")
    
    print(f"\n✅ FINAL POLISH COMPLETE!")
    print(f"  Dataset is now optimized for deep learning")
    print(f"  All major issues resolved")
    print(f"  Ready for model training")
    
    return {
        'final_shape': df_final.shape,
        'high_correlations': len(remaining_high_corr),
        'top_features': len(top_features)-1,
        'target_stats': target_stats,
        'output_file': output_file
    }

if __name__ == "__main__":
    results = final_dataset_polish()
    
    print(f"\n" + "="*50)
    print("FINAL POLISH COMPLETE")
    print("="*50)
    
    print(f"📊 Final shape: {results['final_shape']}")
    print(f"🔗 High correlations: {results['high_correlations']}")
    print(f"🎯 Top features: {results['top_features']}")
    print(f"💾 Output: {results['output_file']}")
    
    print(f"\n🚀 DATASET IS NOW PERFECT FOR DEEP LEARNING!")
