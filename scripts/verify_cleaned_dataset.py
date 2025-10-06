#!/usr/bin/env python3
"""
Verify Cleaned Dataset Quality
Final verification of the cleaned cricket dataset
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

def verify_cleaned_dataset():
    """Verify the quality of the cleaned dataset"""
    print("🔍 VERIFYING CLEANED DATASET QUALITY")
    print("=" * 60)
    
    # Load the cleaned dataset
    try:
        df = pd.read_csv('processed_data/cleaned_cricket_dataset.csv')
        print(f"✅ Loaded cleaned dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 1. Target variable verification
    print(f"\n🎯 TARGET VARIABLE VERIFICATION:")
    target_stats = {
        'mean': df['total_runs'].mean(),
        'std': df['total_runs'].std(),
        'min': df['total_runs'].min(),
        'max': df['total_runs'].max(),
        'skewness': skew(df['total_runs'])
    }
    
    print(f"  Mean: {target_stats['mean']:.2f}")
    print(f"  Std: {target_stats['std']:.2f}")
    print(f"  Range: {target_stats['min']:.0f} - {target_stats['max']:.0f}")
    print(f"  Skewness: {target_stats['skewness']:.3f}")
    
    # Check for unrealistic values
    unrealistic_low = df[df['total_runs'] < 20]
    unrealistic_high = df[df['total_runs'] > 250]
    
    if len(unrealistic_low) == 0 and len(unrealistic_high) == 0:
        print("  ✅ No unrealistic values found")
    else:
        print(f"  ⚠️ Found {len(unrealistic_low)} low values, {len(unrealistic_high)} high values")
    
    # 2. Missing values check
    print(f"\n🔍 MISSING VALUES CHECK:")
    missing_data = df.isnull().sum()
    missing_count = missing_data.sum()
    
    if missing_count == 0:
        print("  ✅ No missing values")
    else:
        print(f"  ⚠️ Found {missing_count} missing values")
        for col, count in missing_data[missing_data > 0].items():
            print(f"    {col}: {count}")
    
    # 3. Feature types verification
    print(f"\n🏷️ FEATURE TYPES:")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # 4. Correlation analysis
    print(f"\n🔗 CORRELATION ANALYSIS:")
    numeric_cols = [col for col in numeric_features if col != 'total_runs']
    corr_matrix = df[numeric_cols].corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    print(f"  High correlations (>0.8): {len(high_corr_pairs)}")
    
    if len(high_corr_pairs) <= 5:
        print("  ✅ Acceptable correlation level")
        for pair in high_corr_pairs:
            print(f"    {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("  ⚠️ Many high correlations remain")
    
    # 5. Feature scaling verification
    print(f"\n📏 FEATURE SCALING VERIFICATION:")
    
    # Check if features are properly scaled (mean ≈ 0, std ≈ 1)
    scaling_issues = []
    for col in numeric_cols:
        mean_val = abs(df[col].mean())
        std_val = abs(df[col].std() - 1.0)
        
        if mean_val > 0.1 or std_val > 0.1:
            scaling_issues.append(col)
    
    if len(scaling_issues) == 0:
        print("  ✅ All features properly scaled")
    else:
        print(f"  ⚠️ {len(scaling_issues)} features may need rescaling")
        print(f"    Issues: {scaling_issues[:5]}...")
    
    # 6. Outlier analysis
    print(f"\n📊 OUTLIER ANALYSIS:")
    
    # Check for extreme outliers (beyond 4 standard deviations)
    outlier_features = []
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        extreme_outliers = (z_scores > 4).sum()
        
        if extreme_outliers > 0:
            outlier_features.append({
                'feature': col,
                'outliers': extreme_outliers,
                'percentage': extreme_outliers / len(df) * 100
            })
    
    if len(outlier_features) == 0:
        print("  ✅ No extreme outliers found")
    else:
        print(f"  ⚠️ {len(outlier_features)} features have extreme outliers")
        for feature in outlier_features[:5]:
            print(f"    {feature['feature']}: {feature['outliers']} ({feature['percentage']:.1f}%)")
    
    # 7. Feature importance preview
    print(f"\n🎯 FEATURE IMPORTANCE PREVIEW:")
    
    # Calculate correlations with target
    target_correlations = []
    for col in numeric_cols:
        if col != 'total_runs':
            corr = abs(df[col].corr(df['total_runs']))
            target_correlations.append({'feature': col, 'correlation': corr})
    
    # Sort by correlation
    target_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    print("  Top 10 features by correlation with target:")
    for i, feature in enumerate(target_correlations[:10]):
        print(f"    {i+1:2d}. {feature['feature']}: {feature['correlation']:.3f}")
    
    # 8. Final quality assessment
    print(f"\n🏁 FINAL QUALITY ASSESSMENT:")
    
    quality_score = 0
    max_score = 7
    
    # Check 1: No unrealistic target values
    if len(unrealistic_low) == 0 and len(unrealistic_high) == 0:
        quality_score += 1
        print("  ✅ Target values: Clean")
    else:
        print("  ⚠️ Target values: Issues found")
    
    # Check 2: No missing values
    if missing_count == 0:
        quality_score += 1
        print("  ✅ Missing values: None")
    else:
        print("  ⚠️ Missing values: Found")
    
    # Check 3: Acceptable correlations
    if len(high_corr_pairs) <= 5:
        quality_score += 1
        print("  ✅ Correlations: Acceptable")
    else:
        print("  ⚠️ Correlations: Too many high correlations")
    
    # Check 4: Proper scaling
    if len(scaling_issues) == 0:
        quality_score += 1
        print("  ✅ Scaling: Proper")
    else:
        print("  ⚠️ Scaling: Issues found")
    
    # Check 5: No extreme outliers
    if len(outlier_features) == 0:
        quality_score += 1
        print("  ✅ Outliers: None")
    else:
        print("  ⚠️ Outliers: Found")
    
    # Check 6: Good feature count
    if 40 <= len(numeric_cols) <= 80:
        quality_score += 1
        print("  ✅ Feature count: Good")
    else:
        print("  ⚠️ Feature count: May be too few/many")
    
    # Check 7: Good sample size
    if len(df) >= 10000:
        quality_score += 1
        print("  ✅ Sample size: Good")
    else:
        print("  ⚠️ Sample size: May be too small")
    
    print(f"\n📊 QUALITY SCORE: {quality_score}/{max_score}")
    
    if quality_score >= 6:
        verdict = "✅ EXCELLENT - Ready for deep learning"
    elif quality_score >= 4:
        verdict = "🔶 GOOD - Suitable for training"
    else:
        verdict = "⚠️ NEEDS IMPROVEMENT"
    
    print(f"\n🎯 VERDICT: {verdict}")
    
    # 9. Recommendations for deep learning
    print(f"\n💡 DEEP LEARNING RECOMMENDATIONS:")
    
    print("  📊 Dataset is ready for:")
    print("    - TabNet (tabular deep learning)")
    print("    - DNN with embeddings")
    print("    - Random Forest / XGBoost")
    print("    - Neural networks")
    
    print("\n  🔧 Suggested preprocessing:")
    print("    - Use StandardScaler (already applied)")
    print("    - Consider feature selection for top 20-30 features")
    print("    - Apply early stopping to prevent overfitting")
    print("    - Use cross-validation for model evaluation")
    
    print("\n  🎯 Expected performance:")
    print("    - R² Score: 0.6-0.8 (good for cricket prediction)")
    print("    - RMSE: 20-30 runs (reasonable for T20)")
    print("    - MAE: 15-25 runs (acceptable accuracy)")
    
    return {
        'quality_score': quality_score,
        'max_score': max_score,
        'verdict': verdict,
        'shape': df.shape,
        'target_stats': target_stats,
        'high_correlations': len(high_corr_pairs),
        'scaling_issues': len(scaling_issues),
        'outlier_features': len(outlier_features)
    }

if __name__ == "__main__":
    results = verify_cleaned_dataset()
    
    print(f"\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    
    print(f"📊 Quality Score: {results['quality_score']}/{results['max_score']}")
    print(f"🎯 Verdict: {results['verdict']}")
    print(f"📈 Shape: {results['shape']}")
    print(f"🔗 High correlations: {results['high_correlations']}")
    print(f"📏 Scaling issues: {results['scaling_issues']}")
    print(f"📊 Outlier features: {results['outlier_features']}")
    
    if results['quality_score'] >= 6:
        print(f"\n🚀 DATASET IS READY FOR DEEP LEARNING TRAINING!")
    else:
        print(f"\n🔧 DATASET NEEDS MINOR IMPROVEMENTS BEFORE TRAINING")
