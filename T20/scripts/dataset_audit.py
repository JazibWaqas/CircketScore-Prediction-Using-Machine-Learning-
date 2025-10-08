#!/usr/bin/env python3
"""
Comprehensive Dataset Audit for Cricket Prediction Model
Data Scientist Analysis for Deep Learning Readiness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

def comprehensive_dataset_audit():
    """Perform comprehensive dataset audit for deep learning readiness"""
    print("🔬 COMPREHENSIVE DATASET AUDIT FOR DEEP LEARNING")
    print("=" * 80)
    
    # Load the final training dataset
    try:
        df = pd.read_csv('processed_data/final_training_dataset.csv')
        print(f"✅ Loaded training dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 1. MISSING & INVALID VALUES
    print(f"\n🔍 1. MISSING & INVALID VALUES ANALYSIS")
    print("-" * 50)
    
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    print("Missing values by column:")
    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percentage.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    # Show columns with missing values
    missing_columns = missing_summary[missing_summary['Missing_Count'] > 0]
    if len(missing_columns) > 0:
        print("\n⚠️ Columns with missing values:")
        for _, row in missing_columns.iterrows():
            print(f"  {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
    else:
        print("✅ No missing values found!")
    
    # Check for invalid values in numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    invalid_values = {}
    
    for col in numeric_columns:
        inf_count = np.isinf(df[col]).sum()
        nan_count = np.isnan(df[col]).sum()
        if inf_count > 0 or nan_count > 0:
            invalid_values[col] = {'inf': inf_count, 'nan': nan_count}
    
    if invalid_values:
        print("\n⚠️ Invalid values found:")
        for col, counts in invalid_values.items():
            print(f"  {col}: {counts['inf']} inf, {counts['nan']} nan")
    else:
        print("✅ No invalid values (inf/nan) found!")
    
    # 2. DATA LEAKAGE CHECK
    print(f"\n🚨 2. DATA LEAKAGE CHECK")
    print("-" * 50)
    
    leakage_indicators = [
        'match_winner', 'player_of_match', 'is_final', 'is_semi_final',
        'winner', 'loser', 'result', 'outcome', 'score_a', 'score_b',
        'total_score', 'match_result', 'winning_team', 'losing_team',
        'powerplay_runs', 'death_overs_runs', 'final_score'
    ]
    
    found_leakage = []
    for indicator in leakage_indicators:
        if indicator in df.columns:
            found_leakage.append(indicator)
    
    if found_leakage:
        print("❌ CRITICAL: Data leakage features found:")
        for feature in found_leakage:
            print(f"  - {feature}")
        print("🚨 These must be removed before training!")
    else:
        print("✅ No data leakage features detected!")
    
    # Check if target variable is present and not used as feature
    target_candidates = ['total_runs', 'score', 'runs', 'target']
    target_variable = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_variable = candidate
            break
    
    if target_variable:
        print(f"✅ Target variable identified: {target_variable}")
    else:
        print("⚠️ No clear target variable found!")
    
    # 3. FEATURE DISTRIBUTION & OUTLIERS
    print(f"\n📈 3. FEATURE DISTRIBUTION & OUTLIERS ANALYSIS")
    print("-" * 50)
    
    print("Numerical features analysis:")
    numeric_summary = df[numeric_columns].describe()
    
    # Check for extreme outliers (beyond 3 standard deviations)
    outlier_analysis = {}
    for col in numeric_columns:
        if col != target_variable:  # Skip target variable
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_analysis[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'range': f"{df[col].min():.2f} to {df[col].max():.2f}",
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
    
    if outlier_analysis:
        print("\n⚠️ Features with significant outliers (>3 IQR):")
        for col, stats in outlier_analysis.items():
            print(f"  {col}: {stats['count']} outliers ({stats['percentage']:.1f}%)")
            print(f"    Range: {stats['range']}, Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    else:
        print("✅ No significant outliers detected!")
    
    # Scaling recommendations
    print("\n📏 Scaling recommendations:")
    for col in numeric_columns:
        if col != target_variable:
            col_std = df[col].std()
            col_mean = df[col].mean()
            col_range = df[col].max() - df[col].min()
            
            if col_range > 100 or col_std > 50:
                print(f"  {col}: RECOMMENDED (range: {col_range:.1f}, std: {col_std:.1f})")
    
    # 4. CATEGORICAL VARIABLES
    print(f"\n🏷️ 4. CATEGORICAL VARIABLES ANALYSIS")
    print("-" * 50)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"Found {len(categorical_columns)} categorical columns:")
    
    categorical_analysis = {}
    for col in categorical_columns:
        unique_count = df[col].nunique()
        most_common = df[col].value_counts().iloc[0]
        most_common_pct = (most_common / len(df)) * 100
        
        categorical_analysis[col] = {
            'unique_count': unique_count,
            'most_common': most_common,
            'most_common_pct': most_common_pct,
            'sparsity': (unique_count / len(df)) * 100
        }
        
        print(f"  {col}: {unique_count} unique values")
        print(f"    Most common: {most_common} ({most_common_pct:.1f}%)")
        print(f"    Sparsity: {categorical_analysis[col]['sparsity']:.1f}%")
    
    # Encoding recommendations
    print("\n🔤 Encoding recommendations:")
    for col, stats in categorical_analysis.items():
        if stats['unique_count'] < 10:
            print(f"  {col}: One-hot encoding (low cardinality)")
        elif stats['unique_count'] < 50:
            print(f"  {col}: Label encoding or embeddings (medium cardinality)")
        else:
            print(f"  {col}: Embeddings or hash encoding (high cardinality)")
    
    # 5. FEATURE CORRELATION
    print(f"\n🔗 5. FEATURE CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Calculate correlation matrix for numerical features
    correlation_matrix = df[numeric_columns].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        print("⚠️ Highly correlated features (>0.8):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        print("Consider removing one from each pair to reduce redundancy.")
    else:
        print("✅ No highly correlated features found!")
    
    # 6. TARGET VARIABLE QUALITY
    print(f"\n🎯 6. TARGET VARIABLE QUALITY ANALYSIS")
    print("-" * 50)
    
    if target_variable:
        target_data = df[target_variable]
        
        print(f"Target variable: {target_variable}")
        print(f"  Mean: {target_data.mean():.2f}")
        print(f"  Median: {target_data.median():.2f}")
        print(f"  Std: {target_data.std():.2f}")
        print(f"  Min: {target_data.min():.2f}")
        print(f"  Max: {target_data.max():.2f}")
        print(f"  Range: {target_data.max() - target_data.min():.2f}")
        
        # Check for skewness
        skewness = skew(target_data)
        print(f"  Skewness: {skewness:.3f}")
        
        if abs(skewness) > 1:
            print("  ⚠️ Highly skewed - consider log transformation")
        elif abs(skewness) > 0.5:
            print("  ⚠️ Moderately skewed - consider transformation")
        else:
            print("  ✅ Approximately normal distribution")
        
        # Check for unrealistic values
        unrealistic_low = target_data[target_data < 20]
        unrealistic_high = target_data[target_data > 300]
        
        if len(unrealistic_low) > 0:
            print(f"  ⚠️ {len(unrealistic_low)} values < 20 (possibly incomplete matches)")
        if len(unrealistic_high) > 0:
            print(f"  ⚠️ {len(unrealistic_high)} values > 300 (very high scores)")
        
        if len(unrealistic_low) == 0 and len(unrealistic_high) == 0:
            print("  ✅ All values appear realistic for T20 cricket")
    
    # 7. DATASET BALANCE
    print(f"\n⚖️ 7. DATASET BALANCE ANALYSIS")
    print("-" * 50)
    
    # Check team balance
    if 'team' in df.columns:
        team_counts = df['team'].value_counts()
        print("Team distribution:")
        print(f"  Most frequent team: {team_counts.iloc[0]} ({team_counts.iloc[0]/len(df)*100:.1f}%)")
        print(f"  Least frequent team: {team_counts.iloc[-1]} ({team_counts.iloc[-1]/len(df)*100:.1f}%)")
        print(f"  Teams represented: {len(team_counts)}")
        
        # Check for imbalance
        max_team_pct = team_counts.iloc[0] / len(df) * 100
        if max_team_pct > 30:
            print("  ⚠️ High team imbalance - consider stratified sampling")
        else:
            print("  ✅ Reasonable team balance")
    
    # Check venue balance
    if 'venue' in df.columns:
        venue_counts = df['venue'].value_counts()
        print(f"\nVenue distribution:")
        print(f"  Unique venues: {len(venue_counts)}")
        print(f"  Most frequent venue: {venue_counts.iloc[0]} ({venue_counts.iloc[0]/len(df)*100:.1f}%)")
        
        # Check for venue dominance
        max_venue_pct = venue_counts.iloc[0] / len(df) * 100
        if max_venue_pct > 20:
            print("  ⚠️ Some venues dominate the dataset")
        else:
            print("  ✅ Good venue diversity")
    
    # Check temporal balance
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].max() - df['date'].min()
        print(f"\nTemporal distribution:")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Time span: {date_range.days} days")
        print(f"  Matches per year: {len(df) / (date_range.days / 365.25):.1f}")
    
    # 8. TRAIN-TEST READINESS
    print(f"\n🎯 8. TRAIN-TEST READINESS")
    print("-" * 50)
    
    print(f"Dataset size: {len(df):,} samples")
    
    if len(df) < 1000:
        print("  ⚠️ Small dataset - may need more data for reliable deep learning")
    elif len(df) < 5000:
        print("  🔶 Moderate dataset - suitable for deep learning with regularization")
    else:
        print("  ✅ Large dataset - good for deep learning training")
    
    # Check feature-to-sample ratio
    feature_count = len(df.columns) - 1  # Exclude target
    sample_to_feature_ratio = len(df) / feature_count
    
    print(f"Feature count: {feature_count}")
    print(f"Sample-to-feature ratio: {sample_to_feature_ratio:.1f}")
    
    if sample_to_feature_ratio < 10:
        print("  ⚠️ Low ratio - risk of overfitting, consider feature selection")
    elif sample_to_feature_ratio < 50:
        print("  🔶 Moderate ratio - use regularization")
    else:
        print("  ✅ Good ratio for deep learning")
    
    # FINAL VERDICT
    print(f"\n🏁 FINAL VERDICT")
    print("=" * 80)
    
    # Scoring system
    score = 0
    max_score = 8
    
    # Missing values (1 point)
    if len(missing_columns) == 0:
        score += 1
        print("✅ Missing values: Clean")
    else:
        print("⚠️ Missing values: Issues found")
    
    # Data leakage (2 points)
    if len(found_leakage) == 0:
        score += 2
        print("✅ Data leakage: None detected")
    else:
        print("❌ Data leakage: Critical issues found")
    
    # Outliers (1 point)
    if len(outlier_analysis) == 0:
        score += 1
        print("✅ Outliers: No significant issues")
    else:
        print("⚠️ Outliers: Some features have outliers")
    
    # Categorical encoding (1 point)
    high_cardinality = sum(1 for stats in categorical_analysis.values() if stats['unique_count'] > 50)
    if high_cardinality == 0:
        score += 1
        print("✅ Categorical variables: Manageable")
    else:
        print("⚠️ Categorical variables: Some high cardinality")
    
    # Correlation (1 point)
    if len(high_corr_pairs) == 0:
        score += 1
        print("✅ Feature correlation: No redundancy")
    else:
        print("⚠️ Feature correlation: Some redundancy")
    
    # Target quality (1 point)
    if target_variable and len(unrealistic_low) == 0 and len(unrealistic_high) == 0:
        score += 1
        print("✅ Target variable: Good quality")
    else:
        print("⚠️ Target variable: Some issues")
    
    # Dataset balance (1 point)
    if max_team_pct < 30 and max_venue_pct < 20:
        score += 1
        print("✅ Dataset balance: Reasonable")
    else:
        print("⚠️ Dataset balance: Some imbalance")
    
    print(f"\n📊 Overall Score: {score}/{max_score}")
    
    if score >= 7:
        verdict = "✅ READY FOR TRAINING"
        print(f"\n{verdict}")
        print("Dataset is well-prepared for deep learning model training.")
    elif score >= 5:
        verdict = "⚠️ NEEDS CLEANING"
        print(f"\n{verdict}")
        print("Dataset has some issues that should be addressed before training.")
    else:
        verdict = "❌ NOT SUITABLE FOR RELIABLE PREDICTION YET"
        print(f"\n{verdict}")
        print("Dataset has significant issues that must be resolved.")
    
    # Feature recommendations
    print(f"\n💡 FEATURE RECOMMENDATIONS:")
    print("-" * 50)
    
    useful_features = []
    drop_features = []
    
    # Recommend dropping highly correlated features
    for pair in high_corr_pairs:
        drop_features.append(pair['feature2'])  # Drop the second feature
    
    # Recommend keeping features with good variance
    for col in numeric_columns:
        if col != target_variable:
            if df[col].std() > 1:  # Good variance
                useful_features.append(col)
    
    print("✅ Keep these features (good variance, no correlation):")
    for feature in useful_features[:10]:  # Show top 10
        print(f"  - {feature}")
    
    if drop_features:
        print("\n🗑️ Consider dropping these features (redundant):")
        for feature in drop_features[:5]:  # Show top 5
            print(f"  - {feature}")
    
    return {
        'verdict': verdict,
        'score': score,
        'max_score': max_score,
        'missing_values': len(missing_columns),
        'data_leakage': len(found_leakage),
        'outliers': len(outlier_analysis),
        'high_correlation': len(high_corr_pairs),
        'useful_features': useful_features,
        'drop_features': drop_features
    }

if __name__ == "__main__":
    results = comprehensive_dataset_audit()
    
    print(f"\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    
    if results['verdict'].startswith("✅"):
        print("🎉 Dataset is ready for deep learning model training!")
    elif results['verdict'].startswith("⚠️"):
        print("🔧 Dataset needs some cleaning before training.")
    else:
        print("🚨 Dataset requires significant improvements before training.")
