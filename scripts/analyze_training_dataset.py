#!/usr/bin/env python3
"""
Analyze Training Dataset Quality
Check if simple_enhanced_train.csv has critical issues that would affect model training
"""

import pandas as pd
import numpy as np

def analyze_training_dataset():
    """Analyze the quality and correctness of the training dataset"""
    print("🔍 ANALYZING TRAINING DATASET QUALITY")
    print("=" * 60)
    
    # Load the training dataset
    try:
        train_df = pd.read_csv('data/simple_enhanced_train.csv')
        print(f"✅ Loaded training dataset: {train_df.shape}")
    except Exception as e:
        print(f"❌ Error loading training dataset: {e}")
        return
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Rows: {len(train_df)}")
    print(f"  Columns: {len(train_df.columns)}")
    print(f"  Memory usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 1. Check target variable (total_runs)
    print(f"\n🎯 TARGET VARIABLE ANALYSIS (total_runs):")
    print(f"  Mean: {train_df['total_runs'].mean():.1f}")
    print(f"  Median: {train_df['total_runs'].median():.1f}")
    print(f"  Std: {train_df['total_runs'].std():.1f}")
    print(f"  Min: {train_df['total_runs'].min()}")
    print(f"  Max: {train_df['total_runs'].max()}")
    print(f"  Missing values: {train_df['total_runs'].isnull().sum()}")
    
    # Check for unrealistic scores
    unrealistic_low = train_df[train_df['total_runs'] < 50]
    unrealistic_high = train_df[train_df['total_runs'] > 250]
    
    print(f"\n⚠️ UNREALISTIC SCORES:")
    print(f"  Scores < 50: {len(unrealistic_low)} ({len(unrealistic_low)/len(train_df)*100:.1f}%)")
    print(f"  Scores > 250: {len(unrealistic_high)} ({len(unrealistic_high)/len(train_df)*100:.1f}%)")
    
    if len(unrealistic_low) > 0:
        print(f"  Lowest scores: {unrealistic_low['total_runs'].tolist()[:5]}")
    if len(unrealistic_high) > 0:
        print(f"  Highest scores: {unrealistic_high['total_runs'].tolist()[:5]}")
    
    # 2. Check for data leakage
    print(f"\n🚨 DATA LEAKAGE CHECK:")
    
    # Check for post-match features that shouldn't be in training data
    suspicious_features = []
    for col in train_df.columns:
        if any(keyword in col.lower() for keyword in ['winner', 'result', 'outcome', 'final']):
            suspicious_features.append(col)
    
    if suspicious_features:
        print(f"  ⚠️ Potentially suspicious features: {suspicious_features}")
    else:
        print(f"  ✅ No obvious data leakage features found")
    
    # Check if match_winner is in the data
    if 'match_winner' in train_df.columns:
        print(f"  🚨 CRITICAL: 'match_winner' found in training data!")
        print(f"      This is post-match information that shouldn't be available during prediction")
        print(f"      Unique values: {train_df['match_winner'].value_counts().head()}")
    
    # 3. Check for missing values
    print(f"\n❌ MISSING VALUES ANALYSIS:")
    missing_data = train_df.isnull().sum()
    high_missing = missing_data[missing_data > len(train_df) * 0.1]  # >10% missing
    
    print(f"  Total missing values: {missing_data.sum()}")
    print(f"  Features with >10% missing: {len(high_missing)}")
    
    if len(high_missing) > 0:
        print(f"  High missing features:")
        for feature, count in high_missing.items():
            percentage = count / len(train_df) * 100
            print(f"    {feature}: {count} ({percentage:.1f}%)")
    
    # 4. Check feature quality
    print(f"\n🔧 FEATURE QUALITY ANALYSIS:")
    
    # Check for constant features
    constant_features = []
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if train_df[col].nunique() <= 1:
            constant_features.append(col)
    
    print(f"  Constant features: {len(constant_features)}")
    if constant_features:
        print(f"    {constant_features}")
    
    # Check for low-variance features
    low_variance_features = []
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col != 'total_runs' and train_df[col].std() < 0.01:
            low_variance_features.append(col)
    
    print(f"  Low variance features: {len(low_variance_features)}")
    if low_variance_features:
        print(f"    {low_variance_features[:5]}...")
    
    # 5. Check for duplicate matches
    print(f"\n🔄 DUPLICATE ANALYSIS:")
    duplicate_matches = train_df.duplicated(subset=['match_id', 'team']).sum()
    print(f"  Duplicate match-team combinations: {duplicate_matches}")
    
    if duplicate_matches > 0:
        print(f"  ⚠️ Found duplicate matches - this could cause overfitting")
    
    # 6. Check venue and team consistency
    print(f"\n🏟️ VENUE AND TEAM CONSISTENCY:")
    
    # Check venue data
    venues = train_df['venue'].nunique()
    venue_ids = train_df['venue_id'].nunique()
    print(f"  Unique venues: {venues}")
    print(f"  Unique venue IDs: {venue_ids}")
    
    if venues != venue_ids:
        print(f"  ⚠️ Venue name vs ID mismatch: {venues} names vs {venue_ids} IDs")
    
    # Check team data
    teams = train_df['team'].nunique()
    team_ids = train_df['team_id'].nunique()
    print(f"  Unique teams: {teams}")
    print(f"  Unique team IDs: {team_ids}")
    
    if teams != team_ids:
        print(f"  ⚠️ Team name vs ID mismatch: {teams} names vs {team_ids} IDs")
    
    # 7. Check temporal consistency
    print(f"\n📅 TEMPORAL CONSISTENCY:")
    
    if 'date' in train_df.columns:
        try:
            train_df['date_parsed'] = pd.to_datetime(train_df['date'])
            date_range = train_df['date_parsed'].max() - train_df['date_parsed'].min()
            print(f"  Date range: {train_df['date_parsed'].min()} to {train_df['date_parsed'].max()}")
            print(f"  Time span: {date_range.days} days")
            
            # Check for future dates
            future_dates = train_df[train_df['date_parsed'] > pd.Timestamp.now()]
            print(f"  Future dates: {len(future_dates)}")
            
        except Exception as e:
            print(f"  ⚠️ Could not parse dates: {e}")
    
    # 8. Check feature correlation with target
    print(f"\n📈 FEATURE-TARGET CORRELATION:")
    
    numeric_features = train_df.select_dtypes(include=[np.number]).columns
    correlations = train_df[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
    
    print(f"  Top 10 features correlated with total_runs:")
    for feature, corr in correlations.head(11).items():  # 11 because total_runs will be 1.0
        if feature != 'total_runs':
            print(f"    {feature}: {corr:.3f}")
    
    # 9. Check for specific cricket logic issues
    print(f"\n🏏 CRICKET LOGIC CHECKS:")
    
    # Check batting_first logic
    if 'batting_first' in train_df.columns:
        batting_first_stats = train_df.groupby('batting_first')['total_runs'].agg(['mean', 'count'])
        print(f"  Batting first vs second innings:")
        print(f"    First innings average: {batting_first_stats.loc[True, 'mean']:.1f}")
        print(f"    Second innings average: {batting_first_stats.loc[False, 'mean']:.1f}")
        
        # This should make sense - first innings teams usually score higher
        if batting_first_stats.loc[True, 'mean'] < batting_first_stats.loc[False, 'mean']:
            print(f"    ⚠️ WARNING: First innings scores are lower than second innings - this seems wrong!")
    
    # Check venue averages
    if 'venue_avg_runs' in train_df.columns:
        venue_avg_range = train_df['venue_avg_runs'].max() - train_df['venue_avg_runs'].min()
        print(f"  Venue average range: {train_df['venue_avg_runs'].min():.1f} to {train_df['venue_avg_runs'].max():.1f}")
        
        if venue_avg_range < 20:
            print(f"    ⚠️ WARNING: Very small venue average range - venues might not be differentiated properly")
    
    # 10. Overall assessment
    print(f"\n🎯 OVERALL DATASET ASSESSMENT:")
    
    issues = []
    
    if 'match_winner' in train_df.columns:
        issues.append("CRITICAL: Data leakage - match_winner in training data")
    
    if len(unrealistic_low) > len(train_df) * 0.05:
        issues.append(f"High number of unrealistic low scores ({len(unrealistic_low)})")
    
    if len(unrealistic_high) > len(train_df) * 0.05:
        issues.append(f"High number of unrealistic high scores ({len(unrealistic_high)})")
    
    if len(constant_features) > 5:
        issues.append(f"Too many constant features ({len(constant_features)})")
    
    if len(low_variance_features) > 10:
        issues.append(f"Too many low-variance features ({len(low_variance_features)})")
    
    if duplicate_matches > 0:
        issues.append(f"Duplicate matches found ({duplicate_matches})")
    
    if len(high_missing) > 5:
        issues.append(f"Too many features with high missing values ({len(high_missing)})")
    
    if issues:
        print(f"  🚨 CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        
        print(f"\n  ❌ RECOMMENDATION: Fix these issues before training models")
    else:
        print(f"  ✅ No critical issues found - dataset appears to be in good condition")
    
    return {
        'total_rows': len(train_df),
        'total_columns': len(train_df.columns),
        'target_mean': train_df['total_runs'].mean(),
        'target_std': train_df['total_runs'].std(),
        'unrealistic_low': len(unrealistic_low),
        'unrealistic_high': len(unrealistic_high),
        'constant_features': len(constant_features),
        'low_variance_features': len(low_variance_features),
        'duplicate_matches': duplicate_matches,
        'high_missing_features': len(high_missing),
        'issues': issues
    }

if __name__ == "__main__":
    results = analyze_training_dataset()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results['issues']:
        print(f"❌ DATASET HAS CRITICAL ISSUES")
        print(f"   Issues found: {len(results['issues'])}")
        print(f"   Recommendation: Fix issues before training")
    else:
        print(f"✅ DATASET APPEARS TO BE IN GOOD CONDITION")
        print(f"   Total rows: {results['total_rows']}")
        print(f"   Target variable: mean={results['target_mean']:.1f}, std={results['target_std']:.1f}")
        print(f"   Ready for model training")
