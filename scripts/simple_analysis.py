"""
Simple Model Performance Analysis
Identify key issues with the model
"""

import pandas as pd
import numpy as np

def simple_analysis():
    """Simple analysis of model issues"""
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv('data/test_dataset.csv')
    print(f"Test dataset: {test_df.shape}")
    
    # Analyze the actual data
    print(f"\nACTUAL DATA ANALYSIS:")
    print(f"Score range: {test_df['total_runs'].min()} to {test_df['total_runs'].max()}")
    print(f"Score mean: {test_df['total_runs'].mean():.1f}")
    print(f"Score std: {test_df['total_runs'].std():.1f}")
    
    # Score distribution
    print(f"\nSCORE DISTRIBUTION:")
    print(f"0-100 runs: {len(test_df[test_df['total_runs'] <= 100])} matches ({len(test_df[test_df['total_runs'] <= 100])/len(test_df)*100:.1f}%)")
    print(f"100-150 runs: {len(test_df[(test_df['total_runs'] > 100) & (test_df['total_runs'] <= 150)])} matches ({len(test_df[(test_df['total_runs'] > 100) & (test_df['total_runs'] <= 150)])/len(test_df)*100:.1f}%)")
    print(f"150-200 runs: {len(test_df[(test_df['total_runs'] > 150) & (test_df['total_runs'] <= 200)])} matches ({len(test_df[(test_df['total_runs'] > 150) & (test_df['total_runs'] <= 200)])/len(test_df)*100:.1f}%)")
    print(f"200+ runs: {len(test_df[test_df['total_runs'] > 200])} matches ({len(test_df[test_df['total_runs'] > 200])/len(test_df)*100:.1f}%)")
    
    # Check for data leakage
    print(f"\nDATA LEAKAGE CHECK:")
    potential_leakage = ['total_wickets', 'total_overs', 'total_balls', 'run_rate',
                        'total_4s', 'total_6s', 'total_boundaries', 'total_extras',
                        'powerplay_runs', 'middle_overs_runs', 'death_overs_runs']
    
    leakage_found = []
    for col in potential_leakage:
        if col in test_df.columns:
            leakage_found.append(col)
    
    if leakage_found:
        print(f"DATA LEAKAGE DETECTED:")
        for col in leakage_found:
            print(f"  - {col}")
        print(f"These features are calculated DURING/AFTER the match!")
    else:
        print(f"No data leakage detected")
    
    # Check feature correlation with target
    print(f"\nFEATURE-TARGET CORRELATION:")
    exclude_cols = ['total_runs', 'toss_winner', 'toss_decision', 'match_winner', 
                   'player_of_match', 'season', 'event_name', 'gender']
    
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    correlations = []
    for col in feature_cols:
        if col in test_df.columns and test_df[col].dtype in ['int64', 'float64']:
            corr = test_df[col].corr(test_df['total_runs'])
            correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"Top 10 features by correlation with total_runs:")
    for col, corr in correlations[:10]:
        print(f"  {col}: {corr:.3f}")
    
    # Identify issues
    print(f"\n" + "="*60)
    print("ISSUES IDENTIFIED")
    print("="*60)
    
    issues = []
    
    # Issue 1: Data leakage
    if leakage_found:
        issues.append("DATA LEAKAGE: Using post-match features")
    
    # Issue 2: Poor feature quality
    low_corr_features = [col for col, corr in correlations if abs(corr) < 0.1]
    if len(low_corr_features) > len(feature_cols) * 0.5:
        issues.append("POOR FEATURES: Many features have low correlation with target")
    
    # Issue 3: Feature engineering
    if len(feature_cols) < 20:
        issues.append("INSUFFICIENT FEATURES: Not enough predictive features")
    
    print(f"\nISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    # Solutions
    print(f"\n" + "="*60)
    print("SOLUTIONS")
    print("="*60)
    
    print(f"\n1. IMMEDIATE FIXES:")
    print(f"   - Remove data leakage features")
    print(f"   - Add more pre-match features")
    print(f"   - Feature selection")
    
    print(f"\n2. FEATURE ENGINEERING:")
    print(f"   - Team batting average (last 5 matches)")
    print(f"   - Opposition bowling average (last 5 matches)")
    print(f"   - Venue-specific team performance")
    print(f"   - Head-to-head record (last 3 meetings)")
    print(f"   - Player form (individual player averages)")
    print(f"   - Match importance (final, semi-final, etc.)")
    
    print(f"\n3. MODEL IMPROVEMENTS:")
    print(f"   - Try different algorithms")
    print(f"   - Ensemble methods")
    print(f"   - Hyperparameter tuning")
    print(f"   - Cross-validation")
    
    print(f"\n4. DATA IMPROVEMENTS:")
    print(f"   - More training data")
    print(f"   - Better data quality")
    print(f"   - More recent matches")
    print(f"   - Player-specific statistics")
    
    return issues

if __name__ == "__main__":
    issues = simple_analysis()
