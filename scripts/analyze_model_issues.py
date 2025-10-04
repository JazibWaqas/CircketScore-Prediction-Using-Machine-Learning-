"""
Analyze Model Performance Issues
Identify why accuracy is poor and how to improve
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_model_issues():
    """Analyze what went wrong with the model performance"""
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
    
    # Analyze features
    print(f"\nFEATURE ANALYSIS:")
    exclude_cols = ['total_runs', 'toss_winner', 'toss_decision', 'match_winner', 
                   'player_of_match', 'season', 'event_name', 'gender']
    
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    # Check feature quality
    print(f"\nFEATURE QUALITY ANALYSIS:")
    for col in feature_cols[:10]:  # Check first 10 features
        if col in test_df.columns:
            print(f"{col}:")
            if test_df[col].dtype in ['int64', 'float64']:
                print(f"  - Range: {test_df[col].min():.2f} to {test_df[col].max():.2f}")
                print(f"  - Mean: {test_df[col].mean():.2f}")
                print(f"  - Std: {test_df[col].std():.2f}")
            else:
                print(f"  - Type: {test_df[col].dtype}")
                print(f"  - Unique values: {test_df[col].nunique()}")
            print(f"  - Missing: {test_df[col].isnull().sum()}")
            print()
    
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
        print(f"❌ DATA LEAKAGE DETECTED:")
        for col in leakage_found:
            print(f"  - {col}")
        print(f"These features are calculated DURING/AFTER the match!")
    else:
        print(f"✅ No data leakage detected")
    
    # Check feature correlation with target
    print(f"\nFEATURE-TARGET CORRELATION:")
    correlations = []
    for col in feature_cols:
        if col in test_df.columns:
            corr = test_df[col].corr(test_df['total_runs'])
            correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"Top 10 features by correlation with total_runs:")
    for col, corr in correlations[:10]:
        print(f"  {col}: {corr:.3f}")
    
    # Check for missing values
    print(f"\nMISSING VALUES ANALYSIS:")
    missing_values = test_df[feature_cols].isnull().sum()
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(test_df)*100:.1f}%)")
    
    # Check for constant features
    print(f"\nCONSTANT FEATURES CHECK:")
    constant_features = []
    for col in feature_cols:
        if col in test_df.columns:
            if test_df[col].nunique() <= 1:
                constant_features.append(col)
    
    if constant_features:
        print(f"❌ CONSTANT FEATURES FOUND:")
        for col in constant_features:
            print(f"  - {col}")
    else:
        print(f"✅ No constant features")
    
    # Check for highly correlated features
    print(f"\nFEATURE CORRELATION ANALYSIS:")
    feature_df = test_df[feature_cols].fillna(0)
    corr_matrix = feature_df.corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.9:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        print(f"❌ HIGHLY CORRELATED FEATURES:")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print(f"✅ No highly correlated features")
    
    # Identify issues and solutions
    print(f"\n" + "="*60)
    print("ISSUES IDENTIFIED AND SOLUTIONS")
    print("="*60)
    
    issues = []
    solutions = []
    
    # Issue 1: Data leakage
    if leakage_found:
        issues.append("❌ DATA LEAKAGE: Using post-match features")
        solutions.append("✅ SOLUTION: Remove all post-match features, use only pre-match context")
    
    # Issue 2: Poor feature quality
    low_corr_features = [col for col, corr in correlations if abs(corr) < 0.1]
    if len(low_corr_features) > len(feature_cols) * 0.5:
        issues.append("❌ POOR FEATURES: Many features have low correlation with target")
        solutions.append("✅ SOLUTION: Engineer better features, add more relevant pre-match data")
    
    # Issue 3: Missing values
    if missing_values.sum() > 0:
        issues.append("❌ MISSING VALUES: Some features have missing data")
        solutions.append("✅ SOLUTION: Handle missing values properly, use imputation")
    
    # Issue 4: Feature engineering
    if len(feature_cols) < 20:
        issues.append("❌ INSUFFICIENT FEATURES: Not enough predictive features")
        solutions.append("✅ SOLUTION: Add more pre-match features (weather, pitch conditions, player form)")
    
    # Issue 5: Model complexity
    issues.append("❌ MODEL COMPLEXITY: May need more sophisticated approach")
    solutions.append("✅ SOLUTION: Try ensemble methods, feature selection, hyperparameter tuning")
    
    print(f"\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
    
    print(f"\nSOLUTIONS:")
    for solution in solutions:
        print(f"  {solution}")
    
    # Specific recommendations
    print(f"\n" + "="*60)
    print("SPECIFIC RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n1. IMMEDIATE FIXES:")
    print(f"   - Remove data leakage features")
    print(f"   - Add more pre-match features")
    print(f"   - Handle missing values")
    print(f"   - Feature selection")
    
    print(f"\n2. FEATURE ENGINEERING:")
    print(f"   - Team batting average (last 5 matches)")
    print(f"   - Opposition bowling average (last 5 matches)")
    print(f"   - Venue-specific team performance")
    print(f"   - Head-to-head record (last 3 meetings)")
    print(f"   - Player form (individual player averages)")
    print(f"   - Match importance (final, semi-final, etc.)")
    print(f"   - Weather conditions")
    print(f"   - Pitch conditions")
    
    print(f"\n3. MODEL IMPROVEMENTS:")
    print(f"   - Try different algorithms (SVM, Neural Networks)")
    print(f"   - Ensemble methods (Voting, Stacking)")
    print(f"   - Hyperparameter tuning")
    print(f"   - Cross-validation")
    print(f"   - Feature selection")
    
    print(f"\n4. DATA IMPROVEMENTS:")
    print(f"   - More training data")
    print(f"   - Better data quality")
    print(f"   - More recent matches")
    print(f"   - Player-specific statistics")
    
    return issues, solutions

if __name__ == "__main__":
    issues, solutions = analyze_model_issues()
