"""
Validate Comprehensive T20 Dataset
Check data quality, completeness, and correctness
"""

import pandas as pd
import numpy as np

def validate_dataset():
    """Validate the comprehensive T20 dataset"""
    print("=== COMPREHENSIVE T20 DATASET VALIDATION ===")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('comprehensive_t20_dataset.csv')
    
    print(f"BASIC STATISTICS:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique matches: {df['match_id'].nunique():,}")
    print(f"   Unique teams: {df['team'].nunique()}")
    print(f"   Unique venues: {df['venue'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total runs range: {df['total_runs'].min()} to {df['total_runs'].max()}")
    
    print(f"\nDATA QUALITY CHECKS:")
    
    # Check 1: Teams per match (should be 2)
    teams_per_match = df.groupby('match_id').size()
    print(f"   Teams per match distribution:")
    print(f"   {teams_per_match.value_counts().head()}")
    
    # Check 2: Missing values
    missing_values = df.isnull().sum()
    print(f"\n   Missing values per column:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"   {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # Check 3: Data consistency
    print(f"\n   Data consistency checks:")
    
    # Check if total_runs makes sense
    invalid_runs = df[(df['total_runs'] < 0) | (df['total_runs'] > 300)]
    print(f"   Invalid total_runs (< 0 or > 300): {len(invalid_runs)}")
    
    # Check if overs make sense
    invalid_overs = df[(df['total_overs'] < 0) | (df['total_overs'] > 25)]
    print(f"   Invalid total_overs (< 0 or > 25): {len(invalid_overs)}")
    
    # Check if boundaries make sense
    invalid_boundaries = df[df['total_boundaries'] > df['total_runs']]
    print(f"   Boundaries > total_runs: {len(invalid_boundaries)}")
    
    # Check 4: Team performance distribution
    print(f"\nTEAM PERFORMANCE ANALYSIS:")
    print(f"   Average runs per team: {df['total_runs'].mean():.1f}")
    print(f"   Median runs per team: {df['total_runs'].median():.1f}")
    print(f"   Standard deviation: {df['total_runs'].std():.1f}")
    
    # Check 5: Venue analysis
    print(f"\nVENUE ANALYSIS:")
    venue_stats = df.groupby('venue')['total_runs'].agg(['count', 'mean', 'std']).round(1)
    print(f"   Top 10 venues by matches:")
    print(venue_stats.sort_values('count', ascending=False).head(10))
    
    # Check 6: Team analysis
    print(f"\nTEAM ANALYSIS:")
    team_stats = df.groupby('team')['total_runs'].agg(['count', 'mean', 'std']).round(1)
    print(f"   Top 10 teams by matches:")
    print(team_stats.sort_values('count', ascending=False).head(10))
    
    # Check 7: Match context
    print(f"\nMATCH CONTEXT:")
    print(f"   Batting first: {df['batting_first'].sum()} ({df['batting_first'].mean()*100:.1f}%)")
    print(f"   Home teams: {df['is_home_team'].sum()} ({df['is_home_team'].mean()*100:.1f}%)")
    print(f"   Finals: {df['is_final'].sum()} ({df['is_final'].mean()*100:.1f}%)")
    print(f"   Semi-finals: {df['is_semi_final'].sum()} ({df['is_semi_final'].mean()*100:.1f}%)")
    
    # Check 8: Feature completeness
    print(f"\nFEATURE COMPLETENESS:")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Features with data: {len(df.columns) - missing_values.sum()}")
    print(f"   Complete records: {len(df) - df.isnull().any(axis=1).sum()}")
    
    # Check 9: Sample data quality
    print(f"\nSAMPLE DATA QUALITY:")
    sample = df.head(3)
    print(f"   Sample match IDs: {sample['match_id'].tolist()}")
    print(f"   Sample teams: {sample['team'].tolist()}")
    print(f"   Sample venues: {sample['venue'].tolist()}")
    print(f"   Sample runs: {sample['total_runs'].tolist()}")
    
    # Check 10: Data relationships
    print(f"\nDATA RELATIONSHIPS:")
    print(f"   Matches with 2 teams: {(teams_per_match == 2).sum()}")
    print(f"   Matches with 1 team: {(teams_per_match == 1).sum()}")
    print(f"   Matches with >2 teams: {(teams_per_match > 2).sum()}")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    
    issues = []
    if len(invalid_runs) > 0:
        issues.append(f"Invalid runs: {len(invalid_runs)}")
    if len(invalid_overs) > 0:
        issues.append(f"Invalid overs: {len(invalid_overs)}")
    if len(invalid_boundaries) > 0:
        issues.append(f"Invalid boundaries: {len(invalid_boundaries)}")
    if missing_values.sum() > 0:
        issues.append(f"Missing values: {missing_values.sum()}")
    if (teams_per_match != 2).sum() > 0:
        issues.append(f"Matches with != 2 teams: {(teams_per_match != 2).sum()}")
    
    if issues:
        print(f"   WARNING: Issues found: {', '.join(issues)}")
    else:
        print(f"   SUCCESS: No major issues found!")
    
    print(f"\nDATASET USEFULNESS:")
    print(f"   Scale: {len(df):,} records from {df['match_id'].nunique():,} matches")
    print(f"   Coverage: {df['team'].nunique()} teams, {df['venue'].nunique()} venues")
    print(f"   Time span: {df['date'].min()} to {df['date'].max()}")
    print(f"   Features: {len(df.columns)} columns")
    print(f"   Context: Toss, venue, opposition, match importance")
    
    return df

if __name__ == "__main__":
    df = validate_dataset()
