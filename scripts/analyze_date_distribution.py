"""
Analyze Date Distribution in T20 Dataset
Understand why 2024-2025 has so many matches
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_date_distribution():
    """Analyze the distribution of matches by year"""
    print("Analyzing Date Distribution in T20 Dataset")
    print("=" * 60)
    
    # Load the validated dataset
    df = pd.read_csv('validated_t20_dataset.csv')
    print(f"Loaded validated dataset: {df.shape}")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Analyze by year
    print("\nMATCHES BY YEAR:")
    year_stats = df.groupby('year').agg({
        'match_id': 'nunique',
        'total_runs': 'mean',
        'team': 'nunique',
        'venue': 'nunique'
    }).round(2)
    year_stats.columns = ['Matches', 'Avg_Runs', 'Teams', 'Venues']
    print(year_stats)
    
    # Analyze by month for 2024-2025
    print("\n2024-2025 MATCHES BY MONTH:")
    recent_data = df[df['year'] >= 2024]
    month_stats = recent_data.groupby(['year', 'month']).agg({
        'match_id': 'nunique',
        'total_runs': 'mean'
    }).round(2)
    month_stats.columns = ['Matches', 'Avg_Runs']
    print(month_stats)
    
    # Check for data quality issues
    print("\nDATA QUALITY CHECK:")
    print(f"Total records: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")
    print(f"Records per match: {len(df) / df['match_id'].nunique():.2f}")
    
    # Check for duplicate matches
    match_counts = df['match_id'].value_counts()
    print(f"Matches with 2 records: {(match_counts == 2).sum():,}")
    print(f"Matches with != 2 records: {(match_counts != 2).sum():,}")
    
    if (match_counts != 2).sum() > 0:
        print("\nMatches with != 2 records:")
        print(match_counts[match_counts != 2].head(10))
    
    # Analyze the 2024-2025 data more closely
    print("\n2024-2025 DETAILED ANALYSIS:")
    recent_matches = recent_data['match_id'].nunique()
    recent_records = len(recent_data)
    print(f"2024-2025 matches: {recent_matches:,}")
    print(f"2024-2025 records: {recent_records:,}")
    print(f"Records per match: {recent_records / recent_matches:.2f}")
    
    # Check if there are any data quality issues
    print("\nPOTENTIAL ISSUES:")
    
    # Check for matches with more than 2 teams
    match_team_counts = df.groupby('match_id')['team'].nunique()
    multi_team_matches = match_team_counts[match_team_counts > 2]
    print(f"Matches with >2 teams: {len(multi_team_matches)}")
    
    if len(multi_team_matches) > 0:
        print("Sample multi-team matches:")
        print(multi_team_matches.head(10))
    
    # Check for matches with <2 teams
    single_team_matches = match_team_counts[match_team_counts < 2]
    print(f"Matches with <2 teams: {len(single_team_matches)}")
    
    # Check for matches with exactly 2 teams
    two_team_matches = match_team_counts[match_team_counts == 2]
    print(f"Matches with exactly 2 teams: {len(two_team_matches)}")
    
    # Analyze the distribution
    print("\nDISTRIBUTION ANALYSIS:")
    print(f"Expected records (2 per match): {df['match_id'].nunique() * 2:,}")
    print(f"Actual records: {len(df):,}")
    print(f"Difference: {len(df) - (df['match_id'].nunique() * 2):,}")
    
    # Check for specific years with high match counts
    print("\nTOP 10 YEARS BY MATCH COUNT:")
    top_years = year_stats.sort_values('Matches', ascending=False).head(10)
    print(top_years)
    
    # Check for 2024-2025 specifically
    print("\n2024-2025 BREAKDOWN:")
    for year in [2024, 2025]:
        year_data = df[df['year'] == year]
        if len(year_data) > 0:
            print(f"{year}: {year_data['match_id'].nunique():,} matches, {len(year_data):,} records")
            print(f"  Teams: {year_data['team'].nunique()}, Venues: {year_data['venue'].nunique()}")
            print(f"  Date range: {year_data['date'].min()} to {year_data['date'].max()}")
    
    return year_stats, month_stats

if __name__ == "__main__":
    year_stats, month_stats = analyze_date_distribution()
