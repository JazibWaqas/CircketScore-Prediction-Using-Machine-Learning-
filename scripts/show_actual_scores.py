#!/usr/bin/env python3
"""
Show Actual Scores in Test Dataset
"""

import pandas as pd

def show_actual_scores():
    """Show actual scores from the test dataset"""
    print("🎯 ACTUAL SCORES IN OUR TEST DATASET:")
    print("=" * 50)
    
    # Load test dataset
    df = pd.read_csv('processed_data/final_test_dataset.csv')
    
    print(f"Total test matches: {len(df)}")
    print(f"Score range: {df['total_runs'].min()} to {df['total_runs'].max()} runs")
    print(f"Average score: {df['total_runs'].mean():.1f} runs")
    print()
    
    print("📊 SAMPLE REAL MATCHES:")
    for i in range(10):
        match = df.iloc[i]
        print(f"Match {i+1}: {match['team']} vs {match['opposition']} - ACTUAL SCORE: {match['total_runs']} runs")
        print(f"  Date: {match['date']}, Venue: {match['venue']}")
        print()
    
    print("🔍 HOW WE'LL MEASURE ACCURACY:")
    print("1. Train model on 13,014 matches")
    print("2. Test model on these 500 matches")
    print("3. Compare predictions vs actual scores:")
    print("   - Model predicts: 150 runs")
    print("   - Actual score: 148 runs")
    print("   - Error: 2 runs (GOOD!)")
    print()
    print("4. Calculate overall accuracy:")
    print("   - Average error across all 500 matches")
    print("   - Percentage of predictions within ±10 runs")
    print("   - R² score (how well model explains the data)")

if __name__ == "__main__":
    show_actual_scores()
