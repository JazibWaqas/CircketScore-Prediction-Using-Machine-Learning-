#!/usr/bin/env python3

import pandas as pd
import pickle

def create_venue_statistics():
    """
    Create a lookup table of actual venue statistics from training data
    """
    print("🏟️ CREATING VENUE STATISTICS LOOKUP")
    print("=" * 50)
    
    # Load training data
    train_df = pd.read_csv('../data/simple_enhanced_train.csv')
    
    # Filter realistic scores
    realistic_df = train_df[(train_df['total_runs'] >= 60) & (train_df['total_runs'] <= 250)]
    
    # Calculate venue statistics
    venue_stats = realistic_df.groupby('venue_id')['total_runs'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # Only include venues with at least 5 matches for reliability
    venue_stats = venue_stats[venue_stats['count'] >= 5]
    
    # Fill missing standard deviation with venue mean
    venue_stats['std'] = venue_stats['std'].fillna(venue_stats['mean'] * 0.2)
    
    # Create lookup dictionary
    venue_lookup = {}
    for _, row in venue_stats.iterrows():
        venue_id = int(row['venue_id'])
        venue_lookup[venue_id] = {
            'avg_runs': float(row['mean']),
            'std_runs': float(row['std']),
            'min_runs': int(row['min']),
            'max_runs': int(row['max']),
            'matches': int(row['count']),
            'difficulty': 1.0  # Default difficulty
        }
    
    # Calculate venue difficulty (higher score = easier batting)
    max_avg = venue_stats['mean'].max()
    min_avg = venue_stats['mean'].min()
    
    for venue_id in venue_lookup:
        avg_score = venue_lookup[venue_id]['avg_runs']
        # Normalize difficulty (0.5 = hard, 1.5 = easy)
        difficulty = 0.5 + ((avg_score - min_avg) / (max_avg - min_avg))
        venue_lookup[venue_id]['difficulty'] = difficulty
    
    print(f"Created venue statistics for {len(venue_lookup)} venues")
    print(f"Score range: {min_avg:.1f} - {max_avg:.1f} runs")
    
    # Show some examples
    print("\n📊 SAMPLE VENUE STATISTICS:")
    sample_venues = list(venue_lookup.keys())[:5]
    for venue_id in sample_venues:
        stats = venue_lookup[venue_id]
        print(f"  Venue {venue_id}: {stats['avg_runs']:.1f} runs (difficulty: {stats['difficulty']:.2f})")
    
    # Save venue lookup
    with open('../models/venue_statistics.pkl', 'wb') as f:
        pickle.dump(venue_lookup, f)
    
    print(f"\n✅ Venue statistics saved to ../models/venue_statistics.pkl")
    
    # Create fallback statistics for venues not in training data
    fallback_stats = {
        'avg_runs': realistic_df['total_runs'].mean(),
        'std_runs': realistic_df['total_runs'].std(),
        'min_runs': realistic_df['total_runs'].min(),
        'max_runs': realistic_df['total_runs'].max(),
        'matches': 0,
        'difficulty': 1.0
    }
    
    with open('../models/venue_fallback.pkl', 'wb') as f:
        pickle.dump(fallback_stats, f)
    
    print(f"✅ Fallback statistics saved for unknown venues")
    print(f"   Fallback avg: {fallback_stats['avg_runs']:.1f} runs")
    
    return venue_lookup

if __name__ == "__main__":
    create_venue_statistics()
