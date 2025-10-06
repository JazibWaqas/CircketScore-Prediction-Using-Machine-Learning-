#!/usr/bin/env python3

import pandas as pd

# Analyze venue scoring patterns
print('🏟️ ANALYZING VENUE SCORING PATTERNS')
print('=' * 50)

# Load training data
train_df = pd.read_csv('../data/simple_enhanced_train.csv')

# Filter realistic scores
realistic_df = train_df[(train_df['total_runs'] >= 60) & (train_df['total_runs'] <= 250)]

# Analyze venue scoring patterns
venue_stats = realistic_df.groupby('venue_id')['total_runs'].agg(['count', 'mean', 'std']).reset_index()
venue_stats = venue_stats[venue_stats['count'] >= 10]  # Venues with at least 10 matches

# Sort by average score
venue_stats = venue_stats.sort_values('mean', ascending=False)

print('TOP 10 HIGH-SCORING VENUES:')
for i, row in venue_stats.head(10).iterrows():
    print(f'  Venue {row["venue_id"]}: {row["mean"]:.1f} runs (std: {row["std"]:.1f}, matches: {row["count"]})')

print()
print('BOTTOM 10 LOW-SCORING VENUES:')
for i, row in venue_stats.tail(10).iterrows():
    print(f'  Venue {row["venue_id"]}: {row["mean"]:.1f} runs (std: {row["std"]:.1f}, matches: {row["count"]})')

print()
print('OVERALL VENUE STATISTICS:')
print(f'  Total venues analyzed: {len(venue_stats)}')
print(f'  Average venue score: {venue_stats["mean"].mean():.1f}')
print(f'  Highest venue average: {venue_stats["mean"].max():.1f}')
print(f'  Lowest venue average: {venue_stats["mean"].min():.1f}')
print(f'  Score range: {venue_stats["mean"].max() - venue_stats["mean"].min():.1f} runs difference')

# Check our current feature preparation
print()
print('🔧 CURRENT FEATURE PREPARATION ANALYSIS:')
print('Checking how we calculate venue features...')

# Simulate our current venue feature calculation
venue_id = 116  # Dubai International Cricket Stadium
venue_avg = 140.0 + (venue_id % 20) * 2.0  # Our current logic
print(f'Venue {venue_id} (Dubai): Our calculated avg = {venue_avg:.1f} runs')

# Compare with actual data
if 116 in venue_stats['venue_id'].values:
    actual_avg = venue_stats[venue_stats['venue_id'] == 116]['mean'].iloc[0]
    print(f'Venue {venue_id} (Dubai): Actual avg = {actual_avg:.1f} runs')
    print(f'Difference: {venue_avg - actual_avg:+.1f} runs (our calculation vs reality)')
else:
    print(f'Venue {venue_id} not found in training data')

# Check some high-scoring venues
high_scoring_venues = venue_stats.head(5)
print()
print('HIGH-SCORING VENUES COMPARISON:')
for i, row in high_scoring_venues.iterrows():
    venue_id = row['venue_id']
    actual_avg = row['mean']
    our_calc = 140.0 + (venue_id % 20) * 2.0
    diff = our_calc - actual_avg
    print(f'Venue {venue_id}: Actual {actual_avg:.1f}, Our calc {our_calc:.1f}, Diff {diff:+.1f}')

print()
print('🎯 ISSUE IDENTIFIED:')
print('Our venue feature calculation is too simplistic!')
print('We need to use actual venue statistics from training data.')
