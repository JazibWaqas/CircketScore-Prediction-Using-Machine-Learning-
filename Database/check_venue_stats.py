#!/usr/bin/env python3

import pickle

# Check the venue statistics we loaded
with open('../models/venue_statistics.pkl', 'rb') as f:
    venue_stats = pickle.load(f)

# Check specific venues
test_venues = [116, 478, 131, 174, 27]

print('🏟️ VENUE STATISTICS CHECK:')
print('=' * 40)

for venue_id in test_venues:
    if venue_id in venue_stats:
        stats = venue_stats[venue_id]
        print(f'Venue {venue_id}:')
        print(f'  Avg runs: {stats["avg_runs"]:.1f}')
        print(f'  Difficulty: {stats["difficulty"]:.2f}')
        print(f'  Matches: {stats["matches"]}')
        print()
    else:
        print(f'Venue {venue_id}: Not found in statistics')
        print()

# Check if venue 478 is actually high-scoring
print('🔍 CHECKING VENUE 478 (should be high-scoring):')
if 478 in venue_stats:
    stats = venue_stats[478]
    print(f'  Average runs: {stats["avg_runs"]:.1f}')
    print(f'  This should be ~181 runs based on our analysis')
    if stats["avg_runs"] > 170:
        print('  ✅ This is indeed a high-scoring venue')
    else:
        print('  ❌ This venue is not high-scoring - issue with our data')
else:
    print('  ❌ Venue 478 not found in statistics')
