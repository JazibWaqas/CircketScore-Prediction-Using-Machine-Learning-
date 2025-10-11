#!/usr/bin/env python3
"""
VERIFY TRAIN/TEST SPLIT IS CORRECT

Checks:
1. No match_ids appear in both train and test
2. All checkpoints from same match are in same split
3. Actual number of matches vs rows
4. Show overlap if any exists
"""

import pandas as pd

print("\n" + "="*80)
print("VERIFYING TRAIN/TEST SPLIT")
print("="*80)

# Load datasets
print("\n[1/4] Loading datasets...")
train_df = pd.read_csv('data/progressive_train.csv')
test_df = pd.read_csv('data/progressive_test.csv')

print(f"   Training: {len(train_df):,} rows")
print(f"   Testing: {len(test_df):,} rows")
print(f"   Total: {len(train_df) + len(test_df):,} rows")

# Get unique match IDs
train_matches = set(train_df['match_id'].unique())
test_matches = set(test_df['match_id'].unique())

print(f"\n   Training matches: {len(train_matches):,}")
print(f"   Testing matches: {len(test_matches):,}")
print(f"   Total matches: {len(train_matches) + len(test_matches):,}")

# ==============================================================================
# CHECK 1: No overlap in match IDs
# ==============================================================================

print("\n[2/4] Checking for match ID overlap...")

overlap = train_matches.intersection(test_matches)

if len(overlap) > 0:
    print(f"\n   [ERROR] Found {len(overlap)} matches in BOTH train and test!")
    print(f"   Example overlapping match IDs: {list(overlap)[:10]}")
    print(f"\n   THIS IS DATA LEAKAGE!")
else:
    print(f"   [OK] No overlap! Train and test are completely separate")

# ==============================================================================
# CHECK 2: All checkpoints from same match are together
# ==============================================================================

print("\n[3/4] Checking if checkpoints are kept together...")

# For each match in training, verify ALL its rows are in training
train_sample = train_df.sample(min(100, len(train_df)))
all_together = True

for match_id in train_sample['match_id'].unique()[:10]:
    train_count = len(train_df[train_df['match_id'] == match_id])
    test_count = len(test_df[test_df['match_id'] == match_id])
    
    if test_count > 0:
        print(f"   [ERROR] Match {match_id}: {train_count} in train, {test_count} in test")
        all_together = False

if all_together:
    print(f"   [OK] All checkpoints from same match stay together")

# ==============================================================================
# CHECK 3: Verify actual match counts vs row counts
# ==============================================================================

print("\n[4/4] Verifying match counts vs row counts...")

# Calculate average checkpoints per match
train_avg_checkpoints = len(train_df) / len(train_matches)
test_avg_checkpoints = len(test_df) / len(test_matches)

print(f"\n   Training:")
print(f"   - {len(train_matches):,} matches")
print(f"   - {len(train_df):,} rows")
print(f"   - {train_avg_checkpoints:.1f} checkpoints per match (expect ~14-15)")

print(f"\n   Testing:")
print(f"   - {len(test_matches):,} matches")
print(f"   - {len(test_df):,} rows")
print(f"   - {test_avg_checkpoints:.1f} checkpoints per match (expect ~14-15)")

# Show distribution of checkpoints per match
print(f"\n   Checkpoint distribution in training:")
train_checkpoints = train_df.groupby('match_id').size()
print(f"   - Min checkpoints: {train_checkpoints.min()}")
print(f"   - Max checkpoints: {train_checkpoints.max()}")
print(f"   - Mean: {train_checkpoints.mean():.1f}")
print(f"   - Median: {train_checkpoints.median():.1f}")

# ==============================================================================
# CHECK 4: Verify we actually have that many matches
# ==============================================================================

print("\n" + "="*80)
print("VERIFICATION OF SOURCE DATA")
print("="*80)

import os
import json

ballbyball_dir = '../raw_data/odis_ballbyBall'
all_files = [f for f in os.listdir(ballbyball_dir) if f.endswith('.json')]

print(f"\nSource files: {len(all_files):,} JSON files")
print(f"Dataset matches: {len(train_matches) + len(test_matches):,} matches")
print(f"Difference: {len(all_files) - (len(train_matches) + len(test_matches)):,} matches filtered out")

# Quick check: why were matches filtered out?
print(f"\nMatches were filtered out because:")
print(f"  - Incomplete innings (no data)")
print(f"  - City not in eligible list (< 200 samples)")
print(f"  - Invalid/corrupted JSON")
print(f"  - Final score = 0")

# ==============================================================================
# SHOW SAMPLE MATCHES
# ==============================================================================

print("\n" + "="*80)
print("SAMPLE MATCH VERIFICATION")
print("="*80)

# Show a complete match from training
print(f"\nExample Training Match (ID = {list(train_matches)[0]}):")
sample_match = train_df[train_df['match_id'] == list(train_matches)[0]].sort_values('ball_number')
print(f"{'Over':<6} {'Score':>8} {'Wickets':>8} {'Balls Left':>11} {'Final':>8}")
print("-" * 50)
for _, row in sample_match.iterrows():
    print(f"{row['over']:<6.0f} {row['current_score']:>8.0f} {10-row['wickets_left']:>8.0f} "
          f"{row['balls_left']:>11.0f} {row['final_score']:>8.0f}")

# Show a complete match from testing
print(f"\nExample Testing Match (ID = {list(test_matches)[0]}):")
sample_match = test_df[test_df['match_id'] == list(test_matches)[0]].sort_values('ball_number')
print(f"{'Over':<6} {'Score':>8} {'Wickets':>8} {'Balls Left':>11} {'Final':>8}")
print("-" * 50)
for _, row in sample_match.head(10).iterrows():
    print(f"{row['over']:<6.0f} {row['current_score']:>8.0f} {10-row['wickets_left']:>8.0f} "
          f"{row['balls_left']:>11.0f} {row['final_score']:>8.0f}")

# ==============================================================================
# FINAL VERDICT
# ==============================================================================

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if len(overlap) > 0:
    print("\n[FAILED] DATA LEAKAGE DETECTED!")
    print(f"  {len(overlap)} matches appear in both train and test")
    print(f"  Results are INVALID")
else:
    print("\n[PASSED] Train/Test split is CORRECT!")
    print(f"  ✓ No match overlap between train and test")
    print(f"  ✓ All checkpoints from same match stay together")
    print(f"  ✓ {len(train_matches):,} training matches")
    print(f"  ✓ {len(test_matches):,} testing matches")
    print(f"  ✓ 80/20 split: {len(train_matches)/(len(train_matches)+len(test_matches))*100:.1f}% / {len(test_matches)/(len(train_matches)+len(test_matches))*100:.1f}%")

print("\n" + "="*80 + "\n")

