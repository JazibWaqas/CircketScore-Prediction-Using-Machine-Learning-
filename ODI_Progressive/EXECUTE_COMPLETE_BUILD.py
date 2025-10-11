#!/usr/bin/env python3
"""
COMPLETE AUTOMATED BUILD - Runs everything without user intervention
"""

import subprocess
import sys
import os

print("\n" + "="*80)
print("AUTOMATED COMPLETE BUILD - ODI PROGRESSIVE PREDICTOR")
print("="*80)

print("\nThis will:")
print("  1. Parse ODI ball-by-ball data")
print("  2. Create training dataset")
print("  3. Train XGBoost model")
print("  4. Test and validate")
print("  5. Report results")

print("\nEstimated time: 3-5 minutes")
print("\n" + "="*80)

# Execute BUILD_AND_TRAIN.py
print("\n[PHASE 1] Building and training model...")
print("-" * 80)

try:
    exec(open('ODI_Progressive/BUILD_AND_TRAIN.py').read())
    print("\n✓ Build and train complete")
except Exception as e:
    print(f"\n❌ Error during build: {e}")
    print("Attempting to continue...")

# Execute TEST_PREDICTIONS.py
print("\n[PHASE 2] Testing predictions...")
print("-" * 80)

try:
    exec(open('ODI_Progressive/TEST_PREDICTIONS.py').read())
    print("\n✓ Testing complete")
except Exception as e:
    print(f"\n❌ Error during testing: {e}")

print("\n" + "="*80)
print("AUTOMATED BUILD COMPLETE")
print("="*80)

