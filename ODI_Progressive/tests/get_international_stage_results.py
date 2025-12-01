#!/usr/bin/env python3
"""
Calculate stage-by-stage results for international validation
"""

import json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Load the validation results
with open('validation_results_v2.json', 'r') as f:
    data = json.load(f)

# We need to load the actual results dataframe
# Let's modify the validation script to save stage-by-stage data
print("This script needs to be run after validate_international_v2.py")
print("Or we need to modify validate_international_v2.py to save stage data")

