# ODI Progressive Score Predictor

**Progressive mid-match prediction with fantasy team building**

---

## Quick Start

### Build & Train Model (One Command)

```bash
python BUILD_AND_TRAIN.py
```

**This will:**
1. Parse 1,500 ODI matches (ball-by-ball)
2. Create ~10,500 training samples
3. Train XGBoost model
4. Evaluate performance
5. Save trained model

**Time:** 3-5 minutes  
**Expected:** R² = 0.90-0.95

---

## What It Does

**Predicts ODI final score from:**
- Current match state (score, wickets, overs)
- Team batting strength (calculated from players)
- Venue and opposition

**Works at any match stage:**
- Pre-match (ball 0): R² ~0.30
- Mid-match (ball 180): R² ~0.92
- Late-match (ball 270): R² ~0.96

---

## Features

**Core (from working Cricket-Score-Predictor):**
- current_score
- balls_left
- wickets_left
- current_run_rate
- last_10_overs

**Our Addition (for fantasy):**
- team_batting_avg (from 11 players)
- batting_team, city

**Total:** 8 features (simple and effective!)

---

## Expected Results

**Overall:** R² = 0.88-0.92, MAE = 8-12 runs

**By Stage:**
- Pre-match: R² = 0.28-0.35
- Mid-match: R² = 0.90-0.93
- Late-match: R² = 0.95-0.97

---

## Usage

```python
import pickle

# Load model
pipe = pickle.load(open('models/odi_progressive_pipe.pkl', 'rb'))

# Predict
scenario = pd.DataFrame([{
    'batting_team': 'India',
    'city': 'Mumbai',
    'current_score': 180,
    'balls_left': 120,
    'wickets_left': 7,
    'crr': 6.0,
    'last_10_overs': 65,
    'team_batting_avg': 38.5
}])

prediction = pipe.predict(scenario)[0]
print(f'Predicted final score: {prediction:.0f} runs')
```

---

## Files

**Scripts:**
- `BUILD_AND_TRAIN.py` - Complete build pipeline
- `TEST_PREDICTIONS.py` - Test with real scenarios

**Models:**
- `models/odi_progressive_pipe.pkl` - Trained model

**Data:**
- Created during build (not saved to reduce size)

