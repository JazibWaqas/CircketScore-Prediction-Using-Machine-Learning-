# 🏏 ODI Cricket Score Prediction with Player-Level Impact

## 🎯 **PROJECT OVERVIEW**

Build an ODI cricket score prediction system that can:
- **Predict match scores** with 85-90% accuracy (within ±20 runs)
- **Analyze individual player impact** (Virat Kohli adds 40+ runs vs average player)
- **Handle what-if scenarios** (team composition changes, player swaps)
- **Recognize star player contributions** (based on batting averages, strike rates)
- **Provide realistic predictions** based on actual player performance
- **Explain predictions** (why score changed when swapping players)

**This is the ACTIVE project** - T20 was deprecated due to format variance making player impact unmeasurable.

---

## 📊 **AVAILABLE DATA**

### **ODI Match Data:**
- **Location**: `../raw_data/odis_ballbyBall/`
- **Files**: 5,761 JSON files (ball-by-ball data)
- **Coverage**: International ODI matches with complete ball-by-ball details

### **ODI Player Performance Data:**
- **Location**: `../raw_data/odi_data/detailed_player_data.csv`
- **Records**: 52,033 player performances
- **Fields**:
  - `match_id, player, team, opposition, venue`
  - `runs, balls_faced, fours, sixes, strike_rate`
  - `wickets, overs_bowled, balls_bowled, runs_conceded, economy`
  - `catches, run_outs, maiden, stumps`
  - `fantasy_points, match_outcome`

---

## 🛠️ **PROJECT STRUCTURE**

```
ODI/
├── data/                   # Training/test datasets
│   └── [Will be created]
│
├── models/                 # Trained ML models
│   └── [Will be created]
│
├── scripts/                # Data processing scripts
│   ├── 1_build_player_database.py
│   ├── 2_create_training_dataset.py
│   ├── 3_train_models.py
│   └── 4_evaluate_predictions.py
│
├── processed_data/         # Cleaned datasets
│   └── [Will be created]
│
├── results/                # Model evaluation results
│   └── [Will be created]
│
└── README.md              # This file
```

---

## 🚀 **DEVELOPMENT ROADMAP**

### **✅ Phase 1: Data Understanding (COMPLETED)**
- [x] Located ODI ball-by-ball data (5,761 matches)
- [x] Found ODI player performance data (52K+ records)
- [x] Verified data structure and completeness
- [x] Confirmed player statistics availability

### **📋 Phase 2: Player Performance Database (Current)**

#### **Goal**: Create comprehensive player statistics lookup

#### **Tasks**:
1. **Aggregate player statistics** from 52K+ records
   - Calculate career batting average per player
   - Calculate career strike rate per player
   - Calculate career bowling average per player
   - Calculate career economy rate per player
   - Count total matches, runs, wickets per player

2. **Build player rating system**
   - Classify players by role (batsman, bowler, all-rounder, wicketkeeper)
   - Create star rating (1-10) based on performance
   - Identify top performers (avg > 35 for batsmen, economy < 5.0 for bowlers)

3. **Create player lookup database**
   ```python
   player_db = {
       'V Kohli': {
           'batting_avg': 58.8,
           'strike_rate': 93.2,
           'role': 'batsman',
           'star_rating': 9.8,
           'matches': 260,
           'runs': 12898
       }
   }
   ```

#### **Expected Output**:
- `processed_data/odi_player_statistics.csv`
- `processed_data/player_ratings.csv`
- `data/player_lookup.json`

---

### **📋 Phase 3: Training Dataset Creation**

#### **Goal**: Build match-level dataset with player features

#### **Tasks**:
1. **Process ball-by-ball match data**
   - Extract match-level information
   - Identify players for each team
   - Calculate team aggregations

2. **Add player-specific features**
   ```python
   features = {
       # Team batting features
       'team_batting_avg': sum(player_batting_avgs) / 11,
       'team_strike_rate': sum(player_strike_rates) / 11,
       'star_batsmen_count': count(players with avg > 35),
       
       # Team bowling features
       'team_bowling_avg': sum(player_bowling_avgs) / 11,
       'team_economy_rate': sum(player_economy_rates) / 11,
       'star_bowlers_count': count(players with economy < 5),
       
       # Team composition
       'all_rounder_count': count(all_rounders),
       'team_balance_score': batting_strength / bowling_strength,
       
       # Opposition features
       'opposition_batting_avg': ...,
       'opposition_bowling_avg': ...,
       
       # Context features
       'venue, tournament, toss_decision, season, match_number'
   }
   ```

3. **Create train/test split**
   - Chronological split (80% old, 20% recent)
   - Ensure data leakage prevention
   - Balance classes and features

#### **Expected Output**:
- `data/odi_training_dataset.csv`
- `data/odi_test_dataset.csv`
- `processed_data/feature_descriptions.txt`

---

### **📋 Phase 4: Model Training**

#### **Goal**: Train ML models with player-aware features

#### **Models to Train**:
1. **Linear Regression** (Baseline)
   - Simple, interpretable
   - Good for understanding feature importance

2. **Random Forest**
   - Handles non-linear relationships
   - Robust to outliers
   - Feature importance analysis

3. **XGBoost**
   - Best performance expected
   - Gradient boosting
   - GPU acceleration

#### **Training Process**:
```python
# Feature engineering
X = dataset[['team_batting_avg', 'team_bowling_avg', 'star_player_count', ...]]
y = dataset['total_runs']

# Train-test split (chronological)
X_train, X_test, y_train, y_test = chronological_split(X, y)

# Train models
models = {
    'Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200),
    'XGBoost': XGBRegressor(tree_method='gpu_hist')
}

# Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate(predictions, y_test)
```

#### **Expected Metrics**:
- **R²**: > 0.85
- **RMSE**: < 20 runs
- **MAE**: < 15 runs
- **Accuracy within ±20 runs**: > 85%

#### **Expected Output**:
- `models/odi_linear_regression.pkl`
- `models/odi_random_forest.pkl`
- `models/odi_xgboost.pkl`
- `results/model_comparison.csv`
- `results/feature_importance.csv`

---

### **📋 Phase 5: Testing & Validation**

#### **Goal**: Validate player impact predictions

#### **Test Scenarios**:
1. **Individual Player Swap**
   ```python
   # Test Babar Azam vs Virat Kohli
   team_with_babar = ['Babar Azam', 'Fakhar Zaman', ...]
   team_with_kohli = ['Virat Kohli', 'Fakhar Zaman', ...]
   
   prediction_babar = predict(team_with_babar)
   prediction_kohli = predict(team_with_kohli)
   
   # Expected: Kohli team scores 30-50 runs more
   ```

2. **Star Player Recognition**
   ```python
   # Test with all star players vs average players
   star_team = ['Kohli', 'Sharma', 'De Villiers', ...]
   average_team = [average_players...]
   
   # Expected: Star team scores 50-80 runs more
   ```

3. **Team Composition Impact**
   ```python
   # Test balanced vs unbalanced team
   balanced = [5 batsmen, 5 bowlers, 1 all-rounder]
   unbalanced = [8 batsmen, 3 bowlers]
   
   # Expected: Balanced team performs better
   ```

4. **Real Match Validation**
   - Test on recent ODI matches
   - Compare predictions vs actual scores
   - Analyze prediction errors

#### **Expected Output**:
- `results/player_impact_test.csv`
- `results/real_match_validation.csv`
- `results/accuracy_report.md`

---

## 📈 **SUCCESS CRITERIA**

### **Model Performance**:
- ✅ **Accuracy**: 85-90% within ±20 runs
- ✅ **R²**: > 0.85
- ✅ **RMSE**: < 20 runs

### **Player Impact**:
- ✅ **Star player impact**: 30-50 runs difference
- ✅ **Team composition**: Measurable effect on predictions
- ✅ **Player swapping**: Realistic score changes

### **System Functionality**:
- ✅ **What-if scenarios**: Works correctly
- ✅ **Real-time predictions**: Fast (<1 second)
- ✅ **Interpretability**: Feature importance clear

---

## 🔧 **TECHNICAL REQUIREMENTS**

### **Python Packages**:
```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

### **Data Requirements**:
- ODI match data (5,761 matches) ✅
- ODI player performance (52K+ records) ✅
- Player statistics extraction ⏳
- Training dataset creation ⏳

---

## 🎯 **NEXT IMMEDIATE STEPS**

1. **Create `scripts/1_build_player_database.py`**
   - Load `../raw_data/odi_data/detailed_player_data.csv`
   - Aggregate player statistics
   - Calculate career averages
   - Create player rating system
   - Save to `processed_data/odi_player_statistics.csv`

2. **Create `scripts/2_create_training_dataset.py`**
   - Load ball-by-ball match data
   - Extract team compositions
   - Add player features from player database
   - Create team-level aggregations
   - Save to `data/odi_training_dataset.csv`

3. **Create `scripts/3_train_models.py`**
   - Load training dataset
   - Train Linear, RF, XGBoost models
   - Evaluate and compare
   - Save models to `models/`

4. **Create `scripts/4_evaluate_predictions.py`**
   - Test player impact scenarios
   - Validate on real matches
   - Generate accuracy reports

---

## 📚 **RESOURCES**

- **ODI Player Data**: `../raw_data/odi_data/detailed_player_data.csv`
- **ODI Matches**: `../raw_data/odis_ballbyBall/*.json`
- **T20 Reference**: `../T20/` (for code patterns, not data)

---

**Let's build an ODI prediction system where player impact is real and measurable!**
