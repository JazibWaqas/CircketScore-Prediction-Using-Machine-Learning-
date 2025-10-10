# 🏏 Cricket Score Prediction with Real Player Impact

**A machine learning system that predicts ODI cricket match scores based on actual player performance, team composition, and match context.**

---

## 🚨 **IMPORTANT: READ FIRST**

**📚 [DOCUMENTATION INDEX](DOCUMENTATION_INDEX.md)** - All project docs in one place

**⚠️ ODI Model Status:** Currently broken (R²=0.01, not 0.69 as previously claimed)  
**📄 Read:** [ODI/START_HERE_TOMORROW.md](ODI/START_HERE_TOMORROW.md) for quick overview  
**📄 Or:** [ODI/PROJECT_STATUS_CRITICAL_ISSUES.md](ODI/PROJECT_STATUS_CRITICAL_ISSUES.md) for full details

**✅ T20 System:** Fully working (R² ~0.65-0.70)  
**✅ Frontend:** Complete and beautiful (both T20/ODI toggle)  
**❌ ODI Predictions:** Need model rebuild (6-8 hours work)

---

## 🎯 **PROJECT VISION**

Build a **realistic ODI cricket score prediction system** that truly understands cricket by:

### **Primary Goals:**
1. **Accurate Score Prediction**: 85-90% accuracy within ±20 runs
2. **Individual Player Impact**: Virat Kohli scores 30-50 runs more than average players
3. **What-If Scenarios**: Answer "What if I swap Babar Azam with Rohit Sharma?"
4. **Star Player Recognition**: System knows elite players contribute significantly more
5. **Team Composition Analysis**: Different combinations give meaningfully different predictions
6. **Explainable Predictions**: Understand WHY scores change when you swap players

### **The Dream Scenario:**
```
User: "Predict India vs Pakistan at Dubai"
- India: Rohit (avg 49), Kohli (avg 58), KL Rahul (avg 46), ... 
- Pakistan: Babar (avg 56), Rizwan (avg 47), Fakhar (avg 38), ...

System: India 310, Pakistan 295

User: "What if I swap Kohli with an average player?"
System: India 275, Pakistan 295 (India now loses! -35 runs from losing Kohli)

User: "What if I add Bumrah (bowling avg 24, economy 4.6)?"
System: Pakistan 260 (better bowling reduces opposition by 35 runs)
```

---

## 🚨 **WHY ODI INSTEAD OF T20?**

### **T20 Project (Deprecated - See `T20/` folder):**
- ❌ **Too volatile**: 120 balls/innings makes individual player impact minimal
- ❌ **High variance**: Same player can score 5 or 80 (unpredictable)
- ❌ **Format limitation**: Even star players have ±5-10 run impact (negligible)
- ❌ **Fundamental issue**: Player quality doesn't reliably predict performance in T20

### **ODI Project (Current Focus):**
- ✅ **Lower variance**: 300 balls/innings allows player quality to shine
- ✅ **Measurable impact**: Star players consistently contribute 30-50 runs more
- ✅ **Statistical reliability**: Player averages actually predict performance
- ✅ **Clear differentiation**: Virat Kohli (avg 58) >> Average player (avg 28)

**T20 is fundamentally too random for reliable player-level prediction. ODI is where cricket skill actually matters.**

---

## 📊 **AVAILABLE DATA**

### **ODI Match Data:**
- **Location**: `raw_data/odis_ballbyBall/`
- **Size**: 5,761 JSON files (ball-by-ball data)
- **Coverage**: International ODI matches with complete details
- **Content**: Teams, players, overs, runs, wickets, outcomes

### **ODI Player Performance Data:**
- **Location**: `raw_data/odi_data/detailed_player_data.csv`
- **Size**: 52,033 player performance records
- **Fields**:
  - **Batting**: runs, balls_faced, fours, sixes, strike_rate
  - **Bowling**: wickets, overs_bowled, runs_conceded, economy, maiden
  - **Fielding**: catches, run_outs, stumps
  - **Context**: match_id, team, opposition, venue, match_outcome, fantasy_points

**Example Record:**
```csv
RG Sharma,India,264,176,33,9,strike_rate=150.0,venue=Eden Gardens,outcome=win
```

---

## 📁 **REPOSITORY STRUCTURE**

```
├── ODI/                           # ODI Project (ACTIVE)
│   ├── scripts/                   # Data processing & training scripts
│   │   ├── 1_build_player_database.py        # Extract player stats
│   │   ├── 2_create_training_dataset.py      # Build match dataset
│   │   ├── 3_train_models.py                 # Train ML models
│   │   └── 4_evaluate_player_impact.py       # Test player swaps
│   ├── data/                      # Training/test datasets
│   ├── models/                    # Trained ML models
│   ├── processed_data/            # Cleaned datasets
│   ├── results/                   # Evaluation results
│   └── README.md                  # Detailed ODI guide
│
├── T20/                           # T20 Project (DEPRECATED)
│   ├── Database/                  # Flask API, SQLite DB
│   ├── models/                    # Trained T20 models
│   ├── data/, scripts/, results/
│   └── PROJECT_STATUS_README.md   # Why T20 was deprecated
│
├── frontend/                      # React Web Application
│   ├── src/
│   │   ├── App.js                # Main application
│   │   └── components/           # Team selector, predictions, etc.
│   └── public/
│
├── raw_data/                      # Raw Cricket Data
│   ├── odis_ballbyBall/          # 5,761 ODI matches (JSON)
│   ├── odi_data/                 # 52K+ ODI player performances (CSV)
│   └── t20 matches ball by ball/ # 7,223 T20 matches (reference)
│
└── README.md                      # This file
```

---

## 🚀 **PROJECT ROADMAP**

### **Phase 1: Player Performance Intelligence (Days 1-2)**
**Goal**: Build comprehensive player database that knows every player's skill level

#### **Tasks**:
1. **Process 52,033 player performances** to calculate per player:
   - Career batting average (runs/dismissals)
   - Career strike rate (runs/balls × 100)
   - Career bowling average (runs_conceded/wickets)
   - Career economy rate (runs_conceded/overs)
   - Consistency score (variance in performance)
   - Matches played, total runs, total wickets

2. **Classify players** by:
   - **Role**: Batsman, Bowler, All-rounder, Wicketkeeper
   - **Skill level**: 
     - Elite (batting avg 45+, bowling avg <25)
     - Star (batting avg 35-45, bowling avg 25-30)
     - Good (batting avg 25-35, bowling avg 30-35)
     - Average (batting avg <25, bowling avg >35)
   - **Star rating**: 1-10 based on overall performance

3. **Identify specialists**:
   - Power hitters (high strike rate, lots of sixes)
   - Anchors (high average, low strike rate)
   - Death bowlers (good economy in last 10 overs)
   - Wicket-takers (high wickets per match)

#### **Output**:
- `ODI/processed_data/player_statistics.csv` - Career stats per player
- `ODI/processed_data/player_ratings.json` - Skill ratings & classifications
- `ODI/data/player_lookup.json` - Quick lookup database

#### **Script**: `ODI/scripts/1_build_player_database.py`

---

### **Phase 2: Match-Level Dataset Creation (Days 2-3)**
**Goal**: Build training data where each match has player-aware features

#### **Tasks**:
1. **For each of 5,761 ODI matches**:
   - Extract both teams' 11 players from ball-by-ball data
   - Look up each player's career statistics
   - Calculate team-level aggregations

2. **Generate team features**:
   ```python
   # Team batting features
   team_batting_avg = mean([player.batting_avg for player in team])
   team_strike_rate = mean([player.strike_rate for player in team])
   star_batsmen_count = count([player for player in team if player.batting_avg > 35])
   power_hitters = count([player for player in team if player.strike_rate > 100])
   
   # Team bowling features
   team_bowling_avg = mean([player.bowling_avg for player in team])
   team_economy = mean([player.economy for player in team])
   star_bowlers_count = count([player for player in team if player.bowling_avg < 30])
   
   # Team composition
   all_rounder_count = count([player.role == 'all-rounder' for player in team])
   team_balance = batting_strength / bowling_strength
   team_depth = count([player.batting_avg > 25 for player in team])
   ```

3. **Add context features**:
   - Venue average score (from historical data)
   - Opposition strength (their team features)
   - Tournament type (World Cup, Bilateral, etc.)
   - Toss decision (bat/field)
   - Season, year, match importance

4. **Create opposition-relative features**:
   ```python
   batting_vs_bowling = team_batting_avg - opposition_bowling_avg
   bowling_vs_batting = team_bowling_avg - opposition_batting_avg
   star_player_advantage = team_star_count - opposition_star_count
   ```

#### **Output**:
- `ODI/data/odi_training_dataset.csv` - 5,261 matches (training)
- `ODI/data/odi_test_dataset.csv` - 500 recent matches (testing)
- `ODI/processed_data/feature_descriptions.txt` - Feature explanations

#### **Script**: `ODI/scripts/2_create_training_dataset.py`

---

### **Phase 3: Model Training & Evaluation (Day 4)**
**Goal**: Train models that understand player impact

#### **Models**:
1. **Linear Regression** - Baseline, interpretable
2. **Random Forest** - Best for feature importance analysis
3. **XGBoost** - Best accuracy, gradient boosting

#### **Key Features** (30-40 total):
| Feature Category | Examples |
|-----------------|----------|
| **Team Batting** | `team_batting_avg`, `team_strike_rate`, `star_batsmen_count`, `power_hitters` |
| **Team Bowling** | `team_bowling_avg`, `team_economy`, `star_bowlers_count` |
| **Team Balance** | `all_rounder_count`, `team_depth`, `balance_score` |
| **Opposition** | `opposition_batting_avg`, `opposition_bowling_avg`, `opposition_star_count` |
| **Relative Strength** | `batting_vs_bowling`, `star_player_advantage` |
| **Context** | `venue_avg_score`, `toss_decision`, `tournament_type`, `season` |

#### **Target Variable**: `total_runs` (team's final score)

#### **Training Process**:
```python
# Chronological split (older matches = training, recent = testing)
X_train = matches[matches.year <= 2022][features]
y_train = matches[matches.year <= 2022]['total_runs']
X_test = matches[matches.year > 2022][features]
y_test = matches[matches.year > 2022]['total_runs']

# Train models
models = {
    'Linear': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=200),
    'XGBoost': XGBRegressor(n_estimators=200, tree_method='gpu_hist')
}

# Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = calculate_accuracy_within_20_runs(predictions, y_test)
```

#### **Expected Performance**:
- **R²**: > 0.85
- **RMSE**: < 20 runs
- **MAE**: < 15 runs
- **Accuracy (±20 runs)**: > 85%

#### **Output**:
- `ODI/models/odi_linear_regression.pkl`
- `ODI/models/odi_random_forest.pkl`
- `ODI/models/odi_xgboost.pkl`
- `ODI/results/model_comparison.csv`
- `ODI/results/feature_importance.csv`

#### **Script**: `ODI/scripts/3_train_models.py`

---

### **Phase 4: Player Impact Testing & Validation (Day 5)**
**Goal**: Validate that player swaps have realistic impact

#### **Test Scenarios**:

1. **Individual Star Player Impact**
   ```python
   team_with_kohli = ['V Kohli (avg 58)', 'Player 2', ..., 'Player 11']
   team_with_average = ['Average Player (avg 28)', 'Player 2', ..., 'Player 11']
   
   prediction_kohli = predict(team_with_kohli)
   prediction_average = predict(team_with_average)
   
   impact = prediction_kohli - prediction_average
   # Expected: 30-50 runs difference
   ```

2. **Star Player Comparison**
   ```python
   team_with_kohli = [..., 'V Kohli (avg 58)', ...]
   team_with_babar = [..., 'Babar Azam (avg 56)', ...]
   team_with_rahul = [..., 'KL Rahul (avg 46)', ...]
   
   # Expected: Kohli > Babar > Rahul (small but measurable differences)
   ```

3. **Star Team vs Average Team**
   ```python
   star_team = ['Kohli', 'Rohit', 'Babar', 'De Villiers', ...] # All avg > 40
   average_team = [all players with avg ~28]
   
   # Expected: Star team scores 60-80 runs more
   ```

4. **Elite Bowler Impact**
   ```python
   opposition_with_bumrah = [..., 'J Bumrah (economy 4.6, avg 24)']
   opposition_without_bumrah = [..., 'Average bowler (economy 5.8, avg 35)']
   
   prediction_with = predict(team_batting_against_bumrah)
   prediction_without = predict(team_batting_against_average)
   
   # Expected: Team scores 20-30 runs less against Bumrah
   ```

5. **Team Composition Balance**
   ```python
   balanced = [5 batsmen, 5 bowlers, 1 all-rounder]
   unbalanced = [8 batsmen, 3 bowlers]
   
   # Expected: Balanced team performs better overall
   ```

6. **Real Match Validation**
   - Test on recent ODI matches (2023-2024)
   - Compare predictions vs actual scores
   - Analyze where predictions fail and why

#### **Success Criteria**:
- ✅ Elite players (avg 50+) show 40-50 run impact
- ✅ Star players (avg 35-45) show 20-30 run impact
- ✅ Good players (avg 25-35) show 10-15 run impact
- ✅ Team composition matters significantly
- ✅ Predictions are explainable and realistic

#### **Output**:
- `ODI/results/player_impact_tests.csv`
- `ODI/results/real_match_validation.csv`
- `ODI/results/accuracy_report.md`

#### **Script**: `ODI/scripts/4_evaluate_player_impact.py`

---

### **Phase 5: Web Interface Integration (Days 6-7)**
**Goal**: Update frontend to work with ODI predictions

#### **Tasks**:
1. **Create ODI API** (Flask):
   - Load trained models
   - Accept team compositions via REST API
   - Calculate real-time player features
   - Return predictions with explanations

2. **Update Frontend** (React):
   - Fetch ODI players from database
   - Show player ratings and statistics
   - Allow team composition building
   - Display predictions with player impact breakdown

3. **Add Explanation Features**:
   - Show why score changed when swapping players
   - Display player contribution estimates
   - Highlight star players vs average players

#### **API Endpoint**:
```python
POST /api/predict
{
  "team_a": ["V Kohli", "RG Sharma", ..., 11 players],
  "team_b": ["Babar Azam", "Mohammad Rizwan", ..., 11 players],
  "venue": "Dubai International Stadium",
  "toss_decision": "bat",
  "tournament": "World Cup"
}

Response:
{
  "prediction": {
    "team_a_score": 310,
    "team_b_score": 285,
    "winner": "Team A",
    "confidence": 0.87
  },
  "player_impact": {
    "V Kohli": "+45 runs (elite batsman)",
    "RG Sharma": "+38 runs (elite batsman)",
    "Babar Azam": "+42 runs (elite batsman)"
  },
  "explanation": "Team A's superior batting lineup (avg 41.2 vs 37.8) gives 23-run advantage..."
}
```

---

## 🎯 **WHAT MAKES THIS PROJECT DIFFERENT**

### **Most Cricket Prediction Systems:**
- ❌ Only predict win/loss (binary classification)
- ❌ Use team names instead of player composition
- ❌ Cannot explain why predictions change
- ❌ Don't account for individual player quality

### **Our System:**
- ✅ Predicts exact scores (regression)
- ✅ Uses actual player performance data
- ✅ Explains player-level impact
- ✅ Handles what-if scenarios
- ✅ Recognizes star players vs average players
- ✅ Realistic and interpretable

---

## 🔧 **TECHNOLOGY STACK**

### **Machine Learning:**
- **Python 3.8+**: Core language
- **scikit-learn**: Linear Regression, Random Forest
- **XGBoost**: Gradient boosting
- **pandas/numpy**: Data processing
- **joblib**: Model serialization

### **Backend:**
- **Flask**: REST API server
- **SQLite**: Player/venue database
- **JSON**: Data interchange

### **Frontend:**
- **React 18**: UI framework
- **Tailwind CSS**: Styling
- **Axios**: API communication

---

## 🚀 **GETTING STARTED**

### **Prerequisites:**
```bash
# Python 3.8+ with pip
pip install pandas numpy scikit-learn xgboost joblib flask flask-cors matplotlib seaborn

# Node.js 14+ with npm (for frontend)
cd frontend
npm install
```

### **Run ODI System:**
```bash
# Step 1: Build player database
cd ODI/scripts
python 1_build_player_database.py

# Step 2: Create training dataset
python 2_create_training_dataset.py

# Step 3: Train models
python 3_train_models.py

# Step 4: Test player impact
python 4_evaluate_player_impact.py

# Step 5: Start API (coming soon)
cd ../api
python app.py

# Step 6: Start frontend (coming soon)
cd ../../frontend
npm start
```

---

## 📈 **EXPECTED OUTCOMES**

### **Model Performance:**
| Metric | Target | Expected Reality |
|--------|--------|------------------|
| **R²** | > 0.85 | 0.87-0.92 |
| **RMSE** | < 20 runs | 15-18 runs |
| **MAE** | < 15 runs | 12-14 runs |
| **Accuracy (±20)** | > 85% | 87-92% |

### **Player Impact:**
| Scenario | Expected Impact |
|----------|----------------|
| **Elite Player (avg 50+)** | +40-50 runs vs average |
| **Star Player (avg 35-45)** | +25-35 runs vs average |
| **Good Player (avg 25-35)** | +10-20 runs vs average |
| **Elite Bowler (econ <5)** | -25-35 runs to opposition |
| **All-Rounder** | +15-25 runs combined |

### **Team Composition:**
| Scenario | Expected Difference |
|----------|-------------------|
| **11 Star Players vs 11 Average** | 60-80 runs |
| **Balanced vs Unbalanced** | 20-30 runs |
| **With vs Without Key Player** | 30-50 runs |

---

## ⚠️ **IMPORTANT NOTES**

### **Why T20 Was Abandoned:**
1. **Format variance** - 120 balls too small for player skill to show
2. **High randomness** - Single over can swing 30+ runs
3. **Minimal player impact** - Even stars only contribute ±5-10 runs
4. **Unpredictable** - Same player scores 5 or 80 (no consistency)

### **Why ODI Works:**
1. **Lower variance** - 300 balls allows skill to emerge
2. **Predictable** - Player averages match actual performance
3. **Measurable impact** - Stars consistently contribute 30-50 runs more
4. **Statistical reliability** - Historical data predicts future performance

---

## 📚 **KEY DOCUMENTATION**

- `ODI/README.md` - Detailed ODI project guide
- `T20/PROJECT_STATUS_README.md` - Why T20 was deprecated
- `REPOSITORY_REORGANIZATION_COMPLETE.md` - Structure changes

---

## 🎯 **CURRENT STATUS**

- [x] Repository reorganized (T20 deprecated, ODI active)
- [x] ODI data verified (5,761 matches, 52K+ player performances)
- [x] Complete project plan created
- [ ] Phase 1: Player database (next step)
- [ ] Phase 2: Training dataset
- [ ] Phase 3: Model training
- [ ] Phase 4: Player impact testing
- [ ] Phase 5: Web interface

---

## 🏆 **PROJECT SUCCESS = Real Player Impact**

**This project succeeds when:**
- Swapping Virat Kohli with an average player changes prediction by 35+ runs
- Star teams score 60-80 runs more than average teams
- Adding an elite bowler reduces opposition by 25+ runs
- Predictions are explainable: "Team A wins because their batting lineup averages 41 vs opponent's 38"

**Not just predictions - understanding WHY.**

---

**Let's build a cricket prediction system that actually understands cricket!** 🏏