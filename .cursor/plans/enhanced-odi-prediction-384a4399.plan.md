<!-- 384a4399-5774-469d-9299-39b6c3befa26 84c42999-eb89-4a02-a3f4-23ce48f68556 -->
# Enhanced ODI Cricket Score Prediction System

## Overview

Enhance the existing ODI prediction system to achieve:

- **High Accuracy**: R² > 0.75, MAE < 25 runs (improving from current R²=0.69, MAE=28.67)
- **Player Impact**: Star players show 20-40 run impact in what-if scenarios
- **Hybrid Features**: Individual key players + aggregated team statistics
- **Validated Performance**: Test with 500 real historical matches

## Current State

- Dataset: `odi_complete_dataset.csv` (7,315 rows, 66 features)
- Models: XGBoost trained (R²=0.69, MAE=28.67)
- Raw Data: 5,761 ball-by-ball matches + 52,033 player performance records
- Issue: Limited player impact detection, accuracy needs improvement

## Implementation Plan

### Phase 1: Enhanced Feature Engineering

**File**: `ODI/scripts/BUILD_ENHANCED_DATASET.py`

Create a new comprehensive dataset with:

1. **Individual Key Player Features** (Top 3-4 players per team):

- Top batsman batting average, strike rate, recent form
- Top bowler economy, wickets, recent form  
- All-rounder contributions
- Encode as: `top_bat_1_avg`, `top_bat_2_avg`, etc.

2. **Recent Form Features** (Last 10 matches):

- Player-specific: runs scored, wickets taken, performance trend
- Team-specific: win rate, average score, momentum
- Calculate rolling averages to avoid data leakage

3. **Venue-Specific Features**:

- Player performance at specific venue (if played there before)
- Team performance at venue (historical average)
- Venue characteristics (avg score, high/low, pitch type)

4. **Opposition-Specific Features**:

- Player performance vs specific opponent
- Head-to-head team statistics
- Historical matchup patterns

5. **Enhanced Match Context**:

- Tournament importance (World Cup=5, Bilateral=1)
- Match pressure (Final=5, Group=1)
- Home advantage indicator
- Rivalry match indicator (India-Pakistan, Aus-Eng)

6. **Aggregated Team Features** (Current + Enhanced):

- Team batting strength (weighted by player quality)
- Team bowling strength (considering all bowlers)
- Team depth (number of players with avg > 30)
- Star player count (elite + star ratings)

7. **Interaction Features**:

- Best batsman vs best bowler quality differential
- Power hitters vs death bowlers
- Team batting strength vs opposition bowling strength

**Output**: `odi_enhanced_dataset.csv` (~100-120 features)

### Phase 2: Data Cleaning & Validation

**File**: `ODI/scripts/CLEAN_AND_VALIDATE_DATASET.py`

1. **Remove Outliers**:

- Scores < 50 runs (abandoned/abnormal matches)
- Scores > 450 runs (data errors)
- Matches with < 40 overs (rain-affected)

2. **Handle Missing Values**:

- Impute player stats with team averages
- Fill missing venue data with global averages
- Document imputation strategy

3. **Check Data Leakage**:

- Verify no future data used in past predictions
- Chronological sorting maintained
- Form calculations use only past matches

4. **Feature Correlation Analysis**:

- Remove highly correlated features (r > 0.95)
- Identify multicollinearity issues
- Keep most predictive features

5. **Stratified Split**:

- Training: 6,815 rows (oldest matches)
- Testing: 500 rows (most recent matches)
- Stratify by score ranges to ensure balanced distribution

**Output**: `odi_enhanced_cleaned.csv` + validation report

### Phase 3: Model Training & Optimization

**File**: `ODI/scripts/TRAIN_ENHANCED_MODELS.py`

1. **Train Multiple Models**:

- **XGBoost**: Hyperparameter tuning (learning_rate, max_depth, n_estimators)
- **Random Forest**: Optimize n_estimators, max_depth, min_samples_split
- Consider Gradient Boosting as backup

2. **Feature Selection**:

- Use feature importance from Random Forest
- Keep top 80-100 most important features
- Remove redundant features

3. **Cross-Validation**:

- 5-fold time-series cross-validation
- Report mean R², MAE, RMSE across folds
- Check for overfitting (train vs test performance)

4. **Hyperparameter Tuning**:

- Grid search for best parameters
- Focus on reducing MAE while maintaining R²
- Early stopping to prevent overfitting

5. **Model Evaluation**:

- R² score (target > 0.75)
- MAE (target < 25 runs)
- RMSE (target < 35 runs)
- Accuracy within ±20 runs (target > 85%)
- Accuracy within ±30 runs (target > 92%)

**Output**:

- `xgboost_ENHANCED.pkl`
- `random_forest_ENHANCED.pkl`
- `scaler_ENHANCED.pkl`
- `feature_names_ENHANCED.pkl`
- Training report with metrics

### Phase 4: Player Impact Validation

**File**: `ODI/scripts/VALIDATE_PLAYER_IMPACT.py`

Test what-if scenarios to ensure player swaps have realistic impact:

1. **Individual Star Player Tests**:

- Swap Virat Kohli (avg 57) with average player (avg 28)
- Expected impact: 25-40 runs difference
- Test with 20 random matches

2. **Elite vs Good Player Tests**:

- Compare Babar Azam (avg 56) vs KL Rahul (avg 46)
- Expected impact: 8-15 runs difference
- Test with 20 random matches

3. **Bowling Impact Tests**:

- Swap Jasprit Bumrah (econ 4.3) with average bowler (econ 5.5)
- Expected opposition score reduction: 15-25 runs
- Test with 20 random matches

4. **Team Composition Tests**:

- All-star team vs all-average team
- Expected difference: 60-80 runs
- Test with 10 synthetic matchups

5. **Validation Metrics**:

- Player impact correlation (should be positive)
- Direction correctness (better player = higher score)
- Magnitude reasonableness (not too small or too large)

**Output**: Player impact validation report with test results

### Phase 5: Real Match Testing & Analysis

**File**: `ODI/scripts/TEST_HISTORICAL_MATCHES.py`

1. **Test on 500 Reserved Matches**:

- Load actual match data (teams, players, venue, outcome)
- Generate predictions using trained model
- Compare predicted vs actual scores
- Calculate accuracy metrics

2. **Error Analysis**:

- Identify patterns in errors (high-scoring vs low-scoring)
- Check if certain teams/venues have higher errors
- Analyze if specific match contexts cause issues

3. **Performance by Category**:

- Accuracy by score range (150-200, 200-250, 250-300, 300+)
- Accuracy by tournament type
- Accuracy by team strength

4. **Visualizations**:

- Predicted vs Actual scatter plot
- Error distribution histogram
- Feature importance bar chart
- Player impact demonstration

**Output**:

- `test_results.csv` (500 matches with predictions)
- `error_analysis_report.md`
- Performance visualizations

### Phase 6: Model Comparison & Selection

**File**: `ODI/scripts/COMPARE_MODELS.py`

1. **Compare All Models**:

- Old COMPLETE model (R²=0.69, MAE=28.67)
- New ENHANCED XGBoost
- New ENHANCED Random Forest

2. **Metrics Comparison**:

- R², MAE, RMSE side-by-side
- Player impact effectiveness
- Inference speed
- Feature importance differences

3. **Select Best Model**:

- Primary: Best accuracy (R² and MAE)
- Secondary: Player impact detection
- Tertiary: Interpretability

4. **Final Model Package**:

- Best model saved as `odi_final_model.pkl`
- All encoders and scalers
- Feature names list
- Usage documentation

**Output**: Model comparison report + final model selection

## Key Files Created/Modified

**New Scripts**:

- `ODI/scripts/BUILD_ENHANCED_DATASET.py` - Enhanced feature engineering
- `ODI/scripts/CLEAN_AND_VALIDATE_DATASET.py` - Data cleaning pipeline
- `ODI/scripts/TRAIN_ENHANCED_MODELS.py` - Model training with tuning
- `ODI/scripts/VALIDATE_PLAYER_IMPACT.py` - Player swap validation
- `ODI/scripts/TEST_HISTORICAL_MATCHES.py` - Real match testing
- `ODI/scripts/COMPARE_MODELS.py` - Model comparison

**New Data Files**:

- `ODI/data/odi_enhanced_dataset.csv` - Full enhanced dataset
- `ODI/data/odi_enhanced_cleaned.csv` - Cleaned version
- `ODI/data/test_matches_500.csv` - Reserved test set

**New Models**:

- `ODI/models/xgboost_ENHANCED.pkl`
- `ODI/models/random_forest_ENHANCED.pkl`
- `ODI/models/scaler_ENHANCED.pkl`
- `ODI/models/feature_names_ENHANCED.pkl`
- `ODI/models/odi_final_model.pkl` - Best selected model

**Results**:

- `ODI/results/training_report.md`
- `ODI/results/player_impact_validation.md`
- `ODI/results/test_results.csv`
- `ODI/results/error_analysis_report.md`
- `ODI/results/model_comparison.md`

## Success Criteria

### Accuracy Metrics (Primary Goal)

- ✅ R² Score > 0.75 (current: 0.69)
- ✅ MAE < 25 runs (current: 28.67)
- ✅ RMSE < 35 runs
- ✅ Accuracy within ±20 runs > 85%
- ✅ Accuracy within ±30 runs > 92%

### Player Impact (Secondary Goal)

- ✅ Elite player (avg 50+): 25-40 run impact
- ✅ Star player (avg 35-50): 15-25 run impact
- ✅ Good player (avg 25-35): 8-15 run impact
- ✅ Elite bowler: 15-25 run opposition reduction
- ✅ Impact direction always correct (better = higher)

### Validation & Testing

- ✅ 500 test matches evaluated
- ✅ Error patterns analyzed and documented
- ✅ No data leakage confirmed
- ✅ Chronological integrity maintained
- ✅ Player swaps produce sensible results

## Technical Approach

**Feature Strategy**:

- Hybrid: 20-30 individual player features + 50-70 aggregated features
- Recent form: Rolling 10-match window
- Venue/opponent: Conditional averages with fallback
- Interaction: Top player vs opposition strength

**Data Quality**:

- Remove abnormal matches (< 50 or > 450 runs)
- Impute missing data conservatively
- Chronological split (train on old, test on new)
- Stratified sampling by score range

**Model Training**:

- XGBoost with hyperparameter tuning
- Random Forest for feature importance
- 5-fold time-series cross-validation
- Early stopping to prevent overfitting

**Validation**:

- Test on 500 recent matches
- Player swap scenarios (20+ tests)
- Error analysis by category
- Model comparison with current baseline

### To-dos

- [ ] Build enhanced dataset with individual player features, recent form, venue/opponent-specific stats, and improved match context
- [ ] Clean dataset by removing outliers, handling missing values, checking for data leakage, and creating stratified train/test split
- [ ] Train XGBoost and Random Forest with hyperparameter tuning, feature selection, and cross-validation
- [ ] Test player swap scenarios to validate that star players show 20-40 run impact
- [ ] Test on 500 reserved matches, perform error analysis, and generate performance visualizations
- [ ] Compare all models, select best performer, and package final model with documentation