# Processed Data Datasets

This folder contains the comprehensive datasets created for the Cricket Score Prediction Machine Learning project. Each dataset focuses on a specific aspect of cricket match analysis and can be used individually or combined for enhanced model training.

## 📊 Dataset Overview

### 1. **Player Impact Dataset** (`player_impact_dataset.csv`)
**Purpose**: Captures individual player performance and impact on match outcomes.

**Key Features (32 total)**:
- **Match Performance**: `match_runs`, `match_balls`, `match_wickets`, `match_4s`, `match_6s`, `match_strike_rate`
- **Career Statistics**: `career_batting_avg`, `career_batting_sr`, `career_bowling_avg`, `career_bowling_econ`
- **Player Impact**: `player_impact_score`, `performance_ratio`, `strike_rate_ratio`
- **Player Classification**: `player_role`, `is_batsman`, `is_bowler`, `is_all_rounder`
- **Experience & Consistency**: `experience_level`, `high_performer`, `consistent_performer`

**Records**: 198,010 player performances across 7,223 matches

**Use Case**: Train models to understand individual player contributions and predict how specific players will perform in different match conditions.

---

### 2. **Venue Conditions Dataset** (`venue_conditions_dataset.csv`)
**Purpose**: Analyzes venue-specific conditions and their impact on scoring patterns.

**Key Features (32 total)**:
- **Weather Conditions**: `temperature`, `humidity`, `wind_speed`, `precipitation`, `dew_factor`
- **Pitch Characteristics**: `pitch_type`, `pitch_bounce`, `pitch_pace`, `pitch_turn`, `pitch_swing`
- **Venue Statistics**: `venue_avg_runs`, `venue_matches`, `venue_high_score`, `venue_low_score`
- **Match Timing**: `is_day_match`, `is_night_match`, `season`, `is_rain_affected`, `is_dew_affected`
- **Derived Metrics**: `weather_impact`, `pitch_difficulty`, `venue_scoring_potential`, `match_conditions_score`

**Records**: 7,223 venue conditions (one per match)

**Use Case**: Predict how venue conditions, weather, and pitch characteristics affect scoring patterns.

---

### 3. **Team Composition Dataset** (`team_composition_dataset.csv`)
**Purpose**: Analyzes team balance, chemistry, and strategic advantages.

**Key Features (32 total)**:
- **Team Structure**: `team_size`, `batting_strength`, `bowling_strength`, `all_rounders`
- **Team Balance**: `batting_ratio`, `bowling_ratio`, `all_rounder_ratio`, `team_balance`, `team_depth`
- **Strategic Elements**: `toss_winner`, `toss_decision`, `toss_impact`, `is_batting_first`, `is_chasing`
- **Team Dynamics**: `role_variety`, `team_chemistry`, `strategic_advantage`
- **Performance Metrics**: `total_runs`, `run_rate`, `powerplay_ratio`, `death_overs_ratio`

**Records**: 14,446 team compositions (2 per match - one for each team)

**Use Case**: Understand how team composition, balance, and strategic decisions affect match outcomes.

---

### 4. **Final Comprehensive Dataset** (`final_comprehensive_dataset.csv`)
**Purpose**: The complete dataset combining all aspects for advanced model training.

**Key Features (95 total)**:
- **Base Features**: All original features from `simple_enhanced_train.csv`
- **Player Impact**: Aggregated team player performance metrics
- **Venue Conditions**: Weather, pitch, and venue-specific data
- **Team Composition**: Balance, chemistry, and strategic advantages
- **Derived Features**: Advanced scoring predictions and match context

**Records**: 13,514 matches with comprehensive feature set

**Use Case**: Train sophisticated models (XGBoost, Random Forest) with complete understanding of all factors affecting cricket scoring.

---

## 🔗 Dataset Relationships

```
Raw Ball-by-Ball Data (7,223 matches)
├── Player Impact Dataset (198,010 player performances)
├── Venue Conditions Dataset (7,223 venue conditions)
├── Team Composition Dataset (14,446 team compositions)
└── Final Comprehensive Dataset (13,514 matches with 95 features)
```

## 📈 Feature Categories

### **Player-Centric Features**
- Individual player performance metrics
- Career statistics and experience
- Player role classification
- Performance consistency indicators

### **Venue-Centric Features**
- Weather and environmental conditions
- Pitch characteristics and behavior
- Venue-specific historical data
- Match timing and seasonality

### **Team-Centric Features**
- Team balance and composition
- Strategic advantages and decisions
- Team chemistry and dynamics
- Opposition analysis

### **Match-Centric Features**
- Match context and importance
- Historical head-to-head data
- Form and momentum indicators
- Pressure situation analysis

## 🎯 Model Training Recommendations

### **For Individual Player Analysis**
Use: `player_impact_dataset.csv`
- Focus on player-specific predictions
- Understand individual contributions
- Analyze player performance patterns

### **For Venue-Specific Predictions**
Use: `venue_conditions_dataset.csv`
- Weather impact analysis
- Pitch condition effects
- Venue scoring patterns

### **For Team Strategy Analysis**
Use: `team_composition_dataset.csv`
- Team balance optimization
- Strategic decision analysis
- Team chemistry assessment

### **For Complete Model Training**
Use: `final_comprehensive_dataset.csv`
- Full-feature model training
- Comprehensive prediction capabilities
- Advanced machine learning applications

## 🚀 Expected Model Performance

The comprehensive dataset is designed to achieve:
- **Higher Accuracy**: 95 features vs. 55 in original dataset
- **Better Player Recognition**: Individual player impact understanding
- **Venue Intelligence**: Weather and pitch condition awareness
- **Strategic Understanding**: Team composition and decision analysis
- **Context Awareness**: Match importance and pressure situations

## 📋 Data Quality

- **Missing Values**: 0 (all datasets are complete)
- **Data Types**: Mixed (numeric, categorical, boolean)
- **Encoding**: Ready for machine learning models
- **Validation**: All datasets have been validated and cleaned

## 🔧 Usage Examples

### **Load Individual Dataset**
```python
import pandas as pd

# Load player impact data
player_data = pd.read_csv('processed_data/player_impact_dataset.csv')

# Load venue conditions
venue_data = pd.read_csv('processed_data/venue_conditions_dataset.csv')

# Load team composition
team_data = pd.read_csv('processed_data/team_composition_dataset.csv')

# Load comprehensive dataset
comprehensive_data = pd.read_csv('processed_data/final_comprehensive_dataset.csv')
```

### **Feature Selection for Different Models**
```python
# For player-focused models
player_features = ['player_impact_score', 'career_batting_avg', 'match_strike_rate']

# For venue-focused models
venue_features = ['temperature', 'humidity', 'pitch_type', 'venue_avg_runs']

# For team-focused models
team_features = ['team_balance', 'team_chemistry', 'strategic_advantage']

# For comprehensive models
all_features = comprehensive_data.columns.tolist()
```

## 📊 Dataset Statistics

| Dataset | Records | Features | Primary Use |
|---------|---------|----------|-------------|
| Player Impact | 198,010 | 32 | Individual player analysis |
| Venue Conditions | 7,223 | 32 | Venue-specific predictions |
| Team Composition | 14,446 | 32 | Team strategy analysis |
| Final Comprehensive | 13,514 | 95 | Complete model training |

## 🎯 Target Variable

All datasets are designed to predict **`total_runs`** - the total runs scored by a team in a T20 match.

**Target Statistics**:
- Mean: 134.35 runs
- Standard Deviation: 44.28 runs
- Range: 1-297 runs
- Distribution: Normal with slight right skew

## 🔄 Data Flow

1. **Raw Data**: Ball-by-ball JSON files (7,223 matches)
2. **Individual Datasets**: Specialized feature extraction
3. **Combined Dataset**: Comprehensive feature integration
4. **Model Training**: Ready for advanced ML algorithms
5. **Prediction**: Real-time cricket score prediction

---

*This comprehensive dataset collection enables sophisticated cricket score prediction models that understand the nuances of individual players, venue conditions, team dynamics, and match context.*
