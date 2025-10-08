# 🎯 ODI Dataset Quality-First Strategy

## 📊 **CORE PRINCIPLE**

**Quality > Quantity**: Better to train on 3,000 high-quality matches with reliable player data than 5,761 mixed-quality matches with unknown players.

---

## 🎯 **QUALITY CRITERIA**

### **Player Quality Criteria:**

#### **Minimum Performance Threshold:**
```
A player is "worthy" for training if:
1. Appeared in >= 10 matches (enough data for reliable average)
2. Has >= 5 batting innings OR >= 5 bowling performances
3. Total career runs >= 100 OR total career wickets >= 5
4. Not a "one-hit wonder" (consistency check)
```

**Why:**
- 1-match players have no reliable statistics
- Need enough data to calculate meaningful averages
- Distinguish between regular players and occasional appearances

#### **Player Classification:**

**Tier 1 - Elite Data (Use Always)**
- Matches: >= 50
- Runs: >= 1000 OR Wickets: >= 50
- Career span: >= 2 years
- **Impact**: Star players like Kohli, Babar, Bumrah

**Tier 2 - Good Data (Use for Training)**
- Matches: 10-49
- Runs: 100-999 OR Wickets: 5-49
- Career span: >= 1 year
- **Impact**: Regular international/domestic players

**Tier 3 - Weak Data (Consider Excluding)**
- Matches: 5-9
- Limited stats
- **Impact**: Occasional players, borderline useful

**Tier 4 - Insufficient Data (EXCLUDE)**
- Matches: < 5
- **Impact**: Too unreliable, noise in dataset

---

### **Match Quality Criteria:**

#### **A match is "high-quality" for training if:**
```
1. >= 16 of 22 players (73%) are Tier 1 or Tier 2
2. Both teams have >= 7 of 11 players with career stats
3. Match completed (not rain-affected/abandoned)
4. Both innings had reasonable scores (not 30 all-out)
```

**Why:**
- Need player stats to calculate team features
- If too many unknown players, team aggregations unreliable
- Completed matches give real target scores
- Extreme scores (30 all-out) are outliers, not representative

---

## 📋 **DATA CLEANING PROCESS**

### **Phase 1: Player Database Quality Control**

**Input**: 52,031 records from odi_data

**Step 1: Player Filtering**
```python
# Remove players with insufficient data
quality_players = []

for player in all_players:
    player_records = df[df['player'] == player]
    
    # Count matches
    total_matches = len(player_records)
    
    # Count batting innings (where they batted)
    batting_innings = len(player_records[player_records['balls_faced'] > 0])
    
    # Count bowling performances (where they bowled)
    bowling_performances = len(player_records[player_records['overs_bowled'] > 0])
    
    # Total contribution
    total_runs = player_records['runs'].sum()
    total_wickets = player_records['wickets'].sum()
    
    # Quality check
    if total_matches >= 10:  # Minimum 10 matches
        if (batting_innings >= 5 and total_runs >= 100) OR \
           (bowling_performances >= 5 and total_wickets >= 5):
            quality_players.append(player)
```

**Expected Filtering:**
- **Original**: 1,872 players
- **After filtering**: ~800-1,000 quality players
- **Removed**: ~900 players (one-hit wonders, rare appearances)

**Output**: `quality_player_database.json`

---

### **Phase 2: Calculate Reliable Career Statistics**

**For each quality player:**

```python
career_stats = {
    'batting': {
        'matches': total_matches_batted,
        'innings': batting_innings,
        'total_runs': total_runs,
        'average': total_runs / dismissals,  # Real average
        'strike_rate': mean(strike_rates),
        'consistency': std(runs_per_innings),
        'high_score': max(runs),
        'boundaries': {
            'fours': total_fours,
            'sixes': total_sixes
        }
    },
    'bowling': {
        'matches': total_matches_bowled,
        'overs': total_overs,
        'wickets': total_wickets,
        'average': runs_conceded / wickets,
        'economy': runs_conceded / overs,
        'consistency': std(wickets_per_match),
        'best_figures': best_bowling
    },
    'reliability_score': calculate_reliability(matches, consistency)
}
```

**Reliability Score:**
```python
reliability = (
    min(matches / 50, 1.0) * 0.4 +      # More matches = more reliable
    (1 - consistency_variance) * 0.3 +   # Consistent = reliable
    career_span_years / 5 * 0.3          # Longer career = reliable
)
```

**Output**: `reliable_player_statistics.csv`

---

### **Phase 3: Match Quality Scoring**

**For each of 5,761 matches:**

```python
def calculate_match_quality(match_json):
    team_a_players = match['info']['players']['Team A']
    team_b_players = match['info']['players']['Team B']
    
    # Check player coverage
    team_a_known = [p for p in team_a_players if p in quality_player_db]
    team_b_known = [p for p in team_b_players if p in quality_player_db]
    
    team_a_coverage = len(team_a_known) / 11
    team_b_coverage = len(team_b_known) / 11
    
    # Quality score
    match_quality = {
        'team_a_coverage': team_a_coverage,
        'team_b_coverage': team_b_coverage,
        'total_coverage': (team_a_coverage + team_b_coverage) / 2,
        'known_players': len(team_a_known) + len(team_b_known),
        'unknown_players': 22 - (len(team_a_known) + len(team_b_known))
    }
    
    # Quality tier
    if match_quality['total_coverage'] >= 0.80:  # 80%+ coverage
        tier = 'Tier 1 - Excellent'
    elif match_quality['total_coverage'] >= 0.65:  # 65-80% coverage
        tier = 'Tier 2 - Good'
    elif match_quality['total_coverage'] >= 0.50:  # 50-65% coverage
        tier = 'Tier 3 - Fair'
    else:
        tier = 'Tier 4 - Poor'
    
    return match_quality, tier
```

**Expected Distribution:**
- Tier 1 (Excellent): ~1,500-2,000 matches
- Tier 2 (Good): ~800-1,200 matches
- Tier 3 (Fair): ~500-800 matches
- Tier 4 (Poor): ~1,500-2,000 matches (exclude)

**Output**: `match_quality_scores.csv`

---

### **Phase 4: Dataset Assembly**

**Selection Criteria:**
```
Use matches where:
1. Match is Tier 1 or Tier 2 (>= 65% player coverage)
2. Both teams have >= 7 known players
3. Match completed (outcome exists)
4. Reasonable scores (50 <= total_runs <= 450)
```

**Expected Final Dataset:**
- **High-quality matches**: 2,500-3,500 matches
- **Known players**: 800-1,000 players
- **Player coverage**: 70-85% of all players per match
- **Training rows**: 5,000-7,000 (2 per match: Team A, Team B)

**Output**: `high_quality_odi_training_dataset.csv`

---

## 🎯 **WHAT THIS MEANS FOR YOUR PROJECT**

### **✅ BENEFITS:**

**1. Reliable Player Impact**
```
Before (all 1,872 players):
- Virat Kohli (avg 58) + Random 1-match player (avg 12) = Unreliable team avg
- Model learns noise, not cricket

After (800-1,000 quality players):
- Virat Kohli (avg 58) + Regular player (avg 32) = Reliable team avg
- Model learns actual cricket patterns
```

**2. Measurable Star Player Effect**
```
With quality data:
- Elite player (avg 50+): Clear +40 runs impact
- Star player (avg 35-45): Clear +20-30 runs impact
- Good player (avg 25-35): Baseline
- Model can distinguish quality
```

**3. Training Efficiency**
```
3,000 high-quality matches > 5,761 mixed-quality matches
- Less noise in data
- Faster training
- Better generalization
- More interpretable results
```

**4. Realistic Predictions**
```
When user selects Virat Kohli:
- Model uses his 260-match career stats (reliable)
- Not his 1-match performance (unreliable)
- Predictions are cricket-realistic
```

---

## 📊 **EXPECTED NUMBERS**

### **Player Database:**
```
Original (odi_data):        1,872 players, 52,031 records
After quality filter:       ~900 players, ~45,000 records
Tier 1 (Elite):            ~150 players (Kohli, Babar, etc.)
Tier 2 (Regular):          ~750 players (solid internationals)
```

### **Match Dataset:**
```
Original (ballbyball):      5,761 matches
Tier 1+2 (High quality):   ~3,000 matches
Expected coverage:          70-85% players known per match
Training rows:              ~6,000 rows (2 per match)
```

### **Model Training:**
```
Features per row:           40-45 features
Training samples:           ~4,800 (80% of 6,000)
Test samples:              ~1,200 (20% of 6,000)
Expected accuracy:          85-90% within ±20 runs
```

---

## 🚀 **IMPLEMENTATION PLAN**

### **Script 1: `filter_quality_players.py`**
```python
Purpose: Filter odi_data to keep only quality players
Input: raw_data/odi_data/detailed_player_data.csv
Output: ODI/processed_data/quality_player_records.csv
        ODI/data/quality_player_database.json

Criteria:
- >= 10 matches
- >= 5 batting innings OR >= 5 bowling performances
- >= 100 runs OR >= 5 wickets
```

### **Script 2: `calculate_career_statistics.py`**
```python
Purpose: Calculate reliable career stats for quality players
Input: ODI/processed_data/quality_player_records.csv
Output: ODI/processed_data/player_career_statistics.csv
        ODI/data/player_lookup.json

Statistics:
- Batting: avg, strike_rate, consistency, reliability
- Bowling: avg, economy, wickets_per_match, reliability
- Classification: Elite/Star/Good/Average
- Role: Batsman/Bowler/All-rounder/Wicketkeeper
```

### **Script 3: `score_match_quality.py`**
```python
Purpose: Score all 5,761 matches for player coverage
Input: raw_data/odis_ballbyBall/*.json
       ODI/data/player_lookup.json
Output: ODI/processed_data/match_quality_scores.csv

For each match:
- Count known vs unknown players
- Calculate coverage percentage
- Assign quality tier
- Flag for inclusion/exclusion
```

### **Script 4: `build_training_dataset.py`**
```python
Purpose: Build final high-quality training dataset
Input: raw_data/odis_ballbyBall/*.json (Tier 1+2 matches)
       ODI/data/player_lookup.json
Output: ODI/data/odi_training_dataset.csv

For each quality match:
- Extract match context (venue, toss, etc.)
- Get 11 players per team
- Calculate team features from player stats
- Add venue, H2H, form features
- Create 2 rows (Team A, Team B)
```

---

## ✅ **SUCCESS METRICS**

### **Data Quality Metrics:**
- ✅ >= 800 players with >= 10 matches each
- ✅ >= 70% player coverage per match
- ✅ >= 2,500 high-quality matches
- ✅ Player reliability score >= 0.6

### **Model Performance Metrics:**
- ✅ R² >= 0.85
- ✅ RMSE <= 20 runs
- ✅ Accuracy (±20 runs) >= 85%
- ✅ Virat Kohli impact: +40 runs vs average
- ✅ Star player impact: +25 runs vs average

### **System Validation:**
- ✅ Player swap changes prediction by 20-40 runs
- ✅ Star team scores 60-80 runs more than average team
- ✅ Predictions explainable and cricket-realistic

---

## 🎯 **BOTTOM LINE**

### **What You're Building:**
A **quality-driven dataset** where:
- Every player has reliable career statistics (>= 10 matches)
- Every match has good player coverage (>= 70%)
- Model learns from **actual cricket patterns**, not noise
- Star player impact is **measurable and realistic**

### **Trade-offs:**
- ❌ Lose ~2,500 low-quality matches
- ❌ Lose ~900 one-hit wonder players
- ✅ Gain **reliable player impact predictions**
- ✅ Gain **interpretable results**
- ✅ Gain **cricket-realistic predictions**

### **Expected Outcome:**
**3,000 high-quality matches with 800-1,000 reliable players** is **perfect** for your goal of building a player-level impact prediction system.

---

**This is exactly what you need: QUALITY players with QUALITY data for QUALITY predictions!** 🎯
