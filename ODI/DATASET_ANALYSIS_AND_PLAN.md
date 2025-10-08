# 📊 ODI Dataset Analysis & Extraction Plan

## 🔍 **AVAILABLE DATA SOURCES**

### **1. ODI Ball-by-Ball Data** (`raw_data/odis_ballbyBall/`)
- **Size**: 5,761 JSON files
- **Source**: Cricsheet (One-day internationals, One-day matches, One-Day Cup)
- **Coverage**: ~2,400+ ODI matches

#### **What We Can Extract:**
```json
{
  "info": {
    "city": "Brisbane",
    "dates": ["2017-01-13"],
    "event": {"match_number": 1, "name": "Pakistan in Australia ODI Series"},
    "gender": "male",
    "match_type": "ODI",
    "outcome": {"by": {"runs": 92}, "winner": "Australia"},
    "overs": 50,
    "player_of_match": ["MS Wade"],
    "players": {
      "Australia": ["DA Warner", "TM Head", ...11 players],
      "Pakistan": ["Azhar Ali", "Sharjeel Khan", ...11 players]
    },
    "registry": {
      "people": {
        "Babar Azam": "8a75e999",  // Unique player IDs
        ...
      }
    },
    "toss": {"decision": "bat", "winner": "Australia"},
    "venue": "The Gabba"
  },
  "innings": [
    {
      "team": "Pakistan",
      "overs": [
        {
          "over": 0,
          "deliveries": [
            {
              "batter": "Azhar Ali",
              "bowler": "MA Starc",
              "runs": {"batter": 4, "extras": 0, "total": 4}
            }
          ]
        }
      ]
    }
  ]
}
```

#### **Key Information Available:**
✅ **Match Context**:
- Match ID (filename)
- Date, city, venue
- Tournament/event name
- Match number
- Gender
- Toss (winner, decision)
- Match outcome (winner, margin)
- Player of match

✅ **Teams & Players**:
- Both teams' names
- 11 players per team (exact playing XI)
- Unique player IDs (registry)

✅ **Ball-by-Ball Details**:
- Every delivery
- Batsman, bowler, runs scored
- Wickets, extras
- Can calculate: innings totals, partnerships, strike rates, economy

---

### **2. ODI Player Performance Data** (`raw_data/odi_data/detailed_player_data.csv`)
- **Size**: 52,033 player performance records
- **Source**: GitHub project (cleaned from Cricsheet)

#### **What We Have:**
```csv
match_id,player,team,runs,balls_faced,fours,sixes,wickets,overs_bowled,balls_bowled,runs_conceded,catches,run_outs,maiden,stumps,match_outcome,opposition_team,strike_rate,economy,fantasy_points,venue
```

#### **Key Information Available:**
✅ **Per-Match Player Performance**:
- Individual batting: runs, balls faced, strike rate, boundaries
- Individual bowling: wickets, overs, economy, maidens
- Individual fielding: catches, run_outs, stumps
- Match context: team, opposition, venue, outcome

✅ **Can Calculate Per Player**:
- Career batting average
- Career strike rate
- Career bowling average
- Career economy rate
- Consistency scores
- Role classification (batsman/bowler/all-rounder)

---

## 📋 **T20 DATASET LESSONS LEARNED**

### **What T20 Final Dataset Had** (34 features):
```
Features Used in Training:
1. team_balance_x              - Team composition balance
2. h2h_avg_runs               - Head-to-head average runs
3. pitch_bounce               - Pitch characteristics
4. team_form_avg_runs         - Recent form
5. venue_avg_runs             - Venue scoring patterns
6. team_batting_avg_x         - Team batting strength
7. opposition_bowling_avg      - Opposition bowling quality
8. team_recent_avg            - Recent performance
9. opposition_recent_avg       - Opposition recent form
10. venue_high_score          - Venue high score
11. opposition_bowling_std     - Opposition bowling variance
12. h2h_matches               - Head-to-head history count
13. event_name                - Tournament type
14. h2h_win_rate              - Head-to-head win percentage
15. team_depth                - Team depth score
16. role_variety              - Role diversity in team
17. team_form_win_rate        - Recent win rate
18. venue_low_score           - Venue low score
19. team_batting_std          - Team batting variance
20. h2h_last_meeting          - Last meeting result
21. venue_matches             - Matches played at venue
22. venue_runs_std            - Venue scoring variance
23. pitch_swing               - Pitch swing characteristics
24. season_month              - Month of season
25. match_number              - Match number in series
26. date                      - Match date
27. humidity                  - Weather (humidity)
28. season                    - Season identifier
29. season_year               - Year
30. team_chemistry            - Team chemistry score
31. toss_decision_bat         - Toss decision (bat)
32. toss_decision_field       - Toss decision (field)
33. gender_female             - Gender (female)
34. gender_male               - Gender (male)

Target: total_runs
```

### **🚨 CRITICAL ISSUES IN T20 DATASET:**

#### **❌ What Was MISSING (Why Player Impact Didn't Work):**
1. **NO Individual Player Statistics**
   - No batting averages for players
   - No strike rates for individual players
   - No bowling averages for players
   - No economy rates for individual players

2. **NO Real Player Data Integration**
   - `team_batting_avg_x` was calculated from team name hash, NOT actual players
   - `opposition_bowling_avg` was from hash, NOT real player performance
   - Player IDs were stored but never used for feature generation

3. **NO Player-Level Features**
   - No star player count
   - No elite batsman identification
   - No power hitter detection
   - No all-rounder count

4. **Hash-Based Pseudo-Features**
   - Many features calculated from `hash(team_name)` or `hash(venue_name)`
   - Not based on actual historical data
   - Completely ignored WHO was playing

#### **✅ What WAS Good (We Should Keep):**
1. **Venue Context**
   - Venue average runs (from historical data)
   - Venue high/low scores
   - Venue scoring variance

2. **Head-to-Head History**
   - H2H average runs
   - H2H win rate
   - H2H matches played

3. **Match Context**
   - Tournament/event name
   - Toss decision
   - Gender
   - Season, month, year

4. **Team Form**
   - Recent average runs
   - Recent win rate

---

## 🎯 **ODI DATASET REQUIREMENTS**

### **MUST HAVE (Critical for Player Impact):**

#### **1. Individual Player Statistics (Per Player)**
```python
player_stats = {
    'Virat Kohli': {
        'batting_avg': 58.8,
        'strike_rate': 93.2,
        'total_runs': 12898,
        'total_innings': 260,
        'role': 'batsman',
        'skill_level': 'Elite',
        'star_rating': 9.8
    }
}
```

#### **2. Team-Level Aggregations (From Actual Players)**
```python
match_features = {
    # Aggregate from 11 actual players
    'team_batting_avg': mean([player1.batting_avg, ..., player11.batting_avg]),
    'team_strike_rate': mean([player1.strike_rate, ..., player11.strike_rate]),
    'team_bowling_avg': mean([player1.bowling_avg, ..., player11.bowling_avg]),
    'team_economy': mean([player1.economy, ..., player11.economy]),
    
    # Player quality indicators
    'elite_batsmen_count': count(players with batting_avg > 45),
    'star_batsmen_count': count(players with batting_avg > 35),
    'power_hitters_count': count(players with strike_rate > 95),
    'elite_bowlers_count': count(players with economy < 4.5),
    'all_rounder_count': count(all-rounders),
    
    # Team composition
    'team_balance_score': batting_strength / bowling_strength,
    'team_depth': count(players with batting_avg > 25),
}
```

#### **3. Opposition Features (From Their Actual Players)**
```python
opposition_features = {
    'opposition_batting_avg': ...,
    'opposition_bowling_avg': ...,
    'opposition_star_count': ...,
}
```

#### **4. Relative Strength Features**
```python
relative_features = {
    'batting_vs_bowling': team_batting_avg - opposition_bowling_avg,
    'bowling_vs_batting': team_bowling_avg - opposition_batting_avg,
    'star_advantage': team_star_count - opposition_star_count,
}
```

### **SHOULD HAVE (Important Context):**
- Venue statistics (avg runs, high/low scores) - from ball-by-ball data
- Head-to-head history - from ball-by-ball data
- Team recent form - from ball-by-ball data
- Tournament context - from ball-by-ball data
- Toss decision
- Season, date, year
- Match importance

### **NICE TO HAVE (Additional Context):**
- Weather conditions (if available)
- Pitch characteristics (if can be inferred)
- Home advantage
- Player current form (recent 5 matches)

---

## 📦 **EXTRACTION PLAN**

### **Phase 1: Build Player Performance Database**
**Input**: `raw_data/odi_data/detailed_player_data.csv`

**Process**:
1. Aggregate 52,033 records by player
2. Calculate career statistics:
   - Batting average = total_runs / total_innings
   - Strike rate = mean(strike_rate)
   - Bowling average = total_runs_conceded / total_wickets
   - Economy rate = mean(economy)
   - Total matches, runs, wickets

3. Classify players:
   - Role: Batsman/Bowler/All-rounder/Wicketkeeper
   - Skill level: Elite/Star/Good/Average
   - Specialties: Power hitter, anchor, death bowler, etc.

4. Create star ratings (1-10)

**Output**: 
- `ODI/data/player_lookup.json` - Complete player database
- `ODI/processed_data/player_statistics.csv` - Stats per player

---

### **Phase 2: Extract Match-Level Data from Ball-by-Ball**
**Input**: `raw_data/odis_ballbyBall/*.json` (5,761 files)

**Process**:
For each match JSON file:

1. **Extract Match Context**:
   ```python
   match_context = {
       'match_id': filename,
       'date': info['dates'][0],
       'venue': info['venue'],
       'city': info['city'],
       'event_name': info['event']['name'],
       'gender': info['gender'],
       'toss_winner': info['toss']['winner'],
       'toss_decision': info['toss']['decision'],
       'match_winner': info['outcome']['winner'],
       'player_of_match': info['player_of_match'][0]
   }
   ```

2. **Extract Team Compositions**:
   ```python
   team_a_players = info['players']['Team A']  # List of 11 players
   team_b_players = info['players']['Team B']  # List of 11 players
   ```

3. **Calculate Team Scores** (from innings data):
   ```python
   team_a_score = sum(all runs in team_a innings)
   team_b_score = sum(all runs in team_b innings)
   ```

4. **Link Players to Stats**:
   ```python
   for player in team_a_players:
       player_stats = player_database[player]  # Look up from Phase 1
   ```

**Output**:
- `ODI/processed_data/match_data.csv` - Basic match info
- `ODI/processed_data/team_compositions.csv` - Players per match

---

### **Phase 3: Build Training Dataset with Player Features**
**Input**: 
- Match data from Phase 2
- Player database from Phase 1

**Process**:
For each match, create TWO rows (one per team):

```python
Row for Team A:
{
    # Match context
    'match_id': ...,
    'date': ...,
    'venue': ...,
    'event_name': ...,
    'toss_decision': ...,
    
    # Team A players (actual 11 players)
    'team_batting_avg': mean([p1.bat_avg, ..., p11.bat_avg]),
    'team_strike_rate': mean([p1.strike_rate, ..., p11.strike_rate]),
    'team_bowling_avg': mean([p1.bowl_avg, ..., p11.bowl_avg]),
    'team_economy': mean([p1.economy, ..., p11.economy]),
    'elite_batsmen_count': count(players with bat_avg > 45),
    'star_batsmen_count': count(players with bat_avg > 35-45),
    'power_hitters_count': count(players with SR > 95),
    'elite_bowlers_count': count(players with econ < 4.5),
    'star_bowlers_count': count(players with econ < 5.0),
    'all_rounder_count': count(all-rounders),
    'team_balance': batting_strength / bowling_strength,
    'team_depth': count(players with bat_avg > 25),
    
    # Opposition (Team B) features
    'opposition_batting_avg': Team B's batting avg,
    'opposition_bowling_avg': Team B's bowling avg,
    'opposition_star_count': Team B's star count,
    
    # Relative features
    'batting_vs_bowling': team_batting_avg - opposition_bowling_avg,
    'star_advantage': star_count - opposition_star_count,
    
    # Venue features (from historical data)
    'venue_avg_runs': average runs at this venue,
    'venue_high_score': highest score at venue,
    'venue_matches': matches played at venue,
    
    # H2H features (from historical data)
    'h2h_avg_runs': avg runs in Team A vs Team B matches,
    'h2h_win_rate': Team A win rate vs Team B,
    'h2h_matches': total matches between teams,
    
    # Form features (last 5-10 matches)
    'team_recent_avg': Team A recent avg runs,
    'team_recent_win_rate': Team A recent win %,
    
    # Target
    'total_runs': Team A's final score in this match
}
```

**Output**:
- `ODI/data/odi_training_dataset.csv` - Complete training data

---

## 🎯 **FINAL DATASET STRUCTURE**

### **Proposed Features (40-45 total)**:

#### **Player-Based Features (12)**:
1. `team_batting_avg` - Mean batting average of 11 players
2. `team_strike_rate` - Mean strike rate
3. `team_bowling_avg` - Mean bowling average
4. `team_economy` - Mean economy rate
5. `elite_batsmen_count` - Count(batting avg > 45)
6. `star_batsmen_count` - Count(batting avg 35-45)
7. `power_hitters_count` - Count(strike rate > 95)
8. `elite_bowlers_count` - Count(economy < 4.5)
9. `star_bowlers_count` - Count(economy 4.5-5.0)
10. `all_rounder_count` - Count(all-rounders)
11. `team_balance` - Batting strength / Bowling strength
12. `team_depth` - Count(batting avg > 25)

#### **Opposition Features (6)**:
13. `opposition_batting_avg`
14. `opposition_strike_rate`
15. `opposition_bowling_avg`
16. `opposition_economy`
17. `opposition_star_count`
18. `opposition_all_rounder_count`

#### **Relative Features (4)**:
19. `batting_vs_bowling` - Our batting vs their bowling
20. `bowling_vs_batting` - Our bowling vs their batting
21. `star_advantage` - Our stars - their stars
22. `balance_advantage` - Our balance - their balance

#### **Venue Features (6)**:
23. `venue_avg_runs` - Historical average at venue
24. `venue_high_score` - Highest score at venue
25. `venue_low_score` - Lowest score at venue
26. `venue_runs_std` - Scoring variance at venue
27. `venue_matches` - Total matches at venue
28. `is_home_venue` - Team plays at home venue

#### **Head-to-Head Features (4)**:
29. `h2h_avg_runs` - Average runs in matchup
30. `h2h_win_rate` - Win rate in matchup
31. `h2h_matches` - Total matches between teams
32. `h2h_last_result` - Result of last meeting

#### **Form Features (4)**:
33. `team_recent_avg` - Recent average (last 5-10 matches)
34. `team_recent_win_rate` - Recent win rate
35. `opposition_recent_avg` - Opposition recent avg
36. `opposition_recent_win_rate` - Opposition recent win rate

#### **Context Features (6)**:
37. `toss_decision` - Bat=1, Field=0
38. `event_type` - World Cup, Bilateral, etc. (categorical)
39. `season_year` - Year
40. `season_month` - Month
41. `gender` - Male=1, Female=0
42. `match_number` - Match number in series

#### **Target Variable**:
43. `total_runs` - Team's final score

---

## ✅ **WHAT THIS FIXES FROM T20**

### **T20 Problems → ODI Solutions**:

| T20 Problem | ODI Solution |
|-------------|-------------|
| ❌ `team_batting_avg` from hash | ✅ Calculate from actual 11 players' career stats |
| ❌ No star player recognition | ✅ `elite_batsmen_count`, `star_batsmen_count` |
| ❌ No player quality indicators | ✅ Multiple skill-level features |
| ❌ No all-rounder tracking | ✅ `all_rounder_count` feature |
| ❌ No power hitter detection | ✅ `power_hitters_count` feature |
| ❌ No bowling quality metrics | ✅ `elite_bowlers_count`, `team_economy` |
| ❌ No relative strength | ✅ `batting_vs_bowling`, `star_advantage` |
| ❌ Player IDs unused | ✅ Link every player to their career stats |

---

## 🎯 **SUCCESS CRITERIA**

### **Dataset is Ready When:**
1. ✅ Every match has 11 actual players with real stats
2. ✅ `team_batting_avg` is calculated from actual player averages
3. ✅ Can identify Virat Kohli (avg 58) vs average player (avg 28)
4. ✅ Team with Kohli has higher `team_batting_avg` than team without
5. ✅ Star player counts accurately reflect team quality
6. ✅ Swapping one player changes team-level features measurably

### **Validation Test:**
```python
# Team with Virat Kohli
team_with_kohli = ['V Kohli', 'Player2', ..., 'Player11']
features_with_kohli = extract_features(team_with_kohli)

# Team with average player
team_with_average = ['Average Player', 'Player2', ..., 'Player11']
features_with_average = extract_features(team_with_average)

# Should see clear difference
assert features_with_kohli['team_batting_avg'] > features_with_average['team_batting_avg']
assert features_with_kohli['star_batsmen_count'] > features_with_average['star_batsmen_count']
```

---

## 📝 **NEXT STEPS**

1. **Validate Extraction Approach** - Confirm this plan makes sense
2. **Write Phase 1 Script** - Build player database from 52K records
3. **Write Phase 2 Script** - Extract match data from 5,761 JSON files
4. **Write Phase 3 Script** - Combine into training dataset
5. **Validate Dataset** - Test player impact is measurable
6. **Train Models** - Linear, RF, XGBoost

---

**This dataset will have REAL player impact because every feature is calculated from ACTUAL player statistics, not hash-based pseudo-values!** 🎯
