# 🔗 How the Two ODI Datasets Combine

## 📊 **THE TWO DATASETS**

### **Dataset 1: `detailed_player_data.csv`**
```csv
match_id,player,team,runs,balls_faced,fours,sixes,wickets,overs_bowled,balls_bowled,runs_conceded,catches,run_outs,maiden,stumps,match_outcome,opposition_team,strike_rate,economy,fantasy_points,venue

Example row:
792295,RG Sharma,India,264,176,33,9,0,0,0,0,0,0,0,0,win,Sri Lanka,150.0,0.0,540,Eden Gardens
```

**What it tells us:**
- ✅ **Individual player performance** in each match
- ✅ **Per-match stats**: How many runs Rohit Sharma scored in match 792295
- ✅ **Easy to aggregate**: Can calculate career averages

**What it DOESN'T tell us:**
- ❌ **Match context**: What was the toss decision? Venue details?
- ❌ **Playing XI**: Who else played in this match?
- ❌ **Team composition**: Was this his full team or just him?

---

### **Dataset 2: `odis_ballbyBall/*.json`**
```json
{
  "info": {
    "match_id": "1000887",
    "venue": "Brisbane Cricket Ground",
    "toss": {"winner": "Australia", "decision": "bat"},
    "players": {
      "Australia": ["DA Warner", "TM Head", "SPD Smith", ...11 players],
      "Pakistan": ["Azhar Ali", "Babar Azam", "Mohammad Rizwan", ...11 players]
    },
    "outcome": {"winner": "Australia", "by": {"runs": 92}}
  },
  "innings": [...]
}
```

**What it tells us:**
- ✅ **Match context**: Venue, toss, tournament, date
- ✅ **Exact playing XI**: All 11 players for each team
- ✅ **Team compositions**: Who played together
- ✅ **Match outcome**: Winner, margin

**What it DOESN'T tell us:**
- ❌ **Player career stats**: What's DA Warner's career batting average?
- ❌ **Pre-calculated stats**: Already has strike rates, economy for this match but not career

---

## 🔗 **HOW THEY COMBINE**

### **Step 1: Build Player Database (from detailed_player_data.csv)**

**Process 52,031 performance records:**
```python
# Read detailed_player_data.csv
df = pd.read_csv('raw_data/odi_data/detailed_player_data.csv')

# Aggregate by player
for player in df['player'].unique():
    player_matches = df[df['player'] == player]
    
    # Calculate career statistics
    career_stats = {
        'player_name': player,
        'total_matches': len(player_matches),
        'career_batting_avg': player_matches['runs'].sum() / len(player_matches[player_matches['balls_faced'] > 0]),
        'career_strike_rate': player_matches['strike_rate'].mean(),
        'career_bowling_avg': player_matches['runs_conceded'].sum() / player_matches['wickets'].sum(),
        'career_economy': player_matches['economy'].mean(),
        'total_runs': player_matches['runs'].sum(),
        'total_wickets': player_matches['wickets'].sum()
    }
    
    # Store in database
    player_database['RG Sharma'] = career_stats
```

**Output:**
```json
{
  "RG Sharma": {
    "total_matches": 260,
    "career_batting_avg": 49.8,
    "career_strike_rate": 89.2,
    "total_runs": 10,866,
    "role": "batsman"
  },
  "Babar Azam": {
    "total_matches": 102,
    "career_batting_avg": 56.8,
    "career_strike_rate": 88.5,
    "total_runs": 5,096,
    "role": "batsman"
  }
}
```

**This gives us: WHO each player is and their CAREER performance**

---

### **Step 2: Process Ball-by-Ball Matches (from odis_ballbyBall)**

**For each match JSON:**
```python
# Read match file
with open('raw_data/odis_ballbyBall/1000887.json') as f:
    match = json.load(f)

# Extract match context
match_context = {
    'match_id': '1000887',
    'venue': match['info']['venue'],
    'toss_winner': match['info']['toss']['winner'],
    'toss_decision': match['info']['toss']['decision'],
    'date': match['info']['dates'][0]
}

# Extract playing XI
team_a_players = match['info']['players']['Australia']
# ['DA Warner', 'TM Head', 'SPD Smith', 'CA Lynn', 'MR Marsh', ...]

team_b_players = match['info']['players']['Pakistan']
# ['Azhar Ali', 'Sharjeel Khan', 'Babar Azam', 'Umar Akmal', ...]

# Calculate team scores (from innings data)
team_a_score = calculate_innings_total(match['innings'][0])
team_b_score = calculate_innings_total(match['innings'][1])
```

**This gives us: WHICH match, WHAT context, WHO played, WHAT scores**

---

### **Step 3: COMBINE - Link Players to Their Career Stats**

**The KEY step where datasets merge:**
```python
# For match 1000887
match_id = '1000887'
team_a_players = ['DA Warner', 'TM Head', 'SPD Smith', ...11 players]

# NOW: Look up each player's CAREER stats from player_database
team_a_features = {
    'team_batting_avg': 0,
    'team_strike_rate': 0,
    'star_players': 0
}

for player in team_a_players:
    # THIS IS THE LINK!
    if player in player_database:  # From detailed_player_data.csv
        player_stats = player_database[player]
        
        # Add to team aggregations
        team_a_features['team_batting_avg'] += player_stats['career_batting_avg']
        team_a_features['team_strike_rate'] += player_stats['career_strike_rate']
        
        # Check if star player
        if player_stats['career_batting_avg'] > 35:
            team_a_features['star_players'] += 1

# Calculate averages
team_a_features['team_batting_avg'] /= len(team_a_players)
team_a_features['team_strike_rate'] /= len(team_a_players)
```

**Result:**
```python
{
    'match_id': '1000887',
    'venue': 'Brisbane Cricket Ground',
    'toss_decision': 'bat',
    
    # Team features (from player database)
    'team_batting_avg': 42.3,  # Average of 11 players' career avgs
    'team_strike_rate': 87.5,
    'star_players': 5,  # Warner, Smith, etc.
    
    # Target
    'total_runs': 369  # Australia's actual score
}
```

---

## 🎯 **CONCRETE EXAMPLE**

### **Match 1000887: Australia vs Pakistan**

**From odis_ballbyBall (Match Context):**
```json
{
  "venue": "Brisbane Cricket Ground",
  "date": "2017-01-13",
  "toss": {"winner": "Australia", "decision": "bat"},
  "players": {
    "Australia": ["DA Warner", "TM Head", "SPD Smith", "CA Lynn", ...],
    "Pakistan": ["Azhar Ali", "Sharjeel Khan", "Babar Azam", ...]
  }
}
```

**From player_database (Built from detailed_player_data.csv):**
```json
{
  "DA Warner": {"career_batting_avg": 45.2, "career_strike_rate": 95.1},
  "SPD Smith": {"career_batting_avg": 42.8, "career_strike_rate": 87.3},
  "Babar Azam": {"career_batting_avg": 56.8, "career_strike_rate": 88.5},
  ...
}
```

**COMBINED Final Training Row:**
```python
{
    # Match context (from ballbyball)
    'match_id': '1000887',
    'venue': 'Brisbane Cricket Ground',
    'toss_decision': 'bat',
    'date': '2017-01-13',
    
    # Team A (Australia) features (calculated from player_database)
    'team_batting_avg': 41.5,  # mean([45.2, 42.8, ...]) for 11 players
    'team_strike_rate': 89.3,
    'star_batsmen_count': 4,  # Warner, Smith, Maxwell, Marsh
    'elite_batsmen_count': 1,  # Warner (>45 avg)
    
    # Team B (Pakistan) features (opposition)
    'opposition_batting_avg': 38.2,
    'opposition_star_count': 3,  # Babar, Azhar, etc.
    
    # Relative features
    'batting_advantage': 41.5 - 38.2 = 3.3,  # Australia stronger
    
    # Target (from ballbyball innings calculation)
    'total_runs': 369  # Australia's actual score
}
```

---

## ✅ **THE KEY INSIGHT**

### **They DON'T Duplicate - They COMPLEMENT:**

| Information | detailed_player_data.csv | odis_ballbyBall |
|-------------|-------------------------|-----------------|
| **Player career stats** | ✅ YES (aggregate from 52K records) | ❌ NO |
| **Match context** | ❌ NO | ✅ YES |
| **Playing XI** | ❌ NO | ✅ YES |
| **Individual match performance** | ✅ YES (per match) | ✅ YES (can calculate) |
| **Team composition** | ❌ NO | ✅ YES |

### **The Combination:**
```
odis_ballbyBall tells us: "In match 1000887, these 11 players played"
                          ↓
player_database tells us: "Here's what each of those 11 players' career stats are"
                          ↓
Combined result:         "Team batting avg = 41.5, star players = 4, predicted score = ?"
```

---

## 🎯 **WHY THIS WORKS**

### **Without Combining (What T20 did wrong):**
```python
# T20 approach (BROKEN)
team_batting_avg = hash('Australia') % 100 / 10  # Random number!
# Result: Australia with Warner = Australia with random players
```

### **With Combining (What we're doing):**
```python
# ODI approach (CORRECT)
team_batting_avg = mean([
    player_database['DA Warner']['career_batting_avg'],    # 45.2
    player_database['SPD Smith']['career_batting_avg'],    # 42.8
    player_database['CA Lynn']['career_batting_avg'],      # 37.1
    # ...all 11 players
])
# Result: Australia with Warner (45 avg) > Australia with random player (25 avg)
```

---

## ✅ **SUMMARY**

**detailed_player_data.csv** = Player profiles (WHO they are, WHAT their career stats are)

**odis_ballbyBall** = Match details (WHICH match, WHO played, WHAT happened)

**Combined** = Match with player-aware features (THESE 11 players with THESE career stats played in THIS match)

**This is how we get player-level impact predictions!** 🎯
