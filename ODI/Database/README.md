# ODI Cricket Prediction Backend

This directory contains the backend API and database setup for ODI cricket score predictions.

## 📁 Files

- `setup_database.py` - Creates SQLite database with teams, venues, and players
- `run_odi_api.py` - Flask API server for predictions
- `requirements.txt` - Python dependencies
- `cricket_prediction_odi.db` - SQLite database (created after running setup)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Setup Database

```bash
python setup_database.py
```

This will:
- Create SQLite database
- Load 28 teams
- Load 303 venues  
- Load 977 players from player_database.json

### 3. Train Models (Required for predictions)

Before running the API, you need to train models. These should be saved in `../models/` with these filenames:
- `odi_linear_regression.pkl`
- `odi_random_forest.pkl`
- `odi_xgboost.pkl`

Training scripts should be created in `../scripts/` directory.

### 4. Start API Server

```bash
python run_odi_api.py
```

The API will start on `http://localhost:5001`

## 📡 API Endpoints

### GET /api/health
Check if API is running and models are loaded

**Response:**
```json
{
  "status": "running",
  "models_loaded": 3,
  "players_loaded": 977
}
```

### GET /api/teams
Get all active teams

**Response:**
```json
[
  {
    "team_id": 0,
    "team_name": "India",
    "team_type": "National",
    "is_active": 1
  },
  ...
]
```

### GET /api/venues
Get all active venues

**Response:**
```json
[
  {
    "venue_id": 0,
    "venue_name": "Lord's Cricket Ground, London",
    "stadium": "Lord's Cricket Ground",
    "city": "London",
    "is_active": 1
  },
  ...
]
```

### GET /api/players
Get all active players (optionally filter by team)

**Query Parameters:**
- `team` (optional) - Filter by team name (e.g., `?team=India`)

**Response:**
```json
[
  {
    "player_id": 0,
    "player_name": "V Kohli",
    "role": "Batsman",
    "skill_level": "Elite",
    "star_rating": 9.0,
    "teams": "India",
    "batting_avg": 57.32,
    "bowling_avg": 0,
    "total_matches": 254,
    "is_active": 1
  },
  ...
]
```

### POST /api/predict
Make a prediction for an ODI match

**Request Body:**
```json
{
  "team_a_id": 1,
  "team_b_id": 2,
  "venue_id": 10,
  "team_a_players": [
    "V Kohli",
    "RG Sharma",
    "JJ Bumrah",
    "RA Jadeja",
    ...
  ],
  "team_b_players": [
    "JE Root",
    "BA Stokes",
    "JC Buttler",
    ...
  ],
  "match_context": {
    "season_year": 2024,
    "season_month": 10,
    "gender": "male",
    "toss_won": "team_a",
    "toss_decision": "bat"
  },
  "model": "random_forest"
}
```

**Response:**
```json
{
  "success": true,
  "prediction_id": 42,
  "predicted_score": 285,
  "model_used": "random_forest",
  "team_a_stats": {
    "team_batting_avg": 42.5,
    "elite_batsmen": 3,
    "star_players": 5,
    ...
  },
  "team_b_stats": {
    "team_batting_avg": 38.2,
    "elite_batsmen": 2,
    "star_players": 4,
    ...
  },
  "features_used": 54
}
```

### GET /api/predictions/history
Get recent prediction history (last 50 predictions)

**Response:**
```json
[
  {
    "prediction_id": 42,
    "team_a_name": "India",
    "team_b_name": "England",
    "venue_name": "Lord's Cricket Ground, London",
    "predicted_score_a": 285,
    "model_used": "random_forest",
    "created_at": "2024-10-08 14:30:00"
  },
  ...
]
```

## 🎯 How It Works

### Feature Calculation

The API calculates aggregated team features from individual player selections:

1. **User selects 11 players** for each team in the frontend
2. **Backend looks up each player** in `player_database.json`
3. **Calculates team statistics:**
   - Average batting average
   - Average strike rate
   - Count of elite/star batsmen
   - Average bowling economy
   - Count of elite/star bowlers
   - All-rounder count
   - Team balance metrics
   - And more...

4. **Builds feature vector** matching training dataset format (54 features)
5. **Feeds to trained model** for prediction

### Database Schema

**teams**
- `team_id` (PK)
- `team_name`
- `team_type` (National/Special XI/County)
- `is_active`

**venues**
- `venue_id` (PK)
- `venue_name`
- `stadium`
- `city`
- `is_active`

**players**
- `player_id` (PK)
- `player_name`
- `role` (Batsman/Bowler/All-rounder/Wicketkeeper)
- `skill_level` (Elite/Star/Regular/Emerging)
- `star_rating` (0-10)
- `teams` (comma-separated)
- `batting_avg`
- `bowling_avg`
- `total_matches`
- `is_active`

**predictions**
- `prediction_id` (PK)
- `team_a_id` (FK)
- `team_b_id` (FK)
- `venue_id` (FK)
- `team_a_players` (JSON)
- `team_b_players` (JSON)
- `predicted_score_a`
- `model_used`
- `match_date`
- `gender`
- `toss_won`
- `toss_decision`
- `created_at`

## 🔧 Development

### Adding New Models

1. Train your model in a script
2. Save it with joblib:
   ```python
   joblib.dump(model, 'odi_your_model.pkl')
   ```
3. Update `run_odi_api.py` to include your model in `model_files` dict
4. Restart the API

### Testing API

Use curl or Postman:

```bash
# Check health
curl http://localhost:5001/api/health

# Get teams
curl http://localhost:5001/api/teams

# Get players for India
curl http://localhost:5001/api/players?team=India
```

## 📊 Frontend Integration

The frontend (React app in `../../frontend/`) should:

1. Fetch teams from `/api/teams`
2. Fetch venues from `/api/venues`
3. When user selects a team, fetch players from `/api/players?team={team_name}`
4. User selects 11 players for each team
5. User fills in match context (date, toss, etc.)
6. Send POST to `/api/predict` with all data
7. Display predicted score

## ⚠️ Important Notes

- **Models must be trained first** - API will start without models but predictions will fail
- **Player names must match exactly** - Use names from `player_database.json`
- **Port 5001** - Make sure this port is available (T20 API uses 5000)
- **CORS enabled** - Frontend can call from different origin

## 📝 Next Steps

1. ✅ Database setup complete
2. ✅ API endpoints implemented
3. ⏳ Train ML models (see `../scripts/`)
4. ⏳ Update frontend to use ODI API
5. ⏳ Test end-to-end predictions

## 🐛 Troubleshooting

**"No models loaded" error:**
- Train models first and save them in `../models/` directory

**"Player not found" error:**
- Check player names match exactly with `player_database.json`

**Database errors:**
- Re-run `setup_database.py` to recreate database

**Port already in use:**
- Change port in `run_odi_api.py` line: `app.run(debug=True, host='0.0.0.0', port=5001)`

