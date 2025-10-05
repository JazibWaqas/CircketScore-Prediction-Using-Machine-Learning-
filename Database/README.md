# Cricket Prediction Database

This folder contains the database and API for the cricket score prediction system.

## Files

- **`cricket_prediction.db`** - SQLite database with T20 players, teams, and venues
- **`app.py`** - Flask API server for the cricket prediction system
- **`setup_database.py`** - Script to create the T20-only database
- **`requirements.txt`** - Python dependencies

## Database Contents

- **8,321 T20 Players** - Only players who actually played T20 cricket
- **172 Teams** - International and franchise teams
- **503 Venues** - Cricket stadiums worldwide
- **Clean Data** - No players who never played T20 (like Wasim Akram, Sachin Tendulkar)

## How to Run

1. **Setup Database** (if needed):
   ```bash
   python setup_database.py
   ```

2. **Start API Server**:
   ```bash
   python app.py
   ```

3. **API Endpoints**:
   - `GET /api/teams` - Get all teams
   - `GET /api/venues` - Get all venues  
   - `GET /api/players` - Get all T20 players
   - `POST /api/predict` - Make score predictions
   - `GET /api/health` - Health check

## Database Schema

### Players Table
- `player_id` - Unique player ID
- `player_name` - Full player name
- `country` - Player's country
- `player_role` - Batsman, Bowler, Wicket-keeper, All-rounder
- `batting_style` - Right-handed, Left-handed
- `bowling_style` - Bowling style
- `is_active` - Active status

### Teams Table
- `team_id` - Unique team ID
- `team_name` - Team name
- `country` - Team's country
- `team_type` - International, Franchise
- `is_active` - Active status

### Venues Table
- `venue_id` - Unique venue ID
- `venue_name` - Venue name
- `city` - City
- `country` - Country
- `capacity` - Stadium capacity
- `venue_type` - Stadium type
- `pitch_type` - Batting, Bowling, Balanced
- `is_active` - Active status

## Notes

- Database contains only T20 players from the training dataset
- All players have actually played T20 cricket
- Legendary players like Wasim Akram, Sachin Tendulkar are excluded (never played T20)
- Modern T20 stars like Virat Kohli, MS Dhoni, Jasprit Bumrah are included
