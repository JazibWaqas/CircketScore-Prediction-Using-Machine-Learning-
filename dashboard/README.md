# ODI Progressive Predictor Dashboard

Complete web application for progressive ODI score prediction with fantasy team building.

## Quick Start

### Backend (Terminal 1)
```bash
cd dashboard/backend
python app.py
```
Backend runs on: `http://localhost:5002`

### Frontend (Terminal 2)
```bash
cd dashboard/frontend
npm start
```
Frontend runs on: `http://localhost:3000`

## Features

- ✅ Select 11 batting players
- ✅ Select 11 bowling players  
- ✅ Set match scenario (pre-match to death overs)
- ✅ Get progressive predictions with confidence levels
- ✅ View team statistics and aggregates
- ✅ Progressive accuracy display (R² improves from 0.35 to 0.94)

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/teams` - List teams
- `GET /api/players/{team}` - Get players
- `GET /api/venues` - List venues  
- `POST /api/predict` - Make prediction
- `POST /api/whatif` - Compare scenarios
- `POST /api/progressive` - Multiple checkpoints

## Model Performance

- Pre-match: R² = 0.35, MAE = 41 runs
- Mid-match (20 overs): R² = 0.75, MAE = 24 runs
- Death overs (40+): R² = 0.94, MAE = 12 runs
- Overall: R² = 0.69, MAE = 25 runs

Validated on 2,904 predictions from 592 international ODI matches.

