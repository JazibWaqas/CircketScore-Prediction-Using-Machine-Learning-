# Database API

This folder contains the Flask API server for cricket score predictions.

## Files

- `run.py` - Original API (outdated, references old lookup tables)
- `run_clean.py` - **NEW CLEAN API** that works with our final training dataset
- `setup_database.py` - Database setup script
- `requirements.txt` - Python dependencies
- `cricket_prediction.db` - SQLite database (if used)

## Usage

### Clean API (Recommended)
```bash
cd Database
python run_clean.py
```

### Original API (Legacy)
```bash
cd Database
python run.py
```

## API Endpoints

- `POST /predict` - Make cricket score predictions
- `GET /health` - Health check
- `GET /models` - Model information

## Requirements

The API expects trained models in the `../models/` folder:
- `final_random_forest.pkl`
- `final_xgboost.pkl`
- `final_linear_regression.pkl`
- `final_scaler.pkl`
- `final_encoders.pkl`

And training data in `../processed_data/final_training_dataset.csv`

## Example Request

```json
{
  "team": "India",
  "opposition": "Australia",
  "venue": "Melbourne Cricket Ground"
}
```

## Example Response

```json
{
  "predictions": {
    "random_forest": 145.2,
    "xgboost": 142.8,
    "linear_regression": 138.5
  },
  "average_prediction": 142.2,
  "team": "India",
  "opposition": "Australia",
  "venue": "Melbourne Cricket Ground"
}
```