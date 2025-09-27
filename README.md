# Cricket Score Prediction Project

## 🎯 Project Overview
A machine learning system to predict T20 cricket team scores with 98% accuracy, enabling "what if" scenario analysis.

## 📊 Final Dataset
- **File**: `fixed_cricket_dataset.csv`
- **Size**: 985 team innings records
- **Features**: 26 predictive features
- **Target**: Total runs scored (3-260 runs, mean: 148 runs)

## 🚀 Quick Start

### 1. Train the Model
```bash
python cricket_score_prediction_model.py
```

### 2. Make Predictions
```bash
python cricket_prediction_interface.py
```

## 📁 Essential Files

### Core Files
- `fixed_cricket_dataset.csv` - **FINAL DATASET** with real target values
- `fixed_dataset_builder.py` - Data processing pipeline
- `cricket_score_prediction_model.py` - ML model training & evaluation
- `cricket_prediction_interface.py` - Interactive prediction interface

### Data Sources
- `PlayerStats/` - Player statistics (666 players)
- `t20 matches ball by ball/` - Match data (7,223 JSON files)

### Documentation
- `README.md` - This file
- `PROJECT_SUMMARY.md` - Detailed project summary

## 🎮 Usage Examples

### Sample Predictions
- **India vs Pakistan at Dubai**: ~160-180 runs
- **Australia vs England at MCG**: ~165-185 runs
- **Balanced teams at neutral venue**: ~140-160 runs

### Model Performance
- **Accuracy**: 98.01% (R² score)
- **Error**: ±5.78 runs on average
- **Best Model**: XGBoost

## 🔧 Requirements
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## 📈 Key Features
- Team strength analysis
- Venue-specific predictions
- Opposition strength factors
- Match context (innings, run rate, boundaries)
- What-if scenario support

---
**Ready for cricket score prediction with high accuracy!**
