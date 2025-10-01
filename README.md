# 🏏 Cricket Score Prediction Using Machine Learning

## 📋 Project Overview

This project aims to build a comprehensive cricket score prediction system using machine learning. The goal is to predict T20 team scores based on team composition, venue, match context, and historical performance data.

## 🎯 Project Goals

- **Predict T20 team scores** with high accuracy
- **Enable "what if" scenarios** - select any team combination and predict scores
- **Context-aware predictions** - consider venue, opposition, toss, match importance
- **Player impact analysis** - understand how individual players affect team performance
- **Interactive frontend** - user-friendly interface for team selection and predictions

## 📊 Datasets

### 1. **T20 Ball-by-Ball Matches** (`t20 matches ball by ball/`)
- **7,223 T20 match files** in JSON format
- **Ball-by-ball data** for each match
- **Team lineups** for each match
- **Match context** (venue, toss, outcome, etc.)
- **Date range:** 2005-2025 (20+ years of data)

### 2. **Player Statistics** (`PlayerStats/`)
- **`all_players.csv`** - Player information (name, country, playing role)
- **`t20_batting.csv`** - T20 batting statistics
- **`t20_bowling.csv`** - T20 bowling statistics
- **`fielding.csv`** - Fielding statistics
- **`country.csv`** - Country information
- **`t20_all_round.csv`** - All-rounder statistics

## 📁 Current Files

### **Core Datasets**
- **`comprehensive_t20_dataset.csv`** - Main dataset with 14,611 team performances
- **`ml_ready_comprehensive_t20_dataset.csv`** - ML-ready numerical dataset

### **Scripts**
- **`build_comprehensive_t20_dataset.py`** - Script to build the comprehensive dataset from JSON files

## 📈 Dataset Statistics

### **Comprehensive T20 Dataset**
- **14,611 team performances** from 7,223 matches
- **172 unique teams** (international + domestic)
- **506 unique venues** worldwide
- **47 features** per team performance

### **Key Features**
- **Team performance:** Total runs, boundaries, run rate, overs
- **Venue context:** Venue-specific scoring patterns
- **Head-to-head:** Historical performance between teams
- **Match context:** Toss decision, batting first, match importance
- **Player lineups:** Actual 11 players from each match

## 🚧 Current Status

### ✅ **Completed**
1. **Dataset Creation** - Built comprehensive dataset from 7,223 T20 matches
2. **Data Processing** - Extracted team performances with rich features
3. **Feature Engineering** - Created venue stats, head-to-head records, team form
4. **Data Cleaning** - Handled missing values and data inconsistencies

### 🔄 **In Progress**
- **Model Development** - Need to build proper predictive models
- **Frontend Development** - Team selection interface

### 📋 **Next Steps**
1. **Build Predictive Models** - Train models on pre-match features only
2. **Create Frontend** - Interactive team selection and prediction interface
3. **Model Validation** - Test on unseen data and validate accuracy
4. **Deployment** - Make the system accessible to users

## 🎯 What We Can Build

### **Team Selection Interface**
- Select any 11 players from any team
- Choose venue and match context
- Get score predictions for both teams

### **"What If" Scenarios**
- "What if India plays Australia at MCG?"
- "What if Team A wins toss and bats first?"
- "What if we change the playing XI?"

### **Player Impact Analysis**
- Individual player performance in specific conditions
- Player combinations and their effectiveness
- Venue-specific player performance

## 🔧 Technical Requirements

### **Python Libraries**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `json` - JSON file processing

### **Data Processing**
- JSON parsing for match data
- Feature engineering for team stats
- Venue and head-to-head calculations
- Team form and context features

## 📝 Usage

### **Building the Dataset**
```bash
python build_comprehensive_t20_dataset.py
```

This will:
1. Process all 7,223 T20 match JSON files
2. Extract team performances and features
3. Calculate venue statistics and head-to-head records
4. Create the comprehensive dataset

### **Dataset Structure**
Each row represents one team's performance in one match:
- **Match info:** ID, date, venue, teams
- **Team performance:** Runs, boundaries, run rate
- **Context:** Toss, batting first, match importance
- **Historical:** Venue stats, head-to-head records, team form

## 🎯 Future Enhancements

1. **Real-time Predictions** - Live match predictions
2. **Player Analytics** - Individual player impact analysis
3. **Venue Analysis** - Detailed venue-specific insights
4. **Match Simulation** - Full match outcome prediction
5. **Mobile App** - Mobile-friendly interface

## 📊 Data Quality

- **Comprehensive coverage** - 20+ years of T20 data
- **Global scope** - International and domestic matches
- **Rich context** - Venue, opposition, match importance
- **Player details** - Actual lineups from each match
- **Clean data** - Handled missing values and inconsistencies

## 🏆 Project Value

This system will enable:
- **Cricket analysts** to understand team performance patterns
- **Coaches** to make data-driven team selection decisions
- **Fans** to explore "what if" scenarios
- **Researchers** to study cricket performance factors

---

**Status:** Dataset creation completed, model development and frontend pending
**Last Updated:** October 2024