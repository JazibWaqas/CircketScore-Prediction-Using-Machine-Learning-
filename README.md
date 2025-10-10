# 🏏 Cricket Score Prediction with Player Impact

**Machine learning system for ODI/T20 cricket score prediction with player impact analysis**

---

## 🚨 **CURRENT STATUS**

### ✅ **Working**
- **Frontend:** Beautiful React UI with T20/ODI toggle, player search/filters ✓
- **T20 System:** Fully functional (R² ~0.65-0.70) ✓
- **ODI API:** All endpoints working ✓
- **Player Database:** 1,872 players with impact coefficients ✓

### ❌ **Broken**
- **ODI Model:** R²=0.01 (claimed 0.69) - predicts ~235 runs every time
- **Cause:** Feature mismatch + never properly validated
- **Fix:** Rebuild model (6-8 hours)

**📄 Full Details:** [ODI/README.md](ODI/README.md)

---

## 🎯 **PROJECT STRUCTURE**

```
├── frontend/              - React app (working) ✓
│   ├── src/
│   │   ├── App.js        - T20/ODI toggle, prediction logic
│   │   └── components/   - TeamSelector, PredictionResults, etc.
│   └── package.json
│
├── ODI/                   - ODI system (model broken, infrastructure working)
│   ├── README.md         - Comprehensive ODI status & fix plan
│   ├── models/           - CURRENT_* (broken), REFERENCE_* (failed experiments)
│   ├── data/             - CURRENT_* (working), BROKEN_* (test data issues)
│   ├── scripts/          - 11 essential scripts (data gen, training reference)
│   └── Database/
│       └── run_odi_api_COMPLETE.py  - API (port 5001)
│
├── T20/                   - T20 system (working) ✓
│   ├── models/           - Trained models (R² ~0.70)
│   ├── data/             - Training datasets
│   └── Database/
│       └── run_final.py  - API (port 5000)
│
└── raw_data/
    ├── odis_ballbyBall/  - 5,761 ODI match JSONs
    └── odi_data/         - Player statistics CSV
```

---

## 📊 **FILE NAMING CONVENTION**

```
CURRENT_*     = Currently being used (may be broken but active)
BROKEN_*      = Known to be broken
REFERENCE_*   = Historical reference only
  ├─ FAILED_* = Failed experiments
  └─ OLD_*    = Previous approaches
```

**Examples:**
- `ODI/models/CURRENT_BROKEN_baseline_xgboost.pkl` - Model API uses (R²=0.01)
- `ODI/data/CURRENT_training_data_7314_matches.csv` - Main dataset (working)
- `ODI/data/BROKEN_test_data_missing_8_features.csv` - Test set with issues
- `ODI/models/REFERENCE_FAILED_enhanced_xgboost.pkl` - Failed 127-feature attempt

---

## 🚀 **QUICK START**

### **Run Everything:**
```bash
# Terminal 1: T20 API (working)
cd T20/Database && python run_final.py

# Terminal 2: ODI API (broken model but API works)
cd ODI/Database && python run_odi_api_COMPLETE.py

# Terminal 3: Frontend
cd frontend && npm start

# Open: http://localhost:3000
```

### **Test ODI Model (See It's Broken):**
```bash
python TEST_MODEL_WITH_REAL_FEATURES.py
# Output: R²=0.01, MAE=56.5, predictions cluster ~235 runs
```

---

## 🔧 **HOW TO FIX ODI MODEL**

### **Recommended: Rebuild from Scratch**

**Why:** Clean slate, proven approach, 90% success rate  
**Time:** 6-8 hours  
**Steps:**

1. **Build Simple Dataset** (2-3 hours)
   - Use 15-20 proven features (team averages, venue stats, form, toss)
   - Ensure train/test have IDENTICAL features
   - Temporal split (last 500 matches = test)

2. **Train XGBoost** (2 hours)
   - Conservative hyperparameters (avoid overfitting)
   - Cross-validation during training
   - Save: model, scaler, feature names

3. **Actually Test** (1 hour)
   - Test on held-out 500 matches
   - Verify R² > 0.65, MAE < 35
   - If not, iterate on features

4. **Validate** (30 min)
   - Test with real historical matches
   - Verify predictions make sense

5. **Deploy** (30 min)
   - Replace CURRENT_BROKEN_* files
   - Test via API and frontend

**Target:** R² > 0.70, MAE < 28, 65%+ within ±30 runs

---

## 📦 **WHAT WE HAVE**

### **Data (All in `ODI/data/`):**
- `CURRENT_training_data_7314_matches.csv` - 7,314 ODI matches, 71 features
- `CURRENT_player_database_977_quality.json` - 977 players with career stats
- `CURRENT_player_impacts_1872_all.json` - Impact coefficients for all 1,872 players
- Lookup tables for teams, venues, players

### **Models (All in `ODI/models/`):**
- `CURRENT_BROKEN_*` - Being used but broken (R²=0.01)
- `REFERENCE_FAILED_*` - Enhanced model attempt (R²=0.52, worse)
- `REFERENCE_OLD_*` - T20-style approach

### **Scripts (11 essential in `ODI/scripts/`):**
- `1_build_player_database.py` - Generated player DB
- `BUILD_COMPLETE_DATASET.py` - Generated training data
- `TRAIN_COMPLETE.py` - Training reference
- `GENERATE_PLAYER_COEFFICIENTS.py` - Generated impact scores

### **Testing:**
- `TEST_MODEL_WITH_REAL_FEATURES.py` - Tests model on held-out data
- `VERIFY_MODEL_ACCURACY.py` - Tests via API with real matches
- `get_real_match.py` - Gets historical match details

---

## 🎨 **FRONTEND FEATURES**

- **Format Toggle:** Switch between T20 and ODI
- **Player Selection:** 
  - Search by name, country, role
  - Filters: Country (all), Role, Quality Tier
  - Dropdown stays open when adding
  - Shows impact scores next to names
- **Match Context:** Venue, toss, tournament, pitch, weather
- **Results:** Shows both team scores, predicted winner, player breakdowns

---

## 🔑 **KEY LESSONS**

1. **Never trust saved metrics** - Always test on held-out data yourself
2. **Feature engineering matters** - 15 good features > 67 questionable ones
3. **Validate early** - Test before building entire system on top
4. **Simple works** - Complexity often makes things worse
5. **Keep player impact separate** - Works well as overlay on base prediction

---

## 📞 **APIS**

### **ODI API** (Port 5001)
```
GET  /api/odi/health      - Status check
GET  /api/odi/teams       - Get 22 teams
GET  /api/odi/players     - Get 1,872 players with impacts
GET  /api/odi/venues      - Get 378 venues with stats
POST /api/odi/predict     - Make prediction (currently broken)
```

### **T20 API** (Port 5000)
```
Fully functional, use as reference
```

---

## 📝 **FOR TOMORROW**

1. Read `ODI/README.md` (this file) - 10 min
2. Decide: Rebuild or Fix
3. If rebuilding: Follow 5-step plan
4. Test thoroughly!
5. Expected: R² > 0.70 achievable in 1 day

---

**Bottom Line:** 90% complete, just need working prediction model! 🎯
