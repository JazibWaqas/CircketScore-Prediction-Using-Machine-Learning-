#!/usr/bin/env python3
"""
Comprehensive Final Dataset Evaluation
Thorough evaluation of the final clean dataset before training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def comprehensive_final_dataset_evaluation():
    """Comprehensive evaluation of the final clean dataset"""
    print("🔬 COMPREHENSIVE FINAL DATASET EVALUATION")
    print("=" * 70)
    
    # Load the final clean dataset
    try:
        df = pd.read_csv('processed_data/final_clean_dataset.csv')
        print(f"✅ Loaded final clean dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Target variable: {'total_runs' if 'total_runs' in df.columns else 'MISSING!'}")
    
    if 'total_runs' in df.columns:
        print(f"  Target statistics:")
        print(f"    Mean: {df['total_runs'].mean():.1f}")
        print(f"    Std: {df['total_runs'].std():.1f}")
        print(f"    Min: {df['total_runs'].min()}")
        print(f"    Max: {df['total_runs'].max()}")
        print(f"    Range: {df['total_runs'].max() - df['total_runs'].min()}")
    
    # 1. CRITICAL DATA LEAKAGE CHECK
    print(f"\n🚨 CRITICAL DATA LEAKAGE CHECK:")
    
    leakage_indicators = [
        'match_winner', 'player_of_match', 'is_final', 'is_semi_final',
        'winner', 'loser', 'result', 'outcome', 'score_a', 'score_b',
        'total_score', 'match_result', 'winning_team', 'losing_team'
    ]
    
    found_leakage = []
    for indicator in leakage_indicators:
        if indicator in df.columns:
            found_leakage.append(indicator)
    
    if found_leakage:
        print(f"  ❌ CRITICAL: Data leakage features found: {found_leakage}")
        print(f"  🚨 These must be removed before training!")
    else:
        print(f"  ✅ No data leakage features found")
    
    # 2. TARGET VARIABLE ANALYSIS
    print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
    
    if 'total_runs' not in df.columns:
        print(f"  ❌ CRITICAL: No target variable found!")
        return
    
    # Check for unrealistic scores
    unrealistic_scores = df[(df['total_runs'] < 20) | (df['total_runs'] > 300)]
    if len(unrealistic_scores) > 0:
        print(f"  ⚠️ Unrealistic scores found: {len(unrealistic_scores)} matches")
        print(f"    Range: {unrealistic_scores['total_runs'].min()} to {unrealistic_scores['total_runs'].max()}")
    
    # Check for missing target values
    missing_targets = df['total_runs'].isnull().sum()
    if missing_targets > 0:
        print(f"  ❌ Missing target values: {missing_targets}")
    else:
        print(f"  ✅ No missing target values")
    
    # 3. FEATURE QUALITY ANALYSIS
    print(f"\n🔍 FEATURE QUALITY ANALYSIS:")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > len(df) * 0.05]
    if len(high_missing) > 0:
        print(f"  ⚠️ Features with >5% missing: {len(high_missing)}")
        for feature, count in high_missing.items():
            percentage = count / len(df) * 100
            print(f"    {feature}: {count} ({percentage:.1f}%)")
    else:
        print(f"  ✅ No features with high missing values")
    
    # Check for constant features
    constant_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'total_runs' and df[col].nunique() <= 2:
            constant_features.append(col)
    
    if constant_features:
        print(f"  ⚠️ Constant features: {len(constant_features)}")
        print(f"    {constant_features[:5]}...")
    else:
        print(f"  ✅ No constant features")
    
    # 4. FEATURE CORRELATION ANALYSIS
    print(f"\n📈 FEATURE CORRELATION ANALYSIS:")
    
    numeric_features = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
    
    strong_features = correlations[correlations > 0.3]
    if 'total_runs' in strong_features.index:
        strong_features = strong_features.drop('total_runs')
    
    moderate_features = correlations[(correlations > 0.1) & (correlations <= 0.3)]
    if 'total_runs' in moderate_features.index:
        moderate_features = moderate_features.drop('total_runs')
    
    weak_features = correlations[correlations <= 0.1]
    if 'total_runs' in weak_features.index:
        weak_features = weak_features.drop('total_runs')
    
    print(f"  Strong correlations (>0.3): {len(strong_features)}")
    print(f"  Moderate correlations (0.1-0.3): {len(moderate_features)}")
    print(f"  Weak correlations (<0.1): {len(weak_features)}")
    
    print(f"  Top 10 strongest features:")
    for feature, corr in strong_features.head(10).items():
        print(f"    {feature}: {corr:.3f}")
    
    # 5. MISSING IMPORTANT CRICKET FEATURES
    print(f"\n🏏 MISSING IMPORTANT CRICKET FEATURES:")
    
    essential_cricket_features = {
        'PLAYER_LEVEL': [
            'star_players_count', 'key_batsman_present', 'key_bowler_present',
            'player_experience_avg', 'player_recent_form', 'player_impact_score'
        ],
        'VENUE_CONDITIONS': [
            'pitch_type', 'weather_conditions', 'dew_factor', 'boundary_dimensions',
            'lighting_conditions', 'temperature', 'humidity'
        ],
        'MATCH_SITUATION': [
            'overs_remaining', 'wickets_in_hand', 'required_run_rate',
            'pressure_situation', 'chasing_score', 'batting_first'
        ],
        'TEAM_DYNAMICS': [
            'team_balance', 'team_chemistry', 'team_strength', 'team_depth',
            'batting_strength', 'bowling_strength'
        ],
        'HISTORICAL_CONTEXT': [
            'h2h_avg_runs', 'team_form_avg_runs', 'recent_form',
            'venue_specific_form', 'opposition_analysis'
        ]
    }
    
    missing_by_category = {}
    for category, features in essential_cricket_features.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        missing_by_category[category] = len(missing)
        
        print(f"  {category}:")
        if available:
            print(f"    ✅ Available: {available}")
        if missing:
            print(f"    ❌ Missing: {missing}")
    
    # 6. TRAIN/TEST SPLIT ANALYSIS
    print(f"\n📊 TRAIN/TEST SPLIT ANALYSIS:")
    
    # Check if we have enough data for proper splitting
    if len(df) < 1000:
        print(f"  ⚠️ Small dataset: {len(df)} records (minimum 1000 recommended)")
    else:
        print(f"  ✅ Sufficient data for train/test split: {len(df):,} records")
    
    # Check for temporal consistency
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].max() - df['date'].min()
        print(f"  📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  📅 Time span: {date_range.days} days")
        
        # Check for temporal leakage
        if df['date'].nunique() < len(df) * 0.8:
            print(f"  ⚠️ Multiple matches on same dates - check for temporal leakage")
        else:
            print(f"  ✅ Good temporal distribution")
    
    # 7. MODEL READINESS TEST
    print(f"\n🤖 MODEL READINESS TEST:")
    
    try:
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['total_runs', 'match_id', 'date', 'venue', 'team', 'opposition']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['total_runs']
        
        print(f"  Features for training: {len(X.columns)}")
        print(f"  Target variable: {y.name}")
        
        # Check for infinite values
        inf_features = []
        for col in X.columns:
            if np.isinf(X[col]).any():
                inf_features.append(col)
        
        if inf_features:
            print(f"  ❌ Features with infinite values: {inf_features}")
        else:
            print(f"  ✅ No infinite values")
        
        # Simple train/test split test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Quick model test
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  Quick test results:")
        print(f"    R² Score: {r2:.3f}")
        print(f"    RMSE: {rmse:.1f}")
        print(f"    MAE: {mae:.1f}")
        
        if r2 > 0.3:
            print(f"  ✅ Model shows promise (R² > 0.3)")
        else:
            print(f"  ⚠️ Model performance may be limited (R² < 0.3)")
        
    except Exception as e:
        print(f"  ❌ Model readiness test failed: {e}")
    
    # 8. OVERALL ASSESSMENT
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    score = 0
    max_score = 10
    
    # Data leakage (2 points)
    if len(found_leakage) == 0:
        score += 2
    else:
        score += 0
    
    # Target variable (2 points)
    if 'total_runs' in df.columns and missing_targets == 0:
        score += 2
    elif 'total_runs' in df.columns:
        score += 1
    
    # Feature quality (2 points)
    if len(high_missing) == 0 and len(constant_features) == 0:
        score += 2
    elif len(high_missing) == 0 or len(constant_features) == 0:
        score += 1
    
    # Strong features (2 points)
    if len(strong_features) >= 10:
        score += 2
    elif len(strong_features) >= 5:
        score += 1
    
    # Missing features (2 points)
    total_missing = sum(missing_by_category.values())
    if total_missing <= 5:
        score += 2
    elif total_missing <= 10:
        score += 1
    
    print(f"  Dataset Quality Score: {score}/{max_score}")
    
    if score >= 8:
        print(f"  ✅ EXCELLENT: Dataset is ready for high-quality training")
    elif score >= 6:
        print(f"  🔶 GOOD: Dataset is usable with minor improvements")
    elif score >= 4:
        print(f"  ⚠️ FAIR: Dataset has significant limitations")
    else:
        print(f"  ❌ POOR: Dataset needs major improvements")
    
    # 9. RECOMMENDATIONS
    print(f"\n💡 RECOMMENDATIONS:")
    
    recommendations = []
    
    if found_leakage:
        recommendations.append(f"🚨 CRITICAL: Remove data leakage features: {found_leakage}")
    
    if 'total_runs' not in df.columns:
        recommendations.append("🚨 CRITICAL: Add target variable 'total_runs'")
    
    if len(high_missing) > 0:
        recommendations.append(f"🔧 Fix {len(high_missing)} features with high missing values")
    
    if len(constant_features) > 0:
        recommendations.append(f"🔧 Remove {len(constant_features)} constant features")
    
    if len(strong_features) < 10:
        recommendations.append("🔧 Add more features with strong correlations")
    
    if total_missing > 10:
        recommendations.append(f"🔧 Add missing cricket features: {total_missing} features")
    
    if score >= 6:
        recommendations.append("✅ Proceed with model training")
    else:
        recommendations.append("❌ Fix critical issues before training")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return {
        'score': score,
        'max_score': max_score,
        'found_leakage': found_leakage,
        'strong_features': len(strong_features),
        'missing_features': total_missing,
        'recommendations': recommendations,
        'model_ready': score >= 6
    }

if __name__ == "__main__":
    results = comprehensive_final_dataset_evaluation()
    
    print(f"\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if results['model_ready']:
        print(f"✅ DATASET IS READY FOR TRAINING")
        print(f"   Quality Score: {results['score']}/{results['max_score']}")
        print(f"   Strong features: {results['strong_features']}")
        print(f"   Data leakage: {len(results['found_leakage'])} issues")
        print(f"   Missing features: {results['missing_features']}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Train models with clean dataset")
        print(f"2. Implement proper train/test split")
        print(f"3. Evaluate model performance")
        print(f"4. Deploy improved models")
    else:
        print(f"❌ DATASET NEEDS IMPROVEMENTS BEFORE TRAINING")
        print(f"   Quality Score: {results['score']}/{results['max_score']}")
        print(f"   Critical issues: {len(results['found_leakage'])}")
        print(f"   Missing features: {results['missing_features']}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Fix critical issues first")
        print(f"2. Add missing essential features")
        print(f"3. Improve data quality")
        print(f"4. Re-evaluate dataset")
