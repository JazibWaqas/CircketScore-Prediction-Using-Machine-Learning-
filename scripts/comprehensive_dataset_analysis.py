#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis
Deep analysis of cleaned dataset from a data scientist perspective
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_dataset_analysis():
    """Comprehensive analysis of the cleaned dataset"""
    print("🔬 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 70)
    
    # Load the cleaned dataset
    try:
        df = pd.read_csv('data/simple_enhanced_train_cleaned.csv')
        print(f"✅ Loaded cleaned dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Time span: {df['date'].min()} to {df['date'].max()}")
    print(f"  Teams: {df['team'].nunique()}")
    print(f"  Venues: {df['venue'].nunique()}")
    
    # 1. FEATURE CATEGORIZATION
    print(f"\n🏷️ FEATURE CATEGORIZATION:")
    
    # Categorize all features
    feature_categories = {
        'IDENTIFIERS': [],
        'TEMPORAL': [],
        'VENUE': [],
        'TEAM': [],
        'MATCH_CONTEXT': [],
        'HISTORICAL': [],
        'DERIVED': [],
        'TARGET': []
    }
    
    for col in df.columns:
        if col in ['match_id', 'date', 'venue', 'team', 'opposition']:
            feature_categories['IDENTIFIERS'].append(col)
        elif col in ['season', 'season_year', 'season_month', 'is_winter', 'is_summer']:
            feature_categories['TEMPORAL'].append(col)
        elif 'venue' in col.lower():
            feature_categories['VENUE'].append(col)
        elif any(x in col.lower() for x in ['team', 'batting', 'bowling']):
            feature_categories['TEAM'].append(col)
        elif any(x in col.lower() for x in ['toss', 'first', 'final', 'playoff', 'important']):
            feature_categories['MATCH_CONTEXT'].append(col)
        elif any(x in col.lower() for x in ['h2h', 'form', 'recent']):
            feature_categories['HISTORICAL'].append(col)
        elif col == 'total_runs':
            feature_categories['TARGET'].append(col)
        else:
            feature_categories['DERIVED'].append(col)
    
    for category, features in feature_categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
            print(f"    {features}")
    
    # 2. FEATURE USEFULNESS ANALYSIS
    print(f"\n🎯 FEATURE USEFULNESS ANALYSIS:")
    
    # Calculate correlation with target
    numeric_features = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_features].corr()['total_runs'].abs().sort_values(ascending=False)
    
    print(f"  STRONG CORRELATIONS (>0.3):")
    strong_features = correlations[correlations > 0.3].drop('total_runs')
    for feature, corr in strong_features.items():
        print(f"    ✅ {feature}: {corr:.3f}")
    
    print(f"  WEAK CORRELATIONS (<0.1):")
    weak_features = correlations[correlations < 0.1]
    if 'total_runs' in weak_features.index:
        weak_features = weak_features.drop('total_runs')
    for feature, corr in weak_features.items():
        print(f"    ❌ {feature}: {corr:.3f}")
    
    # 3. CRICKET DOMAIN ANALYSIS
    print(f"\n🏏 CRICKET DOMAIN ANALYSIS:")
    
    # Check if we have the essential cricket features
    essential_cricket_features = {
        'venue_characteristics': ['venue_avg_runs', 'venue_difficulty'],
        'team_strength': ['team_batting_avg', 'team_balance'],
        'historical_context': ['h2h_avg_runs', 'team_form_avg_runs'],
        'match_context': ['batting_first', 'toss_winner', 'toss_decision']
    }
    
    for category, features in essential_cricket_features.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        print(f"  {category.upper()}:")
        if available:
            print(f"    ✅ Available: {available}")
        if missing:
            print(f"    ❌ Missing: {missing}")
    
    # 4. DATA QUALITY ISSUES
    print(f"\n🔍 DATA QUALITY ISSUES:")
    
    # Check for unrealistic values
    issues = []
    
    # Venue averages that are too extreme
    if 'venue_avg_runs' in df.columns:
        extreme_venues = df[(df['venue_avg_runs'] < 50) | (df['venue_avg_runs'] > 200)]
        if len(extreme_venues) > 0:
            issues.append(f"Extreme venue averages: {len(extreme_venues)} matches")
    
    # Team balances that are unrealistic
    if 'team_balance' in df.columns:
        extreme_balance = df[(df['team_balance'] < 0.1) | (df['team_balance'] > 2.0)]
        if len(extreme_balance) > 0:
            issues.append(f"Extreme team balances: {len(extreme_balance)} matches")
    
    # Check for constant features
    constant_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'total_runs' and df[col].nunique() <= 2:
            constant_features.append(col)
    
    if constant_features:
        issues.append(f"Constant/low-variance features: {constant_features}")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️ {issue}")
    else:
        print(f"  ✅ No major data quality issues found")
    
    # 5. MISSING CRICKET FEATURES ANALYSIS
    print(f"\n❓ MISSING CRICKET FEATURES ANALYSIS:")
    
    missing_features = {
        'PLAYER_LEVEL': [
            'star_players_count', 'key_batsman_present', 'key_bowler_present',
            'player_experience_avg', 'player_recent_form'
        ],
        'CONDITIONS': [
            'pitch_type', 'weather_conditions', 'dew_factor',
            'boundary_dimensions', 'lighting_conditions'
        ],
        'MATCH_SITUATION': [
            'overs_remaining', 'wickets_in_hand', 'required_run_rate',
            'pressure_situation', 'chasing_score'
        ],
        'OPPOSITION_ANALYSIS': [
            'opposition_bowling_strength', 'opposition_fielding_quality',
            'head_to_head_recent', 'opposition_weakness_analysis'
        ],
        'VENUE_SPECIFIC': [
            'venue_pitch_type', 'venue_boundary_size', 'venue_scoring_patterns',
            'venue_difficulty_by_team', 'venue_weather_patterns'
        ]
    }
    
    for category, features in missing_features.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        print(f"  {category}:")
        if available:
            print(f"    ✅ Available: {available}")
        print(f"    ❌ Missing: {missing}")
    
    # 6. FEATURE ENGINEERING OPPORTUNITIES
    print(f"\n🔧 FEATURE ENGINEERING OPPORTUNITIES:")
    
    opportunities = []
    
    # Time-based features
    if 'date' in df.columns:
        opportunities.append("Time-based: day_of_week, is_weekend, month_season")
    
    # Team composition features
    if 'team_players' in df.columns:
        opportunities.append("Team composition: player_count, specialist_ratio")
    
    # Venue-specific team performance
    if 'venue_id' in df.columns and 'team_id' in df.columns:
        opportunities.append("Venue-specific: team_avg_at_venue, team_win_rate_at_venue")
    
    # Opposition-specific features
    if 'opposition' in df.columns:
        opportunities.append("Opposition-specific: team_vs_opposition_avg, recent_form_vs_opposition")
    
    for opportunity in opportunities:
        print(f"  💡 {opportunity}")
    
    # 7. PREDICTION CHALLENGES
    print(f"\n🎯 PREDICTION CHALLENGES:")
    
    challenges = []
    
    # High variance in target
    if df['total_runs'].std() > 40:
        challenges.append(f"High variance in target variable (std: {df['total_runs'].std():.1f})")
    
    # Check for non-linear relationships
    if 'venue_avg_runs' in df.columns:
        venue_variance = df.groupby('venue_avg_runs')['total_runs'].std().mean()
        if venue_variance > 30:
            challenges.append(f"High variance within venues (avg std: {venue_variance:.1f})")
    
    # Team performance variance
    if 'team_id' in df.columns:
        team_variance = df.groupby('team_id')['total_runs'].std().mean()
        if team_variance > 30:
            challenges.append(f"High variance within teams (avg std: {team_variance:.1f})")
    
    if challenges:
        for challenge in challenges:
            print(f"  ⚠️ {challenge}")
    else:
        print(f"  ✅ No major prediction challenges identified")
    
    # 8. RECOMMENDATIONS
    print(f"\n💡 RECOMMENDATIONS:")
    
    recommendations = []
    
    # Feature importance
    top_features = correlations.head(6).drop('total_runs')
    recommendations.append(f"Focus on top features: {', '.join(top_features.index.tolist())}")
    
    # Missing features
    if not any('player' in col.lower() for col in df.columns):
        recommendations.append("Add player-level features for better accuracy")
    
    if not any('weather' in col.lower() or 'pitch' in col.lower() for col in df.columns):
        recommendations.append("Add pitch and weather conditions")
    
    # Data quality
    if any(df[col].std() < 0.01 for col in df.select_dtypes(include=[np.number]).columns if col != 'total_runs'):
        recommendations.append("Remove or fix constant features")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 9. OVERALL ASSESSMENT
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    score = 0
    max_score = 10
    
    # Data quality (2 points)
    if len(issues) == 0:
        score += 2
    elif len(issues) <= 2:
        score += 1
    
    # Feature richness (2 points)
    if len(strong_features) >= 5:
        score += 2
    elif len(strong_features) >= 3:
        score += 1
    
    # Cricket domain coverage (2 points)
    essential_coverage = sum(1 for category in essential_cricket_features.values() 
                           if any(f in df.columns for f in category))
    if essential_coverage >= 3:
        score += 2
    elif essential_coverage >= 2:
        score += 1
    
    # Missing features (2 points)
    missing_count = sum(len(missing) for missing in missing_features.values())
    if missing_count <= 10:
        score += 2
    elif missing_count <= 15:
        score += 1
    
    # Prediction feasibility (2 points)
    if len(challenges) <= 1:
        score += 2
    elif len(challenges) <= 2:
        score += 1
    
    print(f"  Dataset Quality Score: {score}/{max_score}")
    
    if score >= 8:
        print(f"  ✅ EXCELLENT: Dataset is ready for high-quality predictions")
    elif score >= 6:
        print(f"  🔶 GOOD: Dataset is usable but could be improved")
    elif score >= 4:
        print(f"  ⚠️ FAIR: Dataset has significant limitations")
    else:
        print(f"  ❌ POOR: Dataset needs major improvements before use")
    
    return {
        'score': score,
        'max_score': max_score,
        'strong_features': strong_features,
        'weak_features': weak_features,
        'issues': issues,
        'recommendations': recommendations,
        'missing_features': missing_features
    }

if __name__ == "__main__":
    results = comprehensive_dataset_analysis()
    
    print(f"\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if results['score'] >= 6:
        print(f"✅ DATASET IS READY FOR MODEL TRAINING")
        print(f"   Quality Score: {results['score']}/{results['max_score']}")
        print(f"   Strong features available: {len(results['strong_features'])}")
        print(f"   Recommendations to implement: {len(results['recommendations'])}")
    else:
        print(f"❌ DATASET NEEDS IMPROVEMENT BEFORE TRAINING")
        print(f"   Quality Score: {results['score']}/{results['max_score']}")
        print(f"   Critical issues: {len(results['issues'])}")
        print(f"   Missing features: {sum(len(m) for m in results['missing_features'].values())}")
        print(f"   Recommendations: {len(results['recommendations'])}")
    
    print(f"\n🎯 NEXT STEPS:")
    if results['score'] >= 6:
        print(f"1. Train models with current dataset")
        print(f"2. Implement recommended improvements")
        print(f"3. Test model performance")
    else:
        print(f"1. Address critical issues first")
        print(f"2. Add missing essential features")
        print(f"3. Improve data quality")
        print(f"4. Re-assess dataset quality")
