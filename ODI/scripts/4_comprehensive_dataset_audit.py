#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ODI DATASET AUDIT & VALIDATION

Purpose: Perform exhaustive quality checks before model training
Strategy: Validate every aspect - integrity, features, distributions, readiness
Output: Detailed audit report with data quality score and recommendations
"""

import pandas as pd
import numpy as np
import json
import sys
import warnings
from collections import Counter, defaultdict
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class ODIDatasetAuditor:
    """Complete dataset auditor with comprehensive checks"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.quality_scores = {}
        
    def load_dataset(self):
        """Load and basic info"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ODI DATASET AUDIT")
        print("="*80)
        print(f"\nLoading: {self.dataset_path}")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {self.df.shape} (rows × columns)")
        
    def section_header(self, title, number):
        """Print section header"""
        print("\n" + "="*80)
        print(f"{number}. {title}")
        print("="*80)
    
    def subsection(self, title):
        """Print subsection"""
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
    
    # =========================================================================
    # 1. DATASET OVERVIEW
    # =========================================================================
    
    def audit_overview(self):
        """Section 1: Dataset Overview"""
        self.section_header("DATASET OVERVIEW", 1)
        
        print(f"\n📊 Basic Information:")
        print(f"  Rows: {len(self.df):,}")
        print(f"  Columns: {len(self.df.columns)}")
        print(f"  Memory Usage: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        print(f"\n📋 Column Types:")
        type_counts = self.df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count} columns")
        
        print(f"\n🎯 Target Variable:")
        if 'total_runs' in self.df.columns:
            print(f"  ✓ Found: 'total_runs'")
            print(f"    Range: {self.df['total_runs'].min()} - {self.df['total_runs'].max()}")
            print(f"    Mean: {self.df['total_runs'].mean():.2f}")
            print(f"    Median: {self.df['total_runs'].median():.2f}")
        else:
            self.issues.append("Target variable 'total_runs' not found")
            print(f"  ✗ Target variable 'total_runs' NOT FOUND")
        
        print(f"\n📝 Sample Rows (first 3):")
        print(self.df.head(3).to_string())
        
        self.quality_scores['overview'] = 10.0  # Full marks if loaded
    
    # =========================================================================
    # 2. INTEGRITY & QUALITY CHECKS
    # =========================================================================
    
    def audit_integrity(self):
        """Section 2: Data Integrity & Quality"""
        self.section_header("INTEGRITY & QUALITY CHECKS", 2)
        
        score = 10.0
        
        # Missing values
        self.subsection("Missing Values")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(f"  ⚠️ Found {len(missing_df)} columns with missing values:")
            print(missing_df.to_string())
            self.warnings.append(f"{len(missing_df)} columns have missing values")
            score -= min(2.0, len(missing_df) * 0.5)
        else:
            print(f"  ✓ No missing values found!")
        
        # Infinite values
        self.subsection("Infinite Values")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(self.df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            print(f"  ✗ Found infinite values in {len(inf_cols)} columns:")
            for col in inf_cols:
                print(f"    - {col}")
            self.issues.append(f"Infinite values in {len(inf_cols)} columns")
            score -= 2.0
        else:
            print(f"  ✓ No infinite values found!")
        
        # Duplicate rows
        self.subsection("Duplicate Detection")
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"  ⚠️ Found {duplicates:,} duplicate rows ({duplicates/len(self.df)*100:.2f}%)")
            self.warnings.append(f"{duplicates} duplicate rows found")
            score -= 1.0
        else:
            print(f"  ✓ No duplicate rows found!")
        
        # Match ID check
        self.subsection("Match Consistency")
        if 'match_id' in self.df.columns:
            unique_matches = self.df['match_id'].nunique()
            total_rows = len(self.df)
            expected_rows = unique_matches * 2  # 2 rows per match
            
            print(f"  Unique matches: {unique_matches:,}")
            print(f"  Total rows: {total_rows:,}")
            print(f"  Expected rows (2 per match): {expected_rows:,}")
            
            if total_rows == expected_rows:
                print(f"  ✓ Perfect match structure (2 rows per match)")
            else:
                diff = abs(total_rows - expected_rows)
                print(f"  ⚠️ Row count mismatch: {diff} rows difference")
                self.warnings.append(f"Match structure inconsistency: {diff} rows")
                score -= 0.5
        else:
            print(f"  ⚠️ No 'match_id' column found")
            self.warnings.append("Missing match_id column")
        
        # Player count validation
        self.subsection("Player Count Validation")
        if 'team_known_players_count' in self.df.columns:
            player_counts = self.df['team_known_players_count'].value_counts().sort_index()
            print(f"  Player count distribution:")
            for count, freq in player_counts.items():
                print(f"    {count} players: {freq:,} rows ({freq/len(self.df)*100:.1f}%)")
            
            low_coverage = (self.df['team_known_players_count'] < 7).sum()
            if low_coverage > 0:
                print(f"  ⚠️ {low_coverage} rows have < 7 known players")
                self.warnings.append(f"{low_coverage} rows with low player coverage")
                score -= 0.5
            else:
                print(f"  ✓ All rows have adequate player coverage (≥7 players)")
        
        self.quality_scores['integrity'] = max(0, score)
    
    # =========================================================================
    # 3. FEATURE RELEVANCE
    # =========================================================================
    
    def audit_features(self):
        """Section 3: Feature Relevance Check"""
        self.section_header("FEATURE RELEVANCE", 3)
        
        score = 10.0
        
        # Define expected features
        critical_features = {
            'Player Stats': [
                'team_team_batting_avg', 'team_team_strike_rate', 
                'team_team_bowling_avg', 'team_team_economy',
                'team_elite_batsmen', 'team_star_batsmen',
                'team_elite_bowlers', 'team_star_bowlers'
            ],
            'Team Composition': [
                'team_all_rounder_count', 'team_wicketkeeper_count',
                'team_elite_players', 'team_star_players',
                'team_known_players_count'
            ],
            'Opposition Stats': [
                'opp_team_batting_avg', 'opp_team_bowling_avg',
                'opp_team_economy', 'opp_elite_players'
            ],
            'Match Context': [
                'venue', 'season_year', 'season_month',
                'toss_won', 'toss_decision_bat', 'gender'
            ],
            'Relative Features': [
                'batting_advantage', 'star_advantage', 'elite_advantage'
            ]
        }
        
        all_columns = set(self.df.columns)
        
        for category, features in critical_features.items():
            self.subsection(category)
            missing = [f for f in features if f not in all_columns]
            present = [f for f in features if f in all_columns]
            
            print(f"  Present: {len(present)}/{len(features)}")
            
            if missing:
                print(f"  ⚠️ Missing features:")
                for f in missing:
                    print(f"    - {f}")
                self.warnings.append(f"Missing {len(missing)} features in {category}")
                score -= min(2.0, len(missing) * 0.5)
            else:
                print(f"  ✓ All {category} features present!")
        
        # Feature summary
        self.subsection("Feature Summary")
        team_features = [col for col in self.df.columns if col.startswith('team_')]
        opp_features = [col for col in self.df.columns if col.startswith('opp_')]
        
        print(f"  Team features: {len(team_features)}")
        print(f"  Opposition features: {len(opp_features)}")
        print(f"  Total feature columns: {len(self.df.columns)}")
        
        self.quality_scores['features'] = max(0, score)
    
    # =========================================================================
    # 4. STATISTICAL CHECKS
    # =========================================================================
    
    def audit_statistics(self):
        """Section 4: Statistical Analysis"""
        self.section_header("STATISTICAL CHECKS", 4)
        
        score = 10.0
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Descriptive statistics
        self.subsection("Descriptive Statistics (Key Features)")
        key_features = [
            'total_runs', 'team_team_batting_avg', 'team_team_strike_rate',
            'team_team_bowling_avg', 'team_team_economy', 'team_star_players'
        ]
        
        for col in key_features:
            if col in self.df.columns:
                print(f"\n  {col}:")
                print(f"    Mean: {self.df[col].mean():.2f}")
                print(f"    Median: {self.df[col].median():.2f}")
                print(f"    Std: {self.df[col].std():.2f}")
                print(f"    Min: {self.df[col].min():.2f}")
                print(f"    Max: {self.df[col].max():.2f}")
                print(f"    Skewness: {self.df[col].skew():.2f}")
        
        # Outlier detection
        self.subsection("Outlier Detection (IQR Method)")
        outlier_counts = {}
        
        for col in numeric_cols:
            if col != 'match_id':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_counts[col] = outliers
        
        if outlier_counts:
            print(f"  Found outliers in {len(outlier_counts)} features:")
            sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for col, count in sorted_outliers:
                pct = (count / len(self.df)) * 100
                print(f"    {col}: {count:,} ({pct:.1f}%)")
            
            if max(outlier_counts.values()) / len(self.df) > 0.1:
                self.warnings.append("Significant outliers detected (>10% in some features)")
                score -= 1.0
        else:
            print(f"  ✓ No extreme outliers detected!")
        
        # Correlation analysis
        self.subsection("Correlation with Target (total_runs)")
        if 'total_runs' in self.df.columns:
            correlations = self.df[numeric_cols].corrwith(self.df['total_runs']).abs().sort_values(ascending=False)
            
            print(f"  Top 15 correlated features:")
            for i, (col, corr) in enumerate(correlations.head(15).items(), 1):
                if col != 'total_runs':
                    emoji = "🔴" if corr > 0.3 else "🟡" if corr > 0.1 else "⚪"
                    print(f"    {emoji} {col}: {corr:.3f}")
            
            # Check if we have meaningful correlations
            strong_corr = (correlations > 0.2).sum() - 1  # Exclude total_runs itself
            if strong_corr < 5:
                self.warnings.append(f"Only {strong_corr} features have strong correlation (>0.2) with target")
                score -= 1.0
        
        # Distribution checks
        self.subsection("Distribution Analysis")
        highly_skewed = []
        for col in numeric_cols:
            if col != 'match_id':
                skewness = abs(self.df[col].skew())
                if skewness > 2:
                    highly_skewed.append((col, skewness))
        
        if highly_skewed:
            print(f"  ⚠️ {len(highly_skewed)} highly skewed features (|skew| > 2):")
            for col, skew in sorted(highly_skewed, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {col}: {skew:.2f}")
            self.recommendations.append("Consider log transformation for highly skewed features")
        else:
            print(f"  ✓ No highly skewed distributions")
        
        self.quality_scores['statistics'] = max(0, score)
    
    # =========================================================================
    # 5. CATEGORICAL CHECKS
    # =========================================================================
    
    def audit_categorical(self):
        """Section 5: Categorical Features Analysis"""
        self.section_header("CATEGORICAL CHECKS", 5)
        
        score = 10.0
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        self.subsection("Categorical Features Overview")
        print(f"  Total categorical columns: {len(categorical_cols)}")
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            print(f"\n  {col}:")
            print(f"    Unique values: {unique_count:,}")
            
            if unique_count <= 20:
                value_counts = self.df[col].value_counts()
                print(f"    Top 5 values:")
                for val, count in value_counts.head(5).items():
                    print(f"      {val}: {count:,} ({count/len(self.df)*100:.1f}%)")
            else:
                print(f"    High cardinality feature")
        
        # Cardinality analysis
        self.subsection("Cardinality Analysis")
        high_cardinality = {}
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count > 50:
                high_cardinality[col] = unique_count
        
        if high_cardinality:
            print(f"  High-cardinality features (>50 unique values):")
            for col, count in high_cardinality.items():
                print(f"    {col}: {count:,} unique values")
            
            self.recommendations.append("Use embeddings or target encoding for high-cardinality features")
        else:
            print(f"  ✓ No high-cardinality features")
        
        # Encoding recommendations
        self.subsection("Encoding Strategy Recommendations")
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            
            if unique_count <= 5:
                print(f"  {col}: One-Hot Encoding (low cardinality)")
            elif unique_count <= 20:
                print(f"  {col}: Label Encoding or One-Hot")
            elif unique_count <= 100:
                print(f"  {col}: Target Encoding recommended")
            else:
                print(f"  {col}: Embeddings or Target Encoding strongly recommended")
        
        self.quality_scores['categorical'] = max(0, score)
    
    # =========================================================================
    # 6. TARGET VARIABLE VALIDATION
    # =========================================================================
    
    def audit_target(self):
        """Section 6: Target Variable Deep Dive"""
        self.section_header("TARGET VARIABLE VALIDATION", 6)
        
        score = 10.0
        
        if 'total_runs' not in self.df.columns:
            print("  ✗ Target variable 'total_runs' not found!")
            self.issues.append("Missing target variable")
            self.quality_scores['target'] = 0
            return
        
        target = self.df['total_runs']
        
        # Distribution
        self.subsection("Distribution Analysis")
        print(f"  Mean: {target.mean():.2f}")
        print(f"  Median: {target.median():.2f}")
        print(f"  Std Dev: {target.std():.2f}")
        print(f"  Min: {target.min()}")
        print(f"  Max: {target.max()}")
        print(f"  Skewness: {target.skew():.3f}")
        print(f"  Kurtosis: {target.kurtosis():.3f}")
        
        # Realistic ODI range check
        self.subsection("Realistic ODI Score Range")
        very_low = (target < 100).sum()
        low = ((target >= 100) & (target < 150)).sum()
        normal = ((target >= 150) & (target < 350)).sum()
        high = ((target >= 350) & (target < 400)).sum()
        very_high = (target >= 400).sum()
        
        print(f"  Very Low (<100): {very_low:,} ({very_low/len(self.df)*100:.1f}%)")
        print(f"  Low (100-150): {low:,} ({low/len(self.df)*100:.1f}%)")
        print(f"  Normal (150-350): {normal:,} ({normal/len(self.df)*100:.1f}%)")
        print(f"  High (350-400): {high:,} ({high/len(self.df)*100:.1f}%)")
        print(f"  Very High (≥400): {very_high:,} ({very_high/len(self.df)*100:.1f}%)")
        
        if normal / len(self.df) < 0.5:
            self.warnings.append("Less than 50% of scores in normal ODI range (150-350)")
            score -= 1.0
        
        unrealistic_low = (target < 50).sum()
        if unrealistic_low > 0:
            print(f"  ⚠️ {unrealistic_low} matches with <50 runs (likely all-out or abandoned)")
            self.warnings.append(f"{unrealistic_low} unrealistic low scores")
        
        # Balance across categories
        if 'venue' in self.df.columns:
            self.subsection("Target Balance Across Venues")
            venue_stats = self.df.groupby('venue')['total_runs'].agg(['mean', 'count']).sort_values('count', ascending=False)
            print(f"  Top 10 venues by match count:")
            print(venue_stats.head(10).to_string())
        
        if 'team' in self.df.columns:
            self.subsection("Target Balance Across Teams")
            team_stats = self.df.groupby('team')['total_runs'].agg(['mean', 'count']).sort_values('count', ascending=False)
            print(f"  Top 10 teams by match count:")
            print(team_stats.head(10).to_string())
        
        if 'season_year' in self.df.columns:
            self.subsection("Target Balance Across Years")
            year_stats = self.df.groupby('season_year')['total_runs'].agg(['mean', 'count']).sort_values('season_year')
            print(f"  Scores by year:")
            print(year_stats.to_string())
        
        self.quality_scores['target'] = max(0, score)
    
    # =========================================================================
    # 7. DATA LEAKAGE & SPLIT READINESS
    # =========================================================================
    
    def audit_leakage(self):
        """Section 7: Data Leakage Detection"""
        self.section_header("DATA LEAKAGE & SPLIT READINESS", 7)
        
        score = 10.0
        
        # Check for potential leakage columns
        self.subsection("Leakage Detection")
        
        suspicious_keywords = [
            'final', 'result', 'winner', 'outcome', 'cumulative',
            'wickets_lost', 'overs_completed', 'current_score'
        ]
        
        suspicious_cols = []
        for col in self.df.columns:
            for keyword in suspicious_keywords:
                if keyword in col.lower():
                    suspicious_cols.append(col)
                    break
        
        if suspicious_cols:
            print(f"  ⚠️ Found {len(suspicious_cols)} potentially leaking columns:")
            for col in suspicious_cols:
                print(f"    - {col}")
            self.issues.append(f"Potential data leakage in {len(suspicious_cols)} columns")
            score -= 3.0
        else:
            print(f"  ✓ No obvious leakage columns detected")
        
        # Train-test split readiness
        self.subsection("Train-Test Split Readiness")
        
        if 'date' in self.df.columns:
            print(f"  ✓ 'date' column present - chronological split possible")
            
            # Check temporal distribution
            if 'season_year' in self.df.columns:
                year_counts = self.df['season_year'].value_counts().sort_index()
                print(f"\n  Matches per year:")
                for year, count in year_counts.items():
                    print(f"    {year}: {count:,} matches")
                
                # Recommend split
                years = sorted(year_counts.index)
                if len(years) >= 3:
                    split_year = years[-int(len(years) * 0.2)]
                    test_count = self.df[self.df['season_year'] >= split_year].shape[0]
                    print(f"\n  Recommended split: Train on <{split_year}, Test on ≥{split_year}")
                    print(f"    Train size: ~{len(self.df) - test_count:,} rows")
                    print(f"    Test size: ~{test_count:,} rows")
        else:
            print(f"  ⚠️ No 'date' column - random split required")
            self.warnings.append("Missing date column for temporal split")
            score -= 1.0
        
        if 'match_id' in self.df.columns:
            print(f"\n  ✓ 'match_id' column present - can prevent match leakage")
            self.recommendations.append("Use GroupShuffleSplit on match_id to prevent same match in train/test")
        else:
            print(f"\n  ⚠️ No 'match_id' column")
            self.warnings.append("Missing match_id - risk of same match in train/test")
            score -= 1.0
        
        self.quality_scores['leakage'] = max(0, score)
    
    # =========================================================================
    # 8. MODEL SUITABILITY
    # =========================================================================
    
    def audit_model_readiness(self):
        """Section 8: Model Suitability Assessment"""
        self.section_header("MODEL SUITABILITY ASSESSMENT", 8)
        
        # Tree-based models
        self.subsection("Tree-Based Models (Random Forest, XGBoost)")
        
        rows = len(self.df)
        features = len(self.df.columns) - 1  # Exclude target
        ratio = rows / features
        
        print(f"  Sample-to-feature ratio: {ratio:.1f}")
        
        if ratio >= 50:
            print(f"  ✓ Excellent ratio for tree models (≥50)")
            tree_score = 10.0
        elif ratio >= 30:
            print(f"  ✓ Good ratio for tree models (≥30)")
            tree_score = 8.0
        elif ratio >= 20:
            print(f"  ⚠️ Acceptable ratio (≥20), but consider feature selection")
            tree_score = 6.0
        else:
            print(f"  ⚠️ Low ratio (<20), risk of overfitting")
            tree_score = 4.0
        
        print(f"\n  Feature diversity:")
        numeric_features = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(self.df.select_dtypes(include=['object']).columns)
        print(f"    Numeric: {numeric_features}")
        print(f"    Categorical: {categorical_features}")
        print(f"  ✓ Tree models handle both well")
        
        print(f"\n  Readiness Score: {tree_score}/10")
        
        # Neural Networks
        self.subsection("Neural Networks (DNN, TabNet)")
        
        print(f"  Sample size: {rows:,}")
        
        if rows >= 10000:
            print(f"  ✓ Excellent sample size for neural networks (≥10k)")
            nn_score = 10.0
        elif rows >= 5000:
            print(f"  ✓ Good sample size (≥5k)")
            nn_score = 8.0
        elif rows >= 2000:
            print(f"  ⚠️ Adequate but not ideal (≥2k)")
            nn_score = 6.0
        else:
            print(f"  ⚠️ Small sample size for NNs (<2k)")
            nn_score = 4.0
        
        print(f"\n  Categorical encoding needed:")
        if categorical_features > 0:
            print(f"    {categorical_features} categorical features require encoding")
            print(f"    Recommend: Target encoding or embeddings")
        
        print(f"\n  Feature scaling needed:")
        print(f"    Numeric features require standardization/normalization")
        
        print(f"\n  Readiness Score: {nn_score}/10")
        
        self.quality_scores['model_readiness'] = (tree_score + nn_score) / 2
        
        # Overall recommendation
        self.subsection("Model Recommendation")
        
        if tree_score >= nn_score:
            print(f"  🏆 Recommend: Tree-based models (Random Forest, XGBoost)")
            print(f"     Reason: Better suited for current sample size and feature mix")
        else:
            print(f"  🏆 Recommend: Neural Networks")
            print(f"     Reason: Large enough sample size, can capture complex patterns")
    
    # =========================================================================
    # 9. FINAL REPORT
    # =========================================================================
    
    def generate_final_report(self):
        """Section 9: Final Audit Report"""
        self.section_header("FINAL AUDIT REPORT", 9)
        
        # Calculate overall score
        avg_score = np.mean(list(self.quality_scores.values()))
        
        self.subsection("Quality Scores by Category")
        for category, score in self.quality_scores.items():
            emoji = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
            print(f"  {emoji} {category.capitalize()}: {score:.1f}/10")
        
        print(f"\n  {'='*50}")
        overall_emoji = "🟢" if avg_score >= 8 else "🟡" if avg_score >= 6 else "🔴"
        print(f"  {overall_emoji} OVERALL DATA QUALITY: {avg_score:.1f}/10")
        print(f"  {'='*50}")
        
        # Issues summary
        self.subsection("Critical Issues")
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"  ❌ {i}. {issue}")
        else:
            print(f"  ✓ No critical issues found!")
        
        # Warnings summary
        self.subsection("Warnings")
        if self.warnings:
            for i, warning in enumerate(self.warnings, 1):
                print(f"  ⚠️ {i}. {warning}")
        else:
            print(f"  ✓ No warnings!")
        
        # Recommendations
        self.subsection("Preprocessing Recommendations")
        
        # Auto-generate recommendations
        auto_recommendations = []
        
        if 'venue' in self.df.columns:
            auto_recommendations.append("Encode 'venue' using target encoding or embeddings")
        
        if 'team' in self.df.columns:
            auto_recommendations.append("Encode 'team' and 'opposition' using target encoding")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            auto_recommendations.append("Standardize numeric features (StandardScaler or MinMaxScaler)")
        
        if 'date' in self.df.columns:
            auto_recommendations.append("Use chronological split for train/test (80/20 by date)")
        
        if 'match_id' in self.df.columns:
            auto_recommendations.append("Use GroupShuffleSplit on match_id to prevent leakage")
        
        all_recommendations = auto_recommendations + self.recommendations
        
        for i, rec in enumerate(all_recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Training readiness verdict
        self.subsection("Training Readiness Verdict")
        
        if avg_score >= 8 and not self.issues:
            print(f"\n  ✅ READY FOR TRAINING")
            print(f"     - High quality dataset")
            print(f"     - No critical issues")
            print(f"     - Proceed with model training")
        elif avg_score >= 6:
            print(f"\n  ⚠️ PROCEED WITH CAUTION")
            print(f"     - Dataset is usable but has some issues")
            print(f"     - Address warnings before training")
            print(f"     - Monitor model performance closely")
        else:
            print(f"\n  ❌ NOT READY FOR TRAINING")
            print(f"     - Critical issues need resolution")
            print(f"     - Fix issues before proceeding")
        
        # Next steps
        print(f"\n  📋 Next Steps:")
        print(f"     1. Address critical issues (if any)")
        print(f"     2. Apply recommended preprocessing")
        print(f"     3. Create train/test split (chronological)")
        print(f"     4. Train baseline models (Random Forest, XGBoost)")
        print(f"     5. Validate player impact is measurable")
        print(f"     6. Test 'what-if' scenarios (player swaps)")
    
    def run_complete_audit(self):
        """Run all audit checks"""
        self.load_dataset()
        self.audit_overview()
        self.audit_integrity()
        self.audit_features()
        self.audit_statistics()
        self.audit_categorical()
        self.audit_target()
        self.audit_leakage()
        self.audit_model_readiness()
        self.generate_final_report()
        
        print("\n" + "="*80)
        print("AUDIT COMPLETE")
        print("="*80 + "\n")

def main():
    """Main execution"""
    dataset_path = '../data/odi_training_dataset.csv'
    
    auditor = ODIDatasetAuditor(dataset_path)
    auditor.run_complete_audit()

if __name__ == "__main__":
    main()

