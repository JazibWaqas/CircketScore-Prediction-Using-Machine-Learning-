#!/usr/bin/env python3
"""
Refined Dataset Cleaning Process
Creates cleaned_cricket_dataset.csv with proper feature selection and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def refined_dataset_cleaning():
    """Refined dataset cleaning with careful feature selection"""
    print("🧹 REFINED DATASET CLEANING PROCESS")
    print("=" * 60)
    
    # Load the original training dataset
    try:
        df = pd.read_csv('processed_data/final_training_dataset.csv')
        print(f"✅ Loaded original dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"\n📊 STEP 1: INITIAL DATASET OVERVIEW")
    print(f"  Original shape: {df.shape}")
    print(f"  Target variable: total_runs")
    print(f"  Target range: {df['total_runs'].min()} - {df['total_runs'].max()}")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    print(f"\n🎯 STEP 2: TARGET CLEANING")
    # Remove unrealistic scores
    original_count = len(df_clean)
    
    # Remove matches with unrealistic scores
    df_clean = df_clean[(df_clean['total_runs'] >= 20) & (df_clean['total_runs'] <= 250)]
    removed_count = original_count - len(df_clean)
    
    print(f"  Removed {removed_count} matches with unrealistic scores (<20 or >250 runs)")
    print(f"  New shape: {df_clean.shape}")
    print(f"  Target range after cleaning: {df_clean['total_runs'].min()} - {df_clean['total_runs'].max()}")
    
    print(f"\n🔗 STEP 3: CAREFUL CORRELATION HANDLING")
    
    # Get numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_runs' in numeric_cols:
        numeric_cols.remove('total_runs')  # Remove target from correlation analysis
    
    # Calculate correlation matrix
    corr_matrix = df_clean[numeric_cols].corr()
    
    # Find highly correlated pairs (>0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > 0.9:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    print(f"  Found {len(high_corr_pairs)} highly correlated pairs (>0.9)")
    
    # Define features to drop (redundant/derived versions)
    features_to_drop = []
    
    # Drop redundant math-based versions, keep interpretable ones
    redundant_features = {
        'venue_difficulty': 'venue_avg_runs',  # Keep avg_runs, drop difficulty
        'team_form_score': 'team_form_avg_runs',  # Keep avg_runs, drop score
        'h2h_strength': 'h2h_avg_runs',  # Keep avg_runs, drop strength
        'weather_impact': 'temperature',  # Keep temperature, drop derived impact
        'pitch_difficulty': 'pitch_bounce',  # Keep bounce, drop difficulty
        'composition_effectiveness': 'team_depth'  # Keep depth, drop effectiveness
    }
    
    # Drop redundant features
    for redundant, keep in redundant_features.items():
        if redundant in df_clean.columns:
            features_to_drop.append(redundant)
            print(f"  🗑️ Dropping {redundant} (keeping {keep})")
    
    # Handle perfectly correlated pitch features
    pitch_features = ['pitch_pace', 'pitch_turn']
    if all(f in df_clean.columns for f in pitch_features):
        # Keep pitch_bounce as the primary pitch feature
        features_to_drop.extend(['pitch_pace', 'pitch_turn'])
        print(f"  🗑️ Dropping pitch_pace, pitch_turn (keeping pitch_bounce)")
    
    # Handle team balance redundancy
    balance_features = ['team_balance_y', 'bowling_ratio']
    if all(f in df_clean.columns for f in balance_features):
        # Keep batting_ratio as primary balance indicator
        features_to_drop.extend(['team_balance_y', 'bowling_ratio'])
        print(f"  🗑️ Dropping team_balance_y, bowling_ratio (keeping batting_ratio)")
    
    # Remove redundant features
    df_clean = df_clean.drop(columns=features_to_drop)
    print(f"  Features dropped: {len(features_to_drop)}")
    print(f"  New shape: {df_clean.shape}")
    
    print(f"\n📏 STEP 4: OUTLIER HANDLING (CLIPPING)")
    
    # Get numeric columns after dropping redundant features
    numeric_cols_clean = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_runs' in numeric_cols_clean:
        numeric_cols_clean.remove('total_runs')
    
    # Clip outliers using 1st-99th percentile
    clipped_features = []
    for col in numeric_cols_clean:
        lower_bound = df_clean[col].quantile(0.01)
        upper_bound = df_clean[col].quantile(0.99)
        
        original_min = df_clean[col].min()
        original_max = df_clean[col].max()
        
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        clipped_count = ((df_clean[col] != original_min) & (df_clean[col] != original_max)).sum()
        if clipped_count > 0:
            clipped_features.append(col)
    
    print(f"  Clipped outliers in {len(clipped_features)} features")
    if clipped_features:
        print(f"  Clipped features: {clipped_features[:5]}...")
    
    print(f"\n🏷️ STEP 5: CATEGORICAL ENCODING")
    
    # Identify categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Label encode high-cardinality categoricals
    label_encoded_features = []
    for col in categorical_cols:
        if df_clean[col].nunique() > 10:  # High cardinality
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoded_features.append(col)
    
    print(f"  Label encoded: {len(label_encoded_features)} features")
    print(f"  Label encoded features: {label_encoded_features}")
    
    # One-hot encode low-cardinality categoricals
    onehot_encoded_features = []
    for col in categorical_cols:
        if col not in label_encoded_features and df_clean[col].nunique() <= 10:
            # Create dummy variables
            dummies = pd.get_dummies(df_clean[col], prefix=col)
            df_clean = pd.concat([df_clean.drop(columns=[col]), dummies], axis=1)
            onehot_encoded_features.append(col)
    
    print(f"  One-hot encoded: {len(onehot_encoded_features)} features")
    print(f"  One-hot encoded features: {onehot_encoded_features}")
    
    print(f"\n📊 STEP 6: FEATURE SCALING")
    
    # Get final numeric columns for scaling
    final_numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_runs' in final_numeric_cols:
        final_numeric_cols.remove('total_runs')
    
    # Apply scaling to continuous numeric features
    scaler = StandardScaler()
    df_clean[final_numeric_cols] = scaler.fit_transform(df_clean[final_numeric_cols])
    
    print(f"  Scaled {len(final_numeric_cols)} numeric features")
    print(f"  Scaling method: StandardScaler (mean=0, std=1)")
    
    print(f"\n💾 STEP 7: SAVE CLEANED DATASET")
    
    # Save the cleaned dataset
    output_file = 'processed_data/cleaned_cricket_dataset.csv'
    df_clean.to_csv(output_file, index=False)
    
    print(f"  ✅ Saved cleaned dataset: {output_file}")
    print(f"  Final shape: {df_clean.shape}")
    
    print(f"\n📈 STEP 8: FINAL ANALYSIS")
    
    # Calculate final correlations
    final_numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'total_runs' in final_numeric_cols:
        final_numeric_cols.remove('total_runs')
    
    final_corr_matrix = df_clean[final_numeric_cols].corr()
    
    # Find remaining high correlations
    remaining_high_corr = []
    for i in range(len(final_corr_matrix.columns)):
        for j in range(i+1, len(final_corr_matrix.columns)):
            corr_value = abs(final_corr_matrix.iloc[i, j])
            if corr_value > 0.8:
                remaining_high_corr.append({
                    'feature1': final_corr_matrix.columns[i],
                    'feature2': final_corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"  Original shape: {df.shape}")
    print(f"  Final shape: {df_clean.shape}")
    print(f"  Features removed: {df.shape[1] - df_clean.shape[1]}")
    print(f"  Samples removed: {df.shape[0] - df_clean.shape[0]}")
    print(f"  Remaining features: {df_clean.shape[1]}")
    print(f"  Remaining high correlations (>0.8): {len(remaining_high_corr)}")
    
    if remaining_high_corr:
        print(f"\n⚠️ Remaining high correlations:")
        for pair in remaining_high_corr[:5]:  # Show top 5
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    
    # Target variable analysis
    target_stats = {
        'mean': df_clean['total_runs'].mean(),
        'std': df_clean['total_runs'].std(),
        'min': df_clean['total_runs'].min(),
        'max': df_clean['total_runs'].max(),
        'skewness': df_clean['total_runs'].skew()
    }
    
    print(f"\n🎯 TARGET VARIABLE (total_runs):")
    print(f"  Mean: {target_stats['mean']:.2f}")
    print(f"  Std: {target_stats['std']:.2f}")
    print(f"  Range: {target_stats['min']:.0f} - {target_stats['max']:.0f}")
    print(f"  Skewness: {target_stats['skewness']:.3f}")
    
    # Feature categories
    feature_categories = {
        'numeric_features': len(df_clean.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df_clean.select_dtypes(include=['object']).columns),
        'total_features': len(df_clean.columns)
    }
    
    print(f"\n🏷️ FEATURE CATEGORIES:")
    print(f"  Numeric features: {feature_categories['numeric_features']}")
    print(f"  Categorical features: {feature_categories['categorical_features']}")
    print(f"  Total features: {feature_categories['total_features']}")
    
    print(f"\n✅ CLEANING COMPLETE!")
    print(f"  Dataset ready for deep learning training")
    print(f"  All domain-relevant features preserved")
    print(f"  Redundant features removed")
    print(f"  Outliers handled")
    print(f"  Proper encoding applied")
    
    return {
        'original_shape': df.shape,
        'final_shape': df_clean.shape,
        'features_removed': df.shape[1] - df_clean.shape[1],
        'samples_removed': df.shape[0] - df_clean.shape[0],
        'remaining_correlations': len(remaining_high_corr),
        'target_stats': target_stats,
        'output_file': output_file
    }

if __name__ == "__main__":
    results = refined_dataset_cleaning()
    
    print(f"\n" + "="*60)
    print("REFINED DATASET CLEANING COMPLETE")
    print("="*60)
    
    print(f"✅ Cleaned dataset saved: {results['output_file']}")
    print(f"📊 Shape: {results['final_shape']}")
    print(f"🗑️ Features removed: {results['features_removed']}")
    print(f"🗑️ Samples removed: {results['samples_removed']}")
    print(f"🔗 Remaining correlations: {results['remaining_correlations']}")
    
    print(f"\n🚀 READY FOR DEEP LEARNING TRAINING!")
