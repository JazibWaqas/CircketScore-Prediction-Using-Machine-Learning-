#!/usr/bin/env python3
"""
Create a new scaler for the final trained models
This scaler will work with raw features and normalize them properly
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def create_new_scaler():
    """Create a scaler for the new models"""
    print("🏏 CREATING NEW SCALER FOR FINAL TRAINED MODELS")
    print("=" * 60)
    
    # Load the cleaned dataset
    try:
        df = pd.read_csv('processed_data/cleaned_cricket_dataset.csv')
        print(f"✅ Loaded cleaned dataset: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'total_runs']
    X = df[feature_columns]
    
    print(f"✅ Features to scale: {len(feature_columns)}")
    print(f"✅ Sample size: {X.shape[0]:,}")
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    print(f"✅ Scaler created and fitted")
    print(f"   Feature names: {len(scaler.feature_names_in_)} features")
    
    # Test the scaler
    sample_data = X.head(5)
    scaled_data = scaler.transform(sample_data)
    
    print(f"✅ Scaler test successful")
    print(f"   Original shape: {sample_data.shape}")
    print(f"   Scaled shape: {scaled_data.shape}")
    print(f"   Sample scaled values: {scaled_data[0][:5]}")
    
    # Save the scaler
    scaler_path = 'models/final_trained_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Also save feature names for reference
    feature_names_path = 'models/final_trained_feature_names.pkl'
    joblib.dump(feature_columns, feature_names_path)
    print(f"✅ Feature names saved to: {feature_names_path}")
    
    print("\n🎉 NEW SCALER CREATED SUCCESSFULLY!")
    print("Now the API can use this scaler to normalize features properly.")

if __name__ == "__main__":
    create_new_scaler()
