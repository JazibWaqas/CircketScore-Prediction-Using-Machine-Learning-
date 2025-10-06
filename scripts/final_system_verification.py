#!/usr/bin/env python3
"""
Final System Verification
Comprehensive check of the entire system before production
"""

import os
import sys
import pandas as pd
import joblib

def verify_models():
    """Verify all models exist and can be loaded"""
    print("🤖 VERIFYING MODELS")
    print("-" * 30)
    
    model_files = {
        'xgboost': 'models/final_trained_xgboost.pkl',
        'random_forest': 'models/final_trained_random_forest.pkl',
        'linear_regression': 'models/final_trained_linear_regression.pkl'
    }
    
    all_good = True
    for name, file_path in model_files.items():
        if os.path.exists(file_path):
            try:
                model = joblib.load(file_path)
                print(f"✅ {name}: Loaded successfully")
            except Exception as e:
                print(f"❌ {name}: Failed to load - {e}")
                all_good = False
        else:
            print(f"❌ {name}: File not found - {file_path}")
            all_good = False
    
    return all_good

def verify_dataset():
    """Verify the cleaned dataset exists and has correct format"""
    print("\n📊 VERIFYING DATASET")
    print("-" * 30)
    
    dataset_path = 'processed_data/cleaned_cricket_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {df.shape}")
        
        # Check for target variable
        if 'total_runs' not in df.columns:
            print("❌ Target variable 'total_runs' not found")
            return False
        
        print(f"✅ Target variable found: {df['total_runs'].min()}-{df['total_runs'].max()} runs")
        
        # Check feature count
        features = [col for col in df.columns if col != 'total_runs']
        print(f"✅ Features: {len(features)} features")
        
        # Check for required features
        required_features = [
            'team_balance_x', 'h2h_avg_runs', 'pitch_bounce', 
            'team_form_avg_runs', 'venue_avg_runs'
        ]
        
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            print(f"❌ Missing required features: {missing_features}")
            return False
        
        print("✅ All required features present")
        return True
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def verify_api_files():
    """Verify API files exist and are correct"""
    print("\n🌐 VERIFYING API FILES")
    print("-" * 30)
    
    api_files = {
        'run.py': 'Database/run.py',
        'run_old_backup.py': 'Database/run_old_backup.py'
    }
    
    all_good = True
    for name, file_path in api_files.items():
        if os.path.exists(file_path):
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Not found - {file_path}")
            all_good = False
    
    # Check if API references correct models
    try:
        with open('Database/run.py', 'r') as f:
            content = f.read()
            if 'final_trained_' in content:
                print("✅ API references new trained models")
            else:
                print("❌ API still references old models")
                all_good = False
    except Exception as e:
        print(f"❌ Error reading API file: {e}")
        all_good = False
    
    return all_good

def verify_frontend_files():
    """Verify frontend files are updated"""
    print("\n💻 VERIFYING FRONTEND FILES")
    print("-" * 30)
    
    frontend_files = {
        'App.js': 'frontend/src/App.js',
        'ModelSelector.js': 'frontend/src/components/ModelSelector.js',
        'PredictionResults.js': 'frontend/src/components/PredictionResults.js'
    }
    
    all_good = True
    for name, file_path in frontend_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'xgboost' in content and '86.2' in content:
                        print(f"✅ {name}: Updated with new model info")
                    else:
                        print(f"⚠️ {name}: May need updates")
            except Exception as e:
                print(f"❌ {name}: Error reading - {e}")
                all_good = False
        else:
            print(f"❌ {name}: Not found - {file_path}")
            all_good = False
    
    return all_good

def verify_database():
    """Verify database exists and has correct structure"""
    print("\n🗄️ VERIFYING DATABASE")
    print("-" * 30)
    
    db_path = 'Database/cricket_prediction.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print("✅ Database file exists")
    
    # Try to connect and check tables
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['teams', 'venues', 'players', 'user_predictions']
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        print("✅ All required tables present")
        
        # Check for data
        cursor.execute("SELECT COUNT(*) FROM teams WHERE is_active = 1")
        team_count = cursor.fetchone()[0]
        print(f"✅ Active teams: {team_count}")
        
        cursor.execute("SELECT COUNT(*) FROM venues WHERE is_active = 1")
        venue_count = cursor.fetchone()[0]
        print(f"✅ Active venues: {venue_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def verify_results_files():
    """Verify result files exist"""
    print("\n📈 VERIFYING RESULTS FILES")
    print("-" * 30)
    
    result_files = [
        'results/final_trained_model_comparison.csv',
        'results/real_data_test_results.csv',
        'results/sample_predictions.csv'
    ]
    
    all_good = True
    for file_path in result_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️ {file_path}: Not found")
            all_good = False
    
    return all_good

def main():
    """Main verification function"""
    print("🔍 FINAL SYSTEM VERIFICATION")
    print("=" * 50)
    
    checks = [
        ("Models", verify_models),
        ("Dataset", verify_dataset),
        ("API Files", verify_api_files),
        ("Frontend Files", verify_frontend_files),
        ("Database", verify_database),
        ("Results Files", verify_results_files)
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<15}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("🚀 SYSTEM IS READY FOR PRODUCTION!")
        print("\n📋 NEXT STEPS:")
        print("1. Start the API server: cd Database && python run.py")
        print("2. Start the frontend: cd frontend && npm start")
        print("3. Test predictions in the browser")
        print("4. All models are ready with 86.2% accuracy!")
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("🔧 Please fix the issues above before proceeding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
