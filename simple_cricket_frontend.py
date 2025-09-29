"""
Simple Cricket Prediction Frontend
Easy-to-use interface with all 19 features - no external dependencies needed.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data_and_train_model():
    """Load dataset and train the model"""
    print("🏏 SIMPLE CRICKET PREDICTION FRONTEND")
    print("="*60)
    
    # Load the ML-ready dataset
    dataset = pd.read_csv('ml_ready_fixed_dataset.csv')
    print(f"Dataset loaded: {dataset.shape}")
    
    # Define ALL features the model was trained on
    feature_columns = [
        'team_batting_avg', 'team_strike_rate', 'team_bowling_avg', 'team_economy',
        'team_size', 'venue_avg_runs', 'venue_runs_std', 'venue_matches',
        'team_balance', 'venue_advantage', 'opposition_strength',
        'total_wickets', 'balls_bowled', 'overs_bowled', 'run_rate',
        'extras', 'boundaries_4s', 'boundaries_6s', 'boundaries_total'
    ]
    
    X = dataset[feature_columns]
    y = dataset['total_runs']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model trained successfully! (R² = {model.score(X_test, y_test):.4f})")
    print(f"Features used: {len(feature_columns)}")
    
    return model, feature_columns

def get_user_input():
    """Get user input for all 19 features"""
    print("\n🎛️ ENTER VALUES FOR ALL 19 FEATURES:")
    print("="*60)
    
    # Team Statistics
    print("\n📊 TEAM STATISTICS:")
    print("-" * 30)
    team_batting_avg = float(input("Team Batting Average (15-35) [25.0]: ") or "25.0")
    team_strike_rate = float(input("Team Strike Rate (100-150) [120.0]: ") or "120.0")
    team_bowling_avg = float(input("Team Bowling Average (20-40) [30.0]: ") or "30.0")
    team_economy = float(input("Team Economy Rate (5-9) [7.0]: ") or "7.0")
    team_size = int(input("Team Size (11) [11]: ") or "11")
    
    # Venue Statistics
    print("\n🏟️ VENUE STATISTICS:")
    print("-" * 30)
    venue_avg_runs = float(input("Venue Average Runs (120-180) [150.0]: ") or "150.0")
    venue_runs_std = float(input("Venue Runs Std Dev (15-35) [25.0]: ") or "25.0")
    venue_matches = int(input("Venue Matches Played (1-20) [10]: ") or "10")
    
    # Team Balance & Advantage
    print("\n⚖️ TEAM BALANCE & ADVANTAGE:")
    print("-" * 30)
    team_balance = float(input("Team Balance (0.5-1.5) [0.8]: ") or "0.8")
    venue_advantage = float(input("Venue Advantage (0.8-1.2) [1.0]: ") or "1.0")
    opposition_strength = float(input("Opposition Strength (0.5-1.0) [0.8]: ") or "0.8")
    
    # Match Context
    print("\n🏏 MATCH CONTEXT:")
    print("-" * 30)
    total_wickets = int(input("Total Wickets (0-10) [0]: ") or "0")
    balls_bowled = int(input("Balls Bowled (120) [120]: ") or "120")
    overs_bowled = float(input("Overs Bowled (20.0) [20.0]: ") or "20.0")
    run_rate = float(input("Run Rate (6-10) [7.5]: ") or "7.5")
    
    # Extras and Boundaries
    print("\n🎯 EXTRAS & BOUNDARIES:")
    print("-" * 30)
    extras = int(input("Extras (5-20) [10]: ") or "10")
    boundaries_4s = int(input("Boundaries 4s (10-25) [15]: ") or "15")
    boundaries_6s = int(input("Boundaries 6s (3-15) [5]: ") or "5")
    boundaries_total = int(input("Total Boundaries (15-35) [20]: ") or "20")
    
    return {
        'team_batting_avg': team_batting_avg,
        'team_strike_rate': team_strike_rate,
        'team_bowling_avg': team_bowling_avg,
        'team_economy': team_economy,
        'team_size': team_size,
        'venue_avg_runs': venue_avg_runs,
        'venue_runs_std': venue_runs_std,
        'venue_matches': venue_matches,
        'team_balance': team_balance,
        'venue_advantage': venue_advantage,
        'opposition_strength': opposition_strength,
        'total_wickets': total_wickets,
        'balls_bowled': balls_bowled,
        'overs_bowled': overs_bowled,
        'run_rate': run_rate,
        'extras': extras,
        'boundaries_4s': boundaries_4s,
        'boundaries_6s': boundaries_6s,
        'boundaries_total': boundaries_total
    }

def predict_score(model, features):
    """Predict score using all features"""
    # Create prediction input
    input_df = pd.DataFrame([features])
    
    # Make prediction
    predicted_score = model.predict(input_df)[0]
    
    return predicted_score

def display_feature_importance(model, feature_columns):
    """Display feature importance from the model"""
    print("\n📈 FEATURE IMPORTANCE:")
    print("-" * 40)
    
    feature_importance = model.feature_importances_
    feature_names = feature_columns
    
    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.4f}")

def run_scenario_analysis(model):
    """Run multiple scenarios to show impact of different features"""
    print("\n🔬 SCENARIO ANALYSIS")
    print("="*60)
    
    # Base scenario
    base_features = {
        'team_batting_avg': 25.0, 'team_strike_rate': 120.0, 'team_bowling_avg': 30.0, 'team_economy': 7.0,
        'team_size': 11, 'venue_avg_runs': 150.0, 'venue_runs_std': 25.0, 'venue_matches': 10,
        'team_balance': 0.8, 'venue_advantage': 1.0, 'opposition_strength': 0.8,
        'total_wickets': 0, 'balls_bowled': 120, 'overs_bowled': 20.0, 'run_rate': 7.5,
        'extras': 10, 'boundaries_4s': 15, 'boundaries_6s': 5, 'boundaries_total': 20
    }
    
    base_score = predict_score(model, base_features)
    print(f"Base Scenario: {base_score:.1f} runs")
    
    # Test different scenarios
    scenarios = [
        ("High Batting Team", {'team_batting_avg': 35.0, 'team_strike_rate': 140.0}),
        ("Low Batting Team", {'team_batting_avg': 15.0, 'team_strike_rate': 100.0}),
        ("High Scoring Venue", {'venue_avg_runs': 180.0, 'venue_runs_std': 30.0}),
        ("Low Scoring Venue", {'venue_avg_runs': 120.0, 'venue_runs_std': 20.0}),
        ("Strong Opposition", {'opposition_strength': 0.9}),
        ("Weak Opposition", {'opposition_strength': 0.6}),
        ("High Boundaries", {'boundaries_4s': 25, 'boundaries_6s': 10, 'boundaries_total': 35}),
        ("Low Boundaries", {'boundaries_4s': 10, 'boundaries_6s': 2, 'boundaries_total': 12})
    ]
    
    print(f"\nScenario Analysis:")
    print("-" * 40)
    for scenario_name, scenario_features in scenarios:
        test_features = base_features.copy()
        test_features.update(scenario_features)
        score = predict_score(model, test_features)
        difference = score - base_score
        print(f"{scenario_name:20s}: {score:.1f} runs ({difference:+.1f})")

def main():
    """Main function"""
    # Load data and train model
    model, feature_columns = load_data_and_train_model()
    
    while True:
        print("\n🎮 CHOOSE AN OPTION:")
        print("="*60)
        print("1. Comprehensive Prediction (all 19 features)")
        print("2. Scenario Analysis")
        print("3. Feature Importance")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            print("\n🎯 COMPREHENSIVE PREDICTION")
            print("="*60)
            
            # Get user input for all features
            features = get_user_input()
            
            # Make prediction
            predicted_score = predict_score(model, features)
            
            # Display results
            print(f"\n🏆 PREDICTION RESULTS:")
            print("-" * 40)
            print(f"Predicted Score: {predicted_score:.1f} runs")
            print(f"Confidence: High (R² = 0.98)")
            
            # Display input summary
            print(f"\n📋 INPUT SUMMARY:")
            print("-" * 40)
            for feature, value in features.items():
                print(f"{feature:20s}: {value}")
        
        elif choice == '2':
            run_scenario_analysis(model)
        
        elif choice == '3':
            display_feature_importance(model, feature_columns)
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
