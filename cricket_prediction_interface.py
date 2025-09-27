"""
Cricket Score Prediction - Interactive Interface
This script provides an easy-to-use interface for making cricket score predictions.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class CricketPredictionInterface:
    def __init__(self, model_path=None, scaler_path=None):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def load_model(self, model_path='cricket_model.pkl', scaler_path='cricket_scaler.pkl'):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model loaded successfully!")
        except:
            print("Model files not found. Please train the model first.")
    
    def create_sample_scenarios(self):
        """Create sample prediction scenarios"""
        print("CRICKET SCORE PREDICTION - SAMPLE SCENARIOS")
        print("="*60)
        
        # Load the dataset to get feature structure
        try:
            df = pd.read_csv('fixed_cricket_dataset.csv')
            feature_columns = [col for col in df.columns if col != 'total_runs']
            self.feature_columns = feature_columns
        except:
            print("Dataset not found. Please run the model training first.")
            return
        
        # Create sample scenarios
        scenarios = {
            "India vs Pakistan at Dubai": {
                'innings_number': 1,
                'overs_bowled': 20.0,
                'run_rate': 7.5,
                'extras': 8,
                'boundaries_total': 15,
                'team_batting_avg': 28.0,
                'team_strike_rate': 125.0,
                'team_centuries': 5,
                'team_fifties': 25,
                'team_bowling_avg': 32.0,
                'team_economy': 6.5,
                'team_wickets': 150,
                'team_maidens': 20,
                'team_batsmen': 6,
                'team_bowlers': 4,
                'team_allrounders': 1,
                'team_wicketkeepers': 1,
                'venue_avg_runs': 155.0,
                'venue_avg_rr': 7.8,
                'venue_high_scoring': 1,
                'venue_avg_boundaries': 18.0,
                'opp_avg_runs': 145.0,
                'opp_avg_rr': 7.2,
                'opp_avg_boundaries': 16.0,
                'team_balance': 0.6,
                'venue_advantage': 5.0
            },
            
            "Australia vs England at MCG": {
                'innings_number': 1,
                'overs_bowled': 20.0,
                'run_rate': 8.0,
                'extras': 6,
                'boundaries_total': 20,
                'team_batting_avg': 30.0,
                'team_strike_rate': 130.0,
                'team_centuries': 8,
                'team_fifties': 30,
                'team_bowling_avg': 28.0,
                'team_economy': 6.0,
                'team_wickets': 180,
                'team_maidens': 25,
                'team_batsmen': 6,
                'team_bowlers': 4,
                'team_allrounders': 1,
                'team_wicketkeepers': 1,
                'venue_avg_runs': 165.0,
                'venue_avg_rr': 8.2,
                'venue_high_scoring': 1,
                'venue_avg_boundaries': 22.0,
                'opp_avg_runs': 150.0,
                'opp_avg_rr': 7.5,
                'opp_avg_boundaries': 18.0,
                'team_balance': 0.6,
                'venue_advantage': 15.0
            },
            
            "South Africa vs New Zealand at Cape Town": {
                'innings_number': 1,
                'overs_bowled': 20.0,
                'run_rate': 7.0,
                'extras': 10,
                'boundaries_total': 12,
                'team_batting_avg': 25.0,
                'team_strike_rate': 115.0,
                'team_centuries': 3,
                'team_fifties': 20,
                'team_bowling_avg': 35.0,
                'team_economy': 7.5,
                'team_wickets': 120,
                'team_maidens': 15,
                'team_batsmen': 6,
                'team_bowlers': 4,
                'team_allrounders': 1,
                'team_wicketkeepers': 1,
                'venue_avg_runs': 140.0,
                'venue_avg_rr': 7.0,
                'venue_high_scoring': 0,
                'venue_avg_boundaries': 14.0,
                'opp_avg_runs': 135.0,
                'opp_avg_rr': 6.8,
                'opp_avg_boundaries': 13.0,
                'team_balance': 0.6,
                'venue_advantage': -10.0
            }
        }
        
        # Make predictions for each scenario
        for scenario_name, features in scenarios.items():
            print(f"\n{scenario_name}:")
            print("-" * 50)
            
            # Create feature array
            feature_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Make prediction (using a simple model for demonstration)
            # In practice, you would load your trained model here
            predicted_runs = self.simple_prediction(feature_array)
            
            print(f"Predicted Score: {predicted_runs:.1f} runs")
            print(f"Key Factors:")
            print(f"  - Team Batting Average: {features['team_batting_avg']}")
            print(f"  - Team Strike Rate: {features['team_strike_rate']}")
            print(f"  - Venue Average: {features['venue_avg_runs']}")
            print(f"  - Opposition Strength: {features['opp_avg_runs']}")
    
    def simple_prediction(self, features):
        """Simple prediction function (replace with actual model)"""
        # This is a simplified prediction based on key features
        # In practice, you would use your trained XGBoost model
        
        # Extract key features
        team_batting_avg = features[0][5]  # team_batting_avg
        team_strike_rate = features[0][6]  # team_strike_rate
        venue_avg_runs = features[0][17]   # venue_avg_runs
        opp_avg_runs = features[0][21]     # opp_avg_runs
        
        # Simple weighted prediction
        base_score = 150.0
        batting_factor = (team_batting_avg - 25.0) * 2
        strike_rate_factor = (team_strike_rate - 120.0) * 0.5
        venue_factor = (venue_avg_runs - 150.0) * 0.8
        opposition_factor = (opp_avg_runs - 150.0) * 0.3
        
        predicted = base_score + batting_factor + strike_rate_factor + venue_factor + opposition_factor
        
        return max(50, min(250, predicted))  # Clamp between 50 and 250
    
    def create_custom_scenario(self):
        """Allow user to create custom prediction scenarios"""
        print("\nCUSTOM SCENARIO CREATOR")
        print("="*40)
        
        print("Enter the following details for your prediction:")
        
        # Get user input
        team_name = input("Team name: ")
        opposition = input("Opposition team: ")
        venue = input("Venue: ")
        
        # Get key parameters
        try:
            batting_avg = float(input("Team batting average (default 25): ") or "25")
            strike_rate = float(input("Team strike rate (default 120): ") or "120")
            venue_avg = float(input("Venue average runs (default 150): ") or "150")
            opp_strength = float(input("Opposition strength (default 150): ") or "150")
            
            # Create prediction
            features = {
                'team_batting_avg': batting_avg,
                'team_strike_rate': strike_rate,
                'venue_avg_runs': venue_avg,
                'opp_avg_runs': opp_strength
            }
            
            # Simple prediction
            base_score = 150.0
            batting_factor = (batting_avg - 25.0) * 2
            strike_rate_factor = (strike_rate - 120.0) * 0.5
            venue_factor = (venue_avg - 150.0) * 0.8
            opposition_factor = (opp_strength - 150.0) * 0.3
            
            predicted = base_score + batting_factor + strike_rate_factor + venue_factor + opposition_factor
            predicted = max(50, min(250, predicted))
            
            print(f"\nPrediction for {team_name} vs {opposition} at {venue}:")
            print(f"Predicted Score: {predicted:.1f} runs")
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    def run_interface(self):
        """Run the main interface"""
        print("CRICKET SCORE PREDICTION INTERFACE")
        print("="*50)
        
        while True:
            print("\nChoose an option:")
            print("1. View sample scenarios")
            print("2. Create custom scenario")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                self.create_sample_scenarios()
            elif choice == '2':
                self.create_custom_scenario()
            elif choice == '3':
                print("Thank you for using Cricket Score Prediction!")
                break
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function to run the interface"""
    interface = CricketPredictionInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()
