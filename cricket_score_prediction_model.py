"""
Cricket Score Prediction - Machine Learning Model
This script builds and evaluates models to predict T20 cricket team scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class CricketScorePredictor:
    def __init__(self, dataset_path='ml_ready_cricket_dataset.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading and preparing dataset...")
        
        # Load dataset
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Display basic statistics
        print("\nDataset Overview:")
        print(f"Target variable (total_runs) statistics:")
        print(self.df['total_runs'].describe())
        
        # Check for missing values
        print(f"\nMissing values per column:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Prepare features and target
        feature_columns = [col for col in self.df.columns if col != 'total_runs']
        X = self.df[feature_columns]
        y = self.df['total_runs']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return X, y
    
    def build_baseline_model(self):
        """Build baseline Linear Regression model"""
        print("\nBuilding baseline Linear Regression model...")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred_train = lr_model.predict(self.X_train_scaled)
        y_pred_test = lr_model.predict(self.X_test_scaled)
        
        # Evaluate
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.models['Linear Regression'] = lr_model
        self.results['Linear Regression'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse)
        }
        
        print(f"Linear Regression Results:")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {np.sqrt(train_mse):.2f}, Test RMSE: {np.sqrt(test_mse):.2f}")
        
        return lr_model
    
    def build_random_forest_model(self):
        """Build Random Forest model"""
        print("\nBuilding Random Forest model...")
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(self.X_train)
        y_pred_test = rf_model.predict(self.X_test)
        
        # Evaluate
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse)
        }
        
        print(f"Random Forest Results:")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {np.sqrt(train_mse):.2f}, Test RMSE: {np.sqrt(test_mse):.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return rf_model, feature_importance
    
    def build_xgboost_model(self):
        """Build XGBoost model"""
        print("\nBuilding XGBoost model...")
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = xgb_model.predict(self.X_train)
        y_pred_test = xgb_model.predict(self.X_test)
        
        # Evaluate
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse)
        }
        
        print(f"XGBoost Results:")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {np.sqrt(train_mse):.2f}, Test RMSE: {np.sqrt(test_mse):.2f}")
        
        return xgb_model
    
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['train_r2', 'test_r2', 'train_rmse', 'test_rmse']]
        comparison_df.columns = ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE']
        
        print(comparison_df.round(4))
        
        # Find best model
        best_model = comparison_df['Test R²'].idxmax()
        print(f"\nBest Model: {best_model} (Test R²: {comparison_df.loc[best_model, 'Test R²']:.4f})")
        
        return comparison_df
    
    def create_prediction_interface(self):
        """Create a simple prediction interface"""
        print("\n" + "="*60)
        print("PREDICTION INTERFACE")
        print("="*60)
        
        # Use the best model for predictions
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_model = self.models[best_model_name]
        
        print(f"Using {best_model_name} for predictions")
        
        # Create sample predictions
        sample_indices = np.random.choice(self.X_test.index, 5, replace=False)
        sample_data = self.X_test.loc[sample_indices]
        
        if best_model_name == 'Linear Regression':
            sample_predictions = best_model.predict(self.scaler.transform(sample_data))
        else:
            sample_predictions = best_model.predict(sample_data)
        
        print("\nSample Predictions:")
        for i, (idx, pred, actual) in enumerate(zip(sample_indices, sample_predictions, self.y_test.loc[sample_indices])):
            print(f"Sample {i+1}: Predicted = {pred:.1f}, Actual = {actual:.1f}, Error = {abs(pred-actual):.1f}")
    
    def create_what_if_scenarios(self):
        """Create what-if scenario examples"""
        print("\n" + "="*60)
        print("WHAT-IF SCENARIO EXAMPLES")
        print("="*60)
        
        # Get the best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_model = self.models[best_model_name]
        
        # Create example scenarios
        print("Example scenarios (using average values):")
        
        # Get feature columns (excluding target)
        feature_columns = [col for col in self.df.columns if col != 'total_runs']
        
        # Scenario 1: Strong batting team vs weak bowling
        scenario1 = self.X_test.mean().copy()
        scenario1['team_batting_avg'] = 35.0  # High batting average
        scenario1['team_strike_rate'] = 140.0  # High strike rate
        scenario1['team_bowling_avg'] = 40.0  # Weak bowling
        scenario1['venue_avg_runs'] = 160.0  # High-scoring venue
        
        # Ensure we have the right number of features
        scenario1_array = scenario1[feature_columns].values.reshape(1, -1)
        
        if best_model_name == 'Linear Regression':
            pred1 = best_model.predict(self.scaler.transform(scenario1_array))[0]
        else:
            pred1 = best_model.predict(scenario1_array)[0]
        
        print(f"Scenario 1 - Strong batting vs weak bowling: {pred1:.1f} runs")
        
        # Scenario 2: Weak batting team vs strong bowling
        scenario2 = self.X_test.mean().copy()
        scenario2['team_batting_avg'] = 20.0  # Low batting average
        scenario2['team_strike_rate'] = 100.0  # Low strike rate
        scenario2['team_bowling_avg'] = 25.0  # Strong bowling
        scenario2['venue_avg_runs'] = 120.0  # Low-scoring venue
        
        scenario2_array = scenario2[feature_columns].values.reshape(1, -1)
        
        if best_model_name == 'Linear Regression':
            pred2 = best_model.predict(self.scaler.transform(scenario2_array))[0]
        else:
            pred2 = best_model.predict(scenario2_array)[0]
        
        print(f"Scenario 2 - Weak batting vs strong bowling: {pred2:.1f} runs")
        
        # Scenario 3: Balanced teams at neutral venue
        scenario3 = self.X_test.mean().copy()
        scenario3['team_batting_avg'] = 25.0  # Average batting
        scenario3['team_strike_rate'] = 120.0  # Average strike rate
        scenario3['team_bowling_avg'] = 30.0  # Average bowling
        scenario3['venue_avg_runs'] = 150.0  # Average venue
        
        scenario3_array = scenario3[feature_columns].values.reshape(1, -1)
        
        if best_model_name == 'Linear Regression':
            pred3 = best_model.predict(self.scaler.transform(scenario3_array))[0]
        else:
            pred3 = best_model.predict(scenario3_array)[0]
        
        print(f"Scenario 3 - Balanced teams at neutral venue: {pred3:.1f} runs")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("CRICKET SCORE PREDICTION - MACHINE LEARNING ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Build models
        self.build_baseline_model()
        rf_model, feature_importance = self.build_random_forest_model()
        self.build_xgboost_model()
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Create prediction interface
        self.create_prediction_interface()
        
        # Create what-if scenarios
        self.create_what_if_scenarios()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("The model is ready for cricket score prediction!")
        print("You can now use it to predict team scores for different scenarios.")
        
        return self.models, self.results, comparison_df

def main():
    """Main function to run the cricket score prediction analysis"""
    predictor = CricketScorePredictor()
    models, results, comparison = predictor.run_full_analysis()
    
    return models, results, comparison

if __name__ == "__main__":
    models, results, comparison = main()
