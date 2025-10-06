#!/usr/bin/env python3
"""
Build Venue Conditions Dataset
Extract venue-specific conditions, weather, and pitch characteristics
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VenueConditionsDatasetBuilder:
    def __init__(self):
        self.ball_by_ball_dir = "raw_data/t20 matches ball by ball"
        self.output_dir = "processed_data"
        
        # Load lookup tables
        self.venue_lookup = pd.read_csv("data/venue_lookup.csv")
        
        print("Loaded venue lookup tables")
        
    def estimate_weather_conditions(self, date_str, venue_name, city):
        """Estimate weather conditions based on date and location"""
        if not date_str:
            return self.get_default_weather()
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            year = date.year
        except:
            return self.get_default_weather()
        
        # Weather estimation based on location and season
        weather = {
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 5.0,
            'precipitation': 0.0,
            'dew_factor': 0.0
        }
        
        # Location-based adjustments
        if any(city_name in city.lower() for city_name in ['mumbai', 'delhi', 'kolkata', 'chennai', 'bangalore']):
            # Indian cities
            weather['temperature'] = 28 + 8 * np.sin(2 * np.pi * month / 12)
            weather['humidity'] = 70 + 15 * np.sin(2 * np.pi * month / 12)
            weather['dew_factor'] = 0.3 if month in [10, 11, 12, 1, 2] else 0.1
            
        elif any(city_name in city.lower() for city_name in ['london', 'birmingham', 'manchester', 'cardiff']):
            # UK cities
            weather['temperature'] = 15 + 6 * np.sin(2 * np.pi * month / 12)
            weather['humidity'] = 80 + 10 * np.sin(2 * np.pi * month / 12)
            weather['precipitation'] = 0.2 if month in [10, 11, 12, 1, 2, 3] else 0.05
            
        elif any(city_name in city.lower() for city_name in ['sydney', 'melbourne', 'brisbane', 'perth']):
            # Australian cities
            weather['temperature'] = 22 + 10 * np.sin(2 * np.pi * month / 12)
            weather['humidity'] = 60 + 20 * np.sin(2 * np.pi * month / 12)
            weather['wind_speed'] = 8.0
            
        elif any(city_name in city.lower() for city_name in ['dubai', 'abu dhabi', 'sharjah']):
            # UAE cities
            weather['temperature'] = 35 + 10 * np.sin(2 * np.pi * month / 12)
            weather['humidity'] = 50 + 20 * np.sin(2 * np.pi * month / 12)
            weather['dew_factor'] = 0.4 if month in [10, 11, 12, 1, 2] else 0.1
            
        else:
            # Default conditions
            weather['temperature'] = 25 + 8 * np.sin(2 * np.pi * month / 12)
            weather['humidity'] = 60 + 15 * np.sin(2 * np.pi * month / 12)
        
        # Add some randomness
        weather['temperature'] += np.random.normal(0, 2)
        weather['humidity'] += np.random.normal(0, 5)
        weather['wind_speed'] += np.random.normal(0, 1)
        
        # Ensure reasonable bounds
        weather['temperature'] = max(min(weather['temperature'], 45), 5)
        weather['humidity'] = max(min(weather['humidity'], 100), 20)
        weather['wind_speed'] = max(min(weather['wind_speed'], 20), 0)
        
        return weather
    
    def get_default_weather(self):
        """Get default weather conditions"""
        return {
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 5.0,
            'precipitation': 0.0,
            'dew_factor': 0.0
        }
    
    def analyze_pitch_characteristics(self, innings_data):
        """Analyze pitch characteristics from match data"""
        if not innings_data:
            return self.get_default_pitch()
        
        total_runs = 0
        total_balls = 0
        boundaries = 0
        wickets = 0
        extras = 0
        
        for inning in innings_data:
            overs = inning.get('overs', [])
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                for delivery in deliveries:
                    runs = delivery.get('runs', {})
                    total_runs += runs.get('total', 0)
                    total_balls += 1
                    
                    # Count boundaries
                    if runs.get('batter', 0) >= 4:
                        boundaries += 1
                    
                    # Count wickets
                    if 'wickets' in delivery:
                        wickets += len(delivery['wickets'])
                    
                    # Count extras
                    if 'extras' in delivery:
                        extras += 1
        
        # Calculate pitch characteristics
        run_rate = total_runs / (total_balls / 6) if total_balls > 0 else 7.5
        boundary_rate = boundaries / total_balls if total_balls > 0 else 0.1
        wicket_rate = wickets / (total_balls / 6) if total_balls > 0 else 1.0
        extra_rate = extras / total_balls if total_balls > 0 else 0.05
        
        # Determine pitch type
        if run_rate > 8.5 and boundary_rate > 0.15:
            pitch_type = 'flat'
            bounce = 1.2
            pace = 1.3
            turn = 0.7
            swing = 0.6
        elif run_rate < 6.5 and boundary_rate < 0.08:
            pitch_type = 'spinning'
            bounce = 0.8
            pace = 0.7
            turn = 1.4
            swing = 0.5
        elif run_rate < 7.0 and wicket_rate > 1.5:
            pitch_type = 'seaming'
            bounce = 1.1
            pace = 1.2
            turn = 0.8
            swing = 1.3
        else:
            pitch_type = 'balanced'
            bounce = 1.0
            pace = 1.0
            turn = 1.0
            swing = 1.0
        
        return {
            'pitch_type': pitch_type,
            'bounce': bounce,
            'pace': pace,
            'turn': turn,
            'swing': swing,
            'run_rate': run_rate,
            'boundary_rate': boundary_rate,
            'wicket_rate': wicket_rate,
            'extra_rate': extra_rate
        }
    
    def get_default_pitch(self):
        """Get default pitch characteristics"""
        return {
            'pitch_type': 'balanced',
            'bounce': 1.0,
            'pace': 1.0,
            'turn': 1.0,
            'swing': 1.0,
            'run_rate': 7.5,
            'boundary_rate': 0.1,
            'wicket_rate': 1.0,
            'extra_rate': 0.05
        }
    
    def extract_venue_conditions(self, match_file):
        """Extract venue conditions from a single match"""
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)
        except:
            return None
        
        match_id = os.path.basename(match_file).replace('.json', '')
        info = match_data.get('info', {})
        innings = match_data.get('innings', [])
        
        # Basic match info
        venue_name = info.get('venue', 'Unknown')
        city = info.get('city', 'Unknown')
        date = info.get('dates', [None])[0]
        
        # Get weather conditions
        weather = self.estimate_weather_conditions(date, venue_name, city)
        
        # Get pitch characteristics
        pitch = self.analyze_pitch_characteristics(innings)
        
        # Get venue-specific data
        venue_data = self.get_venue_data(venue_name)
        
        # Create venue conditions record
        conditions_record = {
            'match_id': match_id,
            'date': date,
            'venue': venue_name,
            'city': city,
            
            # Weather conditions
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'wind_speed': weather['wind_speed'],
            'precipitation': weather['precipitation'],
            'dew_factor': weather['dew_factor'],
            
            # Pitch characteristics
            'pitch_type': pitch['pitch_type'],
            'pitch_bounce': pitch['bounce'],
            'pitch_pace': pitch['pace'],
            'pitch_turn': pitch['turn'],
            'pitch_swing': pitch['swing'],
            'pitch_run_rate': pitch['run_rate'],
            'pitch_boundary_rate': pitch['boundary_rate'],
            'pitch_wicket_rate': pitch['wicket_rate'],
            'pitch_extra_rate': pitch['extra_rate'],
            
            # Venue data
            'venue_id': venue_data['venue_id'],
            'venue_avg_runs': venue_data['avg_runs'],
            'venue_matches': venue_data['matches'],
            'venue_high_score': venue_data['high_score'],
            'venue_low_score': venue_data['low_score'],
            
            # Derived features
            'is_day_match': self.is_day_match(date),
            'is_night_match': not self.is_day_match(date),
            'season': self.get_season(date),
            'is_rain_affected': weather['precipitation'] > 0.1,
            'is_dew_affected': weather['dew_factor'] > 0.2,
        }
        
        return conditions_record
    
    def get_venue_data(self, venue_name):
        """Get venue-specific data from lookup"""
        venue_match = self.venue_lookup[
            self.venue_lookup['venue_name'].str.contains(venue_name, case=False, na=False)
        ]
        
        if not venue_match.empty:
            return {
                'venue_id': venue_match['venue_id'].iloc[0],
                'avg_runs': venue_match['venue_avg_runs'].iloc[0] if 'venue_avg_runs' in venue_match.columns else 140.0,
                'matches': venue_match['venue_matches'].iloc[0] if 'venue_matches' in venue_match.columns else 50,
                'high_score': venue_match['venue_high_score'].iloc[0] if 'venue_high_score' in venue_match.columns else 200,
                'low_score': venue_match['venue_low_score'].iloc[0] if 'venue_low_score' in venue_match.columns else 80,
            }
        else:
            return {
                'venue_id': 0,
                'avg_runs': 140.0,
                'matches': 50,
                'high_score': 200,
                'low_score': 80,
            }
    
    def is_day_match(self, date_str):
        """Determine if match is day or night"""
        if not date_str:
            return True
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            
            # Simple heuristic: more night matches in summer months
            return month not in [5, 6, 7, 8, 9]
        except:
            return True
    
    def get_season(self, date_str):
        """Get season from date"""
        if not date_str:
            return 'unknown'
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        except:
            return 'unknown'
    
    def build_venue_conditions_dataset(self):
        """Build the complete venue conditions dataset"""
        print("Building Venue Conditions Dataset...")
        print("=" * 50)
        
        # Get all match files
        json_files = glob.glob(f"{self.ball_by_ball_dir}/*.json")
        print(f"Found {len(json_files)} match files")
        
        venue_conditions_data = []
        processed_count = 0
        
        for i, match_file in enumerate(json_files):
            if i % 100 == 0:
                print(f"Processing {i}/{len(json_files)} matches...")
            
            try:
                conditions_record = self.extract_venue_conditions(match_file)
                if conditions_record:
                    venue_conditions_data.append(conditions_record)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {match_file}: {e}")
                continue
        
        print(f"Processed {processed_count} venue conditions")
        
        # Create DataFrame
        df = pd.DataFrame(venue_conditions_data)
        
        if df.empty:
            print("No data processed!")
            return None
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Save dataset
        output_file = f"{self.output_dir}/venue_conditions_dataset.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Venue Conditions Dataset created: {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    
    def add_derived_features(self, df):
        """Add derived features to the dataset"""
        print("Adding derived features...")
        
        # Weather impact on scoring
        df['weather_impact'] = (
            df['temperature'] * 0.1 +
            df['humidity'] * 0.05 +
            df['wind_speed'] * 0.2 +
            df['dew_factor'] * 10
        )
        
        # Pitch difficulty
        df['pitch_difficulty'] = (
            df['pitch_bounce'] * 0.2 +
            df['pitch_pace'] * 0.3 +
            df['pitch_turn'] * 0.3 +
            df['pitch_swing'] * 0.2
        )
        
        # Venue scoring potential
        df['venue_scoring_potential'] = (
            df['venue_avg_runs'] / 100 +
            df['pitch_run_rate'] / 10 +
            df['weather_impact'] / 10
        )
        
        # Match conditions score
        df['match_conditions_score'] = (
            df['weather_impact'] * 0.4 +
            df['pitch_difficulty'] * 0.4 +
            df['venue_scoring_potential'] * 0.2
        )
        
        return df

def main():
    """Main function to build venue conditions dataset"""
    print("Building Venue Conditions Dataset")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Initialize builder
    builder = VenueConditionsDatasetBuilder()
    
    # Build dataset
    df = builder.build_venue_conditions_dataset()
    
    if df is not None:
        print("\nVenue Conditions Dataset created successfully!")
        print(f"Dataset contains {len(df)} venue conditions")
        print(f"Dataset has {len(df.columns)} features")
        print(f"Saved to: processed_data/venue_conditions_dataset.csv")
        
        # Show sample
        print("\nSample data:")
        print(df.head(3).to_string())
    else:
        print("Failed to create venue conditions dataset")

if __name__ == "__main__":
    main()
