import pickle
import json
import os
from config import Config

class ModelLoader:
    def __init__(self):
        self.model = None
        self.player_db = None
        self.venues = {}
        self.load_model()
        self.load_player_database()
        self.load_venues()
    
    def load_model(self):
        """Load the trained ODI Progressive model"""
        try:
            with open(Config.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[OK] Model loaded from {Config.MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            raise
    
    def load_player_database(self):
        """Load player database with batting/bowling stats"""
        try:
            with open(Config.PLAYER_DB_PATH, 'r') as f:
                self.player_db = json.load(f)
            print(f"[OK] Player database loaded: {len(self.player_db)} players")
        except Exception as e:
            print(f"[ERROR] Error loading player database: {e}")
            raise
    
    def load_venues(self):
        """Extract unique venues from test data"""
        try:
            import pandas as pd
            df = pd.read_csv(Config.TEST_DATA_PATH)
            
            # Get unique venues with their average scores
            venue_data = df.groupby('venue').agg({
                'venue_avg_score': 'first',
                'final_score': 'mean'
            }).reset_index()
            
            self.venues = {
                row['venue']: {
                    'avg_score': float(row['venue_avg_score']),
                    'actual_avg': float(row['final_score'])
                }
                for _, row in venue_data.iterrows()
            }
            
            print(f"[OK] Loaded {len(self.venues)} venues")
        except Exception as e:
            print(f"[WARNING] Could not load venues: {e}")
            self.venues = {}
    
    def get_teams(self):
        """Get list of international teams"""
        teams = [
            'India', 'Australia', 'England', 'Pakistan', 'South Africa',
            'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh',
            'Afghanistan', 'Zimbabwe', 'Ireland', 'Scotland', 'Netherlands'
        ]
        return sorted(teams)
    
    def get_players_for_team(self, team):
        """Get players for a specific team from player database"""
        players = []
        for player_name, player_data in self.player_db.items():
            # Simple heuristic: if player has stats, include them
            if 'batting' in player_data or 'bowling' in player_data:
                players.append({
                    'name': player_name,
                    'batting_avg': player_data.get('batting', {}).get('average', 0),
                    'strike_rate': player_data.get('batting', {}).get('strike_rate', 0),
                    'bowling_economy': player_data.get('bowling', {}).get('economy', 0)
                })
        
        # Sort by batting average
        players.sort(key=lambda x: x['batting_avg'], reverse=True)
        return players[:100]  # Return top 100
    
    def get_venues(self):
        """Get list of venues"""
        return [
            {
                'name': venue,
                'avg_score': data['avg_score'],
                'actual_avg': data['actual_avg']
            }
            for venue, data in self.venues.items()
        ]

# Global model loader instance
_model_loader = None

def get_model_loader():
    """Get or create model loader singleton"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader

