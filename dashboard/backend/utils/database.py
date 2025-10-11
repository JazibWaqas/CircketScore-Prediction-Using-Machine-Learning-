import sqlite3
import os

class Database:
    def __init__(self):
        db_path = os.path.join(os.path.dirname(__file__), '../../../ODI_Progressive/cricket_prediction_odi.db')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        print(f"[OK] Connected to database: {db_path}")
    
    def get_all_teams(self):
        """Get all teams from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT team_id, team_name, team_type FROM teams ORDER BY team_name")
        teams = []
        for row in cursor.fetchall():
            teams.append({
                'team_id': row['team_id'],
                'team_name': row['team_name'],
                'team_type': row['team_type']
            })
        return teams
    
    def get_all_players(self):
        """Get all players from database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT player_id, player_name, role, teams, 
                   batting_avg, bowling_avg, total_matches, star_rating
            FROM players 
            ORDER BY batting_avg DESC
        """)
        players = []
        for row in cursor.fetchall():
            # Parse teams to get country
            teams_str = row['teams'] or ''
            country = 'Unknown'
            if teams_str:
                # Extract country from teams string (e.g., "India, Mumbai Indians" -> "India")
                countries = [team.strip() for team in teams_str.split(',')]
                # Find a known country
                known_countries = ['India', 'Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand', 'Sri Lanka', 'Bangladesh', 'West Indies', 'Afghanistan', 'Ireland', 'Zimbabwe']
                for c in countries:
                    if c in known_countries:
                        country = c
                        break
                else:
                    country = countries[0] if countries else 'Unknown'
            
            players.append({
                'player_id': row['player_id'],
                'player_name': row['player_name'],
                'player_role': row['role'] or 'All-rounder',
                'country': country,
                'batting_avg': row['batting_avg'] or 0,
                'bowling_economy': (row['bowling_avg'] or 0) / 2.0,  # Convert bowling avg to economy
                'total_matches': row['total_matches'] or 0,
                'tier': 'regular',
                'has_impact': False,
                'batting_impact': 0,
                'bowling_impact': 0
            })
        return players
    
    def get_all_venues(self):
        """Get all venues from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT venue_id, venue_name, city FROM venues ORDER BY venue_name")
        venues = []
        for row in cursor.fetchall():
            venues.append({
                'venue_id': row['venue_id'],
                'venue_name': row['venue_name'],
                'city': row['city']
            })
        return venues
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Global database instance
_db = None

def get_database():
    """Get or create database singleton"""
    global _db
    if _db is None:
        _db = Database()
    return _db

