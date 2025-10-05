#!/usr/bin/env python3
"""
Minimal Database Setup - Just the essential data for frontend
"""

import sqlite3
import pandas as pd
import os

def create_minimal_database():
    """Create a minimal database with just the essential data"""
    
    print("Creating minimal database...")
    
    # Delete existing database if it exists
    db_path = "cricket_prediction.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create teams table
    cursor.execute('''
        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT NOT NULL,
            country TEXT,
            team_type TEXT DEFAULT 'International',
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create venues table
    cursor.execute('''
        CREATE TABLE venues (
            venue_id INTEGER PRIMARY KEY,
            venue_name TEXT NOT NULL,
            city TEXT,
            country TEXT,
            capacity INTEGER DEFAULT 50000,
            venue_type TEXT DEFAULT 'Stadium',
            pitch_type TEXT DEFAULT 'Balanced',
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create players table
    cursor.execute('''
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            country TEXT,
            batting_style TEXT DEFAULT 'Right-handed',
            bowling_style TEXT DEFAULT 'Right-arm medium',
            player_role TEXT DEFAULT 'All-rounder',
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create user_predictions table
    cursor.execute('''
        CREATE TABLE user_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_a_id INTEGER,
            team_b_id INTEGER,
            venue_id INTEGER,
            team_a_players TEXT,
            team_b_players TEXT,
            predicted_score_a REAL,
            predicted_score_b REAL,
            confidence_score REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    print("Tables created successfully!")
    
    # Load basic data
    print("Loading teams...")
    teams_df = pd.read_csv('../data/team_lookup.csv')
    teams_df['country'] = teams_df['team_name'].apply(lambda x: x.split()[-1] if ' ' in x else x)
    teams_df['team_type'] = 'International'
    teams_df['is_active'] = True
    teams_df.to_sql('teams', conn, if_exists='append', index=False)
    print(f"Loaded {len(teams_df)} teams")
    
    print("Loading venues...")
    venues_df = pd.read_csv('../data/venue_lookup.csv')
    venues_df['city'] = venues_df['venue_name'].apply(lambda x: x.split(',')[0] if ',' in x else 'Unknown')
    venues_df['country'] = venues_df['venue_name'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else 'Unknown')
    venues_df['capacity'] = 50000
    venues_df['venue_type'] = 'Stadium'
    venues_df['pitch_type'] = 'Balanced'
    venues_df['is_active'] = True
    venues_df.to_sql('venues', conn, if_exists='append', index=False)
    print(f"Loaded {len(venues_df)} venues")
    
    print("Loading players...")
    players_df = pd.read_csv('../data/player_lookup.csv')
    players_df['country'] = 'Unknown'
    players_df['batting_style'] = 'Right-handed'
    players_df['bowling_style'] = 'Right-arm medium'
    players_df['player_role'] = 'All-rounder'
    players_df['is_active'] = True
    players_df.to_sql('players', conn, if_exists='append', index=False)
    print(f"Loaded {len(players_df)} players")
    
    conn.commit()
    conn.close()
    
    print("\nMinimal database setup complete!")
    print(f"Database: {db_path}")
    print("Ready for frontend testing!")
    print("\nYou can now start the Flask API server with: python app.py")

if __name__ == "__main__":
    create_minimal_database()
