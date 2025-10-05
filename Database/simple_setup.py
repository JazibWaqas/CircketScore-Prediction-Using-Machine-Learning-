#!/usr/bin/env python3
"""
Simple Database Setup - Just load the essential data
"""

import sqlite3
import pandas as pd
import os

def create_simple_database():
    """Create a simple database with just the essential tables"""
    
    print("Creating simple database...")
    
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
    
    # Create matches table
    cursor.execute('''
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY,
            date TEXT,
            venue_id INTEGER,
            team_a_id INTEGER,
            team_b_id INTEGER,
            toss_winner_id INTEGER,
            toss_decision TEXT,
            match_winner_id INTEGER,
            player_of_match_id INTEGER,
            season TEXT,
            event_name TEXT,
            match_number INTEGER,
            gender TEXT,
            is_final INTEGER DEFAULT 0,
            is_semi_final INTEGER DEFAULT 0,
            is_playoff INTEGER DEFAULT 0
        )
    ''')
    
    # Create team_performances table
    cursor.execute('''
        CREATE TABLE team_performances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            team_id INTEGER,
            opposition_id INTEGER,
            total_runs INTEGER,
            batting_first INTEGER,
            team_players TEXT,
            team_player_ids TEXT,
            venue_avg_runs REAL,
            venue_runs_std REAL,
            venue_matches INTEGER,
            venue_high_score INTEGER,
            venue_low_score INTEGER,
            h2h_matches INTEGER,
            h2h_avg_runs REAL,
            h2h_win_rate REAL,
            team_form_avg_runs REAL,
            team_form_win_rate REAL,
            is_home_team INTEGER,
            team_batting_avg REAL,
            team_batting_std REAL,
            opposition_bowling_avg REAL,
            opposition_bowling_std REAL,
            venue_difficulty REAL,
            team_form_score REAL,
            h2h_strength REAL,
            match_importance REAL,
            team_balance REAL,
            pressure_score REAL,
            team_recent_avg REAL,
            opposition_recent_avg REAL,
            is_home_advantage INTEGER,
            is_important_match INTEGER,
            is_t20_world_cup INTEGER,
            is_ipl INTEGER,
            season_year INTEGER,
            season_month INTEGER,
            is_winter INTEGER,
            is_summer INTEGER
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
    
    # Load a sample of training data
    print("Loading sample training data...")
    train_df = pd.read_csv('../data/simple_enhanced_train.csv')
    
    # Take only first 1000 records to avoid issues
    sample_df = train_df.head(1000).copy()
    sample_df = sample_df.rename(columns={'opposition': 'opposition_id'})
    
    # Convert boolean columns to integers
    bool_columns = ['batting_first', 'is_home_team', 'is_final', 'is_semi_final', 'is_playoff', 
                   'is_home_advantage', 'is_important_match', 'is_t20_world_cup', 'is_ipl', 
                   'is_winter', 'is_summer']
    
    for col in bool_columns:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].astype(int)
    
    sample_df.to_sql('team_performances', conn, if_exists='append', index=False)
    print(f"Loaded {len(sample_df)} team performances")
    
    conn.commit()
    conn.close()
    
    print("\nSimple database setup complete!")
    print(f"Database: {db_path}")
    print("Ready for frontend testing!")

if __name__ == "__main__":
    create_simple_database()
