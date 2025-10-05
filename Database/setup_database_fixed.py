#!/usr/bin/env python3
"""
Cricket Score Prediction - Fixed Database Setup
Creates SQLite database with all tables and populates with data
"""

import sqlite3
import pandas as pd
import pickle
import os
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def create_database():
    """Create complete SQLite database with all tables"""
    
    # Create database
    db_path = "cricket_prediction.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Setting up Complete Cricket Prediction Database...")
    
    # 1. Create teams table
    print("Creating teams table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT NOT NULL,
            country TEXT,
            team_type TEXT DEFAULT 'International',
            established_year INTEGER,
            home_ground_id INTEGER,
            team_logo_url TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 2. Create venues table
    print("Creating venues table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS venues (
            venue_id INTEGER PRIMARY KEY,
            venue_name TEXT NOT NULL,
            city TEXT,
            country TEXT,
            capacity INTEGER,
            venue_type TEXT DEFAULT 'Stadium',
            pitch_type TEXT DEFAULT 'Balanced',
            weather_conditions TEXT,
            established_year INTEGER,
            coordinates_lat REAL,
            coordinates_lng REAL,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 3. Create players table
    print("Creating players table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            country TEXT,
            date_of_birth DATE,
            batting_style TEXT,
            bowling_style TEXT,
            player_role TEXT,
            is_active BOOLEAN DEFAULT 1,
            career_start_year INTEGER,
            career_end_year INTEGER,
            profile_image_url TEXT
        )
    ''')
    
    # 4. Create matches table
    print("Creating matches table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            date DATE NOT NULL,
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
            is_final BOOLEAN DEFAULT 0,
            is_semi_final BOOLEAN DEFAULT 0,
            is_playoff BOOLEAN DEFAULT 0,
            FOREIGN KEY (venue_id) REFERENCES venues(venue_id),
            FOREIGN KEY (team_a_id) REFERENCES teams(team_id),
            FOREIGN KEY (team_b_id) REFERENCES teams(team_id)
        )
    ''')
    
    # 5. Create team_performances table
    print("Creating team_performances table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_performances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            team_id INTEGER,
            opposition_id INTEGER,
            total_runs INTEGER,
            batting_first BOOLEAN,
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
            is_home_team BOOLEAN,
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
            is_home_advantage BOOLEAN,
            is_important_match BOOLEAN,
            is_t20_world_cup BOOLEAN,
            is_ipl BOOLEAN,
            season_year INTEGER,
            season_month INTEGER,
            is_winter BOOLEAN,
            is_summer BOOLEAN,
            FOREIGN KEY (match_id) REFERENCES matches(match_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (opposition_id) REFERENCES teams(team_id)
        )
    ''')
    
    # 6. Create other tables
    print("Creating additional tables...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_stats (
            player_id INTEGER,
            team_id INTEGER,
            season TEXT,
            matches_played INTEGER,
            runs_scored INTEGER,
            balls_faced INTEGER,
            batting_average REAL,
            strike_rate REAL,
            wickets_taken INTEGER,
            balls_bowled INTEGER,
            bowling_average REAL,
            economy_rate REAL,
            catches INTEGER,
            stumpings INTEGER,
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS venue_stats (
            venue_id INTEGER,
            total_matches INTEGER,
            avg_runs_scored REAL,
            runs_std_deviation REAL,
            highest_score INTEGER,
            lowest_score INTEGER,
            batting_first_wins INTEGER,
            fielding_first_wins INTEGER,
            avg_wickets_fallen REAL,
            pitch_conditions TEXT,
            weather_impact REAL,
            FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS head_to_head (
            team_a_id INTEGER,
            team_b_id INTEGER,
            total_matches INTEGER,
            team_a_wins INTEGER,
            team_b_wins INTEGER,
            ties INTEGER,
            avg_runs_team_a REAL,
            avg_runs_team_b REAL,
            last_meeting_date DATE,
            last_meeting_winner_id INTEGER,
            FOREIGN KEY (team_a_id) REFERENCES teams(team_id),
            FOREIGN KEY (team_b_id) REFERENCES teams(team_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_predictions (
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
            actual_score_a REAL,
            actual_score_b REAL,
            prediction_accuracy REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_a_id) REFERENCES teams(team_id),
            FOREIGN KEY (team_b_id) REFERENCES teams(team_id),
            FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            test_date DATE,
            r2_score REAL,
            rmse REAL,
            mae REAL,
            accuracy_10_runs REAL,
            accuracy_15_runs REAL,
            accuracy_20_runs REAL,
            test_records INTEGER,
            training_time_seconds REAL,
            cross_validation_r2 REAL
        )
    ''')
    
    conn.commit()
    print("Database tables created successfully!")
    return conn

def load_data(conn):
    """Load all data from CSV files into database"""
    
    print("\nLoading data into database...")
    
    # Load teams
    print("Loading teams...")
    teams_df = pd.read_csv('../data/team_lookup.csv')
    teams_df['country'] = teams_df['team_name'].apply(lambda x: x.split()[-1] if ' ' in x else x)
    teams_df['team_type'] = 'International'
    teams_df['is_active'] = True
    teams_df.to_sql('teams', conn, if_exists='append', index=False)
    print(f"Loaded {len(teams_df)} teams")
    
    # Load venues
    print("Loading venues...")
    venues_df = pd.read_csv('../data/venue_lookup.csv')
    venues_df['city'] = venues_df['venue_name'].apply(lambda x: x.split(',')[0] if ',' in x else 'Unknown')
    venues_df['country'] = venues_df['venue_name'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else 'Unknown')
    venues_df['capacity'] = 50000  # Default capacity
    venues_df['venue_type'] = 'Stadium'
    venues_df['pitch_type'] = 'Balanced'
    venues_df['is_active'] = True
    venues_df.to_sql('venues', conn, if_exists='append', index=False)
    print(f"Loaded {len(venues_df)} venues")
    
    # Load players
    print("Loading players...")
    players_df = pd.read_csv('../data/player_lookup.csv')
    players_df['country'] = 'Unknown'
    players_df['batting_style'] = 'Right-handed'
    players_df['bowling_style'] = 'Right-arm medium'
    players_df['player_role'] = 'All-rounder'
    players_df['is_active'] = True
    players_df.to_sql('players', conn, if_exists='append', index=False)
    print(f"Loaded {len(players_df)} players")
    
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('../data/simple_enhanced_train.csv')
    
    # Create matches from training data (unique matches only)
    print("Creating matches...")
    unique_matches = train_df.groupby('match_id').first().reset_index()
    
    matches_data = []
    for _, row in unique_matches.iterrows():
        matches_data.append({
            'match_id': row['match_id'],
            'date': row['date'],
            'venue_id': row['venue_id'],
            'team_a_id': row['team_id'],
            'team_b_id': row['opposition'],
            'toss_winner_id': row['toss_winner'],
            'toss_decision': row['toss_decision'],
            'match_winner_id': row['match_winner'],
            'player_of_match_id': row['player_of_match'],
            'season': row['season'],
            'event_name': row['event_name'],
            'match_number': row['match_number'],
            'gender': row['gender'],
            'is_final': row['is_final'],
            'is_semi_final': row['is_semi_final'],
            'is_playoff': row['is_playoff']
        })
    
    matches_df = pd.DataFrame(matches_data)
    matches_df.to_sql('matches', conn, if_exists='append', index=False)
    print(f"Loaded {len(matches_df)} matches")
    
    # Load team performances
    print("Loading team performances...")
    team_perf_data = train_df.copy()
    team_perf_data = team_perf_data.rename(columns={'opposition': 'opposition_id'})
    team_perf_data.to_sql('team_performances', conn, if_exists='append', index=False)
    print(f"Loaded {len(train_df)} team performances")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('../data/simple_enhanced_test.csv')
    test_df = test_df.rename(columns={'opposition': 'opposition_id'})
    test_df.to_sql('team_performances', conn, if_exists='append', index=False)
    print(f"Loaded {len(test_df)} test performances")
    
    # Calculate and load venue stats
    print("Calculating venue statistics...")
    venue_stats = train_df.groupby('venue_id').agg({
        'total_runs': ['mean', 'std', 'max', 'min', 'count'],
        'batting_first': 'sum'
    }).reset_index()
    
    venue_stats.columns = ['venue_id', 'avg_runs_scored', 'runs_std_deviation', 
                          'highest_score', 'lowest_score', 'total_matches', 'batting_first_wins']
    venue_stats['fielding_first_wins'] = venue_stats['total_matches'] - venue_stats['batting_first_wins']
    venue_stats['avg_wickets_fallen'] = 8.0
    venue_stats['pitch_conditions'] = 'Balanced'
    venue_stats['weather_impact'] = 1.0
    
    venue_stats.to_sql('venue_stats', conn, if_exists='append', index=False)
    print(f"Loaded venue statistics for {len(venue_stats)} venues")
    
    # Calculate head-to-head records
    print("Calculating head-to-head records...")
    h2h_data = []
    for team_a in train_df['team_id'].unique():
        for team_b in train_df['team_id'].unique():
            if team_a != team_b:
                matches = train_df[((train_df['team_id'] == team_a) & (train_df['opposition'] == team_b)) | 
                                 ((train_df['team_id'] == team_b) & (train_df['opposition'] == team_a))]
                if len(matches) > 0:
                    team_a_wins = len(matches[matches['match_winner'] == team_a])
                    team_b_wins = len(matches[matches['match_winner'] == team_b])
                    ties = len(matches) - team_a_wins - team_b_wins
                    
                    h2h_data.append({
                        'team_a_id': team_a,
                        'team_b_id': team_b,
                        'total_matches': len(matches),
                        'team_a_wins': team_a_wins,
                        'team_b_wins': team_b_wins,
                        'ties': ties,
                        'avg_runs_team_a': matches[matches['team_id'] == team_a]['total_runs'].mean(),
                        'avg_runs_team_b': matches[matches['team_id'] == team_b]['total_runs'].mean(),
                        'last_meeting_date': matches['date'].max(),
                        'last_meeting_winner_id': matches.iloc[-1]['match_winner']
                    })
    
    h2h_df = pd.DataFrame(h2h_data)
    h2h_df.to_sql('head_to_head', conn, if_exists='append', index=False)
    print(f"Loaded {len(h2h_df)} head-to-head records")
    
    # Load model performance data
    print("Loading model performance data...")
    if os.path.exists('../models/mixed_features_model_comparison.csv'):
        model_perf_df = pd.read_csv('../models/mixed_features_model_comparison.csv')
        model_perf_df['test_date'] = datetime.now().date()
        model_perf_df.to_sql('model_performance', conn, if_exists='append', index=False)
        print("Loaded model performance data")
    
    conn.commit()
    print("\nDatabase setup complete!")
    
    # Print summary
    cursor = conn.cursor()
    tables = ['teams', 'venues', 'players', 'matches', 'team_performances', 
              'venue_stats', 'head_to_head', 'model_performance']
    
    print("\nDatabase Summary:")
    for table in tables:
        count = cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f"   {table}: {count:,} records")
    
    conn.close()
    return "cricket_prediction.db"

if __name__ == "__main__":
    print("Cricket Score Prediction - Database Setup")
    print("=" * 50)
    
    # Create database
    conn = create_database()
    
    # Load data
    db_path = load_data(conn)
    
    print(f"\nSetup complete! Database: {db_path}")
    print("Ready to start the frontend system!")
