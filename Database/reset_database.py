#!/usr/bin/env python3
"""
Reset and recreate the database
"""

import os
import sqlite3

def reset_database():
    """Delete existing database and recreate"""
    
    print("Resetting database...")
    
    # Delete existing database if it exists
    db_path = "cricket_prediction.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Deleted existing database")
    
    # Recreate database with proper schema
    print("Creating fresh database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create teams table
    cursor.execute('''
        CREATE TABLE teams (
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
    
    # Create venues table
    cursor.execute('''
        CREATE TABLE venues (
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
    
    # Create players table
    cursor.execute('''
        CREATE TABLE players (
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
    
    # Create matches table
    cursor.execute('''
        CREATE TABLE matches (
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
    
    # Create team_performances table
    cursor.execute('''
        CREATE TABLE team_performances (
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
    
    # Create other tables
    cursor.execute('''
        CREATE TABLE player_stats (
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
        CREATE TABLE venue_stats (
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
        CREATE TABLE head_to_head (
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
        CREATE TABLE model_performance (
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
    conn.close()
    
    print("Fresh database created successfully!")
    print("You can now run the setup script again")

if __name__ == "__main__":
    reset_database()
