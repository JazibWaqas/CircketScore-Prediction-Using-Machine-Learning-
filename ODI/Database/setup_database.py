#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ODI Cricket Prediction Database Setup

Creates SQLite database with:
- Teams table (from team_lookup.csv)
- Venues table (from venue_lookup.csv)
- Players table (from player_lookup.csv)
- Predictions history table
"""

import sqlite3
import pandas as pd
import os
import sys

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def create_database():
    """Create ODI cricket prediction database"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "cricket_prediction_odi.db")
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Remove old database if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print("✅ Removed old database")
    
    print("\n" + "="*70)
    print("CREATING ODI CRICKET PREDICTION DATABASE")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create teams table
    print("\n📋 Creating teams table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT NOT NULL UNIQUE,
            team_type TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 2. Create venues table
    print("📋 Creating venues table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS venues (
            venue_id INTEGER PRIMARY KEY,
            venue_name TEXT NOT NULL UNIQUE,
            stadium TEXT,
            city TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 3. Create players table
    print("📋 Creating players table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            role TEXT,
            skill_level TEXT,
            star_rating REAL,
            teams TEXT,
            batting_avg REAL,
            bowling_avg REAL,
            total_matches INTEGER,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 4. Create predictions history table
    print("📋 Creating predictions history table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_a_id INTEGER,
            team_b_id INTEGER,
            venue_id INTEGER,
            team_a_players TEXT,
            team_b_players TEXT,
            predicted_score_a REAL,
            predicted_score_b REAL,
            model_used TEXT,
            match_date TEXT,
            gender TEXT,
            toss_won INTEGER,
            toss_decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_a_id) REFERENCES teams (team_id),
            FOREIGN KEY (team_b_id) REFERENCES teams (team_id),
            FOREIGN KEY (venue_id) REFERENCES venues (venue_id)
        )
    ''')
    
    conn.commit()
    print("✅ All tables created successfully!")
    
    # Load and populate teams
    print("\n" + "="*70)
    print("POPULATING TABLES FROM LOOKUP FILES")
    print("="*70)
    
    print("\n📊 Loading teams...")
    teams_df = pd.read_csv(os.path.join(data_dir, 'team_lookup.csv'))
    teams_df.to_sql('teams', conn, if_exists='replace', index=False)
    print(f"✅ Loaded {len(teams_df)} teams")
    
    # Load and populate venues
    print("\n📊 Loading venues...")
    venues_df = pd.read_csv(os.path.join(data_dir, 'venue_lookup.csv'))
    venues_df.to_sql('venues', conn, if_exists='replace', index=False)
    print(f"✅ Loaded {len(venues_df)} venues")
    
    # Load and populate players
    print("\n📊 Loading players...")
    players_df = pd.read_csv(os.path.join(data_dir, 'player_lookup.csv'))
    players_df.to_sql('players', conn, if_exists='replace', index=False)
    print(f"✅ Loaded {len(players_df)} players")
    
    # Verify data
    print("\n" + "="*70)
    print("DATABASE VERIFICATION")
    print("="*70)
    
    cursor.execute("SELECT COUNT(*) FROM teams")
    team_count = cursor.fetchone()[0]
    print(f"\n✅ Teams in database: {team_count}")
    
    cursor.execute("SELECT COUNT(*) FROM venues")
    venue_count = cursor.fetchone()[0]
    print(f"✅ Venues in database: {venue_count}")
    
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    print(f"✅ Players in database: {player_count}")
    
    # Show sample data
    print("\n" + "="*70)
    print("SAMPLE DATA")
    print("="*70)
    
    print("\n🏏 Sample Teams:")
    cursor.execute("SELECT team_name, team_type FROM teams LIMIT 5")
    for row in cursor.fetchall():
        print(f"   - {row[0]} ({row[1]})")
    
    print("\n🏟️  Sample Venues:")
    cursor.execute("SELECT venue_name, city FROM venues LIMIT 5")
    for row in cursor.fetchall():
        city = row[1] if row[1] else "Unknown"
        print(f"   - {row[0]} ({city})")
    
    print("\n👤 Sample Players:")
    cursor.execute("SELECT player_name, role, skill_level, star_rating FROM players ORDER BY star_rating DESC LIMIT 5")
    for row in cursor.fetchall():
        print(f"   - {row[0]} | {row[1]} | {row[2]} (⭐ {row[3]})")
    
    conn.close()
    
    print("\n" + "="*70)
    print("DATABASE SETUP COMPLETE!")
    print("="*70)
    print(f"\n✅ Database created: {db_path}")
    print(f"✅ Total teams: {team_count}")
    print(f"✅ Total venues: {venue_count}")
    print(f"✅ Total players: {player_count}")
    print("\n🚀 Next step: Use run_odi_api.py to start the backend API")
    print("="*70 + "\n")

if __name__ == '__main__':
    create_database()

