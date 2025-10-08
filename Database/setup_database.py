import sqlite3
import pandas as pd
import json
import ast
import os

def create_t20_only_database():
    """Create database with only T20 players from training dataset"""
    db_path = "cricket_prediction.db"
    
    # Remove old database
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed old database")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating T20-only database...")
    
    # 1. Create teams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT NOT NULL,
            country TEXT,
            team_type TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 2. Create venues table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS venues (
            venue_id INTEGER PRIMARY KEY,
            venue_name TEXT NOT NULL,
            city TEXT,
            country TEXT,
            capacity INTEGER,
            venue_type TEXT,
            pitch_type TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 3. Create players table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            country TEXT,
            batting_style TEXT,
            bowling_style TEXT,
            player_role TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # 4. Create user_predictions table
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_a_id) REFERENCES teams (team_id),
            FOREIGN KEY (team_b_id) REFERENCES teams (team_id),
            FOREIGN KEY (venue_id) REFERENCES venues (venue_id)
        )
    ''')
    
    conn.commit()
    print("Tables created successfully!")

    # Load training dataset to get actual T20 player IDs
    print("Loading training dataset to extract T20 players...")
    try:
        train_df = pd.read_csv('../data/enhanced_train_dataset.csv')
        print(f"Loaded training dataset with {len(train_df)} matches")
        
        # Extract all unique player IDs from team_player_ids column
        all_player_ids = set()
        for _, row in train_df.iterrows():
            if pd.notna(row['team_player_ids']):
                try:
                    # Parse the list of player IDs
                    player_ids = ast.literal_eval(row['team_player_ids'])
                    if isinstance(player_ids, list):
                        all_player_ids.update(player_ids)
                except:
                    continue
        
        print(f"Found {len(all_player_ids)} unique T20 player IDs in training data")
        
        # Load teams from team_lookup.csv
    print("Loading teams...")
    teams_df = pd.read_csv('../data/team_lookup.csv')
    teams_df['country'] = teams_df['team_name'].apply(lambda x: x.split()[-1] if ' ' in x else x)
    teams_df['team_type'] = 'International'
    teams_df['is_active'] = True
        
        for _, row in teams_df.iterrows():
            cursor.execute('''
                INSERT INTO teams (team_id, team_name, country, team_type, is_active)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['team_id'], row['team_name'], row['country'], row['team_type'], row['is_active']))
        
        print(f"Loaded {len(teams_df)} teams")

        # Load venues from venue_lookup.csv
    print("Loading venues...")
    venues_df = pd.read_csv('../data/venue_lookup.csv')
    venues_df['city'] = venues_df['venue_name'].apply(lambda x: x.split(',')[0] if ',' in x else 'Unknown')
    venues_df['country'] = venues_df['venue_name'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else 'Unknown')
        venues_df['capacity'] = 50000
    venues_df['venue_type'] = 'Stadium'
    venues_df['pitch_type'] = 'Balanced'
    venues_df['is_active'] = True
        
        for _, row in venues_df.iterrows():
            cursor.execute('''
                INSERT INTO venues (venue_id, venue_name, city, country, capacity, venue_type, pitch_type, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['venue_id'], row['venue_name'], row['city'], row['country'], 
                  row['capacity'], row['venue_type'], row['pitch_type'], row['is_active']))
        
        print(f"Loaded {len(venues_df)} venues")

        # Load only T20 players from player_lookup.csv
        print("Loading T20 players only...")
    players_df = pd.read_csv('../data/player_lookup.csv')
        
        # Filter to only include players who were in T20 matches
        t20_players_df = players_df[players_df['player_id'].isin(all_player_ids)]
        print(f"Filtered to {len(t20_players_df)} T20 players (from {len(players_df)} total players)")
        
        # Add proper roles and countries
        def assign_player_role(name):
            name_lower = name.lower()
            
            # Wicket-keepers
            if any(term in name_lower for term in ['dhoni', 'pant', 'buttler', 'carey', 'rizwan', 'klaasen', 'de kock']):
                return 'Wicket-keeper'
            
            # Bowlers
            if any(term in name_lower for term in ['bumrah', 'shami', 'starc', 'cummins', 'archer', 'rabada', 'afridi', 'rauf', 'ali', 'shah', 'lyon', 'rashid', 'chahal', 'jadeja', 'ashwin']):
                return 'Bowler'
            
            # All-rounders
            if any(term in name_lower for term in ['jadeja', 'pandya', 'stokes', 'maxwell', 'shadab', 'nawaz', 'jansen', 'curran', 'woakes', 'ali']):
                return 'All-rounder'
            
            # Batsmen
            if any(term in name_lower for term in ['kohli', 'rohit', 'sharma', 'smith', 'warner', 'root', 'babar', 'azam', 'fakhar', 'zaman', 'imam', 'faf', 'miller', 'markram', 'finch', 'bairstow']):
                return 'Batsman'
            
            return 'All-rounder'  # Default
        
        def assign_country(name):
            name_lower = name.lower()
            
            if any(term in name_lower for term in ['kohli', 'rohit', 'sharma', 'dhoni', 'bumrah', 'pandya', 'jadeja', 'pant', 'rahul', 'shami', 'chahal', 'kumar', 'singh', 'patel', 'gupta']):
                return 'India'
            elif any(term in name_lower for term in ['babar', 'azam', 'afridi', 'rizwan', 'fakhar', 'zaman', 'shadab', 'rauf', 'ali', 'shah', 'khan', 'ahmed', 'hussain', 'nawaz']):
                return 'Pakistan'
            elif any(term in name_lower for term in ['smith', 'warner', 'cummins', 'starc', 'maxwell', 'hazlewood', 'finch', 'carey', 'lyon', 'stoins', 'archer', 'wood']):
                return 'Australia'
            elif any(term in name_lower for term in ['root', 'stokes', 'buttler', 'archer', 'bairstow', 'ali', 'woakes', 'rashid', 'wood', 'curran']):
                return 'England'
            elif any(term in name_lower for term in ['de kock', 'rabada', 'du plessis', 'miller', 'ngidi', 'shamsi', 'markram', 'jansen', 'klaasen', 'nortje']):
                return 'South Africa'
            elif any(term in name_lower for term in ['williamson', 'boult', 'southee', 'taylor', 'latham', 'santner', 'ferguson', 'conway']):
                return 'New Zealand'
            elif any(term in name_lower for term in ['mendis', 'perera', 'mathews', 'shanaka', 'karunaratne', 'fernando', 'rajapaksa']):
                return 'Sri Lanka'
            elif any(term in name_lower for term in ['mushfiqur', 'tamim', 'mahmudullah', 'shakib', 'mustafizur', 'liton', 'soumya']):
                return 'Bangladesh'
            elif any(term in name_lower for term in ['rashid', 'mujeeb', 'najibullah', 'shahidi', 'ibrahim', 'gurbaz']):
                return 'Afghanistan'
            else:
                return 'Unknown'
        
        # Add roles and countries
        t20_players_df = t20_players_df.copy()
        t20_players_df['player_role'] = t20_players_df['player_name'].apply(assign_player_role)
        t20_players_df['country'] = t20_players_df['player_name'].apply(assign_country)
        t20_players_df['batting_style'] = 'Right-handed'
        t20_players_df['bowling_style'] = 'Right-arm medium'
        t20_players_df['is_active'] = True
        
        # Insert T20 players
        for _, row in t20_players_df.iterrows():
            cursor.execute('''
                INSERT INTO players (player_id, player_name, country, batting_style, bowling_style, player_role, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['player_id'], row['player_name'], row['country'], 
                  row['batting_style'], row['bowling_style'], row['player_role'], row['is_active']))
        
        print(f"Loaded {len(t20_players_df)} T20 players")
        
        # Show some examples
        cursor.execute('''
            SELECT player_name, country, player_role 
            FROM players 
            WHERE player_name IN ('V Kohli', 'MS Dhoni', 'JJ Bumrah', 'R Sharma', 'Imad Wasim')
            ORDER BY player_name
        ''')
        examples = cursor.fetchall()
        
        print("\nFamous T20 players in database:")
        for name, country, role in examples:
            print(f"  {name} - {country} - {role}")
        
        # Show role distribution
        cursor.execute('SELECT player_role, COUNT(*) FROM players GROUP BY player_role ORDER BY COUNT(*) DESC')
        roles = cursor.fetchall()
        
        print(f"\nRole distribution:")
        for role, count in roles:
            print(f"  {role}: {count} players")
        
    except Exception as e:
        print(f"Error loading data: {e}")

    conn.commit()
    conn.close()
    print(f"\nT20-only database created successfully!")
    print(f"Database: {db_path}")
    print("Only players who actually played T20 matches are included!")

if __name__ == "__main__":
    create_t20_only_database()