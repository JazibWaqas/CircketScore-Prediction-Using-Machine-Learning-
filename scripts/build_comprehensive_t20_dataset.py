"""
Build Comprehensive T20 Dataset from Ball-by-Ball JSON Files
Extracts team-level performance data with player lineups and context
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_match_data(json_file_path):
    """Extract comprehensive data from a single T20 match JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
        
        # Basic match info
        info = match_data.get('info', {})
        match_id = os.path.basename(json_file_path).replace('.json', '')
        
        # Match details
        date = info.get('dates', [''])[0] if info.get('dates') else ''
        venue = info.get('venue', '')
        teams = info.get('teams', [])
        toss_winner = info.get('toss', {}).get('winner', '')
        toss_decision = info.get('toss', {}).get('decision', '')
        match_winner = info.get('outcome', {}).get('winner', '')
        player_of_match = info.get('player_of_match', [''])[0] if info.get('player_of_match') else ''
        season = info.get('season', '')
        event_name = info.get('event', {}).get('name', '') if info.get('event') else ''
        match_number = info.get('event', {}).get('match_number', '') if info.get('event') else ''
        gender = info.get('gender', '')
        
        # Player lineups
        players = info.get('players', {})
        
        # Process each team's innings
        team_data = []
        innings_data = match_data.get('innings', [])
        
        for i, innings in enumerate(innings_data):
            team = innings.get('team', '')
            if not team:
                continue
                
            # Get team players
            team_players = players.get(team, [])
            
            # Calculate team performance metrics
            total_runs = 0
            total_wickets = 0
            total_balls = 0
            total_4s = 0
            total_6s = 0
            total_extras = 0
            powerplay_runs = 0
            middle_overs_runs = 0
            death_overs_runs = 0
            
            # Process each over
            overs = innings.get('overs', [])
            for over_data in overs:
                over_num = over_data.get('over', 0)
                deliveries = over_data.get('deliveries', [])
                
                for delivery in deliveries:
                    runs_data = delivery.get('runs', {})
                    batter_runs = runs_data.get('batter', 0)
                    extras = runs_data.get('extras', 0)
                    total_runs_delivery = runs_data.get('total', 0)
                    
                    total_runs += total_runs_delivery
                    total_balls += 1
                    total_extras += extras
                    
                    # Count boundaries
                    if batter_runs == 4:
                        total_4s += 1
                    elif batter_runs == 6:
                        total_6s += 1
                    
                    # Phase-wise runs
                    if over_num < 6:
                        powerplay_runs += total_runs_delivery
                    elif over_num < 16:
                        middle_overs_runs += total_runs_delivery
                    else:
                        death_overs_runs += total_runs_delivery
                    
                    # Count wickets (simplified - would need to check for wicket data)
                    # This is a simplified approach - in reality, wicket data is more complex
            
            # Calculate derived metrics
            total_overs = total_balls / 6.0 if total_balls > 0 else 0
            run_rate = total_runs / total_overs if total_overs > 0 else 0
            total_boundaries = total_4s + total_6s
            
            # Determine opposition
            opposition = [t for t in teams if t != team]
            opposition = opposition[0] if opposition else ''
            
            # Determine if batting first
            batting_first = (i == 0)
            
            # Create team record
            team_record = {
                'match_id': match_id,
                'date': date,
                'venue': venue,
                'team': team,
                'opposition': opposition,
                'team_players': team_players,
                'total_runs': total_runs,
                'total_wickets': total_wickets,
                'total_overs': total_overs,
                'total_balls': total_balls,
                'run_rate': run_rate,
                'total_4s': total_4s,
                'total_6s': total_6s,
                'total_boundaries': total_boundaries,
                'total_extras': total_extras,
                'powerplay_runs': powerplay_runs,
                'middle_overs_runs': middle_overs_runs,
                'death_overs_runs': death_overs_runs,
                'batting_first': batting_first,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'match_winner': match_winner,
                'player_of_match': player_of_match,
                'season': season,
                'event_name': event_name,
                'match_number': match_number,
                'gender': gender,
                'teams': teams
            }
            
            team_data.append(team_record)
        
        return team_data
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return []

def calculate_venue_stats(df):
    """Calculate venue-specific statistics"""
    venue_stats = df.groupby('venue').agg({
        'total_runs': ['mean', 'std', 'count', 'max', 'min']
    }).round(2)
    
    # Flatten the multi-level columns
    venue_stats.columns = ['venue_avg_runs', 'venue_runs_std', 'venue_matches', 'venue_high_score', 'venue_low_score']
    venue_stats = venue_stats.reset_index()
    
    return venue_stats

def calculate_h2h_stats(df):
    """Calculate head-to-head statistics between teams"""
    h2h_stats = []
    
    for team in df['team'].unique():
        team_matches = df[df['team'] == team]
        for opposition in team_matches['opposition'].unique():
            h2h_matches = team_matches[team_matches['opposition'] == opposition]
            
            if len(h2h_matches) > 0:
                h2h_record = {
                    'team': team,
                    'opposition': opposition,
                    'h2h_matches': len(h2h_matches),
                    'h2h_avg_runs': h2h_matches['total_runs'].mean(),
                    'h2h_win_rate': (h2h_matches['match_winner'] == team).mean(),
                    'h2h_last_meeting': h2h_matches['date'].max()
                }
                h2h_stats.append(h2h_record)
    
    return pd.DataFrame(h2h_stats)

def calculate_team_form(df, window=5):
    """Calculate team form (recent performance)"""
    team_form = []
    
    for team in df['team'].unique():
        team_matches = df[df['team'] == team].sort_values('date')
        
        for i in range(len(team_matches)):
            if i >= window - 1:
                recent_matches = team_matches.iloc[i-window+1:i+1]
                form_avg_runs = recent_matches['total_runs'].mean()
                form_win_rate = (recent_matches['match_winner'] == team).mean()
                
                team_form.append({
                    'match_id': team_matches.iloc[i]['match_id'],
                    'team': team,
                    'team_form_avg_runs': form_avg_runs,
                    'team_form_win_rate': form_win_rate
                })
    
    return pd.DataFrame(team_form)

def build_comprehensive_dataset():
    """Build the comprehensive T20 dataset"""
    print("Building Comprehensive T20 Dataset")
    print("=" * 50)
    
    # Get all JSON files
    json_dir = "t20 matches ball by ball"
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} T20 match files")
    
    all_team_data = []
    
    # Process each match file
    for i, json_file in enumerate(json_files):  # Process ALL files
        if i % 100 == 0:
            print(f"Processing match {i+1}/{len(json_files)}")
        
        json_path = os.path.join(json_dir, json_file)
        team_data = extract_match_data(json_path)
        all_team_data.extend(team_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_team_data)
    
    if df.empty:
        print("No data extracted!")
        return None
    
    print(f"Extracted {len(df)} team performances")
    
    # Calculate additional features
    print("Calculating venue statistics...")
    venue_stats = calculate_venue_stats(df)
    df = df.merge(venue_stats, on='venue', how='left')
    
    print("Calculating head-to-head statistics...")
    h2h_stats = calculate_h2h_stats(df)
    df = df.merge(h2h_stats, on=['team', 'opposition'], how='left')
    
    print("Calculating team form...")
    team_form = calculate_team_form(df)
    df = df.merge(team_form, on=['match_id', 'team'], how='left')
    
    # Add derived features
    df['is_home_team'] = df.apply(lambda x: x['venue'].lower() in x['team'].lower(), axis=1)
    df['is_final'] = df['event_name'].str.contains('final', case=False, na=False)
    df['is_semi_final'] = df['event_name'].str.contains('semi', case=False, na=False)
    df['is_playoff'] = df['is_final'] | df['is_semi_final']
    df['target_set'] = df.apply(lambda x: x['total_runs'] if x['batting_first'] else None, axis=1)
    df['target_chased'] = df.apply(lambda x: x['total_runs'] if not x['batting_first'] else None, axis=1)
    df['win_margin'] = df.apply(lambda x: abs(x['total_runs'] - x['target_set']) if x['target_set'] else None, axis=1)
    df['win_type'] = df.apply(lambda x: 'wickets' if x['match_winner'] == x['team'] and not x['batting_first'] else 'runs', axis=1)
    
    # Fill missing values
    df['h2h_matches'] = df['h2h_matches'].fillna(0)
    df['h2h_avg_runs'] = df['h2h_avg_runs'].fillna(df['total_runs'].mean())
    df['h2h_win_rate'] = df['h2h_win_rate'].fillna(0.5)
    df['team_form_avg_runs'] = df['team_form_avg_runs'].fillna(df['total_runs'].mean())
    df['team_form_win_rate'] = df['team_form_win_rate'].fillna(0.5)
    
    # Save dataset
    output_file = 'comprehensive_t20_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Dataset saved as {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display sample
    print("\nSample data:")
    print(df[['match_id', 'team', 'opposition', 'venue', 'total_runs', 'team_players']].head())
    
    return df

if __name__ == "__main__":
    df = build_comprehensive_dataset()
