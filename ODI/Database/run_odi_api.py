#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ODI Cricket Score Prediction API

Backend API that:
1. Loads trained ML models
2. Calculates aggregated team features from individual player selections
3. Generates predictions for ODI matches
"""

import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import datetime
import os
import sys

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app)

# Global variables
models = {}
player_database = {}
feature_names = []

def load_models():
    """Load trained ML models"""
    global models
    
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)
    
    model_dir = '../models'
    model_files = {
        'linear_regression': 'odi_linear_regression.pkl',
        'random_forest': 'odi_random_forest.pkl',
        'xgboost': 'odi_xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        try:
            models[model_name] = joblib.load(model_path)
            print(f"✅ Loaded {model_name} model")
        except Exception as e:
            print(f"⚠️  Could not load {model_name}: {e}")
            print(f"   Model will be available after training")
    
    if models:
        print(f"\n✅ {len(models)} model(s) loaded successfully")
    else:
        print("\n⚠️  No models loaded yet. Train models first!")
    
    return len(models) > 0

def load_player_database():
    """Load player database with statistics"""
    global player_database
    
    print("\n" + "="*70)
    print("LOADING PLAYER DATABASE")
    print("="*70)
    
    player_db_path = '../data/player_database.json'
    try:
        with open(player_db_path, 'r', encoding='utf-8') as f:
            player_database = json.load(f)
        print(f"✅ Loaded {len(player_database)} players from database")
        return True
    except Exception as e:
        print(f"❌ Error loading player database: {e}")
        return False

def get_db_connection():
    """Get SQLite database connection"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'cricket_prediction_odi.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def calculate_team_stats(player_names):
    """
    Calculate aggregated team statistics from selected players
    This matches the features in the training dataset
    """
    if not player_names or len(player_names) == 0:
        return None
    
    # Initialize stats
    stats = {
        'team_batting_avg': 0,
        'team_strike_rate': 0,
        'team_total_runs': 0,
        'elite_batsmen': 0,
        'star_batsmen': 0,
        'power_hitters': 0,
        'team_bowling_avg': 0,
        'team_economy': 0,
        'team_total_wickets': 0,
        'elite_bowlers': 0,
        'star_bowlers': 0,
        'all_rounder_count': 0,
        'wicketkeeper_count': 0,
        'elite_players': 0,
        'star_players': 0,
        'avg_star_rating': 0,
        'team_balance': 0,
        'team_depth': 0,
        'known_players_count': 0
    }
    
    batting_avgs = []
    strike_rates = []
    bowling_avgs = []
    economies = []
    star_ratings = []
    total_wickets = 0
    total_runs = 0
    
    for player_name in player_names:
        if player_name not in player_database:
            continue
        
        player = player_database[player_name]
        stats['known_players_count'] += 1
        
        # Skill level counts
        skill_level = player.get('skill_level', '')
        if skill_level == 'Elite':
            stats['elite_players'] += 1
        elif skill_level == 'Star':
            stats['star_players'] += 1
        
        # Star rating
        star_rating = player.get('star_rating', 0)
        if star_rating > 0:
            star_ratings.append(star_rating)
        
        # Role counts
        role = player.get('role', '')
        if 'All-rounder' in role:
            stats['all_rounder_count'] += 1
        if 'Wicketkeeper' in role:
            stats['wicketkeeper_count'] += 1
        
        # Batting stats
        batting = player.get('batting') or {}
        if batting:
            bat_avg = batting.get('average', 0)
            bat_sr = batting.get('strike_rate', 0)
            bat_runs = batting.get('total_runs', 0)
            
            if bat_avg > 0:
                batting_avgs.append(bat_avg)
            if bat_sr > 0:
                strike_rates.append(bat_sr)
            if bat_runs > 0:
                total_runs += bat_runs
            
            # Check if elite/star batsman
            if skill_level == 'Elite' and 'Bats' in role:
                stats['elite_batsmen'] += 1
            if skill_level == 'Star' and 'Bats' in role:
                stats['star_batsmen'] += 1
            
            # Power hitters (SR > 90)
            if bat_sr > 90:
                stats['power_hitters'] += 1
        
        # Bowling stats
        bowling = player.get('bowling') or {}
        if bowling:
            bowl_avg = bowling.get('average', 0)
            bowl_econ = bowling.get('economy', 0)
            bowl_wickets = bowling.get('total_wickets', 0)
            
            if bowl_avg > 0:
                bowling_avgs.append(bowl_avg)
            if bowl_econ > 0:
                economies.append(bowl_econ)
            if bowl_wickets > 0:
                total_wickets += bowl_wickets
            
            # Check if elite/star bowler
            if skill_level == 'Elite' and 'Bowl' in role:
                stats['elite_bowlers'] += 1
            if skill_level == 'Star' and 'Bowl' in role:
                stats['star_bowlers'] += 1
    
    # Calculate averages
    stats['team_batting_avg'] = np.mean(batting_avgs) if batting_avgs else 0
    stats['team_strike_rate'] = np.mean(strike_rates) if strike_rates else 0
    stats['team_bowling_avg'] = np.mean(bowling_avgs) if bowling_avgs else 0
    stats['team_economy'] = np.mean(economies) if economies else 0
    stats['team_total_runs'] = total_runs
    stats['team_total_wickets'] = total_wickets
    stats['avg_star_rating'] = np.mean(star_ratings) if star_ratings else 0
    
    # Calculate team balance (ratio of specialists to all-rounders)
    if stats['known_players_count'] > 0:
        stats['team_balance'] = stats['all_rounder_count'] / stats['known_players_count']
    
    # Calculate team depth (how many roles covered)
    roles_covered = 0
    if stats['elite_batsmen'] > 0 or stats['star_batsmen'] > 0:
        roles_covered += 1
    if stats['elite_bowlers'] > 0 or stats['star_bowlers'] > 0:
        roles_covered += 1
    if stats['all_rounder_count'] > 0:
        roles_covered += 1
    if stats['wicketkeeper_count'] > 0:
        roles_covered += 1
    stats['team_depth'] = roles_covered
    
    return stats

def build_feature_vector(team_a_stats, team_b_stats, match_context):
    """
    Build feature vector matching the training dataset format
    """
    features = {}
    
    # Match context features
    features['season_year'] = match_context.get('season_year', 2024)
    features['season_month'] = match_context.get('season_month', 1)
    features['gender'] = match_context.get('gender', 'male')
    features['match_type'] = 'ODI'
    features['toss_won'] = 1 if match_context.get('toss_won') == 'team_a' else 0
    features['toss_decision_bat'] = 1 if match_context.get('toss_decision') == 'bat' else 0
    features['toss_decision_field'] = 1 if match_context.get('toss_decision') == 'field' else 0
    
    # Team A stats (prefixed with 'team_')
    for key, value in team_a_stats.items():
        features[f'team_{key}'] = value
    
    # Team B stats (prefixed with 'opp_')
    for key, value in team_b_stats.items():
        features[f'opp_{key}'] = value
    
    # Comparative features
    features['batting_advantage'] = team_a_stats['team_batting_avg'] - team_b_stats['team_batting_avg']
    features['star_advantage'] = team_a_stats['star_players'] - team_b_stats['star_players']
    features['elite_advantage'] = team_a_stats['elite_players'] - team_b_stats['elite_players']
    
    return features

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health')
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'running',
        'models_loaded': len(models),
        'players_loaded': len(player_database)
    })

@app.route('/api/teams')
def get_teams():
    """Get all teams"""
    conn = get_db_connection()
    teams = conn.execute('SELECT * FROM teams WHERE is_active = 1 ORDER BY team_name').fetchall()
    conn.close()
    return jsonify([dict(team) for team in teams])

@app.route('/api/venues')
def get_venues():
    """Get all venues"""
    conn = get_db_connection()
    venues = conn.execute('SELECT * FROM venues WHERE is_active = 1 ORDER BY venue_name').fetchall()
    conn.close()
    return jsonify([dict(venue) for venue in venues])

@app.route('/api/players')
def get_players():
    """Get all players"""
    conn = get_db_connection()
    
    # Get team filter if provided
    team_filter = request.args.get('team', None)
    
    if team_filter:
        # Filter players by team
        players = conn.execute('''
            SELECT * FROM players 
            WHERE is_active = 1 AND teams LIKE ?
            ORDER BY star_rating DESC, player_name
        ''', (f'%{team_filter}%',)).fetchall()
    else:
        players = conn.execute('''
            SELECT * FROM players 
            WHERE is_active = 1
            ORDER BY star_rating DESC, player_name
        ''').fetchall()
    
    conn.close()
    return jsonify([dict(player) for player in players])

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make ODI score prediction
    
    Expected request body:
    {
        "team_a_id": 1,
        "team_b_id": 2,
        "venue_id": 10,
        "team_a_players": ["Player 1", "Player 2", ...],
        "team_b_players": ["Player 1", "Player 2", ...],
        "match_context": {
            "season_year": 2024,
            "season_month": 10,
            "gender": "male",
            "toss_won": "team_a",
            "toss_decision": "bat"
        },
        "model": "random_forest"
    }
    """
    try:
        if not models:
            return jsonify({
                'error': 'No models loaded. Please train models first.',
                'success': False
            }), 503
        
        data = request.get_json()
        
        # Extract data
        team_a_players = data.get('team_a_players', [])
        team_b_players = data.get('team_b_players', [])
        match_context = data.get('match_context', {})
        model_name = data.get('model', 'random_forest')
        
        # Validate
        if not team_a_players or not team_b_players:
            return jsonify({
                'error': 'Please select players for both teams',
                'success': False
            }), 400
        
        if model_name not in models:
            return jsonify({
                'error': f'Model {model_name} not available',
                'success': False
            }), 400
        
        # Calculate team stats
        team_a_stats = calculate_team_stats(team_a_players)
        team_b_stats = calculate_team_stats(team_b_players)
        
        if not team_a_stats or not team_b_stats:
            return jsonify({
                'error': 'Could not calculate team statistics',
                'success': False
            }), 400
        
        # Build feature vector
        features = build_feature_vector(team_a_stats, team_b_stats, match_context)
        
        # Make prediction
        model = models[model_name]
        
        # Convert to DataFrame for model prediction
        # (models expect specific feature order)
        feature_df = pd.DataFrame([features])
        
        # Predict
        prediction = model.predict(feature_df)[0]
        
        # Store prediction in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (team_a_id, team_b_id, venue_id, team_a_players, team_b_players, 
             predicted_score_a, model_used, match_date, gender, toss_won, toss_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('team_a_id'),
            data.get('team_b_id'),
            data.get('venue_id'),
            json.dumps(team_a_players),
            json.dumps(team_b_players),
            float(prediction),
            model_name,
            match_context.get('match_date', datetime.date.today().isoformat()),
            match_context.get('gender', 'male'),
            match_context.get('toss_won', 'team_a'),
            match_context.get('toss_decision', 'bat')
        ))
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'predicted_score': round(float(prediction), 0),
            'model_used': model_name,
            'team_a_stats': team_a_stats,
            'team_b_stats': team_b_stats,
            'features_used': len(features)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/predictions/history')
def get_prediction_history():
    """Get recent predictions"""
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT p.*, 
               ta.team_name as team_a_name,
               tb.team_name as team_b_name,
               v.venue_name
        FROM predictions p
        JOIN teams ta ON p.team_a_id = ta.team_id
        JOIN teams tb ON p.team_b_id = tb.team_id
        JOIN venues v ON p.venue_id = v.venue_id
        ORDER BY p.created_at DESC
        LIMIT 50
    ''').fetchall()
    conn.close()
    return jsonify([dict(pred) for pred in predictions])

def initialize_api():
    """Initialize the API with models and data"""
    print("\n" + "="*70)
    print("ODI CRICKET PREDICTION API")
    print("="*70)
    
    # Load player database
    if not load_player_database():
        print("\n⚠️  Warning: Player database not loaded!")
        print("   Predictions will not work until player_database.json is available")
    
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠️  Warning: No models loaded!")
        print("   API will start, but predictions won't work until models are trained")
        print("\n   To train models:")
        print("   1. Create training scripts in ODI/scripts/")
        print("   2. Train and save models to ODI/models/")
        print("   3. Model filenames: odi_linear_regression.pkl, odi_random_forest.pkl, odi_xgboost.pkl")
    
    print("\n" + "="*70)
    print("API READY!")
    print("="*70)
    print("\n🚀 API Endpoints:")
    print("   GET  /api/health              - Check API status")
    print("   GET  /api/teams               - Get all teams")
    print("   GET  /api/venues              - Get all venues")
    print("   GET  /api/players             - Get all players")
    print("   GET  /api/players?team=India  - Get players by team")
    print("   POST /api/predict             - Make prediction")
    print("   GET  /api/predictions/history - Get prediction history")
    print("\n📡 Server starting on http://localhost:5001")
    print("="*70 + "\n")

if __name__ == '__main__':
    initialize_api()
    app.run(debug=True, host='0.0.0.0', port=5001)

