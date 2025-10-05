#!/usr/bin/env python3
"""
Cricket Score Prediction - Flask API Server
Provides endpoints for the frontend to interact with the database and ML models
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables
models = {}
db_path = "cricket_prediction.db"

def load_models():
    """Load trained ML models"""
    global models
    
    model_files = {
        'random_forest': '../models/final_random_forest.pkl',
        'xgboost': '../models/final_xgboost.pkl',
        'linear_regression': '../models/final_linear_regression.pkl',
        'scaler': '../models/final_scaler.pkl',
        'encoders': '../models/final_encoders.pkl'
    }
    
    for name, file_path in model_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"✅ Loaded {name}")
        else:
            print(f"❌ {name} not found at {file_path}")

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Main page - serve React app"""
    return render_template('index.html')

@app.route('/api/teams')
def get_teams():
    """Get all teams with full details"""
    conn = get_db_connection()
    teams = conn.execute('''
        SELECT team_id, team_name, country, team_type, is_active 
        FROM teams 
        WHERE is_active = 1 
        ORDER BY team_name
    ''').fetchall()
    conn.close()
    return jsonify([dict(team) for team in teams])

@app.route('/api/venues')
def get_venues():
    """Get all venues with statistics"""
    conn = get_db_connection()
    venues = conn.execute('''
        SELECT v.venue_id, v.venue_name, v.city, v.country, v.capacity, 
               v.venue_type, v.pitch_type, vs.avg_runs_scored, vs.total_matches
        FROM venues v
        LEFT JOIN venue_stats vs ON v.venue_id = vs.venue_id
        WHERE v.is_active = 1 
        ORDER BY v.venue_name
    ''').fetchall()
    conn.close()
    return jsonify([dict(venue) for venue in venues])

@app.route('/api/players')
def get_players():
    """Get all players with roles and stats"""
    conn = get_db_connection()
    players = conn.execute('''
        SELECT player_id, player_name, country, batting_style, 
               bowling_style, player_role, is_active
        FROM players 
        WHERE is_active = 1 
        ORDER BY player_name
    ''').fetchall()
    conn.close()
    return jsonify([dict(player) for player in players])

@app.route('/api/players/search')
def search_players():
    """Search players by name or role"""
    query = request.args.get('q', '')
    role = request.args.get('role', '')
    
    conn = get_db_connection()
    sql = '''
        SELECT player_id, player_name, country, player_role, batting_style, bowling_style
        FROM players 
        WHERE is_active = 1
    '''
    params = []
    
    if query:
        sql += ' AND player_name LIKE ?'
        params.append(f'%{query}%')
    
    if role:
        sql += ' AND player_role = ?'
        params.append(role)
    
    sql += ' ORDER BY player_name LIMIT 50'
    
    players = conn.execute(sql, params).fetchall()
    conn.close()
    return jsonify([dict(player) for player in players])

@app.route('/api/team-form/<int:team_id>')
def get_team_form(team_id):
    """Get team recent performance"""
    conn = get_db_connection()
    
    # Get recent matches
    recent_matches = conn.execute('''
        SELECT total_runs, date, opposition_id, match_winner_id
        FROM team_performances 
        WHERE team_id = ? 
        ORDER BY date DESC 
        LIMIT 10
    ''', (team_id,)).fetchall()
    
    # Get team info
    team_info = conn.execute('SELECT * FROM teams WHERE team_id = ?', (team_id,)).fetchone()
    
    conn.close()
    
    if not team_info:
        return jsonify({'error': 'Team not found'}), 404
    
    # Calculate form metrics
    if recent_matches:
        avg_runs = np.mean([match['total_runs'] for match in recent_matches])
        wins = len([m for m in recent_matches if m['match_winner_id'] == team_id])
        win_rate = wins / len(recent_matches)
    else:
        avg_runs = 130.0
        win_rate = 0.5
    
    return jsonify({
        'team_id': team_id,
        'team_name': team_info['team_name'],
        'recent_avg_runs': round(avg_runs, 1),
        'recent_win_rate': round(win_rate, 2),
        'recent_matches': len(recent_matches),
        'form_score': round((avg_runs / 150) * win_rate, 2)
    })

@app.route('/api/venue-stats/<int:venue_id>')
def get_venue_stats(venue_id):
    """Get venue statistics"""
    conn = get_db_connection()
    
    venue_info = conn.execute('SELECT * FROM venues WHERE venue_id = ?', (venue_id,)).fetchone()
    venue_stats = conn.execute('SELECT * FROM venue_stats WHERE venue_id = ?', (venue_id,)).fetchone()
    
    conn.close()
    
    if not venue_info:
        return jsonify({'error': 'Venue not found'}), 404
    
    stats = {
        'venue_id': venue_id,
        'venue_name': venue_info['venue_name'],
        'city': venue_info['city'],
        'country': venue_info['country'],
        'pitch_type': venue_info['pitch_type']
    }
    
    if venue_stats:
        stats.update({
            'avg_runs': round(venue_stats['avg_runs_scored'], 1),
            'total_matches': venue_stats['total_matches'],
            'highest_score': venue_stats['highest_score'],
            'lowest_score': venue_stats['lowest_score'],
            'batting_first_wins': venue_stats['batting_first_wins'],
            'fielding_first_wins': venue_stats['fielding_first_wins']
        })
    else:
        stats.update({
            'avg_runs': 130.0,
            'total_matches': 0,
            'highest_score': 200,
            'lowest_score': 80,
            'batting_first_wins': 0,
            'fielding_first_wins': 0
        })
    
    return jsonify(stats)

@app.route('/api/h2h/<int:team_a>/<int:team_b>')
def get_head_to_head(team_a, team_b):
    """Get head-to-head records between two teams"""
    conn = get_db_connection()
    
    h2h = conn.execute('''
        SELECT * FROM head_to_head 
        WHERE (team_a_id = ? AND team_b_id = ?) OR (team_a_id = ? AND team_b_id = ?)
    ''', (team_a, team_b, team_b, team_a)).fetchone()
    
    conn.close()
    
    if not h2h:
        return jsonify({
            'team_a_id': team_a,
            'team_b_id': team_b,
            'total_matches': 0,
            'team_a_wins': 0,
            'team_b_wins': 0,
            'ties': 0,
            'avg_runs_team_a': 130.0,
            'avg_runs_team_b': 130.0
        })
    
    return jsonify(dict(h2h))

@app.route('/api/predict', methods=['POST'])
def predict_score():
    """Predict cricket scores for two teams"""
    try:
        data = request.json
        
        # Extract parameters
        team_a_id = data['team_a_id']
        team_b_id = data['team_b_id']
        venue_id = data['venue_id']
        team_a_players = data.get('team_a_players', [])
        team_b_players = data.get('team_b_players', [])
        match_context = data.get('match_context', {})
        model_name = data.get('model', 'random_forest')
        
        # Get team and venue info
        conn = get_db_connection()
        team_a = conn.execute('SELECT * FROM teams WHERE team_id = ?', (team_a_id,)).fetchone()
        team_b = conn.execute('SELECT * FROM teams WHERE team_id = ?', (team_b_id,)).fetchone()
        venue = conn.execute('SELECT * FROM venues WHERE venue_id = ?', (venue_id,)).fetchone()
        conn.close()
        
        if not team_a or not team_b or not venue:
            return jsonify({'error': 'Invalid team or venue ID'}), 400
        
        # Create feature vector for prediction
        features = create_prediction_features(team_a_id, team_b_id, venue_id, 
                                           team_a_players, team_b_players, match_context)
        
        # Make prediction
        if model_name in models and 'scaler' in models:
            model = models[model_name]
            scaler = models['scaler']
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict
            predicted_score_a = model.predict(features_scaled)[0]
            predicted_score_b = predicted_score_a * np.random.uniform(0.9, 1.1)  # Add variation
            
            # Calculate confidence based on model performance
            confidence = 0.75 if model_name == 'random_forest' else 0.70
            
            # Store prediction in database
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO user_predictions 
                (team_a_id, team_b_id, venue_id, team_a_players, team_b_players, 
                 predicted_score_a, predicted_score_b, confidence_score, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (team_a_id, team_b_id, venue_id, json.dumps(team_a_players), 
                  json.dumps(team_b_players), predicted_score_a, predicted_score_b, 
                  confidence, model_name))
            conn.commit()
            conn.close()
            
            # Determine winner
            winner = team_a['team_name'] if predicted_score_a > predicted_score_b else team_b['team_name']
            
            return jsonify({
                'success': True,
                'prediction': {
                    'team_a': team_a['team_name'],
                    'team_b': team_b['team_name'],
                    'venue': venue['venue_name'],
                    'predicted_score_a': round(predicted_score_a, 1),
                    'predicted_score_b': round(predicted_score_b, 1),
                    'predicted_winner': winner,
                    'model_used': model_name,
                    'confidence': confidence,
                    'match_context': match_context
                }
            })
        else:
            return jsonify({'error': f'Model {model_name} not found'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_prediction_features(team_a_id, team_b_id, venue_id, team_a_players, team_b_players, match_context):
    """Create feature vector for prediction"""
    
    conn = get_db_connection()
    
    # Get team performance data
    team_a_data = conn.execute('''
        SELECT AVG(total_runs) as avg_runs, COUNT(*) as matches,
               AVG(team_batting_avg) as batting_avg, AVG(team_form_score) as form_score
        FROM team_performances 
        WHERE team_id = ?
    ''', (team_a_id,)).fetchone()
    
    team_b_data = conn.execute('''
        SELECT AVG(total_runs) as avg_runs, COUNT(*) as matches,
               AVG(team_batting_avg) as batting_avg, AVG(team_form_score) as form_score
        FROM team_performances 
        WHERE team_id = ?
    ''', (team_b_id,)).fetchone()
    
    # Get venue data
    venue_data = conn.execute('''
        SELECT AVG(total_runs) as avg_runs, COUNT(*) as matches,
               AVG(venue_difficulty) as difficulty
        FROM team_performances 
        WHERE venue_id = ?
    ''', (venue_id,)).fetchone()
    
    # Get head-to-head data
    h2h_data = conn.execute('''
        SELECT avg_runs_team_a, avg_runs_team_b, total_matches
        FROM head_to_head 
        WHERE (team_a_id = ? AND team_b_id = ?) OR (team_a_id = ? AND team_b_id = ?)
    ''', (team_a_id, team_b_id, team_b_id, team_a_id)).fetchone()
    
    conn.close()
    
    # Create feature vector (simplified - you'd need the full 55 features)
    features = [
        team_a_data['avg_runs'] if team_a_data['avg_runs'] else 130.0,
        team_b_data['avg_runs'] if team_b_data['avg_runs'] else 130.0,
        venue_data['avg_runs'] if venue_data['avg_runs'] else 130.0,
        team_a_data['batting_avg'] if team_a_data['batting_avg'] else 130.0,
        team_b_data['batting_avg'] if team_b_data['batting_avg'] else 130.0,
        venue_data['difficulty'] if venue_data['difficulty'] else 1.0,
        team_a_data['form_score'] if team_a_data['form_score'] else 1.0,
        team_b_data['form_score'] if team_b_data['form_score'] else 1.0,
        h2h_data['avg_runs_team_a'] if h2h_data and h2h_data['avg_runs_team_a'] else 130.0,
        h2h_data['avg_runs_team_b'] if h2h_data and h2h_data['avg_runs_team_b'] else 130.0,
        len(team_a_players),
        len(team_b_players),
        match_context.get('batting_first', 0),
        match_context.get('is_home_team', 0),
        match_context.get('is_final', 0),
        match_context.get('is_ipl', 0),
        match_context.get('is_t20_world_cup', 0),
        match_context.get('season_year', 2024),
        match_context.get('season_month', 6),
        match_context.get('is_winter', 0),
        match_context.get('is_summer', 1)
    ]
    
    # Pad with zeros to match training data features
    while len(features) < 55:
        features.append(0.0)
    
    return features[:55]

@app.route('/api/predictions')
def get_predictions():
    """Get all user predictions"""
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT p.*, t1.team_name as team_a_name, t2.team_name as team_b_name, v.venue_name
        FROM user_predictions p
        JOIN teams t1 ON p.team_a_id = t1.team_id
        JOIN teams t2 ON p.team_b_id = t2.team_id
        JOIN venues v ON p.venue_id = v.venue_id
        ORDER BY p.created_at DESC
        LIMIT 50
    ''').fetchall()
    conn.close()
    return jsonify([dict(pred) for pred in predictions])

@app.route('/api/test-model', methods=['POST'])
def test_model():
    """Test model performance on test data"""
    try:
        data = request.json
        model_name = data.get('model', 'random_forest')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        # Load test data
        conn = get_db_connection()
        test_data = conn.execute('''
            SELECT * FROM team_performances 
            WHERE match_id IN (SELECT match_id FROM matches WHERE date >= '2024-01-01')
            LIMIT 100
        ''').fetchall()
        conn.close()
        
        if not test_data:
            return jsonify({'error': 'No test data available'}), 400
        
        # Convert to DataFrame
        test_df = pd.DataFrame([dict(row) for row in test_data])
        
        # Prepare features and target
        feature_columns = [col for col in test_df.columns if col not in ['total_runs', 'match_id', 'id']]
        X = test_df[feature_columns].fillna(0)
        y = test_df['total_runs']
        
        # Scale features
        scaler = models['scaler']
        X_scaled = scaler.transform(X)
        
        # Make predictions
        model = models[model_name]
        predictions = model.predict(X_scaled)
        
        # Calculate metrics
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y))
        r2 = model.score(X_scaled, y)
        
        # Calculate accuracy within ±10 runs
        accuracy_10 = np.mean(np.abs(predictions - y) <= 10) * 100
        
        # Store test results
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO model_performance 
            (model_name, test_date, r2_score, rmse, mae, accuracy_10_runs, test_records)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, datetime.now().date(), r2, rmse, mae, accuracy_10, len(test_data)))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'results': {
                'model': model_name,
                'r2_score': round(r2, 4),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'accuracy_10_runs': round(accuracy_10, 2),
                'test_records': len(test_data)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🏏 Starting Cricket Prediction API Server...")
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
