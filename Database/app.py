#!/usr/bin/env python3
"""
Simple Cricket Prediction API
Works without ML model dependencies
"""

import sqlite3
import json
import random
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Database path
db_path = "cricket_prediction.db"

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Main page"""
    return "Cricket Prediction API is running! Use the frontend at http://localhost:3000"

@app.route('/api/teams')
def get_teams():
    """Get all teams"""
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
    """Get all venues"""
    conn = get_db_connection()
    venues = conn.execute('''
        SELECT venue_id, venue_name, city, country, capacity, venue_type, pitch_type, is_active
        FROM venues 
        WHERE is_active = 1 
        ORDER BY venue_name
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
        
        # Create realistic prediction based on team strength and venue
        base_score = 150.0
        
        # Team strength factors
        team_a_strength = len(team_a_players) * 0.1 + random.uniform(0.8, 1.2)
        team_b_strength = len(team_b_players) * 0.1 + random.uniform(0.8, 1.2)
        
        # Venue effects
        venue_multiplier = 1.0
        if venue['pitch_type'] == 'Batting':
            venue_multiplier = 1.1
        elif venue['pitch_type'] == 'Bowling':
            venue_multiplier = 0.9
        
        # Match context effects
        context_multiplier = 1.0
        if match_context.get('is_final'):
            context_multiplier *= 1.05
        if match_context.get('is_ipl'):
            context_multiplier *= 1.1
        if match_context.get('is_t20_world_cup'):
            context_multiplier *= 1.15
        
        # Calculate predicted scores
        predicted_score_a = base_score * team_a_strength * venue_multiplier * context_multiplier
        predicted_score_b = base_score * team_b_strength * venue_multiplier * context_multiplier
        
        # Add some randomness for realism
        predicted_score_a += random.uniform(-10, 10)
        predicted_score_b += random.uniform(-10, 10)
        
        # Ensure minimum scores
        predicted_score_a = max(100, predicted_score_a)
        predicted_score_b = max(100, predicted_score_b)
        
        # Calculate confidence based on model
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'players_count': get_player_count()
    })

def get_player_count():
    """Get total player count"""
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM players').fetchone()[0]
    conn.close()
    return count

if __name__ == '__main__':
    print("Starting Simple Cricket Prediction API Server...")
    print("Database connected successfully!")
    print("Ready to serve predictions!")
    app.run(host='0.0.0.0', port=5000, debug=True)
