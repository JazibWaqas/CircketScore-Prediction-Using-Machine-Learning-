import numpy as np
import pandas as pd

def calculate_team_aggregates(players, player_db):
    """
    Calculate batting team aggregates from list of 11 players
    
    Returns:
        dict with team_batting_avg, team_elite_batsmen, team_batting_depth
    """
    try:
        batting_avgs = []
        
        for player_name in players:
            if player_name in player_db and 'batting' in player_db[player_name]:
                avg = player_db[player_name]['batting'].get('average', 0)
                if avg > 0:
                    batting_avgs.append(avg)
        
        if len(batting_avgs) < 5:
            result = {
                'team_batting_avg': 35.0,
                'team_elite_batsmen': 0,
                'team_batting_depth': 0
            }
        else:
            result = {
                'team_batting_avg': np.mean(batting_avgs),
                'team_elite_batsmen': sum(1 for avg in batting_avgs if avg >= 40),
                'team_batting_depth': sum(1 for avg in batting_avgs if avg >= 30)
            }
        
        print(f"Batting aggregates calculated successfully: {result}")
        return result
        
    except Exception as e:
        print(f"ERROR in calculate_team_aggregates: {e}")
        print(f"Players: {players}")
        print(f"Player DB type: {type(player_db)}")
        # Return default values instead of None
        return {
            'team_batting_avg': 35.0,
            'team_elite_batsmen': 0,
            'team_batting_depth': 0
        }

def calculate_bowling_aggregates(players, player_db):
    """
    Calculate bowling aggregates from list of 11 opposition players
    
    Returns:
        dict with opp_bowling_economy, opp_elite_bowlers, opp_bowling_depth
    """
    try:
        bowling_economies = []
        
        for player_name in players:
            if player_name in player_db and 'bowling' in player_db[player_name]:
                economy = player_db[player_name]['bowling'].get('economy', 0)
                if economy > 0:
                    bowling_economies.append(economy)
        
        if len(bowling_economies) < 3:
            result = {
                'opp_bowling_economy': 5.5,
                'opp_elite_bowlers': 0,
                'opp_bowling_depth': 0
            }
        else:
            result = {
                'opp_bowling_economy': np.mean(bowling_economies),
                'opp_elite_bowlers': sum(1 for e in bowling_economies if e < 4.8),
                'opp_bowling_depth': len(bowling_economies)
            }
        
        print(f"Bowling aggregates calculated successfully: {result}")
        return result
        
    except Exception as e:
        print(f"ERROR in calculate_bowling_aggregates: {e}")
        print(f"Players: {players}")
        print(f"Player DB type: {type(player_db)}")
        # Return default values instead of None
        return {
            'opp_bowling_economy': 5.5,
            'opp_elite_bowlers': 0,
            'opp_bowling_depth': 0
        }

def get_batsman_avg(player_name, player_db):
    """Get batting average for a specific player"""
    if player_name in player_db and 'batting' in player_db[player_name]:
        return player_db[player_name]['batting'].get('average', 35.0)
    return 35.0

def make_prediction(model, scenario_data):
    """
    Make prediction using the trained model
    
    Args:
        model: Trained sklearn pipeline
        scenario_data: dict with all required features
    
    Returns:
        float: Predicted final score
    """
    # Create DataFrame with exact feature order
    df = pd.DataFrame([{
        'current_score': scenario_data['current_score'],
        'wickets_fallen': scenario_data['wickets_fallen'],
        'balls_bowled': scenario_data['balls_bowled'],
        'balls_remaining': scenario_data['balls_remaining'],
        'runs_last_10_overs': scenario_data['runs_last_10_overs'],
        'current_run_rate': scenario_data['current_run_rate'],
        'team_batting_avg': scenario_data['team_batting_avg'],
        'team_elite_batsmen': scenario_data['team_elite_batsmen'],
        'team_batting_depth': scenario_data['team_batting_depth'],
        'opp_bowling_economy': scenario_data['opp_bowling_economy'],
        'opp_elite_bowlers': scenario_data['opp_elite_bowlers'],
        'opp_bowling_depth': scenario_data['opp_bowling_depth'],
        'venue_avg_score': scenario_data['venue_avg_score'],
        'batsman_1_avg': scenario_data.get('batsman_1_avg', 0),
        'batsman_2_avg': scenario_data.get('batsman_2_avg', 0),
        'venue': scenario_data['venue']
    }])
    
    prediction = model.predict(df)[0]
    return float(prediction)

def calculate_confidence_interval(mae, stage):
    """
    Calculate confidence interval based on match stage
    
    Args:
        mae: Mean absolute error for this stage
        stage: 'pre-match', 'early', 'mid', 'late', 'death'
    
    Returns:
        tuple: (lower, upper, confidence_label)
    """
    stage_mae = {
        'pre-match': 41,
        'early': 29,
        'mid': 24,
        'late': 18,
        'death': 12
    }
    
    stage_r2 = {
        'pre-match': 0.35,
        'early': 0.62,
        'mid': 0.75,
        'late': 0.86,
        'death': 0.94
    }
    
    mae_value = stage_mae.get(stage, 25)
    r2_value = stage_r2.get(stage, 0.70)
    
    if r2_value >= 0.90:
        confidence = "Very High"
    elif r2_value >= 0.80:
        confidence = "High"
    elif r2_value >= 0.65:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return mae_value, r2_value, confidence

def get_match_stage(balls_bowled):
    """Determine match stage from balls bowled"""
    if balls_bowled <= 10:
        return 'pre-match'
    elif balls_bowled <= 60:
        return 'early'
    elif balls_bowled <= 120:
        return 'mid'
    elif balls_bowled <= 240:
        return 'late'
    else:
        return 'death'

