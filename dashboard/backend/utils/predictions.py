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
            # Use get_batsman_avg which handles defaults based on role
            avg = get_batsman_avg(player_name, player_db)
            batting_avgs.append(avg)  # Always add (uses actual or role-based default)
        
        # Calculate from all 11 players (no more "if < 5 then default entire team")
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
            player_data = player_db.get(player_name) or {}
            bowling_data = player_data.get('bowling') or {}

            # Try to get actual economy
            if bowling_data:
                economy = bowling_data.get('economy')
                # Check if economy exists and is valid (not None, not 0)
                if economy is not None and economy > 0:
                    bowling_economies.append(float(economy))
                else:
                    # Use role-based default if missing
                    role = player_data.get('role') or 'Batsman'
                    if 'Bowler' in role:
                        bowling_economies.append(5.0)
                    elif 'All-rounder' in role:
                        bowling_economies.append(5.5)
                    else:
                        bowling_economies.append(6.0)  # Batsman who bowls occasionally
            else:
                # Player not in database or no bowling data - use role-based default
                role = player_data.get('role') or 'Batsman'
                if 'Bowler' in role:
                    bowling_economies.append(5.0)
                elif 'All-rounder' in role:
                    bowling_economies.append(5.5)
                else:
                    bowling_economies.append(6.0)  # Batsman who bowls occasionally
        
        # Calculate from all players (use defaults for missing)
        if len(bowling_economies) == 0:
            # Fallback if no bowling data at all
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
    """
    Get batting average for a specific player
    Uses actual average if available, otherwise role-based default:
    - Batter: 30
    - All-rounder: 25
    - Bowler: 18
    """
    player_data = player_db.get(player_name) or {}
    batting_data = player_data.get('batting') or {}

    # Try to get actual average from database
    if batting_data:
        avg = batting_data.get('average')
        # Check if average exists and is valid (not None, not 0)
        if avg is not None and avg > 0:  # Has actual data
            return float(avg)
    
    # Only if truly missing (None, 0, or not in DB), use role-based default
    role = player_data.get('role') or 'Batsman'
    if 'Bowler' in role:
        return 18.0
    elif 'All-rounder' in role:
        return 25.0
    else:  # Batsman
        return 30.0

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
    """
    Determine match stage from balls bowled
    Based on RESULTS.md: Pre-Match (0-60), Early (60-120), Mid (120-180), Late (180-240), Death (240+)
    """
    if balls_bowled <= 60:
        return 'pre-match'
    elif balls_bowled <= 120:
        return 'early'
    elif balls_bowled <= 180:
        return 'mid'
    elif balls_bowled <= 240:
        return 'late'
    else:
        return 'death'

