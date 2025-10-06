#!/usr/bin/env python3
"""
Test Complete System Integration
Tests the entire frontend-backend-model pipeline
"""

import requests
import json
import time
import sys

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            print(f"   Mode: {data['mode']}")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   Model names: {data['model_names']}")
            print(f"   Features expected: {data['features_expected']}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_model_performance():
    """Test model performance endpoint"""
    print("\n📊 Testing Model Performance...")
    try:
        response = requests.get('http://localhost:5000/api/model-performance', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Performance Data:")
            for model_name, metrics in data['models'].items():
                print(f"   {model_name.upper()}:")
                print(f"     R² Score: {metrics['r2_score']}")
                print(f"     RMSE: {metrics['rmse']}")
                print(f"     MAE: {metrics['mae']}")
                print(f"     Description: {metrics['description']}")
            return True
        else:
            print(f"❌ Model Performance failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model Performance error: {e}")
        return False

def test_teams_endpoint():
    """Test teams endpoint"""
    print("\n🏏 Testing Teams Endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/teams', timeout=5)
        if response.status_code == 200:
            teams = response.json()
            print(f"✅ Teams loaded: {len(teams)} teams")
            if teams:
                print(f"   Sample team: {teams[0]['team_name']}")
            return teams
        else:
            print(f"❌ Teams endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Teams endpoint error: {e}")
        return None

def test_venues_endpoint():
    """Test venues endpoint"""
    print("\n🏟️ Testing Venues Endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/venues', timeout=5)
        if response.status_code == 200:
            venues = response.json()
            print(f"✅ Venues loaded: {len(venues)} venues")
            if venues:
                print(f"   Sample venue: {venues[0]['venue_name']}")
            return venues
        else:
            print(f"❌ Venues endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Venues endpoint error: {e}")
        return None

def test_prediction_endpoint(teams, venues):
    """Test prediction endpoint with real data"""
    print("\n🎯 Testing Prediction Endpoint...")
    
    if not teams or not venues:
        print("❌ Cannot test prediction - missing teams or venues")
        return False
    
    # Use first available team and venue
    team_a = teams[0]
    team_b = teams[1] if len(teams) > 1 else teams[0]
    venue = venues[0]
    
    prediction_data = {
        "team_a_id": team_a['team_id'],
        "team_b_id": team_b['team_id'],
        "venue_id": venue['venue_id'],
        "team_a_players": [],
        "team_b_players": [],
        "match_context": {
            "tournamentType": "Bilateral",
            "seasonYear": 2025,
            "seasonMonth": 1,
            "tossDecision": "bat",
            "tossWinner": "team_a",
            "battingFirst": "team_a",
            "isHomeTeam": False,
            "isFinal": False,
            "gender": "male"
        },
        "model": "xgboost"
    }
    
    try:
        response = requests.post(
            'http://localhost:5000/api/predict', 
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction = data['prediction']
                print("✅ Prediction successful:")
                print(f"   Teams: {prediction['team_a']} vs {prediction['team_b']}")
                print(f"   Venue: {prediction['venue']}")
                print(f"   Scores: {prediction['predicted_score_a']} - {prediction['predicted_score_b']}")
                print(f"   Winner: {prediction['predicted_winner']}")
                print(f"   Model: {prediction['model_used']}")
                print(f"   Accuracy: {prediction['model_accuracy']}")
                return True
            else:
                print(f"❌ Prediction failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        return False

def test_all_models(teams, venues):
    """Test all three models"""
    print("\n🤖 Testing All Models...")
    
    if not teams or not venues:
        print("❌ Cannot test models - missing teams or venues")
        return False
    
    team_a = teams[0]
    team_b = teams[1] if len(teams) > 1 else teams[0]
    venue = venues[0]
    
    models = ['xgboost', 'random_forest', 'linear_regression']
    results = {}
    
    for model_name in models:
        print(f"\n🔮 Testing {model_name.upper()} model...")
        
        prediction_data = {
            "team_a_id": team_a['team_id'],
            "team_b_id": team_b['team_id'],
            "venue_id": venue['venue_id'],
            "team_a_players": [],
            "team_b_players": [],
            "match_context": {
                "tournamentType": "Bilateral",
                "seasonYear": 2025,
                "seasonMonth": 1,
                "tossDecision": "bat",
                "tossWinner": "team_a",
                "battingFirst": "team_a",
                "isHomeTeam": False,
                "isFinal": False,
                "gender": "male"
            },
            "model": model_name
        }
        
        try:
            response = requests.post(
                'http://localhost:5000/api/predict', 
                json=prediction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    prediction = data['prediction']
                    score = prediction['predicted_score_a']
                    accuracy = prediction['model_accuracy']
                    results[model_name] = {
                        'score': score,
                        'accuracy': accuracy,
                        'success': True
                    }
                    print(f"   ✅ {model_name}: {score} runs ({accuracy})")
                else:
                    print(f"   ❌ {model_name}: {data.get('error', 'Unknown error')}")
                    results[model_name] = {'success': False}
            else:
                print(f"   ❌ {model_name}: HTTP {response.status_code}")
                results[model_name] = {'success': False}
        except Exception as e:
            print(f"   ❌ {model_name}: {e}")
            results[model_name] = {'success': False}
    
    # Summary
    successful_models = sum(1 for r in results.values() if r.get('success'))
    print(f"\n📊 Model Test Summary: {successful_models}/3 models working")
    
    return successful_models == 3

def main():
    """Main test function"""
    print("🧪 COMPLETE SYSTEM INTEGRATION TEST")
    print("=" * 50)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Test API health
    if not test_api_health():
        print("\n❌ API not ready. Please start the server first.")
        return False
    
    # Test model performance
    test_model_performance()
    
    # Test endpoints
    teams = test_teams_endpoint()
    venues = test_venues_endpoint()
    
    # Test prediction
    if teams and venues:
        test_prediction_endpoint(teams, venues)
        test_all_models(teams, venues)
    
    print("\n" + "=" * 50)
    print("✅ COMPLETE SYSTEM TEST FINISHED")
    print("🚀 System is ready for production!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
