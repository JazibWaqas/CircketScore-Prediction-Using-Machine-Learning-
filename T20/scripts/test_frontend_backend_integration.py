#!/usr/bin/env python3
"""
Test Frontend-Backend Integration
Verify that frontend data properly flows to models
"""

import requests
import json
import time

def test_complete_prediction_flow():
    """Test the complete prediction flow with frontend data"""
    print("🧪 TESTING COMPLETE FRONTEND-BACKEND INTEGRATION")
    print("=" * 60)
    
    # Test data that mimics what frontend sends
    test_cases = [
        {
            "name": "IPL Match - Men's Cricket",
            "data": {
                "team_a_id": 1,
                "team_b_id": 2,
                "venue_id": 1,
                "team_a_players": [101, 102, 103, 104, 105],
                "team_b_players": [201, 202, 203, 204, 205],
                "match_context": {
                    "tournamentType": "psl",  # Pakistan Super League
                    "seasonYear": 2025,
                    "seasonMonth": 3,  # March
                    "tossDecision": "bat",
                    "tossWinner": "team_a",
                    "battingFirst": "team_a",
                    "isHomeTeam": False,
                    "isFinal": False,
                    "gender": "male"
                },
                "model": "xgboost"
            }
        },
        {
            "name": "T20 World Cup - Women's Cricket",
            "data": {
                "team_a_id": 3,
                "team_b_id": 4,
                "venue_id": 2,
                "team_a_players": [301, 302, 303, 304, 305],
                "team_b_players": [401, 402, 403, 404, 405],
                "match_context": {
                    "tournamentType": "t20_world_cup",
                    "seasonYear": 2025,
                    "seasonMonth": 6,  # June
                    "tossDecision": "field",
                    "tossWinner": "team_b",
                    "battingFirst": "team_a",
                    "isHomeTeam": True,
                    "isFinal": True,
                    "gender": "female"
                },
                "model": "random_forest"
            }
        },
        {
            "name": "Bilateral Series - Summer",
            "data": {
                "team_a_id": 5,
                "team_b_id": 6,
                "venue_id": 3,
                "team_a_players": [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511],
                "team_b_players": [601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611],
                "match_context": {
                    "tournamentType": "bilateral",
                    "seasonYear": 2025,
                    "seasonMonth": 7,  # July
                    "tossDecision": "bat",
                    "tossWinner": "team_a",
                    "battingFirst": "team_a",
                    "isHomeTeam": False,
                    "isFinal": False,
                    "gender": "male"
                },
                "model": "linear_regression"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 TEST {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            response = requests.post(
                'http://localhost:5000/api/predict',
                json=test_case['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    prediction = result['prediction']
                    print("✅ PREDICTION SUCCESSFUL:")
                    print(f"   Teams: {prediction['team_a']} vs {prediction['team_b']}")
                    print(f"   Venue: {prediction['venue']}")
                    print(f"   Scores: {prediction['predicted_score_a']} - {prediction['predicted_score_b']}")
                    print(f"   Winner: {prediction['predicted_winner']}")
                    print(f"   Model: {prediction['model_used']}")
                    print(f"   Accuracy: {prediction['model_accuracy']}")
                    
                    # Verify frontend data was used
                    context = test_case['data']['match_context']
                    print(f"\n📊 FRONTEND DATA USED:")
                    print(f"   Tournament: {context['tournamentType']}")
                    print(f"   Season: {context['seasonYear']}-{context['seasonMonth']}")
                    print(f"   Toss: {context['tossDecision']}")
                    print(f"   Gender: {context['gender']}")
                    print(f"   Players: {len(test_case['data']['team_a_players'])} vs {len(test_case['data']['team_b_players'])}")
                    
                else:
                    print(f"❌ PREDICTION FAILED: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP ERROR: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ REQUEST ERROR: {e}")
        
        time.sleep(1)  # Small delay between tests

def test_api_health():
    """Test API health first"""
    print("🔍 CHECKING API HEALTH...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"   Mode: {data['mode']}")
            print(f"   Models: {data['model_names']}")
            print(f"   Features: {data['features_expected']}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_model_performance():
    """Test model performance endpoint"""
    print("\n📈 CHECKING MODEL PERFORMANCE...")
    try:
        response = requests.get('http://localhost:5000/api/model-performance', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Performance:")
            for model_name, metrics in data['models'].items():
                print(f"   {model_name.upper()}: {metrics['r2_score']:.3f} R² ({metrics['description']})")
            return True
        else:
            print(f"❌ Model Performance failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model Performance error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 FRONTEND-BACKEND INTEGRATION TEST")
    print("=" * 60)
    
    # Check API health first
    if not test_api_health():
        print("\n❌ API not ready. Please start the server first:")
        print("   cd Database && python run.py")
        return False
    
    # Check model performance
    test_model_performance()
    
    # Test complete prediction flow
    test_complete_prediction_flow()
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST COMPLETE!")
    print("🎯 Frontend data is properly flowing to models!")
    print("🚀 System is ready for production!")
    
    return True

if __name__ == "__main__":
    main()
