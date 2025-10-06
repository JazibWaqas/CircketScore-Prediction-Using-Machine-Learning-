#!/usr/bin/env python3
"""
Test Improved API with Real Feature Data
"""

import requests
import json

def test_improved_api():
    """Test the API with the same match to see if predictions improved"""
    
    print("🧪 TESTING IMPROVED API WITH REAL FEATURE DATA")
    print("=" * 60)
    
    # Test with Australia vs New Zealand at Eden Park (match_id: 211048)
    payload = {
        "team_a_id": 3,  # Australia
        "team_b_id": 108,  # New Zealand
        "venue_id": 119,  # Eden Park
        "team_a_players": [],
        "team_b_players": [],
        "match_context": {
            "isHomeTeam": True,
            "isFinal": False,
            "isPlayoff": False,
            "isT20WorldCup": False,
            "isBilateral": True,
            "isImportantMatch": False,
            "seasonYear": 2005,  # Match was in 2005
            "seasonMonth": 2,   # February
            "isWinter": True,
            "isSummer": False,
            "isMonsoon": False,
            "tournamentType": "bilateral"
        },
        "model": "random_forest"
    }
    
    print("Match: Australia vs New Zealand at Eden Park (2005-02-17)")
    print("Actual Scores: Australia 214, New Zealand 170")
    print("-" * 60)
    
    try:
        response = requests.post(
            "http://localhost:5000/api/predict", 
            json=payload, 
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction', {})
            
            predicted_score = prediction.get('predicted_score_a', 'N/A')
            confidence = prediction.get('confidence', 'N/A')
            
            print(f"✅ API Response received")
            print(f"Predicted Score: {predicted_score}")
            print(f"Confidence: {confidence}")
            
            # Calculate accuracy improvement
            actual_score = 214
            if predicted_score != 'N/A':
                error = abs(predicted_score - actual_score)
                print(f"\n📊 ACCURACY ANALYSIS:")
                print(f"Actual Score: {actual_score}")
                print(f"Predicted Score: {predicted_score}")
                print(f"Error: {error} runs")
                print(f"Accuracy: {(1 - error/actual_score)*100:.1f}%")
                
                # Compare with previous prediction (130)
                previous_prediction = 130
                previous_error = abs(previous_prediction - actual_score)
                improvement = previous_error - error
                
                print(f"\n🎯 IMPROVEMENT ANALYSIS:")
                print(f"Previous Prediction: {previous_prediction} (Error: {previous_error})")
                print(f"New Prediction: {predicted_score} (Error: {error})")
                print(f"Improvement: {improvement} runs")
                
                if improvement > 0:
                    print(f"✅ PREDICTION IMPROVED by {improvement} runs!")
                else:
                    print(f"❌ Prediction got worse by {abs(improvement)} runs")
            else:
                print(f"❌ Could not parse prediction score")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to API server")
        print("Make sure the API server is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with multiple models for comparison
    print(f"\n🔄 TESTING ALL MODELS:")
    print("-" * 40)
    
    models = ["linear_regression", "random_forest", "xgboost"]
    results = {}
    
    for model in models:
        payload["model"] = model
        
        try:
            response = requests.post(
                "http://localhost:5000/api/predict", 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', {})
                score = prediction.get('predicted_score_a', 'N/A')
                results[model] = score
                
                if score != 'N/A':
                    error = abs(score - 214)
                    print(f"{model.upper()}: {score} (Error: {error})")
                else:
                    print(f"{model.upper()}: {score}")
            else:
                print(f"{model.upper()}: Error {response.status_code}")
                
        except Exception as e:
            print(f"{model.upper()}: {e}")
    
    print(f"\n📈 FINAL COMPARISON:")
    print(f"Actual Score: 214")
    for model, score in results.items():
        if score != 'N/A':
            error = abs(score - 214)
            accuracy = (1 - error/214)*100
            print(f"{model.upper()}: {score} (Error: {error}, Accuracy: {accuracy:.1f}%)")

if __name__ == "__main__":
    test_improved_api()
