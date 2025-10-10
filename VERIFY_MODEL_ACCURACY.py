"""
VERIFY MODEL ACCURACY - Test real matches and show actual predictions
"""
import pandas as pd
import requests
import json

print("="*100)
print("VERIFYING ODI MODEL WITH REAL HISTORICAL MATCHES")
print("="*100)

# Load some real international matches from the dataset
df = pd.read_csv(r'C:\Users\OMNIBOOK\Documents\GitHub\CircketScore-Prediction-Using-Machine-Learning-\ODI\data\odi_complete_dataset.csv')

# Filter for international teams only
international_teams = ['India', 'Australia', 'Pakistan', 'England', 'South Africa', 'New Zealand', 
                       'Sri Lanka', 'West Indies', 'Bangladesh', 'Zimbabwe']

int_matches = df[df['team'].isin(international_teams) & df['opposition'].isin(international_teams)]
print(f"\nFound {len(int_matches)} international matches")

# Take 20 random samples
test_matches = int_matches.sample(min(20, len(int_matches)), random_state=42)

print(f"\n{'='*100}")
print(f"TESTING {len(test_matches)} REAL INTERNATIONAL MATCHES")
print(f"{'='*100}\n")

API_URL = "http://localhost:5001/api/odi/predict"

results = []
total_error = 0
total_abs_error = 0

for idx, match in test_matches.iterrows():
    team = match['team']
    opposition = match['opposition']
    venue = match['venue']
    actual_score = match['total_runs']
    
    # Simple test - no players selected, just team averages
    prediction_data = {
        'team_a': team,
        'team_b': opposition,
        'team_a_players': [],  # Empty for now
        'team_b_players': [],
        'match_context': {
            'venue': venue,
            'year': int(match['season_year']) if pd.notna(match['season_year']) else 2020,
            'month': int(match['season_month']) if pd.notna(match['season_month']) else 6,
            'tournament': 'ODI',
            'toss_winner': team if match.get('toss_won', 1) == 1 else opposition,
            'toss_decision': 'bat',
            'pitch_bounce': 1.0,
            'pitch_swing': 0.8,
            'humidity': 60,
            'temperature': 25
        }
    }
    
    try:
        response = requests.post(API_URL, json=prediction_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                predicted_a = result.get('final_prediction_a', 0)
                error = predicted_a - actual_score
                abs_error = abs(error)
                
                total_error += error
                total_abs_error += abs_error
                
                results.append({
                    'team': team,
                    'vs': opposition,
                    'venue': venue[:30],
                    'actual': actual_score,
                    'predicted': predicted_a,
                    'error': error
                })
                
                status = "[OK]" if abs_error <= 30 else "[!!]"
                print(f"{status} {team[:15]:<15} vs {opposition[:15]:<15} | Actual: {actual_score:>3.0f} | Predicted: {predicted_a:>3.0f} | Error: {error:>+4.0f}")
            else:
                print(f"[FAIL] {team} vs {opposition} - API error: {result.get('error', 'Unknown')}")
        else:
            error_msg = response.json().get('error', 'Unknown') if response.headers.get('content-type', '').startswith('application/json') else response.text[:100]
            print(f"[FAIL] {team} vs {opposition} - HTTP {response.status_code}: {error_msg}")
            
    except Exception as e:
        print(f"[FAIL] {team} vs {opposition} - Exception: {e}")

if results:
    print(f"\n{'='*100}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*100}")
    
    avg_error = total_error / len(results)
    mae = total_abs_error / len(results)
    
    within_20 = sum(1 for r in results if abs(r['error']) <= 20)
    within_30 = sum(1 for r in results if abs(r['error']) <= 30)
    within_40 = sum(1 for r in results if abs(r['error']) <= 40)
    
    print(f"\nMean Error (Bias): {avg_error:+.1f} runs")
    print(f"Mean Absolute Error (MAE): {mae:.1f} runs")
    print(f"\nAccuracy within ±20 runs: {within_20}/{len(results)} ({100*within_20/len(results):.1f}%)")
    print(f"Accuracy within ±30 runs: {within_30}/{len(results)} ({100*within_30/len(results):.1f}%)")
    print(f"Accuracy within ±40 runs: {within_40}/{len(results)} ({100*within_40/len(results):.1f}%)")
    
    print(f"\n{'='*100}")
    print(f"INTERPRETATION:")
    print(f"{'='*100}")
    if mae < 25:
        print(f"[+] EXCELLENT - MAE < 25 runs")
    elif mae < 35:
        print(f"[~] GOOD - MAE between 25-35 runs")
    elif mae < 45:
        print(f"[!] ACCEPTABLE - MAE between 35-45 runs")
    else:
        print(f"[-] POOR - MAE > 45 runs")
    
    if within_30 / len(results) > 0.7:
        print(f"[+] GOOD ACCURACY - {100*within_30/len(results):.0f}% predictions within +/-30 runs")
    else:
        print(f"[!] MODERATE ACCURACY - {100*within_30/len(results):.0f}% predictions within +/-30 runs")
else:
    print("\n[-] No successful predictions. Is the API running on http://localhost:5001?")
    print("  Start it with: cd ODI/Database && python run_odi_api_COMPLETE.py")

