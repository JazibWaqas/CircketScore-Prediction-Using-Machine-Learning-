"""
TEST ON REAL 2024-2025 MATCHES - TRUE VALIDATION

Parse recent matches, extract features, test model predictions.
This verifies the model actually works on unseen real data.
"""

import pandas as pd
import pickle
import json
import os
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

print("\n" + "="*80)
print("TESTING ON REAL RECENT MATCHES (2024-2025)")
print("="*80)

# Load model (running from ODI_Progressive folder)
pipe = pickle.load(open('models/odi_progressive_pipe.pkl', 'rb'))
# Player DB - only using for reference (from parent ODI folder)
player_db = json.load(open('../ODI/data/CURRENT_player_database_977_quality.json'))

print("\n✓ Model loaded")

# ==============================================================================
# PARSE RECENT MATCHES (2024-2025)
# ==============================================================================

print("\n[1/2] Finding and parsing recent matches...")

ballbyball_dir = '../raw_data/odis_ballbyBall'
all_files = os.listdir(ballbyball_dir)

# Find matches from 2024-2025
recent_matches = []

for file in all_files:
    if not file.endswith('.json'):
        continue
    
    try:
        filepath = os.path.join(ballbyball_dir, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            match = json.load(f)
        
        date = match['info'].get('dates', ['2020-01-01'])[0]
        
        # Only 2024-2025 matches
        if date >= '2024-01-01':
            recent_matches.append((file, date, match))
    except:
        continue

recent_matches.sort(key=lambda x: x[1])

print(f"   Found {len(recent_matches)} matches from 2024-2025")

if len(recent_matches) == 0:
    print("\n   ⚠ No 2024-2025 matches found in dataset")
    print("   Testing on 2023 matches instead...")
    
    # Try 2023
    for file in all_files:
        if not file.endswith('.json'):
            continue
        try:
            filepath = os.path.join(ballbyball_dir, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                match = json.load(f)
            date = match['info'].get('dates', ['2020-01-01'])[0]
            if date >= '2023-01-01' and date < '2024-01-01':
                recent_matches.append((file, date, match))
        except:
            continue
    
    recent_matches.sort(key=lambda x: x[1])
    print(f"   Found {len(recent_matches)} matches from 2023")

# ==============================================================================
# TEST PREDICTIONS
# ==============================================================================

print(f"\n[2/2] Testing model on {min(50, len(recent_matches))} recent matches...")

test_results = []

for file, date, match in recent_matches[:50]:
    try:
        innings = match['innings'][0]
        batting_team = innings.get('team')
        venue = match['info'].get('venue', 'Unknown')
        city = match['info'].get('city', venue.split(',')[0] if ',' in venue else venue)
        
        # Calculate final score
        final_score = 0
        for over in innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                final_score += delivery.get('runs', {}).get('total', 0)
        
        # Get team players
        players = match['info'].get('players', {}).get(batting_team, [])
        batting_avgs = []
        for p in players:
            if p in player_db and 'batting' in player_db[p]:
                batting_avgs.append(player_db[p]['batting'].get('average', 0))
        team_avg = np.mean(batting_avgs) if batting_avgs else 35.0
        
        # Test at 30 overs (mid-match)
        # Calculate state at 30 overs
        score_at_30 = 0
        wickets_at_30 = 0
        balls_at_30 = 0
        recent_runs = []
        
        for over in innings.get('overs', []):
            if over.get('over', 0) >= 30:
                break
            for delivery in over.get('deliveries', []):
                balls_at_30 += 1
                runs = delivery.get('runs', {}).get('total', 0)
                score_at_30 += runs
                recent_runs.append(runs)
                if 'wickets' in delivery:
                    wickets_at_30 += len(delivery['wickets'])
        
        if balls_at_30 < 170:  # Need at least ~28 overs
            continue
        
        last_10 = sum(recent_runs[-60:]) if len(recent_runs) >= 60 else sum(recent_runs)
        
        # Predict
        scenario = pd.DataFrame([{
            'batting_team': batting_team,
            'city': city,
            'current_score': score_at_30,
            'balls_left': 300 - balls_at_30,
            'wickets_left': 10 - wickets_at_30,
            'crr': (score_at_30 * 6.0 / balls_at_30) if balls_at_30 > 0 else 0,
            'last_10_overs': last_10,
            'team_batting_avg': team_avg
        }])
        
        predicted = pipe.predict(scenario)[0]
        error = predicted - final_score
        
        test_results.append({
            'date': date,
            'team': batting_team,
            'score_at_30': score_at_30,
            'wickets_at_30': wickets_at_30,
            'actual_final': final_score,
            'predicted': predicted,
            'error': error
        })
        
    except Exception as e:
        continue

# ==============================================================================
# ANALYZE RESULTS
# ==============================================================================

if len(test_results) > 0:
    df_results = pd.DataFrame(test_results)
    
    mae = df_results['error'].abs().mean()
    actuals = df_results['actual_final'].values
    preds = df_results['predicted'].values
    r2 = r2_score(actuals, preds) if len(actuals) > 1 else 0
    
    print(f"\n" + "="*80)
    print(f"REAL MATCH VALIDATION RESULTS")
    print(f"="*80)
    
    print(f"\nTested on {len(df_results)} recent real matches")
    print(f"Date range: {df_results['date'].min()} to {df_results['date'].max()}")
    
    print(f"\n📊 PERFORMANCE ON REAL UNSEEN MATCHES:")
    print(f"   R² = {r2:.4f}")
    print(f"   MAE = {mae:.2f} runs")
    
    within_15 = (df_results['error'].abs() <= 15).sum()
    within_30 = (df_results['error'].abs() <= 30).sum()
    
    print(f"\n   Within ±15 runs: {within_15}/{len(df_results)} ({100*within_15/len(df_results):.1f}%)")
    print(f"   Within ±30 runs: {within_30}/{len(df_results)} ({100*within_30/len(df_results):.1f}%)")
    
    # Show examples
    print(f"\n📋 REAL MATCH EXAMPLES:")
    print(f"\n{'Date':<12} {'Team':<15} {'@30 overs':>10} {'Actual':>8} {'Predicted':>10} {'Error':>7}")
    print("-" * 75)
    
    for _, row in df_results.head(15).iterrows():
        print(f"{row['date']:<12} {row['team'][:13]:<15} {row['score_at_30']:>7.0f}/{row['wickets_at_30']:<2.0f} "
              f"{row['actual_final']:>8.0f} {row['predicted']:>10.0f} {row['error']:>+7.0f}")
    
    # Save results
    df_results.to_csv('results/real_match_test.csv', index=False)
    
    with open('results/real_match_validation.txt', 'w') as f:
        f.write("REAL MATCH VALIDATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Tested: {len(df_results)} real matches from {df_results['date'].min()} to {df_results['date'].max()}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  R² = {r2:.4f}\n")
        f.write(f"  MAE = {mae:.2f} runs\n\n")
        f.write(f"Accuracy:\n")
        f.write(f"  Within ±15: {100*within_15/len(df_results):.1f}%\n")
        f.write(f"  Within ±30: {100*within_30/len(df_results):.1f}%\n")
    
    print(f"\n✓ Results saved to results/real_match_validation.txt")
    
    # ==============================================================================
    # VERDICT
    # ==============================================================================
    
    print(f"\n" + "="*80)
    print(f"VERDICT")
    print(f"="*80)
    
    if r2 >= 0.85 and mae <= 15:
        print(f"\n✅ MODEL IS REAL AND WORKING!")
        print(f"   - Performs well on unseen recent matches")
        print(f"   - R² = {r2:.3f} on real data")
        print(f"   - Ready for production")
    elif r2 >= 0.70:
        print(f"\n✓ MODEL WORKS ON REAL DATA")
        print(f"   - R² = {r2:.3f} is functional")
        print(f"   - Not perfect but usable")
    else:
        print(f"\n⚠ MODEL STRUGGLES ON REAL DATA")
        print(f"   - R² = {r2:.3f} on recent matches")
        print(f"   - May have overfitting issues")
        print(f"   - Training R² (0.85) > Real test R² ({r2:.3f})")
    
else:
    print("\n❌ Could not test - no recent matches found")

print(f"\n" + "="*80 + "\n")

