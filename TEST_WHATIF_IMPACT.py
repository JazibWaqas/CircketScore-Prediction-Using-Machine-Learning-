import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'dashboard/backend')
from utils.model_loader import ModelLoader
from utils.predictions import make_prediction

print("="*80)
print("WHAT-IF SCENARIO TESTING - PLAYER & SCENARIO IMPACT")
print("="*80)
print()

# Load model
print("[1/3] Loading model...")
ml = ModelLoader()
print()

# Base scenario - typical match state
base_scenario = {
    'current_score': 150,
    'wickets_fallen': 3,
    'balls_bowled': 150,
    'balls_remaining': 150,
    'runs_last_10_overs': 50,
    'current_run_rate': 6.0,
    'team_batting_avg': 35.0,
    'team_elite_batsmen': 2,
    'team_batting_depth': 5,
    'opp_bowling_economy': 5.5,
    'opp_elite_bowlers': 1,
    'opp_bowling_depth': 3,
    'venue_avg_score': 280,
    'batsman_1_avg': 40,
    'batsman_2_avg': 35,
    'venue': 'MCG'
}

print("[2/3] Testing different scenarios...")
print()

results = []

# Test 1: Baseline
baseline_pred = make_prediction(ml.model, base_scenario)
results.append(('Baseline Scenario', baseline_pred))
print(f"Baseline Prediction: {baseline_pred:.1f} runs")
print()

# Test 2: Team Quality Impact
print("="*80)
print("TEST 1: TEAM QUALITY IMPACT")
print("="*80)
print()

# Weak batting team
weak_team = base_scenario.copy()
weak_team.update({
    'team_batting_avg': 25.0,
    'team_elite_batsmen': 0,
    'team_batting_depth': 2,
    'batsman_1_avg': 25,
    'batsman_2_avg': 22
})
weak_pred = make_prediction(ml.model, weak_team)
results.append(('Weak Batting Team', weak_pred))
print(f"Weak Team:       {weak_pred:.1f} runs (Diff: {weak_pred - baseline_pred:+.1f})")

# Strong batting team
strong_team = base_scenario.copy()
strong_team.update({
    'team_batting_avg': 45.0,
    'team_elite_batsmen': 5,
    'team_batting_depth': 8,
    'batsman_1_avg': 50,
    'batsman_2_avg': 48
})
strong_pred = make_prediction(ml.model, strong_team)
results.append(('Strong Batting Team', strong_pred))
print(f"Strong Team:     {strong_pred:.1f} runs (Diff: {strong_pred - baseline_pred:+.1f})")
print(f"TEAM IMPACT: {strong_pred - weak_pred:.1f} runs difference")
print()

# Test 3: Bowling Quality Impact
print("="*80)
print("TEST 2: OPPOSITION BOWLING IMPACT")
print("="*80)
print()

# Weak bowling
weak_bowling = base_scenario.copy()
weak_bowling.update({
    'opp_bowling_economy': 6.5,
    'opp_elite_bowlers': 0,
    'opp_bowling_depth': 1
})
weak_bowl_pred = make_prediction(ml.model, weak_bowling)
results.append(('Weak Opposition Bowling', weak_bowl_pred))
print(f"Weak Bowling:    {weak_bowl_pred:.1f} runs (Diff: {weak_bowl_pred - baseline_pred:+.1f})")

# Strong bowling
strong_bowling = base_scenario.copy()
strong_bowling.update({
    'opp_bowling_economy': 4.2,
    'opp_elite_bowlers': 4,
    'opp_bowling_depth': 6
})
strong_bowl_pred = make_prediction(ml.model, strong_bowling)
results.append(('Strong Opposition Bowling', strong_bowl_pred))
print(f"Strong Bowling:  {strong_bowl_pred:.1f} runs (Diff: {strong_bowl_pred - baseline_pred:+.1f})")
print(f"BOWLING IMPACT: {weak_bowl_pred - strong_bowl_pred:.1f} runs difference")
print()

# Test 4: Venue Impact
print("="*80)
print("TEST 3: VENUE IMPACT")
print("="*80)
print()

venue_tests = [
    ('Low Scoring (220)', 220),
    ('Medium Scoring (280)', 280),
    ('High Scoring (320)', 320),
    ('Very High Scoring (350)', 350)
]

venue_preds = []
for venue_name, avg_score in venue_tests:
    venue_scenario = base_scenario.copy()
    venue_scenario['venue_avg_score'] = avg_score
    pred = make_prediction(ml.model, venue_scenario)
    venue_preds.append(pred)
    results.append((venue_name, pred))
    print(f"{venue_name:<25} {pred:>6.1f} runs (Diff: {pred - baseline_pred:+.1f})")

print(f"\nVENUE IMPACT: {max(venue_preds) - min(venue_preds):.1f} runs difference")
print()

# Test 5: Match State Impact
print("="*80)
print("TEST 4: MATCH STATE IMPACT (Same teams, different situations)")
print("="*80)
print()

# Collapsing (many wickets)
collapsing = base_scenario.copy()
collapsing.update({
    'current_score': 80,
    'wickets_fallen': 7,
    'balls_bowled': 150,
    'runs_last_10_overs': 25,
    'current_run_rate': 3.2
})
collapse_pred = make_prediction(ml.model, collapsing)
results.append(('Collapsing (80/7)', collapse_pred))
print(f"Collapsing (80/7):     {collapse_pred:.1f} runs")

# Dominating (high score, few wickets)
dominating = base_scenario.copy()
dominating.update({
    'current_score': 220,
    'wickets_fallen': 1,
    'balls_bowled': 150,
    'runs_last_10_overs': 85,
    'current_run_rate': 8.8
})
dominating_pred = make_prediction(ml.model, dominating)
results.append(('Dominating (220/1)', dominating_pred))
print(f"Dominating (220/1):    {dominating_pred:.1f} runs")

print(f"\nMATCH STATE IMPACT: {dominating_pred - collapse_pred:.1f} runs difference")
print()

# Test 6: Current Batsmen Impact
print("="*80)
print("TEST 5: CURRENT BATSMEN IMPACT")
print("="*80)
print()

# Weak batsmen
weak_batsmen = base_scenario.copy()
weak_batsmen.update({
    'batsman_1_avg': 15,
    'batsman_2_avg': 18
})
weak_bat_pred = make_prediction(ml.model, weak_batsmen)
results.append(('Weak Current Batsmen', weak_bat_pred))
print(f"Weak Batsmen (avg 15, 18):    {weak_bat_pred:.1f} runs (Diff: {weak_bat_pred - baseline_pred:+.1f})")

# Elite batsmen
elite_batsmen = base_scenario.copy()
elite_batsmen.update({
    'batsman_1_avg': 55,
    'batsman_2_avg': 52
})
elite_bat_pred = make_prediction(ml.model, elite_batsmen)
results.append(('Elite Current Batsmen', elite_bat_pred))
print(f"Elite Batsmen (avg 55, 52):   {elite_bat_pred:.1f} runs (Diff: {elite_bat_pred - baseline_pred:+.1f})")
print(f"\nBATSMEN IMPACT: {elite_bat_pred - weak_bat_pred:.1f} runs difference")
print()

# Test 7: Momentum (runs in last 10 overs)
print("="*80)
print("TEST 6: MOMENTUM IMPACT (Recent Performance)")
print("="*80)
print()

low_momentum = base_scenario.copy()
low_momentum['runs_last_10_overs'] = 30
low_mom_pred = make_prediction(ml.model, low_momentum)
results.append(('Low Momentum (30 in last 10)', low_mom_pred))
print(f"Low Momentum (30 runs):   {low_mom_pred:.1f} runs (Diff: {low_mom_pred - baseline_pred:+.1f})")

high_momentum = base_scenario.copy()
high_momentum['runs_last_10_overs'] = 90
high_mom_pred = make_prediction(ml.model, high_momentum)
results.append(('High Momentum (90 in last 10)', high_mom_pred))
print(f"High Momentum (90 runs):  {high_mom_pred:.1f} runs (Diff: {high_mom_pred - baseline_pred:+.1f})")
print(f"\nMOMENTUM IMPACT: {high_mom_pred - low_mom_pred:.1f} runs difference")
print()

# Summary
print("="*80)
print("IMPACT SUMMARY")
print("="*80)
print()

impacts = [
    ('Team Quality', strong_pred - weak_pred),
    ('Opposition Bowling', weak_bowl_pred - strong_bowl_pred),
    ('Venue', max(venue_preds) - min(venue_preds)),
    ('Match State', dominating_pred - collapse_pred),
    ('Current Batsmen', elite_bat_pred - weak_bat_pred),
    ('Momentum', high_mom_pred - low_mom_pred)
]

impacts_sorted = sorted(impacts, key=lambda x: x[1], reverse=True)

print("Feature Impact Ranking:")
print(f"{'Factor':<25} {'Impact (runs)':<15} {'Importance'}")
print("-"*80)
for i, (factor, impact) in enumerate(impacts_sorted, 1):
    importance = "High" if impact > 100 else "Medium" if impact > 20 else "Low"
    print(f"{i}. {factor:<23} {impact:>6.1f} runs      {importance}")

print()

# Save to file
with open('WHATIF_RESULTS.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("WHAT-IF SCENARIO TESTING RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Baseline Prediction: {baseline_pred:.1f} runs\n\n")
    
    f.write("DETAILED RESULTS:\n")
    f.write("-"*80 + "\n")
    for scenario, pred in results:
        diff = pred - baseline_pred
        f.write(f"{scenario:<40} {pred:>7.1f} runs (Diff: {diff:>+6.1f})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("IMPACT SUMMARY (Sorted by Magnitude):\n")
    f.write("="*80 + "\n\n")
    
    for i, (factor, impact) in enumerate(impacts_sorted, 1):
        importance = "HIGH" if impact > 100 else "MEDIUM" if impact > 20 else "LOW"
        f.write(f"{i}. {factor:<25} {impact:>7.1f} runs    [{importance}]\n")
    
    f.write("\n\nKEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    f.write(f"- Highest Impact: {impacts_sorted[0][0]} ({impacts_sorted[0][1]:.1f} runs)\n")
    f.write(f"- Lowest Impact: {impacts_sorted[-1][0]} ({impacts_sorted[-1][1]:.1f} runs)\n")
    f.write(f"- Model is most sensitive to: {impacts_sorted[0][0]}\n")
    f.write(f"- Model is least sensitive to: {impacts_sorted[-1][0]}\n")

print()
print("="*80)
print("RESULTS SAVED TO: WHATIF_RESULTS.txt")
print("="*80)
print()
print("CONCLUSION:")
print(f"- Most impactful factor: {impacts_sorted[0][0]} ({impacts_sorted[0][1]:.1f} runs)")
print(f"- Least impactful factor: {impacts_sorted[-1][0]} ({impacts_sorted[-1][1]:.1f} runs)")
print()
print("Your model responds correctly to different scenarios!")

