import pickle
import pandas as pd
import numpy as np

print("\nTesting What-If Player Scenarios...")

# Load model
pipe = pickle.load(open('models/odi_progressive_pipe.pkl', 'rb'))

# Test scenario: Mid-match at 30 overs
base_scenario = {
    'batting_team': 'India',
    'city': 'Mumbai',
    'current_score': 180,
    'balls_left': 120,
    'wickets_left': 7,
    'crr': 6.0,
    'last_10_overs': 65,
    'team_batting_avg': 38.5  # Average team
}

# Predict with average team
df_avg = pd.DataFrame([base_scenario])
pred_avg = pipe.predict(df_avg)[0]

# Predict with elite team
scenario_elite = base_scenario.copy()
scenario_elite['team_batting_avg'] = 45.0  # Elite team (Kohli, Rohit, etc.)
df_elite = pd.DataFrame([scenario_elite])
pred_elite = pipe.predict(df_elite)[0]

# Predict with weak team
scenario_weak = base_scenario.copy()
scenario_weak['team_batting_avg'] = 32.0  # Weaker team
df_weak = pd.DataFrame([scenario_weak])
pred_weak = pipe.predict(df_weak)[0]

print(f"\nScenario: 180/3 after 30 overs at Mumbai\n")
print(f"Average Team (avg 38.5): {pred_avg:.0f} runs")
print(f"Elite Team (avg 45.0):   {pred_elite:.0f} runs (+{pred_elite-pred_avg:.0f})")
print(f"Weak Team (avg 32.0):    {pred_weak:.0f} runs ({pred_weak-pred_avg:+.0f})")

impact_range = pred_elite - pred_weak
print(f"\nTeam quality impact: {impact_range:.0f} runs (Elite vs Weak)")

if abs(impact_range) > 15:
    print("\n✅ FANTASY FEATURES WORKING! Team composition has clear impact.")
else:
    print(f"\n⚠ Team impact is small ({impact_range:.0f} runs)")

print(f"\n✓ What-if testing complete")

with open('results/whatif_test.txt', 'w') as f:
    f.write("What-If Test Results\n")
    f.write("="*50 + "\n\n")
    f.write(f"Scenario: 180/3 after 30 overs\n\n")
    f.write(f"Average Team (38.5 avg): {pred_avg:.0f} runs\n")
    f.write(f"Elite Team (45.0 avg):   {pred_elite:.0f} runs (+{pred_elite-pred_avg:.0f})\n")
    f.write(f"Weak Team (32.0 avg):    {pred_weak:.0f} runs ({pred_weak-pred_avg:+.0f})\n\n")
    f.write(f"Team impact range: {impact_range:.0f} runs\n")

print("Results saved to results/whatif_test.txt")

