import pandas as pd
import json

# Load dataset
df = pd.read_csv('ODI/data/odi_complete_dataset.csv')

# Find a famous recent match - India vs Pakistan at Dubai
india_pak_dubai = df[
    (df['venue'].str.contains('Dubai', na=False)) & 
    ((df['team'] == 'India') & (df['opposition'] == 'Pakistan'))
]

if len(india_pak_dubai) > 0:
    match = india_pak_dubai.iloc[0]
    print("\n" + "="*80)
    print("REAL MATCH #1: INDIA VS PAKISTAN AT DUBAI")
    print("="*80)
    print(f"Date: {match['date']}")
    print(f"Venue: {match['venue']}")
    print(f"Team: {match['team']}")
    print(f"Opposition: {match['opposition']}")
    print(f"ACTUAL SCORE: {int(match['total_runs'])} runs")
    print(f"Venue avg: {match['venue_avg_runs']:.1f} runs")

# Find another match - England vs Australia
eng_aus = df[
    ((df['team'] == 'England') & (df['opposition'] == 'Australia')) |
    ((df['team'] == 'Australia') & (df['opposition'] == 'England'))
].tail(2)

if len(eng_aus) > 0:
    match2 = eng_aus.iloc[0]
    print("\n" + "="*80)
    print("REAL MATCH #2: ENGLAND VS AUSTRALIA")
    print("="*80)
    print(f"Date: {match2['date']}")
    print(f"Venue: {match2['venue']}")
    print(f"Team: {match2['team']}")
    print(f"Opposition: {match2['opposition']}")
    print(f"ACTUAL SCORE: {int(match2['total_runs'])} runs")
    print(f"Venue avg: {match2['venue_avg_runs']:.1f} runs")

# Load ball-by-ball data to get actual player lineups
import os
ballbyball_dir = 'raw_data/odis_ballbyBall'
match_files = [f for f in os.listdir(ballbyball_dir) if f.endswith('.json')][:5]

print("\n" + "="*80)
print("SAMPLE MATCH WITH PLAYERS")
print("="*80)

for filename in match_files:
    try:
        with open(os.path.join(ballbyball_dir, filename), 'r', encoding='utf-8') as f:
            match_data = json.load(f)
        
        info = match_data['info']
        teams = list(info['players'].keys())
        if len(teams) == 2:
            team_a, team_b = teams[0], teams[1]
            team_a_players = info['players'][team_a]
            team_b_players = info['players'][team_b]
            venue = info.get('venue', 'Unknown')
            date = info.get('dates', ['Unknown'])[0]
            
            # Get scores
            innings = match_data.get('innings', [])
            if len(innings) >= 2:
                score_a = sum(d.get('runs', {}).get('total', 0) 
                             for inning in innings if inning.get('team') == team_a
                             for over in inning.get('overs', [])
                             for d in over.get('deliveries', []))
                score_b = sum(d.get('runs', {}).get('total', 0) 
                             for inning in innings if inning.get('team') == team_b
                             for over in inning.get('overs', [])
                             for d in over.get('deliveries', []))
                
                print(f"\nMatch: {team_a} vs {team_b}")
                print(f"Date: {date}")
                print(f"Venue: {venue}")
                print(f"\n{team_a} XI:")
                for i, p in enumerate(team_a_players[:11], 1):
                    print(f"  {i:2d}. {p}")
                print(f"\n{team_b} XI:")
                for i, p in enumerate(team_b_players[:11], 1):
                    print(f"  {i:2d}. {p}")
                print(f"\nACTUAL SCORES:")
                print(f"  {team_a}: {score_a} runs")
                print(f"  {team_b}: {score_b} runs")
                break
    except:
        continue

print("\n" + "="*80)

