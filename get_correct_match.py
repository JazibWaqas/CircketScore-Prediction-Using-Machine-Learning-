import pandas as pd

# Load the test dataset
test_df = pd.read_csv('data/test_dataset.csv')

# Find Pakistan vs India matches
pak_ind_matches = test_df[
    ((test_df['team'].str.contains('Pakistan', case=False, na=False)) & 
     (test_df['opposition'].str.contains('India', case=False, na=False))) |
    ((test_df['team'].str.contains('India', case=False, na=False)) & 
     (test_df['opposition'].str.contains('Pakistan', case=False, na=False)))
]

print("=== REAL PAKISTAN vs INDIA T20 MATCHES ===")
for i, (idx, match) in enumerate(pak_ind_matches.iterrows()):
    print(f"\n--- MATCH {i+1} (Index {idx}) ---")
    print(f"Date: {match['date']}")
    print(f"Venue: {match['venue']}")
    print(f"Event: {match['event_name']}")
    print(f"Team: {match['team']}")
    print(f"Opposition: {match['opposition']}")
    print(f"Total Runs: {match['total_runs']}")
    print(f"Batting First: {match['batting_first']}")
    print(f"Toss Winner: {match['toss_winner']}")
    print(f"Toss Decision: {match['toss_decision']}")
    print(f"Match Winner: {match['match_winner']}")
    print(f"Win Type: {match['win_type']}")
    
    # Get players
    if pd.notna(match['team_players']):
        players_str = str(match['team_players'])
        print(f"Team Players: {players_str[:100]}...")  # First 100 chars
    
    print(f"Season: {match.get('season', 'Unknown')}")
    print(f"Event: {match.get('event_name', 'Unknown')}")

# Let's also check for any high-scoring T20 matches
print("\n=== HIGH SCORING T20 MATCHES ===")
high_scores = test_df[test_df['total_runs'] > 150].head(5)
for i, (idx, match) in enumerate(high_scores.iterrows()):
    print(f"\n--- HIGH SCORE MATCH {i+1} ---")
    print(f"Teams: {match['team']} vs {match['opposition']}")
    print(f"Total Runs: {match['total_runs']}")
    print(f"Venue: {match['venue']}")
    print(f"Date: {match['date']}")
    print(f"Event: {match.get('event_name', 'Unknown')}")
