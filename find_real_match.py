import pandas as pd
import json

# Load the test dataset to find a real Pakistan vs India match
print("=== LOOKING FOR REAL PAKISTAN vs INDIA MATCHES ===")
test_df = pd.read_csv('data/test_dataset.csv')

# Look for Pakistan vs India matches
pak_ind_matches = test_df[
    ((test_df['team'].str.contains('Pakistan', case=False, na=False)) & 
     (test_df['opposition'].str.contains('India', case=False, na=False))) |
    ((test_df['team'].str.contains('India', case=False, na=False)) & 
     (test_df['opposition'].str.contains('Pakistan', case=False, na=False)))
]

print(f"Found {len(pak_ind_matches)} Pakistan vs India matches in test dataset")

if len(pak_ind_matches) > 0:
    # Show the first few matches
    print("\n=== REAL PAKISTAN vs INDIA MATCHES ===")
    for i, match in pak_ind_matches.head(3).iterrows():
        print(f"\n--- MATCH {i+1} ---")
        print(f"Date: {match.get('date', 'Unknown')}")
        print(f"Venue: {match.get('venue', 'Unknown')}")
        print(f"Team: {match.get('team', 'Unknown')}")
        print(f"Opposition: {match.get('opposition', 'Unknown')}")
        print(f"Total Runs: {match.get('total_runs', 'Unknown')}")
        print(f"Batting First: {match.get('batting_first', 'Unknown')}")
        print(f"Toss Winner: {match.get('toss_winner', 'Unknown')}")
        print(f"Toss Decision: {match.get('toss_decision', 'Unknown')}")
        print(f"Match Winner: {match.get('match_winner', 'Unknown')}")
        print(f"Win Type: {match.get('win_type', 'Unknown')}")
        print(f"Event: {match.get('event_name', 'Unknown')}")
        print(f"Season: {match.get('season', 'Unknown')}")
        
        # Try to get player information
        if 'team_players' in match and pd.notna(match['team_players']):
            try:
                players = json.loads(match['team_players']) if isinstance(match['team_players'], str) else match['team_players']
                print(f"Team Players: {players[:5]}...")  # Show first 5 players
            except:
                print(f"Team Players: {match['team_players']}")
else:
    print("No Pakistan vs India matches found in test dataset")
    print("\n=== LOOKING FOR ANY PAKISTAN MATCHES ===")
    pak_matches = test_df[test_df['team'].str.contains('Pakistan', case=False, na=False)]
    print(f"Found {len(pak_matches)} Pakistan matches")
    
    if len(pak_matches) > 0:
        print("\n=== SAMPLE PAKISTAN MATCHES ===")
        for i, match in pak_matches.head(2).iterrows():
            print(f"\n--- MATCH {i+1} ---")
            print(f"Date: {match.get('date', 'Unknown')}")
            print(f"Venue: {match.get('venue', 'Unknown')}")
            print(f"Team: {match.get('team', 'Unknown')}")
            print(f"Opposition: {match.get('opposition', 'Unknown')}")
            print(f"Total Runs: {match.get('total_runs', 'Unknown')}")
            print(f"Event: {match.get('event_name', 'Unknown')}")

print("\n=== LOOKING FOR ANY INDIA MATCHES ===")
ind_matches = test_df[test_df['team'].str.contains('India', case=False, na=False)]
print(f"Found {len(ind_matches)} India matches")

if len(ind_matches) > 0:
    print("\n=== SAMPLE INDIA MATCHES ===")
    for i, match in ind_matches.head(2).iterrows():
        print(f"\n--- MATCH {i+1} ---")
        print(f"Date: {match.get('date', 'Unknown')}")
        print(f"Venue: {match.get('venue', 'Unknown')}")
        print(f"Team: {match.get('team', 'Unknown')}")
        print(f"Opposition: {match.get('opposition', 'Unknown')}")
        print(f"Total Runs: {match.get('total_runs', 'Unknown')}")
        print(f"Event: {match.get('event_name', 'Unknown')}")
