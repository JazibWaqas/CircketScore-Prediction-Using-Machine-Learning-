import pandas as pd
import json

# Load the test dataset
test_df = pd.read_csv('data/test_dataset.csv')

# Get the specific Pakistan vs India match (Match 473)
match = test_df.iloc[473]  # Pakistan batting first
print("=== REAL PAKISTAN vs INDIA T20 MATCH ===")
print(f"Date: {match['date']}")
print(f"Venue: {match['venue']}")
print(f"Event: {match['event_name']}")
print(f"Season: {match['season']}")
print(f"Team: {match['team']}")
print(f"Opposition: {match['opposition']}")
print(f"Total Runs: {match['total_runs']}")
print(f"Batting First: {match['batting_first']}")
print(f"Toss Winner: {match['toss_winner']}")
print(f"Toss Decision: {match['toss_decision']}")
print(f"Match Winner: {match['match_winner']}")
print(f"Win Type: {match['win_type']}")

# Get Pakistan players - handle the data properly
try:
    if pd.notna(match['team_players']):
        pak_players_str = str(match['team_players'])
        print(f"\nPakistan Team Players (Raw): {pak_players_str}")
        
        # Try to parse as JSON
        if pak_players_str.startswith('[') and pak_players_str.endswith(']'):
            pak_players = json.loads(pak_players_str)
        else:
            # If not JSON, split by comma
            pak_players = [p.strip().strip("'\"") for p in pak_players_str.split(',')]
        
        print(f"\nPakistan Team Players:")
        for i, player in enumerate(pak_players, 1):
            print(f"{i}. {player}")
    else:
        print("No Pakistan players data available")
except Exception as e:
    print(f"Error parsing Pakistan players: {e}")

# Get the corresponding India match (Match 474)
india_match = test_df.iloc[474]
print(f"\n=== INDIA MATCH DETAILS ===")
print(f"India Total Runs: {india_match['total_runs']}")
print(f"India Batting First: {india_match['batting_first']}")

try:
    if pd.notna(india_match['team_players']):
        india_players_str = str(india_match['team_players'])
        print(f"\nIndia Team Players (Raw): {india_players_str}")
        
        # Try to parse as JSON
        if india_players_str.startswith('[') and india_players_str.endswith(']'):
            india_players = json.loads(india_players_str)
        else:
            # If not JSON, split by comma
            india_players = [p.strip().strip("'\"") for p in india_players_str.split(',')]
        
        print(f"\nIndia Team Players:")
        for i, player in enumerate(india_players, 1):
            print(f"{i}. {player}")
    else:
        print("No India players data available")
except Exception as e:
    print(f"Error parsing India players: {e}")

print(f"\n=== MATCH SUMMARY ===")
print(f"Pakistan scored: {match['total_runs']} runs")
print(f"India scored: {india_match['total_runs']} runs")
print(f"Winner: {match['match_winner']} by {match['win_type']}")
print(f"Venue: {match['venue']}")
print(f"Tournament: {match['event_name']}")
print(f"Date: {match['date']}")