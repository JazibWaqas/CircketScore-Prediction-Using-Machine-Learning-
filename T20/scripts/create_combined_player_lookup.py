#!/usr/bin/env python3
"""
Create Combined Player Lookup - Full Database
Bridge the gap between ball-by-ball player IDs and career stats IDs
Process ALL players without limitations
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import warnings
warnings.filterwarnings('ignore')

class CombinedPlayerLookupBuilder:
    def __init__(self):
        print("Loading all player data...")
        
        # Load ball-by-ball player lookup (ALL players)
        self.ball_by_ball_lookup = pd.read_csv("data/player_lookup.csv")
        print(f"Ball-by-ball lookup: {len(self.ball_by_ball_lookup)} players")
        
        # Load career stats players (ALL players)
        self.career_stats_players = pd.read_csv("raw_data/PlayerStats/all_players.csv")
        print(f"Career stats players: {len(self.career_stats_players)} players")
        
        # Load all career statistics (ALL data)
        self.batting_stats = pd.read_csv("raw_data/PlayerStats/t20_batting.csv")
        self.bowling_stats = pd.read_csv("raw_data/PlayerStats/t20_bowling.csv")
        self.all_round_stats = pd.read_csv("raw_data/PlayerStats/t20_all_round.csv")
        
        print(f"Batting stats: {len(self.batting_stats)} records")
        print(f"Bowling stats: {len(self.bowling_stats)} records")
        print(f"All-round stats: {len(self.all_round_stats)} records")
        
        # Load country lookup if available
        try:
            self.country_lookup = pd.read_csv("raw_data/PlayerStats/country.csv")
            print(f"Country lookup: {len(self.country_lookup)} countries")
        except:
            self.country_lookup = None
            print("Country lookup not found - will use country_id directly")
        
    def clean_player_name(self, name):
        """Clean player name for better matching"""
        if pd.isna(name):
            return ""
        
        name = str(name).strip()
        # Remove common suffixes and prefixes
        name = name.replace("(c)", "").replace("(wk)", "").replace("*", "")
        name = name.replace("(captain)", "").replace("(wicketkeeper)", "")
        # Remove extra spaces and normalize
        name = " ".join(name.split())
        return name
    
    def create_career_stats_lookup(self):
        """Create comprehensive career stats lookup"""
        print("Creating career stats lookup...")
        
        career_stats_dict = {}
        
        for _, player in self.career_stats_players.iterrows():
            career_id = player['id']
            clean_name = self.clean_player_name(player['name'])
            
            if not clean_name:
                continue
                
            # Get batting stats
            batting = self.batting_stats[self.batting_stats['id'] == career_id]
            bowling = self.bowling_stats[self.bowling_stats['id'] == career_id]
            all_round = self.all_round_stats[self.all_round_stats['id'] == career_id]
            
            # Extract batting stats
            batting_stats = {}
            if not batting.empty:
                batting_row = batting.iloc[0]
                batting_stats = {
                    'batting_matches': batting_row['matches'] if not pd.isna(batting_row['matches']) else 0,
                    'batting_runs': batting_row['runs'] if not pd.isna(batting_row['runs']) else 0,
                    'batting_avg': batting_row['average_score'] if not pd.isna(batting_row['average_score']) else None,
                    'batting_sr': batting_row['strike_rate'] if not pd.isna(batting_row['strike_rate']) else None,
                    'batting_50s': batting_row['50'] if not pd.isna(batting_row['50']) else 0,
                    'batting_100s': batting_row['100s'] if not pd.isna(batting_row['100s']) else 0,
                    'batting_4s': batting_row['4s'] if not pd.isna(batting_row['4s']) else 0,
                    'batting_6s': batting_row['6s'] if not pd.isna(batting_row['6s']) else 0,
                    'batting_span': batting_row['span'] if not pd.isna(batting_row['span']) else None,
                }
            
            # Extract bowling stats
            bowling_stats = {}
            if not bowling.empty:
                bowling_row = bowling.iloc[0]
                bowling_stats = {
                    'bowling_matches': bowling_row['mt'] if not pd.isna(bowling_row['mt']) else 0,
                    'bowling_wickets': bowling_row['wk'] if not pd.isna(bowling_row['wk']) else 0,
                    'bowling_avg': bowling_row['bwa'] if not pd.isna(bowling_row['bwa']) else None,
                    'bowling_econ': bowling_row['bwe'] if not pd.isna(bowling_row['bwe']) else None,
                    'bowling_sr': bowling_row['bwsr'] if not pd.isna(bowling_row['bwsr']) else None,
                    'bowling_span': bowling_row['sp'] if not pd.isna(bowling_row['sp']) else None,
                }
            
            # Extract all-round stats
            all_round_stats = {}
            if not all_round.empty:
                all_round_row = all_round.iloc[0]
                all_round_stats = {
                    'all_round_batting_avg': all_round_row['bta'] if not pd.isna(all_round_row['bta']) else None,
                    'all_round_bowling_avg': all_round_row['bbad'] if not pd.isna(all_round_row['bbad']) else None,
                    'all_round_span': all_round_row['sp'] if not pd.isna(all_round_row['sp']) else None,
                }
            
            # Get country name if available
            country_name = None
            if self.country_lookup is not None:
                country_info = self.country_lookup[self.country_lookup['id'] == player['country_id']]
                if not country_info.empty:
                    country_name = country_info.iloc[0]['country']
            
            career_stats_dict[clean_name] = {
                'career_id': career_id,
                'original_name': player['name'],
                'gender': player['gender'],
                'batting_style': player['bating_style'],
                'bowling_style': player['bowling_style'],
                'playing_role': player['playing_role'],
                'country_id': player['country_id'],
                'country_name': country_name,
                **batting_stats,
                **bowling_stats,
                **all_round_stats
            }
        
        print(f"Career stats lookup created for {len(career_stats_dict)} players")
        return career_stats_dict
    
    def match_players_by_name(self, career_stats_dict):
        """Match ALL players between datasets using fuzzy string matching"""
        print(f"\nMatching {len(self.ball_by_ball_lookup)} players...")
        
        # Clean names in ball-by-ball lookup
        self.ball_by_ball_lookup['clean_name'] = self.ball_by_ball_lookup['player_name'].apply(self.clean_player_name)
        
        matched_players = []
        unmatched_players = []
        
        # Process ALL players
        for idx, player in self.ball_by_ball_lookup.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(self.ball_by_ball_lookup)} players...")
            
            ball_by_ball_id = player['player_id']
            ball_by_ball_name = player['clean_name']
            
            if not ball_by_ball_name:
                unmatched_players.append({
                    'ball_by_ball_id': ball_by_ball_id,
                    'ball_by_ball_name': player['player_name'],
                    'match_score': 0,
                    'reason': 'Empty name'
                })
                continue
            
            # Try exact match first
            if ball_by_ball_name in career_stats_dict:
                career_data = career_stats_dict[ball_by_ball_name]
                matched_players.append({
                    'ball_by_ball_id': ball_by_ball_id,
                    'ball_by_ball_name': player['player_name'],
                    'career_id': career_data['career_id'],
                    'career_name': career_data['original_name'],
                    'gender': career_data['gender'],
                    'batting_style': career_data['batting_style'],
                    'bowling_style': career_data['bowling_style'],
                    'playing_role': career_data['playing_role'],
                    'country_id': career_data['country_id'],
                    'country_name': career_data['country_name'],
                    'match_score': 100,
                    'match_type': 'exact',
                    **{k: v for k, v in career_data.items() if k not in ['career_id', 'original_name', 'gender', 'batting_style', 'bowling_style', 'playing_role', 'country_id', 'country_name']}
                })
                continue
            
            # Try fuzzy matching with lower threshold for better coverage
            best_match = process.extractOne(
                ball_by_ball_name, 
                list(career_stats_dict.keys()),
                scorer=fuzz.ratio
            )
            
            if best_match and best_match[1] >= 75:  # Lowered threshold for better matching
                career_name = best_match[0]
                career_data = career_stats_dict[career_name]
                matched_players.append({
                    'ball_by_ball_id': ball_by_ball_id,
                    'ball_by_ball_name': player['player_name'],
                    'career_id': career_data['career_id'],
                    'career_name': career_data['original_name'],
                    'gender': career_data['gender'],
                    'batting_style': career_data['batting_style'],
                    'bowling_style': career_data['bowling_style'],
                    'playing_role': career_data['playing_role'],
                    'country_id': career_data['country_id'],
                    'country_name': career_data['country_name'],
                    'match_score': best_match[1],
                    'match_type': 'fuzzy',
                    **{k: v for k, v in career_data.items() if k not in ['career_id', 'original_name', 'gender', 'batting_style', 'bowling_style', 'playing_role', 'country_id', 'country_name']}
                })
            else:
                unmatched_players.append({
                    'ball_by_ball_id': ball_by_ball_id,
                    'ball_by_ball_name': player['player_name'],
                    'match_score': best_match[1] if best_match else 0,
                    'reason': 'No good match found'
                })
        
        print(f"Match processing completed!")
        return matched_players, unmatched_players
    
    def create_combined_lookup(self):
        """Create the final combined player lookup for ALL players"""
        print("Creating comprehensive combined player lookup...")
        
        # Create career stats lookup
        career_stats_dict = self.create_career_stats_lookup()
        
        # Match ALL players
        matched_players, unmatched_players = self.match_players_by_name(career_stats_dict)
        
        print(f"Matched players: {len(matched_players)}")
        print(f"Unmatched players: {len(unmatched_players)}")
        
        # Create combined lookup with ALL players
        combined_lookup = []
        
        # Add matched players
        for player in matched_players:
            combined_player = {
                'ball_by_ball_id': player['ball_by_ball_id'],
                'ball_by_ball_name': player['ball_by_ball_name'],
                'career_id': player['career_id'],
                'career_name': player['career_name'],
                'gender': player['gender'],
                'batting_style': player['batting_style'],
                'bowling_style': player['bowling_style'],
                'playing_role': player['playing_role'],
                'country_id': player['country_id'],
                'country_name': player['country_name'],
                'match_score': player['match_score'],
                'match_type': player['match_type'],
                'has_career_stats': True,
                'batting_matches': player.get('batting_matches', 0),
                'batting_runs': player.get('batting_runs', 0),
                'batting_avg': player.get('batting_avg'),
                'batting_sr': player.get('batting_sr'),
                'batting_50s': player.get('batting_50s', 0),
                'batting_100s': player.get('batting_100s', 0),
                'batting_4s': player.get('batting_4s', 0),
                'batting_6s': player.get('batting_6s', 0),
                'batting_span': player.get('batting_span'),
                'bowling_matches': player.get('bowling_matches', 0),
                'bowling_wickets': player.get('bowling_wickets', 0),
                'bowling_avg': player.get('bowling_avg'),
                'bowling_econ': player.get('bowling_econ'),
                'bowling_sr': player.get('bowling_sr'),
                'bowling_span': player.get('bowling_span'),
                'all_round_batting_avg': player.get('all_round_batting_avg'),
                'all_round_bowling_avg': player.get('all_round_bowling_avg'),
                'all_round_span': player.get('all_round_span'),
            }
            combined_lookup.append(combined_player)
        
        # Add unmatched players (no career stats available)
        for player in unmatched_players:
            combined_player = {
                'ball_by_ball_id': player['ball_by_ball_id'],
                'ball_by_ball_name': player['ball_by_ball_name'],
                'career_id': None,
                'career_name': None,
                'gender': None,
                'batting_style': None,
                'bowling_style': None,
                'playing_role': None,
                'country_id': None,
                'country_name': None,
                'match_score': player['match_score'],
                'match_type': 'unmatched',
                'has_career_stats': False,
                'batting_matches': 0,
                'batting_runs': 0,
                'batting_avg': None,
                'batting_sr': None,
                'batting_50s': 0,
                'batting_100s': 0,
                'batting_4s': 0,
                'batting_6s': 0,
                'batting_span': None,
                'bowling_matches': 0,
                'bowling_wickets': 0,
                'bowling_avg': None,
                'bowling_econ': None,
                'bowling_sr': None,
                'bowling_span': None,
                'all_round_batting_avg': None,
                'all_round_bowling_avg': None,
                'all_round_span': None,
            }
            combined_lookup.append(combined_player)
        
        return pd.DataFrame(combined_lookup), matched_players, unmatched_players
    
    def save_results(self, combined_lookup, matched_players, unmatched_players):
        """Save ALL results"""
        print("Saving results...")
        
        # Save combined lookup (ALL players)
        combined_lookup.to_csv('data/combined_player_lookup.csv', index=False)
        print(f"Saved combined lookup: data/combined_player_lookup.csv ({len(combined_lookup)} players)")
        
        # Save detailed match summary
        match_summary = pd.DataFrame(matched_players)
        match_summary.to_csv('data/player_match_summary.csv', index=False)
        print(f"Saved match summary: data/player_match_summary.csv ({len(matched_players)} players)")
        
        # Save unmatched players
        unmatched_summary = pd.DataFrame(unmatched_players)
        unmatched_summary.to_csv('data/unmatched_players.csv', index=False)
        print(f"Saved unmatched players: data/unmatched_players.csv ({len(unmatched_players)} players)")
        
        # Print comprehensive statistics
        total_players = len(combined_lookup)
        matched_count = len(matched_players)
        unmatched_count = len(unmatched_players)
        exact_matches = len([p for p in matched_players if p['match_type'] == 'exact'])
        fuzzy_matches = len([p for p in matched_players if p['match_type'] == 'fuzzy'])
        
        print(f"\n=== COMPREHENSIVE PLAYER LOOKUP STATISTICS ===")
        print(f"Total players processed: {total_players:,}")
        print(f"Matched with career stats: {matched_count:,} ({matched_count/total_players*100:.1f}%)")
        print(f"  - Exact matches: {exact_matches:,} ({exact_matches/total_players*100:.1f}%)")
        print(f"  - Fuzzy matches: {fuzzy_matches:,} ({fuzzy_matches/total_players*100:.1f}%)")
        print(f"Unmatched (no career stats): {unmatched_count:,} ({unmatched_count/total_players*100:.1f}%)")
        
        # Show top players by career stats
        matched_df = pd.DataFrame(matched_players)
        if not matched_df.empty:
            print(f"\n=== TOP PLAYERS WITH CAREER STATS ===")
            # Show players with most batting runs
            top_batters = matched_df.nlargest(10, 'batting_runs')[['ball_by_ball_name', 'career_name', 'batting_runs', 'batting_avg', 'batting_sr']]
            print("\nTop 10 Batsmen by Runs:")
            print(top_batters.to_string(index=False))
            
            # Show players with most bowling wickets
            top_bowlers = matched_df.nlargest(10, 'bowling_wickets')[['ball_by_ball_name', 'career_name', 'bowling_wickets', 'bowling_avg', 'bowling_econ']]
            print("\nTop 10 Bowlers by Wickets:")
            print(top_bowlers.to_string(index=False))
        
        # Show sample matches
        print(f"\n=== SAMPLE MATCHED PLAYERS ===")
        sample_matched = pd.DataFrame(matched_players).head(15)
        for _, player in sample_matched.iterrows():
            print(f"{player['ball_by_ball_name']} (ID: {player['ball_by_ball_id']}) -> {player['career_name']} (ID: {player['career_id']}) - {player['match_score']}% match")

def main():
    print("Creating Combined Player Lookup - FULL DATABASE")
    print("=" * 60)
    
    builder = CombinedPlayerLookupBuilder()
    combined_lookup, matched_players, unmatched_players = builder.create_combined_lookup()
    builder.save_results(combined_lookup, matched_players, unmatched_players)
    
    print("\n" + "=" * 60)
    print("Combined player lookup created successfully!")
    print("Full database processed with no limitations.")
    print("Ready to rebuild player impact dataset with real career stats!")

if __name__ == "__main__":
    main()
