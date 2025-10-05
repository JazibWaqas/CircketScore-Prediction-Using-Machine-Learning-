import sqlite3

conn = sqlite3.connect('cricket_prediction.db')
cursor = conn.cursor()

print("=== PAKISTAN PLAYERS ===")
pak_players = cursor.execute('SELECT player_name, country, player_role FROM players WHERE country = "Pakistan" ORDER BY player_name LIMIT 15').fetchall()
for p in pak_players:
    print(f"{p[0]} - {p[2]}")

print("\n=== INDIA PLAYERS ===")
ind_players = cursor.execute('SELECT player_name, country, player_role FROM players WHERE country = "India" ORDER BY player_name LIMIT 15').fetchall()
for p in ind_players:
    print(f"{p[0]} - {p[2]}")

print("\n=== VENUES ===")
venues = cursor.execute('SELECT venue_name, city, country FROM venues ORDER BY venue_name LIMIT 10').fetchall()
for v in venues:
    print(f"{v[0]} ({v[1]}, {v[2]})")

print("\n=== TEAMS ===")
teams = cursor.execute('SELECT team_name, country FROM teams ORDER BY team_name LIMIT 10').fetchall()
for t in teams:
    print(f"{t[0]} ({t[1]})")

conn.close()
