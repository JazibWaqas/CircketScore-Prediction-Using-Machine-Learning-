import sqlite3

conn = sqlite3.connect('cricket_prediction.db')
cursor = conn.cursor()

print("=== CRICKET VENUES ===")
venues = cursor.execute('SELECT venue_name, city, country FROM venues WHERE venue_name LIKE "%Melbourne%" OR venue_name LIKE "%Sydney%" OR venue_name LIKE "%Dubai%" OR venue_name LIKE "%Abu Dhabi%" OR venue_name LIKE "%Sharjah%" OR venue_name LIKE "%Lahore%" OR venue_name LIKE "%Karachi%" OR venue_name LIKE "%Mumbai%" OR venue_name LIKE "%Delhi%" ORDER BY venue_name').fetchall()
for v in venues:
    print(f"{v[0]} ({v[1]}, {v[2]})")

print("\n=== TEAMS ===")
teams = cursor.execute('SELECT team_name, country FROM teams WHERE team_name LIKE "%Pakistan%" OR team_name LIKE "%India%" ORDER BY team_name').fetchall()
for t in teams:
    print(f"{t[0]} ({t[1]})")

print("\n=== ALL VENUES (First 10) ===")
all_venues = cursor.execute('SELECT venue_name, city, country FROM venues ORDER BY venue_name LIMIT 10').fetchall()
for v in all_venues:
    print(f"{v[0]} ({v[1]}, {v[2]})")

conn.close()
