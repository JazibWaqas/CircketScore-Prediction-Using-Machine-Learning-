import sqlite3

conn = sqlite3.connect('cricket_prediction.db')
cursor = conn.cursor()

# Search for famous players
famous_names = [
    "Virat", "Rohit", "Babar", "Shaheen", "Rizwan", "Shadab", "Haris", "Fakhar",
    "Suryakumar", "Hardik", "Jadeja", "Bumrah", "Dhoni", "Tendulkar", "Akram",
    "Kohli", "Sharma", "Azam", "Afridi", "Khan", "Patel", "Singh", "Kumar"
]

print("=== SEARCHING FOR FAMOUS PLAYERS ===")
for name in famous_names:
    players = cursor.execute('SELECT player_name, country, player_role FROM players WHERE player_name LIKE ? ORDER BY player_name', (f'%{name}%',)).fetchall()
    if players:
        print(f"\n--- Players with '{name}' ---")
        for p in players[:5]:  # Show first 5 matches
            print(f"{p[0]} - {p[1]} - {p[2]}")

print("\n=== PAKISTAN PLAYERS (First 20) ===")
pak_players = cursor.execute('SELECT player_name, country, player_role FROM players WHERE country = "Pakistan" ORDER BY player_name LIMIT 20').fetchall()
for p in pak_players:
    print(f"{p[0]} - {p[2]}")

print("\n=== INDIA PLAYERS (First 20) ===")
ind_players = cursor.execute('SELECT player_name, country, player_role FROM players WHERE country = "India" ORDER BY player_name LIMIT 20').fetchall()
for p in ind_players:
    print(f"{p[0]} - {p[2]}")

conn.close()
