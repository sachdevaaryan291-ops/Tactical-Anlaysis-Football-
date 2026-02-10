# =========================================================
# Football Tactical Analysis Prototype using Simulated Data
# =========================================================

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. CREATE DATA DIRECTORY (IF NOT EXISTS)
# ---------------------------------------------------------
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------
# 2. SIMULATED MATCH DATA GENERATION
# ---------------------------------------------------------
np.random.seed(42)

players = [f"P{i}" for i in range(1, 23)]
minutes = range(1, 91)

data = []

for minute in minutes:
    for player in players:
        data.append({
            "minute": minute,
            "player": player,
            "team": "Team_A" if player in players[:11] else "Team_B",
            "x": np.random.uniform(0, 100),
            "y": np.random.uniform(0, 100),
            "speed": np.random.uniform(0, 8),
            "event": np.random.choice(
                ["pass", "shot", "pressure", "carry", "none"],
                p=[0.35, 0.08, 0.20, 0.25, 0.12]
            )
        })

df = pd.DataFrame(data)
df.to_csv("data/simulated_match_data.csv", index=False)

print("\n✅ Simulated football match data generated.\n")

# ---------------------------------------------------------
# 3. PROJECT: TEAM SHAPE & FORMATION DETECTION
# ---------------------------------------------------------

# Load data
df = pd.read_csv("data/simulated_match_data.csv")

# Filter Team A safely
team = df[df["team"] == "Team_A"].copy()

# Positional clustering
positions = team[["x", "y"]]

kmeans = KMeans(n_clusters=4, random_state=42)
team["zone_cluster"] = kmeans.fit_predict(positions)

# Output results
print("📊 Project: Team Shape Detection")
print("Detected team shape clusters:")
print(team["zone_cluster"].value_counts())

print("\n=== SCRIPT EXECUTED SUCCESSFULLY ===")

# ---------------------------------------------------------
# 4. PROJECT 2: PRESSING INTENSITY ANALYSIS
# ---------------------------------------------------------

# Filter pressing actions
pressing_actions = df[df["event"] == "pressure"]

# Pressing intensity per minute
pressing_intensity = pressing_actions.groupby("minute").size()

print("\n📊 Project 2 – Pressing Intensity Analysis")
print("Average pressing actions per minute:",
      round(pressing_intensity.mean(), 2))

print("Peak pressing minute:",
      pressing_intensity.idxmax())

# ---------------------------------------------------------
# 5. PROJECT 3: SPACE OCCUPATION ANALYSIS
# ---------------------------------------------------------

team_a = df[df["team"] == "Team_A"]

# Divide pitch into vertical zones
team_a["pitch_zone"] = pd.cut(
    team_a["x"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=["Far Left", "Left", "Center", "Right", "Far Right"]
)

zone_usage = team_a["pitch_zone"].value_counts()

print("\n📊 Project 3 – Space Occupation Analysis")
print(zone_usage)

# ---------------------------------------------------------
# 6. PROJECT 4: TACTICAL xG PROXY ANALYSIS
# ---------------------------------------------------------

shots = df[df["event"] == "shot"].copy()

# Assign proxy xG based on shot location
shots["xG"] = np.where(
    shots["x"] > 70,
    np.random.uniform(0.3, 0.6, len(shots)),
    np.random.uniform(0.05, 0.25, len(shots))
)

print("\n📊 Project 4 – Tactical xG Proxy")
print("Average xG per shot:", round(shots["xG"].mean(), 3))

# ---------------------------------------------------------
# 7. PROJECT 5: TRANSITION PLAY ANALYSIS
# ---------------------------------------------------------

transitions = df[
    (df["event"] == "carry") &
    (df["speed"] > 5)
]

print("\n📊 Project 5 – Transition Play Analysis")
print("High-speed transition actions:", len(transitions))


# ---------------------------------------------------------
# 8. PROJECT 6: AI-GENERATED MATCH NARRATIVE
# ---------------------------------------------------------

event_counts = df["event"].value_counts()

match_narrative = f"""
📊 Project 6 – Tactical Match Narrative

The team completed {event_counts.get('pass', 0)} passes, indicating structured possession.
A total of {event_counts.get('pressure', 0)} pressing actions highlight defensive aggression.
The match featured {event_counts.get('shot', 0)} shots, reflecting attacking intent,
with transitions playing a key role in chance creation.
"""

print(match_narrative)

# ---------------------------------------------------------
# 9. PROJECT 7: PLAYER ROLE EFFECTIVENESS
# ---------------------------------------------------------

role_score = (
    df.groupby("player")["event"]
    .count()
    .sort_values(ascending=False)
)

print("📊 Project 7 – Player Role Effectiveness (Top 5)")
print(role_score.head())

# ---------------------------------------------------------
# 10. PROJECT 8: OPPONENT WEAKNESS IDENTIFICATION
# ---------------------------------------------------------

opponent = df[df["team"] == "Team_B"]

# Defensive third vulnerability
weak_zone_actions = opponent[opponent["x"] < 30]

print("\n📊 Project 8 – Opponent Weakness Analysis")
print("Actions in defensive third:", len(weak_zone_actions))

# ---------------------------------------------------------
# 11. PROJECT 9: TEAM COMPACTNESS INDEX
# ---------------------------------------------------------

team_a = df[df["team"] == "Team_A"]

# Compactness measured as average distance from team centroid
centroid_x = team_a["x"].mean()
centroid_y = team_a["y"].mean()

team_a["distance_from_centroid"] = np.sqrt(
    (team_a["x"] - centroid_x) ** 2 +
    (team_a["y"] - centroid_y) ** 2
)

compactness_index = team_a["distance_from_centroid"].mean()

print("\n📊 Project 9 – Team Compactness Index")
print("Compactness Index:", round(compactness_index, 2))


# ---------------------------------------------------------
# 12. PROJECT 10: SET-PIECE TACTICAL IMPACT ANALYSIS
# ---------------------------------------------------------

# Proxy: shots taken from wide areas as set-piece outcomes
set_piece_shots = df[
    (df["event"] == "shot") &
    ((df["y"] < 20) | (df["y"] > 80))
]

print("\n📊 Project 10 – Set-Piece Tactical Impact")
print("Set-piece related shots (proxy):", len(set_piece_shots))



