import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ---------------- Page Config ----------------
st.set_page_config(page_title="League Ranking – Performance & Momentum", layout="wide")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"  # Pasta onde estão os CSVs

# ---------------- Helpers ----------------
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        return df
    return pd.DataFrame()

def calculate_team_stats(df, view_mode):
    results = []
    teams = pd.concat([df["Home"], df["Away"]]).unique()

    for team in teams:
        if view_mode == "Home only":
            team_games = df[df["Home"] == team].copy()
        elif view_mode == "Away only":
            team_games = df[df["Away"] == team].copy()
        else:  # General
            team_games = df[(df["Home"] == team) | (df["Away"] == team)].copy()

        # Ignorar times sem jogos válidos
        team_games = team_games.dropna(subset=["Goals_FT_H", "Goals_FT_A"])
        if team_games.empty:
            continue

        # --- Estatísticas consolidadas ---
        total_games = len(team_games)
        wins, roi = 0, 0
        streak = []

        for _, row in team_games.iterrows():
            home_goals, away_goals = row["Goals_FT_H"], row["Goals_FT_A"]

            if row["Home"] == team:  # Jogos em casa
                if home_goals > away_goals:
                    wins += 1
                    roi += row["Odd_H"] - 1
                    streak.append("W")
                else:
                    roi -= 1
                    streak.append("L")
            elif row["Away"] == team:  # Jogos fora
                if away_goals > home_goals:
                    wins += 1
                    roi += row["Odd_A"] - 1
                    streak.append("W")
                else:
                    roi -= 1
                    streak.append("L")

        winrate = wins / total_games if total_games > 0 else 0
        avg_diff_power = team_games["Diff_Power"].mean()
        avg_diff_momentum = (team_games["M_H"] - team_games["M_A"]).mean()

        results.append({
            "Team": team,
            "Games": total_games,
            "Winrate (%)": round(winrate * 100, 1),
            "ROI": round(roi, 2),
            "Avg Diff_Power": round(avg_diff_power, 2),
            "Avg Diff_Momentum": round(avg_diff_momentum, 2),
            "Streak": "".join(streak[-5:])
        })

    return pd.DataFrame(results)

# ---------------- Load Data ----------------
df_all = load_all_games(GAMES_FOLDER)

if df_all.empty:
    st.error("No game data found in GamesDay folder.")
    st.stop()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("⚙️ Filters")

league = st.sidebar.selectbox("Select League", sorted(df_all["League"].unique()))
period = st.sidebar.selectbox("Select Period", ["Last 10 Games", "Last 30 Days", "Full Season"])
order_by = st.sidebar.selectbox("Order by", ["ROI", "Winrate (%)", "Avg Diff_Power", "Avg Diff_Momentum"])

# ---------------- Header with Toggle ----------------
col1, col2 = st.columns([4, 2])
with col1:
    st.title("📊 League Ranking – Performance & Momentum")
with col2:
    view_mode = st.radio(
        "View Mode",
        ["General", "Home only", "Away only"],
        horizontal=True
    )

# ---------------- Apply Period Filter ----------------
df_filtered = df_all[df_all["League"] == league].copy()

if period == "Last 30 Days":
    cutoff = datetime.now() - timedelta(days=30)
    df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], errors="coerce")
    df_filtered = df_filtered[df_filtered["Date"] >= cutoff]
elif period == "Last 10 Games":
    # Will be handled later per team (take last 10 games for each)
    pass

# ---------------- Calculate Stats ----------------
df_stats = calculate_team_stats(df_filtered, view_mode)

# Handle "Last 10 Games" filter
if period == "Last 10 Games":
    stats_last10 = []
    for team in df_stats["Team"].unique():
        team_games = df_filtered[(df_filtered["Home"] == team) | (df_filtered["Away"] == team)].copy()
        team_games = team_games.sort_values("Date").tail(10)
        if not team_games.empty:
            stats_last10.append(calculate_team_stats(team_games, view_mode))
    if stats_last10:
        df_stats = pd.concat(stats_last10, ignore_index=True)

# ---------------- Rank & Sort ----------------
df_stats = df_stats.sort_values(order_by, ascending=False).reset_index(drop=True)
df_stats.insert(0, "Rank", [f"{i+1}" for i in range(len(df_stats))])

# ---------------- Display ----------------
st.subheader(f"📌 {league} – {period} – {view_mode}")
st.dataframe(df_stats, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("""
---
**Notes:**  
- **Diff_Power** = historical team strength  
- **Diff_Momentum** = recent trend (M_H – M_A)  
- **ROI** = return of flat betting 1 unit on the team  
""")
