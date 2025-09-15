import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ---------------- Page Config ----------------
st.set_page_config(page_title="League Ranking – Performance & Momentum", layout="wide")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"

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

        # 🔹 remover duplicados por jogo (data + times)
        if all(col in df.columns for col in ["Date", "Home", "Away"]):
            df = df.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")

        return df
    return pd.DataFrame()

def calculate_team_stats(df, view_mode, league):
    results = []

    # 🔹 garantir que o cálculo respeita a liga selecionada
    df = df[df["League"] == league].copy()
    if df.empty:
        return pd.DataFrame()

    teams = pd.concat([df["Home"], df["Away"]]).unique()

    for team in teams:
        if view_mode == "Home only":
            team_games = df[df["Home"] == team].copy()
        elif view_mode == "Away only":
            team_games = df[df["Away"] == team].copy()
        else:  # General
            team_games = df[(df["Home"] == team) | (df["Away"] == team)].copy()

        # Valid games only
        team_games = team_games.dropna(subset=["Goals_H_FT", "Goals_A_FT"])
        if team_games.empty:
            continue

        total_games = len(team_games)
        wins, draws, losses = 0, 0, 0
        streak = []

        for _, row in team_games.iterrows():
            home_goals, away_goals = row["Goals_H_FT"], row["Goals_A_FT"]

            if row["Home"] == team:  # Home games
                if home_goals > away_goals:
                    wins += 1
                    streak.append("W")
                elif home_goals == away_goals:
                    draws += 1
                    streak.append("D")
                else:
                    losses += 1
                    streak.append("L")

            elif row["Away"] == team:  # Away games
                if away_goals > home_goals:
                    wins += 1
                    streak.append("W")
                elif away_goals == home_goals:
                    draws += 1
                    streak.append("D")
                else:
                    losses += 1
                    streak.append("L")

        # Rates
        winrate = (wins / total_games) * 100 if total_games > 0 else 0
        drawrate = (draws / total_games) * 100 if total_games > 0 else 0
        lossrate = (losses / total_games) * 100 if total_games > 0 else 0

        avg_diff_power = team_games["Diff_Power"].mean() if "Diff_Power" in team_games else np.nan
        avg_diff_momentum = (team_games["M_H"] - team_games["M_A"]).mean() if "M_H" in team_games and "M_A" in team_games else np.nan

        results.append({
            "Team": team,
            "Games": total_games,
            "Winrate (%)": round(winrate, 1),
            "Drawrate (%)": round(drawrate, 1),
            "Lossrate (%)": round(lossrate, 1),
            "Avg Diff_Power": round(avg_diff_power, 2) if pd.notnull(avg_diff_power) else None,
            "Avg Diff_Momentum": round(avg_diff_momentum, 2) if pd.notnull(avg_diff_momentum) else None,
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
order_by = st.sidebar.selectbox("Order by", ["Winrate (%)", "Drawrate (%)", "Lossrate (%)", "Avg Diff_Power", "Avg Diff_Momentum"])

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
    teams = pd.concat([df_filtered["Home"], df_filtered["Away"]]).unique()
    df_last10 = []
    for team in teams:
        games_team = df_filtered[(df_filtered["Home"] == team) | (df_filtered["Away"] == team)].copy()
        games_team["Date"] = pd.to_datetime(games_team["Date"], errors="coerce")
        games_team = games_team.sort_values("Date").tail(10)
        df_last10.append(games_team)
    if df_last10:
        df_filtered = pd.concat(df_last10, ignore_index=True)

# ---------------- Calculate Stats ----------------
df_stats = calculate_team_stats(df_filtered, view_mode, league)

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
- **Winrate / Drawrate / Lossrate** = percentage based on valid games played  
""")
