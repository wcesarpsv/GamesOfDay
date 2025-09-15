import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ---------------- Page Config ----------------
st.set_page_config(page_title="League Ranking – Performance & Momentum", layout="wide")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["copa", "copas", "cup", "uefa"]

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
            print(f"⚠️ Erro lendo {file}: {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        if all(col in df.columns for col in ["Date", "Home", "Away"]):
            df = df.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
        return df
    return pd.DataFrame()

# ---------------- Stats Calculations ----------------
def calculate_stats(df, league, mode):
    df = df[df["League"] == league].copy()
    df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT"])
    results = []

    if mode == "Home only":
        teams = df["Home"].unique()
    elif mode == "Away only":
        teams = df["Away"].unique()
    else:  # General
        teams = pd.concat([df["Home"], df["Away"]]).unique()

    for team in teams:
        if mode == "Home only":
            team_games = df[df["Home"] == team].copy()
        elif mode == "Away only":
            team_games = df[df["Away"] == team].copy()
        else:
            team_games = df[(df["Home"] == team) | (df["Away"] == team)].copy()

        if team_games.empty:
            continue

        total_games = len(team_games)
        wins, draws, losses = 0, 0, 0
        streak = []

        for _, row in team_games.iterrows():
            h, a = row["Goals_H_FT"], row["Goals_A_FT"]

            if row["Home"] == team:  # time jogou em casa
                if h > a: wins, streak = wins+1, streak+["W"]
                elif h == a: draws, streak = draws+1, streak+["D"]
                else: losses, streak = losses+1, streak+["L"]

            elif row["Away"] == team:  # time jogou fora
                if a > h: wins, streak = wins+1, streak+["W"]
                elif a == h: draws, streak = draws+1, streak+["D"]
                else: losses, streak = losses+1, streak+["L"]

        winrate = round((wins / total_games) * 100, 1)
        drawrate = round((draws / total_games) * 100, 1)
        lossrate = round((losses / total_games) * 100, 1)

        results.append({
            "Team": team,
            "Games": total_games,
            "Winrate (%)": winrate,
            "Drawrate (%)": drawrate,
            "Lossrate (%)": lossrate,
            "Avg Diff_Power": round(team_games["Diff_Power"].mean(), 2) if "Diff_Power" in team_games else None,
            "Avg Diff_Momentum": round((team_games["M_H"] - team_games["M_A"]).mean(), 2) if "M_H" in team_games and "M_A" in team_games else None,
            "Streak": "".join(streak[-5:])
        })

    return pd.DataFrame(results)

# ---------------- Load Data ----------------
df_all = load_all_games(GAMES_FOLDER)

if df_all.empty:
    st.error("⚠️ Nenhum dado encontrado na pasta GamesDay.")
    st.stop()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("⚙️ Filters")

all_leagues = sorted(df_all["League"].unique())
leagues = [l for l in all_leagues if not any(bad in str(l).lower() for bad in EXCLUDED_LEAGUE_KEYWORDS)]

league = st.sidebar.selectbox("Select League", leagues)
period = st.sidebar.selectbox("Select Period", ["Last 10 Games", "Last 30 Days", "Full Season"])
order_by = st.sidebar.selectbox("Order by", ["Winrate (%)", "Drawrate (%)", "Lossrate (%)", "Avg Diff_Power", "Avg Diff_Momentum"])

# ---------------- Header with Toggle ----------------
col1, col2 = st.columns([4, 2])
with col1:
    st.title("📊 League Ranking – Performance & Momentum")
with col2:
    view_mode = st.radio("View Mode", ["General", "Home only", "Away only"], horizontal=True)

# ---------------- Apply Period Filter ----------------
df_filtered = df_all[df_all["League"] == league].copy()
df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], errors="coerce")

if period == "Last 30 Days":
    cutoff = datetime.now() - timedelta(days=30)
    df_filtered = df_filtered[df_filtered["Date"] >= cutoff]

elif period == "Last 10 Games":
    teams = []
    if view_mode == "Home only":
        teams = df_filtered["Home"].unique()
    elif view_mode == "Away only":
        teams = df_filtered["Away"].unique()
    else:
        teams = pd.concat([df_filtered["Home"], df_filtered["Away"]]).unique()

    df_last10 = []
    for team in teams:
        if view_mode == "Home only":
            games_team = df_filtered[df_filtered["Home"] == team].copy()
        elif view_mode == "Away only":
            games_team = df_filtered[df_filtered["Away"] == team].copy()
        else:
            games_team = df_filtered[(df_filtered["Home"] == team) | (df_filtered["Away"] == team)].copy()

        games_team = games_team.sort_values("Date").tail(10)
        df_last10.append(games_team)

    if df_last10:
        df_filtered = pd.concat(df_last10, ignore_index=True)

# ---------------- Calculate Stats ----------------
df_stats = calculate_stats(df_filtered, league, view_mode)

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
- **Winrate / Drawrate / Lossrate** = percentages based on valid games played  
- **Last 10 Games** now respects General / Home / Away mode  
""")
