import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import date
import re

st.set_page_config(page_title="Strategy Backtest – 1X2", layout="wide")
st.title("📈 Strategy Backtest – 1X2")

# ──────────────────────────────────────────────────────────────────────────────
# 🔒 Internal league filter (NOT shown in UI)
EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "UEFA", "Friendly", "Super Cup"]
_EXC_PATTERN = re.compile("|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS)), flags=re.IGNORECASE) if EXCLUDED_LEAGUE_KEYWORDS else None

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def range_filter_hybrid(label: str, data_min: float, data_max: float, step: float, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input("Min", value=float(data_min), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_min")
    max_val = c2.number_input("Max", value=float(data_max), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_max")

    if min_val > max_val:
        min_val, max_val = max_val, min_val
        st.session_state[f"{key_prefix}_min"] = float(min_val)
        st.session_state[f"{key_prefix}_max"] = float(max_val)

    slider_val = st.sidebar.slider("Drag to adjust",
                                   min_value=float(data_min),
                                   max_value=float(data_max),
                                   value=(float(min_val), float(max_val)),
                                   step=step,
                                   key=f"{key_prefix}_slider")

    source = st.sidebar.radio("Filter source", ["Slider", "Manual"], horizontal=True, key=f"{key_prefix}_src")
    st.sidebar.divider()

    if source == "Slider":
        return float(slider_val[0]), float(slider_val[1])
    else:
        return float(min_val), float(max_val)

def date_range_filter_hybrid(label: str, series_dates: pd.Series, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")

    dates = pd.to_datetime(series_dates, errors="coerce").dt.date.dropna().unique()
    dates = sorted(dates)
    if not dates:
        return None, None

    dmin, dmax = dates[0], dates[-1]

    c1, c2 = st.sidebar.columns(2)
    d_from = c1.date_input("From", value=dmin, min_value=dmin, max_value=dmax, key=f"{key_prefix}_from")
    d_to   = c2.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key=f"{key_prefix}_to")

    idx_min, idx_max = 0, len(dates) - 1
    idx_from, idx_to = st.sidebar.slider(
        "Drag to adjust (by date index)",
        min_value=idx_min,
        max_value=idx_max,
        value=(idx_min, idx_max),
        key=f"{key_prefix}_slider"
    )

    source = st.sidebar.radio("Filter source", ["Slider", "Manual"], horizontal=True, key=f"{key_prefix}_src")
    st.sidebar.divider()

    if source == "Slider":
        start_d, end_d = dates[min(idx_from, idx_to)], dates[max(idx_from, idx_to)]
    else:
        start_d, end_d = (d_from, d_to) if d_from <= d_to else (d_to, d_from)

    return start_d, end_d

# ──────────────────────────────────────────────────────────────────────────────
# Load CSVs
# ──────────────────────────────────────────────────────────────────────────────
GAMES_FOLDER = "GamesDay"

if not os.path.isdir(GAMES_FOLDER):
    st.error(f"❌ Folder '{GAMES_FOLDER}' not found.")
    st.stop()

all_dfs = []
for file in sorted(os.listdir(GAMES_FOLDER)):
    if file.endswith(".csv"):
        df_path = os.path.join(GAMES_FOLDER, file)
        try:
            df = pd.read_csv(df_path)
        except Exception:
            continue

        required = {"Goals_H_FT","Goals_A_FT","Diff_Power","Diff_HT_P","Odd_H","Odd_D","Odd_A","Date"}
        if not required.issubset(df.columns):
            continue

        if _EXC_PATTERN and "League" in df.columns:
            df = df[~df["League"].astype(str).str.contains(_EXC_PATTERN, na=False)]

        df = df.dropna(subset=["Goals_H_FT","Goals_A_FT"])
        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data with goal columns found.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)

if _EXC_PATTERN and "League" in df_all.columns:
    df_all = df_all[~df_all["League"].astype(str).str.contains(_EXC_PATTERN, na=False)]

df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 🎯 Filters
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Draw", "Away"])
st.sidebar.divider()

date_start, date_end = date_range_filter_hybrid("🗓️ Period (Date)", df_all["Date"], key_prefix="date")
if date_start is None or date_end is None:
    st.error("❌ Invalid dates.")
    st.stop()

step_metrics = 0.01
step_odds = 0.01

dp_min, dp_max = float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())
diff_power_sel = range_filter_hybrid("📊 Diff_Power", dp_min, dp_max, step=step_metrics, key_prefix="diff_power")

htp_min, htp_max = float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())
diff_ht_p_sel = range_filter_hybrid("📉 Diff_HT_P", htp_min, htp_max, step=step_metrics, key_prefix="diff_htp")

oh_min, oh_max = float(df_all["Odd_H"].min()), float(df_all["Odd_H"].max())
odd_h_sel = range_filter_hybrid("💰 Odd_H (Home win)", oh_min, oh_max, step=step_odds, key_prefix="odd_h")

od_min, od_max = float(df_all["Odd_D"].min()), float(df_all["Odd_D"].max())
odd_d_sel = range_filter_hybrid("💰 Odd_D (Draw)", od_min, od_max, step=step_odds, key_prefix="odd_d")

oa_min, oa_max = float(df_all["Odd_A"].min()), float(df_all["Odd_A"].max())
odd_a_sel = range_filter_hybrid("💰 Odd_A (Away win)", oa_min, oa_max, step=step_odds, key_prefix="odd_a")

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = df_all[
    (df_all["Date"] >= date_start) & (df_all["Date"] <= date_end) &
    (df_all["Diff_Power"] >= diff_power_sel[0]) & (df_all["Diff_Power"] <= diff_power_sel[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_all["Diff_HT_P"] <= diff_ht_p_sel[1]) &
    (df_all["Odd_H"] >= odd_h_sel[0]) & (df_all["Odd_H"] <= odd_h_sel[1]) &
    (df_all["Odd_D"] >= odd_d_sel[0]) & (df_all["Odd_D"] <= odd_d_sel[1]) &
    (df_all["Odd_A"] >= odd_a_sel[0]) & (df_all["Odd_A"] <= odd_a_sel[1])
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Profit Calculation
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row):
    h, a = row["Goals_H_FT"], row["Goals_A_FT"]
    if bet_on == "Home":
        return (row["Odd_H"] - 1) if h > a else -1
    elif bet_on == "Draw":
        return (row["Odd_D"] - 1) if h == a else -1
    else:
        return (row["Odd_A"] - 1) if a > h else -1

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # 📊 Resumo por Liga
    st.subheader("📊 Performance by League")
    league_summary = (
        filtered_df.groupby("League")
        .agg(
            Matches=("League", "size"),
            Wins=("Bet Result", lambda x: (x > 0).sum()),
            Total_Profit=("Bet Result", "sum"),
            Mean_Odd=("Odd_H" if bet_on=="Home" else "Odd_D" if bet_on=="Draw" else "Odd_A", "mean"),
        )
        .reset_index()
    )
    league_summary["Winrate"] = league_summary["Wins"] / league_summary["Matches"]
    league_summary["ROI"] = league_summary["Total_Profit"] / league_summary["Matches"]

    # Filtro interativo de ligas
    leagues_available = sorted(league_summary["League"].unique())
    selected_leagues = st.sidebar.multiselect("📌 Select leagues", leagues_available, default=leagues_available)
    league_summary = league_summary[league_summary["League"].isin(selected_leagues)]
    filtered_df = filtered_df[filtered_df["League"].isin(selected_leagues)]

    st.dataframe(league_summary, use_container_width=True)

        # 📈 Evolução temporal por liga
    st.subheader("📈 League Evolution Over Time")

    fig, ax = plt.subplots(figsize=(10, 5))

    for league in selected_leagues:
        df_league = filtered_df[filtered_df["League"] == league].copy()
        if df_league.empty:
            continue
        df_league = df_league.sort_values("Date")
        df_league["Cumulative Profit"] = df_league["Bet Result"].cumsum()
        ax.plot(df_league["Date"], df_league["Cumulative Profit"], label=league)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit by League Over Time")
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)


    # 📈 Profit acumulado geral
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(filtered_df)), filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title(f"Cumulative Profit by Bet (1X2 – {bet_on}, Stake=1)")
    st.pyplot(fig)

    # 📊 Metrics globais
    n_matches = len(filtered_df)
    wins = (filtered_df["Bet Result"] > 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    odd_map = {"Home": "Odd_H", "Draw": "Odd_D", "Away": "Odd_A"}
    mean_odd = filtered_df[odd_map[bet_on]].mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader("📊 Backtest Results (Global)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean Odd", f"{mean_odd:.2f}")
    col4.metric("ROI", f"{roi:.1%}")

    # 📝 Tabela final
    st.subheader("📝 Filtered Matches")
    st.dataframe(
        filtered_df[[
            "Date", "League", "Home", "Away",
            "Odd_H", "Odd_D", "Odd_A",
            "Diff_Power", "Diff_HT_P",
            "Goals_H_FT", "Goals_A_FT",
            "Bet Result", "Cumulative Profit"
        ]],
        use_container_width=True
    )
else:
    st.warning("⚠️ No matches found with selected filters.")
