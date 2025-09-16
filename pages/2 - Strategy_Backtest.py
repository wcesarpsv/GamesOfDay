# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import date
import re

st.set_page_config(page_title="Strategy Backtest – 1X2", layout="wide")
st.title("📈 Strategy Backtest – 1X2")

# ──────────────────────────────────────────────────────────────────────────────
# 🔒 Internal league filter (NOT shown in UI)
EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "UEFA", "Friendly", "Super Cup"]
_EXC_PATTERN = re.compile("|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS)),
                          flags=re.IGNORECASE) if EXCLUDED_LEAGUE_KEYWORDS else None

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

        required = {"Goals_H_FT","Goals_A_FT","Diff_Power","OU_Total","Odd_H","Odd_D","Odd_A","Date","M_H","M_A"}
        if not required.issubset(df.columns):
            continue

        # 🔧 Garante colunas extras
        for col in ["Diff_HT_P", "M_HT_H", "M_HT_A"]:
            if col not in df.columns:
                df[col] = float("nan")

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
# 🎯 Filters (Dynamic Cascading)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filter Matches")

# 🔘 Reset
if st.sidebar.button("🔄 Reset filters"):
    for key in list(st.session_state.keys()):
        if any(prefix in key for prefix in ["mh", "ma", "diff_power", "odd_h", "odd_d", "odd_a",
                                            "date", "bet_on", "diff_htp", "mht_h", "mht_a"]):
            del st.session_state[key]
    st.rerun()

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Draw", "Away"], key="bet_on")
st.sidebar.divider()

df_filtered = df_all.copy()

# Date
date_start, date_end = date_range_filter_hybrid("🗓️ Period (Date)", df_filtered["Date"], key_prefix="date")
if date_start and date_end:
    df_filtered = df_filtered[(df_filtered["Date"] >= date_start) & (df_filtered["Date"] <= date_end)]

# M_H
if not df_filtered.empty:
    mh_min, mh_max = float(df_filtered["M_H"].min()), float(df_filtered["M_H"].max())
    mh_sel = range_filter_hybrid("📊 M_H", mh_min, mh_max, step=0.01, key_prefix="mh")
    df_filtered = df_filtered[(df_filtered["M_H"] >= mh_sel[0]) & (df_filtered["M_H"] <= mh_sel[1])]

# M_A
if not df_filtered.empty:
    ma_min, ma_max = float(df_filtered["M_A"].min()), float(df_filtered["M_A"].max())
    ma_sel = range_filter_hybrid("📊 M_A", ma_min, ma_max, step=0.01, key_prefix="ma")
    df_filtered = df_filtered[(df_filtered["M_A"] >= ma_sel[0]) & (df_filtered["M_A"] <= ma_sel[1])]

# Diff_Power
if not df_filtered.empty:
    dp_min, dp_max = float(df_filtered["Diff_Power"].min()), float(df_filtered["Diff_Power"].max())
    diff_power_sel = range_filter_hybrid("📊 Diff_Power", dp_min, dp_max, step=0.01, key_prefix="diff_power")
    df_filtered = df_filtered[(df_filtered["Diff_Power"] >= diff_power_sel[0]) & (df_filtered["Diff_Power"] <= diff_power_sel[1])]

# Odds
if not df_filtered.empty:
    oh_min, oh_max = float(df_filtered["Odd_H"].min()), float(df_filtered["Odd_H"].max())
    odd_h_sel = range_filter_hybrid("💰 Odd_H (Home win)", oh_min, oh_max, step=0.01, key_prefix="odd_h")
    df_filtered = df_filtered[(df_filtered["Odd_H"] >= odd_h_sel[0]) & (df_filtered["Odd_H"] <= odd_h_sel[1])]

if not df_filtered.empty:
    od_min, od_max = float(df_filtered["Odd_D"].min()), float(df_filtered["Odd_D"].max())
    odd_d_sel = range_filter_hybrid("💰 Odd_D (Draw)", od_min, od_max, step=0.01, key_prefix="odd_d")
    df_filtered = df_filtered[(df_filtered["Odd_D"] >= odd_d_sel[0]) & (df_filtered["Odd_D"] <= odd_d_sel[1])]

if not df_filtered.empty:
    oa_min, oa_max = float(df_filtered["Odd_A"].min()), float(df_filtered["Odd_A"].max())
    odd_a_sel = range_filter_hybrid("💰 Odd_A (Away win)", oa_min, oa_max, step=0.01, key_prefix="odd_a")
    df_filtered = df_filtered[(df_filtered["Odd_A"] >= odd_a_sel[0]) & (df_filtered["Odd_A"] <= odd_a_sel[1])]

# Extras
extra_filters = st.sidebar.multiselect(
    "➕ Extra filters (optional)",
    options=["Diff_HT_P", "M_HT_H", "M_HT_A"]
)

if "Diff_HT_P" in extra_filters and not df_filtered.empty:
    htp_min, htp_max = float(df_filtered["Diff_HT_P"].min()), float(df_filtered["Diff_HT_P"].max())
    diff_ht_p_sel = range_filter_hybrid("📉 Diff_HT_P", htp_min, htp_max, step=0.01, key_prefix="diff_htp")
    df_filtered = df_filtered[(df_filtered["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_filtered["Diff_HT_P"] <= diff_ht_p_sel[1])]

if "M_HT_H" in extra_filters and not df_filtered.empty:
    mht_h_min, mht_h_max = float(df_filtered["M_HT_H"].min()), float(df_filtered["M_HT_H"].max())
    mht_h_sel = range_filter_hybrid("📊 M_HT_H", mht_h_min, mht_h_max, step=0.01, key_prefix="mht_h")
    df_filtered = df_filtered[(df_filtered["M_HT_H"] >= mht_h_sel[0]) & (df_filtered["M_HT_H"] <= mht_h_sel[1])]

if "M_HT_A" in extra_filters and not df_filtered.empty:
    mht_a_min, mht_a_max = float(df_filtered["M_HT_A"].min()), float(df_filtered["M_HT_A"].max())
    mht_a_sel = range_filter_hybrid("📊 M_HT_A", mht_a_min, mht_a_max, step=0.01, key_prefix="mht_a")
    df_filtered = df_filtered[(df_filtered["M_HT_A"] >= mht_a_sel[0]) & (df_filtered["M_HT_A"] <= mht_a_sel[1])]

# ──────────────────────────────────────────────────────────────────────────────
# Profit Calculation + Results
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row):
    h, a = row["Goals_H_FT"], row["Goals_A_FT"]
    if bet_on == "Home":
        return (row["Odd_H"] - 1) if h > a else -1
    elif bet_on == "Draw":
        return (row["Odd_D"] - 1) if h == a else -1
    else:
        return (row["Odd_A"] - 1) if a > h else -1

if not df_filtered.empty:
    df_filtered["Bet Result"] = df_filtered.apply(calculate_profit, axis=1)
    df_filtered["Cumulative Profit"] = df_filtered["Bet Result"].cumsum()

    # 📈 Profit acumulado geral
    fig = px.line(
        df_filtered.reset_index(),
        x=df_filtered.reset_index().index,
        y="Cumulative Profit",
        title=f"Cumulative Profit by Bet (1X2 – {bet_on}, Stake=1)",
        labels={"index": "Bet Number", "Cumulative Profit": "Profit (units)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Metrics globais
    n_matches = len(df_filtered)
    wins = (df_filtered["Bet Result"] > 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    odd_map = {"Home": "Odd_H", "Draw": "Odd_D", "Away": "Odd_A"}
    mean_odd = df_filtered[odd_map[bet_on]].mean()
    total_profit = df_filtered["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader("📊 Backtest Results (Global)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean Odd", f"{mean_odd:.2f}")
    col4.metric("ROI", f"{roi:.1%}")

    # 📝 Tabela final
    st.subheader("📝 Filtered Matches")
    st.dataframe(df_filtered[[
        "Date", "League", "Home", "Away",
        "Odd_H", "Odd_D", "Odd_A",
        "Diff_Power","OU_Total", "M_H", "M_A",
        "Diff_HT_P", "M_HT_H", "M_HT_A",
        "Goals_H_FT", "Goals_A_FT",
        "Bet Result", "Cumulative Profit"
    ]], use_container_width=True)

    # 📊 Resumo por Liga
    league_summary = (
        df_filtered.groupby("League")
        .agg(
            Matches=("League", "size"),
            Wins=("Bet Result", lambda x: (x > 0).sum()),
            Total_Profit=("Bet Result", "sum"),
            Mean_Odd=(odd_map[bet_on], "mean"),
        )
        .reset_index()
    )
    league_summary["Winrate"] = league_summary["Wins"] / league_summary["Matches"]
    league_summary["ROI"] = league_summary["Total_Profit"] / league_summary["Matches"]

    leagues_available = sorted(league_summary["League"].unique())
    selected_leagues = st.sidebar.multiselect("📌 Select leagues", leagues_available, default=leagues_available)
    league_summary = league_summary[league_summary["League"].isin(selected_leagues)]
    df_filtered = df_filtered[df_filtered["League"].isin(selected_leagues)]

    # 📈 Profit acumulado por liga
    plot_data = []
    for league in selected_leagues:
        df_league = df_filtered[df_filtered["League"] == league].copy()
        if df_league.empty:
            continue
        df_league = df_league.sort_values("Date")
        df_league["Cumulative Profit"] = df_league["Bet Result"].cumsum()
        df_league["Bet Number"] = range(1, len(df_league) + 1)
        df_league["LeagueName"] = league
        plot_data.append(df_league)

    if plot_data:
        df_plot = pd.concat(plot_data)
        fig = px.line(
            df_plot,
            x="Bet Number",
            y="Cumulative Profit",
            color="LeagueName",
            hover_data=["LeagueName", "Bet Number", "Cumulative Profit"],
            title="Cumulative Profit by League",
            labels={"Cumulative Profit": "Profit (units)", "Bet Number": "Number of Bets"}
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                         height=800)
        st.plotly_chart(fig, use_container_width=True)

    # 📊 Performance por Liga
    st.subheader("📊 Performance by League")
    st.dataframe(league_summary, use_container_width=True)

else:
    st.warning("⚠️ No matches found with selected filters.")
