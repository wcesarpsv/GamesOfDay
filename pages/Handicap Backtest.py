import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
from datetime import date

st.set_page_config(page_title="Strategy Backtest – Asian Handicap", layout="wide")
st.title("📈 Strategy Backtest – Asian Handicap")

# ──────────────────────────────────────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────────────────────────────────────
GAMES_FOLDER = "GamesDay"
ODDS_ARE_NET = True
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa"]
_EXC_PATTERN = re.compile("|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS)), flags=re.IGNORECASE) if EXCLUDED_LEAGUE_KEYWORDS else None

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_net_odds(x):
    try:
        v = float(x)
        return v if ODDS_ARE_NET else (v - 1.0)
    except Exception:
        return None

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

def settle_ah_with_odds(goals_h, goals_a, ah_components_home, bet_on: str, net_odds: float) -> float:
    if not ah_components_home or pd.isna(goals_h) or pd.isna(goals_a) or net_odds is None:
        return 0.0
    profits = []
    score_diff = goals_h - goals_a
    for h_home in ah_components_home:
        if bet_on == "Home":
            margin = score_diff + h_home
        else:
            h_away = -h_home
            margin = (goals_a - goals_h) + h_away
        if margin > 0:
            profits.append(net_odds)
        elif abs(margin) < 1e-9:
            profits.append(0.0)
        else:
            profits.append(-1.0)
    return sum(profits) / len(profits)

# ──────────────────────────────────────────────────────────────────────────────
# Load CSVs
# ──────────────────────────────────────────────────────────────────────────────
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
        required = {"Goals_H_FT","Goals_A_FT","Diff_Power","M_H","M_A","Odd_H_Asi","Odd_A_Asi","Date","Asian_Line_Away_raw","AH_components_home","AH_clean_home"}
        if not required.issubset(df.columns):
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data found in GamesAsian.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
if _EXC_PATTERN and "League" in df_all.columns:
    df_all = df_all[~df_all["League"].astype(str).str.contains(_EXC_PATTERN, na=False)]
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Filters
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

# 🔄 Reset button
if st.sidebar.button("🔄 Reset filters"):
    for key in list(st.session_state.keys()):
        if any(prefix in key for prefix in ["mh", "ma", "diff_power", "ah_for_side", "date", "odd_hasi", "odd_aasi", "diff_htp"]):
            del st.session_state[key]
    st.rerun()

# 🎯 Bet On
bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"], key="bet_on")
st.sidebar.divider()

# 🗓️ Date
date_start, date_end = date_range_filter_hybrid("🗓️ Period (Date)", df_all["Date"], key_prefix="date")

# 📊 M_H
mh_min, mh_max = float(df_all["M_H"].min()), float(df_all["M_H"].max())
mh_sel = range_filter_hybrid("📊 M_H", mh_min, mh_max, step=0.01, key_prefix="mh")

# 📊 M_A
ma_min, ma_max = float(df_all["M_A"].min()), float(df_all["M_A"].max())
ma_sel = range_filter_hybrid("📊 M_A", ma_min, ma_max, step=0.01, key_prefix="ma")

# 📊 Diff_Power
dp_min, dp_max = float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())
diff_power_sel = range_filter_hybrid("📊 Diff_Power", dp_min, dp_max, step=0.01, key_prefix="diff_power")

# ⚖️ Asian Handicap (side)
df_all["AH_clean_for_side"] = df_all["AH_clean_home"] if bet_on == "Home" else -df_all["AH_clean_home"]
ah_min, ah_max = float(df_all["AH_clean_for_side"].min()), float(df_all["AH_clean_for_side"].max())
ah_sel = range_filter_hybrid("⚖️ Asian Handicap (side line)", ah_min, ah_max, step=0.25, key_prefix="ah_for_side")

# ➕ Extras
extra_filters = st.sidebar.multiselect("➕ Extra filters", options=["Odd_H_Asi", "Odd_A_Asi", "Diff_HT_P"])

if "Odd_H_Asi" in extra_filters:
    oh_min, oh_max = float(df_all["Odd_H_Asi"].min()), float(df_all["Odd_H_Asi"].max())
    odd_hasi_sel = range_filter_hybrid("💰 Odd_H_Asi", oh_min, oh_max, step=0.01, key_prefix="odd_hasi")
else:
    odd_hasi_sel = (float("-inf"), float("inf"))

if "Odd_A_Asi" in extra_filters:
    oa_min, oa_max = float(df_all["Odd_A_Asi"].min()), float(df_all["Odd_A_Asi"].max())
    odd_aasi_sel = range_filter_hybrid("💰 Odd_A_Asi", oa_min, oa_max, step=0.01, key_prefix="odd_aasi")
else:
    odd_aasi_sel = (float("-inf"), float("inf"))

if "Diff_HT_P" in extra_filters:
    htp_min, htp_max = float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())
    diff_htp_sel = range_filter_hybrid("📉 Diff_HT_P", htp_min, htp_max, step=0.01, key_prefix="diff_htp")
else:
    diff_htp_sel = (float("-inf"), float("inf"))

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = df_all[
    (df_all["Date"] >= date_start) & (df_all["Date"] <= date_end) &
    (df_all["M_H"] >= mh_sel[0]) & (df_all["M_H"] <= mh_sel[1]) &
    (df_all["M_A"] >= ma_sel[0]) & (df_all["M_A"] <= ma_sel[1]) &
    (df_all["Diff_Power"] >= diff_power_sel[0]) & (df_all["Diff_Power"] <= diff_power_sel[1]) &
    (df_all["AH_clean_for_side"] >= ah_sel[0]) & (df_all["AH_clean_for_side"] <= ah_sel[1]) &
    (df_all["Odd_H_Asi"] >= odd_hasi_sel[0]) & (df_all["Odd_H_Asi"] <= odd_hasi_sel[1]) &
    (df_all["Odd_A_Asi"] >= odd_aasi_sel[0]) & (df_all["Odd_A_Asi"] <= odd_aasi_sel[1]) &
    (df_all["Diff_HT_P"] >= diff_htp_sel[0]) & (df_all["Diff_HT_P"] <= diff_htp_sel[1])
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Profit Calculation
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row):
    net_odds = row["Odd_H_Asi"] if bet_on == "Home" else row["Odd_A_Asi"]
    return settle_ah_with_odds(row["Goals_H_FT"], row["Goals_A_FT"], row["AH_components_home"], bet_on, net_odds)

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # 📈 Profit acumulado
    fig = px.line(
        filtered_df.reset_index(),
        x=filtered_df.reset_index().index,
        y="Cumulative Profit",
        title=f"Cumulative Profit (Asian Handicap – {bet_on}, Stake=1)",
        labels={"index": "Bet Number", "Cumulative Profit": "Profit (units)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Metrics globais
    n_matches = len(filtered_df)
    wins = (filtered_df["Bet Result"] > 0).sum()
    pushes = (filtered_df["Bet Result"] == 0).sum()
    losses = (filtered_df["Bet Result"] < 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0
    mean_ah = filtered_df["AH_clean_for_side"].mean()

    st.subheader("📊 Backtest Results (Global)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean AH (side)", f"{mean_ah:+.2f}")
    col4.metric("ROI", f"{roi:.1%}")

    # 📝 Tabela final
    st.subheader("📝 Filtered Matches")
    st.dataframe(filtered_df[[
        "Date", "League", "Home", "Away",
        "Asian_Line_Away_raw", "Asian_Line_Away",
        "AH_clean_home", "AH_clean_for_side",
        "Diff_Power", "M_H", "M_A", "Diff_HT_P",
        "Odd_H_Asi", "Odd_A_Asi",
        "Goals_H_FT", "Goals_A_FT",
        "Bet Result", "Cumulative Profit"
    ]], use_container_width=True)

    # 📊 Resumo por Liga
    league_summary = (
        filtered_df.groupby("League")
        .agg(
            Matches=("League", "size"),
            Wins=("Bet Result", lambda x: (x > 0).sum()),
            Total_Profit=("Bet Result", "sum"),
            Mean_Odd=("Odd_H_Asi" if bet_on=="Home" else "Odd_A_Asi", "mean"),
        )
        .reset_index()
    )
    league_summary["Winrate"] = league_summary["Wins"] / league_summary["Matches"]
    league_summary["ROI"] = league_summary["Total_Profit"] / league_summary["Matches"]

    st.subheader("📊 Performance by League")
    st.dataframe(league_summary, use_container_width=True)

else:
    st.warning("⚠️ No matches found with selected filters.")
