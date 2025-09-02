# Live_Backtest.py

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import re

st.set_page_config(page_title="Live Backtest – 1X2", layout="wide")
st.title("🔮 Live Backtest – 1X2")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def range_filter(label: str, data_min: float, data_max: float, step: float, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")
    min_val, max_val = st.sidebar.slider(
        "Range",
        min_value=float(data_min),
        max_value=float(data_max),
        value=(float(data_min), float(data_max)),
        step=step,
        key=key_prefix
    )
    return float(min_val), float(max_val)

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

        required = {"Goals_H_FT","Goals_A_FT","Diff_Power","Odd_H","Odd_A","Date","M_H","M_A"}
        if not required.issubset(df.columns):
            continue

        df = df.dropna(subset=["Goals_H_FT","Goals_A_FT"])
        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data with goal columns found.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 🎯 Filters
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filters")

# Escolha de aposta (apenas Home / Away)
bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"], key="bet_on")
st.sidebar.divider()

step_metrics = 0.01
step_odds = 0.01

# 📊 M_H
mh_min, mh_max = float(df_all["M_H"].min()), float(df_all["M_H"].max())
mh_sel = range_filter("📊 M_H", mh_min, mh_max, step=step_metrics, key_prefix="mh")

# 📊 M_A
ma_min, ma_max = float(df_all["M_A"].min()), float(df_all["M_A"].max())
ma_sel = range_filter("📊 M_A", ma_min, ma_max, step=step_metrics, key_prefix="ma")

# 📊 Diff_Power
dp_min, dp_max = float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())
diff_power_sel = range_filter("📊 Diff_Power", dp_min, dp_max, step=step_metrics, key_prefix="diff_power")

# 💰 Odd_H
oh_min, oh_max = float(df_all["Odd_H"].min()), float(df_all["Odd_H"].max())
odd_h_sel = range_filter("💰 Odd_H (Home win)", oh_min, oh_max, step=step_odds, key_prefix="odd_h")

# 💰 Odd_A
oa_min, oa_max = float(df_all["Odd_A"].min()), float(df_all["Odd_A"].max())
odd_a_sel = range_filter("💰 Odd_A (Away win)", oa_min, oa_max, step=step_odds, key_prefix="odd_a")

# Percentual de teste
test_size = st.sidebar.slider("📐 Percentage for Test (%)", 5, 50, 10, step=5) / 100.0
split_mode = st.sidebar.radio("🔀 Data Division", ["Radom", "Chronological"], horizontal=True)

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = df_all[
    (df_all["M_H"] >= mh_sel[0]) & (df_all["M_H"] <= mh_sel[1]) &
    (df_all["M_A"] >= ma_sel[0]) & (df_all["M_A"] <= ma_sel[1]) &
    (df_all["Diff_Power"] >= diff_power_sel[0]) & (df_all["Diff_Power"] <= diff_power_sel[1]) &
    (df_all["Odd_H"] >= odd_h_sel[0]) & (df_all["Odd_H"] <= odd_h_sel[1]) &
    (df_all["Odd_A"] >= odd_a_sel[0]) & (df_all["Odd_A"] <= odd_a_sel[1])

].copy()

if filtered_df.empty:
    st.warning("⚠️ No matches found with selected filters.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Split train/test
# ──────────────────────────────────────────────────────────────────────────────
if split_mode == "Aleatória":
    train_df = filtered_df.sample(frac=1-test_size, random_state=42)
    test_df = filtered_df.drop(train_df.index)
else:  # Cronológica
    cutoff = int(len(filtered_df) * (1-test_size))
    train_df = filtered_df.iloc[:cutoff]
    test_df = filtered_df.iloc[cutoff:]

# ──────────────────────────────────────────────────────────────────────────────
# Profit Calculation
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row, bet_on):
    h, a = row["Goals_H_FT"], row["Goals_A_FT"]
    if bet_on == "Home":
        return (row["Odd_H"] - 1) if h > a else -1
    else:
        return (row["Odd_A"] - 1) if a > h else -1

def evaluate_dataset(df, bet_on, title):
    if df.empty:
        st.warning(f"⚠️ No data for {title}")
        return

    df = df.copy()
    df["Bet Result"] = df.apply(lambda r: calculate_profit(r, bet_on), axis=1)
    df["Cumulative Profit"] = df["Bet Result"].cumsum()

    # 📈 Profit acumulado
    fig = px.line(
        df.reset_index(),
        x=df.reset_index().index,
        y="Cumulative Profit",
        title=f"{title} – Cumulative Profit ({bet_on})",
        labels={"index": "Bet Number", "Cumulative Profit": "Profit (units)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Metrics
    n_matches = len(df)
    wins = (df["Bet Result"] > 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    odd_map = {"Home": "Odd_H", "Away": "Odd_A"}
    mean_odd = df[odd_map[bet_on]].mean()
    total_profit = df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader(f"📊 Results – {title}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean Odd", f"{mean_odd:.2f}")
    col4.metric("ROI", f"{roi:.1%}")

    st.dataframe(df[[
        "Date", "League", "Home", "Away",
        "Odd_H", "Odd_A",
        "Diff_Power", "M_H", "M_A",
        "Goals_H_FT", "Goals_A_FT",
        "Bet Result", "Cumulative Profit"
    ]], use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ──────────────────────────────────────────────────────────────────────────────
evaluate_dataset(train_df, bet_on, "Treino (Train Set)")
evaluate_dataset(test_df, bet_on, "Teste (Test Set)")
