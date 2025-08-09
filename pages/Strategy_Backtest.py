import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest – 1X2 (Com Odds)", layout="wide")
st.title("📈 Strategy Backtest – 1X2 (Com Odds)")

# 🔹 Pasta com os CSVs
GAMES_FOLDER = "GamesDay"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: filtro híbrido (Manual + Slider)
# ──────────────────────────────────────────────────────────────────────────────
def range_filter_hibrido(label: str, data_min: float, data_max: float, step: float, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input("Min", value=float(data_min), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_min")
    max_val = c2.number_input("Max", value=float(data_max), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_max")

    # garante ordem
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        st.session_state[f"{key_prefix}_min"] = float(min_val)
        st.session_state[f"{key_prefix}_max"] = float(max_val)

    slider_val = st.sidebar.slider("Arraste para ajustar",
                                   min_value=float(data_min),
                                   max_value=float(data_max),
                                   value=(float(min_val), float(max_val)),
                                   step=step,
                                   key=f"{key_prefix}_slider")

    fonte = st.sidebar.radio("Fonte do filtro", ["Slider", "Manual"], horizontal=True, key=f"{key_prefix}_src")
    st.sidebar.divider()

    if fonte == "Slider":
        return float(slider_val[0]), float(slider_val[1])
    else:
        return float(min_val), float(max_val)

# ──────────────────────────────────────────────────────────────────────────────
# Carrega CSVs válidos
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
        # precisa ter gols FT para avaliar resultado e as colunas usadas nos filtros
        required = {"Goals_H_FT","Goals_A_FT","Diff_Power","Diff_HT_P","Odd_H","Odd_D","Odd_A","Date"}
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
# 🎚️ Filtros (híbridos)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

# passos finos para métricas e odds
step_metrics = 0.01
step_odds = 0.01

# limites seguros (float)
dp_min, dp_max = float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())
diff_power_sel = range_filter_hibrido("📊 Diff_Power", dp_min, dp_max, step=step_metrics, key_prefix="diff_power")

htp_min, htp_max = float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())
diff_ht_p_sel = range_filter_hibrido("📉 Diff_HT_P", htp_min, htp_max, step=step_metrics, key_prefix="diff_htp")

oh_min, oh_max = float(df_all["Odd_H"].min()), float(df_all["Odd_H"].max())
odd_h_sel = range_filter_hibrido("💰 Odd_H (Home win)", oh_min, oh_max, step=step_odds, key_prefix="odd_h")

od_min, od_max = float(df_all["Odd_D"].min()), float(df_all["Odd_D"].max())
odd_d_sel = range_filter_hibrido("💰 Odd_D (Draw)", od_min, od_max, step=step_odds, key_prefix="odd_d")

oa_min, oa_max = float(df_all["Odd_A"].min()), float(df_all["Odd_A"].max())
odd_a_sel = range_filter_hibrido("💰 Odd_A (Away win)", oa_min, oa_max, step=step_odds, key_prefix="odd_a")

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# ──────────────────────────────────────────────────────────────────────────────
# 🧮 Aplica filtros
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power_sel[0]) & (df_all["Diff_Power"] <= diff_power_sel[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_all["Diff_HT_P"] <= diff_ht_p_sel[1]) &
    (df_all["Odd_H"] >= odd_h_sel[0]) & (df_all["Odd_H"] <= odd_h_sel[1]) &
    (df_all["Odd_D"] >= odd_d_sel[0]) & (df_all["Odd_D"] <= odd_d_sel[1]) &
    (df_all["Odd_A"] >= odd_a_sel[0]) & (df_all["Odd_A"] <= odd_a_sel[1])
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# 🧠 Cálculo do resultado da aposta (stake = 1)
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row):
    if bet_on == "Home":
        return (row["Odd_H"] - 1) if row["Goals_H_FT"] > row["Goals_A_FT"] else -1
    else:
        return (row["Odd_A"] - 1) if row["Goals_A_FT"] > row["Goals_H_FT"] else -1

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # 📈 Gráfico de lucro acumulado
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(filtered_df)), filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit by Bet (1X2, Stake=1)")
    st.pyplot(fig)

    # 🔢 Métricas
    n_matches = len(filtered_df)
    wins = (filtered_df["Bet Result"] > 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    odd_col = "Odd_H" if bet_on == "Home" else "Odd_A"
    mean_odd = filtered_df[odd_col].mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader("📊 Backtest Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean Odd", f"{mean_odd:.2f}")
    col4.metric("ROI", f"{roi:.1%}")

    # 📋 Tabela
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
