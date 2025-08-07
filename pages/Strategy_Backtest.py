# pages/Strategy_Backtest.py

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ─── 1) Ler parâmetros via URL ──────────────────────────────────────────────────
raw_params = st.experimental_get_query_params()
preset_filters = {}
for key, val in raw_params.items():
    try:
        preset_filters[key] = float(val[0])
    except:
        pass
# ex: preset_filters = {'Diff_Power': 1.23, 'Odd_H': 2.50}

# ─── 2) Layout da página ──────────────────────────────────────────────────────
st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("📈 Strategy Backtest")

# ─── 3) Carregar todos os CSVs válidos ─────────────────────────────────────────
GAMES_FOLDER = "GamesDay"
all_dfs = []
for fname in sorted(os.listdir(GAMES_FOLDER)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(GAMES_FOLDER, fname))
    # pular arquivos sem colunas de gols
    if 'Goals_H_FT' not in df.columns or 'Goals_A_FT' not in df.columns:
        continue
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data with goal columns found.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values("Date").reset_index(drop=True)

# ─── 4) Sidebar: sliders com presets ───────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

# --- Diff_Power ---
min_dp, max_dp = float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())
if "Diff_Power" in preset_filters:
    v = preset_filters["Diff_Power"]
    diff_power = st.sidebar.slider(
        "📊 Diff_Power",
        min_dp, max_dp,
        (v, v),
        disabled=True
    )
else:
    diff_power = st.sidebar.slider(
        "📊 Diff_Power",
        min_dp, max_dp,
        (min_dp, max_dp)
    )

# --- Diff_HT_P ---
min_ht, max_ht = float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())
if "Diff_HT_P" in preset_filters:
    v = preset_filters["Diff_HT_P"]
    diff_ht_p = st.sidebar.slider(
        "📉 Diff_HT_P",
        min_ht, max_ht,
        (v, v),
        disabled=True
    )
else:
    diff_ht_p = st.sidebar.slider(
        "📉 Diff_HT_P",
        min_ht, max_ht,
        (min_ht, max_ht)
    )

# --- Odd_H ---
min_oh, max_oh = float(df_all["Odd_H"].min()), float(df_all["Odd_H"].max())
if "Odd_H" in preset_filters:
    v = preset_filters["Odd_H"]
    odd_h = st.sidebar.slider(
        "💰 Odd_H (Home win)",
        min_oh, max_oh,
        (v, v),
        disabled=True
    )
else:
    odd_h = st.sidebar.slider(
        "💰 Odd_H (Home win)",
        min_oh, max_oh,
        (min_oh, max_oh)
    )

# --- Odd_D ---
min_od, max_od = float(df_all["Odd_D"].min()), float(df_all["Odd_D"].max())
if "Odd_D" in preset_filters:
    v = preset_filters["Odd_D"]
    odd_d = st.sidebar.slider(
        "💰 Odd_D (Draw)",
        min_od, max_od,
        (v, v),
        disabled=True
    )
else:
    odd_d = st.sidebar.slider(
        "💰 Odd_D (Draw)",
        min_od, max_od,
        (min_od, max_od)
    )

# --- Odd_A ---
min_oa, max_oa = float(df_all["Odd_A"].min()), float(df_all["Odd_A"].max())
if "Odd_A" in preset_filters:
    v = preset_filters["Odd_A"]
    odd_a = st.sidebar.slider(
        "💰 Odd_A (Away win)",
        min_oa, max_oa,
        (v, v),
        disabled=True
    )
else:
    odd_a = st.sidebar.slider(
        "💰 Odd_A (Away win)",
        min_oa, max_oa,
        (min_oa, max_oa)
    )

# --- Bet on ---
bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# ─── 5) Filtrar DataFrame ─────────────────────────────────────────────────────
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power[0]) & (df_all["Diff_Power"] <= diff_power[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p[0]) & (df_all["Diff_HT_P"] <= diff_ht_p[1]) &
    (df_all["Odd_H"] >= odd_h[0]) & (df_all["Odd_H"] <= odd_h[1]) &
    (df_all["Odd_D"] >= odd_d[0]) & (df_all["Odd_D"] <= odd_d[1]) &
    (df_all["Odd_A"] >= odd_a[0]) & (df_all["Odd_A"] <= odd_a[1])
].copy()

# ─── 6) Calcular resultados das apostas ────────────────────────────────────────
def calculate_profit(row):
    if bet_on == "Home":
        return (row["Odd_H"] - 1) if row["Goals_H_FT"] > row["Goals_A_FT"] else -1
    else:
        return (row["Odd_A"] - 1) if row["Goals_A_FT"] > row["Goals_H_FT"] else -1

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # ─── 7) Gráfico de lucro acumulado ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["Cumulative Profit"].values, marker="o")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("Cumulative Profit by Bet")
    st.pyplot(fig)

    # ─── 8) Métricas de Backtest ────────────────────────────────────────────────
    n_matches = len(filtered_df)
    wins = (filtered_df["Bet Result"] > 0).sum()
    winrate = wins / n_matches if n_matches else 0
    odd_col = "Odd_H" if bet_on == "Home" else "Odd_A"
    mean_odd = filtered_df[odd_col].mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0

    st.subheader("📊 Backtest Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of Matches", f"{n_matches}")
    c2.metric("Winrate", f"{winrate:.1%}")
    c3.metric("Mean Odd", f"{mean_odd:.2f}")
    c4.metric("ROI", f"{roi:.1%}")

    # ─── 9) Tabela de partidas filtradas ────────────────────────────────────────
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
