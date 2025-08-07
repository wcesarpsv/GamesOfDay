import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_FOLDER = Path("GamesDay")
FILE_PATTERN = "Jogosdodia_*.csv"
DATE_COL = "Date"
RESULT_COLS = ["Goals_H_FT", "Goals_A_FT"]
FILTER_COLS = ["Diff_Power", "Diff_HT_P", "Odd_H", "Odd_D", "Odd_A"]

# ──────────────────────────────────────────────────────────────────────────────
# Caching data load to speed up Streamlit reruns
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(folder: Path, pattern: str) -> pd.DataFrame:
    files = list(folder.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["SourceFile"] = f.name
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Falha ao ler {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# Clean & prepare the DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop unnamed cols, all-NA cols, strip whitespace
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")
    
    # parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    
    # convert numeric columns
    for col in FILTER_COLS + RESULT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RESULT_COLS)  # only matches with final result
    
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Profit simulation (vectorized)
# ──────────────────────────────────────────────────────────────────────────────
def simulate_profit(df: pd.DataFrame, bet_on: str) -> pd.Series:
    if bet_on == "Home":
        return np.where(
            df.Goals_H_FT > df.Goals_A_FT,
            df.Odd_H - 1,
            -1
        )
    elif bet_on == "Away":
        return np.where(
            df.Goals_A_FT > df.Goals_H_FT,
            df.Odd_A - 1,
            -1
        )
    else:
        # Draw option if ever needed
        return np.where(
            df.Goals_H_FT == df.Goals_A_FT,
            df.Odd_D - 1,
            -1
        )

# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("📊 Strategy Backtest")

# 1) Load & clean
raw_df = load_data(DATA_FOLDER, FILE_PATTERN)
if raw_df.empty:
    st.error(f"❌ Nenhum CSV encontrado em '{DATA_FOLDER}'.")
    st.stop()

df = clean_data(raw_df)

# 2) Sidebar filters
st.sidebar.header("🔎 Filtros de Partidas")

# date range filter
min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()
date_range = st.sidebar.date_input("Período", [min_date, max_date])
df = df[df[DATE_COL].between(pd.to_datetime(date_range[0]),
                              pd.to_datetime(date_range[1]))]

# numeric sliders
col_ranges = {}
for col in FILTER_COLS:
    if col in df.columns:
        mn, mx = float(df[col].min()), float(df[col].max())
        col_ranges[col] = st.sidebar.slider(col, mn, mx, (mn, mx))
        df = df[df[col].between(*col_ranges[col])]

# bet direction
bet_option = st.sidebar.selectbox("🎲 Apostar em", ["Home", "Away"])

# 3) Simulate and metrics
df = df.sort_values(DATE_COL)
df["Profit"] = simulate_profit(df, bet_option)
df["Cumulative Profit"] = df["Profit"].cumsum()

total_bets = len(df)
wins = (df["Profit"] > 0).sum()
win_rate = wins / total_bets * 100 if total_bets else 0
total_invested = total_bets * 1  # 1 unit per bet
roi = (df["Profit"].sum() / total_invested * 100) if total_invested else 0

# display KPIs
col1, col2, col3, col4 = st.metrics_container().columns(4)
col1.metric("Total de Apostas", total_bets)
col2.metric("Vitórias", wins)
col3.metric("Win Rate", f"{win_rate:.1f}%")
col4.metric("ROI", f"{roi:.1f}%")

# 4) Plot cumulative profit
if not df.empty:
    fig, ax = plt.subplots()
    ax.plot(df[DATE_COL], df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("📈 Lucro Acumulado ao Longo do Tempo")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("⚠️ Sem partidas após filtros.")

# 5) DataFrame & download
if not df.empty:
    display_cols = [DATE_COL, "League", "Home", "Away"] + RESULT_COLS + ["Odd_H", "Odd_D", "Odd_A", "Profit", "Cumulative Profit"]
    st.dataframe(df[display_cols])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", csv, "backtest.csv", "text/csv")
