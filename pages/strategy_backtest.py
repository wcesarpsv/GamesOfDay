import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
DATA_FOLDER = Path("GamesDay")
FILE_PATTERN = "Jogosdodia_*.csv"
DATE_COL = "Date"
# Vamos usar sempre estes nomes internos:
RESULT_COLS = ["Goals_H_FT", "Goals_A_FT"]
FILTER_COLS = ["Diff_Power", "Diff_HT_P", "Odd_H", "Odd_D", "Odd_A"]

# ──────────────────────────────────────────────────────────────────────────────
# Carregamento (cache)
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
# Limpeza e preparação
# ──────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # remover colunas Unnamed, all-NA e strip
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")

    # se o CSV tiver 'Goals_FT_H' e 'Goals_FT_A', renomeie para nosso padrão
    rename_map = {}
    if "Goals_FT_H" in df.columns: rename_map["Goals_FT_H"] = "Goals_H_FT"
    if "Goals_FT_A" in df.columns: rename_map["Goals_FT_A"] = "Goals_A_FT"
    if rename_map:
        df = df.rename(columns=rename_map)

    # parse de datas
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # conversão numérica de filtros e resultados
    for col in FILTER_COLS + RESULT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # apenas linhas com resultados finais
    df = df.dropna(subset=RESULT_COLS)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Simulação de lucro (vetorizada)
# ──────────────────────────────────────────────────────────────────────────────
def simulate_profit(df: pd.DataFrame, bet_on: str) -> pd.Series:
    if bet_on == "Home":
        return np.where(df.Goals_H_FT > df.Goals_A_FT, df.Odd_H - 1, -1)
    else:  # Away
        return np.where(df.Goals_A_FT > df.Goals_H_FT, df.Odd_A - 1, -1)

# ──────────────────────────────────────────────────────────────────────────────
# App principal
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("📊 Strategy Backtest")

# 1) Load & clean
raw_df = load_data(DATA_FOLDER, FILE_PATTERN)
if raw_df.empty:
    st.error(f"❌ Nenhum CSV encontrado em '{DATA_FOLDER}'.")
    st.stop()

df = clean_data(raw_df)
if df.empty:
    st.error("❌ Nenhum dado válido após limpeza de datas e resultados.")
    # para ajudar a debugar, mostramos as colunas encontradas
    st.write("Colunas originais:", raw_df.columns.tolist())
    st.write("Depois da limpeza, colunas:", df.columns.tolist())
    st.write("Amostra do raw_df:", raw_df.head())
    st.stop()

# 2) Filtros na sidebar
st.sidebar.header("🔎 Filtros de Partidas")

# filtro por período
min_date = df[DATE_COL].dt.date.min()
max_date = df[DATE_COL].dt.date.max()
date_range = st.sidebar.date_input("Período", [min_date, max_date])
df = df[df[DATE_COL].between(pd.to_datetime(date_range[0]),
                              pd.to_datetime(date_range[1]))]

# sliders numéricos
for col in FILTER_COLS:
    if col in df.columns:
        mn, mx = float(df[col].min()), float(df[col].max())
        sel = st.sidebar.slider(col, mn, mx, (mn, mx))
        df = df[df[col].between(sel[0], sel[1])]

# direção da aposta
bet_option = st.sidebar.selectbox("🎲 Apostar em", ["Home", "Away"])

# 3) Simular e métricas
df = df.sort_values(DATE_COL)
df["Profit"] = simulate_profit(df, bet_option)
df["Cumulative Profit"] = df["Profit"].cumsum()

total_bets = len(df)
wins = (df["Profit"] > 0).sum()
win_rate = wins / total_bets * 100 if total_bets else 0
roi = df["Profit"].sum() / total_bets * 100 if total_bets else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de Apostas", total_bets)
c2.metric("Vitórias", wins)
c3.metric("Win Rate", f"{win_rate:.1f}%")
c4.metric("ROI", f"{roi:.1f}%")

# 4) Gráfico de lucro acumulado
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

# 5) Tabela e download
if not df.empty:
    display_cols = [DATE_COL, "League", "Home", "Away"] + RESULT_COLS + ["Odd_H", "Odd_D", "Odd_A", "Profit", "Cumulative Profit"]
    st.dataframe(df[display_cols])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", csv, "backtest.csv", "text/csv")
