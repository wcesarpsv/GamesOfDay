import pandas as pd
import streamlit as st
import os
import glob
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Strategy Backtest", layout="wide")
st.title("📈 Strategy Backtest")

DATA_FOLDER = "GamesDay"

# 🧩 Ler e combinar todos os arquivos CSV válidos com os campos necessários
@st.cache_data
def carregar_dados_validos():
    arquivos = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    dfs = []

    for arquivo in arquivos:
        try:
            df = pd.read_csv(arquivo)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            df = df.dropna(axis=1, how='all')

            # Checar se as colunas essenciais existem
            colunas_essenciais = {'Date', 'Home', 'Away', 'Goals_H_FT', 'Goals_A_FT',
                                  'Odd_H', 'Odd_D', 'Odd_A', 'Diff_HT_P', 'Diff_Power'}
            if colunas_essenciais.issubset(df.columns):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
                dfs.append(df)
        except Exception:
            continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# 📥 Carregar os dados
df = carregar_dados_validos()

if df.empty:
    st.error("❌ No valid game data with final scores was found.")
    st.stop()

# 🎛️ Filtros interativos
st.sidebar.header("🔧 Filter Matches")

# Valores máximos e mínimos dinâmicos
diff_power_min, diff_power_max = float(df["Diff_Power"].min()), float(df["Diff_Power"].max())
diff_ht_min, diff_ht_max = float(df["Diff_HT_P"].min()), float(df["Diff_HT_P"].max())
odd_h_min, odd_h_max = float(df["Odd_H"].min()), float(df["Odd_H"].max())
odd_d_min, odd_d_max = float(df["Odd_D"].min()), float(df["Odd_D"].max())
odd_a_min, odd_a_max = float(df["Odd_A"].min()), float(df["Odd_A"].max())

# Sliders
range_diff_power = st.sidebar.slider("📊 Diff_Power", diff_power_min, diff_power_max, (diff_power_min, diff_power_max))
range_diff_ht = st.sidebar.slider("⏱️ Diff_HT_P", diff_ht_min, diff_ht_max, (diff_ht_min, diff_ht_max))
range_odd_h = st.sidebar.slider("🏠 Odd_H (Home win)", odd_h_min, odd_h_max, (odd_h_min, odd_h_max))
range_odd_d = st.sidebar.slider("🤝 Odd_D (Draw)", odd_d_min, odd_d_max, (odd_d_min, odd_d_max))
range_odd_a = st.sidebar.slider("🛫 Odd_A (Away win)", odd_a_min, odd_a_max, (odd_a_min, odd_a_max))

# Tipo de aposta
tipo_aposta = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 🧮 Aplicar os filtros
df_filtrado = df[
    (df["Diff_Power"].between(*range_diff_power)) &
    (df["Diff_HT_P"].between(*range_diff_ht)) &
    (df["Odd_H"].between(*range_odd_h)) &
    (df["Odd_D"].between(*range_odd_d)) &
    (df["Odd_A"].between(*range_odd_a))
].copy()

# 🧾 Simulação de apostas
if tipo_aposta == "Home":
    df_filtrado["Bet_Odd"] = df_filtrado["Odd_H"]
    df_filtrado["Win"] = df_filtrado["Goals_H_FT"] > df_filtrado["Goals_A_FT"]
elif tipo_aposta == "Away":
    df_filtrado["Bet_Odd"] = df_filtrado["Odd_A"]
    df_filtrado["Win"] = df_filtrado["Goals_A_FT"] > df_filtrado["Goals_H_FT"]

df_filtrado["Profit"] = df_filtrado.apply(lambda x: x["Bet_Odd"] - 1 if x["Win"] else -1, axis=1)

# ✅ Métricas
total_bets = len(df_filtrado)
wins = df_filtrado["Win"].sum()
winrate = (wins / total_bets) * 100 if total_bets else 0
total_profit = df_filtrado["Profit"].sum()

st.subheader("📊 Backtest Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bets", total_bets)
col2.metric("Wins", int(wins))
col3.metric("Win Rate", f"{winrate:.2f}%")
col4.metric("Total Profit", f"{total_profit:.2f} units")

# 📈 Gráfico do lucro acumulado
if total_bets > 0:
    df_filtrado = df_filtrado.sort_values("Date")
    df_filtrado["Cumulative Profit"] = df_filtrado["Profit"].cumsum()

    st.subheader("📈 Profit Over Time")
    fig, ax = plt.subplots()
    ax.plot(df_filtrado["Date"], df_filtrado["Cumulative Profit"], marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit")
    ax.grid(True)
    st.pyplot(fig)

    # 🧾 Tabela final
    with st.expander("🔍 Show filtered matches"):
        st.dataframe(df_filtrado[[
            "Date", "League", "Home", "Away", "Goals_H_FT", "Goals_A_FT",
            "Diff_Power", "Diff_HT_P", "Odd_H", "Odd_D", "Odd_A", "Bet_Odd", "Win", "Profit"
        ]].reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No matches found with the selected filters.")
