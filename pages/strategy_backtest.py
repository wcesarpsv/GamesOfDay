
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("📈 Strategy Backtest")

# 📁 Diretório dos arquivos CSV
PASTA = "GamesDay"

# 📥 Juntar todos os arquivos CSV
arquivos = [f for f in os.listdir(PASTA) if f.endswith(".csv")]
df_list = []
for arquivo in arquivos:
    caminho = os.path.join(PASTA, arquivo)
    try:
        df = pd.read_csv(caminho)
        df["source_file"] = arquivo
        df_list.append(df)
    except:
        continue

if not df_list:
    st.error("❌ No CSV files found or failed to load.")
    st.stop()

df = pd.concat(df_list, ignore_index=True)

# 🧹 Limpeza inicial
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Date"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# 🎯 Garantir colunas necessárias
required_columns = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "Diff_HT_P", "Goals_H_FT", "Goals_A_FT"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
    st.stop()

# 🎛️ Filtros
st.sidebar.header("🎯 Filter Matches")
diff_power_range = st.sidebar.slider("📊 Diff_Power", float(df["Diff_Power"].min()), float(df["Diff_Power"].max()), (float(df["Diff_Power"].min()), float(df["Diff_Power"].max())))
diff_htp_range = st.sidebar.slider("⏱️ Diff_HT_P", float(df["Diff_HT_P"].min()), float(df["Diff_HT_P"].max()), (float(df["Diff_HT_P"].min()), float(df["Diff_HT_P"].max())))
odd_h_range = st.sidebar.slider("💰 Odd_H (Home win)", float(df["Odd_H"].min()), float(df["Odd_H"].max()), (float(df["Odd_H"].min()), float(df["Odd_H"].max())))
odd_d_range = st.sidebar.slider("💰 Odd_D (Draw)", float(df["Odd_D"].min()), float(df["Odd_D"].max()), (float(df["Odd_D"].min()), float(df["Odd_D"].max())))
odd_a_range = st.sidebar.slider("💰 Odd_A (Away win)", float(df["Odd_A"].min()), float(df["Odd_A"].max()), (float(df["Odd_A"].min()), float(df["Odd_A"].max())))

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 📊 Aplicar filtros
filtered_df = df[
    (df["Diff_Power"].between(*diff_power_range)) &
    (df["Diff_HT_P"].between(*diff_htp_range)) &
    (df["Odd_H"].between(*odd_h_range)) &
    (df["Odd_D"].between(*odd_d_range)) &
    (df["Odd_A"].between(*odd_a_range))
].copy()

# 🧮 Simular apostas
if bet_on == "Home":
    filtered_df["Bet Result"] = filtered_df.apply(lambda row: row["Odd_H"] - 1 if row["Goals_H_FT"] > row["Goals_A_FT"] else -1, axis=1)
elif bet_on == "Away":
    filtered_df["Bet Result"] = filtered_df.apply(lambda row: row["Odd_A"] - 1 if row["Goals_A_FT"] > row["Goals_H_FT"] else -1, axis=1)

filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()
filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors="coerce")

# 📈 Gráfico
st.subheader("📉 Cumulative Profit")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(filtered_df) + 1), filtered_df["Cumulative Profit"], marker="o")
ax.set_xlabel("Bet Number")
ax.set_ylabel("Cumulative Profit")
ax.set_title("Backtest Profit Over Time")
ax.grid(True)
st.pyplot(fig)

# 📋 Detalhes
st.subheader("📋 Filtered Matches")
st.dataframe(filtered_df[["Date", "League", "Home", "Away", "Odd_H", "Odd_D", "Odd_A", "Diff_Power", "Diff_HT_P", "Goals_H_FT", "Goals_A_FT", "Bet Result", "Cumulative Profit"]])
