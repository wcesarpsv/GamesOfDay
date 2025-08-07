import pandas as pd
import streamlit as st
import os
import glob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("📊 Strategy Backtest")

# 📁 Load all CSVs from the folder
DATA_FOLDER = "GamesDay"
all_files = glob.glob(os.path.join(DATA_FOLDER, "Jogosdodia_*.csv"))

# 🧠 Combine all CSVs into one DataFrame
df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df["SourceFile"] = os.path.basename(file)
    df_list.append(df)

if not df_list:
    st.error("❌ No CSV files found in the 'GamesDay' folder.")
    st.stop()

df = pd.concat(df_list, ignore_index=True)

# 🧹 Clean up columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how='all')

# 📆 Ensure Date column exists and is parsed correctly
if 'Date' not in df.columns:
    st.error("❌ Missing 'Date' column in the data.")
    st.stop()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

# 🎯 Filter only rows with match results
required_cols = ['Goals_H_FT', 'Goals_A_FT']
if not all(col in df.columns for col in required_cols):
    st.warning("⚠️ Some data is missing final match results. Only rows with results will be used.")
df = df.dropna(subset=required_cols)

# 🎛️ Sidebar filters
st.sidebar.header("🎯 Filter Matches")

col_ranges = {}
for col in ['Diff_Power', 'Diff_HT_P', 'Odd_H', 'Odd_D', 'Odd_A']:
    if col in df.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        col_ranges[col] = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))

# 📌 Bet direction
bet_option = st.sidebar.selectbox("🎲 Bet on", ["Home", "Away"])

# 🎯 Apply filters
filtered_df = df.copy()
for col, (min_val, max_val) in col_ranges.items():
    filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

# 💵 Profit simulation
def simulate_profit(row):
    if bet_option == "Home":
        if row['Goals_H_FT'] > row['Goals_A_FT']:
            return row['Odd_H'] - 1
        else:
            return -1
    elif bet_option == "Away":
        if row['Goals_A_FT'] > row['Goals_H_FT']:
            return row['Odd_A'] - 1
        else:
            return -1

filtered_df["Profit"] = filtered_df.apply(simulate_profit, axis=1)

# ✅ Sort by date and calculate cumulative profit
filtered_df = filtered_df.sort_values("Date")
filtered_df["Cumulative Profit"] = filtered_df["Profit"].cumsum()

# Garantir que a coluna Date seja datetime
filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors='coerce')

# 📈 Plot
if not filtered_df.empty:
    fig, ax = plt.subplots()
    ax.plot(filtered_df["Date"], filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("📈 Cumulative Profit Over Time")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 📋 Show filtered results
    st.dataframe(filtered_df[["Date", "League", "Home", "Away", "Goals_H_FT", "Goals_A_FT", "Odd_H", "Odd_D", "Odd_A", "Profit", "Cumulative Profit"]])
else:
    st.warning("⚠️ No matches found with the selected filters.")
