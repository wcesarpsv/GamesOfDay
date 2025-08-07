
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest", layout="wide")

st.title("📈 Strategy Backtest")

# 🔹 Folder containing match data CSVs
GAMES_FOLDER = "GamesDay"

# ⬇️ Load all valid CSVs with goal data
all_dfs = []

for file in sorted(os.listdir(GAMES_FOLDER)):  # Sort files alphabetically (oldest first)
    if file.endswith(".csv"):
        df_path = os.path.join(GAMES_FOLDER, file)
        df = pd.read_csv(df_path)

        # ⚠️ Skip if goal columns are missing
        if 'Goals_H_FT' not in df.columns or 'Goals_A_FT' not in df.columns:
            continue

        # ⏱️ Ensure date is parsed
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        all_dfs.append(df)

# 🚨 Stop if nothing valid
if not all_dfs:
    st.error("❌ No valid data with goal columns found.")
    st.stop()

# 🧱 Combine all valid CSVs
df_all = pd.concat(all_dfs, ignore_index=True)

# 📅 Order by date
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# 🎚️ Filter sliders
st.sidebar.header("🎯 Filter Matches")

diff_power = st.sidebar.slider("📊 Diff_Power", float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max()), (float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())))
diff_ht_p = st.sidebar.slider("📉 Diff_HT_P", float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max()), (float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())))
odd_h = st.sidebar.slider("💰 Odd_H (Home win)", float(df_all["Odd_H"].min()), float(df_all["Odd_H"].max()), (float(df_all["Odd_H"].min()), float(df_all["Odd_H"].max())))
odd_d = st.sidebar.slider("💰 Odd_D (Draw)", float(df_all["Odd_D"].min()), float(df_all["Odd_D"].max()), (float(df_all["Odd_D"].min()), float(df_all["Odd_D"].max())))
odd_a = st.sidebar.slider("💰 Odd_A (Away win)", float(df_all["Odd_A"].min()), float(df_all["Odd_A"].max()), (float(df_all["Odd_A"].min()), float(df_all["Odd_A"].max())))

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 🧮 Apply filters
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power[0]) & (df_all["Diff_Power"] <= diff_power[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p[0]) & (df_all["Diff_HT_P"] <= diff_ht_p[1]) &
    (df_all["Odd_H"] >= odd_h[0]) & (df_all["Odd_H"] <= odd_h[1]) &
    (df_all["Odd_D"] >= odd_d[0]) & (df_all["Odd_D"] <= odd_d[1]) &
    (df_all["Odd_A"] >= odd_a[0]) & (df_all["Odd_A"] <= odd_a[1])
].copy()

# 🧠 Calculate bet result
def calculate_profit(row):
    if bet_on == "Home":
        if row["Goals_H_FT"] > row["Goals_A_FT"]:
            return row["Odd_H"] - 1
        else:
            return -1
    elif bet_on == "Away":
        if row["Goals_A_FT"] > row["Goals_H_FT"]:
            return row["Odd_A"] - 1
        else:
            return -1
    return 0

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # 📈 Plot profit by bet number
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(filtered_df)), filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("Cumulative Profit by Bet")
    st.pyplot(fig)

    # 📋 Show filtered table
    st.subheader("📝 Filtered Matches")
    st.dataframe(filtered_df[["Date", "League", "Home", "Away", "Odd_H", "Odd_D", "Odd_A", "Diff_Power", "Diff_HT_P", "Goals_H_FT", "Goals_A_FT", "Bet Result", "Cumulative Profit"]])
else:
    st.warning("⚠️ No matches found with selected filters.")
