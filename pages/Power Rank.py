import streamlit as st
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(page_title="League Ranking – Performance & Momentum", layout="wide")
st.title("📊 League Ranking – Performance & Momentum")

st.markdown("""
This page allows you to compare team performance within a league, 
based on ROI, Winrate, Diff_Power, and Diff_Momentum.
""")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("⚙️ Filters")

league = st.sidebar.selectbox(
    "Select League",
    ["Premier League", "Serie B (Brazil)", "MLS", "K-League"]
)

period = st.sidebar.selectbox(
    "Select Period",
    ["Last 10 Games", "Last 30 Days", "Full Season"]
)

order_by = st.sidebar.selectbox(
    "Order by",
    ["ROI", "Winrate", "Diff_Power", "Diff_Momentum"]
)

st.sidebar.markdown("----")
min_games = st.sidebar.checkbox("Show only teams with more than 5 games")
highlight_streaks = st.sidebar.checkbox("Highlight streaks with 3+ wins/losses")

# ---------------- Mock Data (example only) ----------------
data = {
    "Rank": ["🥇 1", "🥈 2", "🥉 3", "4", "5", "6", "7"],
    "Team": ["Arsenal", "Liverpool", "Tottenham", "Newcastle", "Aston Villa", "Chelsea", "Nott'm Forest"],
    "Games": [6, 6, 5, 6, 6, 5, 5],
    "Winrate": ["83%", "67%", "60%", "50%", "33%", "20%", "0%"],
    "ROI (%)": ["+24.5", "+12.0", "+8.3", "+3.5", "-5.0", "-18.7", "-25.0"],
    "Streak": ["4W", "2W", "1L", "2W 1L", "3L", "4L", "5L"],
    "Avg Diff_Power": [18.2, 15.4, 12.7, 10.2, 6.8, 7.5, -4.3],
    "Avg Diff_Momentum": [0.75, 0.62, 0.48, 0.05, -0.22, -0.35, -0.70],
    "Notes": ["Hot streak", "Solid form", "Consistent", "Oscillating", "Declining", "Disappointing", "Very poor form"]
}

df_mock = pd.DataFrame(data)

# ---------------- Display Table ----------------
st.subheader(f"📌 {league} – {period}")
st.dataframe(df_mock, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("""
---
**Notes:**  
- **Diff_Power** = historical team strength  
- **Diff_Momentum** = recent trend (M_H – M_A)  
- **ROI** = hypothetical betting return  
""")
