import streamlit as st
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(page_title="League Ranking – Performance & Momentum", layout="wide")

# ---------------- Header with Toggle ----------------
col1, col2 = st.columns([4, 2])
with col1:
    st.title("📊 League Ranking – Performance & Momentum")
with col2:
    view_mode = st.radio(
        "View Mode",
        ["General", "Home only", "Away only"],
        horizontal=True
    )

st.markdown(f"**Current mode:** {view_mode}")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("⚙️ Filters")
league = st.sidebar.selectbox("Select League", ["Premier League", "Serie B", "MLS", "K-League"])
period = st.sidebar.selectbox("Select Period", ["Last 10 Games", "Last 30 Days", "Full Season"])
order_by = st.sidebar.selectbox("Order by", ["ROI", "Winrate", "Diff_Power", "Diff_Momentum"])

# ---------------- Mock Data ----------------
data = {
    "Rank": ["🥇 1", "🥈 2", "🥉 3", "4", "5"],
    "Team": ["Arsenal", "Liverpool", "Tottenham", "Newcastle", "Chelsea"],
    "Games": [6, 6, 5, 6, 5],
    "Winrate": ["83%", "67%", "60%", "50%", "20%"],
    "ROI (%)": ["+24.5", "+12.0", "+8.3", "+3.5", "-18.7"],
    "Streak": ["4W", "2W", "1L", "2W 1L", "4L"],
    "Avg Diff_Power": [18.2, 15.4, 12.7, 10.2, 7.5],
    "Avg Diff_Momentum": [0.75, 0.62, 0.48, 0.05, -0.35]
}
df_mock = pd.DataFrame(data)

# ---------------- Display Table ----------------
st.subheader(f"📌 {league} – {period} – {view_mode}")
st.dataframe(df_mock, use_container_width=True)
