import streamlit as st
import pandas as pd
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="Today's Picks – Momentum Thermometer", layout="wide")
st.title("📊 Today's Picks – Momentum Thermometer")

# ---------------- Legend ----------------
st.markdown("""
**🔍 Recommendation Rules (based on M_Diff):**

- 🔵 **Home (Strong)** → M_Diff ≥ +0.9 with high Diff_Power  
- 🔴 **Away (Strong)** → M_Diff ≤ -0.9 with low Diff_Power  
- ✅ **Home / Away** → Clear directional advantage  
- 🟦 **Home/Draw** → Slight edge for Home  
- 🟥 **Away/Draw** → Slight edge for Away  
- ❌ **Avoid** → Balanced match (M_Diff between -0.3 and +0.3)
""")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

# ---------------- Color Helpers ----------------
def color_diff_power(val):
    if pd.isna(val): return ''
    if -8 <= val <= 8:
        intensity = 1 - (abs(val) / 8)
        return f'background-color: rgba(255, 255, 0, {0.3 + 0.4 * intensity})'
    if val > 0:
        intensity = min(1, val / 25)
        return f'background-color: rgba(0, 255, 0, {0.3 + 0.4 * intensity})'
    if val < 0:
        intensity = min(1, abs(val) / 25)
        return f'background-color: rgba(255, 0, 0, {0.3 + 0.4 * intensity})'

def color_probability(val):
    if pd.isna(val): return ''
    intensity = min(1, val / 100)
    return f'background-color: rgba(0, 255, 0, {0.2 + 0.6 * intensity})'

# ---------------- Core Functions ----------------
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def load_last_csv(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: return pd.DataFrame()
    latest_file = max(files)
    return pd.read_csv(os.path.join(folder, latest_file))

def filter_leagues(df):
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)]

def prepare_history(df):
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

def recommend_bet(m_h, m_a, diff_power, power_support=10):
    m_diff = m_h - m_a
    abs_diff = abs(m_diff)
    if abs_diff < 0.3:
        return "❌ Avoid"
    if 0.3 <= m_diff < 0.6:
        return "🟦 Home/Draw"
    elif -0.6 < m_diff <= -0.3:
        return "🟥 Away/Draw"
    if 0.6 <= m_diff < 0.9:
        return "✅ Home"
    elif -0.9 < m_diff <= -0.6:
        return "✅ Away"
    if m_diff >= 0.9:
        return "🔵 Home (Strong)" if diff_power > power_support else "✅ Home"
    elif m_diff <= -0.9:
        return "🔴 Away (Strong)" if diff_power < -power_support else "✅ Away"
    return "❌ Avoid"

def count_similar_matches(history_df, m_h, m_a, diff_power, side, m_diff_margin=0.3, power_margin=10):
    m_diff = m_h - m_a
    history_df = history_df.copy()
    history_df['M_Diff'] = history_df['M_H'] - history_df['M_A']
    if side == "Home":
        win_mask = history_df['Goals_H_FT'] > history_df['Goals_A_FT']
    elif side == "Away":
        win_mask = history_df['Goals_A_FT'] > history_df['Goals_H_FT']
    else:
        return 0, None
    mask = (
        history_df['M_Diff'].between(m_diff - m_diff_margin, m_diff + m_diff_margin) &
        history_df['Diff_Power'].between(diff_power - power_margin, diff_power + power_margin)
    )
    filtered = history_df[mask]
    total = len(filtered)
    if total == 0:
        return 0, None
    wins = win_mask[filtered.index].sum()
    return total, round((wins / total) * 100, 1)

def extract_side(reco):
    if "Home" in reco:
        return "Home"
    elif "Away" in reco:
        return "Away"
    else:
        return None

# ---------------- Load Data ----------------
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

games_today = filter_leagues(load_last_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# ---------------- Apply Logic ----------------
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
games_today['Recommendation'] = games_today.apply(
    lambda row: recommend_bet(row['M_H'], row['M_A'], row['Diff_Power']), axis=1
)
games_today['Side'] = games_today['Recommendation'].apply(extract_side)

results = games_today.apply(
    lambda row: count_similar_matches(history, row['M_H'], row['M_A'], row['Diff_Power'], row['Side']),
    axis=1
)

games_today['Games_Analyzed'] = [r[0] for r in results]
games_today['Win_Probability'] = [r[1] for r in results]

games_today = games_today.sort_values(by='Win_Probability', ascending=False)

# ---------------- Display Table ----------------
st.dataframe(
    games_today[['Date','Time','League','Home','Away','Odd_H','Odd_D','Odd_A','M_H','M_A','M_Diff','Diff_Power','Recommendation','Games_Analyzed','Win_Probability']]
    .style
    .applymap(color_diff_power, subset=['Diff_Power'])
    .applymap(color_probability, subset=['Win_Probability'])
    .format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'M_H': '{:.2f}', 'M_A': '{:.2f}', 'M_Diff': '{:.2f}',
        'Diff_Power': '{:.2f}', 'Win_Probability': '{:.1f}%', 'Games_Analyzed': '{:,.0f}'
    })
)

