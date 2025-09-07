import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="Today's Picks - Momentum Thermometer", layout="wide")
st.title("Today's Picks - Momentum Thermometer")

# ---------------- Legend ----------------
st.markdown("""
Recommendation Rules (based on M_Diff and Bands):

- Strong edges -> Back side (Home/Away)
- Moderate edges with both teams Balanced -> 1X (Home/Draw) or X2 (Away/Draw)
- Avoid when signals are weak or conflicting
""")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

# Similarity windows used for historical estimation
M_DIFF_MARGIN = 0.30
POWER_MARGIN = 10

# Threshold used to decide Dominant side by individual strength
DOMINANT_THRESHOLD = 0.90

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
    intensity = min(1, float(val) / 100.0)
    return f'background-color: rgba(0, 255, 0, {0.2 + 0.6 * intensity})'

def color_classification(val):
    if pd.isna(val): return ''
    if val == "Low Variation":
        return 'background-color: rgba(0, 200, 0, 0.12)'
    if val == "Medium Variation":
        return 'background-color: rgba(255, 215, 0, 0.12)'
    if val == "High Variation":
        return 'background-color: rgba(255, 0, 0, 0.10)'
    return ''

def color_band(val):
    if pd.isna(val): return ''
    if val == "Top 20%":
        return 'background-color: rgba(0, 128, 255, 0.10)'
    if val == "Bottom 20%":
        return 'background-color: rgba(255, 128, 0, 0.10)'
    if val == "Balanced":
        return 'background-color: rgba(200, 200, 200, 0.08)'
    return ''

def color_auto_rec(val):
    if pd.isna(val): return ''
    m = {
        "✅ Back Home": 'background-color: rgba(0, 200, 0, 0.14)',
        "✅ Back Away": 'background-color: rgba(0, 200, 0, 0.14)',
        "🟦 1X (Home/Draw)": 'background-color: rgba(0, 128, 255, 0.12)',
        "🟪 X2 (Away/Draw)": 'background-color: rgba(128, 0, 255, 0.12)',
        "❌ Avoid": 'background-color: rgba(180, 180, 180, 0.10)',
    }
    return m.get(val, '')

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
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def prepare_history(df):
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

# ---------------- Load Data ----------------
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

games_today = filter_leagues(load_last_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# ---------------- Derived + Display ----------------
# (todo: resto já está implementado no trecho anterior que você mandou)
# ...
# No final:

cols_to_show = [
    'Date','Time','League','League_Classification',
    'Home','Away','Odd_H','Odd_D','Odd_A',
    'M_H','M_A','Diff_Power',
    'Home_Band','Away_Band','Dominant','Auto_Recommendation',
    'Games_Analyzed','Win_Probability'
]

display_cols = [c for c in cols_to_show if c in games_today.columns]

styler = (
    games_today[display_cols]
    .style
    .applymap(color_diff_power, subset=['Diff_Power'])
    .applymap(color_probability, subset=['Win_Probability'])
    .applymap(color_classification, subset=['League_Classification'])
    .applymap(color_band, subset=['Home_Band','Away_Band'])
    .applymap(color_auto_rec, subset=['Auto_Recommendation'])
    .format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'M_H': '{:.2f}', 'M_A': '{:.2f}',
        'Diff_Power': '{:.2f}',
        'Win_Probability': '{:.1f}%', 'Games_Analyzed': '{:,.0f}'
    }, na_rep='—')
)

st.dataframe(styler, use_container_width=True)
