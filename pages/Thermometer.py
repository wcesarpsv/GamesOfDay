import streamlit as st
import pandas as pd
import numpy as np
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
    if val == "Balanced P20-80":
        return 'background-color: rgba(200, 200, 200, 0.08)'
    return ''

def color_auto_rec(val):
    if pd.isna(val): return ''
    m = {
        "✅ Back Home": 'background-color: rgba(0, 200, 0, 0.14)',
        "✅ Back Away": 'background-color: rgba(0, 200, 0, 0.14)',
        "🟦 1X (Home/Draw)": 'background-color: rgba(0, 128, 255, 0.12)',
        "🟪 X2 (Away/Draw)": 'background-color: rgba(128, 0, 255, 0.12)',
        "🟥 Against Home": 'background-color: rgba(255, 0, 0, 0.10)',
        "🟥 Against Away": 'background-color: rgba(255, 0, 0, 0.10)',
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
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)]

def prepare_history(df):
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

# --- League Variation (Low/Medium/High)
def classify_leagues_variation(history_df):
    """
    Variation_Total = (max(M_H)-min(M_H)) + (max(M_A)-min(M_A))
    Thresholds: Low < 3.0, 3.0–6.0 Medium, > 6.0 High
    """
    agg = (
        history_df.groupby('League')
        .agg(
            M_H_Min=('M_H','min'), M_H_Max=('M_H','max'),
            M_A_Min=('M_A','min'), M_A_Max=('M_A','max'),
            Hist_Games=('M_H','count')
        ).reset_index()
    )
    agg['Variation_Total'] = (agg['M_H_Max'] - agg['M_H_Min']) + (agg['M_A_Max'] - agg['M_A_Min'])
    def label(v):
        if v > 6.0: return "High Variation"
        if v >= 3.0: return "Medium Variation"
        return "Low Variation"
    agg['League_Classification'] = agg['Variation_Total'].apply(label)
    return agg[['League','League_Classification','Variation_Total','Hist_Games']]

# --- Per-league P20/P80 bands for Diff_M
def compute_league_bands(history_df):
    """
    Per-league P20/P80 for Diff_M = M_H - M_A.
    Your dataset already guarantees ≥10 league matches per team, so we assume enough data.
    """
    hist = history_df.copy()
    hist['Diff_M'] = hist['M_H'] - hist['M_A']
    q = (
        hist.groupby('League')['Diff_M']
            .quantile([0.20, 0.80])
            .unstack()
            .rename(columns={0.2:'P20_Diff', 0.8:'P80_Diff'})
            .reset_index()
    )
    return q[['League','P20_Diff','P80_Diff']]

# --- Original recommendation (kept)
def recommend_bet(m_h, m_a, diff_power, power_support=10):
    m_diff = m_h - m_a
    abs_diff = abs(m_diff)
    if abs_diff < 0.296:
        return "❌ Avoid"
    if 0.296 <= m_diff < 0.6:
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

# --- Count similar matches (kept)
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

# --- NEW: Dominant side driver (symmetrical for strong/weak on both teams)
def dominant_side(row, threshold=0.9):
    """
    Identify which side drives the imbalance (symmetric):
      - Both extremes (Home strong & Away weak): M_H >= thr and M_A <= -thr
      - Both extremes (Away strong & Home weak): M_A >= thr and M_H <= -thr
      - Home strong:  M_H >= thr
      - Home weak:    M_H <= -thr
      - Away strong:  M_A >= thr
      - Away weak:    M_A <= -thr
      - Mixed / Neutral: otherwise
    """
    m_h, m_a = row['M_H'], row['M_A']
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold:
        return "Home strong"
    if m_h <= -threshold:
        return "Home weak"
    if m_a >= threshold:
        return "Away strong"
    if m_a <= -threshold:
        return "Away weak"
    return "Mixed / Neutral"

# --- NEW: Auto recommendation badge based on Band + Dominant
def auto_recommendation(band, dominant):
    """
    Simple rules:
      - Top 20%:
          * Home strong / Both(Home↑ & Away↓) -> ✅ Back Home
          * Away weak -> ✅ Back Home or 🟥 Against Away (choose safer)
          * Away strong / Home weak (rare with Top band) -> ❌ Avoid
      - Bottom 20%:
          * Away strong / Both(Away↑ & Home↓) -> ✅ Back Away
          * Home weak -> 🟥 Against Home or ✅ Back Away
          * Home strong / Away weak (rare with Bottom band) -> ❌ Avoid
      - Balanced: ❌ Avoid
    """
    if pd.isna(band) or pd.isna(dominant):
        return "❌ Avoid"

    band = str(band)
    dominant = str(dominant)

    if band == "Top 20%":
        if dominant in ["Home strong", "Both extremes (Home↑ & Away↓)"]:
            return "✅ Back Home"
        if dominant == "Away weak":
            return "✅ Back Home"
        # counter-intuitive edge cases
        return "❌ Avoid"

    if band == "Bottom 20%":
        if dominant in ["Away strong", "Both extremes (Away↑ & Home↓)"]:
            return "✅ Back Away"
        if dominant == "Home weak":
            return "✅ Back Away"
        return "❌ Avoid"

    # Balanced
    return "❌ Avoid"

# ---------------- Load Data ----------------
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

# Compute league classification and per-league bands
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

games_today = filter_leagues(load_last_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# ---------------- Apply Logic ----------------
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
games_today['Recommendation'] = games_today.apply(
    lambda row: recommend_bet(row['M_H'], row['M_A'], row['Diff_Power']), axis=1
)
games_today['Side'] = games_today['Recommendation'].apply(extract_side)

# Attach league classification and bands
games_today = games_today.merge(league_class, on='League', how='left')
games_today = games_today.merge(league_bands, on='League', how='left')

# Band per match (vectorized)
games_today['Band'] = np.where(
    games_today['M_Diff'] <= games_today['P20_Diff'], 'Bottom 20%',
    np.where(games_today['M_Diff'] >= games_today['P80_Diff'], 'Top 20%', 'Balanced P20-80')
)

# Dominant side
games_today['Dominant'] = games_today.apply(dominant_side, axis=1)

# Auto recommendation badge
games_today['Auto_Recommendation'] = games_today.apply(
    lambda r: auto_recommendation(r['Band'], r['Dominant']), axis=1
)

# Similar matches & win prob (as in your original)
results = games_today.apply(
    lambda row: count_similar_matches(history, row['M_H'], row['M_A'], row['Diff_Power'], row['Side']),
    axis=1
)
games_today['Games_Analyzed'] = [r[0] for r in results]
games_today['Win_Probability'] = [r[1] for r in results]

games_today = games_today.sort_values(by='Win_Probability', ascending=False)

# ---------------- Display Table ----------------
cols_to_show = [
    'Date','Time','League','League_Classification','Band',
    'Home','Away','Odd_H','Odd_D','Odd_A',
    'M_H','M_A','M_Diff','Diff_Power',
    'Dominant','Auto_Recommendation',
    'Recommendation','Games_Analyzed','Win_Probability'
]

styler = (
    games_today[cols_to_show]
    .style
    .applymap(color_diff_power, subset=['Diff_Power'])
    .applymap(color_probability, subset=['Win_Probability'])
    .applymap(color_classification, subset=['League_Classification'])
    .applymap(color_band, subset=['Band'])
    .applymap(color_auto_rec, subset=['Auto_Recommendation'])
    .format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'M_H': '{:.2f}', 'M_A': '{:.2f}', 'M_Diff': '{:.2f}',
        'Diff_Power': '{:.2f}', 'Win_Probability': '{:.1f}%', 'Games_Analyzed': '{:,.0f}'
    })
)

st.dataframe(styler)
