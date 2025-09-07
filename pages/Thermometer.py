import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="Today's Picks - Momentum Thermometer", layout="wide")
st.title("Today's Picks - Momentum Thermometer")

# ---------------- Legend ----------------
st.markdown("""
Recommendation Rules (based only on Bands):

- Top20% vs Bottom20% → Back side of Top20
- Top20% vs Balanced → Back/Draw for the Top20 side
- Bottom20% vs Balanced → Back/Draw for the Balanced side
- Balanced vs Balanced → Avoid
- Top20% vs Top20% → Avoid
- Bottom20% vs Bottom20% → Avoid
""")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

M_DIFF_MARGIN = 0.30
POWER_MARGIN = 10

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

# League variation classification (Low/Medium/High)
def classify_leagues_variation(history_df):
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

# Per-league P20/P80 for M_H (Home) and M_A (Away)
def compute_league_bands(history_df):
    hist = history_df.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']

    home_q = (
        hist.groupby('League')['M_H']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Home_P20', 0.8:'Home_P80'})
            .reset_index()
    )

    away_q = (
        hist.groupby('League')['M_A']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Away_P20', 0.8:'Away_P80'})
            .reset_index()
    )

    return home_q.merge(away_q, on='League', how='inner')

# 🚨 NOVA LÓGICA
def auto_recommendation(row):
    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')

    # Top20 vs Bottom20
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return "✅ Back Home"
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return "✅ Back Away"

    # Top20 vs Balanced
    if band_home == 'Top 20%' and band_away == 'Balanced':
        return "🟦 1X (Home/Draw)"
    if band_home == 'Balanced' and band_away == 'Top 20%':
        return "🟪 X2 (Away/Draw)"

    # Bottom20 vs Balanced
    if band_home == 'Bottom 20%' and band_away == 'Balanced':
        return "🟪 X2 (Away/Draw)"
    if band_home == 'Balanced' and band_away == 'Bottom 20%':
        return "🟦 1X (Home/Draw)"

    # Outros casos
    return "❌ Avoid"

def event_side_for_winprob(auto_rec):
    if pd.isna(auto_rec): return None
    s = str(auto_rec)
    if 'Back Home' in s: return 'HOME'
    if 'Back Away' in s: return 'AWAY'
    if '1X' in s:       return '1X'
    if 'X2' in s:       return 'X2'
    return None

def win_prob_for_recommendation(history, row,
                                m_diff_margin=M_DIFF_MARGIN,
                                power_margin=POWER_MARGIN):
    m_h, m_a = row['M_H'], row['M_A']
    diff_m   = m_h - m_a
    diff_pow = row['Diff_Power']

    hist = history.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']

    mask = (
        hist['M_Diff'].between(diff_m - m_diff_margin, diff_m + m_diff_margin) &
        hist['Diff_Power'].between(diff_pow - power_margin, diff_pow + power_margin)
    )
    sample = hist[mask]
    n = len(sample)
    if n == 0:
        return 0, None

    target = event_side_for_winprob(row['Auto_Recommendation'])
    if target == 'HOME':
        p = (sample['Goals_H_FT'] > sample['Goals_A_FT']).mean()
    elif target == 'AWAY':
        p = (sample['Goals_A_FT'] > sample['Goals_H_FT']).mean()
    elif target == '1X':
        p = ((sample['Goals_H_FT'] >= sample['Goals_A_FT'])).mean()
    elif target == 'X2':
        p = ((sample['Goals_A_FT'] >= sample['Goals_H_FT'])).mean()
    else:
        p = None

    return n, (round(float(p)*100, 1) if p is not None else None)

# ---------------- Load Data ----------------
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

games_today = filter_leagues(load_last_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# ---------------- Derived Metrics ----------------
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
games_today = games_today.merge(league_class, on='League', how='left')
games_today = games_today.merge(league_bands, on='League', how='left')

games_today['Home_Band'] = np.where(
    games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
    np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
)
games_today['Away_Band'] = np.where(
    games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
    np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
)

games_today['Auto_Recommendation'] = games_today.apply(lambda r: auto_recommendation(r), axis=1)

ga_wp = games_today.apply(lambda r: win_prob_for_recommendation(history, r), axis=1)
games_today['Games_Analyzed']  = [x[0] for x in ga_wp]
games_today['Win_Probability'] = [x[1] for x in ga_wp]

games_today = games_today.sort_values(
    by=['Win_Probability'],
    ascending=False,
    na_position='last'
).reset_index(drop=True)

# ---------------- Display Table ----------------
cols_to_show = [
    'Date','Time','League','League_Classification',
    'Home','Away','Odd_H','Odd_D','Odd_A',
    'M_H','M_A','Diff_Power',
    'Home_Band','Away_Band','Auto_Recommendation',
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
