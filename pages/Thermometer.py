import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="Today's Picks - Momentum Thermometer", layout="wide")
st.title("Today's Picks - Momentum Thermometer")

# ---------------- Legend ----------------
st.markdown("""
### Recommendation Rules (Momentum Thermometer):

1. Strong edges -> 🟢 Back Home / 🟠 Back Away  
2. Moderate edges -> 1X (Home/Draw) or X2 (Away/Draw) when both are Balanced (with thresholds)  
3. If none of the above -> 1X if Home Balanced vs Away Bottom20%, or X2 if Away Balanced vs Home Bottom20%  
4. If none of the above -> 1X if Home Top20% vs Away Balanced, or X2 if Away Top20% vs Home Balanced  
5. **Balanced + Odd_D 2.5–6.0 + M_H (0–1) or M_A (0–0.5) -> ⚪ Back Draw**  
6. Otherwise -> Avoid
""")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa"]

M_DIFF_MARGIN = 0.30
POWER_MARGIN = 10
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
    if val == "Low Variation": return 'background-color: rgba(0, 200, 0, 0.12)'
    if val == "Medium Variation": return 'background-color: rgba(255, 215, 0, 0.12)'
    if val == "High Variation": return 'background-color: rgba(255, 0, 0, 0.10)'
    return ''

def color_band(val):
    if pd.isna(val): return ''
    if val == "Top 20%": return 'background-color: rgba(0, 128, 255, 0.10)'
    if val == "Bottom 20%": return 'background-color: rgba(255, 128, 0, 0.10)'
    if val == "Balanced": return 'background-color: rgba(200, 200, 200, 0.08)'
    return ''

def color_auto_rec(val):
    if pd.isna(val): return ''
    m = {
        "🟢 Back Home": 'background-color: rgba(0, 200, 0, 0.14)',
        "🟠 Back Away": 'background-color: rgba(255, 215, 0, 0.14)',
        "⚪ Back Draw": 'background-color: rgba(200, 200, 200, 0.14)',
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

def compute_league_bands(history_df):
    hist = history_df.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']
    diff_q = (
        hist.groupby('League')['M_Diff']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'P20_Diff', 0.8:'P80_Diff'})
            .reset_index()
    )
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
    out = diff_q.merge(home_q, on='League', how='inner').merge(away_q, on='League', how='inner')
    return out

def dominant_side(row, threshold=DOMINANT_THRESHOLD):
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

# ---------------- Auto Recommendation ----------------
def auto_recommendation(row,
                        diff_mid_lo=0.20, diff_mid_hi=0.80,
                        diff_mid_hi_highvar=0.75, power_gate=1, power_gate_highvar=5):

    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    league_cls= row.get('League_Classification', 'Medium Variation')
    m_a       = row.get('M_A')
    m_h       = row.get('M_H')
    odd_d     = row.get('Odd_D')

    # 1) Strong edges -> Direct Back
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '🟢 Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '🔵 Back Away'

    if dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if diff_m is not None and diff_m >= 0.90:
            return '🟢 Back Home'
    if dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home == 'Balanced':
        if diff_m is not None and diff_m <= -0.90:
            return '🟪 X2 (Away/Draw)'

    # 2) Both Balanced (with thresholds)
    if (band_home == 'Balanced') and (band_away == 'Balanced') and (diff_m is not None) and (diff_pow is not None):
        if league_cls == 'High Variation':
            if (diff_m >= 0.45 and diff_m < diff_mid_hi_highvar and diff_pow >= power_gate_highvar):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -0.45 and diff_m > -diff_mid_hi_highvar and diff_pow <= -power_gate_highvar):
                return '🟪 X2 (Away/Draw)'
        else:
            if (diff_m >= diff_mid_lo and diff_m < diff_mid_hi and diff_pow >= power_gate):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -diff_mid_lo and diff_m > -diff_mid_hi and diff_pow <= -power_gate):
                return '🟪 X2 (Away/Draw)'

    # 3) Balanced vs Bottom20%
    if (band_home == 'Balanced') and (band_away == 'Bottom 20%'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Balanced') and (band_home == 'Bottom 20%'):
        return '🟪 X2 (Away/Draw)'

    # 4) Top20% vs Balanced
    if (band_home == 'Top 20%') and (band_away == 'Balanced'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Top 20%') and (band_home == 'Balanced'):
        return '🟪 X2 (Away/Draw)'

    # 5) Novo filtro Draw
    if (odd_d is not None and 2.5 <= odd_d <= 6.0) and (diff_pow is not None and -10 <= diff_pow <= 10):
        if (m_h is not None and 0 <= m_h <= 1) or (m_a is not None and 0 <= m_a <= 0.5):
            return '⚪ Back Draw'

    # 6) Fallback
    return '❌ Avoid'


# ---------------- Win Probability Helpers ----------------
def event_side_for_winprob(auto_rec):
    if pd.isna(auto_rec): return None
    s = str(auto_rec)
    if 'Back Home' in s: return 'HOME'
    if 'Back Away' in s: return 'AWAY'
    if 'Back Draw' in s: return 'DRAW'
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
    elif target == 'DRAW':
        p = (sample['Goals_A_FT'] == sample['Goals_H_FT']).mean()
    elif target == '1X':
        p = ((sample['Goals_H_FT'] >= sample['Goals_A_FT'])).mean()
    elif target == 'X2':
        p = ((sample['Goals_A_FT'] >= sample['Goals_H_FT'])).mean()
    else:
        p_home = (sample['Goals_H_FT'] > sample['Goals_A_FT']).mean()
        p_away = (sample['Goals_A_FT'] > sample['Goals_H_FT']).mean()
        p = max(p_home, p_away)

    return n, (round(float(p)*100, 1) if p is not None else None)

# ---------------- Select CSV (Latest or Previous) ----------------
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select matchday file:", options, index=len(options)-1)

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# ---------------- Derived Metrics ----------------
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

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

games_today['Dominant'] = games_today.apply(dominant_side, axis=1)
games_today['Auto_Recommendation'] = games_today.apply(lambda r: auto_recommendation(r), axis=1)

# ---------------- Win Probability ----------------
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
    'Home_Band','Away_Band','Dominant','Auto_Recommendation',
    'Games_Analyzed','Win_Probability'
]

missing_cols = [c for c in cols_to_show if c not in games_today.columns]
if missing_cols:
    st.warning(f"Some expected columns are missing in today's data: {missing_cols}")

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

st.dataframe(styler, use_container_width=True, height=1000)
