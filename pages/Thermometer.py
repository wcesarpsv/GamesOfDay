import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------- Page Config ----------------
# Define o título da aba e o layout da página
st.set_page_config(page_title="Today's Picks - Momentum Thermometer", layout="wide")
st.title("Today's Picks - Momentum Thermometer")

# ---------------- Legend ----------------
# Explicação das regras que serão usadas nas recomendações
st.markdown("""
### Recommendation Rules (based on M_Diff and Bands):

- Strong edges -> Back side (Home/Away)
- Moderate edges -> 1X (Home/Draw) ou X2 (Away/Draw)  
   (duas lógicas: ambos Balanced **ou** Balanced vs Bottom20%)
- Avoid -> quando os sinais são fracos ou conflitantes
""")

# ---------------- Configs ----------------
# Diretório onde ficam os CSVs e parâmetros de análise
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa"]

M_DIFF_MARGIN = 0.30    # tolerância para encontrar jogos semelhantes no histórico
POWER_MARGIN = 10       # tolerância para Diff_Power
DOMINANT_THRESHOLD = 0.90  # limite para definir força dominante

# ---------------- Color Helpers ----------------
# Funções para colorir a tabela no Streamlit (visualização)
def color_diff_power(val):
    if pd.isna(val): 
        return ''
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
    if pd.isna(val): 
        return ''
    intensity = min(1, float(val) / 100.0)
    return f'background-color: rgba(0, 255, 0, {0.2 + 0.6 * intensity})'

def color_classification(val):
    if pd.isna(val): 
        return ''
    if val == "Low Variation":
        return 'background-color: rgba(0, 200, 0, 0.12)'
    if val == "Medium Variation":
        return 'background-color: rgba(255, 215, 0, 0.12)'
    if val == "High Variation":
        return 'background-color: rgba(255, 0, 0, 0.10)'
    return ''

def color_band(val):
    if pd.isna(val): 
        return ''
    if val == "Top 20%":
        return 'background-color: rgba(0, 128, 255, 0.10)'
    if val == "Bottom 20%":
        return 'background-color: rgba(255, 128, 0, 0.10)'
    if val == "Balanced":
        return 'background-color: rgba(200, 200, 200, 0.08)'
    return ''

def color_auto_rec(val):
    if pd.isna(val): 
        return ''
    m = {
        "✅ Back Home": 'background-color: rgba(0, 200, 0, 0.14)',
        "✅ Back Away": 'background-color: rgba(0, 200, 0, 0.14)',
        "🟦 1X (Home/Draw)": 'background-color: rgba(0, 128, 255, 0.12)',
        "🟪 X2 (Away/Draw)": 'background-color: rgba(128, 0, 255, 0.12)',
        "❌ Avoid": 'background-color: rgba(180, 180, 180, 0.10)',
    }
    return m.get(val, '')

# ---------------- Core Functions ----------------
# Carrega todos os CSVs (histórico)
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# Carrega apenas o último CSV (jogos do dia)
def load_last_csv(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    latest_file = max(files)
    return pd.read_csv(os.path.join(folder, latest_file))

# Remove ligas indesejadas (copas, amistosos etc.)
def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# Prepara histórico: apenas jogos finalizados com gols
def prepare_history(df):
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

# Classifica ligas como Low/Medium/High variation
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

# Calcula bandas (P20/P80) para M_Diff, M_H e M_A por liga
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

# Define se algum time é dominante
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
# Agora cobre duas lógicas para 1X/X2: ambos Balanced OU Balanced vs Bottom20%
def auto_recommendation(row,
                        diff_mid_lo=0.20, diff_mid_hi=0.80,
                        diff_mid_hi_highvar=0.75, power_gate=1, power_gate_highvar=5):

    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    league_cls= row.get('League_Classification', 'Medium Variation')

    # 1) Strong edges -> Back direto
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '✅ Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '✅ Back Away'

    if dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if diff_m is not None and diff_m >= 0.90:
            return '✅ Back Home'
    if dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home != 'Top 20%':
        if diff_m is not None and diff_m <= -0.90:
            return '✅ Back Away'

    # 2) Conservative edges (1X ou X2)
    both_balanced = (band_home == 'Balanced') and (band_away == 'Balanced')
    home_balanced_vs_away_bottom = (band_home == 'Balanced') and (band_away == 'Bottom 20%')
    away_balanced_vs_home_bottom = (band_away == 'Balanced') and (band_home == 'Bottom 20%')

    if (both_balanced or home_balanced_vs_away_bottom or away_balanced_vs_home_bottom) \
       and (diff_m is not None) and (diff_pow is not None):

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

    # 3) Caso nenhum critério seja atendido
    return '❌ Avoid'
