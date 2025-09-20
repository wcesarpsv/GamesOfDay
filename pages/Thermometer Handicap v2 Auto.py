import streamlit as st
import pandas as pd
import numpy as np
import os

########################################
########## Bloco 1 – Configs ###########
########################################
st.set_page_config(page_title="Today's Picks - Momentum Thermometer", layout="wide")
st.title("Today's Picks - Momentum Thermometer")

# Legenda
st.markdown("""
### Recommendation Rules (Momentum Thermometer - Data-driven):

1. Primeiro procura maior Winrate em vitórias puras (Home, Away, Draw).  
   - Se Winrate ≥ 50% → escolha.  
2. Caso contrário → procura 1X ou X2.  
3. Se nada ≥ 50% → ❌ Avoid.  
4. Colunas adicionais:
   - **Win_Probability** (% histórico)
   - **EV** (Expected Value, apenas auditoria)
   - **Bands** (Balanced / Top20 / Bottom20), mas também usados como features numéricas internas.
""")

# Pastas e exclusões
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa","afc"]

# Parâmetros
M_DIFF_MARGIN = 0.30
POWER_MARGIN = 10
DOMINANT_THRESHOLD = 0.90


########################################
####### Bloco 2 – Color Helpers ########
########################################
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
    if val >= 50:
        return 'background-color: rgba(0, 200, 0, 0.14)'  # verde
    else:
        return 'background-color: rgba(255, 0, 0, 0.14)'  # vermelho

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

def color_ev(val):
    if pd.isna(val): return ''
    if val > 0:
        return 'background-color: rgba(0, 200, 0, 0.14)'
    else:
        return 'background-color: rgba(255, 0, 0, 0.14)'


########################################
###### Bloco 3 – Core Functions ########
########################################
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

def add_band_features(df):
    """Adiciona Home_Band, Away_Band e versões numéricas ao histórico."""
    if df.empty:
        return df
    BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}
    df["Home_Band_Num"] = df["Home_Band"].map(BAND_MAP) if "Home_Band" in df else None
    df["Away_Band_Num"] = df["Away_Band"].map(BAND_MAP) if "Away_Band" in df else None
    return df


########################################
### Bloco 4 – Win Prob / EV Helpers ####
########################################

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
                                base_m_diff=0.30,
                                base_power=10,
                                min_games=10,
                                max_m_diff=1.0,
                                max_power=25):
    """
    Calcula Win Probability com ranges automáticos:
    - Começa estreito (precisão alta)
    - Se não houver jogos suficientes, expande ranges até achar
    - Inclui fallback com bands e odds implícitas
    """
    m_h, m_a = row['M_H'], row['M_A']
    diff_m   = m_h - m_a
    diff_pow = row['Diff_Power']

    hist = history.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']

    # Inicializa ranges
    m_diff_margin = base_m_diff
    power_margin = base_power
    sample = pd.DataFrame()
    n = 0

    while n < min_games and (m_diff_margin <= max_m_diff and power_margin <= max_power):
        mask = (
            hist['M_Diff'].between(diff_m - m_diff_margin, diff_m + m_diff_margin) &
            hist['Diff_Power'].between(diff_pow - power_margin, diff_pow + power_margin)
        )

        # Bands (flexível: aceita Balanced como similar)
        home_candidates = [row['Home_Band_Num']]
        away_candidates = [row['Away_Band_Num']]
        if row['Home_Band_Num'] in [1, 3]: home_candidates.append(2)
        if row['Away_Band_Num'] in [1, 3]: away_candidates.append(2)

        mask = mask & hist['Home_Band_Num'].isin(home_candidates) & hist['Away_Band_Num'].isin(away_candidates)

        sample = hist[mask]
        n = len(sample)

        # Se não achar suficiente → expande ranges
        if n < min_games:
            m_diff_margin += 0.20
            power_margin += 5

    if row.get('Auto_Recommendation') == '❌ Avoid':
        return n, None
    if n == 0:
        # fallback absoluto: usar odd implícita
        target = event_side_for_winprob(row['Auto_Recommendation'])
        if target == 'HOME' and row.get("Odd_H"):
            return 0, round(100 / row["Odd_H"], 1)
        if target == 'AWAY' and row.get("Odd_A"):
            return 0, round(100 / row["Odd_A"], 1)
        if target == 'DRAW' and row.get("Odd_D"):
            return 0, round(100 / row["Odd_D"], 1)
        return 0, None

    # ---- Calcula probabilidade real
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


########################################
####### Bloco 5 – Auto Selection #######
########################################
def auto_recommendation_hybrid(row, history,
                               min_games=5,
                               min_winrate=45.0):
    """
    Híbrido:
    1. Tenta pelas regras fixas (bands, dominant, thresholds).
    2. Valida com histórico (Win Probability + EV).
    3. Se cair em Avoid → chama fallback automático (dynamic winrate).
    """

    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    league_cls= row.get('League_Classification', 'Medium Variation')
    m_a       = row.get('M_A')
    m_h       = row.get('M_H')
    odd_d     = row.get('Odd_D')

    rec = None
    # === Regras determinísticas ===
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        rec = '🟢 Back Home'
    elif band_home == 'Bottom 20%' and band_away == 'Top 20%':
        rec = '🟠 Back Away'
    elif dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if diff_m is not None and diff_m >= 0.90:
            rec = '🟢 Back Home'
    elif dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home == 'Balanced':
        if diff_m is not None and diff_m <= -0.90:
            rec = '🟪 X2 (Away/Draw)'
    elif (band_home == 'Balanced') and (band_away == 'Balanced') and (diff_m is not None) and (diff_pow is not None):
        if league_cls == 'High Variation':
            if (diff_m >= 0.45 and diff_m < 0.75 and diff_pow >= 5):
                rec = '🟦 1X (Home/Draw)'
            if (diff_m <= -0.45 and diff_m > -0.75 and diff_pow <= -5):
                rec = '🟪 X2 (Away/Draw)'
        else:
            if (diff_m >= 0.20 and diff_m < 0.80 and diff_pow >= 1):
                rec = '🟦 1X (Home/Draw)'
            if (diff_m <= -0.20 and diff_m > -0.80 and diff_pow <= -1):
                rec = '🟪 X2 (Away/Draw)'
    elif (band_home == 'Balanced') and (band_away == 'Bottom 20%'):
        rec = '🟦 1X (Home/Draw)'
    elif (band_away == 'Balanced') and (band_home == 'Bottom 20%'):
        rec = '🟪 X2 (Away/Draw)'
    elif (band_home == 'Top 20%') and (band_away == 'Balanced'):
        rec = '🟦 1X (Home/Draw)'
    elif (band_away == 'Top 20%') and (band_home == 'Balanced'):
        rec = '🟪 X2 (Away/Draw)'
    elif (odd_d is not None and 2.5 <= odd_d <= 6.0) and (diff_pow is not None and -10 <= diff_pow <= 10):
        if (m_h is not None and 0 <= m_h <= 1) or (m_a is not None and 0 <= m_a <= 0.5):
            rec = '⚪ Back Draw'

    if rec is None:
        rec = '❌ Avoid'

    # === Validação no histórico ===
    row_copy = row.copy()
    row_copy["Auto_Recommendation"] = rec
    n, p = win_prob_for_recommendation(history, row_copy)

    # Odds de referência
    odd_ref = None
    if rec == "🟢 Back Home": odd_ref = row.get("Odd_H")
    elif rec == "🟠 Back Away": odd_ref = row.get("Odd_A")
    elif rec == "⚪ Back Draw": odd_ref = row.get("Odd_D")
    elif rec == "🟦 1X (Home/Draw)" and row.get("Odd_1X"): odd_ref = row["Odd_1X"]
    elif rec == "🟪 X2 (Away/Draw)" and row.get("Odd_X2"): odd_ref = row["Odd_X2"]

    ev = (p/100.0) * odd_ref - 1 if (odd_ref and p) else None

    # Se falhar nas regras → fallback automático
    if rec == "❌ Avoid" or (p is None) or (n < min_games) or (p < min_winrate):
        return auto_recommendation_dynamic_winrate(row, history, min_games, min_winrate)

    return rec, p, ev, n




########################################
######## Bloco 6 – Load Data ###########
########################################

files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select matchday file:", options, index=len(options)-1)

# Carrega jogos do dia
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Mantém apenas jogos futuros (sem resultado ainda)
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# Carrega histórico
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

# Calcula variação de ligas e bands por quantis
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

# Cria coluna de diferença
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']

# Faz os merges (para trazer quantis)
games_today = games_today.merge(league_class, on='League', how='left')
games_today = games_today.merge(league_bands, on='League', how='left')

# ==== Criar bandas textuais nos jogos do dia ====
games_today['Home_Band'] = np.where(
    games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
    np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
)
games_today['Away_Band'] = np.where(
    games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
    np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
)

# ==== Mapear bandas para valores numéricos (1,2,3) ====
BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}
games_today["Home_Band_Num"] = games_today["Home_Band"].map(BAND_MAP)
games_today["Away_Band_Num"] = games_today["Away_Band"].map(BAND_MAP)

# ==== Criar bandas também no histórico ====
history = history.merge(league_bands, on="League", how="left")

history['Home_Band'] = np.where(
    history['M_H'] <= history['Home_P20'], 'Bottom 20%',
    np.where(history['M_H'] >= history['Home_P80'], 'Top 20%', 'Balanced')
)
history['Away_Band'] = np.where(
    history['M_A'] <= history['Away_P20'], 'Bottom 20%',
    np.where(history['M_A'] >= history['Away_P80'], 'Top 20%', 'Balanced')
)
history["Home_Band_Num"] = history["Home_Band"].map(BAND_MAP)
history["Away_Band_Num"] = history["Away_Band"].map(BAND_MAP)

# ==== Aproximação das odds 1X e X2 ====
def compute_double_chance_odds(df):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df['Odd_H']
    probs['p_D'] = 1 / df['Odd_D']
    probs['p_A'] = 1 / df['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)  # normaliza (remove vig)

    df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

games_today = compute_double_chance_odds(games_today)

# ==== Continuar pipeline normal ====
games_today['Dominant'] = games_today.apply(dominant_side, axis=1)

# Aplica recomendação + métricas (respeitando lógica híbrida)
recs = games_today.apply(lambda r: auto_recommendation_hybrid(r, history), axis=1)
games_today["Auto_Recommendation"] = [x[0] for x in recs]
games_today["Win_Probability"] = [x[1] for x in recs]
games_today["EV"] = [x[2] for x in recs]
games_today["Games_Analyzed"] = [x[3] for x in recs]




########################################
######## Bloco 7 – Exibição ############
########################################
cols_to_show = [
    'Date','Time','League','League_Classification',
    'Home','Away','Odd_H','Odd_D','Odd_A',
    'M_H','M_A','Diff_Power',
    'Home_Band','Away_Band',
    'Dominant','Auto_Recommendation',
    'Games_Analyzed','Win_Probability','EV'
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
    .applymap(color_ev, subset=['EV'])
    .format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'M_H': '{:.2f}', 'M_A': '{:.2f}',
        'Diff_Power': '{:.2f}',
        'Win_Probability': '{:.1f}%', 'EV': '{:.2f}', 'Games_Analyzed': '{:,.0f}'
    }, na_rep='—')
)

st.dataframe(styler, use_container_width=True, height=1000)
