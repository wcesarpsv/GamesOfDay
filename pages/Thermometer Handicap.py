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
                                m_diff_margin=M_DIFF_MARGIN,
                                power_margin=POWER_MARGIN):
    """Agora inclui Home_Band_Num e Away_Band_Num no filtro histórico"""
    m_h, m_a = row['M_H'], row['M_A']
    diff_m   = m_h - m_a
    diff_pow = row['Diff_Power']

    hist = history.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']

    mask = (
        hist['M_Diff'].between(diff_m - m_diff_margin, diff_m + m_diff_margin) &
        hist['Diff_Power'].between(diff_pow - power_margin, diff_pow + power_margin) &
        (hist['Home_Band_Num'] == row['Home_Band_Num']) &
        (hist['Away_Band_Num'] == row['Away_Band_Num'])
    )
    sample = hist[mask]
    n = len(sample)

    if row.get('Auto_Recommendation') == '❌ Avoid':
        return n, None
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


########################################
####### Bloco 5 – Auto Selection #######
########################################
def auto_recommendation_dynamic_winrate(row, history,
                                        m_diff_margin=M_DIFF_MARGIN,
                                        power_margin=POWER_MARGIN,
                                        min_games=30):
    """Escolhe recomendação baseada no maior Winrate, com fallback 1X/X2"""
    candidates_main = ["🟢 Back Home", "🟠 Back Away", "⚪ Back Draw"]
    candidates_fallback = ["🟦 1X (Home/Draw)", "🟪 X2 (Away/Draw)"]

    best_rec, best_prob, best_ev, best_n = None, None, None, None

    # 1) Checa vitórias puras
    for rec in candidates_main:
        row_copy = row.copy()
        row_copy["Auto_Recommendation"] = rec
        n, p = win_prob_for_recommendation(history, row_copy,
                                           m_diff_margin, power_margin)
        if p is None or n < min_games:
            continue

        odd_ref = None
        if rec == "🟢 Back Home": odd_ref = row.get("Odd_H")
        elif rec == "🟠 Back Away": odd_ref = row.get("Odd_A")
        elif rec == "⚪ Back Draw": odd_ref = row.get("Odd_D")
        ev = (p/100.0) * odd_ref - 1 if odd_ref and odd_ref > 1.0 else None

        if (best_prob is None) or (p > best_prob):
            best_rec, best_prob, best_ev, best_n = rec, p, ev, n

    if best_prob is not None and best_prob >= 50:
        return best_rec, best_prob, best_ev, best_n

    # 2) Se não, checa 1X/X2
    for rec in candidates_fallback:
        row_copy = row.copy()
        row_copy["Auto_Recommendation"] = rec
        n, p = win_prob_for_recommendation(history, row_copy,
                                           m_diff_margin, power_margin)
        if p is None or n < min_games:
            continue

        odd_ref = None
        if rec == "🟦 1X (Home/Draw)" and row.get("Odd_H") and row.get("Odd_D"):
            odd_ref = 1 / (1/row["Odd_H"] + 1/row["Odd_D"])
        elif rec == "🟪 X2 (Away/Draw)" and row.get("Odd_A") and row.get("Odd_D"):
            odd_ref = 1 / (1/row["Odd_A"] + 1/row["Odd_D"])
        ev = (p/100.0) * odd_ref - 1 if odd_ref and odd_ref > 1.0 else None

        if (best_prob is None) or (p > best_prob):
            best_rec, best_prob, best_ev, best_n = rec, p, ev, n

    if best_prob is None or best_prob < 50:
        return "❌ Avoid", best_prob, best_ev, best_n

    return best_rec, best_prob, best_ev, best_n


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

# ==== Continuar pipeline normal ====
games_today['Dominant'] = games_today.apply(dominant_side, axis=1)

# Aplica recomendação + métricas
recs = games_today.apply(lambda r: auto_recommendation_dynamic_winrate(r, history), axis=1)
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
