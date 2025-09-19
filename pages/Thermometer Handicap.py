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

- **Auto Recommendation** agora considera:
  - Win Probability histórica (%)
  - Expected Value (EV) com base nas odds (Home, Away, Draw) ou aproximadas (1X, X2)
- Se ambas forem boas → Entrada pré-jogo  
- Se apenas Win Probability for boa → Analisar live
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

def color_ev(val):
    if pd.isna(val): return ''
    if val > 0:
        return 'background-color: rgba(0, 200, 0, 0.14)'  # verde
    else:
        return 'background-color: rgba(255, 0, 0, 0.14)'  # vermelho


def add_band_features(df):
    """
    Adiciona colunas Home_Band, Away_Band (texto) e suas versões numéricas.
    """
    if df.empty: 
        return df

    BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}

    df['Home_Band'] = np.where(
        df['M_H'] <= df['M_H'].quantile(0.20), 'Bottom 20%',
        np.where(df['M_H'] >= df['M_H'].quantile(0.80), 'Top 20%', 'Balanced')
    )
    df['Away_Band'] = np.where(
        df['M_A'] <= df['M_A'].quantile(0.20), 'Bottom 20%',
        np.where(df['M_A'] >= df['M_A'].quantile(0.80), 'Top 20%', 'Balanced')
    )

    df["Home_Band_Num"] = df["Home_Band"].map(BAND_MAP)
    df["Away_Band_Num"] = df["Away_Band"].map(BAND_MAP)

    return df


def win_prob_for_recommendation(history, row,
                                m_diff_margin=M_DIFF_MARGIN,
                                power_margin=POWER_MARGIN):
    """
    Calcula Win Probability considerando também Home_Band_Num e Away_Band_Num.
    """
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
    """
    Seleciona a recomendação com base no maior Winrate histórico.
    1) Prioriza vitórias puras (Home, Away, Draw) se >= 50%
    2) Caso contrário, considera 1X / X2
    3) Se nada >= 50%, retorna Avoid
    """

    candidates_main = ["🟢 Back Home", "🟠 Back Away", "⚪ Back Draw"]
    candidates_fallback = ["🟦 1X (Home/Draw)", "🟪 X2 (Away/Draw)"]

    best_rec, best_prob, best_ev, best_n = None, None, None, None

    # --- 1) Checa vitórias puras ---
    for rec in candidates_main:
        row_copy = row.copy()
        row_copy["Auto_Recommendation"] = rec
        n, p = win_prob_for_recommendation(history, row_copy,
                                           m_diff_margin=m_diff_margin,
                                           power_margin=power_margin)
        if p is None or n < min_games:
            continue

        odd_ref = None
        if rec == "🟢 Back Home":
            odd_ref = row.get("Odd_H")
        elif rec == "🟠 Back Away":
            odd_ref = row.get("Odd_A")
        elif rec == "⚪ Back Draw":
            odd_ref = row.get("Odd_D")

        ev = None
        if odd_ref is not None and odd_ref > 1.0:
            ev = (p/100.0) * odd_ref - 1

        if (best_prob is None) or (p > best_prob):
            best_rec, best_prob, best_ev, best_n = rec, p, ev, n

    # Se maior Winrate >= 50% já retorna
    if best_prob is not None and best_prob >= 50:
        return best_rec, best_prob, best_ev, best_n

    # --- 2) Se não, checa 1X / X2 ---
    for rec in candidates_fallback:
        row_copy = row.copy()
        row_copy["Auto_Recommendation"] = rec
        n, p = win_prob_for_recommendation(history, row_copy,
                                           m_diff_margin=m_diff_margin,
                                           power_margin=power_margin)
        if p is None or n < min_games:
            continue

        odd_ref = None
        if rec == "🟦 1X (Home/Draw)" and row.get("Odd_H") and row.get("Odd_D"):
            odd_ref = 1 / (1/row["Odd_H"] + 1/row["Odd_D"])
        elif rec == "🟪 X2 (Away/Draw)" and row.get("Odd_A") and row.get("Odd_D"):
            odd_ref = 1 / (1/row["Odd_A"] + 1/row["Odd_D"])

        ev = None
        if odd_ref is not None and odd_ref > 1.0:
            ev = (p/100.0) * odd_ref - 1

        if (best_prob is None) or (p > best_prob):
            best_rec, best_prob, best_ev, best_n = rec, p, ev, n

    # --- 3) Se nada >= 50%, evita ---
    if best_prob is None or best_prob < 50:
        return "❌ Avoid", best_prob, best_ev, best_n

    return best_rec, best_prob, best_ev, best_n


# Bands via quantis da liga (já calculados)
games_today['Home_Band'] = np.where(
    games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
    np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
)
games_today['Away_Band'] = np.where(
    games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
    np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
)

# Bandas numéricas
BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}
games_today["Home_Band_Num"] = games_today["Home_Band"].map(BAND_MAP)
games_today["Away_Band_Num"] = games_today["Away_Band"].map(BAND_MAP)

# Também adiciona no histórico
history = add_band_features(history)


########################################
######## Bloco 7 – Exibição ############
########################################
cols_to_show = [
    'Date','Time','League','League_Classification',
    'Home','Away','Odd_H','Odd_D','Odd_A',
    'M_H','M_A','Diff_Power',
    'Home_Band','Away_Band','Dominant','Auto_Recommendation',
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
