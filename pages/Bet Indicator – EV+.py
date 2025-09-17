import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator v3.2 (RF + League + Diff_M)", layout="wide")
st.title("📊 AI-Powered Bet Indicator – Random Forest + Kelly")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
MODELS_FOLDER = "Models"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa","afc","sudamericana","copa"]

os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- Helpers ----------------
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

def load_last_or_penultimate_csv(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()

    files = sorted(files)  # ordena pela data no nome (ou ordem alfabética)
    
    # Selecionar último ou penúltimo via Streamlit
    option = st.radio(
        "📂 Escolha o arquivo para carregar:",
        options=["Today Matches", "Yesterday Matches"],
        index=0,  # padrão = último
        horizontal=True
    )

    if option == "Today Matches":
        selected_file = files[-1]
    else:
        if len(files) >= 2:
            selected_file = files[-2]
        else:
            st.warning("⚠️ Só existe um arquivo disponível, carregando o último.")
            selected_file = files[-1]

    return pd.read_csv(os.path.join(folder, selected_file))

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# ---------------- Load Data ----------------
st.info("📂 Loading historical data...")
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
if all_games.empty:
    st.warning("No valid historical data found.")
    st.stop()

history = all_games.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
if history.empty:
    st.warning("No valid historical results found.")
    st.stop()

games_today = filter_leagues(load_last_or_penultimate_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

if games_today.empty:
    st.warning("No valid games today.")
    st.stop()

# ---------------- Target multiclasses ----------------
history['Target'] = history.apply(
    lambda row: 0 if row['Goals_H_FT'] > row['Goals_A_FT']
    else (1 if row['Goals_H_FT'] == row['Goals_A_FT'] else 2),
    axis=1
)

# ---------------- Features ----------------
history['Diff_M'] = history['M_H'] - history['M_A']
games_today['Diff_M'] = games_today['M_H'] - games_today['M_A']

base_features = ['Odd_H','Odd_A','Odd_D','M_H','M_A','Diff_Power','Diff_M']

# One-hot encode League
history_leagues = pd.get_dummies(history['League'], prefix="League")
games_today_leagues = pd.get_dummies(games_today['League'], prefix="League")

# Garantir que os dummies tenham as mesmas colunas
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

# Montar features finais
X = pd.concat([history[base_features], history_leagues], axis=1)
y = history['Target']

X_today = pd.concat([games_today[base_features], games_today_leagues], axis=1)

# ---------------- Train model ----------------
model_multi = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample"
)
model_multi.fit(X, y)

# ---------------- Predict Today's Games ----------------
probs = model_multi.predict_proba(X_today)

games_today['p_home'] = probs[:,0]
games_today['p_draw'] = probs[:,1]
games_today['p_away'] = probs[:,2]

# ---------------- Kelly Criterion Helpers ----------------
def odd_min(prob):
    return 1 / prob if prob > 0 else np.inf

def kelly_fraction_pct(prob, odds, scale=0.1):
    b = odds - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    return max(0, f_star * 100 * scale)  # % da banca

def calc_kelly_row(row, margin_pre=0.1, margin_live=0.2, scale=0.1):
    # Probabilidades e odds
    probs = {"Home": row['p_home'], "Draw": row['p_draw'], "Away": row['p_away']}
    odds = {"Home": row['Odd_H'], "Draw": row['Odd_D'], "Away": row['Odd_A']}
    
    # Mercado com maior probabilidade
    best_market = max(probs, key=probs.get)
    best_prob = probs[best_market]
    best_odds = odds[best_market]
    
    # Odd mínima base
    min_odd_base = odd_min(best_prob)
    min_odd_pre = min_odd_base * (1 + margin_pre)
    min_odd_live = min_odd_base * (1 + margin_live)
    
    # Stake pré ou live (% da banca)
    if best_odds >= min_odd_pre:
        stake_pre = kelly_fraction_pct(best_prob, best_odds, scale=scale)
        stake_live = 0
    else:
        stake_pre = 0
        stake_live = kelly_fraction_pct(best_prob, min_odd_live, scale=scale)
    
    return pd.Series({
        "Best_Market": best_market,
        "Best_Prob": best_prob,
        "Best_Odds": best_odds,
        "Odd_Min_Base": min_odd_base,
        "Odd_Min_Pre": min_odd_pre,
        "Odd_Min_Live": min_odd_live,
        "Stake_Pre(%)": stake_pre,
        "Stake_Live(%)": stake_live
    })

# ---------------- Apply Kelly to today's games ----------------
games_today = games_today.join(games_today.apply(calc_kelly_row, axis=1))

# ---------------- Funções de gradiente ----------------
def color_prob(val, color):
    alpha = int(val * 255)  # intensidade proporcional à probabilidade
    return f'background-color: rgba({color}, {alpha/255:.2f})'

def style_probs(val, col):
    if col == 'p_home':
        return color_prob(val, "0,200,0")  # verde
    elif col == 'p_draw':
        return color_prob(val, "150,150,150")  # cinza
    elif col == 'p_away':
        return color_prob(val, "255,140,0")  # laranja
    return ''

def highlight_stakes(val, col):
    if col == 'Stake_Pre(%)' and val > 0:
        return 'background-color: rgba(0,200,0,0.3); font-weight: bold;'
    if col == 'Stake_Live(%)' and val > 0:
        return 'background-color: rgba(255,140,0,0.3); font-weight: bold;'
    return ''

# ---------------- Display ----------------
cols_final = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Odd_H', 'Odd_D', 'Odd_A',
    'p_home', 'p_draw', 'p_away',
    'Best_Market',
    'Odd_Min_Base', "Odd_Min_Live",
    'Stake_Pre(%)', 'Stake_Live(%)'
]

styled_df = (
    games_today[cols_final]
    .style.format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'p_home': '{:.1%}', 'p_draw': '{:.1%}', 'p_away': '{:.1%}',
        'Odd_Min_Base': '{:.2f}', 'Odd_Min_Live': '{:.2f}',
        'Stake_Pre(%)': '{:.2f}%', 'Stake_Live(%)': '{:.2f}%'
    }, na_rep='—')
    .applymap(lambda v: style_probs(v, 'p_home'), subset=['p_home'])
    .applymap(lambda v: style_probs(v, 'p_draw'), subset=['p_draw'])
    .applymap(lambda v: style_probs(v, 'p_away'), subset=['p_away'])
    .applymap(lambda v: highlight_stakes(v, 'Stake_Pre(%)'), subset=['Stake_Pre(%)'])
    .applymap(lambda v: highlight_stakes(v, 'Stake_Live(%)'), subset=['Stake_Live(%)'])
    .set_properties(subset=['Best_Market'], **{'text-align': 'center'})  # centraliza
)

st.dataframe(styled_df, use_container_width=True, height=1000)
