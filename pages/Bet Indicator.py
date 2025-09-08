import streamlit as st
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator (XGBoost)", layout="wide")
st.title("📊 AI-Powered Bet Indicator – XGBoost")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

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

def load_last_csv(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    latest_file = max(files)
    return pd.read_csv(os.path.join(folder, latest_file))

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# ---------------- Load Data ----------------
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
if all_games.empty:
    st.warning("No valid historical data found.")
    st.stop()

# precisa de gols para treino
history = all_games.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
if history.empty:
    st.warning("No valid historical results found.")
    st.stop()

games_today = filter_leagues(load_last_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

if games_today.empty:
    st.warning("No valid games today.")
    st.stop()

# ---------------- Prepare Data ----------------
history['BetHomeWin'] = (history['Goals_H_FT'] > history['Goals_A_FT']).astype(int)
history['BetAwayWin'] = (history['Goals_A_FT'] > history['Goals_H_FT']).astype(int)

features = ['Odd_H','Odd_A','Odd_D','M_H','M_A','Diff_Power']
X = history[features]

# ---------------- Train Models ----------------
# Home model
y_home = history['BetHomeWin']
X_train, X_val, y_train, y_val = train_test_split(X, y_home, test_size=0.2, shuffle=False)
model_home = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model_home.fit(X_train, y_train)

# Away model
y_away = history['BetAwayWin']
X_train, X_val, y_train, y_val = train_test_split(X, y_away, test_size=0.2, shuffle=False)
model_away = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model_away.fit(X_train, y_train)

# ---------------- Predict Today's Games ----------------
X_today = games_today[features].copy()

games_today['p_home'] = model_home.predict_proba(X_today)[:,1]
games_today['p_away'] = model_away.predict_proba(X_today)[:,1]

games_today['EV_Home'] = (games_today['p_home'] * games_today['Odd_H']) - 1
games_today['EV_Away'] = (games_today['p_away'] * games_today['Odd_A']) - 1

def choose_bet(row):
    ev_home, ev_away = row['EV_Home'], row['EV_Away']
    if ev_home > 0 and ev_home >= ev_away:
        return f"Back Home (EV={ev_home:.1%})"
    elif ev_away > 0 and ev_away > ev_home:
        return f"Back Away (EV={ev_away:.1%})"
    else:
        return "No Bet"

games_today['Bet_Indicator'] = games_today.apply(choose_bet, axis=1)

# ---------------- Display Results ----------------
cols_to_show = [
    'Date','Time','League','Home','Away',
    'Odd_H','Odd_D','Odd_A',
    'Diff_Power','M_H','M_A','Bet_Indicator'
]

styler = (
    games_today[cols_to_show]
    .style
    .format({
        'Odd_H':'{:.2f}','Odd_D':'{:.2f}','Odd_A':'{:.2f}',
        'M_H':'{:.2f}','M_A':'{:.2f}','Diff_Power':'{:.2f}'
    }, na_rep='—')
    .applymap(lambda v: 'background-color: rgba(0,200,0,0.2)' if isinstance(v,str) and "Back" in v else '')
    .applymap(lambda v: 'background-color: rgba(255,200,200,0.3)' if isinstance(v,str) and "No Bet" in v else '')
)

st.dataframe(styler, use_container_width=True, height=1000)
