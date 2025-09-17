import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator – Home vs Away", layout="wide")
st.title("📊 AI-Powered Bet Indicator – Home vs Away (Binary)")

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

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# ---------------- Load Historical Data ----------------
st.info("📂 Loading historical data...")
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
if all_games.empty:
    st.warning("No valid historical data found.")
    st.stop()

history = all_games.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
if history.empty:
    st.warning("No valid historical results found.")
    st.stop()

# ---------------- Matchday Selector ----------------
option = st.radio(
    "Select Matches",
    ("Today Matches", "Yesterday Matches"),
    horizontal=True
)

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No match files available.")
    st.stop()

if option == "Today Matches":
    selected_file = files[-1]  # latest file
elif option == "Yesterday Matches":
    if len(files) >= 2:
        selected_file = files[-2]  # second to last file
    else:
        st.warning("No yesterday matches available.")
        st.stop()

games_today = filter_leagues(pd.read_csv(os.path.join(GAMES_FOLDER, selected_file)))

# Keep only upcoming games (no final scores yet)
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

if games_today.empty:
    st.warning("No valid matches found for the selected day.")
    st.stop()

# ---------------- Target binary (Home=0, Away=1) ----------------
history = history[history['Goals_H_FT'] != history['Goals_A_FT']]  # remove draws
history['Target'] = history.apply(
    lambda row: 0 if row['Goals_H_FT'] > row['Goals_A_FT'] else 1,
    axis=1
)

# ---------------- Features ----------------
history['Diff_M'] = history['M_H'] - history['M_A']
games_today['Diff_M'] = games_today['M_H'] - games_today['M_A']

base_features = ['Odd_H','Odd_D','Odd_A','M_H','M_A','Diff_Power','Diff_M']

# One-hot encode League
history_leagues = pd.get_dummies(history['League'], prefix="League")
games_today_leagues = pd.get_dummies(games_today['League'], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

# Final features
X = pd.concat([history[base_features], history_leagues], axis=1)
y = history['Target']
X_today = pd.concat([games_today[base_features], games_today_leagues], axis=1)

# ---------------- Train & Evaluate ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_bin = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample"
)
model_bin.fit(X_train, y_train)

# Validation
preds = model_bin.predict(X_val)
probs = model_bin.predict_proba(X_val)

acc = accuracy_score(y_val, preds)
ll = log_loss(y_val, probs)
bs = brier_score_loss(y_val, probs[:,1])

winrate_home = (preds[y_val==0] == 0).mean()
winrate_away = (preds[y_val==1] == 1).mean()

# Show stats
st.markdown("### 📊 Model Statistics (Validation)")
df_stats = pd.DataFrame([{
    "Model": "Home vs Away (Binary)",
    "Accuracy": f"{acc:.3f}",
    "LogLoss": f"{ll:.3f}",
    "Brier": f"{bs:.3f}",
    "Winrate_Home": f"{winrate_home:.2%}",
    "Winrate_Away": f"{winrate_away:.2%}"
}])
st.dataframe(df_stats, use_container_width=True)

# ---------------- Predict Selected Games ----------------
probs_today = model_bin.predict_proba(X_today)
games_today['p_home'] = probs_today[:,0]
games_today['p_away'] = probs_today[:,1]

# ---------------- Display ----------------
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Odd_H', 'Odd_A',
    'p_home', 'p_away'
]

def color_prob(val, color):
    alpha = int((1 - val) * 255)
    return f'background-color: rgba({color}, {alpha/255:.2f})'

def style_probs(val, col):
    if col == 'p_home':
        return color_prob(val, "0,200,0")  # green
    elif col == 'p_away':
        return color_prob(val, "255,140,0")  # orange
    return ''

styled_df = (
    games_today[cols_to_show]
    .style.format({
        'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}',
        'p_home': '{:.1%}', 'p_away': '{:.1%}'
    }, na_rep='—')
    .applymap(lambda v: style_probs(v, 'p_home'), subset=['p_home'])
    .applymap(lambda v: style_probs(v, 'p_away'), subset=['p_away'])
)

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df, use_container_width=True, height=1000)
