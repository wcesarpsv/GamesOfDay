import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator v3.2 (RF + League + Diff_M)", layout="wide")
st.title("📊 AI-Powered Bet Indicator – Random Forest")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
MODELS_FOLDER = "Models"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa","afc"]

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
st.info("📂 Loading historical data...")
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
if all_games.empty:
    st.warning("No valid historical data found.")
    st.stop()

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

# ---------------- Train & Evaluate ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model_multi = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample"
)
model_multi.fit(X_train, y_train)

preds = model_multi.predict(X_val)
probs_val = model_multi.predict_proba(X_val)

# Metrics
acc = accuracy_score(y_val, preds)
ll = log_loss(y_val, probs_val)

# Brier multiclasse
y_onehot = pd.get_dummies(y_val).values
brier_multi = np.mean(np.sum((probs_val - y_onehot) ** 2, axis=1))

# Winrates por classe
winrate_home = (preds[y_val==0] == 0).mean()
winrate_draw = (preds[y_val==1] == 1).mean()
winrate_away = (preds[y_val==2] == 2).mean()

df_stats = pd.DataFrame([{
    "Model": "1X2",
    "Accuracy": f"{acc:.3f}",
    "LogLoss": f"{ll:.3f}",
    "Brier": f"{brier_multi:.3f} (multi)",
    "Winrate_Home": f"{winrate_home:.2%}",
    "Winrate_Draw": f"{winrate_draw:.2%}",
    "Winrate_Away": f"{winrate_away:.2%}"
}])

st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)

# ---------------- Predict Today's Games ----------------
probs = model_multi.predict_proba(X_today)
games_today['p_home'] = probs[:,0]
games_today['p_draw'] = probs[:,1]
games_today['p_away'] = probs[:,2]

# ---------------- Display ----------------
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Odd_H', 'Odd_D', 'Odd_A',
    'p_home', 'p_draw', 'p_away'
]

# Funções de gradiente
def color_prob(val, color):
    alpha = int((1 - val) * 255)
    return f'background-color: rgba({color}, {alpha/255:.2f})'

def style_probs(val, col):
    if col == 'p_home':
        return color_prob(val, "0,200,0")  # verde
    elif col == 'p_draw':
        return color_prob(val, "150,150,150")  # cinza
    elif col == 'p_away':
        return color_prob(val, "255,140,0")  # laranja
    return ''

styled_df = (
    games_today[cols_to_show]
    .style.format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'p_home': '{:.1%}', 'p_draw': '{:.1%}', 'p_away': '{:.1%}'
    }, na_rep='—')
    .applymap(lambda v: style_probs(v, 'p_home'), subset=['p_home'])
    .applymap(lambda v: style_probs(v, 'p_draw'), subset=['p_draw'])
    .applymap(lambda v: style_probs(v, 'p_away'), subset=['p_away'])
)

st.markdown("### 📌 Predictions for Today's Games")
st.dataframe(styled_df, use_container_width=True, height=1000)
