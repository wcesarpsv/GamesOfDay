import streamlit as st
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator v2.2 (XGBoost)", layout="wide")
st.title("📊 AI-Powered Bet Indicator – XGBoost (v2.2)")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
MODELS_FOLDER = "Models"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- User Option ----------------
train_option = st.sidebar.selectbox(
    "Choose model mode:",
    ["Use saved models (fast)", "Train new models (slow)"]
)

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

# ---------------- League Stats ----------------
st.info("📊 Calculating league statistics...")
league_stats = (
    history.groupby("League")
    .agg(
        Games=("Goals_H_FT", "count"),
        DrawRate=("Goals_H_FT", lambda x: (x == history.loc[x.index, "Goals_A_FT"]).mean()),
        HomeWinRate=("Goals_H_FT", lambda x: (x > history.loc[x.index, "Goals_A_FT"]).mean()),
        AwayWinRate=("Goals_H_FT", lambda x: (x < history.loc[x.index, "Goals_A_FT"]).mean()),
    )
    .reset_index()
)

history = history.merge(league_stats[["League","DrawRate","HomeWinRate","AwayWinRate"]], on="League", how="left")
games_today = games_today.merge(league_stats[["League","DrawRate","HomeWinRate","AwayWinRate"]], on="League", how="left")

# ---------------- Targets ----------------
history['BetHomeWin'] = (history['Goals_H_FT'] > history['Goals_A_FT']).astype(int)
history['BetAwayWin'] = (history['Goals_A_FT'] > history['Goals_H_FT']).astype(int)
history['BetDrawWin'] = (history['Goals_H_FT'] == history['Goals_A_FT']).astype(int)

# ---------------- Features ----------------
base_features = ['Odd_H','Odd_A','Odd_D','M_H','M_A','Diff_Power','DrawRate','HomeWinRate','AwayWinRate']
X_base = history[base_features]

# One-hot leagues
X_leagues = pd.get_dummies(history['League'], prefix="League")
X = pd.concat([X_base, X_leagues], axis=1)

# ---------------- Train or Load Models ----------------
models = {}
feature_names = None

def train_and_save(target, filename, step):
    y = history[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    model.save_model(os.path.join(MODELS_FOLDER, filename))
    progress.progress(step)
    return model, X.columns

def load_model(filename):
    model = XGBClassifier()
    model.load_model(os.path.join(MODELS_FOLDER, filename))
    return model

if train_option == "Train new models (slow)":
    st.info("🚀 Training models... this may take a while")
    progress = st.progress(0)
    model_home, feature_names = train_and_save("BetHomeWin", "home.json", 33)
    model_away, _ = train_and_save("BetAwayWin", "away.json", 66)
    model_draw, _ = train_and_save("BetDrawWin", "draw.json", 100)
else:
    try:
        model_home = load_model("home.json")
        model_away = load_model("away.json")
        model_draw = load_model("draw.json")
        feature_names = X.columns
        st.success("✅ Loaded saved models successfully.")
    except Exception as e:
        st.warning("⚠️ Saved models not found. Training new ones...")
        progress = st.progress(0)
        model_home, feature_names = train_and_save("BetHomeWin", "home.json", 33)
        model_away, _ = train_and_save("BetAwayWin", "away.json", 66)
        model_draw, _ = train_and_save("BetDrawWin", "draw.json", 100)

# ---------------- Predict Today's Games ----------------
st.info("🔮 Generating predictions for today's matches...")
with st.spinner("Calculating probabilities and EV..."):
    X_today_base = games_today[base_features]
    X_today_leagues = pd.get_dummies(games_today['League'], prefix="League")
    X_today = pd.concat([X_today_base, X_today_leagues], axis=1)

    # Align columns with training
    for col in feature_names:
        if col not in X_today.columns:
            X_today[col] = 0
    X_today = X_today[feature_names]

    games_today['p_home'] = model_home.predict_proba(X_today)[:,1]
    games_today['p_away'] = model_away.predict_proba(X_today)[:,1]
    games_today['p_draw'] = model_draw.predict_proba(X_today)[:,1]

    games_today['EV_Home'] = (games_today['p_home'] * games_today['Odd_H']) - 1
    games_today['EV_Away'] = (games_today['p_away'] * games_today['Odd_A']) - 1
    games_today['EV_Draw'] = (games_today['p_draw'] * games_today['Odd_D']) - 1

# ---------------- Bet Decision ----------------
def choose_bet(row):
    evs = {
        "Home": row['EV_Home'],
        "Away": row['EV_Away'],
        "Draw": row['EV_Draw']
    }
    best = max(evs, key=evs.get)
    if evs[best] > 0:
        return f"Back {best} (EV={evs[best]:.1%})"
    else:
        return "No Bet"

games_today['Bet_Indicator'] = games_today.apply(choose_bet, axis=1)

# ---------------- Display Results ----------------
cols_to_show = [
    'Date','Time','League','Home','Away',
    'Odd_H','Odd_D','Odd_A',
    'Diff_Power','M_H','M_A',
    'p_home','p_draw','p_away',
    'EV_Home','EV_Draw','EV_Away','Bet_Indicator'
]

def color_bet(val):
    if pd.isna(val): return ''
    if "Home" in str(val): return 'background-color: rgba(0,128,255,0.2)'  # Blue
    if "Away" in str(val): return 'background-color: rgba(128,0,255,0.2)'  # Purple
    if "Draw" in str(val): return 'background-color: rgba(255,215,0,0.3)'  # Yellow
    if "No Bet" in str(val): return 'background-color: rgba(200,200,200,0.2)'  # Gray
    return ''

styler = (
    games_today[cols_to_show]
    .style
    .format({
        'Odd_H':'{:.2f}','Odd_D':'{:.2f}','Odd_A':'{:.2f}',
        'M_H':'{:.2f}','M_A':'{:.2f}','Diff_Power':'{:.2f}',
        'p_home':'{:.1%}','p_draw':'{:.1%}','p_away':'{:.1%}',
        'EV_Home':'{:.1%}','EV_Draw':'{:.1%}','EV_Away':'{:.1%}'
    }, na_rep='—')
    .applymap(color_bet, subset=['Bet_Indicator'])
)

st.dataframe(styler, use_container_width=True, height=1000)

# ---------------- Save Results ----------------
output_folder = os.path.join("GamesDay", "BetIndicator")
os.makedirs(output_folder, exist_ok=True)

today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
output_file = os.path.join(output_folder, f"Recommendations_{today_str}.csv")
games_today[cols_to_show].to_csv(output_file, index=False)

st.success(f"✅ Recommendations saved at: {output_file}")