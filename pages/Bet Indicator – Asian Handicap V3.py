##################### BLOCO 1 – IMPORTS & CONFIG #####################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap V3", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (V3 – Enriched Features)")

PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup","copas","uefa","afc","sudamericana","copa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


##################### BLOCO 2 – HELPERS #####################
def preprocess_df(df):
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, feature_cols, filename):
    with open(os.path.join(MODELS_FOLDER, filename), "wb") as f:
        joblib.dump((model, feature_cols), f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None


##################### BLOCO 3 – LOAD DATA + HANDICAP TARGET #####################
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT","Goals_A_FT","Asian_Line"]).copy()

if set(["Date","Home","Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home","Away","Goals_H_FT","Goals_A_FT"], keep="first")

today = datetime.now().strftime("%Y-%m-%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Date" in games_today.columns:
    games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
games_today = games_today[games_today["Date"] == today].copy()

include_yesterday = st.sidebar.checkbox("Include yesterday's matches", value=False)
if include_yesterday:
    games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
    if "Date" in games_today.columns:
        games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games_today = games_today[games_today["Date"].isin([today,yesterday])].copy()

if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

def convert_asian_line(line_str):
    try:
        if pd.isna(line_str) or line_str == "":
            return None
        line_str = str(line_str).strip()
        if "/" not in line_str:
            return float(line_str)
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
    except:
        return None

history["Asian_Line_Display"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Display"] = games_today["Asian_Line"].apply(convert_asian_line)

def calc_handicap_result(margin, asian_line_str, invert=False):
    if pd.isna(asian_line_str): return np.nan
    if invert: margin = -margin
    try:
        parts = [float(x) for x in str(asian_line_str).split("/")]
    except:
        return np.nan
    results = []
    for line in parts:
        if margin > line: results.append(1.0)
        elif margin == line: results.append(0.5)
        else: results.append(0.0)
    return np.mean(results)

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Handicap_Home_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1)
history["Handicap_Away_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1)

history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)


##################### BLOCO 4 – FEATURE ENGINEERING #####################
feature_blocks = {
    "odds": [
        "Odd_H","Odd_D","Odd_A",
        "Odd_H_Asi","Odd_A_Asi",
        "Odd_1X","Odd_X2"
    ],
    "strength": [
        "Diff_Power","M_H","M_A","M_Diff",
        "Diff_HT_P","M_HT_H","M_HT_A",
        "Asian_Line_Display","Win_Probability","Games_Analyzed"
    ],
    "categorical": [
        "Home_Band_Num","Away_Band_Num",
        "Dominant","League_Classification"
    ]
}


##################### BLOCO 4B – FEATURE ENGINEERING EXTRA #####################
# (use o bloco corrigido que te mandei antes, com merges certos)
# --- COLE AQUI O BLOCO 4B CORRIGIDO COMPLETO ---


##################### BLOCO 4C – BUILD FEATURE MATRIX (V3) #####################
def build_feature_matrix(df, leagues, blocks, fit_encoder=False, encoder=None):
    dfs = []

    # Odds + Strength
    for block_name, cols in blocks.items():
        if block_name == "categorical": continue
        available_cols = [c for c in cols if c in df.columns]
        if available_cols: dfs.append(df[available_cols])

    # League OneHot
    if leagues is not None and not leagues.empty:
        dfs.append(leagues)

    # Outras categóricas OneHot
    cat_cols = [c for c in ["Dominant","League_Classification"] if c in df.columns]
    if cat_cols:
        if fit_encoder:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        dfs.append(encoded_df)

    # Bands Num
    for col in ["Home_Band_Num","Away_Band_Num"]:
        if col in df.columns: dfs.append(df[[col]])

    X = pd.concat(dfs, axis=1)
    return X, encoder

# Histórico (treino)
history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X_ah_home, encoder_cat = build_feature_matrix(history, history_leagues, feature_blocks, fit_encoder=True)
X_ah_away, _ = build_feature_matrix(history, history_leagues, feature_blocks, fit_encoder=False, encoder=encoder_cat)

# Jogos de hoje (predição)
X_today_ah_home, _ = build_feature_matrix(games_today, games_today_leagues, feature_blocks, fit_encoder=False, encoder=encoder_cat)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)

X_today_ah_away, _ = build_feature_matrix(games_today, games_today_leagues, feature_blocks, fit_encoder=False, encoder=encoder_cat)
X_today_ah_away = X_today_ah_away.reindex(columns=X_ah_away.columns, fill_value=0)

numeric_cols = feature_blocks["odds"] + feature_blocks["strength"]
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]


##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model",["Random Forest","XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version",["v1","v2","v3"])
retrain = st.sidebar.checkbox("Retrain models",value=False)
normalize_features = st.sidebar.checkbox("Normalize features (odds + strength)",value=False)


##################### BLOCO 6 – TRAIN & EVALUATE #####################
# (sem mudanças, apenas garante que nomes agora recebem *_v3.pkl)


##################### BLOCO 7 – TRAINING MODELS (V3) #####################
stats = []
res, model_ah_home_v3 = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home_v3"); stats.append(res)
res, model_ah_away_v3 = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away_v3"); stats.append(res)
res, model_ah_home_v3c = train_and_evaluate_v2(X_ah_home, history["Target_AH_Home"], "AH_Home_v3"); stats.append(res)
res, model_ah_away_v3c = train_and_evaluate_v2(X_ah_away, history["Target_AH_Away"], "AH_Away_v3"); stats.append(res)

stats_df = pd.DataFrame(stats)[["Model","Accuracy","LogLoss","BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) TOP – v3")
st.dataframe(stats_df, use_container_width=True)


##################### BLOCO 8 – PREDICTIONS (V3) #####################
if ml_version_choice == "v1":
    model_ah_home, cols1 = model_ah_home_v1
    model_ah_away, cols2 = model_ah_away_v1
elif ml_version_choice == "v2":
    model_ah_home, cols1 = model_ah_home_v2
    model_ah_away, cols2 = model_ah_away_v2
else:  # v3
    model_ah_home, cols1 = model_ah_home_v3
    model_ah_away, cols2 = model_ah_away_v3

X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

if normalize_features:
    scaler = StandardScaler()
    scaler.fit(X_ah_home[numeric_cols])
    X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
    X_today_ah_away[numeric_cols] = scaler.transform(X_today_ah_away[numeric_cols])

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no","p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]
    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for cls, col in zip(model_ah_away.classes_, ["p_ah_away_no","p_ah_away_yes"]):
        games_today[col] = probs_away[:, cls]

def color_prob(val, color):
    if pd.isna(val): return ""
    alpha = float(np.clip(val,0,1))
    return f"background-color: rgba({color},{alpha:.2f})"

styled_df = (
    games_today[[
        "Date","Time","League","Home","Away",
        "Odd_H","Odd_D","Odd_A",
        "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H":"{:.2f}","Odd_D":"{:.2f}","Odd_A":"{:.2f}",
        "Asian_Line_Display":"{:.2f}",
        "Odd_H_Asi":"{:.2f}","Odd_A_Asi":"{:.2f}",
        "p_ah_home_yes":"{:.1%}","p_ah_away_yes":"{:.1%}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v,"0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v,"255,140,0"), subset=["p_ah_away_yes"])
)

st.markdown(f"### 📌 Predictions for Today's Matches – Asian Handicap ({ml_version_choice})")
st.dataframe(styled_df, use_container_width=True, height=800)
