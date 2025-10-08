########################################
########## BLOCO 1 – IMPORTS ###########
########################################
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bet Indicator – Asian Handicap (AIL v1)", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (Home vs Away) + AIL v1")

# ---------------- Configurações ----------------
PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


########################################
###### BLOCO 2 – HELPERS BÁSICOS #######
########################################
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

@st.cache_data(show_spinner=False)
def load_all_games(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
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


########################################
### BLOCO AIL – CORE FUNCTIONS (Option A)
########################################
MMD_ABS_DIFF_POWER = 2.0
MMD_ABS_HC_BALANCE = 0.05
LEAGUE_MEI_MIN_N = 200
WINSOR_P_LO, WINSOR_P_HI = 0.01, 0.99

def _to_float_safe(x):
    if pd.isna(x): return None
    try:
        if isinstance(x, str): x = x.strip().replace(",", ".")
        return float(x)
    except:
        return None

def invert_asian_line_str(line_str):
    val = _to_float_safe(line_str)
    if val is None: return None
    inv = -val
    out = f"{inv:+g}"
    return out.replace("+", "") if out.startswith("+") else out

def make_display_lines(df):
    df = df.copy()
    df["Asian_Line_Away_Display"] = df["Asian_Line"]
    df["Asian_Line_Home_Display"] = df["Asian_Line"].apply(invert_asian_line_str)
    return df

def deduplicate_matches(df):
    df = df.copy()
    if "Id" in df.columns:
        df = df.sort_values(by=["Date", "Home", "Away"]).drop_duplicates("Id", keep="last")
    else:
        df = df.sort_values(by=["Date", "Home", "Away"]).drop_duplicates(["Date", "Home", "Away"], keep="last")
    return df

def _ah_outcome(gf, ga, line):
    diff = gf - ga
    margin = diff - line
    if margin > 0.5: return "Win"
    if margin < -0.5: return "Lose"
    return "Push"

def compute_targets_home_away(df):
    df = df.copy()
    ah_away = df["Asian_Line"].apply(_to_float_safe)
    ah_home = ah_away.apply(lambda v: -v if v is not None else None)
    gH = df["Goals_H_FT"].fillna(0).astype(int)
    gA = df["Goals_A_FT"].fillna(0).astype(int)
    df["AH_Target_Home"] = [_ah_outcome(h, a, l) if l is not None else None for h, a, l in zip(gH, gA, ah_home)]
    df["AH_Target_Away"] = [_ah_outcome(a, h, l) if l is not None else None for h, a, l in zip(gH, gA, ah_away)]
    return df

def ensure_training_interactions(df):
    df = df.copy()
    cols = [
        "Market_Error_Home","Market_Error_Away","Market_Error_Diff",
        "Underdog_Value_Home","Underdog_Value_Away","Underdog_Value_Diff",
        "Favorite_Crash_Home","Favorite_Crash_Away","Favorite_Crash_Diff"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


########################################
##### BLOCO 3 – LOAD + TARGETS AH ######
########################################
st.info("📂 Loading data...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

for col in ["Goals_H_Today", "Goals_A_Today"]:
    if col not in games_today.columns:
        games_today[col] = np.nan

history = filter_leagues(load_all_games(GAMES_FOLDER))
if history.empty:
    st.stop()

history = deduplicate_matches(history)
history = make_display_lines(history)
history = compute_targets_home_away(history)
history = history.dropna(subset=["Goals_H_FT","Goals_A_FT","Asian_Line"]).copy()

if {"Date","Home","Away"}.issubset(history.columns):
    history = history.drop_duplicates(subset=["Date","Home","Away"], keep="last")

if games_today.empty or len(games_today) == 0:
    st.warning("⚠️ No valid matches for today.")
    st.stop()


########################################
#### BLOCO 4.5 – AIL-ML INTERACTIONS ###
########################################
history = ensure_training_interactions(history)
games_today = ensure_training_interactions(games_today)


########################################
##### BLOCO 5 – FEATURE BLOCKS #########
########################################
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": ["Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","M_HT_H","M_HT_A","Asian_Line_Home_Display"],
    "aggression": [
        "Market_Error_Home","Market_Error_Away","Market_Error_Diff",
        "Underdog_Value_Home","Underdog_Value_Away","Underdog_Value_Diff",
        "Favorite_Crash_Home","Favorite_Crash_Away","Favorite_Crash_Diff"
    ],
    "categorical": []
}

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League").reindex(columns=history_leagues.columns, fill_value=0)
feature_blocks["categorical"] = list(history_leagues.columns)

def build_feature_matrix(df, leagues, blocks):
    dfs = []
    for block_name, cols in blocks.items():
        if block_name == "categorical":
            dfs.append(leagues)
        elif cols:
            avail = [c for c in cols if c in df.columns]
            if avail:
                dfs.append(df[avail])
    return pd.concat(dfs, axis=1)

X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()
X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()
numeric_cols = [c for c in X_ah_home.columns if np.issubdtype(X_ah_home[c].dtype, np.number)]


########################################
###### BLOCO 6 – SIDEBAR & ML ##########
########################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features", value=False)

def brier_multiclass(y_true, probs, classes):
    y_true_bin = pd.get_dummies(y_true, columns=classes).reindex(columns=classes, fill_value=0)
    return np.mean(np.sum((probs - y_true_bin.values) ** 2, axis=1))


def train_and_evaluate(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v4.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {"Model": f"{name}_v1", "Accuracy": accuracy_score(y, preds),
                   "LogLoss": log_loss(y, probs), "BrierScore": brier_multiclass(y, probs, model.classes_)}
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_test = X_train.copy(), X_test.copy()

    if normalize_features and numeric_cols:
        train_med = X_train[numeric_cols].median()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(train_med)
        X_test[numeric_cols]  = X_test[numeric_cols].fillna(train_med)
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42) if ml_model_choice=="Random Forest" else \
             XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {"Model": f"{name}_v1", "Accuracy": accuracy_score(y_test, preds),
           "LogLoss": log_loss(y_test, probs), "BrierScore": brier_multiclass(y_test, probs, model.classes_)}
    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


########################################
######## BLOCO 7 – TREINAMENTO #########
########################################
stats = []
res, model_ah_home_v1 = train_and_evaluate(X_ah_home, history["AH_Target_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v1 = train_and_evaluate(X_ah_away, history["AH_Target_Away"], "AH_Away"); stats.append(res)
stats_df = pd.DataFrame(stats)[["Model", "Accuracy", "LogLoss", "BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) – AIL Option A")
st.dataframe(stats_df, use_container_width=True)


########################################
######## BLOCO 8 – PREDICTIONS #########
########################################
model_ah_home, cols1 = model_ah_home_v1
model_ah_away, cols2 = model_ah_away_v1
X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for i, cls in enumerate(model_ah_home.classes_):
        games_today[f"p_ah_home_{cls.lower()}"] = probs_home[:, i]
    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for i, cls in enumerate(model_ah_away.classes_):
        games_today[f"p_ah_away_{cls.lower()}"] = probs_away[:, i]

def color_prob(val, rgb):
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({rgb}, {alpha:.2f})"

st.markdown(f"### 📌 Predictions for {selected_date_str}")

cols_show = [
    "Date","Time","League","Home","Away","Goals_H_Today","Goals_A_Today",
    "Odd_H","Odd_D","Odd_A","Asian_Line_Home_Display","Odd_H_Asi","Odd_A_Asi"
] + [c for c in games_today.columns if c.startswith("p_ah_home_") or c.startswith("p_ah_away_")]
cols_show = [c for c in cols_show if c in games_today.columns]

pred_df = games_today[cols_show].copy()
fmt_map = {c: "{:.2f}" for c in ["Odd_H","Odd_D","Odd_A","Odd_H_Asi","Odd_A_Asi"] if c in pred_df.columns}
for c in pred_df.columns:
    if c.startswith("p_ah_"): fmt_map[c] = "{:.1%}"
styled_df = (
    pred_df
    .style.format(fmt_map, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=[c for c in pred_df.columns if "p_ah_home" in c])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=[c for c in pred_df.columns if "p_ah_away" in c])
)
st.dataframe(styled_df, use_container_width=True, height=800)
