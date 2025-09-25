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
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (Home vs Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa"]

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
st.info("📂 Loading data...")

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home", "Away","Goals_H_FT", "Goals_A_FT"], keep="first")
else:
    history = history.drop_duplicates(keep="first")

if history.empty:
    st.stop()

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
    games_today = games_today[games_today["Date"].isin([today, yesterday])].copy()

if set(["Date", "Home", "Away"]).issubset(games_today.columns):
    games_today = games_today.drop_duplicates(subset=["Home", "Away","Goals_H_FT","Goals_A_FT"], keep="first")

if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.warning("⚠️ No matches found for today (or yesterday, if selected).")
    st.stop()

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
    if pd.isna(asian_line_str):
        return np.nan
    if invert:
        margin = -margin
    try:
        parts = [float(x) for x in str(asian_line_str).split('/')]
    except:
        return np.nan
    results = []
    for line in parts:
        if margin > line:
            results.append(1.0)
        elif margin == line:
            results.append(0.5)
        else:
            results.append(0.0)
    return np.mean(results)

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Handicap_Home_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1)
history["Handicap_Away_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1)

history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)


##################### BLOCO 4 – FEATURE ENGINEERING (V3) #####################

feature_blocks = {
    # Odds principais + asiáticas + double chance
    "odds": [
        "Odd_H", "Odd_D", "Odd_A",
        "Odd_H_Asi", "Odd_A_Asi",
        "Odd_1X", "Odd_X2"
    ],

    # Força, momentum e contexto
    "strength": [
        "Diff_Power", "M_H", "M_A", "M_Diff",
        "Diff_HT_P", "M_HT_H", "M_HT_A",
        "Asian_Line_Display",
        "Win_Probability", "Games_Analyzed"
    ],

    # Categóricas enriquecidas
    "categorical": [
        "Home_Band_Num", "Away_Band_Num",
        "Dominant", "League_Classification"
    ]
}



##################### BLOCO 4B – FEATURE ENGINEERING EXTRA #####################

# --- Odds Double Chance (1X, X2) ---
def compute_double_chance_odds(df):
    df = df.copy()
    if set(["Odd_H", "Odd_D", "Odd_A"]).issubset(df.columns):
        probs = pd.DataFrame()
        probs["p_H"] = 1 / df["Odd_H"]
        probs["p_D"] = 1 / df["Odd_D"]
        probs["p_A"] = 1 / df["Odd_A"]
        probs = probs.div(probs.sum(axis=1), axis=0)
        df["Odd_1X"] = 1 / (probs["p_H"] + probs["p_D"])
        df["Odd_X2"] = 1 / (probs["p_A"] + probs["p_D"])
    return df

history = compute_double_chance_odds(history)
games_today = compute_double_chance_odds(games_today)

# --- Diferença de Momentum ---
history["M_Diff"] = history["M_H"] - history["M_A"]
games_today["M_Diff"] = games_today["M_H"] - games_today["M_A"]

# --- Classificação de ligas e bandas ---
def classify_leagues_variation(history_df):
    agg = (
        history_df.groupby("League")
        .agg(
            M_H_Min=("M_H", "min"), M_H_Max=("M_H", "max"),
            M_A_Min=("M_A", "min"), M_A_Max=("M_A", "max"),
            Hist_Games=("M_H", "count")
        ).reset_index()
    )
    agg["Variation_Total"] = (agg["M_H_Max"] - agg["M_H_Min"]) + (agg["M_A_Max"] - agg["M_A_Min"])
    def label(v):
        if v > 6.0: return "High Variation"
        if v >= 3.0: return "Medium Variation"
        return "Low Variation"
    agg["League_Classification"] = agg["Variation_Total"].apply(label)
    return agg[["League", "League_Classification", "Variation_Total", "Hist_Games"]]

def compute_league_bands(history_df):
    hist = history_df.copy()
    hist["M_Diff"] = hist["M_H"] - hist["M_A"]
    diff_q = (
        hist.groupby("League")["M_Diff"]
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2: "P20_Diff", 0.8: "P80_Diff"})
            .reset_index()
    )
    home_q = (
        hist.groupby("League")["M_H"]
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2: "Home_P20", 0.8: "Home_P80"})
            .reset_index()
    )
    away_q = (
        hist.groupby("League")["M_A"]
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2: "Away_P20", 0.8: "Away_P80"})
            .reset_index()
    )
    out = diff_q.merge(home_q, on="League", how="inner").merge(away_q, on="League", how="inner")
    return out

def dominant_side(row, threshold=0.90):
    m_h, m_a = row["M_H"], row["M_A"]
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold: return "Home strong"
    if m_h <= -threshold: return "Home weak"
    if m_a >= threshold: return "Away strong"
    if m_a <= -threshold: return "Away weak"
    return "Mixed / Neutral"

league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

# --- aplicar merges corretamente ---
for name, df in [("history", history), ("games_today", games_today)]:
    df = df.merge(league_class, on="League", how="left")
    df = df.merge(league_bands, on="League", how="left")

    df["Home_Band"] = np.where(
        df["M_H"] <= df["Home_P20"], "Bottom 20%",
        np.where(df["M_H"] >= df["Home_P80"], "Top 20%", "Balanced")
    )
    df["Away_Band"] = np.where(
        df["M_A"] <= df["Away_P20"], "Bottom 20%",
        np.where(df["M_A"] >= df["Away_P80"], "Top 20%", "Balanced")
    )
    df["Dominant"] = df.apply(dominant_side, axis=1)
    df["Home_Band_Num"] = df["Home_Band"].map({"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3})
    df["Away_Band_Num"] = df["Away_Band"].map({"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3})

    if name == "history":
        history = df
    else:
        games_today = df

# --- Placeholders caso não existam ---
if "Win_Probability" not in history.columns:
    history["Win_Probability"] = np.nan
    games_today["Win_Probability"] = np.nan
if "Games_Analyzed" not in history.columns:
    history["Games_Analyzed"] = np.nan
    games_today["Games_Analyzed"] = np.nan



##################### BLOCO 4C – BUILD FEATURE MATRIX (V3) #####################

from sklearn.preprocessing import OneHotEncoder

def build_feature_matrix(df, leagues, blocks, fit_encoder=False, encoder=None):
    dfs = []

    # --- Odds + Strength ---
    for block_name, cols in blocks.items():
        if block_name == "categorical":
            continue  # tratamos depois
        available_cols = [c for c in cols if c in df.columns]
        if available_cols:
            dfs.append(df[available_cols])

    # --- League OneHot (já gerado antes) ---
    if leagues is not None and not leagues.empty:
        dfs.append(leagues)

    # --- Outras categóricas (Dominant, League_Classification) ---
    cat_cols = [c for c in ["Dominant", "League_Classification"] if c in df.columns]
    if cat_cols:
        if fit_encoder:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        dfs.append(encoded_df)

    # --- Bands Num (já numéricas) ---
    for col in ["Home_Band_Num", "Away_Band_Num"]:
        if col in df.columns:
            dfs.append(df[[col]])

    X = pd.concat(dfs, axis=1)

    return X, encoder




##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features (odds + strength)", value=False)


##################### BLOCO 6 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2C_v1.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {"Model": f"{name}_v1", "Accuracy": accuracy_score(y, preds),
                   "LogLoss": log_loss(y, probs), "BrierScore": brier_score_loss(y, probs[:,1])}
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if ml_model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    else:
        model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              use_label_encoder=False, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {"Model": f"{name}_v1", "Accuracy": accuracy_score(y_test, preds),
           "LogLoss": log_loss(y_test, probs), "BrierScore": brier_score_loss(y_test, probs[:,1])}

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


def train_and_evaluate_v2(X, y, name, use_calibration=True):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2C_v2.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {"Model": f"{name}_v2", "Accuracy": accuracy_score(y, preds),
                   "LogLoss": log_loss(y, probs), "BrierScore": brier_score_loss(y, probs[:,1])}
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if ml_model_choice == "Random Forest":
        base_model = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced",
                                            random_state=42, n_jobs=-1)
    else:
        base_model = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                   use_label_encoder=False, random_state=42,
                                   scale_pos_weight=(sum(y == 0) / sum(y == 1)) if sum(y == 1) > 0 else 1)

    if use_calibration:
        try:
            model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=2)
        except TypeError:
            model = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=2)
        model.fit(X_train, y_train)
    else:
        if ml_model_choice == "XGBoost":
            base_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)
        else:
            base_model.fit(X_train, y_train)
        model = base_model

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {"Model": f"{name}_v2", "Accuracy": accuracy_score(y_test, preds),
           "LogLoss": log_loss(y_test, probs), "BrierScore": brier_score_loss(y_test, probs[:,1])}

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


##################### BLOCO 7 – TRAINING MODELS (V3) #####################

stats = []

# Treino Handicap Home
res, model_ah_home_v3 = train_and_evaluate(
    X_ah_home, history["Target_AH_Home"], "AH_Home_v3"
)
stats.append(res)

# Treino Handicap Away
res, model_ah_away_v3 = train_and_evaluate(
    X_ah_away, history["Target_AH_Away"], "AH_Away_v3"
)
stats.append(res)

# Versão calibrada (v3)
res, model_ah_home_v3c = train_and_evaluate_v2(
    X_ah_home, history["Target_AH_Home"], "AH_Home_v3"
)
stats.append(res)

res, model_ah_away_v3c = train_and_evaluate_v2(
    X_ah_away, history["Target_AH_Away"], "AH_Away_v3"
)
stats.append(res)

# Tabela de estatísticas
stats_df = pd.DataFrame(stats)[["Model", "Accuracy", "LogLoss", "BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) TOP – v3")
st.dataframe(stats_df, use_container_width=True)



##################### BLOCO 8 – PREDICTIONS (V3) #####################

# Escolha do modelo v3
if ml_version_choice == "v1":
    model_ah_home, cols1 = model_ah_home_v1
    model_ah_away, cols2 = model_ah_away_v1
elif ml_version_choice == "v2":
    model_ah_home, cols1 = model_ah_home_v2
    model_ah_away, cols2 = model_ah_away_v2
else:  # v3
    model_ah_home, cols1 = model_ah_home_v3
    model_ah_away, cols2 = model_ah_away_v3

# Reindex para alinhar colunas
X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

# Normalização se marcado no sidebar
if normalize_features:
    scaler = StandardScaler()
    scaler.fit(X_ah_home[numeric_cols])
    X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
    X_today_ah_away[numeric_cols] = scaler.transform(X_today_ah_away[numeric_cols])

# Geração de probabilidades
if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no", "p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]

    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for cls, col in zip(model_ah_away.classes_, ["p_ah_away_no", "p_ah_away_yes"]):
        games_today[col] = probs_away[:, cls]

# Função para colorir probabilidades
def color_prob(val, color):
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({color}, {alpha:.2f})"

# Tabela final
styled_df = (
    games_today[[
        "Date","Time","League","Home","Away",
        "Odd_H","Odd_D","Odd_A",
        "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
        "Asian_Line_Display": "{:.2f}",
        "Odd_H_Asi": "{:.2f}", "Odd_A_Asi": "{:.2f}",
        "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_ah_away_yes"])
)

st.markdown(f"### 📌 Predictions for Today's Matches – Asian Handicap ({ml_version_choice})")
st.dataframe(styled_df, use_container_width=True, height=800)
