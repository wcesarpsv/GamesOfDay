##################### BLOCO 1 – IMPORTS & CONFIG #####################
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime

st.set_page_config(page_title="Bet Indicator – Asian Handicap v2.0", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap v2.0 (AIL + Market + Value + Momentum)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "AsianHandicap_v2"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

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

files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
if date_match:
    selected_date_str = date_match.group(0)
else:
    selected_date_str = datetime.now().strftime("%Y-%m-%d")

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

if os.path.exists(livescore_file):
    results_df = pd.read_csv(livescore_file)
    results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
    games_today = games_today.merge(results_df, left_on='Id', right_on='game_id', how='left', suffixes=('', '_RAW'))
    games_today['Goals_H_Today'] = games_today['home_goal']
    games_today['Goals_A_Today'] = games_today['away_goal']
else:
    games_today['Goals_H_Today'] = np.nan
    games_today['Goals_A_Today'] = np.nan

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

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
history["Target_AH_Home"] = (history["Handicap_Home_Result"] > 0.5).astype(int)
history["Target_AH_Away"] = (history["Handicap_Away_Result"] > 0.5).astype(int)


##################### BLOCO 4 – FEATURE ENGINEERING (AGGRESSION) #####################
def add_aggression_features(df):
    aggression_features = []
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away']):
        df['Handicap_Balance'] = df['Aggression_Home'] - df['Aggression_Away']
        df['Underdog_Indicator'] = -df['Handicap_Balance']
        if 'M_H' in df.columns and 'M_A' in df.columns:
            df['Power_vs_Perception_Home'] = df['M_H'] - df['Aggression_Home']
            df['Power_vs_Perception_Away'] = df['M_A'] - df['Aggression_Away']
            df['Power_Perception_Diff'] = df['Power_vs_Perception_Home'] - df['Power_vs_Perception_Away']
        aggression_features.extend(['Aggression_Home', 'Aggression_Away', 'Handicap_Balance',
                                    'Underdog_Indicator', 'Power_Perception_Diff'])
    return df, aggression_features

history, aggression_features = add_aggression_features(history)
games_today, _ = add_aggression_features(games_today)

feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": ["Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A", "Asian_Line_Display"],
    "aggression": aggression_features,
    "categorical": []
}


##################### BLOCO 4.5 – AIL EXTRA FEATURES #####################
def add_market_error_features(df):
    features = []
    if all(col in df.columns for col in ['Diff_Power', 'Asian_Line_Display']):
        df['Market_Error_Home'] = df['Diff_Power'] - df['Asian_Line_Display']
        df['Market_Error_Away'] = -df['Market_Error_Home']
        df['Market_Error_Diff'] = df['Market_Error_Home'].abs()
        features = ['Market_Error_Home', 'Market_Error_Away', 'Market_Error_Diff']
    return df, features

def add_value_score(df):
    features = []
    if all(col in df.columns for col in ['Odd_H', 'Odd_A']):
        df['Implied_H'] = 1 / df['Odd_H']
        df['Implied_A'] = 1 / df['Odd_A']
        if all(col in df.columns for col in ['p_ah_home_yes', 'p_ah_away_yes']):
            df['ValueScore_Home'] = df['p_ah_home_yes'] - df['Implied_H']
            df['ValueScore_Away'] = df['p_ah_away_yes'] - df['Implied_A']
            features = ['ValueScore_Home', 'ValueScore_Away']
    return df, features

def add_aggression_momentum(df):
    features = []
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'Diff_HT_P']):
        df['Aggression_Momentum_Score_Home'] = (-df['Aggression_Home']) * df['Diff_HT_P']
        df['Aggression_Momentum_Score_Away'] = (-df['Aggression_Away']) * (-df['Diff_HT_P'])
        features = ['Aggression_Momentum_Score_Home', 'Aggression_Momentum_Score_Away']
    return df, features

history, me_features = add_market_error_features(history)
games_today, _ = add_market_error_features(games_today)

history, val_features = add_value_score(history)
games_today, _ = add_value_score(games_today)

history, am_features = add_aggression_momentum(history)
games_today, _ = add_aggression_momentum(games_today)

feature_blocks.update({
    "market": me_features,
    "value": val_features,
    "momentum": am_features
})

numeric_cols = (
    feature_blocks["odds"]
    + feature_blocks["strength"]
    + feature_blocks["aggression"]
    + feature_blocks["market"]
    + feature_blocks["value"]
    + feature_blocks["momentum"]
)
numeric_cols = [c for c in numeric_cols if c in history.columns]
st.success("✅ AIL Extra Features added: Market_Error, Value_Score, and Aggression_Momentum")


##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize numeric features", value=True)


##################### BLOCO 6 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_v5.pkl"
    feature_cols = X.columns.tolist()

    # ------------------------------------------
    # 1️⃣ Verifica se modelo já salvo deve ser carregado
    # ------------------------------------------
    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {
                "Model": f"{name}_v2",
                "Accuracy": accuracy_score(y, preds),
                "LogLoss": log_loss(y, probs),
                "BrierScore": brier_score_loss(y, probs[:, 1])
            }
            return res, (model, cols)

    # ------------------------------------------
    # 2️⃣ Split de treino e teste
    # ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ------------------------------------------
    # 3️⃣ Normalização opcional
    # ------------------------------------------
    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ------------------------------------------
    # 4️⃣ Seleção do modelo base
    # ------------------------------------------
    if ml_model_choice == "Random Forest":
        base_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        base_model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )

    # ------------------------------------------
    # 5️⃣ Calibração (compatível com qualquer versão sklearn)
    # ------------------------------------------
    try:
        model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=2)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=2)

    # ------------------------------------------
    # 6️⃣ Treino e avaliação
    # ------------------------------------------
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {
        "Model": f"{name}_v2",
        "Accuracy": accuracy_score(y_test, preds),
        "LogLoss": log_loss(y_test, probs),
        "BrierScore": brier_score_loss(y_test, probs[:, 1])
    }

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


##################### BLOCO 7 – TRAINING & COMPARISON #####################
st.info("🚀 Training and Comparing AIL v1 vs AIL v2.0 models...")

# ------------------------------------------
# 1️⃣ TRAIN V1 (BASELINE)
# ------------------------------------------
def train_v1_baseline(X, y, name):
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return {
        "Model": f"{name}_v1",
        "Accuracy": accuracy_score(y_test, preds),
        "LogLoss": log_loss(y_test, probs),
        "BrierScore": brier_score_loss(y_test, probs[:, 1])
    }, model

# ------------------------------------------
# 2️⃣ TRAIN V2 (AIL+)
# ------------------------------------------
stats = []
X = history[numeric_cols].dropna()
y_home = history["Target_AH_Home"].loc[X.index]
y_away = history["Target_AH_Away"].loc[X.index]

# AIL v1 baseline
res1_home, model1_home = train_v1_baseline(X, y_home, "AH_Home")
res1_away, model1_away = train_v1_baseline(X, y_away, "AH_Away")

# AIL v2.0 (atual)
res2_home, model2_home = train_and_evaluate(X, y_home, "AH_Home")
res2_away, model2_away = train_and_evaluate(X, y_away, "AH_Away")

stats.extend([res1_home, res1_away, res2_home, res2_away])
stats_df = pd.DataFrame(stats)[["Model", "Accuracy", "LogLoss", "BrierScore"]]

# ------------------------------------------
# 3️⃣ COMPARATIVO (Δ entre versões)
# ------------------------------------------
def compare_metrics(v1, v2):
    df = pd.DataFrame({
        "Metric": ["Accuracy", "LogLoss", "BrierScore"],
        "v1": [v1["Accuracy"], v1["LogLoss"], v1["BrierScore"]],
        "v2": [v2["Accuracy"], v2["LogLoss"], v2["BrierScore"]]
    })
    df["Δ (v2-v1)"] = df["v2"] - df["v1"]
    df["Trend"] = df["Δ (v2-v1)"].apply(lambda x: "🟢↑" if x > 0 else "🔻↓" if x < 0 else "⚪")
    return df

compare_home = compare_metrics(res1_home, res2_home)
compare_away = compare_metrics(res1_away, res2_away)

st.markdown("### 📊 Model Statistics – AIL v1 vs AIL v2.0")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🏠 AH_Home Comparison")
    st.dataframe(compare_home.style.format({
        "v1": "{:.3f}", "v2": "{:.3f}", "Δ (v2-v1)": "{:+.3f}"
    }), use_container_width=True)
with col2:
    st.subheader("🚌 AH_Away Comparison")
    st.dataframe(compare_away.style.format({
        "v1": "{:.3f}", "v2": "{:.3f}", "Δ (v2-v1)": "{:+.3f}"
    }), use_container_width=True)

# ------------------------------------------
# 4️⃣ ESCOLHER MODELOS (usar v2 por padrão)
# ------------------------------------------
model_ah_home, model_ah_away = model2_home, model2_away


##################### BLOCO 8 – PREDICTIONS #####################
st.markdown("### 🎯 Predictions – Today's Matches")

X_today = games_today[numeric_cols].copy()
X_today = X_today.reindex(columns=X.columns, fill_value=0)

if normalize_features:
    scaler = StandardScaler()
    scaler.fit(X[numeric_cols])
    X_today[numeric_cols] = scaler.transform(X_today[numeric_cols])

# ✅ Desempacotar corretamente o modelo e colunas
model_ah_home, cols_home = model2_home
model_ah_away, cols_away = model2_away

# Garantir compatibilidade (caso algum modelo não tenha predict_proba)
if not hasattr(model_ah_home, "predict_proba"):
    model_ah_home.predict_proba = lambda X: np.stack(
        [1 - model_ah_home.predict(X), model_ah_home.predict(X)], axis=1
    )
if not hasattr(model_ah_away, "predict_proba"):
    model_ah_away.predict_proba = lambda X: np.stack(
        [1 - model_ah_away.predict(X), model_ah_away.predict(X)], axis=1
    )

# ------------------------------------------
# 5️⃣ Fazer as previsões com degradê
# ------------------------------------------
if not games_today.empty:
    games_today["p_ah_home_yes"] = model_ah_home.predict_proba(X_today)[:, 1]
    games_today["p_ah_away_yes"] = model_ah_away.predict_proba(X_today)[:, 1]

    def color_prob(val, color_rgb):
        if pd.isna(val):
            return ""
        alpha = float(np.clip(val, 0, 1))
        return f"background-color: rgba({color_rgb}, {alpha:.2f}); color: white; font-weight: bold;"

    styled_df = (
        games_today[[
            "Date", "League", "Home", "Away",
            "Odd_H", "Odd_D", "Odd_A",
            "Asian_Line_Display",
            "p_ah_home_yes", "p_ah_away_yes"
        ]]
        .style.format({
            "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
            "Asian_Line_Display": "{:.2f}",
            "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}"
        })
        .applymap(lambda v: color_prob(v, "0,180,0"), subset=["p_ah_home_yes"])   # Verde → Home
        .applymap(lambda v: color_prob(v, "255,210,0"), subset=["p_ah_away_yes"]) # Amarelo → Away
    )

    st.dataframe(styled_df, width="stretch", height=800)
else:
    st.warning("⚠️ Nenhum jogo encontrado para previsão.")


##################### BLOCO 9 – PERFORMANCE DASHBOARD #####################
st.markdown("### 📊 AIL v2.0 – Daily Performance Overview")

if not games_today.empty and "p_ah_home_yes" in games_today.columns:

    avg_home = games_today["p_ah_home_yes"].mean()
    avg_away = games_today["p_ah_away_yes"].mean()

    strong_home = (games_today["p_ah_home_yes"] > 0.60).sum()
    strong_away = (games_today["p_ah_away_yes"] > 0.60).sum()

    gap_mean = (games_today["p_ah_home_yes"] - games_today["p_ah_away_yes"]).abs().mean()

    # ROI proxy simples (média das diferenças entre probabilidade prevista e implícita)
    if all(col in games_today.columns for col in ["Odd_H", "Odd_A"]):
        games_today["Implied_H"] = 1 / games_today["Odd_H"]
        games_today["Implied_A"] = 1 / games_today["Odd_A"]
        roi_home = (games_today["p_ah_home_yes"] - games_today["Implied_H"]).mean()
        roi_away = (games_today["p_ah_away_yes"] - games_today["Implied_A"]).mean()
    else:
        roi_home = roi_away = np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏠 Avg Prob (Home)", f"{avg_home:.1%}")
    col2.metric("🚌 Avg Prob (Away)", f"{avg_away:.1%}")
    col3.metric("🔥 Strong Signals", f"{strong_home + strong_away}")
    col4.metric("⚖️ Balance Gap", f"{gap_mean:.1%}")

    col5, col6 = st.columns(2)
    col5.metric("💵 ROI Proxy Home", f"{roi_home:+.2%}")
    col6.metric("💵 ROI Proxy Away", f"{roi_away:+.2%}")

else:
    st.warning("⚠️ No predictions available to summarize.")

