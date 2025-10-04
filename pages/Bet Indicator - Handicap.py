##################### BLOCO 1 – IMPORTS & CONFIG #####################
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
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (Home vs Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa","trophy"]

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

# ========== NOVO: SELEÇÃO DE DATA IGUAL AO CÓDIGO MODELO ==========
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# Últimos dois arquivos (Hoje e Ontem) - igual ao código modelo
options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Extrair a data do arquivo selecionado (YYYY-MM-DD)
date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
if date_match:
    selected_date_str = date_match.group(0)
else:
    selected_date_str = datetime.now().strftime("%Y-%m-%d")

# Carregar os jogos do dia selecionado
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# ========== NOVO: MERGE COM LIVESCORE (IGUAL AO CÓDIGO MODELO) ==========
livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

# Ensure goal columns exist
if 'Goals_H_Today' not in games_today.columns:
    games_today['Goals_H_Today'] = np.nan
if 'Goals_A_Today' not in games_today.columns:
    games_today['Goals_A_Today'] = np.nan

# Merge with the correct LiveScore file
if os.path.exists(livescore_file):
    st.info(f"LiveScore file found: {livescore_file}")
    results_df = pd.read_csv(livescore_file)

    # FILTER OUT CANCELED AND POSTPONED GAMES
    results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
    
    required_cols = [
        'game_id', 'status', 'home_goal', 'away_goal',
        'home_ht_goal', 'away_ht_goal',
        'home_corners', 'away_corners',
        'home_yellow', 'away_yellow',
        'home_red', 'away_red'
    ]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if missing_cols:
        st.error(f"The file {livescore_file} is missing these columns: {missing_cols}")
    else:
        games_today = games_today.merge(
            results_df,
            left_on='Id',
            right_on='game_id',
            how='left',
            suffixes=('', '_RAW')
        )

        # Update goals only for finished games
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
        
        # ADD RED CARD COLUMNS
        games_today['Home_Red'] = games_today['home_red']
        games_today['Away_Red'] = games_today['away_red']
else:
    st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")

# ========== CONTINUAÇÃO DO CÓDIGO ORIGINAL ==========
# Carregar histórico para treinamento
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home", "Away","Goals_H_FT", "Goals_A_FT"], keep="first")
else:
    history = history.drop_duplicates(keep="first")

if history.empty:
    st.stop()

# Filtrar apenas jogos sem placar final (para previsão)
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


##################### BLOCO 4 – FEATURE ENGINEERING #####################
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": [
        "Diff_Power", "M_H", "M_A", "Diff_M",
        "Diff_HT_P", "M_HT_H", "M_HT_A",
        "Asian_Line_Display"   # ✅ agora usamos a versão convertida em float
    ],
    "categorical": []
}

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)
feature_blocks["categorical"] = list(history_leagues.columns)

def build_feature_matrix(df, leagues, blocks):
    dfs = []
    for block_name, cols in blocks.items():
        if block_name == "categorical":
            dfs.append(leagues)
        else:
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                dfs.append(df[available_cols])
    return pd.concat(dfs, axis=1)

X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()

X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()

numeric_cols = feature_blocks["odds"] + feature_blocks["strength"]
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]


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


##################### BLOCO 7 – TRAINING MODELS #####################
stats = []
res, model_ah_home_v1 = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v1 = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)
res, model_ah_home_v2 = train_and_evaluate_v2(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v2 = train_and_evaluate_v2(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)

stats_df = pd.DataFrame(stats)[["Model", "Accuracy", "LogLoss", "BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) – v1 vs v2")
st.dataframe(stats_df, use_container_width=True)


##################### BLOCO 8 – PREDICTIONS #####################
if ml_version_choice == "v1":
    model_ah_home, cols1 = model_ah_home_v1
    model_ah_away, cols2 = model_ah_away_v1
else:
    model_ah_home, cols1 = model_ah_home_v2
    model_ah_away, cols2 = model_ah_away_v2

X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

if normalize_features:
    scaler = StandardScaler()
    scaler.fit(X_ah_home[numeric_cols])
    X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
    X_today_ah_away[numeric_cols] = scaler.transform(X_today_ah_away[numeric_cols])

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no", "p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]

    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for cls, col in zip(model_ah_away.classes_, ["p_ah_away_no", "p_ah_away_yes"]):
        games_today[col] = probs_away[:, cls]

def color_prob(val, color):
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({color}, {alpha:.2f})"

# ========== NOVO: COLUNAS DE GOLS ADICIONADAS APÓS AWAY ==========
styled_df = (
    games_today[[
        "Date","Time","League","Home","Away",
        "Goals_H_Today", "Goals_A_Today",  # NOVO: Colunas de gols adicionadas aqui
        "Odd_H","Odd_D","Odd_A",
        "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
        "Asian_Line_Display": "{:.2f}",
        "Odd_H_Asi": "{:.2f}", "Odd_A_Asi": "{:.2f}",
        "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}",
        "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}"  # NOVO: Formatação dos gols
    }, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_ah_away_yes"])
)

st.markdown(f"### 📌 Predictions for {selected_date_str} – Asian Handicap ({ml_version_choice})")
st.dataframe(styled_df, use_container_width=True, height=800)
