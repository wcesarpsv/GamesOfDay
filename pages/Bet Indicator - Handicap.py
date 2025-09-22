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

st.set_page_config(page_title="Bet Indicator – Asian Handicap", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (Home vs Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "AsianHandicap"   # prefixo único
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
from datetime import datetime, timedelta

st.info("📂 Loading data...")

# Load histórico completo
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# Garantir que não haja duplicatas: Date + Home + Away
if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
else:
    history = history.drop_duplicates(keep="first")

if history.empty:
    st.stop()

# ---------------- Jogos de hoje ----------------
today = datetime.now().strftime("%Y-%m-%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))

if "Date" in games_today.columns:
    games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

# Apenas jogos de hoje
games_today = games_today[games_today["Date"] == today].copy()

include_yesterday = st.sidebar.checkbox("Include yesterday's matches", value=False)
if include_yesterday:
    games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
    if "Date" in games_today.columns:
        games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games_today = games_today[games_today["Date"].isin([today, yesterday])].copy()

# Remover duplicatas
if set(["Date", "Home", "Away"]).issubset(games_today.columns):
    games_today = games_today.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")

# Apenas jogos sem placar final
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.warning("⚠️ No matches found for today (or yesterday, if selected).")
    st.stop()

# ---------------- Função para converter Asian_Line para decimal ----------------
def convert_asian_line(line_str):
    """
    Converte linha asiática fracionada (ex: '0/0.5') para decimal (ex: 0.25).
    """
    try:
        # Se vier vazio ou None
        if pd.isna(line_str) or line_str == "":
            return None

        # Sempre tratar como string
        line_str = str(line_str).strip()

        # Se não houver "/", retorna como float
        if "/" not in line_str:
            return float(line_str)

        # Divide a linha e calcula a média
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
    
    except:
        return None

# Criar coluna de exibição no formato decimal
history["Asian_Line_Display"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Display"] = games_today["Asian_Line"].apply(convert_asian_line)

# ---------------- Handicap Asian Logic ----------------
def calc_handicap_result(margin, asian_line_str, invert=False):
    """Calcula o resultado do handicap asiático (0.0, 0.5, 1.0)."""
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

# Margem de gols
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]

# Resultado bruto
history["Handicap_Home_Result"] = history.apply(
    lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1
)
history["Handicap_Away_Result"] = history.apply(
    lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1
)

# Targets binários
history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)



##################### BLOCO 4 – FEATURE ENGINEERING #####################
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A", "Odd_H_Asi", "Odd_A_Asi", "OU_Total"],
    "strength": ["Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A"],
    "categorical": []
}

# One-hot encoding de ligas
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

# Histórico
X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()

# Jogos de hoje
X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)

X_today_ah_away = X_today_ah_home.copy()

# 🔹 Não normalizar aqui — será feito dentro do train_and_evaluate para evitar leakage
numeric_cols = sum([cols for name, cols in feature_blocks.items() if name != "categorical"], [])
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]



##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
retrain = st.sidebar.checkbox("Retrain models", value=False)



##################### BLOCO 6 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2C.pkl"

    feature_cols = X.columns.tolist()

    # Tenta carregar modelo salvo
    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols, scaler = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {
                "Model": name,
                "Accuracy": accuracy_score(y, preds),
                "LogLoss": log_loss(y, probs),
                "BrierScore": brier_score_loss(y, probs[:,1])  # binário corrigido
            }
            return res, (model, cols, scaler)

    # Split temporal (dataset deve estar ordenado por data!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 🔹 Normalização SEM leakage
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if ml_model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    else:
        model = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            use_label_encoder=False, random_state=42
        )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "LogLoss": log_loss(y_test, probs),
        "BrierScore": brier_score_loss(y_test, probs[:,1])  # binário corrigido
    }

    save_model((model, feature_cols, scaler), feature_cols, filename)
    return res, (model, feature_cols, scaler)




##################### BLOCO 7 – TRAINING MODELS #####################
stats = []
res, model_ah_home = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home")
stats.append(res)

res, model_ah_away = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away")
stats.append(res)

st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(pd.DataFrame(stats), use_container_width=True)



##################### BLOCO 8 – PREDICTIONS #####################
model_ah_home, cols1, scaler_home = model_ah_home
model_ah_away, cols2, scaler_away = model_ah_away

# Reindexar para alinhar features
X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

# 🔹 Aplicar o scaler correto (o mesmo usado no treino)
if not games_today.empty:
    if scaler_home is not None:
        X_today_ah_home[numeric_cols] = scaler_home.transform(X_today_ah_home[numeric_cols])
    if scaler_away is not None:
        X_today_ah_away[numeric_cols] = scaler_away.transform(X_today_ah_away[numeric_cols])

    # Previsões para Home
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no", "p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]

    # Previsões para Away
    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for cls, col in zip(model_ah_away.classes_, ["p_ah_away_no", "p_ah_away_yes"]):
        games_today[col] = probs_away[:, cls]


##################### BLOCO 9 – DISPLAY #####################
def color_prob(val, color):
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({color}, {alpha:.2f})"

styled_df = (
    games_today[[
        "Date","Time","League","Home","Away",
        "Odd_H", "Odd_D", "Odd_A",
        "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H": "{:.2f}",
        "Odd_D": "{:.2f}",
        "Odd_A": "{:.2f}",
        "Asian_Line_Display": "{:.2f}",  # Linha no formato decimal
        "Odd_H_Asi": "{:.2f}",
        "Odd_A_Asi": "{:.2f}",
        "p_ah_home_yes": "{:.1%}",
        "p_ah_away_yes": "{:.1%}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_ah_away_yes"])
)

st.markdown("### 📌 Predictions for Today's Matches – Asian Handicap")
st.dataframe(styled_df, use_container_width=True, height=800)


