########################################################
# Bloco 1 – Imports & Config
########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import math
import io
from scipy.stats import skellam
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Bet Indicator – Forecast V2", layout="wide")
st.title("📊 Bet Indicator – Forecast V2 (Enhanced Features)")

# Paths
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "copas", "uefa", "nordeste", "afc","trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


########################################################
# Bloco 2 – Funções auxiliares
########################################################
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(os.path.join(folder, f)) for f in files], ignore_index=True)

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    
    today_file = files[-1]
    yesterday_file = files[-2] if len(files) >= 2 else None

    st.markdown("### 📂 Select matches to display")
    col1, col2 = st.columns(2)
    today_checked = col1.checkbox("Today Matches", value=True)
    yesterday_checked = col2.checkbox("Yesterday Matches", value=False)

    selected_dfs = []
    if today_checked:
        selected_dfs.append(pd.read_csv(os.path.join(folder, today_file)))
    if yesterday_checked and yesterday_file:
        selected_dfs.append(pd.read_csv(os.path.join(folder, yesterday_file)))

    if not selected_dfs:
        return pd.DataFrame()
    return pd.concat(selected_dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, filename):
    path = os.path.join(MODELS_FOLDER, filename)
    with open(path, "wb") as f:
        joblib.dump(model, f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None


########################################################
# Bloco 3 – Carregar Dados
########################################################
st.info("📂 Loading data...")

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

if history.empty:
    st.error("⚠️ No valid historical data found in GamesDay.")
    st.stop()

games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.error("⚠️ No valid matches selected.")
    st.stop()


########################################################
# Bloco 4 – Targets
########################################################
history["Target"] = history.apply(
    lambda row: 0 if row["Goals_H_FT"] > row["Goals_A_FT"]
    else (1 if row["Goals_H_FT"] == row["Goals_A_FT"] else 2),
    axis=1,
)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)


########################################################
# Bloco 5 – Features Avançadas
########################################################
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

history["M_Diff"] = history["M_H"] - history["M_A"]
games_today["M_Diff"] = games_today["M_H"] - games_today["M_A"]


########################################################
# Bloco 6 – Configurações ML
########################################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"]
)
retrain = st.sidebar.checkbox("Retrain models", value=False)


########################################################
# Bloco 7 – Treino & Avaliação
########################################################
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_fc_v2.pkl"
    model = None

    if not retrain:
        model = load_model(filename)

    if model is None:
        if ml_model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced_subsample"
            )
        elif ml_model_choice == "Random Forest Tuned":
            rf_params = {
                "1X2": {'n_estimators': 600, 'max_depth': 14, 'min_samples_split': 10,
                        'min_samples_leaf': 1, 'max_features': 'sqrt'},
            }
            model = RandomForestClassifier(random_state=42, class_weight="balanced_subsample", **rf_params[name])
        elif ml_model_choice == "XGBoost Tuned":
            xgb_params = {
                "1X2": {'n_estimators': 219, 'max_depth': 9, 'learning_rate': 0.05,
                        'subsample': 0.9, 'colsample_bytree': 0.8,
                        'eval_metric': 'mlogloss', 'use_label_encoder': False},
            }
            model = XGBClassifier(random_state=42, **xgb_params[name])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        save_model(model, filename)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)
    if num_classes == 2:
        bs = brier_score_loss(y_val, probs[:, 1])
        bs = f"{bs:.3f}"
    else:
        y_onehot = pd.get_dummies(y_val).values
        bs_raw = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        bs = f"{bs_raw:.3f} (multi)"
    return {"Model": name, "Accuracy": f"{acc:.3f}", "LogLoss": f"{ll:.3f}", "Brier": bs}, model


########################################################
# Bloco 8 – Treinar Modelo + Previsões ML
########################################################
features = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "M_Diff", "Diff_HT_P"]
# Remover NaNs de forma sincronizada entre features e target
data_train = history.dropna(subset=features + ["Target"]).copy()
X_1x2 = data_train[features]
y_1x2 = data_train["Target"]


res, model_multi = train_and_evaluate(X_1x2, y_1x2, "1X2", 3)
games_today = games_today.dropna(subset=features)
games_today["p_home"], games_today["p_draw"], games_today["p_away"] = model_multi.predict_proba(games_today[features]).T


########################################################
# Bloco 9 – Conversão Asian + Skellam Model
########################################################
def convert_asian_line(line_str):
    try:
        if pd.isna(line_str) or line_str == "":
            return np.nan
        line_str = str(line_str).strip().replace("–", "-")
        if "/" in line_str:
            parts = [float(x) for x in line_str.split("/")]
            return np.mean(parts)
        return float(line_str)
    except:
        return np.nan

def skellam_handicap(mu_h, mu_a, line):
    try:
        mu_h = float(np.clip(mu_h, 0.05, 5.0))
        mu_a = float(np.clip(mu_a, 0.05, 5.0))
    except:
        return np.nan, np.nan, np.nan
    if pd.isna(line):
        return np.nan, np.nan, np.nan
    line = float(line)

    if abs(line - round(line)) < 1e-9:
        k = int(round(line))
        win = 1 - skellam.cdf(k, mu_h, mu_a)
        push = skellam.pmf(k, mu_h, mu_a)
        lose = skellam.cdf(k - 1, mu_h, mu_a)
        return win, push, lose

    if abs(line * 4 - round(line * 4)) < 1e-9:
        low_line = line - 0.25
        high_line = line + 0.25
        def single_line_prob(l):
            k = int(round(l))
            win = 1 - skellam.cdf(k, mu_h, mu_a)
            push = skellam.pmf(k, mu_h, mu_a)
            lose = skellam.cdf(k - 1, mu_h, mu_a)
            return win, push, lose
        res_low = single_line_prob(low_line)
        res_high = single_line_prob(high_line)
        return np.mean([res_low[0], res_high[0]]), np.mean([res_low[1], res_high[1]]), np.mean([res_low[2], res_high[2]])
    return np.nan, np.nan, np.nan

games_today["Asian_Home"] = games_today["Asian_Line"].apply(convert_asian_line)
games_today["Skellam_AH_Win"], games_today["Skellam_AH_Push"], games_today["Skellam_AH_Lose"] = zip(
    *games_today.apply(
        lambda r: skellam_handicap(r["XG2_H"], r["XG2_A"], r["Asian_Home"])
        if pd.notna(r["XG2_H"]) and pd.notna(r["XG2_A"]) and pd.notna(r["Asian_Home"])
        else (np.nan, np.nan, np.nan),
        axis=1
    )
)


########################################################
# Bloco 10 – Layout em Abas
########################################################
tab1, tab2 = st.tabs(["🤖 ML Forecast", "🎲 Skellam Model (1X2 + AH)"])

with tab1:
    st.markdown("### 📊 ML Predictions")
    st.dataframe(
        games_today[["League","Home","Away","Odd_H","Odd_D","Odd_A","p_home","p_draw","p_away"]],
        use_container_width=True,
        height=700
    )

with tab2:
    st.markdown("### 🎲 Skellam Probabilities by Handicap")
    st.dataframe(
        games_today[[
            "League","Home","Away","Asian_Line","Asian_Home",
            "XG2_H","XG2_A","Skellam_AH_Win","Skellam_AH_Push","Skellam_AH_Lose"
        ]].style.format({
            "Asian_Home": "{:+.2f}",
            "XG2_H": "{:.2f}", "XG2_A": "{:.2f}",
            "Skellam_AH_Win": "{:.1%}", "Skellam_AH_Push": "{:.1%}", "Skellam_AH_Lose": "{:.1%}"
        }),
        use_container_width=True,
        height=700
    )
