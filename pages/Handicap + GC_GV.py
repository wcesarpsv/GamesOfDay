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
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (Home vs Away)")

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
    history = history.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
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
    games_today = games_today.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")

if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.warning("⚠️ No matches found for today (or yesterday, if selected).")
    st.stop()

# Handicap calculation
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
history["Handicap_Home_Result"] = history.apply(
    lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1
)
history["Handicap_Away_Result"] = history.apply(
    lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1
)

history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)


##################### BLOCO EXTRA – CONVERSÃO ASIAN LINE #####################
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

history["Asian_Line_Decimal"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Decimal"] = games_today["Asian_Line"].apply(convert_asian_line)


##################### BLOCO EXTRA – COSTO & VALOR DO GOL (ASIAN) #####################
def calc_profit_asian(odd_liquid, result):
    if pd.isna(result) or pd.isna(odd_liquid):
        return 0
    return odd_liquid * result - (1 - result)

history["Bet_Result_AH_Home"] = history.apply(
    lambda r: calc_profit_asian(r["Odd_H_Asi"], r["Handicap_Home_Result"]), axis=1
)
history["Bet_Result_AH_Away"] = history.apply(
    lambda r: calc_profit_asian(r["Odd_A_Asi"], r["Handicap_Away_Result"]), axis=1
)

history["Custo_Gol_AH_Home"] = np.where(history["Goals_H_FT"] > 0, history["Odd_H_Asi"] / history["Goals_H_FT"], 0)
history["Custo_Gol_AH_Away"] = np.where(history["Goals_A_FT"] > 0, history["Odd_A_Asi"] / history["Goals_A_FT"], 0)

history["Valor_Gol_AH_Home"] = np.where(history["Goals_H_FT"] > 0,
                                        history["Bet_Result_AH_Home"] / history["Goals_H_FT"], 0)
history["Valor_Gol_AH_Away"] = np.where(history["Goals_A_FT"] > 0,
                                        history["Bet_Result_AH_Away"] / history["Goals_A_FT"], 0)

def rolling_stats(sub_df, col, window=5, min_periods=1):
    return sub_df.sort_values("Date")[col].rolling(window=window, min_periods=min_periods).mean()

history = history.sort_values("Date")
history["Media_CustoGol_AH_Home"] = history.groupby("Home", group_keys=False).apply(lambda x: rolling_stats(x, "Custo_Gol_AH_Home")).shift(1)
history["Media_ValorGol_AH_Home"] = history.groupby("Home", group_keys=False).apply(lambda x: rolling_stats(x, "Valor_Gol_AH_Home")).shift(1)
history["Media_CustoGol_AH_Away"] = history.groupby("Away", group_keys=False).apply(lambda x: rolling_stats(x, "Custo_Gol_AH_Away")).shift(1)
history["Media_ValorGol_AH_Away"] = history.groupby("Away", group_keys=False).apply(lambda x: rolling_stats(x, "Valor_Gol_AH_Away")).shift(1)

for col in ["Custo_Gol_AH_Home","Custo_Gol_AH_Away","Valor_Gol_AH_Home","Valor_Gol_AH_Away",
            "Media_CustoGol_AH_Home","Media_ValorGol_AH_Home","Media_CustoGol_AH_Away","Media_ValorGol_AH_Away"]:
    games_today[col] = np.nan

for idx, row in games_today.iterrows():
    home = row["Home"]
    away = row["Away"]
    last_home = history[history["Home"] == home].sort_values("Date").tail(1)
    last_away = history[history["Away"] == away].sort_values("Date").tail(1)
    if not last_home.empty:
        games_today.at[idx, "Media_CustoGol_AH_Home"] = last_home["Media_CustoGol_AH_Home"].values[-1]
        games_today.at[idx, "Media_ValorGol_AH_Home"] = last_home["Media_ValorGol_AH_Home"].values[-1]
    if not last_away.empty:
        games_today.at[idx, "Media_CustoGol_AH_Away"] = last_away["Media_CustoGol_AH_Away"].values[-1]
        games_today.at[idx, "Media_ValorGol_AH_Away"] = last_away["Media_ValorGol_AH_Away"].values[-1]


##################### BLOCO 4 – FEATURE ENGINEERING #####################
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A", "Odd_H_Asi", "Odd_A_Asi", "OU_Total"],
    "strength": ["Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A", "Asian_Line_Decimal"],
    "goals": ["Media_CustoGol_AH_Home","Media_ValorGol_AH_Home",
              "Media_CustoGol_AH_Away","Media_ValorGol_AH_Away"],
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

scaler = StandardScaler()
numeric_cols = sum([cols for name, cols in feature_blocks.items() if name != "categorical"], [])
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]

X_ah_home[numeric_cols] = scaler.fit_transform(X_ah_home[numeric_cols])
X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
X_ah_away[numeric_cols] = X_ah_home[numeric_cols]
X_today_ah_away[numeric_cols] = X_today_ah_home[numeric_cols]


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
    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {
                "Model": name,
                "Accuracy": accuracy_score(y, preds),
                "LogLoss": log_loss(y, probs),
                "BrierScore": brier_score_loss(pd.get_dummies(y).values.ravel(), probs.ravel())
            }
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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
        "BrierScore": brier_score_loss(pd.get_dummies(y_test).values.ravel(), probs.ravel())
    }

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


##################### BLOCO 7 – TRAINING MODELS #####################
stats = []
res, model_ah_home = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)

st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(pd.DataFrame(stats), use_container_width=True)


##################### BLOCO 8 – PREDICTIONS #####################
model_ah_home, cols1 = model_ah_home
model_ah_away, cols2 = model_ah_away

X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no", "p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]

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
        "Asian_Line","Asian_Line_Decimal","Odd_H_Asi","Odd_A_Asi",
        "Media_CustoGol_AH_Home","Media_ValorGol_AH_Home",
        "Media_CustoGol_AH_Away","Media_ValorGol_AH_Away",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H": "{:.2f}","Odd_D": "{:.2f}","Odd_A": "{:.2f}",
        "Asian_Line_Decimal": "{:.2f}",
        "Odd_H_Asi": "{:.2f}","Odd_A_Asi": "{:.2f}",
        "Media_CustoGol_AH_Home": "{:.2f}","Media_ValorGol_AH_Home": "{:.2f}",
        "Media_CustoGol_AH_Away": "{:.2f}","Media_ValorGol_AH_Away": "{:.2f}",
        "p_ah_home_yes": "{:.1%}","p_ah_away_yes": "{:.1%}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_ah_away_yes"])
)

st.markdown("### 📌 Predictions for Today's Matches – Asian Handicap")
st.dataframe(styled_df, use_container_width=True, height=800)
