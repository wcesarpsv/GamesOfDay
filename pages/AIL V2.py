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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime

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

def convert_asian_line(line_str):
    """Converte string de linha asiática (ex.: '-0.25/0') em média numérica. Retorna float ou None."""
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

def invert_asian_line_str(line_str):
    """Inverte o sinal de cada parte da linha (para trocar referência Away ↔ Home). Ex.: '-0.25/0' → '0.25/0'"""
    if pd.isna(line_str):
        return np.nan
    try:
        parts = [p.strip() for p in str(line_str).split('/')]
        inv_parts = [str(-float(p)) for p in parts]
        return '/'.join(inv_parts)
    except:
        return np.nan

def calc_handicap_result(margin, asian_line_str, invert=False):
    """Retorna média de pontos por linha (1 win, 0.5 push, 0 loss)."""
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

# -------- Aggression Features do seu código --------
def add_aggression_features(df: pd.DataFrame):
    """
    Aggression ∈ [-1,1]
      >0  = dá handicap com frequência (favorito)
      <0  = recebe handicap com frequência (underdog)
    """
    df = df.copy()
    aggression_features = []
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away']):
        df['Handicap_Balance'] = df['Aggression_Home'] - df['Aggression_Away']
        df['Underdog_Indicator'] = -df['Handicap_Balance']  # Positivo = Home underdog
        if 'M_H' in df.columns and 'M_A' in df.columns:
            df['Power_vs_Perception_Home'] = df['M_H'] - df['Aggression_Home']
            df['Power_vs_Perception_Away'] = df['M_A'] - df['Aggression_Away']
            df['Power_Perception_Diff'] = df['Power_vs_Perception_Home'] - df['Power_vs_Perception_Away']
        aggression_features.extend(['Aggression_Home', 'Aggression_Away', 'Handicap_Balance',
                                    'Underdog_Indicator', 'Power_Perception_Diff'])
    if all(col in df.columns for col in ['HandScore_Home', 'HandScore_Away']):
        df['HandScore_Diff'] = df['HandScore_Home'] - df['HandScore_Away']
        aggression_features.append('HandScore_Diff')
    if all(col in df.columns for col in ['OverScore_Home', 'OverScore_Away']):
        df['OverScore_Diff'] = df['OverScore_Home'] - df['OverScore_Away']
        df['Total_OverScore'] = df['OverScore_Home'] + df['OverScore_Away']
        aggression_features.extend(['OverScore_Diff', 'Total_OverScore'])
    return df, aggression_features




########################################
##### BLOCO 3 – LOAD + TARGETS AH ######
########################################
st.info("📂 Loading data...")

# Seleção de arquivo do dia
files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

# Carregar jogos do dia
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Garantir colunas básicas
for col in ["Goals_H_Today", "Goals_A_Today"]:
    if col not in games_today.columns:
        games_today[col] = np.nan

# Merge com LiveScore
livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
if os.path.exists(livescore_file):
    results_df = pd.read_csv(livescore_file)
    results_df = results_df[~results_df["status"].isin(["Cancel", "Postp."])]
    if {"game_id","home_goal","away_goal"}.issubset(results_df.columns):
        games_today = games_today.merge(
            results_df[["game_id","home_goal","away_goal","status"]],
            left_on="Id", right_on="game_id", how="left"
        )
        games_today.loc[games_today["status"]=="FT", "Goals_H_Today"] = games_today["home_goal"]
        games_today.loc[games_today["status"]=="FT", "Goals_A_Today"] = games_today["away_goal"]
else:
    st.warning(f"No LiveScore file found for {selected_date_str}")

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
if history.empty:
    st.stop()

# --- Importar funções do AIL Option A ---
from AIL_OptionA_Module import (
    deduplicate_matches, make_display_lines, compute_targets_home_away
)

# Deduplicação + displays + targets AH
history = deduplicate_matches(history)
history = make_display_lines(history)
history = compute_targets_home_away(history)

# Limpeza básica de gols nulos
history = history.dropna(subset=["Goals_H_FT","Goals_A_FT","Asian_Line"]).copy()

# Excluir duplicatas simples
if {"Date","Home","Away"}.issubset(history.columns):
    history = history.drop_duplicates(subset=["Date","Home","Away"], keep="last")

# Jogos a prever (sem FT)
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()
if games_today.empty:
    st.warning("⚠️ No matches found for today.")
    st.stop()





########################################
#### BLOCO 4 – AIL (INTELIGÊNCIA) ######
########################################
# 4.1 – Aggression features (seu bloco)
history, aggression_features = add_aggression_features(history)
games_today, _ = add_aggression_features(games_today)

# 4.2 – AIL – funções
AIL_CFG = {"hs_neutral": 5.0, "aggr_neutral": 0.05}

def _sign(x: float) -> int:
    if pd.isna(x): return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _classify_market_alignment(agg: float, hs: float, cfg=AIL_CFG) -> str:
    ag = 0.0 if pd.isna(agg) else agg
    hs_ = 0.0 if pd.isna(hs) else hs
    if abs(ag) < cfg["aggr_neutral"] and abs(hs_) < cfg["hs_neutral"]:
        return "ALIGNED: Neutral"
    if ag > 0 and hs_ > 0:  return "FAVORITE RELIABLE"
    if ag > 0 and hs_ < 0:  return "MARKET OVERRATES"
    if ag < 0 and hs_ > 0:  return "UNDERDOG VALUE"
    if ag < 0 and hs_ < 0:  return "WEAK (Market Right)"
    return "ALIGNED: Neutral"

def _normalized_gap(a_home: float, a_away: float, eps: float = 1e-6) -> float:
    if pd.isna(a_home) or pd.isna(a_away): return np.nan
    denom = abs(a_home) + abs(a_away) + eps
    return (a_home - a_away) / denom

def build_aggression_intelligence(history: pd.DataFrame, games_today: pd.DataFrame):
    df = games_today.copy()

    for col in ["Aggression_Home","HandScore_Home","Aggression_Away","HandScore_Away","Diff_Power","Diff_HT_P"]:
        if col not in df.columns: df[col] = np.nan
    if "Handicap_Balance" not in df.columns:
        df["Handicap_Balance"] = df["Aggression_Home"] - df["Aggression_Away"]

    # Update 1 – Classes
    df["Market_Class_Home"] = [_classify_market_alignment(a,h) for a,h in zip(df["Aggression_Home"], df["HandScore_Home"])]
    df["Market_Class_Away"] = [_classify_market_alignment(a,h) for a,h in zip(df["Aggression_Away"], df["HandScore_Away"])]

    def _match_value_tag(row) -> str:
        home_tag = _classify_market_alignment(row.Aggression_Home, row.HandScore_Home)
        away_tag = _classify_market_alignment(row.Aggression_Away, row.HandScore_Away)
        if "UNDERDOG VALUE" in home_tag: return "VALUE: HOME"
        if "UNDERDOG VALUE" in away_tag: return "VALUE: AWAY"
        if "MARKET OVERRATES" in home_tag: return "FADE: HOME"
        if "MARKET OVERRATES" in away_tag: return "FADE: AWAY"
        if _sign(row.Diff_Power) > 0: return "ALIGN: HOME"
        if _sign(row.Diff_Power) < 0: return "ALIGN: AWAY"
        return "BALANCED"
    df["AIL_Match_Tag"] = df.apply(_match_value_tag, axis=1)

    # Update 2/6 – MEI & HomeBias por liga (history)
    if history is not None and not history.empty and "League" in history.columns:
        cols_req = ["League","Aggression_Home","HandScore_Home","Aggression_Away","HandScore_Away"]
        hist_ok = history[[c for c in cols_req if c in history.columns]].dropna(how="any")
        if not hist_ok.empty:
            def _mei_grp(g: pd.DataFrame) -> float:
                parts = []
                if {"Aggression_Home","HandScore_Home"}.issubset(g.columns):
                    parts.append(g[["Aggression_Home","HandScore_Home"]].rename(columns={"Aggression_Home":"Aggression","HandScore_Home":"HandScore"}))
                if {"Aggression_Away","HandScore_Away"}.issubset(g.columns):
                    parts.append(g[["Aggression_Away","HandScore_Away"]].rename(columns={"Aggression_Away":"Aggression","HandScore_Away":"HandScore"}))
                if not parts: return np.nan
                cat = pd.concat(parts, axis=0)
                if cat["Aggression"].nunique()<2 or cat["HandScore"].nunique()<2: return np.nan
                return float(cat["Aggression"].corr(cat["HandScore"]))
            league_mei = hist_ok.groupby("League", dropna=False).apply(_mei_grp).rename("League_MEI").reset_index()

            def _home_bias(g: pd.DataFrame) -> float:
                ah = g["Aggression_Home"].dropna(); aa = g["Aggression_Away"].dropna()
                if ah.empty or aa.empty: return np.nan
                return float(ah.mean() - aa.mean())
            league_homebias = hist_ok.groupby("League", dropna=False).apply(_home_bias).rename("League_HomeBias").reset_index()

            df = df.merge(league_mei, on="League", how="left")
            df = df.merge(league_homebias, on="League", how="left")
        else:
            df["League_MEI"] = np.nan; df["League_HomeBias"] = np.nan
    else:
        df["League_MEI"] = np.nan; df["League_HomeBias"] = np.nan

    # Update 3 – Divergência Mercado x Modelo
    df["Market_Model_Divergence"] = [1 if _sign(dp)!=_sign(hb) else 0 for dp,hb in zip(df["Diff_Power"], df["Handicap_Balance"])]

    # Update 4 – Aggression x Momentum (Home e Away)
    # Diff_HT_P = Home - Away  (confirmado)
    df["Aggression_Momentum_Score_Home"] = (-1.0 * df["Aggression_Home"]) * df["Diff_HT_P"]
    df["Aggression_Momentum_Score_Away"] = (-1.0 * df["Aggression_Away"]) * (-df["Diff_HT_P"])  # Away momentum = -(Home - Away)

    # Update 5 – (opcional) Trend recentes se existirem
    if "HandScore_Home_Recent5" in df.columns:
        df["Market_Adjustment_Score_Home"] = df["HandScore_Home_Recent5"].astype(float) - df["HandScore_Home"].astype(float) - df["Aggression_Home"].astype(float)
    else:
        df["Market_Adjustment_Score_Home"] = np.nan
    if "HandScore_Away_Recent5" in df.columns:
        df["Market_Adjustment_Score_Away"] = df["HandScore_Away_Recent5"].astype(float) - df["HandScore_Away"].astype(float) - df["Aggression_Away"].astype(float)
    else:
        df["Market_Adjustment_Score_Away"] = np.nan

    # Gap normalizado
    df["Aggression_Gap_Norm"] = [_normalized_gap(h,a) for h,a in zip(df["Aggression_Home"], df["Aggression_Away"])]

    # Score consolidado (considera também AWAY)
    def _consolidated_value_score(row) -> float:
        score = 0.0
        score += 0.75 * row.get("Market_Model_Divergence", 0)
        # Valorizações
        if str(row.get("Market_Class_Home","")).startswith("UNDERDOG VALUE"): score += 0.5
        if str(row.get("Market_Class_Away","")).startswith("UNDERDOG VALUE"): score += 0.5
        if str(row.get("Market_Class_Home","")).startswith("FAVORITE RELIABLE"): score += 0.25
        if str(row.get("Market_Class_Away","")).startswith("FAVORITE RELIABLE"): score += 0.25
        # Momentum subestimação (Home e Away, pequena escala)
        am_h = row.get("Aggression_Momentum_Score_Home", 0.0)
        am_a = row.get("Aggression_Momentum_Score_Away", 0.0)
        for am in (am_h, am_a):
            if not pd.isna(am): score += 0.001 * am
        # Ligas ineficientes (MEI baixo/negativo) reforçam
        mei = row.get("League_MEI", np.nan)
        if not pd.isna(mei): score += 0.25 * (0 - max(0.0, mei))
        return float(score)
    df["AIL_Value_Score"] = df.apply(_consolidated_value_score, axis=1)

    return df

# 4.3 – Executar AIL
games_today = build_aggression_intelligence(history, games_today)



########################################
#### BLOCO 4.5 – AIL-ML INTERACTIONS ###
########################################
from AIL_OptionA_Module import (
    ensure_training_interactions,
    compute_market_model_divergence,
    compute_aggression_momentum_scores,
    compute_league_mei_and_bias,
    compute_ail_value_score
)

# Garantir colunas de interações para treino
history = ensure_training_interactions(history)
games_today = ensure_training_interactions(games_today)

# Extras apenas para UI (fora do treino)
games_today = compute_market_model_divergence(games_today)
games_today = compute_aggression_momentum_scores(games_today)
games_today = compute_league_mei_and_bias(games_today)
games_today = compute_ail_value_score(games_today)




########################################
##### BLOCO 5 – FEATURE BLOCKS #########
########################################
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": [
        "Diff_Power","M_H","M_A","Diff_M",
        "Diff_HT_P","M_HT_H","M_HT_A",
        "Asian_Line_Home_Display"
    ],
    "aggression": [
        # Interações principais de treino (as 3 famílias)
        "Market_Error_Home","Market_Error_Away","Market_Error_Diff",
        "Underdog_Value_Home","Underdog_Value_Away","Underdog_Value_Diff",
        "Favorite_Crash_Home","Favorite_Crash_Away","Favorite_Crash_Diff"
    ],
    "categorical": []  # dummies de liga + classes AIL (se desejar para UI)
}

# One-hot das ligas
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

# Construir matrizes
X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()

X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()



########################################
###### BLOCO 6 – SIDEBAR & ML ##########
########################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features", value=True)

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
                   "LogLoss": log_loss(y, probs), "BrierScore": brier_score_loss(y, probs[:,1])}
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features and numeric_cols:
        # imputar mediana (só do treino) e então escalar
        train_med = X_train[numeric_cols].median()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(train_med)
        X_test[numeric_cols]  = X_test[numeric_cols].fillna(train_med)

        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

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
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v4.pkl"
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

    if normalize_features and numeric_cols:
        # imputar mediana (só do treino) e então escalar
        train_med = X_train[numeric_cols].median()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(train_med)
        X_test[numeric_cols]  = X_test[numeric_cols].fillna(train_med)

        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

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


########################################
######## BLOCO 7 – TREINAMENTO #########
########################################
stats = []
res, model_ah_home_v1 = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v1 = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)
res, model_ah_home_v2 = train_and_evaluate_v2(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v2 = train_and_evaluate_v2(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)

stats_df = pd.DataFrame(stats)[["Model", "Accuracy", "LogLoss", "BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) – v1 vs v2")
st.dataframe(stats_df, use_container_width=True)



########################################
######## BLOCO 8 – PREDICTIONS #########
########################################
if ml_version_choice == "v1":
    model_ah_home, cols1 = model_ah_home_v1
    model_ah_away, cols2 = model_ah_away_v1
else:
    model_ah_home, cols1 = model_ah_home_v2
    model_ah_away, cols2 = model_ah_away_v2

X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

# Normalização/Imputação para o "hoje"
if normalize_features and numeric_cols:
    scaler = StandardScaler()

    # mediana do histórico (usa X_ah_home construído acima)
    med = X_ah_home[numeric_cols].median()

    # preparar base para ajustar o scaler (sem NaN)
    X_ah_home_fit = X_ah_home[numeric_cols].fillna(med)
    scaler.fit(X_ah_home_fit)

    # imputar + transformar hoje
    X_today_ah_home[numeric_cols] = X_today_ah_home[numeric_cols].fillna(med)
    X_today_ah_away[numeric_cols] = X_today_ah_away[numeric_cols].fillna(med)
    X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
    X_today_ah_away[numeric_cols] = scaler.transform(X_today_ah_away[numeric_cols])

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no", "p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]

    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for cls, col in zip(model_ah_away.classes_, ["p_ah_away_no", "p_ah_away_yes"]):
        games_today[col] = probs_away[:, cls]

def color_prob(val, rgb):
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({rgb}, {alpha:.2f})"

st.markdown(f"### 📌 Predictions for {selected_date_str} – Asian Handicap ({ml_version_choice})")

# montar colunas disponíveis de forma segura
cols_show = [
    "Date","Time","League","Home","Away",
    "Goals_H_Today", "Goals_A_Today",
    "Odd_H","Odd_D","Odd_A",
    "Asian_Line_Home_Display","Odd_H_Asi","Odd_A_Asi",
    "p_ah_home_yes","p_ah_away_yes"
]
cols_show = [c for c in cols_show if c in games_today.columns]
pred_df = games_today[cols_show].copy()

fmt_map = {
    "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
    "Asian_Line_Home_Display": "{:.2f}",
    "Odd_H_Asi": "{:.2f}", "Odd_A_Asi": "{:.2f}",
    "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}",
    "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}"
}
fmt_map = {k:v for k,v in fmt_map.items() if k in pred_df.columns}

styled_df = (
    pred_df
    .style.format(fmt_map, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=[c for c in ["p_ah_home_yes"] if c in pred_df.columns])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=[c for c in ["p_ah_away_yes"] if c in pred_df.columns])
)
st.dataframe(styled_df, use_container_width=True, height=800)


########################################
#### BLOCO 9 – AIL VALUE RADAR (UI) ####
########################################
st.markdown("### 🧠 AIL – Today’s Value Radar")
radar_cols = [
    "Home","Away","League","Asian_Line_Home_Display",
    "Market_Class_Home","Market_Class_Away","AIL_Match_Tag",
    "p_ah_home_yes","p_ah_away_yes",
    "Aggression_Home","Aggression_Away","HandScore_Home","HandScore_Away",
    "Diff_Power","Diff_HT_P",
    "Aggression_Momentum_Score_Home","Aggression_Momentum_Score_Away",
    "Market_Model_Divergence","Aggression_Gap_Norm","League_MEI","League_HomeBias",
    "AIL_Value_Score"
]
radar_cols = [c for c in radar_cols if c in games_today.columns]
radar = games_today[radar_cols].copy()

# formatação amigável
if "Asian_Line_Home_Display" in radar.columns:
    radar["Asian_Line_Home_Display"] = radar["Asian_Line_Home_Display"].apply(
        lambda x: f"+{x:.2f}" if pd.notnull(x) and x>0 else (f"{x:.2f}" if pd.notnull(x) else "N/A")
    )
for pcol in [c for c in ["p_ah_home_yes","p_ah_away_yes"] if c in radar.columns]:
    radar[pcol] = radar[pcol].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "—")

st.dataframe(radar.sort_values("AIL_Value_Score", ascending=False), use_container_width=True)


########################################
### BLOCO 9.5 – AIL VIS: AGG x HS ######
########################################
import matplotlib.pyplot as plt

st.markdown("### 📈 Aggression × HandScore – Quadrantes de Valor")

def _plot_aggression_handscore(df: pd.DataFrame, side: str = "Home"):
    ax = plt.figure(figsize=(6, 5)).gca()
    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)

    x = pd.to_numeric(df[f"Aggression_{side}"], errors="coerce")
    y = pd.to_numeric(df[f"HandScore_{side}"], errors="coerce")
    mask = x.notna() & y.notna()

    ax.scatter(x[mask], y[mask], alpha=0.6, s=20)

    ax.set_xlabel(f"Aggression_{side} (−1 zebra ↔ +1 favorito)")
    ax.set_ylabel(f"HandScore_{side} (− falha ↔ + cobre)")
    ax.set_title(f"Aggression vs HandScore – {side}")

    # limites e anotações estáveis
    y_max = float(y[mask].max()) if mask.any() else 0.5
    y_min = float(y[mask].min()) if mask.any() else -0.5
    ax.text(-0.95, max(y_max, 0), "Underdog Value\n(x<0, y>0)", fontsize=9)
    ax.text( 0.05, max(y_max, 0), "Favorite Reliable\n(x>0, y>0)", fontsize=9)
    ax.text( 0.05, min(y_min, 0), "Market Overrates\n(x>0, y<0)", fontsize=9)
    ax.text(-0.95, min(y_min, 0), "Weak Underdog\n(x<0, y<0)", fontsize=9)

    st.pyplot(ax.figure)

col_h, col_a = st.columns(2)
with col_h:
    _plot_aggression_handscore(games_today, side="Home")
with col_a:
    _plot_aggression_handscore(games_today, side="Away")



########################################
### BLOCO 9.6 – AIL EXPLANATIONS #######
########################################
st.markdown("### 🗒️ AIL – Explicações por Jogo")

def explain_match(row: pd.Series) -> str:
    home, away = row.get("Home","?"), row.get("Away","?")
    # A linha armazenada é do AWAY; exibimos ambas as visões (Home = sinal invertido)
    asian_away = row.get("Asian_Line_Away_Display", np.nan)
    if pd.notnull(asian_away):
        try:
            asian_away_f = float(asian_away)
            home_line = -asian_away_f
            away_line =  asian_away_f
            line_txt = f"{home} {home_line:+.2f} / {away} {away_line:+.2f}"
        except:
            # fallback textual
            asian_home = row.get("Asian_Line_Home_Display", np.nan)
            line_txt = f"{home} {asian_home} / {away} ({asian_away})"
    else:
        asian_home = row.get("Asian_Line_Home_Display", np.nan)
        if pd.notnull(asian_home):
            try:
                asian_home_f = float(asian_home)
                line_txt = f"{home} {asian_home_f:+.2f} / {away} {(-asian_home_f):+.2f}"
            except:
                line_txt = f"{home} {asian_home} / {away} (oposto)"
        else:
            line_txt = "N/A"

    p_home = row.get("p_ah_home_yes", np.nan)
    p_away = row.get("p_ah_away_yes", np.nan)
    p_txt = f"Prob AH – Home: {p_home:.1%} | Away: {p_away:.1%}" if pd.notnull(p_home) and pd.notnull(p_away) else "Prob AH – N/A"

    tag = row.get("AIL_Match_Tag","—")
    mclass_h = row.get("Market_Class_Home","—")
    mclass_a = row.get("Market_Class_Away","—")

    # Sinal curto
    signal = ""
    if isinstance(tag, str):
        if "VALUE: AWAY" in tag: signal = "🎯 Valor no visitante"
        elif "VALUE: HOME" in tag: signal = "🎯 Valor no mandante"
        elif "FADE: HOME" in tag: signal = "📉 Fade no mandante"
        elif "FADE: AWAY" in tag: signal = "📉 Fade no visitante"
        else: signal = "⚖️ Equilíbrio/Alinhado"
    else:
        signal = "⚖️ Equilíbrio/Alinhado"

    return (
        f"**{home} vs {away}**  \n"
        f"🧮 Asian Line: {line_txt}  \n"
        f"🏷️ Classes – Home: {mclass_h} | Away: {mclass_a}  \n"
        f"📊 {p_txt}  \n"
        f"🧠 Sinal AIL: **{tag}** → {signal}"
    )

# Render
for _, r in games_today.iterrows():
    st.markdown(explain_match(r))
    st.markdown("---")
