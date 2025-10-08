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

# -------- Aggression Features do seu código --------
def add_aggression_features(df: pd.DataFrame):
    """
    Aggression positivo = DÁ mais handicap (favorito)
    Aggression negativo = RECEBE mais handicap (underdog)
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

# Garantir colunas para merge
if 'Goals_H_Today' not in games_today.columns:
    games_today['Goals_H_Today'] = np.nan
if 'Goals_A_Today' not in games_today.columns:
    games_today['Goals_A_Today'] = np.nan

# Merge com LiveScore do dia
livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
if os.path.exists(livescore_file):
    st.info(f"LiveScore file found: {livescore_file}")
    results_df = pd.read_csv(livescore_file)
    results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
    required_cols = [
        'game_id', 'status', 'home_goal', 'away_goal',
        'home_ht_goal', 'away_ht_goal',
        'home_corners', 'away_corners',
        'home_yellow', 'away_yellow',
        'home_red', 'away_red'
    ]
    missing_cols = [c for c in required_cols if c not in results_df.columns]
    if missing_cols:
        st.error(f"The file {livescore_file} is missing these columns: {missing_cols}")
    else:
        games_today = games_today.merge(
            results_df, left_on='Id', right_on='game_id',
            how='left', suffixes=('', '_RAW')
        )
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
        games_today['Home_Red'] = games_today['home_red']
        games_today['Away_Red'] = games_today['away_red']
else:
    st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")

# Histórico
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home", "Away", "Goals_H_FT", "Goals_A_FT"], keep="first")
else:
    history = history.drop_duplicates(keep="first")
if history.empty:
    st.stop()

# Apenas jogos sem FT hoje (para prever)
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()
if games_today.empty:
    st.warning("⚠️ No matches found for today (or yesterday, if selected).")
    st.stop()

# Targets AH históricos
history["Asian_Line_Display"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Display"] = games_today["Asian_Line"].apply(convert_asian_line)

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Handicap_Home_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1)
history["Handicap_Away_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1)
history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)


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

    # Score consolidado (agora considerando também AWAY)
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
#### BLOCO 4.5 – AIL-ML INTERACTIONS ####
########################################
def add_ail_ml_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensina explicitamente a lógica mercado x resultado:
    - Market_Error: Aggression * HandScore (positivo = mercado acertou; negativo = errou)
    - Underdog_Value: (-Aggression) * max(HandScore, 0)      (zebra que cobre)
    - Favorite_Crash: ( Aggression) * min(HandScore, 0)      (favorito que falha)
    Cria versões Home e Away e diffs.
    """
    out = df.copy()

    # Safety
    for c in ["Aggression_Home","Aggression_Away","HandScore_Home","HandScore_Away"]:
        if c not in out.columns: out[c] = np.nan

    # Home
    out["Market_Error_Home"] = out["Aggression_Home"] * out["HandScore_Home"]
    out["Underdog_Value_Home"] = (-out["Aggression_Home"]) * np.maximum(0.0, out["HandScore_Home"].astype(float))
    out["Favorite_Crash_Home"] = ( out["Aggression_Home"]) * np.minimum(0.0, out["HandScore_Home"].astype(float))

    # Away
    out["Market_Error_Away"] = out["Aggression_Away"] * out["HandScore_Away"]
    out["Underdog_Value_Away"] = (-out["Aggression_Away"]) * np.maximum(0.0, out["HandScore_Away"].astype(float))
    out["Favorite_Crash_Away"] = ( out["Aggression_Away"]) * np.minimum(0.0, out["HandScore_Away"].astype(float))

    # Diffs (sinal útil pra ML)
    out["Market_Error_Diff"] = out["Market_Error_Home"] - out["Market_Error_Away"]
    out["Underdog_Value_Diff"] = out["Underdog_Value_Home"] - out["Underdog_Value_Away"]
    out["Favorite_Crash_Diff"] = out["Favorite_Crash_Home"] - out["Favorite_Crash_Away"]

    return out

# Aplicar nas bases
history = add_ail_ml_interactions(history)
games_today = add_ail_ml_interactions(games_today)





########################################
##### BLOCO 5 – FEATURE BLOCKS #########
########################################
# Bloco de features original + AIL
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": ["Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A", "Asian_Line_Display"],
    "aggression": [],   # preencheremos abaixo
    "categorical": []   # dummies de liga + classes do AIL
}

# Aggression originais + AIL novas
base_aggr = ['Aggression_Home','Aggression_Away','Handicap_Balance','Underdog_Indicator','Power_Perception_Diff','HandScore_Diff']
ail_new = [
    "Market_Model_Divergence","Aggression_Momentum_Score_Home","Aggression_Momentum_Score_Away",
    "Aggression_Gap_Norm","League_MEI","League_HomeBias","AIL_Value_Score"
]
aggr_all = [c for c in (base_aggr + ail_new) if c in games_today.columns or c in history.columns]
feature_blocks["aggression"] = aggr_all



####################

# --- NOVAS features AIL-ML (interações explícitas) ---
ail_ml_interactions = [
    "Market_Error_Home","Market_Error_Away","Market_Error_Diff",
    "Underdog_Value_Home","Underdog_Value_Away","Underdog_Value_Diff",
    "Favorite_Crash_Home","Favorite_Crash_Away","Favorite_Crash_Diff"
]

# Garantir que existam (em history/games_today) antes de incluir
ail_ml_interactions = [c for c in ail_ml_interactions if (c in games_today.columns or c in history.columns)]

# Injetar no bloco de aggression (numéricas)
feature_blocks["aggression"] = list(dict.fromkeys(feature_blocks["aggression"] + ail_ml_interactions))

# Atualizar numeric_cols (mantendo binárias de fora)
numeric_cols = (
    feature_blocks["odds"]
    + feature_blocks["strength"]
    + [c for c in feature_blocks["aggression"] if c not in ["Market_Model_Divergence"]]
)
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]


##########

# Categóricas: Ligas + classes AIL
history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League").reindex(columns=history_leagues.columns, fill_value=0)
feature_blocks["categorical"] = list(history_leagues.columns) + ["Market_Class_Home","Market_Class_Away","AIL_Match_Tag"]

def build_feature_matrix(df, leagues, blocks):
    dfs = []
    for block_name, cols in blocks.items():
        if block_name == "categorical":
            # Ligas (dummies)
            dfs.append(leagues)
            # Categóricas textuais (one-hot simples)
            cat_cols = [c for c in ["Market_Class_Home","Market_Class_Away","AIL_Match_Tag"] if c in df.columns]
            if cat_cols:
                dummies = pd.get_dummies(df[cat_cols], columns=cat_cols, dummy_na=False)
                dfs.append(dummies)
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

# Numéricas para normalização
numeric_cols = (
    feature_blocks["odds"]
    + feature_blocks["strength"]
    + [c for c in feature_blocks["aggression"] if c not in ["Market_Model_Divergence"]]  # binária fica de fora
)
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]



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
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v3.pkl"
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
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v3.pkl"
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

if normalize_features and numeric_cols:
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

st.markdown(f"### 📌 Predictions for {selected_date_str} – Asian Handicap ({ml_version_choice})")
styled_df = (
    games_today[[
        "Date","Time","League","Home","Away",
        "Goals_H_Today", "Goals_A_Today",
        "Odd_H","Odd_D","Odd_A",
        "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
        "Asian_Line_Display": "{:.2f}",
        "Odd_H_Asi": "{:.2f}", "Odd_A_Asi": "{:.2f}",
        "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}",
        "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_ah_away_yes"])
)
st.dataframe(styled_df, use_container_width=True, height=800)




########################################
#### BLOCO 9 – AIL VALUE RADAR (UI) ####
########################################
st.markdown("### 🧠 AIL – Today’s Value Radar")
radar_cols = [
    "Home","Away","League","Asian_Line_Display",
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
if "Asian_Line_Display" in radar.columns:
    radar["Asian_Line_Display"] = radar["Asian_Line_Display"].apply(lambda x: f"+{x:.2f}" if pd.notnull(x) and x>0 else (f"{x:.2f}" if pd.notnull(x) else "N/A"))
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
    x = df[f"Aggression_{side}"].astype(float)
    y = df[f"HandScore_{side}"].astype(float)
    ax.scatter(x, y, alpha=0.6, s=20)

    ax.set_xlabel(f"Aggression_{side} (−1 zebra ↔ +1 favorito)")
    ax.set_ylabel(f"HandScore_{side} (− falha ↔ + cobre)")
    ax.set_title(f"Aggression vs HandScore – {side}")

    # Anotações dos quadrantes
    ax.text(-0.95, max(y.fillna(0).max(), 0) if y.notna().any() else 0.5, "Underdog Value\n(x<0, y>0)", fontsize=9)
    ax.text( 0.05, max(y.fillna(0).max(), 0) if y.notna().any() else 0.5, "Favorite Reliable\n(x>0, y>0)", fontsize=9)
    ax.text( 0.05, min(y.fillna(0).min(), 0) if y.notna().any() else -0.5, "Market Overrates\n(x>0, y<0)", fontsize=9)
    ax.text(-0.95, min(y.fillna(0).min(), 0) if y.notna().any() else -0.5, "Weak Underdog\n(x<0, y<0)", fontsize=9)

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
    asian_home = row.get("Asian_Line_Display", np.nan)

    # Interpretação da linha: se Asian_Line_Display é do Home,
    # então Home recebe esse valor, Away recebe o oposto.
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
    if "VALUE: AWAY" in tag: signal = "🎯 Valor no visitante"
    elif "VALUE: HOME" in tag: signal = "🎯 Valor no mandante"
    elif "FADE: HOME" in tag: signal = "📉 Fade no mandante"
    elif "FADE: AWAY" in tag: signal = "📉 Fade no visitante"
    else: signal = "⚖️ Equilíbrio/Alinhado"

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



