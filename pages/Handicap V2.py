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

# ========== SELEÇÃO DE DATA ==========
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# Últimos dois arquivos (Hoje e Ontem)
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

# ========== MERGE COM LIVESCORE ==========
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

# ========== CARREGAR HISTÓRICO ==========
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


##################### BLOCO 4 – FEATURE ENGINEERING COM AGGRESSION #####################

# NOVO: ADICIONAR FEATURES DE AGGRESSION
def add_aggression_features(df):
    """
    Aggression positivo = DÁ mais handicap (favorito)
    Aggression negativo = RECEBE mais handicap (underdog)
    """
    # Features básicas de aggression (se existirem)
    aggression_features = []
    
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away']):
        # Features estratégicas para handicap
        df['Handicap_Balance'] = df['Aggression_Home'] - df['Aggression_Away']
        df['Underdog_Indicator'] = -df['Handicap_Balance']  # Positivo = Home underdog
        
        # Power vs Market Perception
        if 'M_H' in df.columns and 'M_A' in df.columns:
            df['Power_vs_Perception_Home'] = df['M_H'] - df['Aggression_Home']
            df['Power_vs_Perception_Away'] = df['M_A'] - df['Aggression_Away']
            df['Power_Perception_Diff'] = df['Power_vs_Perception_Home'] - df['Power_vs_Perception_Away']
        
        aggression_features.extend(['Aggression_Home', 'Aggression_Away', 'Handicap_Balance', 
                                  'Underdog_Indicator', 'Power_Perception_Diff'])
    
    # Adicionar HandScore se disponível
    if all(col in df.columns for col in ['HandScore_Home', 'HandScore_Away']):
        df['HandScore_Diff'] = df['HandScore_Home'] - df['HandScore_Away']
        aggression_features.append('HandScore_Diff')
    
    # Adicionar OverScore se disponível
    if all(col in df.columns for col in ['OverScore_Home', 'OverScore_Away']):
        df['OverScore_Diff'] = df['OverScore_Home'] - df['OverScore_Away']
        df['Total_OverScore'] = df['OverScore_Home'] + df['OverScore_Away']
        aggression_features.extend(['OverScore_Diff', 'Total_OverScore'])
    
    return df, aggression_features

# Aplicar às bases
history, aggression_features = add_aggression_features(history)
games_today, _ = add_aggression_features(games_today)

# BLOCO DE FEATURES ATUALIZADO
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": [
        "Diff_Power", "M_H", "M_A", "Diff_M",
        "Diff_HT_P", "M_HT_H", "M_HT_A",
        "Asian_Line_Display"
    ],
    "aggression": aggression_features,  # NOVO BLOCO DINÂMICO
    "categorical": []
}

# Filtrar apenas as features que existem no dataframe
available_aggression = [col for col in feature_blocks["aggression"] if col in history.columns]
feature_blocks["aggression"] = available_aggression

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)
feature_blocks["categorical"] = list(history_leagues.columns)

def build_feature_matrix(df, leagues, blocks):
    dfs = []
    for block_name, cols in blocks.items():
        if block_name == "categorical":
            dfs.append(leagues)
        elif cols:  # Só adiciona se a lista não estiver vazia
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                dfs.append(df[available_cols])
    return pd.concat(dfs, axis=1)

X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()

X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()

# ATUALIZAR numeric_cols para incluir as novas features
numeric_cols = (feature_blocks["odds"] + feature_blocks["strength"] + 
                feature_blocks["aggression"])
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]

# BLOCO EXTRA – ANÁLISE DAS NOVAS FEATURES
st.markdown("### 🔍 Análise: Aggression vs Asian Handicap")

if not history.empty and all(col in history.columns for col in ['Target_AH_Home', 'Target_AH_Away'] + aggression_features):
    
    # Análise de correlação
    available_aggression_for_analysis = [f for f in aggression_features if f in history.columns]
    
    if available_aggression_for_analysis:
        # Correlação com target Home
        corr_home = history[available_aggression_for_analysis + ['Target_AH_Home']].corr()['Target_AH_Home'].drop('Target_AH_Home')
        
        # Correlação com target Away  
        corr_away = history[available_aggression_for_analysis + ['Target_AH_Away']].corr()['Target_AH_Away'].drop('Target_AH_Away')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlação com Handicap Home Win:**")
            corr_df_home = pd.DataFrame({
                'Feature': corr_home.index,
                'Correlation': corr_home.values
            }).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df_home.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
        
        with col2:
            st.write("**Correlação com Handicap Away Win:**")
            corr_df_away = pd.DataFrame({
                'Feature': corr_away.index, 
                'Correlation': corr_away.values
            }).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df_away.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
        
        # Destacar features mais promissoras
        strong_features_home = corr_df_home[abs(corr_df_home['Correlation']) > 0.05]
        strong_features_away = corr_df_away[abs(corr_df_away['Correlation']) > 0.05]
        
        if not strong_features_home.empty or not strong_features_away.empty:
            st.success("🎯 **Features mais promissoras (correlação > |0.05|):**")
            if not strong_features_home.empty:
                st.write("Para Handicap Home:", ", ".join(strong_features_home['Feature'].tolist()))
            if not strong_features_away.empty:
                st.write("Para Handicap Away:", ", ".join(strong_features_away['Feature'].tolist()))


##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features (odds + strength + aggression)", value=True)


##################### BLOCO 6 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v2.pkl"
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
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v2.pkl"
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

# ========== EXIBIÇÃO DAS PREDIÇÕES ==========
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

st.markdown(f"### 📌 Predictions for {selected_date_str} – Asian Handicap ({ml_version_choice})")
st.dataframe(styled_df, use_container_width=True, height=800)

# NOVO: MOSTRAR AS FEATURES DE AGGRESSION NOS JOGOS DE HOJE
if aggression_features:
    st.markdown("### 🔎 Aggression Features nos Jogos de Hoje")
    aggression_cols = [col for col in aggression_features if col in games_today.columns]
    if aggression_cols:
        aggression_display = games_today[["Home", "Away"] + aggression_cols].copy()
        # Formatar valores numéricos
        for col in aggression_cols:
            if aggression_display[col].dtype in ['float64', 'int64']:
                aggression_display[col] = aggression_display[col].round(3)
        
        st.dataframe(aggression_display, use_container_width=True)



##################### BLOCO 8.5 – ASIAN HANDICAP INDICATOR CLARO (AMBOS LADOS) #####################

def create_ah_indicator_both_sides(row):
    """
    Cria um indicador claro para Asian Handicap considerando ambos os lados
    """
    underdog_ind = row.get('Underdog_Indicator', 0)
    p_ah_home = row.get('p_ah_home_yes', 0.5)
    p_ah_away = row.get('p_ah_away_yes', 0.5)
    
    # Diferença de probabilidades
    prob_diff = p_ah_home - p_ah_away
    
    # Determinar vantagem baseada nas probabilidades (PRINCIPAL)
    if prob_diff > 0.15:  # Home tem vantagem forte
        return f"🏠⤴️ VANTAGEM FORTE HOME (Prob: {p_ah_home:.1%} vs {p_ah_away:.1%})"
    elif prob_diff > 0.05:  # Home tem vantagem
        return f"🏠⤴️ VANTAGEM HOME (Prob: {p_ah_home:.1%} vs {p_ah_away:.1%})"
    elif prob_diff < -0.15:  # Away tem vantagem forte
        return f"🚌⤴️ VANTAGEM FORTE AWAY (Prob: {p_ah_away:.1%} vs {p_ah_home:.1%})"
    elif prob_diff < -0.05:  # Away tem vantagem
        return f"🚌⤴️ VANTAGEM AWAY (Prob: {p_ah_away:.1%} vs {p_ah_home:.1%})"
    else:
        return "⚖️ EQUILIBRADO (Sem vantagem clara)"

def get_ah_recommendation_both_sides(row):
    """
    Recomendação direta baseada PRINCIPALMENTE nas probabilidades
    """
    underdog_ind = row.get('Underdog_Indicator', 0)
    p_ah_home = row.get('p_ah_home_yes', 0.5)
    p_ah_away = row.get('p_ah_away_yes', 0.5)
    
    # REGRA PRINCIPAL: Apostar no lado com maior probabilidade
    prob_diff = p_ah_home - p_ah_away
    
    # Verificar consistência com underdog indicator
    home_consistent = (prob_diff > 0 and underdog_ind < 0)  # Home favorito e tem maior prob
    away_consistent = (prob_diff < 0 and underdog_ind > 0)  # Away underdog e tem maior prob
    
    # Recomendações baseadas nas probabilidades
    if p_ah_home > 0.60 and home_consistent:
        return f"✅ APOSTAR HOME AH (Prob: {p_ah_home:.1%}, Conf: Alta)"
    elif p_ah_away > 0.60 and away_consistent:
        return f"✅ APOSTAR AWAY AH (Prob: {p_ah_away:.1%}, Conf: Alta)"
    elif p_ah_home > 0.55 and home_consistent:
        return f"🎯 FORTE SINAL HOME AH (Prob: {p_ah_home:.1%})"
    elif p_ah_away > 0.55 and away_consistent:
        return f"🎯 FORTE SINAL AWAY AH (Prob: {p_ah_away:.1%})"
    elif p_ah_home > p_ah_away and p_ah_home > 0.52:
        return f"📈 SINAL HOME AH (Prob: {p_ah_home:.1%})"
    elif p_ah_away > p_ah_home and p_ah_away > 0.52:
        return f"📈 SINAL AWAY AH (Prob: {p_ah_away:.1%})"
    else:
        # Motivos específicos para aguardar
        if max(p_ah_home, p_ah_away) < 0.52:
            return "⏸️ AGUARDAR: Probabilidades baixas"
        elif abs(prob_diff) < 0.03:
            return "⏸️ AGUARDAR: Diferença muito pequena"
        else:
            return "⏸️ AGUARDAR: Sinal insuficiente"

def get_ah_side_analysis(row):
    """
    Análise detalhada de ambos os lados - MAIS CLARA
    """
    p_ah_home = row.get('p_ah_home_yes', 0.5)
    p_ah_away = row.get('p_ah_away_yes', 0.5)
    underdog_ind = row.get('Underdog_Indicator', 0)
    
    # Determinar status mais claro
    home_status = "FAVORITO" if underdog_ind < 0 else "UNDERDOG" if underdog_ind > 0 else "NEUTRO"
    away_status = "UNDERDOG" if underdog_ind < 0 else "FAVORITO" if underdog_ind > 0 else "NEUTRO"
    
    return f"Home: {p_ah_home:.1%} ({home_status}) | Away: {p_ah_away:.1%} ({away_status})"

# Aplicar aos dados
games_today['AH_Indicator'] = games_today.apply(create_ah_indicator_both_sides, axis=1)
games_today['AH_Recommendation'] = games_today.apply(get_ah_recommendation_both_sides, axis=1)
games_today['AH_Side_Analysis'] = games_today.apply(get_ah_side_analysis, axis=1)

# NOVA TABELA SIMPLIFICADA COM INDICADORES CLAROS
st.markdown("### 🎯 ASIAN HANDICAP INDICATOR - AMBOS LADOS")

simple_display = games_today[[
    "Home", "Away", "Asian_Line_Display",
    "AH_Indicator", "AH_Recommendation",
    "p_ah_home_yes", "p_ah_away_yes"
]].copy()

# Formatação
simple_display["Asian_Line_Display"] = simple_display["Asian_Line_Display"].apply(
    lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}" if pd.notnull(x) else "N/A"
)

# Função de estilo melhorada
def color_recommendation(val):
#     if "APOSTAR HOME" in str(val):
#         return "background-color: #90EE90; font-weight: bold"
#     elif "APOSTAR AWAY" in str(val):
#         return "background-color: #87CEEB; font-weight: bold"
#     elif "FORTE SINAL" in str(val):
#         return "background-color: #FFD700"
#     elif "AGUARDAR" in str(val):
#         return "background-color: #FFB6C1"
    return ""

styled_simple = (
    simple_display
    .style.format({
        "p_ah_home_yes": "{:.1%}",
        "p_ah_away_yes": "{:.1%}"
    })
    .applymap(color_recommendation, subset=["AH_Recommendation"])
)

st.dataframe(styled_simple, use_container_width=True)

# RESUMO DOS SINAIS MELHORADO
st.markdown("### 📊 RESUMO DOS SINAIS - AMBOS LADOS")

apostar_home = len([x for x in games_today['AH_Recommendation'] if "APOSTAR HOME" in x])
apostar_away = len([x for x in games_today['AH_Recommendation'] if "APOSTAR AWAY" in x])
forte_home = len([x for x in games_today['AH_Recommendation'] if "FORTE SINAL HOME" in x])
forte_away = len([x for x in games_today['AH_Recommendation'] if "FORTE SINAL AWAY" in x])
aguardar_count = len([x for x in games_today['AH_Recommendation'] if "AGUARDAR" in x])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("✅ Apostar Home", apostar_home)
col2.metric("✅ Apostar Away", apostar_away)
col3.metric("🎯 Forte Home", forte_home)
col4.metric("🎯 Forte Away", forte_away)
col5.metric("⏸️ Aguardar", aguardar_count)

# ANÁLISE DE EFETIVIDADE
st.markdown("### 🔍 ANÁLISE DE VANTAGEM")

if not games_today.empty:
    avg_home_prob = games_today['p_ah_home_yes'].mean()
    avg_away_prob = games_today['p_ah_away_yes'].mean()
    home_advantage_count = len(games_today[games_today['p_ah_home_yes'] > games_today['p_ah_away_yes']])
    away_advantage_count = len(games_today[games_today['p_ah_away_yes'] > games_today['p_ah_home_yes']])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Prob Média Home", f"{avg_home_prob:.1%}")
    col2.metric("📊 Prob Média Away", f"{avg_away_prob:.1%}")
    col3.metric("⚖️ Vantagem Home", f"{home_advantage_count}/{len(games_today)}")
