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
            val = float(line_str)
            # CORREÇÃO 1: Corrige -0.0 para 0.0
            return 0.0 if abs(val) < 1e-10 else val
        parts = [float(x) for x in line_str.split("/")]
        avg = sum(parts) / len(parts)
        # CORREÇÃO 1: Corrige -0.0 para 0.0
        return 0.0 if abs(avg) < 1e-10 else avg
    except:
        return None

def fix_zero_display(line):
    """Função auxiliar para corrigir zeros na exibição"""
    if pd.isna(line):
        return line
    return 0.0 if abs(line) < 1e-10 else line

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

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

# Jogos do dia
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

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home", "Away", "Goals_H_FT", "Goals_A_FT"], keep="first")
else:
    history = history.drop_duplicates(keep="first")
if history.empty:
    st.stop()

# ==============================
# 🔒 ANTI-LEAKAGE FILTER
# ==============================
# Filtra o histórico para incluir apenas jogos anteriores à data selecionada
if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"Treinando modelo apenas com jogos até {selected_date_str} (sem vazamento temporal).")
        if history.empty:
            st.warning("⚠️ Nenhum dado histórico anterior à data selecionada — não é possível treinar.")
            st.stop()
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")



# CORREÇÃO 2: Corrigir odds asiáticas (valor líquido → bruto)
def correct_asiatic_odds(df):
    """Corrige odds asiáticas de valor líquido para valor bruto"""
    df = df.copy()
    if 'Odd_H_Asi' in df.columns:
        df['Odd_H_Asi'] = df['Odd_H_Asi'] + 1
    if 'Odd_A_Asi' in df.columns:
        df['Odd_A_Asi'] = df['Odd_A_Asi'] + 1
    return df

# Aplicar correção das odds
history = correct_asiatic_odds(history)
games_today = correct_asiatic_odds(games_today)

# Apenas jogos sem FT hoje (para prever)
# 🔁 Para simular dias passados: manter todos os jogos, mesmo com resultados
if "Goals_H_FT" in games_today.columns:
    if games_today["Goals_H_FT"].isna().all():
        st.info("✅ Todos os jogos do dia ainda sem resultado (modo previsão).")
    else:
        st.info("📊 Exibindo todos os jogos (modo simulação histórica / backtest).")

if games_today.empty:
    st.warning("⚠️ No matches found for today (or yesterday, if selected).")
    st.stop()

# ATENÇÃO: a Asian_Line no CSV refere-se ao AWAY.
# Criamos duas visões: Away (original) e Home (sinal invertido).
history["Asian_Line_Away_Display"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Away_Display"] = games_today["Asian_Line"].apply(convert_asian_line)

# CORREÇÃO 1: Aplicar correção de zeros
history["Asian_Line_Home_Display"] = history["Asian_Line_Away_Display"].apply(lambda x: -fix_zero_display(x) if pd.notna(x) else x)
games_today["Asian_Line_Home_Display"] = games_today["Asian_Line_Away_Display"].apply(lambda x: -fix_zero_display(x) if pd.notna(x) else x)

# Garantir tipo numérico (blinda CSVs com strings)
history["Asian_Line_Away_Display"] = pd.to_numeric(history["Asian_Line_Away_Display"], errors="coerce")
games_today["Asian_Line_Away_Display"] = pd.to_numeric(games_today["Asian_Line_Away_Display"], errors="coerce")
history["Asian_Line_Home_Display"] = pd.to_numeric(history["Asian_Line_Home_Display"], errors="coerce")
games_today["Asian_Line_Home_Display"] = pd.to_numeric(games_today["Asian_Line_Home_Display"], errors="coerce")

# Targets AH históricos
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]

# HOME usa a linha invertida (porque a original é do AWAY)
history["Handicap_Home_Result"] = history.apply(
    lambda r: calc_handicap_result(r["Margin"], invert_asian_line_str(r["Asian_Line"]), invert=False), axis=1
)
# AWAY segue usando a linha original com invert=True
history["Handicap_Away_Result"] = history.apply(
    lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1
)

history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x > 0.5 else 0)
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
#### BLOCO 4.X – AIL Dynamic Learning ####
########################################
# Este bloco adiciona duas inteligências complementares ao AIL:
#  1️⃣ Pesos dinâmicos por liga (aprendidos com base na correlação entre variáveis AIL e HandScore real)
#  2️⃣ Métrica de consistência de mercado por time (volatilidade de Aggression/HandScore)
#  3️⃣ Integração no cálculo de um novo AIL_Value_Score_Dynamic (mais contextual)

st.markdown("### 🧠 AIL Dynamic Learning – League Weights + Market Consistency")

# ----------------------------------------------
# 1️⃣ Aprendizado de pesos dinâmicos por liga
# ----------------------------------------------
@st.cache_data
def learn_league_weights(history_df: pd.DataFrame):
    """Aprende pesos por liga com base na correlação entre variáveis AIL e desempenho real (HandScore_Diff)."""
    components = [
        "Market_Model_Divergence",
        "Aggression_Momentum_Score_Home",
        "Aggression_Momentum_Score_Away",
        "Underdog_Value_Diff",
        "Favorite_Crash_Diff"
    ]

    weights_by_league = {}
    for lg, g in history_df.groupby("League"):
        corrs = {}
        for c in components:
            if c in g.columns and "HandScore_Diff" in g.columns:
                corrs[c] = g[c].corr(g["HandScore_Diff"])
        # Normalização para evitar explosões numéricas
        total = sum(abs(v) for v in corrs.values() if not pd.isna(v))
        if total > 0:
            corrs = {k: v / total for k, v in corrs.items()}
        weights_by_league[lg] = corrs
    return weights_by_league

# Aprende os pesos de cada liga com base no histórico disponível
league_weights = learn_league_weights(history)
st.success(f"✅ Pesos dinâmicos aprendidos para {len(league_weights)} ligas.")

# ----------------------------------------------
# 2️⃣ Consistência de mercado por time
# ----------------------------------------------
@st.cache_data
def compute_market_consistency(history_df: pd.DataFrame):
    """
    Calcula a consistência do mercado (volatilidade) por time,
    com base no desvio padrão de Aggression e HandScore ao longo do tempo.
    Quanto maior o valor, mais imprevisível é o time.
    """
    # Cálculo separado para mandante e visitante
    agg_home = history_df.groupby("Home")["Aggression_Home"].std().rename("Agg_Std_Home")
    agg_away = history_df.groupby("Away")["Aggression_Away"].std().rename("Agg_Std_Away")
    hs_home = history_df.groupby("Home")["HandScore_Home"].std().rename("HS_Std_Home")
    hs_away = history_df.groupby("Away")["HandScore_Away"].std().rename("HS_Std_Away")

    # Média entre agressão e handscore (proxy de consistência)
    df_home = pd.concat([agg_home, hs_home], axis=1).mean(axis=1).rename("Market_Consistency_Home")
    df_away = pd.concat([agg_away, hs_away], axis=1).mean(axis=1).rename("Market_Consistency_Away")

    # Normalização (z-score)
    df_home = (df_home - df_home.mean()) / df_home.std(ddof=0)
    df_away = (df_away - df_away.mean()) / df_away.std(ddof=0)

    return df_home, df_away

market_consistency_home, market_consistency_away = compute_market_consistency(history)
st.success("✅ Consistência de mercado calculada por time (volatilidade de Aggression/HandScore).")

# Merge no games_today
games_today = games_today.merge(
    market_consistency_home, left_on="Home", right_index=True, how="left"
)
games_today = games_today.merge(
    market_consistency_away, left_on="Away", right_index=True, how="left"
)

# ----------------------------------------------
# 3️⃣ AIL Value Score Dinâmico (com pesos e consistência)
# ----------------------------------------------
def compute_dynamic_value_score(row):
    """Cálculo contextualizado de valor por confronto (com pesos da liga e ajuste de consistência)."""
    league = row.get("League", None)
    comps = {
        "Market_Model_Divergence": row.get("Market_Model_Divergence", 0),
        "Aggression_Momentum_Score_Home": row.get("Aggression_Momentum_Score_Home", 0),
        "Aggression_Momentum_Score_Away": row.get("Aggression_Momentum_Score_Away", 0),
        "Underdog_Value_Diff": row.get("Underdog_Value_Diff", 0),
        "Favorite_Crash_Diff": row.get("Favorite_Crash_Diff", 0)
    }

    # Pega os pesos específicos da liga (ou padrão neutro)
    weights = league_weights.get(league, {})
    score = 0.0
    for k, v in comps.items():
        w = weights.get(k, 0.2)  # peso padrão 0.2 se não houver histórico
        score += w * v

    # Penaliza times com mercado muito previsível (consistência baixa)
    mc_home = row.get("Market_Consistency_Home", 0)
    mc_away = row.get("Market_Consistency_Away", 0)
    avg_mc = np.nanmean([mc_home, mc_away])
    if not np.isnan(avg_mc):
        score -= 0.1 * avg_mc

    # Reforça ligas ineficientes (MEI negativo)
    mei = row.get("League_MEI", np.nan)
    if not np.isnan(mei):
        score += 0.25 * (0 - max(0.0, mei))

    return float(score)

games_today["AIL_Value_Score_Dynamic"] = games_today.apply(compute_dynamic_value_score, axis=1)

# ----------------------------------------------
# 4️⃣ Exibição dos resultados
# ----------------------------------------------
st.markdown("#### 📊 AIL – Liga & Time Contextual Intelligence")
show_cols = [
    "League",
    "Home", "Away",
    "AIL_Value_Score", "AIL_Value_Score_Dynamic",
    "League_MEI", "League_HomeBias",
    "Market_Consistency_Home", "Market_Consistency_Away"
]
show_cols = [c for c in show_cols if c in games_today.columns]

st.dataframe(
    games_today[show_cols]
    .style.format({
        "AIL_Value_Score": "{:.3f}",
        "AIL_Value_Score_Dynamic": "{:.3f}",
        "League_MEI": "{:.2f}",
        "Market_Consistency_Home": "{:.2f}",
        "Market_Consistency_Away": "{:.2f}"
    })
    .background_gradient(subset=["AIL_Value_Score_Dynamic"], cmap="RdYlGn"),
    use_container_width=True, height=520
)

st.caption(
    "O AIL_Value_Score_Dynamic combina pesos aprendidos por liga com ajustes de consistência do mercado por time. "
    "Isso permite adaptar o modelo à eficiência e volatilidade específicas de cada contexto competitivo."
)



########################################
#### BLOCO 4.Y – AIL Insights Generator ####
########################################
# Mostra ícones de ajuda ❓ no cabeçalho e remove qualquer coluna oculta auxiliar.

import streamlit as st
import numpy as np
import pandas as pd

st.markdown("### 💡 AIL Insights Generator – Contextual Summary")

# ----------------------------------------------
# 1️⃣ Função geradora de insights
# ----------------------------------------------
def generate_insight(row):
    league_mei = row.get("League_MEI", np.nan)
    homebias = row.get("League_HomeBias", np.nan)
    mc_home = row.get("Market_Consistency_Home", np.nan)
    mc_away = row.get("Market_Consistency_Away", np.nan)
    val = row.get("AIL_Value_Score_Dynamic", 0)

    if league_mei > 0.6:
        league_txt = "Liga eficiente"
    elif league_mei < 0.3:
        league_txt = "Liga ineficiente"
    else:
        league_txt = "Liga moderadamente eficiente"

    bias_txt = (
        "com viés pró-mandante"
        if homebias > 0.3
        else "com leve tendência neutra"
        if abs(homebias) <= 0.3
        else "com viés pró-visitante"
    )

    if mc_home < 0 and mc_away > 0:
        cons_txt = "mandante previsível e visitante imprevisível"
    elif mc_home > 0 and mc_away < 0:
        cons_txt = "mandante imprevisível e visitante estável"
    elif mc_home > 0 and mc_away > 0:
        cons_txt = "ambos imprevisíveis"
    else:
        cons_txt = "ambos consistentes"

    if val > 0.5:
        lado = f"Home ({row.get('Home','?')})"
        intensidade = f"🔥 Forte (+{val:.2f})"
        insight = f"💎 {league_txt} {bias_txt}; {cons_txt} → valor contextual pró-mandante."
    elif val < -0.5:
        lado = f"Away ({row.get('Away','?')})"
        intensidade = f"🔻 Forte ({val:.2f})"
        insight = f"⚠️ {league_txt} {bias_txt}; {cons_txt} → valor contextual pró-visitante."
    else:
        lado = "Neutro"
        intensidade = f"⚪ Fraco ({val:.2f})"
        insight = f"⚖️ {league_txt} {bias_txt}; {cons_txt} → sem valor claro detectado."

    return pd.Series({"Insight": insight, "Lado sugerido": lado, "Intensidade": intensidade})


# ----------------------------------------------
# 2️⃣ Aplicação e renomeações
# ----------------------------------------------
if "AIL_Value_Score_Dynamic" in games_today.columns:
    insights_df = games_today.copy()
    insights_df[["Insight","Lado sugerido","Intensidade"]] = insights_df.apply(generate_insight, axis=1)

    rename_map = {}
    if "Leagues" in insights_df.columns and "League" not in insights_df.columns:
        rename_map["Leagues"] = "League"
    if "HomeTeam" in insights_df.columns and "Home" not in insights_df.columns:
        rename_map["HomeTeam"] = "Home"
    if "AwayTeam" in insights_df.columns and "Away" not in insights_df.columns:
        rename_map["AwayTeam"] = "Away"
    if rename_map:
        insights_df = insights_df.rename(columns=rename_map)

    # ----------------------------------------------
    # 3️⃣ Cabeçalho com ícones ❓ integrados
    # ----------------------------------------------
    # st.markdown("""
    # <style>
    # .help-icon {
    #     font-size: 16px;
    #     color: #aaa;
    #     margin-left: 6px;
    #     cursor: pointer;
    # }
    # </style>
    # """, unsafe_allow_html=True)

    # st.markdown("""
    # #### 📊 Resumo de Insights AIL  
    # **Legenda:**  
    # Insight ❓ = interpretação contextual  Intensidade ❓ = força do sinal de valor
    # """)

    cols_to_show = [c for c in ["League","Home","Away","Goals_H_Today","Goals_A_Today","Insight","Lado sugerido","Intensidade"]
                    if c in insights_df.columns]

    # Exibe apenas as colunas principais
    st.dataframe(
        insights_df[cols_to_show]
        .style
        .format({
            "Goals_H_Today": "{:.0f}",
            "Goals_A_Today": "{:.0f}"
        })
        .set_properties(**{"white-space": "pre-wrap"}),
        use_container_width=True,
        height=600
    )

    # ----------------------------------------------
    # 4️⃣ Explicações clicáveis (abaixo da tabela)
    # ----------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("❓ Explicação – Insight"):
            st.markdown("""
            O campo **Insight** combina:
            - Eficiência e viés da liga (`League_MEI`, `League_HomeBias`);
            - Consistência de mercado dos times (`Market_Consistency_*`);
            - Direção do valor esperado.

            Resultado: uma leitura textual do contexto de valor detectado pelo AIL.
            """)
    with c2:
        with st.expander("❓ Explicação – Intensidade"):
            st.markdown("""
            A **Intensidade** expressa a força do sinal de valor:
            - Baseada em `AIL_Value_Score_Dynamic`;
            - Valores positivos → favorecem o **mandante**;
            - Valores negativos → favorecem o **visitante**;
            - Quanto mais distante de 0, mais forte o sinal.
            """)

    st.caption("💬 Clique nos ícones ❓ acima para entender o significado de cada métrica.")

else:
    st.warning("⚠️ A coluna 'AIL_Value_Score_Dynamic' não foi encontrada em games_today. Gere o BLOCO 4.X antes deste.")




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

# CORREÇÃO 3: Adicionar features explícitas dos quadrantes
def add_quadrant_features(df):
    """Adiciona features explícitas para os quadrantes Aggression × HandScore"""
    df = df.copy()
    
    # Home Quadrants (binárias)
    df['Home_Underdog_Value'] = ((df['Aggression_Home'] < 0) & (df['HandScore_Home'] > 0)).astype(int)
    df['Home_Favorite_Reliable'] = ((df['Aggression_Home'] > 0) & (df['HandScore_Home'] > 0)).astype(int)
    df['Home_Market_Overrates'] = ((df['Aggression_Home'] > 0) & (df['HandScore_Home'] < 0)).astype(int)
    df['Home_Weak_Underdog'] = ((df['Aggression_Home'] < 0) & (df['HandScore_Home'] < 0)).astype(int)
    
    # Away Quadrants (binárias)
    df['Away_Underdog_Value'] = ((df['Aggression_Away'] < 0) & (df['HandScore_Away'] > 0)).astype(int)
    df['Away_Favorite_Reliable'] = ((df['Aggression_Away'] > 0) & (df['HandScore_Away'] > 0)).astype(int)
    df['Away_Market_Overrates'] = ((df['Aggression_Away'] > 0) & (df['HandScore_Away'] < 0)).astype(int)
    df['Away_Weak_Underdog'] = ((df['Aggression_Away'] < 0) & (df['HandScore_Away'] < 0)).astype(int)
    
    # Strength of signal (contínuas)
    df['Home_Value_Strength'] = (-df['Aggression_Home']) * df['HandScore_Home']
    df['Away_Value_Strength'] = (-df['Aggression_Away']) * df['HandScore_Away']
    
    # Value differential
    df['Value_Strength_Diff'] = df['Home_Value_Strength'] - df['Away_Value_Strength']
    
    return df

# Aplicar correção dos quadrantes
history = add_quadrant_features(history)
games_today = add_quadrant_features(games_today)



########################################
##### BLOCO 5 – FEATURE BLOCKS #########
########################################
# Bloco de features original + AIL
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A", "Odd_H_Asi", "Odd_A_Asi"],  # CORREÇÃO 2: odds asiáticas incluídas
    "strength": [
        "Diff_Power", "M_H", "M_A", "Diff_M",
        "Diff_HT_P", "M_HT_H", "M_HT_A",
        # usar a visão coerente com o margin (Home − Away):
        "Asian_Line_Home_Display"
    ],
    "aggression": [],   # preencheremos abaixo
    "quadrants": [],    # CORREÇÃO 3: novo bloco para features de quadrantes
    "categorical": []   # dummies de liga + classes do AIL
}

# Aggression originais + AIL novas
base_aggr = ['Aggression_Home','Aggression_Away','Handicap_Balance','Underdog_Indicator','Power_Perception_Diff','HandScore_Diff']
ail_new = [
    "Market_Model_Divergence","Aggression_Momentum_Score_Home","Aggression_Momentum_Score_Away",
    "Aggression_Gap_Norm","League_MEI","League_HomeBias","AIL_Value_Score"
]
ail_ml_interactions = [
    "Market_Error_Home","Market_Error_Away","Market_Error_Diff",
    "Underdog_Value_Home","Underdog_Value_Away","Underdog_Value_Diff",
    "Favorite_Crash_Home","Favorite_Crash_Away","Favorite_Crash_Diff"
]

# Garantir que existam antes de incluir
aggr_all = [c for c in (base_aggr + ail_new + ail_ml_interactions) if (c in games_today.columns or c in history.columns)]
feature_blocks["aggression"] = list(dict.fromkeys(aggr_all))

# CORREÇÃO 3: Features de quadrantes
quadrant_features = [
    'Home_Underdog_Value', 'Home_Favorite_Reliable', 'Home_Market_Overrates', 'Home_Weak_Underdog',
    'Away_Underdog_Value', 'Away_Favorite_Reliable', 'Away_Market_Overrates', 'Away_Weak_Underdog',
    'Home_Value_Strength', 'Away_Value_Strength', 'Value_Strength_Diff'
]
quadrant_features = [c for c in quadrant_features if (c in games_today.columns or c in history.columns)]
feature_blocks["quadrants"] = quadrant_features

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

# Montar matrizes
X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()

X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()

# Numéricas para normalização (calcular SÓ agora)
numeric_cols = (
    feature_blocks["odds"]
    + feature_blocks["strength"]
    + [c for c in feature_blocks["aggression"] if c not in ["Market_Model_Divergence"]]  # binária fica de fora
    + [c for c in feature_blocks["quadrants"] if not c.endswith(('Value', 'Reliable', 'Overrates', 'Underdog'))]  # excluir binárias
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
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v7.pkl"
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
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2CH_v7.pkl"
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
###### BLOCO 6.5 – NO ODDS MODE ########
########################################
# Adiciona opção na sidebar
ignore_odds = st.sidebar.checkbox("🎯 Ignore Odds Features (No Odds Mode)", value=False)

# Aplicar lógica condicional
if ignore_odds:
    st.sidebar.info("Odds removidas do modelo para avaliar sensibilidade e valor real.")
    
    # Remover blocos de odds
    feature_blocks["odds"] = []
    
    # Atualizar prefixo de salvamento (sem sobrescrever os modelos padrão)
    PAGE_PREFIX = PAGE_PREFIX + "_NoOdds"

    # Atualizar lista de colunas numéricas (sem odds)
    numeric_cols = (
        feature_blocks["strength"]
        + [c for c in feature_blocks["aggression"] if c not in ["Market_Model_Divergence"]]
        + [c for c in feature_blocks["quadrants"] if not c.endswith(('Value', 'Reliable', 'Overrates', 'Underdog'))]
    )
    numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]

    model_mode_label = "No Odds"
else:
    st.sidebar.info("Odds incluídas no modelo (modo padrão).")
    model_mode_label = "With Odds"






########################################
######## BLOCO 7 – TREINAMENTO #########
########################################
stats = []
mode_col = model_mode_label  # captura modo atual (With Odds / No Odds)
res, model_ah_home_v1 = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res["Mode"] = mode_col

res, model_ah_away_v1 = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)
res["Mode"] = mode_col

res, model_ah_home_v2 = train_and_evaluate_v2(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res["Mode"] = mode_col

res, model_ah_away_v2 = train_and_evaluate_v2(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)
res["Mode"] = mode_col

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

# st.markdown(f"### 📌 Predictions for {selected_date_str} – Asian Handicap ({ml_version_choice})")

# # montar colunas disponíveis de forma segura
# cols_show = [
#     "Date","Time","League","Home","Away",
#     "Goals_H_Today", "Goals_A_Today",
#     "Odd_H","Odd_D","Odd_A",
#     "Asian_Line_Home_Display","Odd_H_Asi","Odd_A_Asi",
#     "p_ah_home_yes","p_ah_away_yes"
# ]
# cols_show = [c for c in cols_show if c in games_today.columns]
# pred_df = games_today[cols_show].copy()

# fmt_map = {
#     "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
#     "Asian_Line_Home_Display": "{:.2f}",
#     "Odd_H_Asi": "{:.2f}", "Odd_A_Asi": "{:.2f}",
#     "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}",
#     "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}"
# }
# fmt_map = {k:v for k,v in fmt_map.items() if k in pred_df.columns}

# styled_df = (
#     pred_df
#     .style.format(fmt_map, na_rep="—")
#     .applymap(lambda v: color_prob(v, "0,200,0"), subset=[c for c in ["p_ah_home_yes"] if c in pred_df.columns])
#     .applymap(lambda v: color_prob(v, "255,140,0"), subset=[c for c in ["p_ah_away_yes"] if c in pred_df.columns])
# )
# st.dataframe(styled_df, use_container_width=True, height=800)



########################################
### BLOCO 8.4 – XG ESTIMATOR VIA MODEL METRICS
########################################
# st.markdown("### ⚙️ Gerando Expected Goals (xG2_H / xG2_A) via Métricas Internas")

def model_based_xg(row, total_goals_avg=2.6):
    """
    Gera xG2_H e xG2_A baseados em métricas internas (M, HandScore).
    Mantém escala média de gols total (~2.6).
    """
    # pega métricas principais
    m_h = row.get("M_H", np.nan)
    m_a = row.get("M_A", np.nan)
    hs_h = row.get("HandScore_Home", np.nan)
    hs_a = row.get("HandScore_Away", np.nan)

    if pd.isna(m_h) or pd.isna(m_a):
        return np.nan, np.nan

    # pesos: prioriza M_, mas mistura com HandScore (0.7/0.3)
    w_m, w_hs = 0.7, 0.3
    s_home = (w_m * m_h) + (w_hs * (hs_h if not pd.isna(hs_h) else 0))
    s_away = (w_m * m_a) + (w_hs * (hs_a if not pd.isna(hs_a) else 0))

    s_home = max(0.01, s_home)
    s_away = max(0.01, s_away)

    total_s = s_home + s_away
    if total_s == 0:
        return np.nan, np.nan

    xg_home = total_goals_avg * (s_home / total_s)
    xg_away = total_goals_avg * (s_away / total_s)

    xg_home = np.clip(xg_home, 0.3, 3.5)
    xg_away = np.clip(xg_away, 0.3, 3.5)

    return xg_home, xg_away


if not {"XG2_H", "XG2_A"}.issubset(games_today.columns):
    # st.info("Gerando xG2_H e xG2_A a partir das métricas internas do modelo...")
    est = games_today.apply(lambda r: model_based_xg(r), axis=1, result_type="expand")
    est.columns = ["XG2_H", "XG2_A"]
    games_today = pd.concat([games_today, est], axis=1)
else:
    st.success("XG2_H e XG2_A já presentes no dataset.")

# st.write("📊 Exemplo de xG interno (5 primeiros):")
# st.dataframe(games_today[["Home","Away","M_H","M_A","HandScore_Home","HandScore_Away","XG2_H","XG2_A"]].head(5))


########################################
### BLOCO 8.5 – ML-BASED XG ESTIMATOR (xG2_H, xG2_A)
########################################
from sklearn.ensemble import RandomForestRegressor

st.markdown("### 🤖 ML-based xG Estimator (Experimental)")

use_ml_xg = st.checkbox("Use ML-based xG Estimator", value=False)

# Features para treinar o modelo
xg_features = [
    "M_H", "M_A",
    "HandScore_Home", "HandScore_Away",
    "Aggression_Home", "Aggression_Away",
    "Diff_Power"
]

# Verifica se as colunas estão disponíveis
missing_cols = [c for c in xg_features if c not in history.columns]
if missing_cols:
    st.warning(f"⚠️ Não é possível treinar o modelo de xG — faltando colunas: {missing_cols}")
    use_ml_xg = False

if use_ml_xg and not history.empty and set(["Goals_H_FT","Goals_A_FT"]).issubset(history.columns):
    # Prepara dados
    X_xg = history[xg_features].copy()
    y_h = history["Goals_H_FT"].astype(float)
    y_a = history["Goals_A_FT"].astype(float)

    # Imputação simples
    X_xg = X_xg.fillna(X_xg.median())

    # Treina dois modelos separados (Home e Away)
    model_xg_home = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    model_xg_away = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )

    model_xg_home.fit(X_xg, y_h)
    model_xg_away.fit(X_xg, y_a)

    # Gera previsões para os jogos de hoje
    X_today_xg = games_today[xg_features].copy()
    X_today_xg = X_today_xg.fillna(X_xg.median())

    games_today["XG2_H"] = model_xg_home.predict(X_today_xg)
    games_today["XG2_A"] = model_xg_away.predict(X_today_xg)

    # Clamping dos valores
    games_today["XG2_H"] = games_today["XG2_H"].clip(0.3, 3.5)
    games_today["XG2_A"] = games_today["XG2_A"].clip(0.3, 3.5)

    st.success("✅ xG2_H e xG2_A gerados via modelo ML (RandomForest).")

    # Exemplo visual (opcional)
    st.dataframe(
        games_today[["Home","Away","M_H","M_A","HandScore_Home","HandScore_Away",
                     "Aggression_Home","Aggression_Away","Diff_Power","XG2_H","XG2_A"]]
        .head(10)
        .style.format({"M_H":"{:.2f}","M_A":"{:.2f}",
                       "HandScore_Home":"{:.2f}","HandScore_Away":"{:.2f}",
                       "Aggression_Home":"{:.2f}","Aggression_Away":"{:.2f}",
                       "Diff_Power":"{:.2f}","XG2_H":"{:.2f}","XG2_A":"{:.2f}"})
    )

else:
    if not use_ml_xg:
        st.info("Modo ML-based xG desativado. Usando modelo analítico (model_based_xg).")
    elif history.empty:
        st.warning("Histórico vazio — não foi possível treinar o modelo ML.")








########################################
### BLOCO 8.6 – AH PROBABILITIES (HOME & AWAY, POISSON)
########################################
from scipy.stats import poisson

st.markdown("### 🎯 Probabilidade AH – Home & Away (Poisson via xG interno)")

def _expand_quarter_line(line_value):
    if pd.isna(line_value):
        return []
    frac = abs(line_value) - abs(int(line_value))
    s = 1.0 if line_value >= 0 else -1.0
    if np.isclose(frac, 0.25):
        a = int(line_value)
        b = a - 0.5 * s
        return [a, b]
    if np.isclose(frac, 0.75):
        a = int(line_value) + 0.5 * s
        b = int(line_value) + 1.0 * s
        return [a, b]
    return [line_value]

def _parse_home_line_parts(row):
    try:
        if "Asian_Line" in row and pd.notna(row["Asian_Line"]):
            s = invert_asian_line_str(row["Asian_Line"])
            parts = [float(x) for x in str(s).split("/")]
            return parts
    except Exception:
        pass
    line = row.get("Asian_Line_Home_Display", np.nan)
    if pd.isna(line): return []
    return _expand_quarter_line(float(line))

def _parse_away_line_parts(row):
    try:
        if "Asian_Line" in row and pd.notna(row["Asian_Line"]):
            parts = [float(x) for x in str(row["Asian_Line"]).split("/")]
            return parts
    except Exception:
        pass
    line = row.get("Asian_Line_Away_Display", np.nan)
    if pd.isna(line):
        alt = row.get("Asian_Line_Home_Display", np.nan)
        if pd.notna(alt): line = -float(alt)
    if pd.isna(line): return []
    return _expand_quarter_line(float(line))

def _score_matrix(xGh, xGa, max_goals=10):
    gh = np.arange(0, max_goals + 1)
    ga = np.arange(0, max_goals + 1)
    P = np.outer(poisson.pmf(gh, xGh), poisson.pmf(ga, xGa))
    return P

def _ah_probs_single_line(P, line):
    max_goals = P.shape[0] - 1
    margins = np.subtract.outer(np.arange(0, max_goals + 1), np.arange(0, max_goals + 1))
    p_win  = float(P[margins > line].sum())
    p_push = float(P[np.isclose(margins, line)].sum())
    p_lose = float(P[margins < line].sum())
    total = p_win + p_push + p_lose
    if total > 0:
        p_win /= total; p_push /= total; p_lose /= total
    return p_win, p_push, p_lose

def _ah_probs_split(P, parts):
    if not parts:
        return np.nan, np.nan, np.nan
    acc = np.zeros(3)
    for lp in parts:
        acc += _ah_probs_single_line(P, lp)
    return tuple(acc / len(parts))

def _add_fair_odds(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c + "_FairOdd"] = np.where(out[c] > 0, 1.0 / out[c], np.nan)
    return out


# ---- Processamento principal ----
if {"XG2_H", "XG2_A"}.issubset(games_today.columns):
    matrices = []
    for _, r in games_today.iterrows():
        if pd.isna(r["XG2_H"]) or pd.isna(r["XG2_A"]):
            matrices.append(None)
        else:
            matrices.append(_score_matrix(r["XG2_H"], r["XG2_A"], 10))

    home_probs, away_probs = [], []
    for i, r in games_today.iterrows():
        P = matrices[i]
        if P is None:
            home_probs.append((np.nan, np.nan, np.nan))
            away_probs.append((np.nan, np.nan, np.nan))
            continue

        home_parts = _parse_home_line_parts(r)
        h_w, h_p, h_l = _ah_probs_split(P, home_parts)
        home_probs.append((h_w, h_p, h_l))

        away_parts = _parse_away_line_parts(r)
        aw_w, aw_p, aw_l = (np.nan, np.nan, np.nan)
        if away_parts:
            parts_home_equiv = [-lp for lp in away_parts]
            w_h, u_h, l_h = _ah_probs_split(P, parts_home_equiv)
            aw_w, aw_p, aw_l = l_h, u_h, w_h
        away_probs.append((aw_w, aw_p, aw_l))

    games_today[["p_AH_Home_Win","p_AH_Home_Push","p_AH_Home_Lose"]] = pd.DataFrame(home_probs, index=games_today.index)
    games_today[["p_AH_Away_Win","p_AH_Away_Push","p_AH_Away_Lose"]] = pd.DataFrame(away_probs, index=games_today.index)
    games_today = _add_fair_odds(games_today, [
        "p_AH_Home_Win","p_AH_Home_Push","p_AH_Home_Lose",
        "p_AH_Away_Win","p_AH_Away_Push","p_AH_Away_Lose"
    ])

    # ---- Exibição HOME ----
    st.markdown("#### 🏠 Home – Probabilidades AH (Win/Push/Lose) + Fair Odds")
    cols_home = ["Home","Away","Goals_H_Today","Goals_A_Today","Asian_Line_Home_Display","XG2_H","XG2_A",
                 "p_AH_Home_Win","p_AH_Home_Push","p_AH_Home_Lose"]
    #,"p_AH_Home_Win_FairOdd","p_AH_Home_Push_FairOdd","p_AH_Home_Lose_FairOdd"
    cols_home = [c for c in cols_home if c in games_today.columns]
    fmt = {
        "Asian_Line_Home_Display": "{:+.2f}",
        "XG2_H": "{:.2f}","XG2_A": "{:.2f}",
        "Goals_H_Today": "{:.0f}","Goals_A_Today": "{:.0f}",
        "p_AH_Home_Win": "{:.1%}","p_AH_Home_Push": "{:.1%}","p_AH_Home_Lose": "{:.1%}",
        "p_AH_Home_Win_FairOdd": "{:.2f}","p_AH_Home_Push_FairOdd": "{:.2f}","p_AH_Home_Lose_FairOdd": "{:.2f}"
    }
    st.dataframe(
        games_today[cols_home]
        .style.format(fmt)
        .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_AH_Home_Win"])
        .applymap(lambda v: color_prob(v, "255,255,0"), subset=["p_AH_Home_Push"])
        .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_AH_Home_Lose"]),
        use_container_width=True, height=520
    )

else:
    st.warning("⚠️ Para cálculo Poisson AH é necessário ter XG2_H e XG2_A.")


########################################
#### BLOCO 8.7 – AH (Skellam via xG ajustado por Handicap) – versão final corrigida
########################################
from scipy.stats import skellam
import math

st.markdown("#### 🏠 Home – Probabilidades AH (Skellam via xG ajustado por Handicap)")

def adjust_xg_for_handicap(xg_home: float, line_home: float):
    """Aplica o handicap diretamente no xG do Home."""
    if pd.isna(xg_home) or pd.isna(line_home):
        return np.nan
    if line_home < 0:  # favorito
        return xg_home - abs(line_home)
    else:  # zebra
        return xg_home + abs(line_home)

def skellam_probs(mu_h, mu_a):
    """Probabilidades via Skellam (Home>0, Home=0, Home<0) com estabilidade numérica."""
    try:
        mu_h = float(np.clip(mu_h, 0.01, 5.0))
        mu_a = float(np.clip(mu_a, 0.01, 5.0))
        p_win = 1 - skellam.cdf(0, mu_h, mu_a)
        p_push = skellam.pmf(0, mu_h, mu_a)
        p_lose = 1 - p_win - p_push
        if np.isnan(p_win) or np.isnan(p_push) or np.isnan(p_lose):
            return 0.0, 0.0, 0.0
        return p_win, p_push, p_lose
    except Exception:
        return 0.0, 0.0, 0.0

def fair_odds(p):
    return (1/p) if (p and p > 0) else np.nan

# Garante coluna da linha do HOME
if "Asian_Line_Home_Display" not in games_today.columns:
    if "Asian_Line_Away_Display" in games_today.columns:
        games_today["Asian_Line_Home_Display"] = -games_today["Asian_Line_Away_Display"]
    else:
        games_today["Asian_Line_Home_Display"] = np.nan

rows = []
for idx, r in games_today.iterrows():
    try:
        xh = float(r.get("XG2_H", np.nan))
        xa = float(r.get("XG2_A", np.nan))
        Lh = float(np.clip(r.get("Asian_Line_Home_Display", np.nan), -3, 3))
    except Exception:
        rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    if pd.isna(xh) or pd.isna(xa) or pd.isna(Lh):
        rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    # ✅ Ajusta xG do Home e aplica limite mínimo de 0.01
    xh_hand = adjust_xg_for_handicap(xh, Lh)
    xh_hand = max(xh_hand, 0.01)

    # Probabilidades via Skellam
    pW, pP, pL = skellam_probs(xh_hand, xa)

    # Melhor lado
    best_side = "BackHome" if pW > 0.50 else "BackAway"

    rows.append((xh_hand, pW, pP, pL, fair_odds(pW), best_side))

games_today[["XG2_H_Hand","p_AH_Home_Win_Sk","p_AH_Home_Push_Sk",
             "p_AH_Home_Lose_Sk","p_AH_Home_Win_Sk_FairOdd","Best_Side"]] = pd.DataFrame(rows, index=games_today.index)

cols_home_sk = [
    "Home","Away","Goals_H_Today","Goals_A_Today","Asian_Line_Home_Display",
    "XG2_H","XG2_A","XG2_H_Hand",
    "p_AH_Home_Win_Sk","p_AH_Home_Push_Sk","p_AH_Home_Lose_Sk",
    "Best_Side"
]
cols_home_sk = [c for c in cols_home_sk if c in games_today.columns]

fmt_sk = {
    "Asian_Line_Home_Display": "{:+.2f}",
    "Goals_H_Today": "{:.0f}","Goals_A_Today": "{:.0f}",
    "XG2_H": "{:.2f}","XG2_A": "{:.2f}","XG2_H_Hand": "{:.2f}",
    "p_AH_Home_Win_Sk": "{:.1%}","p_AH_Home_Push_Sk": "{:.1%}","p_AH_Home_Lose_Sk": "{:.1%}",
    "p_AH_Home_Win_Sk_FairOdd": "{:.2f}"
}

st.dataframe(
    games_today[cols_home_sk]
    .style.format(fmt_sk)
    .applymap(lambda v: "background-color: rgba(0,200,0,0.2);" if isinstance(v,str) and v=="BackHome" else
                        ("background-color: rgba(255,100,100,0.2);" if isinstance(v,str) and v=="BackAway" else ""), subset=["Best_Side"]),
    use_container_width=True, height=520
)

st.caption("xG ajustado: XG2_H_Hand = XG2_H ± |Handicap| (negativo subtrai, positivo soma). Probabilidades calculadas via Skellam (Home>0, =0, <0). Linhas extremas são limitadas entre -3 e +3 para estabilidade numérica.")



########################################
#### BLOCO 8.8 – AH (Skellam via xG ajustado por Handicap) – Versão Away
########################################
from scipy.stats import skellam
import math

st.markdown("#### 🛫 Away – Probabilidades AH (Skellam via xG ajustado por Handicap)")

def adjust_xg_for_handicap_away(xg_away: float, line_away: float):
    """Aplica o handicap diretamente no xG do Away."""
    if pd.isna(xg_away) or pd.isna(line_away):
        return np.nan
    if line_away < 0:  # Away é favorito (dá gols)
        return xg_away - abs(line_away)
    else:  # Away é zebra (recebe gols)
        return xg_away + abs(line_away)

def skellam_probs_away(mu_a, mu_h):
    """Probabilidades para o Away vencer, empatar, perder (Skellam simétrico)."""
    try:
        mu_a = float(np.clip(mu_a, 0.01, 5.0))
        mu_h = float(np.clip(mu_h, 0.01, 5.0))
        p_win = 1 - skellam.cdf(0, mu_a, mu_h)
        p_push = skellam.pmf(0, mu_a, mu_h)
        p_lose = 1 - p_win - p_push
        if np.isnan(p_win) or np.isnan(p_push) or np.isnan(p_lose):
            return 0.0, 0.0, 0.0
        return p_win, p_push, p_lose
    except Exception:
        return 0.0, 0.0, 0.0

def fair_odds(p):
    return (1/p) if (p and p > 0) else np.nan

# Garante coluna da linha do AWAY
if "Asian_Line_Away_Display" not in games_today.columns:
    if "Asian_Line_Home_Display" in games_today.columns:
        games_today["Asian_Line_Away_Display"] = -games_today["Asian_Line_Home_Display"]
    else:
        games_today["Asian_Line_Away_Display"] = np.nan

rows = []
for idx, r in games_today.iterrows():
    try:
        xa = float(r.get("XG2_A", np.nan))
        xh = float(r.get("XG2_H", np.nan))
        La = float(np.clip(r.get("Asian_Line_Away_Display", np.nan), -3, 3))
    except Exception:
        rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    if pd.isna(xa) or pd.isna(xh) or pd.isna(La):
        rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    # ✅ Ajusta xG do Away e aplica limite mínimo de 0.01
    xa_hand = adjust_xg_for_handicap_away(xa, La)
    xa_hand = max(xa_hand, 0.01)

    # Probabilidades via Skellam (agora Away como "mu1")
    pW, pP, pL = skellam_probs_away(xa_hand, xh)

    # Melhor lado
    best_side = "BackAway" if pW > 0.5 else "BackHome"

    rows.append((xa_hand, pW, pP, pL, fair_odds(pW), best_side))

games_today[["XG2_A_Hand","p_AH_Away_Win_Sk","p_AH_Away_Push_Sk",
             "p_AH_Away_Lose_Sk","p_AH_Away_Win_Sk_FairOdd","Best_Side_Away"]] = pd.DataFrame(rows, index=games_today.index)

cols_away_sk = [
    "Home","Away",
    "Goals_H_Today","Goals_A_Today",
    "Asian_Line_Away_Display",
    "XG2_H","XG2_A","XG2_A_Hand",
    "p_AH_Away_Win_Sk","p_AH_Away_Push_Sk","p_AH_Away_Lose_Sk",
    "Best_Side_Away"
]
cols_away_sk = [c for c in cols_away_sk if c in games_today.columns]

fmt_sk = {
    "Asian_Line_Away_Display": "{:+.2f}",
    "Goals_H_Today": "{:.0f}","Goals_A_Today": "{:.0f}",
    "XG2_H": "{:.2f}","XG2_A": "{:.2f}","XG2_A_Hand": "{:.2f}",
    "p_AH_Away_Win_Sk": "{:.1%}","p_AH_Away_Push_Sk": "{:.1%}","p_AH_Away_Lose_Sk": "{:.1%}",
    "p_AH_Away_Win_Sk_FairOdd": "{:.2f}"
}

st.dataframe(
    games_today[cols_away_sk]
    .style.format(fmt_sk)
    .applymap(lambda v: "background-color: rgba(0,200,0,0.2);" if isinstance(v,str) and v=="BackAway" else
                        ("background-color: rgba(255,100,100,0.2);" if isinstance(v,str) and v=="BackHome" else ""), subset=["Best_Side_Away"]),
    use_container_width=True, height=520
)

st.caption("xG ajustado: XG2_A_Hand = XG2_A ± |Handicap| (negativo subtrai, positivo soma). Probabilidades calculadas via Skellam (Away>0, =0, <0). Linhas extremas são limitadas entre -3 e +3 para estabilidade numérica.")




