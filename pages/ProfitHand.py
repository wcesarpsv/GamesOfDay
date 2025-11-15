from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# ========================= CONFIG STREAMLIT =========================
st.set_page_config(page_title="Análise de Quadrantes 3D - Bet Indicator", layout="wide")
st.title("🎯 Análise 3D de 16 Quadrantes - ML Inteligente (Home & Away)")

# ========================= CONFIGURAÇÕES GERAIS =========================
PAGE_PREFIX = "QuadrantesML_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "coppa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# =====================================================
# Helpers básicos
# =====================================================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        return pd.DataFrame()
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df_tmp = pd.read_csv(os.path.join(folder, f))
            df_tmp = preprocess_df(df_tmp)
            dfs.append(df_tmp)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

# =====================================================
# Asian Line
# =====================================================
def convert_asian_line_to_decimal(line_str):
    if pd.isna(line_str) or line_str == "":
        return None
    line_str = str(line_str).strip()
    if line_str in ["0", "0.0"]:
        return 0.0

    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num
        except ValueError:
            return None

    try:
        parts = [float(p) for p in line_str.split("/")]
        avg = sum(parts) / len(parts)
        first_part = parts[0]
        if first_part < 0:
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result
    except (ValueError, TypeError):
        return None

def calc_handicap_result(margin, asian_line_decimal, invert=False):
    if pd.isna(asian_line_decimal) or pd.isna(margin):
        return np.nan
    if invert:
        margin = -margin
    if margin > asian_line_decimal:
        return 1.0
    elif margin == asian_line_decimal:
        return 0.5
    else:
        return 0.0

# =====================================================
# Cálculo de M / MT a partir do HandScore
# =====================================================
def calcular_zscores_detalhados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Z-scores a partir do HandScore:
    - M_H, M_A: Z-score do time em relação à liga (performance relativa)
    - MT_H, MT_A: Z-score do time em relação a si mesmo (consistência)
    """
    df = df.copy()
    st.info("📊 Calculando Z-scores (M, MT) a partir do HandScore...")

    # 1) Z-score por liga
    if {'League', 'HandScore_Home', 'HandScore_Away'}.issubset(df.columns):
        league_stats = df.groupby('League').agg({
            'HandScore_Home': ['mean', 'std'],
            'HandScore_Away': ['mean', 'std']
        }).round(3)
        league_stats.columns = ['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std']
        league_stats['HS_H_std'] = league_stats['HS_H_std'].replace(0, 1)
        league_stats['HS_A_std'] = league_stats['HS_A_std'].replace(0, 1)

        df = df.merge(league_stats, on='League', how='left')
        df['M_H'] = (df['HandScore_Home'] - df['HS_H_mean']) / df['HS_H_std']
        df['M_A'] = (df['HandScore_Away'] - df['HS_A_mean']) / df['HS_A_std']

        df['M_H'] = np.clip(df['M_H'], -5, 5)
        df['M_A'] = np.clip(df['M_A'], -5, 5)
    else:
        st.warning("⚠️ League ou HandScore_Home/HandScore_Away ausentes - M_H/M_A = 0")
        df['M_H'] = 0.0
        df['M_A'] = 0.0

    # 2) Z-score por time (consistência)
    if {'Home', 'Away', 'HandScore_Home', 'HandScore_Away'}.issubset(df.columns):
        home_team_stats = df.groupby('Home').agg({'HandScore_Home': ['mean', 'std']}).round(3)
        home_team_stats.columns = ['HT_mean', 'HT_std']
        away_team_stats = df.groupby('Away').agg({'HandScore_Away': ['mean', 'std']}).round(3)
        away_team_stats.columns = ['AT_mean', 'AT_std']

        home_team_stats['HT_std'] = home_team_stats['HT_std'].replace(0, 1)
        away_team_stats['AT_std'] = away_team_stats['AT_std'].replace(0, 1)

        df = df.merge(home_team_stats, left_on='Home', right_index=True, how='left')
        df = df.merge(away_team_stats, left_on='Away', right_index=True, how='left')

        df['MT_H'] = (df['HandScore_Home'] - df['HT_mean']) / df['HT_std']
        df['MT_A'] = (df['HandScore_Away'] - df['AT_mean']) / df['AT_std']

        df['MT_H'] = np.clip(df['MT_H'], -5, 5)
        df['MT_A'] = np.clip(df['MT_A'], -5, 5)

        df = df.drop(['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std',
                      'HT_mean', 'HT_std', 'AT_mean', 'AT_std'], axis=1, errors='ignore')
    else:
        st.warning("⚠️ Home/Away ou HandScore_Home/HandScore_Away ausentes - MT_H/MT_A = 0")
        df['MT_H'] = 0.0
        df['MT_A'] = 0.0

    df[['M_H', 'M_A', 'MT_H', 'MT_A']] = df[['M_H', 'M_A', 'MT_H', 'MT_A']].fillna(0)
    st.success("✅ M_H, M_A, MT_H, MT_A calculados e normalizados")
    return df

def clean_features_for_training(X: pd.DataFrame) -> pd.DataFrame:
    X_clean = X.copy()
    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean)

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    nan_count = X_clean.isna().sum().sum()
    if nan_count > 0:
        st.warning(f"⚠️ {nan_count} NaNs encontrados nas features - preenchendo com mediana/0")

    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)

    for col in X_clean.columns:
        if X_clean[col].dtype in [np.float64, np.float32, float]:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower, upper)

    X_clean = X_clean.replace([np.inf, -np.inf], 0).fillna(0)
    st.success(f"✅ Features limpas para treino: {X_clean.shape}")
    return X_clean

# =====================================================
# Clusterização 3D
# =====================================================
def aplicar_clusterizacao_3d(df, max_clusters=5, random_state=42):
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        df['Cluster3D_Desc'] = 'Dados Insuficientes'
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()
    n_samples = len(X_cluster)

    if n_samples < 2:
        df['Cluster3D_Label'] = 0
        df['Cluster3D_Desc'] = 'Amostra Única'
        return df

    n_clusters = min(max_clusters, max(2, n_samples // 3))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init='k-means++',
        n_init=min(10, n_samples)
    )
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
    centroids['Cluster'] = range(n_clusters)
    centroids['Tamanho'] = [sum(df['Cluster3D_Label'] == i) for i in range(n_clusters)]

    def classificar_cluster(dx, dy, dz):
        if abs(dx) > 0.5 and abs(dy) > 1.0 and abs(dz) > 1.0:
            return '🔥 Alta Variância 3D'
        elif dx > 0.3 and dy > 0.5:
            return '⚡ Home Dominante + Momentum'
        elif dx < -0.3 and dy < -0.5:
            return '⚡ Away Dominante + Momentum'
        elif abs(dx) < 0.2 and abs(dy) < 0.3 and abs(dz) < 0.3:
            return '⚖️ Equilibrado'
        elif dy > 0.8 or dz > 0.8:
            return '📈 Momentum Positivo'
        elif dy < -0.8 or dz < -0.8:
            return '📉 Momentum Negativo'
        else:
            return '🌀 Padrão Misto'

    cluster_descriptions = {}
    for i in range(n_clusters):
        centroid = centroids.iloc[i]
        cluster_descriptions[i] = classificar_cluster(centroid['dx'], centroid['dy'], centroid['dz'])

    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(cluster_descriptions)
    return df

# =====================================================
# Cálculo de regressão à média
# =====================================================
def calcular_regressao_media(df):
    df = df.copy()
    for col in ['M_H', 'M_A', 'MT_H', 'MT_A', 'Aggression_Home', 'Aggression_Away']:
        if col not in df.columns:
            df[col] = 0.0

    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])

    df['Regressao_Force_Home'] = -np.sign(df['M_H']) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df['M_A']) * (df['Extremidade_Away'] ** 0.7)

    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))

    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] +
                              0.4 * (1 - df['Aggression_Home'].fillna(0)))
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] +
                              0.4 * (1 - df['Aggression_Away'].fillna(0)))

    conditions_home = [
        df['Regressao_Force_Home'] > 1.0,
        df['Regressao_Force_Home'] > 0.3,
        df['Regressao_Force_Home'] > -0.3,
        df['Regressao_Force_Home'] > -1.0,
        df['Regressao_Force_Home'] <= -1.0
    ]
    choices_home = ['📈 FORTE MELHORA', '📈 MELHORA', '⚖️ ESTÁVEL', '📉 QUEDA', '📉 FORTE QUEDA']
    df['Tendencia_Home'] = np.select(conditions_home, choices_home, default='⚖️ ESTÁVEL')

    conditions_away = [
        df['Regressao_Force_Away'] > 1.0,
        df['Regressao_Force_Away'] > 0.3,
        df['Regressao_Force_Away'] > -0.3,
        df['Regressao_Force_Away'] > -1.0,
        df['Regressao_Force_Away'] <= -1.0
    ]
    choices_away = ['📈 FORTE MELHORA', '📈 MELHORA', '⚖️ ESTÁVEL', '📉 QUEDA', '📉 FORTE QUEDA']
    df['Tendencia_Away'] = np.select(conditions_away, choices_away, default='⚖️ ESTÁVEL')
    return df

# =====================================================
# Distâncias 3D
# =====================================================
def calcular_distancias_3d(df):
    df = df.copy()
    for col in ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']:
        if col not in df.columns:
            df[col] = 0.0

    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    angle_xy = np.arctan2(dy, dx)
    angle_xz = np.arctan2(dz, dx)
    angle_yz = np.arctan2(dz, dy)

    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Angle_XZ'] = np.degrees(angle_xz)
    df['Quadrant_Angle_YZ'] = np.degrees(angle_yz)

    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Quadrant_Sin_XZ'] = np.sin(angle_xz)
    df['Quadrant_Cos_XZ'] = np.cos(angle_xz)
    df['Quadrant_Sin_YZ'] = np.sin(angle_yz)
    df['Quadrant_Cos_YZ'] = np.cos(angle_yz)

    df['Quadrant_Sin_Combo'] = np.sin(angle_xy + angle_xz + angle_yz)
    df['Quadrant_Cos_Combo'] = np.cos(angle_xy + angle_xz + angle_yz)

    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df

# =====================================================
# Sistema 16 quadrantes
# =====================================================
QUADRANTES_16 = {
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte",       "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado",    "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro",      "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},
    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte",       "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado",    "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro",      "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},
    9: {"nome": "Under Moderado Neutro",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},
    13: {"nome": "Under Forte Neutro",    "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",  "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",     "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for qid, cfg in QUADRANTES_16.items():
        if cfg['agg_min'] <= agg <= cfg['agg_max'] and cfg['hs_min'] <= hs <= cfg['hs_max']:
            return qid
    return 0

# =====================================================
# Carregar dados
# =====================================================
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

if not os.path.exists(GAMES_FOLDER):
    st.error(f"Pasta {GAMES_FOLDER} não encontrada.")
    st.stop()

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("Nenhum CSV encontrado na pasta GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = preprocess_df(games_today)
    games_today = filter_leagues(games_today)

    history = filter_leagues(load_all_games(GAMES_FOLDER))
    if not history.empty:
        history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"], how="any")
    return games_today, history

games_today, history = load_cached_data(selected_file)

if games_today.empty:
    st.error("CSV do dia carregado, mas sem dados após filtros.")
    st.stop()

# =====================================================
# LiveScore merge
# =====================================================
def load_and_merge_livescore(games_today, selected_date_str):
    games_today = setup_livescore_columns(games_today)
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    if not os.path.exists(livescore_file):
        return games_today
    try:
        results_df = pd.read_csv(livescore_file)
    except Exception:
        return games_today

    if 'status' not in results_df.columns:
        return games_today
    results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

    required_cols = [
        'Id', 'status', 'home_goal', 'away_goal',
        'home_ht_goal', 'away_ht_goal',
        'home_corners', 'away_corners',
        'home_yellow', 'away_yellow',
        'home_red', 'away_red'
    ]
    missing_cols = [c for c in required_cols if c not in results_df.columns]
    if missing_cols:
        return games_today

    games_today = games_today.merge(
        results_df,
        left_on='Id',
        right_on='Id',
        how='left',
        suffixes=('', '_RAW')
    )

    games_today['Goals_H_Today'] = games_today['home_goal']
    games_today['Goals_A_Today'] = games_today['away_goal']
    games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

    games_today['Home_Red'] = games_today['home_red']
    games_today['Away_Red'] = games_today['away_red']

    return games_today

games_today = load_and_merge_livescore(games_today, selected_date_str)

# =====================================================
# Asian Line decimal + Anti-leak
# =====================================================
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

history = history.dropna(subset=['Asian_Line_Decimal'])
st.info(f"📊 Histórico com Asian Line válida: {len(history)} jogos")

if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

# =====================================================
# Target AH Home
# =====================================================
# ---------------- TARGET = PROFIT PURO DO HANDICAP (HOME & AWAY) ----------------
import math

def split_asian_line(line: float):
    """
    Quebra uma linha asiática em duas metades quando for 0.25 ou 0.75.
    Ex:
      +0.25 -> [0.0, +0.5]
      -0.25 -> [0.0, -0.5]
      +0.75 -> [+0.5, +1.0]
      -0.75 -> [-0.5, -1.0]
    Linhas normais (0.0, 0.5, 1.0, 1.5, ...) voltam como [line]
    """
    if pd.isna(line):
        return []

    frac = abs(line) % 1
    sign = 1 if line >= 0 else -1
    base = math.floor(abs(line))

    if frac == 0.25:
        # 0.25 → metade em inteiro, metade em +0.5
        l1 = sign * base
        l2 = sign * (base + 0.5)
        return [l1, l2]
    elif frac == 0.75:
        # 0.75 → metade em +0.5, metade em +1.0
        l1 = sign * (base + 0.5)
        l2 = sign * (base + 1.0)
        return [l1, l2]
    else:
        return [line]

def calc_profit_home_unit(margin: float, line: float) -> float:
    """
    Lucro unitário (stake=1) para o HOME na linha asiática já convertida p/ Home (Asian_Line_Decimal).
    Retorna valor entre -1 e +1:
      +1   = win cheio
      +0.5 = half win
       0   = push
      -0.5 = half loss
      -1   = loss cheio
    """
    if pd.isna(margin) or pd.isna(line):
        return 0.0

    lines = split_asian_line(line)
    if not lines:
        return 0.0

    profits = []
    for l in lines:
        adj = margin + l  # GH - GA + linha_home
        if adj > 0:
            profits.append(1.0)
        elif adj == 0:
            profits.append(0.0)
        else:
            profits.append(-1.0)

    return float(np.mean(profits))

# 🔹 Margin FT
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]

# 🔹 Profit puro Home (na perspectiva do handicap da casa)
history["Target_Profit_Home"] = history.apply(
    lambda r: calc_profit_home_unit(r["Margin"], r["Asian_Line_Decimal"]), axis=1
)

# 🔹 Profit puro Away é sempre o oposto (mesma linha, lado contrário)
history["Target_Profit_Away"] = -history["Target_Profit_Home"]

st.info(
    f"🎯 Targets de lucro gerados: "
    f"Home ∈ [{history['Target_Profit_Home'].min():.1f}, {history['Target_Profit_Home'].max():.1f}], "
    f"Away espelhado."
)


# =====================================================
# Calcular M/MT usando history + games_today juntos
# =====================================================
history['_is_history'] = 1
games_today['_is_history'] = 0
df_all = pd.concat([history, games_today], ignore_index=True)
df_all = calcular_zscores_detalhados(df_all)

history = df_all[df_all['_is_history'] == 1].drop(columns=['_is_history'])
games_today = df_all[df_all['_is_history'] == 0].drop(columns=['_is_history'])

# =====================================================
# Regressão à média + 3D + clusters
# =====================================================
history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)

history = calcular_distancias_3d(history)
games_today = calcular_distancias_3d(games_today)

history = aplicar_clusterizacao_3d(history)
games_today = aplicar_clusterizacao_3d(games_today)

# Classificar quadrantes
games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)
history['Quadrante_Home'] = history.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
history['Quadrante_Away'] = history.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)

# =====================================================
# Features inteligentes + modelo
# =====================================================
def adicionar_features_inteligentes_ml(df):
    df = df.copy()

    for col in ['Regressao_Force_Home', 'Regressao_Force_Away',
                'Tendencia_Home', 'Tendencia_Away',
                'Media_Score_Home', 'Media_Score_Away',
                'Aggression_Home', 'Aggression_Away']:
        if col not in df.columns:
            if col.startswith('Tendencia'):
                df[col] = '⚖️ ESTÁVEL'
            else:
                df[col] = 0.0

    df['eh_fav_forte_com_momentum'] = (
        (df['Quadrante_Home'].isin([1, 2, 3, 4])) &
        (df['M_H'] > 0.5) &
        (df['Regressao_Force_Home'] > 0)
    ).astype(int)

    df['eh_under_forte_sem_momentum'] = (
        (df['Quadrante_Away'].isin([13, 14, 15, 16])) &
        (df['M_A'] < -0.5) &
        (df['Regressao_Force_Away'] < 0)
    ).astype(int)

    df['eh_forte_melhora_home'] = (df['Tendencia_Home'] == '📈 FORTE MELHORA').astype(int)
    df['eh_forte_melhora_away'] = (df['Tendencia_Away'] == '📈 FORTE MELHORA').astype(int)
    df['eh_forte_queda_home'] = (df['Tendencia_Home'] == '📉 FORTE QUEDA').astype(int)
    df['eh_forte_queda_away'] = (df['Tendencia_Away'] == '📉 FORTE QUEDA').astype(int)

    df['conflito_agg_regressao_home'] = (
        (df['Aggression_Home'] > 0.3) &
        (df['Regressao_Force_Home'] < -0.8)
    ).astype(int)

    df['conflito_agg_regressao_away'] = (
        (df['Aggression_Away'] < -0.3) &
        (df['Regressao_Force_Away'] < -0.8)
    ).astype(int)

    df['momentum_confirma_home'] = (
        (df['Aggression_Home'] > 0.3) &
        (df['M_H'] > 0) &
        (df['Regressao_Force_Home'] > 0)
    ).astype(int)

    df['momentum_confirma_away'] = (
        (df['Aggression_Away'] < -0.3) &
        (df['M_A'] > 0) &
        (df['Regressao_Force_Away'] > 0)
    ).astype(int)

    df['momentum_negativo_alarmante_home'] = (
        (df['M_H'] < -1.0) &
        (df['Regressao_Force_Home'] < -0.5)
    ).astype(int)

    df['momentum_negativo_alarmante_away'] = (
        (df['M_A'] < -1.0) &
        (df['Regressao_Force_Away'] < -0.5)
    ).astype(int)

    df['padrao_fav_forte_vs_under_forte'] = (
        (df['Quadrante_Home'].isin([1, 2, 3, 4])) &
        (df['Quadrante_Away'].isin([13, 14, 15, 16]))
    ).astype(int)

    df['padrao_fav_moderado_vs_under_moderado'] = (
        (df['Quadrante_Home'].isin([5, 6, 7, 8])) &
        (df['Quadrante_Away'].isin([9, 10, 11, 12]))
    ).astype(int)

    aggression_proxy_home = (df['Aggression_Home'] + 1) / 2
    aggression_proxy_away = (1 - (df['Aggression_Away'] + 1) / 2)

    df['score_confianca_composto'] = (
        (aggression_proxy_home * 0.3) +
        (aggression_proxy_away * 0.3) +
        (df['Media_Score_Home'] * 0.2) +
        (df['Media_Score_Away'] * 0.2)
    )

    return df

def treinar_modelo_inteligente(history, games_today):
    """
    Treina modelo ML para prever LUCRO ESPERADO do handicap:
      - Target_Profit_Home (stake=1 u)
      - Target_Profit_Away (espelhado)
    E gera:
      - Pred_Profit_Home, Pred_Profit_Away  (EV em unidades)
      - EV_Main (melhor lado)
      - Quadrante_ML_Score_* em formato "prob-like" (0–1) para reaproveitar o resto do sistema
    """
    # ---------------- 1) Features inteligentes + 3D + clusters ----------------
    history = adicionar_features_inteligentes_ml(history)
    games_today = adicionar_features_inteligentes_ml(games_today)

    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)

    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    # ---------------- 2) One-hot de liga e cluster ----------------
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D'
    ]

    features_regressao = [
        'Media_Score_Home', 'Media_Score_Away',
        'Regressao_Force_Home', 'Regressao_Force_Away',
        'Extremidade_Home', 'Extremidade_Away'
    ]

    features_inteligentes = [
        'eh_fav_forte_com_momentum', 'eh_under_forte_sem_momentum',
        'eh_forte_melhora_home', 'eh_forte_melhora_away',
        'eh_forte_queda_home', 'eh_forte_queda_away',
        'conflito_agg_regressao_home', 'conflito_agg_regressao_away',
        'momentum_confirma_home', 'momentum_confirma_away',
        'momentum_negativo_alarmante_home', 'momentum_negativo_alarmante_away',
        'padrao_fav_forte_vs_under_forte', 'padrao_fav_moderado_vs_under_moderado',
        'score_confianca_composto'
    ]

    available_3d = [f for f in features_3d if f in history.columns]
    available_regressao = [f for f in features_regressao if f in history.columns]
    available_inteligentes = [f for f in features_inteligentes if f in history.columns]

    extras_3d = history[available_3d].fillna(0)
    extras_regressao = history[available_regressao].fillna(0)
    extras_inteligentes = history[available_inteligentes].fillna(0)

    X = pd.concat(
        [ligas_dummies, clusters_dummies, extras_3d, extras_regressao, extras_inteligentes],
        axis=1
    )

    # ---------------- 3) Targets de lucro puro ----------------
    y_home = history['Target_Profit_Home'].fillna(0.0).astype(float)
    y_away = history['Target_Profit_Away'].fillna(0.0).astype(float)

    # ---------------- 4) Limpeza de features ----------------
    X = clean_features_for_training(X)

    # ---------------- 5) Modelos de regressão (Home & Away) ----------------
    model_home = RandomForestRegressor(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=40,
        random_state=42,
        n_jobs=-1
    )

    model_away = RandomForestRegressor(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=40,
        random_state=42,
        n_jobs=-1
    )

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # ---------------- 6) Preparar dados de hoje ----------------
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(
        columns=ligas_dummies.columns, fill_value=0
    )
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(
        columns=clusters_dummies.columns, fill_value=0
    )
    extras_today_3d = games_today[available_3d].fillna(0)
    extras_today_reg = games_today[available_regressao].fillna(0)
    extras_today_int = games_today[available_inteligentes].fillna(0)

    X_today = pd.concat(
        [ligas_today, clusters_today, extras_today_3d, extras_today_reg, extras_today_int],
        axis=1
    )
    X_today = clean_features_for_training(X_today)

    # ---------------- 7) Predição de lucro esperado ----------------
    pred_profit_home = model_home.predict(X_today)  # EV em unidades
    pred_profit_away = model_away.predict(X_today)

    games_today['Pred_Profit_Home'] = pred_profit_home
    games_today['Pred_Profit_Away'] = pred_profit_away

    # Melhor lado (EV maior)
    games_today['Best_Side'] = np.where(
        games_today['Pred_Profit_Home'] >= games_today['Pred_Profit_Away'],
        'HOME', 'AWAY'
    )

    games_today['EV_Home'] = games_today['Pred_Profit_Home']
    games_today['EV_Away'] = games_today['Pred_Profit_Away']
    games_today['EV_Main'] = np.where(
        games_today['Best_Side'] == 'HOME',
        games_today['EV_Home'],
        games_today['EV_Away']
    )

    # Para reaproveitar o resto do sistema (que espera "probabilidades" 0–1)
    # mapeamos EV ∈ [-1, +1] para Score ∈ [0, 1]
    games_today['Quadrante_ML_Score_Home'] = 0.5 + games_today['Pred_Profit_Home'] / 2
    games_today['Quadrante_ML_Score_Away'] = 0.5 + games_today['Pred_Profit_Away'] / 2
    games_today['Quadrante_ML_Score_Main'] = games_today[
        ['Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away']
    ].max(axis=1)

    # ML_Side segue o melhor lado por EV
    games_today['ML_Side'] = games_today['Best_Side']
    games_today['ML_Confidence'] = games_today['Quadrante_ML_Score_Main']

    # ---------------- 8) Importância das features (modelo Home) ----------------
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features (Profit Handicap)")
    st.dataframe(importances.head(20).to_frame("Importância"), use_container_width=True)

    st.info(
        f"EV médio Home: {games_today['EV_Home'].mean():.3f} u | "
        f"EV médio Away: {games_today['EV_Away'].mean():.3f} u"
    )

    st.success("✅ Modelo Inteligente de LUCRO treinado com sucesso!")
    return model_home, model_away, games_today


# =====================================================
# Treinar modelo com history e aplicar nos jogos do dia
# =====================================================
if history.empty:
    st.error("Histórico vazio após filtros - não foi possível treinar o modelo.")
    st.stop()

st.markdown("### 🤖 Treinando modelo 3D Inteligente com histórico filtrado...")
modelo_home, modelo_away, games_today = treinar_modelo_inteligente(history, games_today)
st.success("✅ Modelo 3D Inteligente treinado e aplicado nos jogos de hoje!")

# =====================================================
# Plot 2D dos quadrantes
# =====================================================
def plot_quadrantes_16(df, side="Home"):
    fig, ax = plt.subplots(figsize=(12, 8))
    for quadrante_id in range(1, 17):
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(x, y, alpha=0.7, s=40, label=QUADRANTES_16[quadrante_id]['nome'])
    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    for y in [-45, -30, -15, 15, 30, 45]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel(f'Aggression_{side} (-1 zebra ↔ +1 favorito)')
    ax.set_ylabel(f'HandScore_{side} (-60 a +60)')
    ax.set_title(f'16 Quadrantes - {side} (Visão 2D)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

st.markdown("### 📈 Visualização dos 16 Quadrantes (2D)")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

# =====================================================
# Visualização 3D (mantendo sua lógica principal)
# =====================================================
st.markdown("## 🎯 Visualização Interativa 3D – Tamanho Fixo")

if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())
    selected_league = st.selectbox(
        "Selecione a liga para análise:",
        options=["⚽ Todas as ligas"] + leagues,
        index=0
    )
    if selected_league != "⚽ Todas as ligas":
        df_filtered = games_today[games_today["League"] == selected_league].copy()
    else:
        df_filtered = games_today.copy()
else:
    st.warning("⚠️ Nenhuma coluna de 'League' encontrada — exibindo todos os jogos.")
    df_filtered = games_today.copy()

max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância 3D):", 10, min(max_n, 200), 40, step=5)

def create_fixed_3d_plot(df_plot, n_to_show, selected_league):
    fig_3d = go.Figure()
    X_RANGE = [-1.2, 1.2]
    Y_RANGE = [-4.0, 4.0]
    Z_RANGE = [-4.0, 4.0]

    df_plot = df_plot.copy()
    if 'Quadrant_Dist_3D' in df_plot.columns:
        df_plot = df_plot.sort_values('Quadrant_Dist_3D', ascending=False).head(n_to_show)

    for _, row in df_plot.iterrows():
        xh = row.get("Aggression_Home", 0) or 0
        yh = row.get("M_H", 0) if not pd.isna(row.get("M_H")) else 0
        zh = row.get("MT_H", 0) if not pd.isna(row.get("MT_H")) else 0

        xa = row.get("Aggression_Away", 0) or 0
        ya = row.get("M_A", 0) if not pd.isna(row.get("M_A")) else 0
        za = row.get("MT_A", 0) if not pd.isna(row.get("MT_A")) else 0

        if all(v == 0 for v in [xh, yh, zh, xa, ya, za]):
            continue

        fig_3d.add_trace(go.Scatter3d(
            x=[xh, xa],
            y=[yh, ya],
            z=[zh, za],
            mode='lines+markers',
            line=dict(color='gray', width=4),
            marker=dict(size=5),
            hoverinfo='text',
            hovertext=(
                f"<b>{row.get('Home','N/A')} vs {row.get('Away','N/A')}</b><br>"
                f"🏆 {row.get('League','N/A')}<br>"
                f"📏 Dist 3D: {row.get('Quadrant_Dist_3D', np.nan):.2f}<br>"
                f"📍 Agg_H: {xh:.2f} | Agg_A: {xa:.2f}<br>"
                f"⚙️ M_H: {row.get('M_H', np.nan):.2f} | M_A: {row.get('M_A', np.nan):.2f}<br>"
                f"🔥 MT_H: {row.get('MT_H', np.nan):.2f} | MT_A: {row.get('MT_A', np.nan):.2f}"
            ),
            showlegend=False
        ))

    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Home"],
        y=df_plot["M_H"],
        z=df_plot["MT_H"],
        mode='markers+text',
        name='Home',
        marker=dict(
            color='royalblue',
            size=8,
            opacity=0.9,
            symbol='circle',
            line=dict(color='darkblue', width=2)
        ),
        text=df_plot["Home"],
        textposition="top center",
        hoverinfo='skip'
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Away"],
        y=df_plot["M_A"],
        z=df_plot["MT_A"],
        mode='markers+text',
        name='Away',
        marker=dict(
            color='orangered',
            size=8,
            opacity=0.9,
            symbol='diamond',
            line=dict(color='darkred', width=2)
        ),
        text=df_plot["Away"],
        textposition="top center",
        hoverinfo='skip'
    ))

    fig_3d.update_layout(
        title=f"Top {len(df_plot)} Distâncias 3D – Tamanho Fixo" + (f" | {selected_league}" if selected_league != "⚽ Todas as ligas" else ""),
        scene=dict(
            xaxis=dict(title='Aggression', range=X_RANGE),
            yaxis=dict(title='Momentum Liga (M)', range=Y_RANGE),
            zaxis=dict(title='Momentum Time (MT)', range=Z_RANGE),
            aspectmode="cube"
        ),
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig_3d

fig_3d_fixed = create_fixed_3d_plot(df_filtered, n_to_show, selected_league)
st.plotly_chart(fig_3d_fixed, use_container_width=True)


# ========================= DICIONÁRIO DOS QUADRANTES 3D =========================
QUADRANTES_3D_LABELS = {
    1: "Fav Forte",
    2: "Fav Moderado",
    3: "Neutro",
    4: "Under Moderado",
    5: "Under Forte",
    6: "Underdog Forte",
    7: "Underdog Moderado",
    8: "Zona Desvio",
    9: "Clutch Attack",
    10: "Clutch Defense",
    11: "Stress Alto Home",
    12: "Stress Alto Away",
    13: "Regressão Forte Home",
    14: "Regressão Forte Away",
    15: "Alerta Home",
    16: "Alerta Away"
}



# =====================================================
# Recomendação / Score 3D / Live
# =====================================================
def adicionar_indicadores_explicativos_3d_16_dual(df):
    """
    Adiciona indicadores e recomendações do Sistema 3D Inteligente
    com lógica de EV para apostar apenas quando lucrativo.
    """

    df = df.copy()

    # Garantir preenchimento
    for col in ['EV_Main', 'Pred_Profit_Home', 'Pred_Profit_Away']:
        if col not in df.columns:
            df[col] = 0.0
    
    df['Quadrante_Home_Label'] = df['Quadrante_Home'].apply(
        lambda x: QUADRANTES_3D_LABELS.get(int(x), "Sem Quadrante")
    )
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].apply(
        lambda x: QUADRANTES_3D_LABELS.get(int(x), "Sem Quadrante")
    )

    # ------------------- Regras com EV + Momentum + Regressão -------------------
    def gerar_recomendacao_3d_ev(row):
        ev_main = row.get('EV_Main', 0.0)
        ev_home = row.get('Pred_Profit_Home', 0.0)
        ev_away = row.get('Pred_Profit_Away', 0.0)
        ml_side = row.get('ML_Side', 'PASS')
        score_home = row.get('Quadrante_ML_Score_Home', 0.5)
        score_away = row.get('Quadrante_ML_Score_Away', 0.5)
        momentum_h = row.get('M_H', 0.0)
        momentum_a = row.get('M_A', 0.0)
        tendencia_h = row.get('Tendencia_Home', '⚖️')
        tendencia_a = row.get('Tendencia_Away', '⚖️')

        # **PASS** → EV ruim
        if ev_main <= 0:
            return f"⏸ PASS (EV={ev_main:.2f}u)"

        # Se EV positivo → analisar padrões inteligentes
        # Prioridade à escolha do modelo por lucro esperado
        if ml_side == "HOME":
            return f"🏠 HOME +AH (EV={ev_home:.2f}u)"
        else:
            return f"🚀 AWAY +AH (EV={ev_away:.2f}u)"

    df['Recomendacao_EV'] = df.apply(gerar_recomendacao_3d_ev, axis=1)

    # ----------------- Classificação do Potencial 3D -----------------
    df['Classificacao_Potencial_3D'] = df['Quadrante_ML_Score_Main'].apply(
        lambda p: '🌟 ALTO POTENCIAL 3D' if p >= 0.65
        else '💼 VALOR SOLIDO 3D' if p >= 0.55
        else '⚖️ EQUILIBRADO'
    )

    # -------------------- Indicação final exibida --------------------
    df['Recomendacao'] = df['Recomendacao_EV']

    return df


def calcular_pontuacao_3d_quadrante_16(quadrante_id, momentum=0):
    scores_base = {
        1: 85, 2: 80, 3: 75, 4: 70,
        5: 70, 6: 65, 7: 60, 8: 55,
        9: 50, 10: 45, 11: 40, 12: 35,
        13: 35, 14: 30, 15: 25, 16: 20
    }
    base = scores_base.get(quadrante_id, 50)
    adjusted = base + momentum * 10
    return max(0, min(100, adjusted))

def gerar_score_combinado_3d_16(df):
    df = df.copy()
    if 'Quadrant_Dist_3D' not in df.columns:
        df['Quadrant_Dist_3D'] = 0.0
    if 'Quadrante_ML_Score_Main' not in df.columns:
        df['Quadrante_ML_Score_Main'] = 0.5

    df['Score_Base_Home'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Home'], x.get('M_H', 0)), axis=1
    )
    df['Score_Base_Away'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Away'], x.get('M_A', 0)), axis=1
    )

    df['Score_Combinado_3D'] = (df['Score_Base_Home'] * 0.5 +
                                df['Score_Base_Away'] * 0.3 +
                                df['Quadrant_Dist_3D'] * 0.2)

    df['Score_Final_3D'] = df['Score_Combinado_3D'] * df['Quadrante_ML_Score_Main']

    conditions = [
        df['Score_Final_3D'] >= 60,
        df['Score_Final_3D'] >= 45,
        df['Score_Final_3D'] >= 30,
        df['Score_Final_3D'] < 30
    ]
    choices = ['🌟 ALTO POTENCIAL 3D', '💼 VALOR SOLIDO 3D', '⚖️ NEUTRO 3D', '🔴 BAIXO POTENCIAL 3D']
    df['Classificacao_Potencial_3D'] = np.select(conditions, choices, default='⚖️ NEUTRO 3D')
    return df

# =====================================================
# Live: handicap e lucro
# =====================================================
def determine_handicap_result(row):
    try:
        gh = float(row['Goals_H_Today'])
        ga = float(row['Goals_A_Today'])
        line = float(row['Asian_Line_Decimal'])
    except (ValueError, TypeError, KeyError):
        return None

    margin = gh - ga
    diff = margin + line

    if abs(diff) < 1e-6:
        return "PUSH"
    elif diff > 0.5:
        return "HOME_COVERED"
    elif 0 < diff <= 0.5:
        return "HALF_HOME_COVERED"
    elif -0.5 < diff < 0:
        return "HALF_HOME_NOT_COVERED"
    elif diff <= -0.5:
        return "HOME_NOT_COVERED"
    else:
        return None

def check_handicap_recommendation_correct(rec, handicap_result):
    if pd.isna(rec) or handicap_result is None:
        return None
    rec = str(rec).upper()
    if any(k in rec for k in ['HOME']):
        return handicap_result in ["HOME_COVERED", "HALF_HOME_COVERED"]
    if any(k in rec for k in ['AWAY']):
        return handicap_result in ["HOME_NOT_COVERED", "HALF_HOME_NOT_COVERED"]
    return None

def calculate_handicap_profit(rec, handicap_result, odd_home, odd_away, asian_line_decimal):
    if pd.isna(rec) or handicap_result is None:
        return 0
    rec = str(rec).upper()
    is_home_bet = 'HOME' in rec and 'AWAY' not in rec
    is_away_bet = 'AWAY' in rec

    if not (is_home_bet or is_away_bet):
        return 0

    odd = odd_home if is_home_bet else odd_away
    result = str(handicap_result).upper()

    if result == "PUSH":
        return 0
    if result == "HALF_HOME_COVERED":
        return odd / 2 if is_home_bet else -0.5
    if result == "HALF_HOME_NOT_COVERED":
        return -0.5 if is_home_bet else odd / 2
    if result == "HOME_COVERED":
        return odd if is_home_bet else -1
    if result == "HOME_NOT_COVERED":
        return -1 if is_home_bet else odd
    return 0

def update_real_time_data_3d(df):
    df = df.copy()
    if "Score_Final_3D" not in df.columns:
        st.error("❌ 'Score_Final_3D' não encontrado – gere o score antes.")
        return df

    min_sf3d = st.slider(
        "📈 Score_Final_3D mínimo para considerar (0–100):",
        0, 70, 30, 1
    )
    df = df[df["Score_Final_3D"] >= min_sf3d].copy()
    st.info(f"✅ Considerando {len(df)} jogos com Score_Final_3D ≥ {min_sf3d}")

    df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)

    df['Quadrante_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct(r.get('Recomendacao'), r.get('Handicap_Result')),
        axis=1
    )

    odd_home_col = "Odd_H_Asi"
    odd_away_col = "Odd_A_Asi"
    if odd_home_col not in df.columns or odd_away_col not in df.columns:
        st.warning("⚠️ Odd_H_Asi / Odd_A_Asi não encontradas. Profit_Quadrante = 0.")
        df["Profit_Quadrante"] = 0.0
        return df

    df['Profit_Quadrante'] = df.apply(
        lambda r: calculate_handicap_profit(
            r.get('Recomendacao'),
            r.get('Handicap_Result'),
            r.get(odd_home_col),
            r.get(odd_away_col),
            r.get('Asian_Line_Decimal')
        ),
        axis=1
    )

    df['Bet_Result_Label'] = df['Profit_Quadrante'].apply(
        lambda x: "✅ Win" if x > 0 else ("❌ Loss" if x < 0 else "⚖️ Push")
    )
    return df

def generate_live_summary_3d(df):
    finished_games = df.dropna(subset=['Handicap_Result'])
    if finished_games.empty:
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": 0,
            "Apostas Quadrante 3D": 0,
            "Acertos Quadrante 3D": 0,
            "Winrate Quadrante 3D": "0%",
            "Profit Quadrante 3D": 0,
            "ROI Quadrante 3D": "0%"
        }
    quadrante_bets = finished_games[finished_games['Quadrante_Correct'].notna()]
    total_bets = len(quadrante_bets)
    correct_bets = quadrante_bets['Quadrante_Correct'].sum()
    winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
    total_profit = quadrante_bets['Profit_Quadrante'].sum()
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    return {
        "Total Jogos": len(df),
        "Jogos Finalizados": len(finished_games),
        "Apostas Quadrante 3D": total_bets,
        "Acertos Quadrante 3D": int(correct_bets),
        "Winrate Quadrante 3D": f"{winrate:.1f}%",
        "Profit Quadrante 3D": f"{total_profit:.2f}u",
        "ROI Quadrante 3D": f"{roi:.1f}%"
    }

def estilo_tabela_3d_quadrantes(df):
    styler = df.style
    if 'Quadrante_ML_Score_Home' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')
    if 'Quadrante_ML_Score_Away' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')
    if 'Score_Final_3D' in df.columns:
        styler = styler.background_gradient(subset=['Score_Final_3D'], cmap='RdYlGn')
    if 'M_H' in df.columns and 'M_A' in df.columns:
        styler = styler.background_gradient(subset=['M_H', 'M_A'], cmap='coolwarm')
    return styler

def resumo_3d_16_quadrantes_hoje(df):
    st.markdown("### 📋 Resumo Executivo - Sistema 3D Inteligente Hoje")
    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)
    alto_potencial_3d = (df['Classificacao_Potencial_3D'] == '🌟 ALTO POTENCIAL 3D').sum()
    valor_solido_3d = (df['Classificacao_Potencial_3D'] == '💼 VALOR SOLIDO 3D').sum()

    col_hv = df['Classificacao_Valor_Home'] if 'Classificacao_Valor_Home' in df.columns else pd.Series(['']*len(df))
    col_av = df['Classificacao_Valor_Away'] if 'Classificacao_Valor_Away' in df.columns else pd.Series(['']*len(df))

    alto_valor_home = (col_hv == '🏆 ALTO VALOR').sum()
    alto_valor_away = (col_av == '🏆 ALTO VALOR').sum()

    col_fav = df['eh_fav_forte_com_momentum'] if 'eh_fav_forte_com_momentum' in df.columns else pd.Series([0]*len(df))
    fav_forte_momentum = (col_fav == 1).sum()

    col_m_h = df['eh_forte_melhora_home'] if 'eh_forte_melhora_home' in df.columns else pd.Series([0]*len(df))
    col_m_a = df['eh_forte_melhora_away'] if 'eh_forte_melhora_away' in df.columns else pd.Series([0]*len(df))
    forte_melhora = ((col_m_h == 1) | (col_m_a == 1)).sum()

    col_conf_h = df['conflito_agg_regressao_home'] if 'conflito_agg_regressao_home' in df.columns else pd.Series([0]*len(df))
    col_conf_a = df['conflito_agg_regressao_away'] if 'conflito_agg_regressao_away' in df.columns else pd.Series([0]*len(df))
    conflitos_value = ((col_conf_h == 1) | (col_conf_a == 1)).sum()

    col_score = df['score_confianca_composto'] if 'score_confianca_composto' in df.columns else pd.Series([0]*len(df))
    score_conf_medio = col_score.mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("🌟 Alto Potencial 3D", alto_potencial_3d)
    with col2:
        st.metric("💪 Fav Forte + Momentum", fav_forte_momentum)
        st.metric("📈 Forte Melhora", forte_melhora)
    with col3:
        st.metric("💼 Valor Sólido 3D", valor_solido_3d)
        st.metric("🎯 Alto Valor", alto_valor_home + alto_valor_away)
    with col4:
        st.metric("🔥 Conflitos Value", conflitos_value)
        st.metric("📊 Score Confiança Médio", f"{score_conf_medio:.2f}")

    if 'Recomendacao' in df.columns:
        st.markdown("#### 📊 Distribuição de Recomendações 3D Inteligentes")
        st.dataframe(df['Recomendacao'].value_counts(), use_container_width=True)

# ========================= EXECUÇÃO FINAL =========================
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes (ML Inteligente)")

if games_today.empty:
    st.error("📭 Nenhum jogo disponível hoje. Verifique o CSV da pasta GamesDay.")
    st.stop()

ranking_3d = games_today.copy()
ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
ranking_3d = gerar_score_combinado_3d_16(ranking_3d)
ranking_3d = update_real_time_data_3d(ranking_3d)
ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

st.success(f"🎯 {len(ranking_3d)} jogos processados pelo Sistema 3D Inteligente")

st.markdown("## 📡 Live Score Monitor - Sistema 3D Inteligente")
live_summary_3d = generate_live_summary_3d(ranking_3d)
st.json(live_summary_3d)

resumo_3d_16_quadrantes_hoje(ranking_3d)

num_show = st.slider(
    "📌 Quantos jogos exibir na tabela?",
    min_value=5,
    max_value=len(ranking_3d),
    value=min(40, len(ranking_3d))
)

df_show = ranking_3d.head(num_show).copy()

st.markdown("### 📋 Lista de Recomendações - Ordenado por Score 3D")
st.dataframe(
    estilo_tabela_3d_quadrantes(df_show)
    .format({
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'Asian_Line_Decimal': '{:.2f}',
        'Profit_Quadrante': '{:.2f}',
        'Quadrante_ML_Score_Home': '{:.1%}',
        'Quadrante_ML_Score_Away': '{:.1%}',
        'Score_Final_3D': '{:.1f}',
        'M_H': '{:.2f}',
        'M_A': '{:.2f}',
        'Quadrant_Dist_3D': '{:.2f}',
        'Momentum_Diff': '{:.2f}',
        'Media_Score_Home': '{:.2f}',
        'Media_Score_Away': '{:.2f}',
        'Regressao_Force_Home': '{:.2f}',
        'Regressao_Force_Away': '{:.2f}',
        'score_confianca_composto': '{:.2f}'
    }, na_rep="-"),
    use_container_width=True
)

if "Profit_Quadrante" in ranking_3d.columns and ranking_3d['Profit_Quadrante'].notna().any():
    st.success("📈 Monitoramento de lucro operacional ativo!")
else:
    st.info("🕗 Aguardando LiveScore ou Odds Asiáticas para monitorar o lucro em tempo real...")





# ========================= 📊 TABELA EXTRA – PICK QUALITY TABLE =========================

st.markdown("### 📝 Tabela Especial — Resultados e EV das Oportunidades")

colunas_extra = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Asian_Line_Decimal',
    'Goals_H_Today', 'Goals_A_Today',
    'Best_Side', 'EV_Home', 'EV_Away',
    'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
    'ML_Confidence', 'Recomendacao_EV',
    'Handicap_Result', 'Profit_Quadrante', 'Bet_Result_Label'
]

df_final_extra = ranking_3d.copy()

# garantir compatibilidade: se não tiver alguma, preenche
for c in colunas_extra:
    if c not in df_final_extra.columns:
        df_final_extra[c] = "-"

df_final_extra = df_final_extra[colunas_extra]

# Ordenação por Score Final 3D (mantendo coerência com recomendação)
if 'Score_Final_3D' in ranking_3d.columns:
    df_final_extra = df_final_extra.loc[ranking_3d.sort_values('Score_Final_3D', ascending=False).index]

st.dataframe(
    df_final_extra.head(num_show).style.format({
        'Asian_Line_Decimal': '{:.2f}',
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'EV_Home': '{:.2f}',
        'EV_Away': '{:.2f}',
        'Quadrante_ML_Score_Home': '{:.2f}',
        'Quadrante_ML_Score_Away': '{:.2f}',
        'ML_Confidence': '{:.2f}',
        'Profit_Quadrante': '{:.2f}'
    }, na_rep="-"),
    use_container_width=True
)

st.info("📌 Esta tabela mostra EV + Resultados e deixa claro onde buscamos valor real!")





# ========================= 🧠 ANALISE ESTRATÉGICA PURA (SEM EV) =========================

st.markdown("### 🧠 Análise Estratégica Baseada em Momentum + Regressão (Sem EV)")

def analise_estrategica_sem_ev(row):
    mh = row.get('M_H', 0)
    ma = row.get('M_A', 0)
    tend_h = row.get('Tendencia_Home', '⚖️')
    tend_a = row.get('Tendencia_Away', '⚖️')
    cluster_h = row.get('Cluster3D_Desc', '')
    cluster_a = row.get('Cluster3D_Desc', '')

    texto = []

    # Tendências e Momentum
    if mh > 1.0:
        texto.append("🔥 Home em forte evolução")
    elif mh < -1.0:
        texto.append("📉 Home em forte queda")

    if ma > 1.0:
        texto.append("🚀 Away em forte evolução")
    elif ma < -1.0:
        texto.append("📉 Away em forte queda")

    # Clusters mostrando vantagem tática
    if "Under" in cluster_a or "Underdog" in cluster_a:
        texto.append("📊 Away com perfil de Underdog Forte")
    if "Fav Forte" in cluster_h:
        texto.append("💪 Perfil de Favorito Forte em casa")

    # Conflito entre expectativas e tendência
    if "FORTE QUEDA" in tend_h and mh < 0:
        texto.append("⚠ Favorito da casa inflado → cuidado")
    if "FORTE MELHORA" in tend_a and ma > 0:
        texto.append("💡 Underdog em ascensão → pode ter valor")

    # Conclusão simples caso nenhum insight forte
    if not texto:
        texto.append("↔ Equilíbrio — jogo difícil de ler")

    return " | ".join(texto)

ranking_3d['Analysis_Text_Pure'] = ranking_3d.apply(analise_estrategica_sem_ev, axis=1)

colunas_analise_pura = [
    'League', 'Home', 'Away',
    'Asian_Line_Decimal',
    'M_H', 'M_A',
    'Tendencia_Home', 'Tendencia_Away',
    'Cluster3D_Label', 'Cluster3D_Desc',
    'Analysis_Text_Pure'
]

# Ordena antes para manter Score_Final_3D válido
ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

df_ana_pura = ranking_3d[colunas_analise_pura].copy()

st.dataframe(
    df_ana_pura.head(num_show).style.format({
        'M_H': '{:.2f}',
        'M_A': '{:.2f}',
        'Asian_Line_Decimal': '{:.2f}',
    }, na_rep="-"),
    use_container_width=True
)

st.markdown("✔ Esta análise é independente do EV e mostra os sinais reais do comportamento dos times.")




st.markdown("---")
st.markdown("🏁 Fim da execução — Sistema 3D Inteligente totalmente operacional 🚀")
