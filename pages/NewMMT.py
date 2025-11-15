from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
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

# ========================= LIVE SCORE =========================
def setup_livescore_columns(df):
    """Garante que as colunas do Live Score existam no DataFrame"""
    df = df.copy()
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

# ========================= HELPERS BÁSICOS =========================
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
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

# ========================= ASIAN LINE CORRIGIDA =========================
def convert_asian_line_to_decimal(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    Ex: '0/0.5' (away) -> -0.25 (home invertido)
    """
    if pd.isna(line_str) or line_str == "":
        return None

    line_str = str(line_str).strip()

    # Caso especial: linha zero
    if line_str in ["0", "0.0"]:
        return 0.0

    # Caso simples — número único
    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num  # inverte (Away → Home)
        except ValueError:
            return None

    # Caso duplo — média dos dois lados com tratamento de sinal
    try:
        parts = [float(p) for p in line_str.split("/")]
        avg = sum(parts) / len(parts)
        first_part = parts[0]
        if first_part < 0:
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result  # inverte no final (Away → Home)
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

# ========================= CLUSTERIZAÇÃO 3D =========================
def aplicar_clusterizacao_3d(df, max_clusters=5, random_state=42):
    """
    Cria clusters espaciais 3D com número DINÂMICO baseado na quantidade de dados.
    """
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

    st.markdown("### 🧭 Clusters 3D Criados (Dinâmicos)")
    st.dataframe(centroids.style.format({
        'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}', 'Tamanho': '{:.0f}'
    }))

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
        cluster_descriptions[i] = classificar_cluster(
            centroid['dx'], centroid['dy'], centroid['dz']
        )

    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(cluster_descriptions)

    cluster_stats = df.groupby('Cluster3D_Label').agg({
        'dx': 'mean', 'dy': 'mean', 'dz': 'mean',
        'Cluster3D_Desc': 'first'
    }).round(3)

    st.markdown("### 📊 Estatísticas dos Clusters")
    st.dataframe(cluster_stats)

    return df

# ========================= Z-SCORES via HANDSCORE (M e MT) =========================
def calcular_zscores_detalhados(df):
    """
    Calcula Z-scores a partir do HandScore:
    - M_H, M_A: Z-score do time em relação à liga (performance relativa)
    - MT_H, MT_A: Z-score do time em relação a si mesmo (consistência)
    """
    df = df.copy()

    if 'M_H' not in df.columns:
        df['M_H'] = 0.0
    if 'M_A' not in df.columns:
        df['M_A'] = 0.0
    if 'MT_H' not in df.columns:
        df['MT_H'] = 0.0
    if 'MT_A' not in df.columns:
        df['MT_A'] = 0.0

    if 'League' in df.columns and 'HandScore_Home' in df.columns and 'HandScore_Away' in df.columns:
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
        df['M_H'] = 0.0
        df['M_A'] = 0.0

    if 'Home' in df.columns and 'Away' in df.columns:
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

        df = df.drop(
            ['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std',
             'HT_mean', 'HT_std', 'AT_mean', 'AT_std'],
            axis=1,
            errors='ignore'
        )
    else:
        df['MT_H'] = 0.0
        df['MT_A'] = 0.0

    df[['M_H', 'M_A', 'MT_H', 'MT_A']] = df[['M_H', 'M_A', 'MT_H', 'MT_A']].fillna(0)
    return df

def clean_features_for_training(X):
    """
    Remove infinitos, NaNs e limita outliers das features
    """
    X_clean = X.copy()

    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean)

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)

    for col in X_clean.columns:
        if pd.api.types.is_numeric_dtype(X_clean[col]):
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)

    X_clean = X_clean.replace([np.inf, -np.inf], 0).fillna(0)
    return X_clean

# ========================= CACHE - CARREGAMENTO PESADO =========================
@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)

    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = preprocess_df(history)
    return games_today, history

# ========================= CARREGAR DADOS =========================
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

if not os.path.exists(GAMES_FOLDER):
    st.error(f"Pasta {GAMES_FOLDER} não encontrada.")
    st.stop()

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("Nenhum CSV encontrado na pasta GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Selecione o arquivo de Matchday:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
selected_date = pd.to_datetime(selected_date_str)

games_today, history = load_cached_data(selected_file)

# ========================= LIVE SCORE MERGE =========================
def load_and_merge_livescore(games_today, selected_date_str):
    games_today = setup_livescore_columns(games_today)
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    if os.path.exists(livescore_file):
        results_df = pd.read_csv(livescore_file)
        if 'status' in results_df.columns:
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
            st.error(f"❌ LiveScore faltando colunas: {missing_cols}")
            return games_today
        games_today = games_today.merge(
            results_df,
            on='Id',
            how='left',
            suffixes=('', '_RAW')
        )
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        if 'status' in games_today.columns:
            games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
        games_today['Home_Red'] = games_today.get('home_red', np.nan)
        games_today['Away_Red'] = games_today.get('away_red', np.nan)
        st.success(f"✅ LiveScore merged: {len(results_df)} jogos carregados")
        return games_today
    else:
        st.warning(f"⚠️ Nenhum LiveScore encontrado para {selected_date_str}")
        return games_today

games_today = load_and_merge_livescore(games_today, selected_date_str)

# ========================= HISTÓRICO: ASIAN LINE & TARGET =========================
history = preprocess_df(history)
history = filter_leagues(history)

if not {'Goals_H_FT', 'Goals_A_FT', 'Asian_Line'}.issubset(history.columns):
    st.error("Histórico sem colunas necessárias: Goals_H_FT, Goals_A_FT, Asian_Line")
    st.stop()

history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
history = history.dropna(subset=['Goals_H_FT', 'Goals_A_FT', 'Asian_Line_Decimal']).copy()

if "Date" in history.columns:
    history['Date'] = pd.to_datetime(history['Date'], errors='coerce')
    history = history[history['Date'] < selected_date].copy()

history['Margin'] = history['Goals_H_FT'] - history['Goals_A_FT']
history["Target_AH_Home"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=False) > 0.5 else 0,
    axis=1
)

# ========================= QUADRANTES 16 =========================
st.markdown("## 🎯 Sistema 3D de 16 Quadrantes")

QUADRANTES_16 = {
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte",       "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado",    "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro",      "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},

    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte",       "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado",    "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro",      "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},

    9:  {"nome": "Under Moderado Neutro",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado",  "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",     "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},

    13: {"nome": "Under Forte Neutro",       "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",     "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",        "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte",  "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id
    return 0

for df_ in [games_today, history]:
    df_['Quadrante_Home'] = df_.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    df_['Quadrante_Away'] = df_.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )

# ========================= M/MT via HANDSCORE =========================
history = calcular_zscores_detalhados(history)
games_today = calcular_zscores_detalhados(games_today)

# ========================= REGRESSÃO À MÉDIA =========================
def calcular_regressao_media(df):
    df = df.copy()
    df['Extremidade_Home'] = (df.get('M_H', 0).abs() + df.get('MT_H', 0).abs())
    df['Extremidade_Away'] = (df.get('M_A', 0).abs() + df.get('MT_A', 0).abs())

    df['Regressao_Force_Home'] = -np.sign(df.get('M_H', 0)) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df.get('M_A', 0)) * (df['Extremidade_Away'] ** 0.7)

    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))

    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] +
                              0.4 * (1 - df.get('Aggression_Home', 0)))
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] +
                              0.4 * (1 - df.get('Aggression_Away', 0)))

    cond_h = [
        df['Regressao_Force_Home'] > 1.0,
        df['Regressao_Force_Home'] > 0.3,
        df['Regressao_Force_Home'] > -0.3,
        df['Regressao_Force_Home'] > -1.0,
        df['Regressao_Force_Home'] <= -1.0
    ]
    choices = ['📈 FORTE MELHORA', '📈 MELHORA', '⚖️ ESTÁVEL', '📉 QUEDA', '📉 FORTE QUEDA']
    df['Tendencia_Home'] = np.select(cond_h, choices, default='⚖️ ESTÁVEL')

    cond_a = [
        df['Regressao_Force_Away'] > 1.0,
        df['Regressao_Force_Away'] > 0.3,
        df['Regressao_Force_Away'] > -0.3,
        df['Regressao_Force_Away'] > -1.0,
        df['Regressao_Force_Away'] <= -1.0
    ]
    df['Tendencia_Away'] = np.select(cond_a, choices, default='⚖️ ESTÁVEL')
    return df

history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)

# ========================= DISTÂNCIAS 3D =========================
def calcular_distancias_3d(df):
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        for col in ['Quadrant_Dist_3D', 'Quadrant_Separation_3D',
                    'Quadrant_Angle_XY', 'Quadrant_Angle_XZ', 'Quadrant_Angle_YZ',
                    'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
                    'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
                    'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
                    'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
                    'Vector_Sign', 'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D']:
            df[col] = np.nan
        return df

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

history = calcular_distancias_3d(history)
games_today = calcular_distancias_3d(games_today)

# ========================= VISUALIZAÇÃO 2D DOS QUADRANTES =========================
def plot_quadrantes_16(df, side="Home"):
    fig, ax = plt.subplots(figsize=(10, 7))

    for quadrante_id in range(1, 17):
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(x, y, alpha=0.7, s=35, label=QUADRANTES_16[quadrante_id]['nome'])

    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    for y in [-45, -30, -15, 15, 30, 45]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    ax.set_xlabel(f'Aggression_{side} (-1 zebra ↔ +1 favorito)')
    ax.set_ylabel(f'HandScore_{side} (-60 a +60)')
    ax.set_title(f'16 Quadrantes - {side} (Visão 2D)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

st.markdown("### 📈 Visualização dos 16 Quadrantes (2D)")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

# ========================= VISUALIZAÇÃO 3D (MESMO ESQUEMA QUE VC JÁ TINHA) =========================
# (mantive igual à sua versão anterior – não mexe na ML, só visual)

df_filtered = games_today.copy()
if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())
    selected_league = st.selectbox(
        "Selecione a liga para análise 3D:",
        options=["⚽ Todas as ligas"] + list(leagues),
        index=0
    )
    if selected_league != "⚽ Todas as ligas":
        df_filtered = games_today[games_today["League"] == selected_league].copy()
else:
    selected_league = "⚽ Todas as ligas"

max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância 3D):",
                      10, min(max_n, 200), min(40, max_n), step=5)

# filtro angular simplificado (como já estava)
col_ang1, col_ang2, col_ang3 = st.columns(3)
with col_ang1:
    angulo_xy_range = st.slider("Ângulo XY (Aggression × M_Liga):", -180, 180, (-45, 45), step=5)
with col_ang2:
    angulo_xz_range = st.slider("Ângulo XZ (Aggression × M_Time):", -180, 180, (-45, 45), step=5)
with col_ang3:
    magnitude_min = st.slider("Magnitude Mínima 3D:", 0.0, 5.0, 0.5, 0.1)

aplicar_filtro = st.button("🎯 Aplicar Filtros Angulares")

def filtrar_por_angulo(df, angulo_xy_range, angulo_xz_range, magnitude_min):
    df_filtrado = df.copy()
    dx = df_filtrado['Aggression_Home'] - df_filtrado['Aggression_Away']
    dy = df_filtrado['M_H'] - df_filtrado['M_A']
    dz = df_filtrado['MT_H'] - df_filtrado['MT_A']
    angulo_xy = np.degrees(np.arctan2(dy, dx))
    angulo_xz = np.degrees(np.arctan2(dz, dx))
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    mask_xy = (angulo_xy >= angulo_xy_range[0]) & (angulo_xy <= angulo_xy_range[1])
    mask_xz = (angulo_xz >= angulo_xz_range[0]) & (angulo_xz <= angulo_xz_range[1])
    mask_mag = magnitude >= magnitude_min
    mask = mask_xy & mask_xz & mask_mag

    df_filtrado = df_filtrado[mask].copy()
    df_filtrado['Angulo_XY'] = angulo_xy[mask]
    df_filtrado['Angulo_XZ'] = angulo_xz[mask]
    df_filtrado['Magnitude_3D_Filtro'] = magnitude[mask]
    return df_filtrado

df_plot = df_filtered.copy()
if aplicar_filtro:
    df_plot = filtrar_por_angulo(df_plot, angulo_xy_range, angulo_xz_range, magnitude_min)
    st.success(f"✅ Filtro angular aplicado: {len(df_plot)} jogos")
else:
    df_plot = df_plot.nlargest(n_to_show, "Quadrant_Dist_3D")

def create_fixed_3d_plot(df_plot, n_to_show, selected_league):
    fig_3d = go.Figure()
    X_RANGE = [-1.2, 1.2]
    Y_RANGE = [-4.0, 4.0]
    Z_RANGE = [-4.0, 4.0]

    for _, row in df_plot.iterrows():
        xh = row.get("Aggression_Home", 0) or 0
        yh = row.get("M_H", 0) or 0
        zh = row.get("MT_H", 0) or 0
        xa = row.get("Aggression_Away", 0) or 0
        ya = row.get("M_A", 0) or 0
        za = row.get("MT_A", 0) or 0

        if all(v == 0 for v in [xh, yh, zh, xa, ya, za]):
            continue

        fig_3d.add_trace(go.Scatter3d(
            x=[xh, xa], y=[yh, ya], z=[zh, za],
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
        mode='markers',
        name='Home',
        marker=dict(color='royalblue', size=7, opacity=0.9)
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Away"],
        y=df_plot["M_A"],
        z=df_plot["MT_A"],
        mode='markers',
        name='Away',
        marker=dict(color='orangered', size=7, opacity=0.9)
    ))

    titulo_3d = f"Top {min(n_to_show, len(df_plot))} Distâncias 3D"
    if selected_league != "⚽ Todas as ligas":
        titulo_3d += f" | {selected_league}"

    fig_3d.update_layout(
        title=titulo_3d,
        scene=dict(
            xaxis=dict(title='Aggression', range=X_RANGE),
            yaxis=dict(title='Momentum Liga', range=Y_RANGE),
            zaxis=dict(title='Momentum Time', range=Z_RANGE),
            aspectmode="cube"
        ),
        template="plotly_dark",
        height=650,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig_3d

st.markdown("## 🎯 Visualização Interativa 3D – Tamanho Fixo")
fig_3d_fixed = create_fixed_3d_plot(df_plot, n_to_show, selected_league)
st.plotly_chart(fig_3d_fixed, use_container_width=True)

# ========================= CLUSTERIZAÇÃO 3D =========================
history = aplicar_clusterizacao_3d(history)
games_today = aplicar_clusterizacao_3d(games_today)

# ========================= FEATURES INTELIGENTES + TREINO ML =========================
def adicionar_features_inteligentes_ml(df):
    df = df.copy()
    df['eh_fav_forte_com_momentum'] = (
        (df['Quadrante_Home'].isin([1, 2, 3, 4])) &
        (df.get('M_H', 0) > 0.5) &
        (df.get('Regressao_Force_Home', 0) > 0)
    ).astype(int)

    df['eh_under_forte_sem_momentum'] = (
        (df['Quadrante_Away'].isin([13, 14, 15, 16])) &
        (df.get('M_A', 0) < -0.5) &
        (df.get('Regressao_Force_Away', 0) < 0)
    ).astype(int)

    df['eh_forte_melhora_home'] = (df.get('Tendencia_Home', '') == '📈 FORTE MELHORA').astype(int)
    df['eh_forte_melhora_away'] = (df.get('Tendencia_Away', '') == '📈 FORTE MELHORA').astype(int)
    df['eh_forte_queda_home'] = (df.get('Tendencia_Home', '') == '📉 FORTE QUEDA').astype(int)
    df['eh_forte_queda_away'] = (df.get('Tendencia_Away', '') == '📉 FORTE QUEDA').astype(int)

    df['conflito_agg_regressao_home'] = (
        (df.get('Aggression_Home', 0) > 0.3) &
        (df.get('Regressao_Force_Home', 0) < -0.8)
    ).astype(int)

    df['conflito_agg_regressao_away'] = (
        (df.get('Aggression_Away', 0) < -0.3) &
        (df.get('Regressao_Force_Away', 0) < -0.8)
    ).astype(int)

    df['momentum_confirma_home'] = (
        (df.get('Aggression_Home', 0) > 0.3) &
        (df.get('M_H', 0) > 0) &
        (df.get('Regressao_Force_Home', 0) > 0)
    ).astype(int)

    df['momentum_confirma_away'] = (
        (df.get('Aggression_Away', 0) < -0.3) &
        (df.get('M_A', 0) > 0) &
        (df.get('Regressao_Force_Away', 0) > 0)
    ).astype(int)

    df['momentum_negativo_alarmante_home'] = (
        (df.get('M_H', 0) < -1.0) &
        (df.get('Regressao_Force_Home', 0) < -0.5)
    ).astype(int)

    df['momentum_negativo_alarmante_away'] = (
        (df.get('M_A', 0) < -1.0) &
        (df.get('Regressao_Force_Away', 0) < -0.5)
    ).astype(int)

    df['padrao_fav_forte_vs_under_forte'] = (
        df['Quadrante_Home'].isin([1, 2, 3, 4]) &
        df['Quadrante_Away'].isin([13, 14, 15, 16])
    ).astype(int)

    df['padrao_fav_moderado_vs_under_moderado'] = (
        df['Quadrante_Home'].isin([5, 6, 7, 8]) &
        df['Quadrante_Away'].isin([9, 10, 11, 12])
    ).astype(int)

    aggression_proxy_home = (df.get('Aggression_Home', 0) + 1) / 2
    aggression_proxy_away = 1 - (df.get('Aggression_Away', 0) + 1) / 2

    df['score_confianca_composto'] = (
        (aggression_proxy_home * 0.3) +
        (aggression_proxy_away * 0.3) +
        (df.get('Media_Score_Home', 0) * 0.2) +
        (df.get('Media_Score_Away', 0) * 0.2)
    )

    return df

def treinar_modelo_inteligente(history, games_today):
    history = adicionar_features_inteligentes_ml(history)
    games_today = adicionar_features_inteligentes_ml(games_today)

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

    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    available_3d = [f for f in features_3d if f in history.columns]
    available_reg = [f for f in features_regressao if f in history.columns]
    available_int = [f for f in features_inteligentes if f in history.columns]

    extras_3d = history[available_3d].fillna(0)
    extras_reg = history[available_reg].fillna(0)
    extras_int = history[available_int].fillna(0)

    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, extras_reg, extras_int], axis=1)
    X = clean_features_for_training(X)

    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model_home.fit(X, y_home)

    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)

    extras_today_3d = games_today[available_3d].fillna(0)
    extras_today_reg = games_today[available_reg].fillna(0)
    extras_today_int = games_today[available_int].fillna(0)

    X_today = pd.concat(
        [ligas_today, clusters_today, extras_today_3d, extras_today_reg, extras_today_int],
        axis=1
    )
    X_today = clean_features_for_training(X_today)

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home >= proba_away, 'HOME', 'AWAY')
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today[['Prob_Home', 'Prob_Away']].max(axis=1)

    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.markdown("### 🔍 Top Features (Com Inteligência)")
    st.dataframe(importances.head(20).to_frame("Importância"))

    return model_home, games_today

# ========================= SISTEMA DE INDICAÇÕES 3D / SCORE / LIVE =========================
def adicionar_indicadores_explicativos_3d_16_dual(df):
    df = df.copy()
    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))

    # garantir colunas ML (fallback neutro se não treinado)
    if 'Quadrante_ML_Score_Home' not in df.columns:
        df['Quadrante_ML_Score_Home'] = 0.5
    if 'Quadrante_ML_Score_Away' not in df.columns:
        df['Quadrante_ML_Score_Away'] = 0.5
    if 'ML_Side' not in df.columns:
        df['ML_Side'] = np.where(df['Quadrante_ML_Score_Home'] >= df['Quadrante_ML_Score_Away'], 'HOME', 'AWAY')
    if 'Quadrante_ML_Score_Main' not in df.columns:
        df['Quadrante_ML_Score_Main'] = df[['Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away']].max(axis=1)

    cond_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_val = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(cond_home, choices_val, default='⚖️ NEUTRO')

    cond_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    df['Classificacao_Valor_Away'] = np.select(cond_away, choices_val, default='⚖️ NEUTRO')

    def gerar_recomendacao(row):
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        momentum_h = row.get('M_H', 0)
        momentum_a = row.get('M_A', 0)
        tendencia_h = row.get('Tendencia_Home', '⚖️ ESTÁVEL')
        tendencia_a = row.get('Tendencia_Away', '⚖️ ESTÁVEL')
        ml_side = row['ML_Side']

        if '📈 FORTE MELHORA' in tendencia_h and score_home >= 0.58:
            return f'🎯 HOME EM FORTE MELHORA ({score_home:.1%})'
        if '📈 FORTE MELHORA' in tendencia_a and score_away >= 0.58:
            return f'🎯 AWAY EM FORTE MELHORA ({score_away:.1%})'
        if '📉 FORTE QUEDA' in tendencia_h and score_away >= 0.55:
            return f'🔻 HOME EM QUEDA → AWAY ({score_away:.1%})'
        if '📉 FORTE QUEDA' in tendencia_a and score_home >= 0.55:
            return f'🔻 AWAY EM QUEDA → HOME ({score_home:.1%})'
        if ml_side == 'HOME' and score_home >= 0.60 and momentum_h > 0:
            return f'📈 MODELO CONFIA HOME ({score_home:.1%})'
        if ml_side == 'AWAY' and score_away >= 0.60 and momentum_a > 0:
            return f'📈 MODELO CONFIA AWAY ({score_away:.1%})'
        return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao, axis=1)
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)
    return df

def calcular_pontuacao_3d_quadrante_16(quadrante_id, momentum=0):
    scores_base = {
        1: 85, 2: 80, 3: 75, 4: 70,
        5: 70, 6: 65, 7: 60, 8: 55,
        9: 50, 10: 45, 11: 40, 12: 35,
        13: 35, 14: 30, 15: 25, 16: 20
    }
    base_score = scores_base.get(quadrante_id, 50)
    adjusted_score = base_score + momentum * 10
    return max(0, min(100, adjusted_score))

def gerar_score_combinado_3d_16(df):
    df = df.copy()
    df['Score_Base_Home'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Home'], x.get('M_H', 0)),
        axis=1
    )
    df['Score_Base_Away'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Away'], x.get('M_A', 0)),
        axis=1
    )
    df['Score_Combinado_3D'] = (
        df['Score_Base_Home'] * 0.5 +
        df['Score_Base_Away'] * 0.3 +
        df.get('Quadrant_Dist_3D', 0) * 0.2
    )
    if 'Quadrante_ML_Score_Main' not in df.columns:
        df['Quadrante_ML_Score_Main'] = 0.5
    df['Score_Final_3D'] = df['Score_Combinado_3D'] * df['Quadrante_ML_Score_Main']

    cond = [
        df['Score_Final_3D'] >= 60,
        df['Score_Final_3D'] >= 45,
        df['Score_Final_3D'] >= 30,
        df['Score_Final_3D'] < 30
    ]
    choices = ['🌟 ALTO POTENCIAL 3D', '💼 VALOR SOLIDO 3D', '⚖️ NEUTRO 3D', '🔴 BAIXO POTENCIAL 3D']
    df['Classificacao_Potencial_3D'] = np.select(cond, choices, default='⚖️ NEUTRO 3D')
    return df

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
    if any(k in rec for k in ['HOME', 'VALUE NO HOME', 'FAVORITO HOME', 'MODELO CONFIA HOME']):
        return handicap_result in ["HOME_COVERED", "HALF_HOME_COVERED"]
    if any(k in rec for k in ['AWAY', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
        return handicap_result in ["HOME_NOT_COVERED", "HALF_HOME_NOT_COVERED"]
    return None

def calculate_handicap_profit(rec, handicap_result, odd_home, odd_away, asian_line_decimal):
    if pd.isna(rec) or handicap_result is None:
        return 0
    rec = str(rec).upper()
    is_home_bet = any(k in rec for k in ['HOME', 'FAVORITO HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away_bet = any(k in rec for k in ['AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])

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
        df['Profit_Quadrante'] = 0.0
        df['Handicap_Result'] = None
        df['Quadrante_Correct'] = None
        return df

    min_sf3d = st.slider(
        "📈 Score_Final_3D mínimo para considerar (0–100):",
        0, 70, 30, 1,
        help="Somente recomendações com Score_Final_3D ≥ este valor serão consideradas."
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
        st.warning("⚠️ Odd_H_Asi / Odd_A_Asi não encontradas. Ajuste os nomes das colunas de odds asiáticas.")
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
            "Profit Quadrante 3D": "0.00u",
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
    def cor_classificacao_3d(valor):
        txt = str(valor)
        if any(k in txt for k in ['🌟 ALTO POTENCIAL 3D', '💼 VALOR SOLIDO 3D',
                                  '🏆 ALTO VALOR', '🔴 ALTO RISCO',
                                  'VALUE', 'EVITAR']):
            return 'font-weight: bold'
        return ''
    colunas_para_estilo = []
    for col in ['Classificacao_Potencial_3D', 'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao']:
        if col in df.columns:
            colunas_para_estilo.append(col)
    styler = df.style
    if colunas_para_estilo:
        styler = styler.applymap(cor_classificacao_3d, subset=colunas_para_estilo)
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


# ========================= EXECUÇÃO FINAL (COM BOTÃO B) =========================
st.markdown("## 🧠 Modelo 3D Inteligente - Treino & Recomendações")

if games_today.empty:
    st.error("📭 Nenhum jogo disponível hoje. Verifique o CSV da pasta GamesDay.")
    st.stop()

treinar = st.button("🚀 Treinar / Atualizar Modelo 3D Inteligente")

if treinar:
    if not history.empty:
        model_home, games_today_ml = treinar_modelo_inteligente(history, games_today)
        st.session_state['games_today_ml'] = games_today_ml
        st.success("✅ Modelo 3D Inteligente treinado e aplicado nos jogos de hoje!")
    else:
        st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D.")

# Base de trabalho
if 'games_today_ml' in st.session_state:
    base_df = st.session_state['games_today_ml'].copy()
else:
    base_df = games_today.copy()
    st.info("ℹ️ Modelo ainda não treinado nesta sessão. Treinando, usamos somente o espaço 3D (sem ML forte).")

ranking_3d = base_df.copy()
ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
ranking_3d = gerar_score_combinado_3d_16(ranking_3d)
ranking_3d = update_real_time_data_3d(ranking_3d)
ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes (ML Inteligente)")

if ranking_3d.empty:
    st.warning("⚠️ Nenhum jogo passou no filtro de Score_Final_3D / dados disponíveis.")
    st.json({
        "games_today_len": len(games_today),
        "cols": list(games_today.columns)
    })
    st.stop()

st.success(f"🎯 {len(ranking_3d)} jogos processados pelo Sistema 3D Inteligente")

# Resumo Live
st.markdown("## 📡 Live Score Monitor - Sistema 3D Inteligente")
live_summary_3d = generate_live_summary_3d(ranking_3d)
st.json(live_summary_3d)

# Resumo executivo
resumo_3d_16_quadrantes_hoje(ranking_3d)

# Tabela final
num_show = st.slider(
    "📌 Quantos jogos exibir na tabela?",
    min_value=5,
    max_value=len(ranking_3d),
    value=min(40, len(ranking_3d))
)
df_show = ranking_3d.head(num_show).copy()

colunas_3d = [
    'League', 'Time',
    'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
    'Recomendacao', 'ML_Side', 'Cluster3D_Desc',
    'Quadrante_Home_Label', 'Quadrante_Away_Label',
    'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
    'Score_Final_3D', 'Classificacao_Potencial_3D',
    'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
    'M_H', 'M_A', 'Quadrant_Dist_3D', 'Momentum_Diff',
    'Media_Score_Home', 'Media_Score_Away',
    'Regressao_Force_Home', 'Regressao_Force_Away',
    'eh_fav_forte_com_momentum', 'eh_under_forte_sem_momentum',
    'eh_forte_melhora_home', 'eh_forte_melhora_away',
    'score_confianca_composto',
    'Asian_Line_Decimal', 'Handicap_Result',
    'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
]
cols_finais_3d = [c for c in colunas_3d if c in df_show.columns]

st.markdown("### 📋 Lista de Recomendações - Ordenado por Score 3D")
st.dataframe(
    estilo_tabela_3d_quadrantes(df_show[cols_finais_3d])
    .format({
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'Asian_Line_Decimal': '{:.2f}',
        'Home_Red': '{:.0f}',
        'Away_Red': '{:.0f}',
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

st.markdown("---")
st.markdown("🏁 Fim da execução — Sistema 3D Inteligente totalmente operacional 🚀")
