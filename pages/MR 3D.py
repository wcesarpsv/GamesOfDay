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

st.set_page_config(page_title="Análise de Quadrantes 3D - Bet Indicator", layout="wide")
st.title("🎯 Análise 3D de 16 Quadrantes - ML Avançado (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------
LIVESCORE_FOLDER = "LiveScore"

def setup_livescore_columns(df):
    """Garante que as colunas do Live Score existam no DataFrame"""
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

# ---------------- Helpers Básicos ----------------
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

def convert_asian_line(line_str):
    """Converte string de linha asiática em média numérica"""
    try:
        if pd.isna(line_str) or line_str == "":
            return None
        line_str = str(line_str).strip()
        if "/" not in line_str:
            val = float(line_str)
            return 0.0 if abs(val) < 1e-10 else val
        parts = [float(x) for x in line_str.split("/")]
        avg = sum(parts) / len(parts)
        return 0.0 if abs(avg) < 1e-10 else avg
    except:
        return None

def calc_handicap_result(margin, asian_line_str, invert=False):
    """Retorna média de pontos por linha (1 win, 0.5 push, 0 loss)"""
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

def convert_asian_line_to_decimal(line_str):
    """Converte qualquer formato de Asian Line para valor decimal único"""
    if pd.isna(line_str) or line_str == "":
        return None

    try:
        line_str = str(line_str).strip()

        # Se não tem "/" é valor único
        if "/" not in line_str:
            return float(line_str)

        # Se tem "/" é linha fracionada - calcular média
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)

    except (ValueError, TypeError):
        return None

from sklearn.cluster import KMeans

# ==============================================================
# 🧩 BLOCO – CLUSTERIZAÇÃO 3D (KMEANS)
# ==============================================================

def aplicar_clusterizacao_3d(df, n_clusters=2, random_state=42):
    """
    Cria clusters espaciais com base em Aggression, Momentum Liga e Momentum Time.
    Retorna o DataFrame com a nova coluna 'Cluster3D_Label'.
    """

    df = df.copy()

    # Garante as colunas necessárias
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        return df

    # Diferenças espaciais (vetor 3D)
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    # KMeans 3D
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init='k-means++',   # garante convergência estável
        n_init=10           # mais robusto
    )
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    # 🧠 Calcular centroide de cada cluster para diagnóstico
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
    centroids['Cluster'] = range(n_clusters)

    st.markdown("### 🧭 Clusters 3D Criados (KMeans)")
    st.dataframe(centroids.style.format({'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}'}))

    # Adicionar também uma descrição textual leve (para visualização)
    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map({
        0: '⚡ Agressivos + Momentum Positivo',
        1: '💤 Reativos + Momentum Negativo',
        2: '⚖️ Equilibrados',
        3: '🔥 Alta Variância',
        4: '🌪️ Caóticos / Transição'
    }).fillna('🌀 Outro')

    return df

# ---------------- Carregar Dados ----------------
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

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

# ---------------- CACHE INTELIGENTE ----------------
@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_cached_data(selected_file):
    """Cache apenas dos dados pesados"""
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
    
    return games_today, history

# No lugar do carregamento atual, use:
games_today, history = load_cached_data(selected_file)

# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today, selected_date_str):
    """Carrega e faz merge dos dados do Live Score"""

    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    # Setup das colunas
    games_today = setup_livescore_columns(games_today)

    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)

        # Filtrar jogos cancelados/adiados
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
            st.error(f"❌ LiveScore file missing columns: {missing_cols}")
            return games_today
        else:
            # Fazer merge com os jogos do dia
            games_today = games_today.merge(
                results_df,
                left_on='Id',
                right_on='game_id',
                how='left',
                suffixes=('', '_RAW')
            )

            # Atualizar gols apenas para jogos finalizados
            games_today['Goals_H_Today'] = games_today['home_goal']
            games_today['Goals_A_Today'] = games_today['away_goal']
            games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

            # Atualizar cartões vermelhos
            games_today['Home_Red'] = games_today['home_red']
            games_today['Away_Red'] = games_today['away_red']

            st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
            return games_today
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

# Aplicar Live Score
games_today = load_and_merge_livescore(games_today, selected_date_str)

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# ---------------- CONVERSÃO ASIAN LINE ----------------
# Aplicar conversão no histórico e jogos de hoje
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

# Filtrar apenas jogos com linha válida no histórico
history = history.dropna(subset=['Asian_Line_Decimal'])
st.info(f"📊 Histórico com Asian Line válida: {len(history)} jogos")

# Filtro anti-leakage temporal
if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

# Targets AH históricos
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Target_AH_Home"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False) > 0.5 else 0, axis=1
)

# ---------------- SISTEMA 3D DE 16 QUADRANTES ----------------
st.markdown("## 🎯 Sistema 3D de 16 Quadrantes")

QUADRANTES_16 = {
    # 🔵 QUADRANTE 1-4: FORTE FAVORITO (+0.75 a +1.0)
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte",       "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado",    "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro",      "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},

    # 🟢 QUADRANTE 5-8: FAVORITO MODERADO (+0.25 a +0.75)
    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte",       "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado",    "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro",      "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},

    # 🟡 QUADRANTE 9-12: UNDERDOG MODERADO (-0.75 a -0.25)
    9: {"nome": "Under Moderado Neutro",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},

    # 🔴 QUADRANTE 13-16: FORTE UNDERDOG (-1.0 a -0.75)
    13: {"nome": "Under Forte Neutro",    "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",  "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",     "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    """Classifica Aggression e HandScore em um dos 16 quadrantes"""
    if pd.isna(agg) or pd.isna(hs):
        return 0  # Neutro/Indefinido

    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])

        if agg_ok and hs_ok:
            return quadrante_id

    return 0  # Caso não se enquadre em nenhum quadrante

# Aplicar classificação aos dados
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

########################################
#### 🧠 BLOCO – Cálculo de MT_H e MT_A (Momentum do Time)
########################################
def calcular_momentum_time(df, window=6):
    """
    Calcula o Momentum do Time (MT_H / MT_A) com base no HandScore,
    usando média móvel e normalização z-score por time.
    
    - MT_H: momentum do time em casa (últimos jogos como mandante)
    - MT_A: momentum do time fora (últimos jogos como visitante)
    - Valores típicos: [-3.5, +3.5]
    """
    df = df.copy()

    # Garante existência das colunas
    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    # Lista de todos os times (Home + Away)
    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        # ---------------- HOME ----------------
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:  # precisa de histórico mínimo
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = zscore

        # ---------------- AWAY ----------------
        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = zscore

    # Preenche eventuais NaN com 0 (neutro)
    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)

    return df

# ✅ Aplicar antes do cálculo 3D
history = calcular_momentum_time(history)
games_today = calcular_momentum_time(games_today)

# ---------------- CÁLCULO DE REGRESSÃO À MÉDIA ----------------
def calcular_regressao_media(df):
    """
    Calcula tendência de regressão à média baseada em:
    - M_H, M_A: Z-score do momentum na liga  
    - MT_H, MT_A: Z-score do momentum do time
    - Distância da média e volatilidade histórica
    """
    df = df.copy()
    
    # 1. SCORE DE EXTREMIDADE (quão longe da média)
    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])
    
    # 2. FORÇA DE REGRESSÃO (quanto tende a voltar à média)
    # Times com momentum muito alto tendem a regredir (valores negativos)
    # Times com momentum muito baixo tendem a melhorar (valores positivos)
    df['Regressao_Force_Home'] = -np.sign(df['M_H']) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df['M_A']) * (df['Extremidade_Away'] ** 0.7)
    
    # 3. PROBABILIDADE DE REGRESSÃO (0-1 scale)
    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))
    
    # 4. MEDIA SCORE FINAL (combina regressão com aggression atual)
    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] + 
                             0.4 * (1 - df['Aggression_Home']))  # Inverte aggression
    
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] + 
                             0.4 * (1 - df['Aggression_Away']))
    
    # 5. CLASSIFICAÇÃO DE REGRESSÃO
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

# Aplicar regressão à média ao histórico e dados de hoje
history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)

# ---------------- CÁLCULO DE DISTÂNCIAS 3D (Aggression × M × MT) ----------------
def calcular_distancias_3d(df):
    """
    Calcula distância 3D e ângulos usando Aggression, Momentum (liga) e Momentum (time)
    Versão neutra + features compostas (sin/cos combinados e sinal vetorial).
    """
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in [
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Quadrant_Angle_XY', 'Quadrant_Angle_XZ', 'Quadrant_Angle_YZ',
            'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
            'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
            'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
            'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo', 'Vector_Sign',
            'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D'
        ]:
            df[col] = np.nan
        return df

    # --- Diferenças puras ---
    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    # --- Distância Euclidiana pura ---
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    # --- Ângulos entre planos ---
    angle_xy = np.arctan2(dy, dx)
    angle_xz = np.arctan2(dz, dx)
    angle_yz = np.arctan2(dz, dy)

    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Angle_XZ'] = np.degrees(angle_xz)
    df['Quadrant_Angle_YZ'] = np.degrees(angle_yz)

    # --- Projeções trigonométricas básicas ---
    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Quadrant_Sin_XZ'] = np.sin(angle_xz)
    df['Quadrant_Cos_XZ'] = np.cos(angle_xz)
    df['Quadrant_Sin_YZ'] = np.sin(angle_yz)
    df['Quadrant_Cos_YZ'] = np.cos(angle_yz)

    # --- 🧩 1) Combinações trigonométricas compostas ---
    df['Quadrant_Sin_Combo'] = np.sin(angle_xy + angle_xz + angle_yz)
    df['Quadrant_Cos_Combo'] = np.cos(angle_xy + angle_xz + angle_yz)

    # --- 🧭 2) Sinal vetorial (direção espacial total) ---
    df['Vector_Sign'] = np.sign(dx * dy * dz)

    # --- Separação neutra 3D ---
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3

    # --- Diferenças individuais ---
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz

    # --- Magnitude total ---
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df

# Aplicar cálculo 3D ao games_today
games_today = calcular_distancias_3d(games_today)

# ---------------- VISUALIZAÇÃO DOS 16 QUADRANTES (2D) ----------------
def plot_quadrantes_16(df, side="Home"):
    """Plot dos 16 quadrantes com cores e anotações"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Definir cores por categoria
    cores_categorias = {
        'Fav Forte': 'lightcoral',
        'Fav Moderado': 'lightpink', 
        'Under Moderado': 'lightblue',
        'Under Forte': 'lightsteelblue'
    }

    # Plotar cada ponto com cor da categoria
    for quadrante_id in range(1, 17):
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            categoria = QUADRANTES_16[quadrante_id]['nome'].split()[0] + ' ' + QUADRANTES_16[quadrante_id]['nome'].split()[1]
            cor = cores_categorias.get(categoria, 'gray')

            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(x, y, c=cor, 
                      label=QUADRANTES_16[quadrante_id]['nome'],
                      alpha=0.7, s=50)

    # Linhas divisórias dos quadrantes (Aggression)
    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    # Linhas divisórias dos quadrantes (HandScore)  
    for y in [-45, -30, -15, 15, 30, 45]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Anotações dos quadrantes
    annot_config = [
        (0.875, 52.5, "Fav Forte\nMuito Forte", 8), (0.875, 37.5, "Fav Forte\nForte", 8),
        (0.875, 22.5, "Fav Forte\nModerado", 8), (0.875, 0, "Fav Forte\nNeutro", 8),
        (0.5, 52.5, "Fav Moderado\nMuito Forte", 8), (0.5, 37.5, "Fav Moderado\nForte", 8),
        (0.5, 22.5, "Fav Moderado\nModerado", 8), (0.5, 0, "Fav Moderado\nNeutro", 8),
        (-0.5, 0, "Under Moderado\nNeutro", 8), (-0.5, -22.5, "Under Moderado\nModerado", 8),
        (-0.5, -37.5, "Under Moderado\nForte", 8), (-0.5, -52.5, "Under Moderado\nMuito Forte", 8),
        (-0.875, 0, "Under Forte\nNeutro", 8), (-0.875, -22.5, "Under Forte\nModerado", 8),
        (-0.875, -37.5, "Under Forte\nForte", 8), (-0.875, -52.5, "Under Forte\nMuito Forte", 8)
    ]

    for x, y, text, fontsize in annot_config:
        ax.text(x, y, text, ha='center', fontsize=fontsize, weight='bold')

    ax.set_xlabel(f'Aggression_{side} (-1 zebra ↔ +1 favorito)')
    ax.set_ylabel(f'HandScore_{side} (-60 a +60)')
    ax.set_title(f'16 Quadrantes - {side} (Visão 2D)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Exibir gráficos 2D
st.markdown("### 📈 Visualização dos 16 Quadrantes (2D)")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

# ---------------- VISUALIZAÇÃO INTERATIVA 3D COM TAMANHO FIXO ----------------
import plotly.graph_objects as go

st.markdown("## 🎯 Visualização Interativa 3D – Tamanho Fixo")

# Filtros interativos
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

# Controle de número de confrontos
max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância 3D):", 10, min(max_n, 200), 40, step=5)

# ---------------- FILTRO ANGULAR 3D ----------------
st.markdown("### 🎯 Filtro Angular 3D")

col_ang1, col_ang2, col_ang3 = st.columns(3)

with col_ang1:
    angulo_xy_range = st.slider(
        "Ângulo XY - Aggression × Momentum Liga:",
        -180, 180, (-45, 45),
        step=5,
        help="Filtra jogos por inclinação entre Aggression (X) e Momentum Liga (Y)"
    )

with col_ang2:
    angulo_xz_range = st.slider(
        "Ângulo XZ - Aggression × Momentum Time:",
        -180, 180, (-45, 45), 
        step=5,
        help="Filtra jogos por inclinação entre Aggression (X) e Momentum Time (Z)"
    )

with col_ang3:
    magnitude_min = st.slider(
        "Magnitude Mínima 3D:",
        0.0, 5.0, 0.5, 0.1,
        help="Filtra por distância mínima da origem (intensidade do sinal 3D)"
    )

# Botão para aplicar filtros
aplicar_filtro = st.button("🎯 Aplicar Filtros Angulares", type="primary")

# ---------------- FUNÇÃO DE FILTRAGEM ANGULAR ----------------
def filtrar_por_angulo(df, angulo_xy_range, angulo_xz_range, magnitude_min):
    """Filtra jogos por ângulos e magnitude no espaço 3D"""
    df_filtrado = df.copy()
    
    # Garantir que as colunas necessárias existem
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df_filtrado.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas ausentes para filtro angular: {missing_cols}")
        return df_filtrado
    
    # Calcular diferenças
    dx = df_filtrado['Aggression_Home'] - df_filtrado['Aggression_Away']
    dy = df_filtrado['M_H'] - df_filtrado['M_A']  # Momentum Liga
    dz = df_filtrado['MT_H'] - df_filtrado['MT_A']  # Momentum Time
    
    # Ângulo XY (Aggression × Momentum Liga)
    angulo_xy = np.degrees(np.arctan2(dy, dx))
    
    # Ângulo XZ (Aggression × Momentum Time)  
    angulo_xz = np.degrees(np.arctan2(dz, dx))
    
    # Magnitude 3D
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Aplicar filtros
    mask_xy = (angulo_xy >= angulo_xy_range[0]) & (angulo_xy <= angulo_xy_range[1])
    mask_xz = (angulo_xz >= angulo_xz_range[0]) & (angulo_xz <= angulo_xz_range[1]) 
    mask_mag = magnitude >= magnitude_min
    
    df_filtrado = df_filtrado[mask_xy & mask_xz & mask_mag]
    
    # Adicionar colunas de ângulo para análise
    df_filtrado['Angulo_XY'] = angulo_xy[mask_xy & mask_xz & mask_mag]
    df_filtrado['Angulo_XZ'] = angulo_xz[mask_xy & mask_xz & mask_mag]
    df_filtrado['Magnitude_3D_Filtro'] = magnitude[mask_xy & mask_xz & mask_mag]
    
    return df_filtrado

# ---------------- PREPARAR DADOS PARA VISUALIZAÇÃO ----------------
# Primeiro criar df_plot como antes
df_plot = df_filtered.copy()

# DEPOIS aplicar filtro angular se solicitado
if aplicar_filtro:
    df_plot = filtrar_por_angulo(df_plot, angulo_xy_range, angulo_xz_range, magnitude_min)
    st.success(f"✅ Filtro aplicado! {len(df_plot)} jogos encontrados com os critérios angulares.")
    
    # Mostrar estatísticas dos ângulos
    if not df_plot.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ângulo XY Médio", f"{df_plot['Angulo_XY'].mean():.1f}°")
        with col2:
            st.metric("Ângulo XZ Médio", f"{df_plot['Angulo_XZ'].mean():.1f}°")
        with col3:
            st.metric("Magnitude Média", f"{df_plot['Magnitude_3D_Filtro'].mean():.2f}")
else:
    # Sem filtro, usar os top por distância como antes
    df_plot = df_plot.nlargest(n_to_show, "Quadrant_Dist_3D")

# ---------------- FILTRO DE REGRESSÃO À MÉDIA ----------------
st.sidebar.markdown("### 🔄 Filtro de Regressão à Média")

oportunidade_regressao = st.sidebar.selectbox(
    "Buscar oportunidades de regressão:",
    [
        "Todas as oportunidades",
        "🎯 Discordância Forte: Melhora vs Queda", 
        "📈 Times em Forte Melhora (Subvalorizados)",
        "📉 Times em Forte Queda (Sobrevalorizados)",
        "🔥 Conflito: ML vs Regressão (Value Spots)"
    ]
)

def filtrar_oportunidades_regressao(df, filtro):
    """Filtra jogos por oportunidades de regressão à média"""
    if filtro == "Todas as oportunidades":
        return df
    
    elif filtro == "🎯 Discordância Forte: Melhora vs Queda":
        # Home melhorando vs Away piorando (ou vice-versa)
        mask = (
            ((df['Regressao_Force_Home'] > 0.5) & (df['Regressao_Force_Away'] < -0.5)) |
            ((df['Regressao_Force_Home'] < -0.5) & (df['Regressao_Force_Away'] > 0.5))
        )
        return df[mask]
    
    elif filtro == "📈 Times em Forte Melhora (Subvalorizados)":
        # Times com forte tendência de melhora
        mask = (df['Regressao_Force_Home'] > 1.0) | (df['Regressao_Force_Away'] > 1.0)
        return df[mask]
    
    elif filtro == "📉 Times em Forte Queda (Sobrevalorizados)":
        # Times com forte tendência de queda
        mask = (df['Regressao_Force_Home'] < -1.0) | (df['Regressao_Force_Away'] < -1.0)
        return df[mask]
    
    elif filtro == "🔥 Conflito: ML vs Regressão (Value Spots)":
        # Onde o ML recomenda um lado mas a regressão recomenda o oposto
        mask = (
            ((df['ML_Side'] == 'HOME') & (df['Regressao_Force_Home'] < -0.8)) |
            ((df['ML_Side'] == 'AWAY') & (df['Regressao_Force_Away'] < -0.8))
        )
        return df[mask]

# Aplicar filtro se selecionado
if oportunidade_regressao != "Todas as oportunidades":
    df_plot = filtrar_oportunidades_regressao(df_plot, oportunidade_regressao)
    st.sidebar.success(f"🔍 {len(df_plot)} oportunidades de regressão encontradas!")

# ---------------------- CONFIGURAÇÃO COM TAMANHO FIXO ----------------------
def create_fixed_3d_plot(df_plot, n_to_show, selected_league):
    """Cria gráfico 3D com tamanho fixo para referência espacial consistente"""
    
    fig_3d = go.Figure()

    # RANGES FIXOS PARA REFERÊNCIA ESPACIAL
    X_RANGE = [-1.2, 1.2]      # Aggression (-1.2 a +1.2)
    Y_RANGE = [-4.0, 4.0]      # Momentum Liga (-4.0 a +4.0)  
    Z_RANGE = [-4.0, 4.0]      # Momentum Time (-4.0 a +4.0)

    for _, row in df_plot.iterrows():
        # Garantir valores válidos (fallback = 0)
        xh = row.get("Aggression_Home", 0) or 0
        yh = row.get("M_H", 0) if not pd.isna(row.get("M_H")) else 0
        zh = row.get("MT_H", 0) if not pd.isna(row.get("MT_H")) else 0

        xa = row.get("Aggression_Away", 0) or 0
        ya = row.get("M_A", 0) if not pd.isna(row.get("M_A")) else 0
        za = row.get("MT_A", 0) if not pd.isna(row.get("MT_A")) else 0

        # Verificar se há dados válidos para traçar
        if all(v == 0 for v in [xh, yh, zh, xa, ya, za]):
            continue

        # Plotar linha de conexão (Home → Away)
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
                f"🎯 Home: {QUADRANTES_16.get(row.get('Quadrante_Home'), {}).get('nome', 'N/A')}<br>"
                f"🎯 Away: {QUADRANTES_16.get(row.get('Quadrante_Away'), {}).get('nome', 'N/A')}<br>"
                f"📏 Dist 3D: {row.get('Quadrant_Dist_3D', np.nan):.2f}<br>"
                f"📍 Agg_H: {xh:.2f} | Agg_A: {xa:.2f}<br>"
                f"⚙️ M_H: {row.get('M_H', np.nan):.2f} | M_A: {row.get('M_A', np.nan):.2f}<br>"
                f"🔥 MT_H: {row.get('MT_H', np.nan):.2f} | MT_A: {row.get('MT_A', np.nan):.2f}<br>"
                f"📈 Tendência H: {row.get('Tendencia_Home', 'N/A')}<br>"
                f"📈 Tendência A: {row.get('Tendencia_Away', 'N/A')}"
            ),
            showlegend=False
        ))

    # Adicionar pontos Home (azul)
    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Home"],
        y=df_plot["M_H"], 
        z=df_plot["MT_H"],
        mode='markers+text',
        name='Home',
        marker=dict(
            color='royalblue',
            size=10,
            opacity=0.9,
            symbol='circle',
            line=dict(color='darkblue', width=2)
        ),
        text=df_plot["Home"],
        textposition="top center",
        hoverinfo='skip'
    ))

    # Adicionar pontos Away (vermelho)
    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Away"],
        y=df_plot["M_A"],
        z=df_plot["MT_A"], 
        mode='markers+text',
        name='Away',
        marker=dict(
            color='orangered',
            size=10,
            opacity=0.9,
            symbol='diamond',
            line=dict(color='darkred', width=2)
        ),
        text=df_plot["Away"],
        textposition="top center",
        hoverinfo='skip'
    ))

    # ---------------------- PLANOS DE REFERÊNCIA FIXOS ----------------------
    # Plano XY (z=0) - para referência
    x_plane = np.array([X_RANGE[0], X_RANGE[1], X_RANGE[1], X_RANGE[0]])
    y_plane = np.array([Y_RANGE[0], Y_RANGE[0], Y_RANGE[1], Y_RANGE[1]])
    z_plane = np.array([0, 0, 0, 0])
    
    fig_3d.add_trace(go.Mesh3d(
        x=x_plane, y=y_plane, z=z_plane,
        color='lightgray',
        opacity=0.1,
        name='Plano Neutro (Z=0)'
    ))

    # Linhas dos eixos principais
    fig_3d.add_trace(go.Scatter3d(
        x=[X_RANGE[0], X_RANGE[1]], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='red', width=4),
        name='Eixo X (Aggression)',
        showlegend=False
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[0, 0], y=[Y_RANGE[0], Y_RANGE[1]], z=[0, 0],
        mode='lines', 
        line=dict(color='green', width=4),
        name='Eixo Y (Momentum Liga)',
        showlegend=False
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[Z_RANGE[0], Z_RANGE[1]],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Eixo Z (Momentum Time)',
        showlegend=False
    ))

    # ---------------------- LAYOUT COM TAMANHO FIXO ----------------------
    titulo_3d = f"Top {n_to_show} Distâncias 3D – Tamanho Fixo"
    if selected_league != "⚽ Todas as ligas":
        titulo_3d += f" | {selected_league}"

    fig_3d.update_layout(
        title=dict(
            text=titulo_3d,
            x=0.5,
            font=dict(size=16, color='white')
        ),
        scene=dict(
            # RANGES FIXOS PARA REFERÊNCIA CONSISTENTE
            xaxis=dict(
                title='Aggression (-1 zebra ↔ +1 favorito)',
                range=X_RANGE,
                backgroundcolor="rgba(20,20,20,0.1)",
                gridcolor="gray",
                showbackground=True,
                gridwidth=2,
                zerolinecolor="red",
                zerolinewidth=4
            ),
            yaxis=dict(
                title='Momentum (Liga)',
                range=Y_RANGE, 
                backgroundcolor="rgba(20,20,20,0.1)",
                gridcolor="gray",
                showbackground=True,
                gridwidth=2,
                zerolinecolor="green",
                zerolinewidth=4
            ),
            zaxis=dict(
                title='Momentum (Time)',
                range=Z_RANGE,
                backgroundcolor="rgba(20,20,20,0.1)", 
                gridcolor="gray",
                showbackground=True,
                gridwidth=2,
                zerolinecolor="blue",
                zerolinewidth=4
            ),
            
            # CONFIGURAÇÃO DE CÂMERA FIXA
            aspectmode="cube",  # FORÇA PROPORÇÕES IGUAIS
            camera=dict(
                eye=dict(x=0.0, y=0.0, z=1.2),  # POSIÇÃO FIXA DA CÂMERA
                up=dict(x=0.3, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        template="plotly_dark",
        height=800,  # ALTURA FIXA
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    return fig_3d

# Criar e exibir o gráfico 3D com tamanho fixo
fig_3d_fixed = create_fixed_3d_plot(df_plot, n_to_show, selected_league)
st.plotly_chart(fig_3d_fixed, use_container_width=True)

# ---------------------- LEGENDA DE REFERÊNCIA ----------------------
st.markdown("""
### 🎯 Legenda do Espaço 3D Fixo

**Eixos com Ranges Fixos:**
- **X (Vermelho)**: Aggression → `-1.2` (Zebra Extrema) ↔ `+1.2` (Favorito Extremo)
- **Y (Verde)**: Momentum Liga → `-4.0` (Muito Negativo) ↔ `+4.0` (Muito Positivo)  
- **Z (Azul)**: Momentum Time → `-4.0` (Muito Negativo) ↔ `+4.0` (Muito Positivo)

**Referências Visuais:**
- 📍 **Plano Cinza**: Ponto neutro (Z=0) - momentum time equilibrado
- 🔵 **Bolas Azuis**: Times da Casa (Home)
- 🔴 **Losangos Vermelhos**: Visitantes (Away)
- ⚫ **Linhas Cinzas**: Conexões entre confrontos
""")

# Aplicar clusterização 3D antes do treino
history = aplicar_clusterizacao_3d(history, n_clusters=5)
games_today = aplicar_clusterizacao_3d(games_today, n_clusters=5)

# ---------------- MODELO ML COM REGRESSÃO À MÉDIA ----------------
def treinar_modelo_com_regressao(history, games_today):
    """
    Treina modelo ML 3D incluindo features de regressão à média
    """
    # Garantir features 3D e clusters
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    # Features categóricas (liga + cluster)
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    # Features contínuas vetoriais + REGRESSÃO
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D'
    ]
    
    # NOVAS FEATURES DE REGRESSÃO
    features_regressao = [
        'Media_Score_Home', 'Media_Score_Away',
        'Regressao_Force_Home', 'Regressao_Force_Away',
        'Extremidade_Home', 'Extremidade_Away'
    ]

    extras_3d = history[features_3d].fillna(0)
    extras_regressao = history[features_regressao].fillna(0)

    # Combinar todas as features
    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, extras_regressao], axis=1)

    # Target
    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    # Preparar dados de hoje
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d].fillna(0)
    extras_regressao_today = games_today[features_regressao].fillna(0)

    X_today = pd.concat([ligas_today, clusters_today, extras_today, extras_regressao_today], axis=1)

    # Previsões
    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today['ML_Confidence']

    # Importância das features
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features (Com Regressão à Média)")
    st.dataframe(importances.head(20).to_frame("Importância"), use_container_width=True)

    # Verificar se features de regressão estão no topo
    regressao_no_top = len([f for f in importances.head(15).index if 'Regressao' in f or 'Media_Score' in f])
    st.info(f"📊 Features de Regressão à Média no Top 15: {regressao_no_top}")

    st.success("✅ Modelo 3D treinado com Regressão à Média!")
    return model_home, games_today

# ---------------- SISTEMA DE INDICAÇÕES 3D PARA 16 QUADRANTES ----------------
def adicionar_indicadores_explicativos_3d_16_dual(df):
    """Adiciona classificações e recomendações explícitas para sistema 3D"""
    df = df.copy()

    # Mapear quadrantes para labels
    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))

    # 1. CLASSIFICAÇÃO DE VALOR PARA HOME (3D)
    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    # 2. CLASSIFICAÇÃO DE VALOR PARA AWAY (3D)
    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

    # 3. RECOMENDAÇÃO DE APOSTA 3D PARA 16 QUADRANTES (ATUALIZADA COM REGRESSÃO)
    def gerar_recomendacao_3d_16_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']
        momentum_h = row.get('M_H', 0)
        momentum_a = row.get('M_A', 0)
        tendencia_h = row.get('Tendencia_Home', '⚖️ ESTÁVEL')
        tendencia_a = row.get('Tendencia_Away', '⚖️ ESTÁVEL')

        # Padrões 3D específicos incorporando momentum E REGRESSÃO
        if 'Fav Forte' in home_q and 'Under Forte' in away_q and momentum_h > 1.0 and '📈' in tendencia_h:
            return f'💪 FAVORITO HOME SUPER FORTE (+Momentum +Regressão) ({score_home:.1%})'
        elif 'Under Forte' in home_q and 'Fav Forte' in away_q and momentum_a > 1.0 and '📈' in tendencia_a:
            return f'💪 FAVORITO AWAY SUPER FORTE (+Momentum +Regressão) ({score_away:.1%})'
        elif '📈 FORTE MELHORA' in tendencia_h and score_home >= 0.58:
            return f'🎯 HOME EM FORTE MELHORA (Regressão) ({score_home:.1%})'
        elif '📈 FORTE MELHORA' in tendencia_a and score_away >= 0.58:
            return f'🎯 AWAY EM FORTE MELHORA (Regressão) ({score_away:.1%})'
        elif '📉 FORTE QUEDA' in tendencia_h and score_away >= 0.55:
            return f'🔻 HOME EM FORTE QUEDA → AWAY (Regressão) ({score_away:.1%})'
        elif '📉 FORTE QUEDA' in tendencia_a and score_home >= 0.55:
            return f'🔻 AWAY EM FORTE QUEDA → HOME (Regressão) ({score_home:.1%})'
        elif 'Fav Moderado' in home_q and 'Under Moderado' in away_q and momentum_h > 0.5:
            return f'🎯 VALUE NO HOME (+Momentum) ({score_home:.1%})'
        elif 'Under Moderado' in home_q and 'Fav Moderado' in away_q and momentum_a > 0.5:
            return f'🎯 VALUE NO AWAY (+Momentum) ({score_away:.1%})'
        elif ml_side == 'HOME' and score_home >= 0.60 and momentum_h > 0:
            return f'📈 MODELO CONFIA HOME (+Momentum) ({score_home:.1%})'
        elif ml_side == 'AWAY' and score_away >= 0.60 and momentum_a > 0:
            return f'📈 MODELO CONFIA AWAY (+Momentum) ({score_away:.1%})'
        elif momentum_h < -1.0 and score_away >= 0.55:
            return f'🔻 HOME EM MOMENTUM NEGATIVO → AWAY ({score_away:.1%})'
        elif momentum_a < -1.0 and score_home >= 0.55:
            return f'🔻 AWAY EM MOMENTUM NEGATIVO → HOME ({score_home:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_3d_16_dual, axis=1)

    # 4. RANKING POR MELHOR PROBABILIDADE 3D
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- EXECUÇÃO PRINCIPAL 3D ----------------
# Executar treinamento 3D com regressão
if not history.empty:
    modelo_home, games_today = treinar_modelo_com_regressao(history, games_today)
    st.success("✅ Modelo 3D com Regressão à Média treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D")

# [RESTANTE DO CÓDIGO PERMANECE IGUAL...]
# ... (funções analisar_padroes_3d_quadrantes_16_dual, gerar_estrategias_3d_16_quadrantes, 
# calcular_pontuacao_3d_quadrante_16, gerar_score_combinado_3d_16, etc.)

# ---------------- EXIBIÇÃO DOS RESULTADOS 3D ----------------
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição 3D
    ranking_3d = games_today.copy()

    # Aplicar indicadores explicativos 3D
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)

    # Aplicar scoring combinado 3D
    ranking_3d = gerar_score_combinado_3d_16(ranking_3d)

    # [RESTANTE DO CÓDIGO DE EXIBIÇÃO PERMANECE IGUAL...]
    # ... (funções de atualização em tempo real, tabelas, resumos, etc.)

# ADICIONAR COLUNAS DE REGRESSÃO NA EXIBIÇÃO FINAL
colunas_3d.extend([
    'Tendencia_Home', 'Tendencia_Away',
    'Media_Score_Home', 'Media_Score_Away', 
    'Regressao_Force_Home', 'Regressao_Force_Away'
])

st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML com Regressão à Média** implementado com sucesso!")
st.info("""
**Novas Funcionalidades:**
- 🔄 **Regressão à Média**: Identifica times sobrevalorizados/subvalorizados
- 🎯 **Filtros Angulares**: Seleção por inclinação no espaço 3D  
- 📈 **Tendências de Melhora/Queda**: Baseado em z-score de momentum
- 🔍 **Oportunidades de Value**: Conflitos entre ML e regressão
- 📊 **Features Avançadas**: Media Score e Força de Regressão no modelo
""")
