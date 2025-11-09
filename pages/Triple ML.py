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


def convert_asian_line_to_decimal(value):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).

    Regras oficiais e consistentes com Pinnacle/Bet365:
      '0/0.5'   -> +0.25  (para away) → invertido: -0.25 (para home)
      '-0.5/0'  -> -0.25  (para away) → invertido: +0.25 (para home)
      '-1/1.5'  -> -0.25  → +0.25
      '1/1.5'   -> +1.25  → -1.25
      '1.5'     -> +1.50  → -1.50
      '0'       ->  0.00  →  0.00

    Retorna: float
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Caso simples — número único
    if "/" not in value:
        try:
            num = float(value)
            return -num  # Inverte sinal (Away → Home)
        except ValueError:
            return np.nan

    # Caso duplo — média dos dois lados
    try:
        parts = [float(p) for p in value.split("/")]
        avg = np.mean(parts)
        # Mantém o sinal do primeiro número
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        # Inverte o sinal no final (Away → Home)
        return -result
    except ValueError:
        return np.nan


# ============================================================
# 🎯 CÁLCULO DO TARGET – COBERTURA REAL DE HANDICAP (AH)
# ============================================================

def calculate_ah_home_target(row):
    """
    Calcula o target binário indicando se o time da casa cobriu o handicap asiático.
    Mantém compatibilidade total com o código original (ML1, ML2, LiveScore, Comparativo AH x 1x2).

    Retorna:
        1 -> Home cobriu o handicap
        0 -> Home não cobriu (inclui Push)
    """
    gh = row.get("Goals_H_FT")
    ga = row.get("Goals_A_FT")
    line_home = row.get("Asian_Line_Decimal")

    # Verificação de dados ausentes
    if pd.isna(gh) or pd.isna(ga) or pd.isna(line_home):
        return np.nan

    # Cálculo ajustado do placar considerando o handicap da casa
    adjusted = (gh + line_home) - ga

    # Resultado final
    if adjusted > 0:
        return 1   # Home cobre o handicap
    elif adjusted < 0:
        return 0   # Home não cobre o handicap
    else:
        return 0   # Push (tratado como 0 para manter consistência)

# ==============================================================
# 🧩 BLOCO – CLUSTERIZAÇÃO 3D (KMEANS)
# ==============================================================

def aplicar_clusterizacao_3d(df, n_clusters=4, random_state=42):
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
        3: '🔥 Alta Variância'
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
            'Id', 'status', 'home_goal', 'away_goal',
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
                right_on='Id',
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
history["Target_AH_Home"] = history.apply(calculate_ah_home_target, axis=1)

# ------------------------------------------------------------
# Aplicação do cálculo do target no histórico
# ------------------------------------------------------------
history["Target_AH_Home"] = history.apply(calculate_ah_home_target, axis=1)

# Remover partidas inválidas
history = history.dropna(subset=["Target_AH_Home"]).copy()

# Garantir tipo inteiro
history["Target_AH_Home"] = history["Target_AH_Home"].astype(int)

# Target alternativo para o time visitante (mantendo compatibilidade dual)
history["Target_AH_Away"] = 1 - history["Target_AH_Home"]

# Verificação de consistência mínima
if history["Target_AH_Home"].nunique() < 2:
    st.warning("⚠️ Target insuficiente (todas as classes iguais) — verifique colunas de gols/linha.")

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

# ============================================================
# 🎯 SISTEMA DE MÚLTIPLOS TARGETS (CHECKBOX)
# ============================================================

st.sidebar.markdown("## 🎯 Configuração de Targets ML")

# Checkbox para selecionar targets
st.sidebar.markdown("### Selecionar Estratégias de Target:")
use_target_original = st.sidebar.checkbox("🎯 Target Original (AH Home)", value=True)
use_target_espacial = st.sidebar.checkbox("🧭 Target Espacial Inteligente", value=True)
use_target_zona_risco = st.sidebar.checkbox("⚠️ Target Zona de Risco", value=True)
use_target_confianca = st.sidebar.checkbox("📊 Target Confiança Espacial", value=True)

# Funções dos novos targets
def calcular_confianca_espacial(row):
    """Calcula nível de confiança baseado na disposição espacial"""
    try:
        dx = row.get('Aggression_Home', 0) - row.get('Aggression_Away', 0)
        dy = row.get('M_H', 0) - row.get('M_A', 0)
        dz = row.get('MT_H', 0) - row.get('MT_A', 0)
        distancia_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Ângulo de instabilidade
        angulo_xy = np.degrees(np.arctan2(dy, dx)) % 360 if dx != 0 else 0
        angulo_instavel = ((45 <= angulo_xy <= 135) or (225 <= angulo_xy <= 315))
        
        # Classificação de confiança
        if distancia_3d > 1.2 and not angulo_instavel:
            return "ALTA_CONFIANCA"
        elif angulo_instavel and distancia_3d < 0.6:
            return "ALTA_INSTABILIDADE"
        else:
            return "MEDIA_CONFIANCA"
    except:
        return "MEDIA_CONFIANCA"

def criar_target_espacial_inteligente(row):
    """
    Target binário que considera relações espaciais
    """
    try:
        dx = row.get('Aggression_Home', 0) - row.get('Aggression_Away', 0)
        dy = row.get('M_H', 0) - row.get('M_A', 0)
        dz = row.get('MT_H', 0) - row.get('MT_A', 0)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Critérios espaciais para value bet
        distancia_ok = dist > 0.7
        angulo_estavel = abs(np.degrees(np.arctan2(dy, dx))) < 40 if dx != 0 else True
        
        # Cluster estável (usando fallback se não existir)
        cluster_val = row.get('Cluster3D_Label', 0)
        cluster_confiavel = cluster_val in [0, 2]  # clusters mais estáveis
        
        if distancia_ok and angulo_estavel and cluster_confiavel:
            return 1  # VALUE BET ESPACIAL
        else:
            return 0  # EVITAR - alta instabilidade
    except:
        return 0

def criar_target_zona_risco(row):
    """
    Classifica jogos em zonas de risco baseado na posição 3D
    """
    try:
        dist_3d = row.get('Quadrant_Dist_3D', 0)
        vector_sign = row.get('Vector_Sign', 0)
        angulo_xy = row.get('Quadrant_Angle_XY', 0)
        
        # Zona Verde - Alta previsibilidade
        if dist_3d > 1.0 and vector_sign > 0 and abs(angulo_xy) < 30:
            return "ZONA_VERDE"
        # Zona Vermelha - Alta imprevisibilidade
        elif dist_3d < 0.5 or abs(angulo_xy) > 60:
            return "ZONA_VERMELHA"
        # Zona Amarela - Risco moderado
        else:
            return "ZONA_AMARELA"
    except:
        return "ZONA_AMARELA"

# ============================================================
# 🏗️ FUNÇÃO DE TREINAMENTO MULTI-TARGET
# ============================================================

def treinar_modelo_personalizado(history_subset, games_today, target_col):
    """
    Treina modelo com subconjunto específico de dados
    """
    # Features base (mesmas do original)
    ligas_dummies = pd.get_dummies(history_subset['League'], prefix='League')
    
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ', 
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D', "Asian_Line_Decimal"
    ]
    
    features_cluster = ['Cluster3D_Label', 'C3D_ZScore', 'C3D_Sin', 'C3D_Cos']
    
    # Garantir que as colunas existem
    available_features = []
    for feature in features_3d + features_cluster:
        if feature in history_subset.columns:
            available_features.append(feature)
    
    X = pd.concat([ligas_dummies, history_subset[available_features]], axis=1).fillna(0)
    y = history_subset[target_col].astype(int)
    
    # Modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Previsões para hoje
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    X_today = pd.concat([ligas_today, games_today[available_features]], axis=1).fillna(0)
    
    # Garantir mesma ordem de colunas
    missing_cols = set(X.columns) - set(X_today.columns)
    for col in missing_cols:
        X_today[col] = 0
    X_today = X_today[X.columns]
    
    proba = model.predict_proba(X_today)[:, 1]
    games_today[f'Prob_{target_col}'] = proba
    games_today[f'ML_Side_{target_col}'] = np.where(proba > 0.5, 'HOME', 'AWAY')
    games_today[f'Confidence_{target_col}'] = np.maximum(proba, 1-proba)
    
    return model, games_today

def treinar_modelo_3d_clusters_single(history, games_today):
    """
    🔧 Modelo original para compatibilidade
    """
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    history = aplicar_clusterizacao_3d(history, n_clusters=4)
    games_today = aplicar_clusterizacao_3d(games_today, n_clusters=4)

    # Feature engineering
    history['Cluster3D_Label'] = history['Cluster3D_Label'].astype(float)
    games_today['Cluster3D_Label'] = games_today['Cluster3D_Label'].astype(float)

    mean_c = history['Cluster3D_Label'].mean()
    std_c = history['Cluster3D_Label'].std(ddof=0) or 1
    history['C3D_ZScore'] = (history['Cluster3D_Label'] - mean_c) / std_c
    games_today['C3D_ZScore'] = (games_today['Cluster3D_Label'] - mean_c) / std_c

    history['C3D_Sin'] = np.sin(history['Cluster3D_Label'])
    history['C3D_Cos'] = np.cos(history['Cluster3D_Label'])
    games_today['C3D_Sin'] = np.sin(games_today['Cluster3D_Label'])
    games_today['C3D_Cos'] = np.cos(games_today['Cluster3D_Label'])

    # Features
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D', "Asian_Line_Decimal"
    ]

    features_cluster = ['Cluster3D_Label', 'C3D_ZScore', 'C3D_Sin', 'C3D_Cos']
    
    available_features = []
    for feature in features_3d + features_cluster:
        if feature in history.columns:
            available_features.append(feature)
            
    X = pd.concat([ligas_dummies, history[available_features]], axis=1).fillna(0)
    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model_home.fit(X, y_home)

    # Previsões
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    X_today = pd.concat([ligas_today, games_today[available_features]], axis=1).fillna(0)
    
    # Garantir mesma ordem
    missing_cols = set(X.columns) - set(X_today.columns)
    for col in missing_cols:
        X_today[col] = 0
    X_today = X_today[X.columns]

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today['ML_Confidence']

    return model_home, games_today

def treinar_modelos_multi_target(history, games_today):
    """
    Treina modelos separados para cada target selecionado
    """
    modelos = {}
    resultados = {}
    
    # Garantir que temos os dados 3D calculados
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    # Aplicar clusterização
    history = aplicar_clusterizacao_3d(history, n_clusters=4)
    games_today = aplicar_clusterizacao_3d(games_today, n_clusters=4)
    
    # 🎯 TARGET 1: ORIGINAL (AH Home)
    if use_target_original:
        st.markdown("### 🎯 Modelo 1: Target Original (AH Home)")
        try:
            modelo_original, games_original = treinar_modelo_3d_clusters_single(history, games_today)
            modelos['ORIGINAL'] = modelo_original
            resultados['ORIGINAL'] = games_original
            st.success("✅ Modelo Original treinado")
        except Exception as e:
            st.error(f"❌ Erro no modelo original: {e}")
    
    # 🧭 TARGET 2: ESPACIAL INTELIGENTE
    if use_target_espacial:
        st.markdown("### 🧭 Modelo 2: Target Espacial Inteligente")
        try:
            # Criar target espacial
            history_espacial = history.copy()
            history_espacial['Target_Espacial'] = history_espacial.apply(criar_target_espacial_inteligente, axis=1)
            
            # Filtrar apenas jogos espacialmente confiáveis
            history_confiavel = history_espacial[history_espacial['Target_Espacial'] == 1]
            
            if len(history_confiavel) > 10:  # Mínimo de amostras
                # Usar target original mas apenas nos espacialmente confiáveis
                history_confiavel = history_confiavel.dropna(subset=['Target_AH_Home'])
                modelo_espacial, games_espacial = treinar_modelo_personalizado(
                    history_confiavel, games_today, 'Target_AH_Home'
                )
                modelos['ESPACIAL'] = modelo_espacial
                resultados['ESPACIAL'] = games_espacial
                st.success(f"✅ Modelo Espacial treinado com {len(history_confiavel)} jogos confiáveis")
            else:
                st.warning("⚠️ Dados insuficientes para modelo espacial")
        except Exception as e:
            st.error(f"❌ Erro no modelo espacial: {e}")
    
    # ⚠️ TARGET 3: ZONA DE RISCO
    if use_target_zona_risco:
        st.markdown("### ⚠️ Modelo 3: Target Zona de Risco")
        try:
            history_zona = history.copy()
            history_zona['Zona_Risco'] = history_zona.apply(criar_target_zona_risco, axis=1)
            
            # Focar apenas na Zona Verde (alta confiança)
            history_verde = history_zona[history_zona['Zona_Risco'] == 'ZONA_VERDE']
            
            if len(history_verde) > 10:
                # Usar target original mas apenas na zona verde
                history_verde = history_verde.dropna(subset=['Target_AH_Home'])
                modelo_zona, games_zona = treinar_modelo_personalizado(
                    history_verde, games_today, 'Target_AH_Home'
                )
                modelos['ZONA_RISCO'] = modelo_zona
                resultados['ZONA_RISCO'] = games_zona
                st.success(f"✅ Modelo Zona Risco treinado com {len(history_verde)} jogos Zona Verde")
            else:
                st.warning("⚠️ Dados insuficientes para modelo zona risco")
        except Exception as e:
            st.error(f"❌ Erro no modelo zona risco: {e}")
    
    # 📊 TARGET 4: CONFIANÇA ESPACIAL
    if use_target_confianca:
        st.markdown("### 📊 Modelo 4: Target Confiança Espacial")
        try:
            history_confianca = history.copy()
            history_confianca['Confianca_Espacial'] = history_confianca.apply(calcular_confianca_espacial, axis=1)
            
            # Focar apenas em alta confiança
            history_alta_conf = history_confianca[history_confianca['Confianca_Espacial'] == 'ALTA_CONFIANCA']
            
            if len(history_alta_conf) > 10:
                history_alta_conf = history_alta_conf.dropna(subset=['Target_AH_Home'])
                modelo_confianca, games_confianca = treinar_modelo_personalizado(
                    history_alta_conf, games_today, 'Target_AH_Home'
                )
                modelos['CONFIANCA'] = modelo_confianca
                resultados['CONFIANCA'] = games_confianca
                st.success(f"✅ Modelo Confiança treinado com {len(history_alta_conf)} jogos de alta confiança")
            else:
                st.warning("⚠️ Dados insuficientes para modelo confiança")
        except Exception as e:
            st.error(f"❌ Erro no modelo confiança: {e}")
    
    return modelos, resultados

# ============================================================
# 📊 PAINEL COMPARATIVO DE RESULTADOS
# ============================================================

# ============================================================
# 📊 PAINEL COMPARATIVO DE RESULTADOS (CORRIGIDO)
# ============================================================

# ============================================================
# 📊 PAINEL COMPARATIVO DE RESULTADOS (CORRIGIDO DEFINITIVO)
# ============================================================

def exibir_comparativo_modelos(resultados):
    """
    Exibe comparação lado a lado dos diferentes modelos
    """
    st.markdown("## 📊 Comparativo de Modelos")
    
    if not resultados:
        st.warning("⚠️ Nenhum modelo selecionado para comparação")
        return
    
    # Criar tabela comparativa
    comparativo = []
    
    for modelo_nome, df in resultados.items():
        if not df.empty:
            # Estatísticas básicas
            total_jogos = len(df)
            
            # Encontrar colunas de probabilidade
            prob_cols = [c for c in df.columns if 'Prob_' in c and 'Home' in c]
            confidence_cols = [c for c in df.columns if 'Confidence_' in c]
            ml_side_cols = [c for c in df.columns if 'ML_Side_' in c]
            
            prob_media = df[prob_cols[0]].mean() if prob_cols else 0
            confidence_media = df[confidence_cols[0]].mean() if confidence_cols else 0
            
            # Contar recomendações
            if ml_side_cols:
                home_recomendations = len(df[df[ml_side_cols[0]] == 'HOME'])
                away_recomendations = len(df[df[ml_side_cols[0]] == 'AWAY'])
            else:
                home_recomendations = away_recomendations = 0
            
            comparativo.append({
                'Modelo': modelo_nome,
                'Total Jogos': total_jogos,
                'Prob Média': prob_media,  # Manter como float
                'Confiança Média': confidence_media,  # Manter como float
                'Recomendações HOME': home_recomendations,
                'Recomendações AWAY': away_recomendations
            })
    
    if comparativo:
        df_comparativo = pd.DataFrame(comparativo)
        
        # Criar DataFrame para exibição (com valores formatados)
        df_display = df_comparativo.copy()
        df_display['Prob Média'] = df_display['Prob Média'].apply(lambda x: f"{x:.1%}")
        df_display['Confiança Média'] = df_display['Confiança Média'].apply(lambda x: f"{x:.1%}")
        
        # Reordenar colunas para exibição
        df_display = df_display[['Modelo', 'Total Jogos', 'Prob Média', 'Confiança Média', 
                               'Recomendações HOME', 'Recomendações AWAY']]
        
        # Exibir sem background_gradient para evitar o erro
        st.dataframe(df_display, use_container_width=True)
        
        # Exibir métricas em cards para melhor visualização
        st.markdown("### 📈 Métricas dos Modelos")
        cols = st.columns(len(df_comparativo))
        
        for idx, (_, row) in enumerate(df_comparativo.iterrows()):
            with cols[idx]:
                st.metric(
                    label=f"**{row['Modelo']}**",
                    value=f"{row['Prob Média']:.1%}",
                    delta=f"Conf: {row['Confiança Média']:.1%}",
                    help=f"Total: {row['Total Jogos']} jogos | HOME: {row['Recomendações HOME']} | AWAY: {row['Recomendações AWAY']}"
                )
        
        # Exibir recomendações de cada modelo
        st.markdown("### 🎯 Recomendações Detalhadas por Modelo")
        for modelo_nome, df in resultados.items():
            with st.expander(f"📋 {modelo_nome} - {len(df)} jogos", expanded=False):
                # Encontrar colunas relevantes
                prob_cols = [c for c in df.columns if 'Prob_' in c and 'Home' in c]
                ml_side_cols = [c for c in df.columns if 'ML_Side_' in c]
                confidence_cols = [c for c in df.columns if 'Confidence_' in c]
                
                if prob_cols and ml_side_cols and confidence_cols:
                    cols_exibir = ['Home', 'Away', 'League', 
                                  prob_cols[0], ml_side_cols[0], confidence_cols[0]]
                    
                    cols_exibir = [c for c in cols_exibir if c in df.columns]
                    
                    # Formatar as colunas numéricas para exibição
                    display_df = df[cols_exibir].copy()
                    if prob_cols[0] in display_df.columns:
                        display_df[prob_cols[0]] = display_df[prob_cols[0]].apply(lambda x: f"{x:.1%}")
                    if confidence_cols[0] in display_df.columns:
                        display_df[confidence_cols[0]] = display_df[confidence_cols[0]].apply(lambda x: f"{x:.1%}")
                    
                    # Ordenar por confiança
                    if confidence_cols[0] in df.columns:
                        display_df = display_df.sort_values(confidence_cols[0], ascending=False)
                    
                    st.dataframe(display_df.head(20), use_container_width=True)
                    
                    # Estatísticas rápidas
                    home_count = len(display_df[display_df[ml_side_cols[0]] == 'HOME'])
                    away_count = len(display_df[display_df[ml_side_cols[0]] == 'AWAY'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏠 Recomendações HOME", home_count)
                    with col2:
                        st.metric("✈️ Recomendações AWAY", away_count)
                    with col3:
                        st.metric("📊 Confiança Média", f"{df[confidence_cols[0]].mean():.1%}")

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ============================================================

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Treinar Modelos Selecionados", type="primary"):
    with st.spinner("Treinando modelos selecionados..."):
        modelos, resultados = treinar_modelos_multi_target(history, games_today)
        
        if resultados:
            exibir_comparativo_modelos(resultados)
            
            # Salvar resultados para análise posterior
            for nome, df in resultados.items():
                df.to_csv(f"resultados_{nome}_{selected_date_str}.csv", index=False)
                st.success(f"💾 Resultados {nome} salvos em CSV")
                
            st.balloons()
else:
    st.info("👆 Selecione os modelos no sidebar e clique para treinar")

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

# Exibir gráficos 2D se não estiver treinando
if not st.sidebar.button:
    st.markdown("### 📈 Visualização dos 16 Quadrantes (2D)")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_quadrantes_16(games_today, "Home"))
    with col2:
        st.pyplot(plot_quadrantes_16(games_today, "Away"))

st.markdown("---")
st.success("🎯 **Sistema 3D de Múltiplos Targets** implementado com sucesso!")
st.info("""
**Como usar:**
1. ✅ Selecione no sidebar quais modelos testar
2. 🚀 Clique em "Treinar Modelos Selecionados" 
3. 📊 Compare os resultados na tabela comparativa
4. 💾 CSVs são salvos automaticamente para análise
5. 🔄 Teste diferentes combinações dia a dia
""")
