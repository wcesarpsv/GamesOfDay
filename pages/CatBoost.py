from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import math

st.set_page_config(page_title="Análise de Quadrantes 3D - Bet Indicator", layout="wide")
st.title("🎯 Análise 3D de 16 Quadrantes - ML Inteligente (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "coppa", "trophy"]

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

# 🟢🟢🟢 CORREÇÃO CRÍTICA - FUNÇÃO ASIAN LINE CORRIGIDA 🟢🟢🟢
def convert_asian_line_to_decimal(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).

    Regras oficiais e consistentes com Pinnacle/Bet365:
      '0/0.5'   -> +0.25  (para away) → invertido: -0.25 (para home)
      '-0.5/0'  -> -0.25  (para away) → invertido: +0.25 (para home)
      '-1/1.5'  -> -1.25  → +1.25
      '1/1.5'   -> +1.25  → -1.25
      '1.5'     -> +1.50  → -1.50
      '0'       ->  0.00  →  0.00

    Retorna: float ou None se inválido
    """
    if pd.isna(line_str) or line_str == "":
        return None

    line_str = str(line_str).strip()

    # Caso especial: linha zero
    if line_str == "0" or line_str == "0.0":
        return 0.0

    # Caso simples — número único
    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num  # ✅ CORREÇÃO: Inverte sinal (Away → Home)
        except ValueError:
            return None

    # Caso duplo — média dos dois lados com tratamento de sinal
    try:
        parts = [float(p) for p in line_str.split("/")]
        
        # Calcula média mantendo a lógica de sinal
        avg = sum(parts) / len(parts)
        
        # Determina o sinal base baseado no primeiro elemento
        first_part = parts[0]
        if first_part < 0:
            result = -abs(avg)
        else:
            result = abs(avg)
            
        # ✅ CORREÇÃO CRÍTICA: Inverte o sinal no final (Away → Home)
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

# =======================
# 🔢 FAIXAS DE HANDICAP
# =======================
def criar_faixa_handicap(line):
    """
    Cria faixas globais de handicap já na perspectiva do Home (Asian_Line_Decimal):
      line <= -0.75  -> Home super favorito
      -0.75 < line <= -0.25 -> Home favorito leve
      -0.25 < line < 0.25 -> Jogo equilibrado
      0.25 <= line < 0.75 -> Home underdog leve
      line >= 0.75 -> Home underdog pesado
    """
    if pd.isna(line):
        return "GLOBAL"

    try:
        x = float(line)
    except (TypeError, ValueError):
        return "GLOBAL"

    if x <= -0.75:
        return "FAV_PESADO"
    elif x <= -0.25:
        return "FAV_LEVE"
    elif x < 0.25:
        return "EQUILIBRADO"
    elif x < 0.75:
        return "DOG_LEVE"
    else:
        return "DOG_PESADO"

from sklearn.cluster import KMeans

# ==============================================================
# 🧩 BLOCO – CLUSTERIZAÇÃO 3D (KMEANS)
# ==============================================================

def aplicar_clusterizacao_3d(df, max_clusters=5, random_state=42):
    """
    Cria clusters espaciais 3D com número DINÂMICO baseado na quantidade de dados.
    """
    df = df.copy()

    # Garante as colunas necessárias
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        df['Cluster3D_Desc'] = 'Dados Insuficientes'
        return df

    # Calcula diferenças espaciais
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()
    
    # 🎯 DECISÃO INTELIGENTE: Ajusta número de clusters baseado nos dados
    n_samples = len(X_cluster)
    
    if n_samples < 2:
        # Menos de 2 amostras - não faz sentido clusterizar
        df['Cluster3D_Label'] = 0
        df['Cluster3D_Desc'] = 'Amostra Única'
        st.info("ℹ️ Apenas 1 jogo encontrado - cluster único criado")
        return df
    
    # Calcula número ideal de clusters (máximo 30% dos dados ou max_clusters)
    n_clusters = min(max_clusters, max(2, n_samples // 3))  # Pelo menos 2 clusters
    
    st.info(f"🎯 Clusterização: {n_samples} amostras → {n_clusters} clusters")
    
    # KMeans com número dinâmico de clusters
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init='k-means++',
        n_init=min(10, n_samples)  # Ajusta n_init também
    )
    
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    # 🧠 Calcular e mostrar centroides
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
    centroids['Cluster'] = range(n_clusters)
    centroids['Tamanho'] = [sum(df['Cluster3D_Label'] == i) for i in range(n_clusters)]

    st.markdown("### 🧭 Clusters 3D Criados (Dinâmicos)")
    st.dataframe(centroids.style.format({
        'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}',
        'Tamanho': '{:.0f}'
    }))

    # 🎨 Descrição inteligente dos clusters baseado nos centroides
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

    # Aplica descrição
    cluster_descriptions = {}
    for i in range(n_clusters):
        centroid = centroids.iloc[i]
        cluster_descriptions[i] = classificar_cluster(centroid['dx'], centroid['dy'], centroid['dz'])
    
    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(cluster_descriptions)

    # 📊 Estatísticas dos clusters
    cluster_stats = df.groupby('Cluster3D_Label').agg({
        'dx': 'mean', 'dy': 'mean', 'dz': 'mean',
        'Cluster3D_Desc': 'first'
    }).round(3)
    
    st.markdown("### 📊 Estatísticas dos Clusters")
    st.dataframe(cluster_stats)

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

# ---------------- CONVERSÃO ASIAN LINE CORRIGIDA ----------------
# ✅ AGORA aplica a conversão CORRETA no histórico e jogos de hoje
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

# Targets AH históricos CORRIGIDOS
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Target_AH_Home"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=False) > 0.5 else 0, axis=1
)

history["Target_AH_Away"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=True) > 0.5 else 0, axis=1
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

# ===================================
# 🔧 CALIBRAÇÃO DINÂMICA DE THRESHOLDS (REGRESSÃO)
# ===================================
def calibrar_thresholds_regressao_por_handicap(history):
    """
    Usa o histórico para calibrar thresholds de Regressao_Force_Home/Away
    por faixa de handicap (FAV_PESADO, EQUILIBRADO, DOG_PESADO, etc.).
    """
    hist = history.copy()

    required_cols = ['Asian_Line_Decimal', 'Regressao_Force_Home', 'Regressao_Force_Away']
    missing = [c for c in required_cols if c not in hist.columns]
    if missing:
        st.warning(f"⚠️ Não foi possível calibrar thresholds dinâmicos. Colunas faltando: {missing}")
        return {}

    hist['Faixa_Handicap'] = hist['Asian_Line_Decimal'].apply(criar_faixa_handicap)

    thresholds = {}

    for faixa, grupo in hist.groupby('Faixa_Handicap'):
        if len(grupo) < 50:
            continue

        qs_home = grupo['Regressao_Force_Home'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        qs_away = grupo['Regressao_Force_Away'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()

        thresholds[faixa] = {
            'home': {
                'q20': qs_home[0.2], 'q40': qs_home[0.4],
                'q60': qs_home[0.6], 'q80': qs_home[0.8],
            },
            'away': {
                'q20': qs_away[0.2], 'q40': qs_away[0.4],
                'q60': qs_away[0.6], 'q80': qs_away[0.8],
            }
        }

    if 'GLOBAL' not in thresholds and len(hist) >= 50:
        qs_home = hist['Regressao_Force_Home'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        qs_away = hist['Regressao_Force_Away'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        thresholds['GLOBAL'] = {
            'home': {
                'q20': qs_home[0.2], 'q40': qs_home[0.4],
                'q60': qs_home[0.6], 'q80': qs_home[0.8],
            },
            'away': {
                'q20': qs_away[0.2], 'q40': qs_away[0.4],
                'q60': qs_away[0.6], 'q80': qs_away[0.8],
            }
        }

    st.info(f"✅ Thresholds dinâmicos calibrados para {len(thresholds)} faixas de handicap")
    return thresholds

def aplicar_thresholds_regressao_por_handicap(df, thresholds):
    """
    Aplica os thresholds dinâmicos para gerar:
      - Tendencia_Home
      - Tendencia_Away
    """
    if not thresholds:
        st.warning("⚠️ Nenhum threshold dinâmico disponível. Mantendo tendências padrão (se existirem).")
        return df

    df = df.copy()
    df['Faixa_Handicap'] = df['Asian_Line_Decimal'].apply(criar_faixa_handicap)

    def classificar(row, side):
        faixa = row.get('Faixa_Handicap', 'GLOBAL')
        thr_faixa = thresholds.get(faixa, thresholds.get('GLOBAL', None))
        if thr_faixa is None:
            return '⚖️ ESTÁVEL'

        if side == 'home':
            val = row.get('Regressao_Force_Home', np.nan)
            qs = thr_faixa['home']
        else:
            val = row.get('Regressao_Force_Away', np.nan)
            qs = thr_faixa['away']

        if pd.isna(val):
            return '⚖️ ESTÁVEL'

        if val > qs['q80']:
            return '📈 FORTE MELHORA'
        elif val > qs['q60']:
            return '📈 MELHORA'
        elif val > qs['q40']:
            return '⚖️ ESTÁVEL'
        elif val > qs['q20']:
            return '📉 QUEDA'
        else:
            return '📉 FORTE QUEDA'

    df['Tendencia_Home'] = df.apply(lambda r: classificar(r, 'home'), axis=1)
    df['Tendencia_Away'] = df.apply(lambda r: classificar(r, 'away'), axis=1)

    return df

# ---------------- CÁLCULO DE REGRESSÃO À MÉDIA ----------------
def calcular_regressao_media(df):
    """
    Calcula tendência de regressão à média baseada em M_H/M_A e MT_H/MT_A.
    """
    df = df.copy()

    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])
    
    alpha = 0.65
    beta = 0.35
    
    sign_home = alpha * np.sign(df['M_H']) + beta * np.sign(df['MT_H'])
    sign_away = alpha * np.sign(df['M_A']) + beta * np.sign(df['MT_A'])
    
    df['Regressao_Force_Home'] = -sign_home * (df['Extremidade_Home'] ** 0.70)
    df['Regressao_Force_Away'] = -sign_away * (df['Extremidade_Away'] ** 0.70)
    
    df['Regressao_Force_Home'] = df['Regressao_Force_Home'].replace([np.inf, -np.inf], 0).fillna(0)
    df['Regressao_Force_Away'] = df['Regressao_Force_Away'].replace([np.inf, -np.inf], 0).fillna(0)

    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] + 
                             0.4 * (1 - df['Aggression_Home']))
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] + 
                             0.4 * (1 - df['Aggression_Away']))

    return df

# Aplicar regressão à média
history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)

# ============================================
# 🔥 CALIBRAÇÃO E APLICAÇÃO DOS THRESHOLDS 3D
# ============================================
thresholds_reg = calibrar_thresholds_regressao_por_handicap(history)
history = aplicar_thresholds_regressao_por_handicap(history, thresholds_reg)
games_today = aplicar_thresholds_regressao_por_handicap(games_today, thresholds_reg)

# ---------------- CÁLCULO DE DISTÂNCIAS 3D ----------------
def calcular_distancias_3d(df):
    """
    Calcula distância 3D e ângulos usando Aggression, Momentum (liga) e Momentum (time)
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

games_today = calcular_distancias_3d(games_today)

# ---------------- VISUALIZAÇÃO 16 QUADRANTES 2D ----------------
def plot_quadrantes_16(df, side="Home"):
    """Plot dos 16 quadrantes com cores e anotações"""
    fig, ax = plt.subplots(figsize=(14, 10))

    cores_categorias = {
        'Fav Forte Forte': 'gold',
        'Fav Forte': 'blue',
        'Fav Moderado Forte': 'gold',
        'Fav Moderado': 'black', 
        'Under Moderado': 'black',
        'Under Forte': 'red'
    }

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

    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    for y in [-45, -30, -15, 15, 30, 45]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

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

st.markdown("### 📈 Visualização dos 16 Quadrantes (2D)")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

# ---------------- VISUALIZAÇÃO INTERATIVA 3D COM TAMANHO FIXO ----------------
import plotly.graph_objects as go

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

aplicar_filtro = st.button("🎯 Aplicar Filtros Angulares", type="primary")

def filtrar_por_angulo(df, angulo_xy_range, angulo_xz_range, magnitude_min):
    df_filtrado = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df_filtrado.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas ausentes para filtro angular: {missing_cols}")
        return df_filtrado

    dx = df_filtrado['Aggression_Home'] - df_filtrado['Aggression_Away']
    dy = df_filtrado['M_H'] - df_filtrado['M_A']
    dz = df_filtrado['MT_H'] - df_filtrado['MT_A']

    angulo_xy = np.degrees(np.arctan2(dy, dx))
    angulo_xz = np.degrees(np.arctan2(dz, dx))
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    mask_xy = (angulo_xy >= angulo_xy_range[0]) & (angulo_xy <= angulo_xy_range[1])
    mask_xz = (angulo_xz >= angulo_xz_range[0]) & (angulo_xz <= angulo_xz_range[1]) 
    mask_mag = magnitude >= magnitude_min

    df_filtrado = df_filtrado[mask_xy & mask_xz & mask_mag]

    df_filtrado['Angulo_XY'] = angulo_xy[mask_xy & mask_xz & mask_mag]
    df_filtrado['Angulo_XZ'] = angulo_xz[mask_xy & mask_xz & mask_mag]
    df_filtrado['Magnitude_3D_Filtro'] = magnitude[mask_xy & mask_xz & mask_mag]

    return df_filtrado

df_plot = df_filtered.copy()

if aplicar_filtro:
    df_plot = filtrar_por_angulo(df_plot, angulo_xy_range, angulo_xz_range, magnitude_min)
    st.success(f"✅ Filtro aplicado! {len(df_plot)} jogos encontrados com os critérios angulares.")

    if not df_plot.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ângulo XY Médio", f"{df_plot['Angulo_XY'].mean():.1f}°")
        with col2:
            st.metric("Ângulo XZ Médio", f"{df_plot['Angulo_XZ'].mean():.1f}°")
        with col3:
            st.metric("Magnitude Média", f"{df_plot['Magnitude_3D_Filtro'].mean():.2f}")
else:
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
    if filtro == "Todas as oportunidades":
        return df

    elif filtro == "🎯 Discordância Forte: Melhora vs Queda":
        mask = (
            ((df['Regressao_Force_Home'] > 0.5) & (df['Regressao_Force_Away'] < -0.5)) |
            ((df['Regressao_Force_Home'] < -0.5) & (df['Regressao_Force_Away'] > 0.5))
        )
        return df[mask]

    elif filtro == "📈 Times em Forte Melhora (Subvalorizados)":
        mask = (df['Regressao_Force_Home'] > 1.0) | (df['Regressao_Force_Away'] > 1.0)
        return df[mask]

    elif filtro == "📉 Times em Forte Queda (Sobrevalorizados)":
        mask = (df['Regressao_Force_Home'] < -1.0) | (df['Regressao_Force_Away'] < -1.0)
        return df[mask]

    elif filtro == "🔥 Conflito: ML vs Regressão (Value Spots)":
        mask = (
            ((df['ML_Side'] == 'HOME') & (df['Regressao_Force_Home'] < -0.8)) |
            ((df['ML_Side'] == 'AWAY') & (df['Regressao_Force_Away'] < -0.8))
        )
        return df[mask]

if oportunidade_regressao != "Todas as oportunidades":
    df_plot = filtrar_oportunidades_regressao(df_plot, oportunidade_regressao)
    st.sidebar.success(f"🔍 {len(df_plot)} oportunidades de regressão encontradas!")

# ---------------------- CONFIGURAÇÃO COM TAMANHO FIXO ----------------------
def create_fixed_3d_plot(df_plot, n_to_show, selected_league):
    fig_3d = go.Figure()

    X_RANGE = [-1.2, 1.2]
    Y_RANGE = [-4.0, 4.0]
    Z_RANGE = [-4.0, 4.0]

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

    x_plane = np.array([X_RANGE[0], X_RANGE[1], X_RANGE[1], X_RANGE[0]])
    y_plane = np.array([Y_RANGE[0], Y_RANGE[0], Y_RANGE[1], Y_RANGE[1]])
    z_plane = np.array([0, 0, 0, 0])

    fig_3d.add_trace(go.Mesh3d(
        x=x_plane, y=y_plane, z=z_plane,
        color='lightgray',
        opacity=0.1,
        name='Plano Neutro (Z=0)'
    ))

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
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.2, y=1.4, z=0.9),
                up=dict(x=0.3, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    return fig_3d

fig_3d_fixed = create_fixed_3d_plot(df_plot, n_to_show, selected_league)
st.plotly_chart(fig_3d_fixed, use_container_width=True)

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
history = aplicar_clusterizacao_3d(history)
games_today = aplicar_clusterizacao_3d(games_today)

# ---------------- 🆕 FEATURES INTELIGENTES ----------------
def adicionar_features_inteligentes_ml(df):
    df = df.copy()
    
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
    
    df['eh_forte_melhora_home'] = (
        (df['Tendencia_Home'] == '📈 FORTE MELHORA')
    ).astype(int)
    
    df['eh_forte_melhora_away'] = (
        (df['Tendencia_Away'] == '📈 FORTE MELHORA')
    ).astype(int)
    
    df['eh_forte_queda_home'] = (
        (df['Tendencia_Home'] == '📉 FORTE QUEDA')
    ).astype(int)
    
    df['eh_forte_queda_away'] = (
        (df['Tendencia_Away'] == '📉 FORTE QUEDA')
    ).astype(int)
    
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

# ---------------- CALIBRAÇÃO DA MARGEM MÍNIMA (ProbDiffThreshold) ----------------
def calibrar_prob_diff_threshold(history, X_features, modelo_cb_home, modelo_cb_away):
    """
    Calibra a melhor margem mínima |Prob_Home - Prob_Away| (ProbDiffThreshold)
    maximizando ROI histórico. Odds: prioriza Asian (Odd_H_Asi/Odd_A_Asi) depois 1X2.
    """
    df = history.copy().reset_index(drop=True)

    proba_home_hist = modelo_cb_home.predict_proba(X_features)[:, 1]
    proba_away_hist = modelo_cb_away.predict_proba(X_features)[:, 1]

    df['Prob_Home_CB'] = proba_home_hist
    df['Prob_Away_CB'] = proba_away_hist
    df['ProbDiff'] = (df['Prob_Home_CB'] - df['Prob_Away_CB']).abs()

    def pick_odds(row):
        oh_asi = row.get('Odd_H_Asi', np.nan)
        oa_asi = row.get('Odd_A_Asi', np.nan)
        if not pd.isna(oh_asi) and not pd.isna(oa_asi):
            return oh_asi, oa_asi
        oh = row.get('Odd_H', np.nan)
        oa = row.get('Odd_A', np.nan)
        if not pd.isna(oh) and not pd.isna(oa):
            return oh, oa
        return np.nan, np.nan

    odds = df.apply(pick_odds, axis=1, result_type='expand')
    df['Odd_H_Calib'] = odds[0]
    df['Odd_A_Calib'] = odds[1]

    df = df.dropna(subset=['Odd_H_Calib', 'Odd_A_Calib', 'ProbDiff', 'Target_AH_Home', 'Target_AH_Away'])

    if df.empty or len(df) < 150:
        st.warning("⚠️ Poucos dados válidos para calibrar ProbDiffThreshold – usando valor padrão 0.10")
        return 0.10, df

    thresholds = np.arange(0.02, 0.31, 0.01)

    best_t = 0.10
    best_roi = -1e9
    best_n = 0

    for t in thresholds:
        sample = df[df['ProbDiff'] >= t]
        if len(sample) < 80:
            continue

        profits = []
        for _, r in sample.iterrows():
            if r['Prob_Home_CB'] > r['Prob_Away_CB']:
                is_home = True
            elif r['Prob_Away_CB'] > r['Prob_Home_CB']:
                is_home = False
            else:
                continue

            odd = r['Odd_H_Calib'] if is_home else r['Odd_A_Calib']
            target = r['Target_AH_Home'] if is_home else r['Target_AH_Away']

            if target == 1:
                profit = odd - 1.0
            else:
                profit = -1.0
            profits.append(profit)

        if not profits:
            continue

        total_profit = np.sum(profits)
        n_bets = len(profits)
        roi = total_profit / n_bets

        if roi > best_roi:
            best_roi = roi
            best_t = t
            best_n = n_bets

    st.info(f"🔧 ProbDiffThreshold calibrado: {best_t:.3f} | ROI hist: {best_roi:.3f} u/aposta em {best_n} bets")
    return best_t, df

# ---------------- 🧠 TREINO DO MODELO INTELIGENTE COM CATBOOST (HOME & AWAY) ----------------
def treinar_modelo_inteligente(history, games_today):
    """
    Treina modelos CatBoost Home & Away com features 3D + Regressão + Features Inteligentes
    e aplica ProbDiffThreshold calibrado para definir BetSignal_CB.
    """
    history = adicionar_features_inteligentes_ml(history)
    games_today = adicionar_features_inteligentes_ml(games_today)
    
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

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

    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, extras_regressao, extras_inteligentes], axis=1)

    y_home = history['Target_AH_Home'].astype(int)
    y_away = history['Target_AH_Away'].astype(int)

    modelo_cb_home = CatBoostClassifier(
        iterations=900,
        learning_rate=0.06,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        thread_count=-1,
        verbose=False
    )

    modelo_cb_away = CatBoostClassifier(
        iterations=900,
        learning_rate=0.06,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        thread_count=-1,
        verbose=False
    )

    st.info("🤖 Treinando CatBoost (Home)...")
    modelo_cb_home.fit(X, y_home)
    st.info("🤖 Treinando CatBoost (Away)...")
    modelo_cb_away.fit(X, y_away)

    # 🔧 Calibrar ProbDiffThreshold usando o histórico
    probdiff_threshold, _ = calibrar_prob_diff_threshold(history, X, modelo_cb_home, modelo_cb_away)

    # Preparar dados de hoje
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[available_3d].fillna(0)
    extras_regressao_today = games_today[available_regressao].fillna(0)
    extras_inteligentes_today = games_today[available_inteligentes].fillna(0)

    X_today = pd.concat([ligas_today, clusters_today, extras_today, extras_regressao_today, extras_inteligentes_today], axis=1)

    proba_home_today = modelo_cb_home.predict_proba(X_today)[:, 1]
    proba_away_today = modelo_cb_away.predict_proba(X_today)[:, 1]

    games_today['Prob_Home'] = proba_home_today
    games_today['Prob_Away'] = proba_away_today
    games_today['ProbDiff'] = (games_today['Prob_Home'] - games_today['Prob_Away']).abs()

    def decide_betsignal(row, threshold):
        if row['ProbDiff'] < threshold:
            return 'NO BET'
        if row['Prob_Home'] > row['Prob_Away']:
            return 'HOME'
        elif row['Prob_Away'] > row['Prob_Home']:
            return 'AWAY'
        else:
            return 'NO BET'

    games_today['BetSignal_CB'] = games_today.apply(lambda r: decide_betsignal(r, probdiff_threshold), axis=1)
    games_today['ProbDiffThreshold_Usado'] = probdiff_threshold

    games_today['ML_Side'] = games_today['BetSignal_CB'].replace({'NO BET': 'NEUTRO'})
    games_today['ML_Confidence'] = np.where(
        games_today['BetSignal_CB'] == 'HOME',
        games_today['Prob_Home'],
        np.where(games_today['BetSignal_CB'] == 'AWAY', games_today['Prob_Away'], 0.0)
    )

    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = np.maximum(games_today['Prob_Home'], games_today['Prob_Away'])

    importances_home = pd.Series(modelo_cb_home.get_feature_importance(), index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features – CatBoost (Home)")
    st.dataframe(importances_home.head(20).to_frame("Importância"), use_container_width=True)

    st.info(f"📏 ProbDiffThreshold final usado: {probdiff_threshold:.3f}")
    st.success("✅ Modelos CatBoost Inteligentes (Home & Away) treinados com sucesso!")

    return modelo_cb_home, modelo_cb_away, games_today

# ---------------- SISTEMA DE INDICAÇÕES 3D PARA 16 QUADRANTES ----------------
def adicionar_indicadores_explicativos_3d_16_dual(df):
    df = df.copy()

    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))

    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

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

    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

def estilo_tabela_3d_quadrantes(df):
    def cor_classificacao_3d(valor):
        if '🌟 ALTO POTENCIAL 3D' in str(valor): return 'font-weight: bold'
        elif '💼 VALOR SOLIDO 3D' in str(valor): return 'font-weight: bold'
        elif '🔴 BAIXO POTENCIAL 3D' in str(valor): return 'font-weight: bold'
        elif '🏆 ALTO VALOR' in str(valor): return 'font-weight: bold'
        elif '🔴 ALTO RISCO' in str(valor): return 'font-weight: bold'
        elif 'VALUE' in str(valor): return 'font-weight: bold'
        elif 'EVITAR' in str(valor): return 'font-weight: bold'
        else: return ''

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
    if 'M_H' in df.columns:
        styler = styler.background_gradient(subset=['M_H', 'M_A'], cmap='coolwarm')

    return styler

# ---------------- SISTEMA DE SCORING 3D ----------------
def calcular_pontuacao_3d_quadrante_16(quadrante_id, momentum=0):
    scores_base = {
        1: 85, 2: 80, 3: 75, 4: 70,
        5: 70, 6: 65, 7: 60, 8: 55,
        9: 50, 10: 45, 11: 40, 12: 35,
        13: 35, 14: 30, 15: 25, 16: 20
    }

    base_score = scores_base.get(quadrante_id, 50)
    momentum_boost = momentum * 10
    adjusted_score = base_score + momentum_boost
    return max(0, min(100, adjusted_score))

def gerar_score_combinado_3d_16(df):
    df = df.copy()

    df['Score_Base_Home'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Home'], x.get('M_H', 0)), axis=1
    )
    df['Score_Base_Away'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Away'], x.get('M_A', 0)), axis=1
    )

    df['Score_Combinado_3D'] = (df['Score_Base_Home'] * 0.5 + df['Score_Base_Away'] * 0.3 + 
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

# ---------------- ANÁLISE DE PADRÕES 3D ----------------
def analisar_padroes_3d_quadrantes_16_dual(df):
    st.markdown("### 🔍 Análise de Padrões 3D por Combinação")

    padroes_3d = {
        'Fav Forte Forte (+Momentum) vs Under Forte Muito Forte (-Momentum)': {
            'descricao': '🎯 **MELHOR PADRÃO 3D HOME** - Favorito forte com momentum vs underdog muito fraco sem momentum',
            'lado_recomendado': 'HOME',
            'prioridade': 1,
            'score_min': 0.65,
            'momentum_min_home': 0.5,
            'momentum_max_away': -0.5
        },
        'Under Forte Muito Forte (-Momentum) vs Fav Forte Forte (+Momentum)': {
            'descricao': '🎯 **MELHOR PADRÃO 3D AWAY** - Underdog muito fraco sem momentum vs favorito forte com momentum',
            'lado_recomendado': 'AWAY', 
            'prioridade': 1,
            'score_min': 0.65,
            'momentum_max_home': -0.5,
            'momentum_min_away': 0.5
        },
        'Fav Moderado Forte (+Momentum) vs Under Moderado Forte (-Momentum)': {
            'descricao': '💪 **PADRÃO 3D VALUE HOME** - Favorito moderado com momentum vs underdog moderado fraco sem momentum',
            'lado_recomendado': 'HOME',
            'prioridade': 2,
            'score_min': 0.58,
            'momentum_min_home': 0.3,
            'momentum_max_away': -0.3
        }
    }

    padroes_ordenados = sorted(padroes_3d.items(), key=lambda x: x[1]['prioridade'])

    for padrao, info in padroes_ordenados:
        home_q, away_q = padrao.split(' vs ')[0], padrao.split(' vs ')[1]
        home_q_base = home_q.split(' (')[0] if ' (' in home_q else home_q
        away_q_base = away_q.split(' (')[0] if ' (' in away_q else away_q

        jogos = df[
            (df['Quadrante_Home_Label'] == home_q_base) & 
            (df['Quadrante_Away_Label'] == away_q_base)
        ]

        if 'momentum_min_home' in info:
            jogos = jogos[jogos['M_H'] >= info['momentum_min_home']]
        if 'momentum_max_home' in info:
            jogos = jogos[jogos['M_H'] <= info['momentum_max_home']]
        if 'momentum_min_away' in info:
            jogos = jogos[jogos['M_A'] >= info['momentum_min_away']]
        if 'momentum_max_away' in info:
            jogos = jogos[jogos['M_A'] <= info['momentum_max_away']]

        if info['lado_recomendado'] == 'HOME':
            score_col = 'Quadrante_ML_Score_Home'
        else:
            score_col = 'Quadrante_ML_Score_Away'

        if 'score_min' in info:
            jogos = jogos[jogos[score_col] >= info['score_min']]

        if not jogos.empty:
            st.write(f"**{padrao}**")
            st.write(f"{info['descricao']}")
            st.write(f"📈 **Score mínimo**: {info.get('score_min', 0.50):.1%}")
            st.write(f"🎯 **Jogos encontrados**: {len(jogos)}")

            cols_padrao = ['Ranking', 'Home', 'Away', 'League', score_col, 'M_H', 'M_A', 'Recomendacao', 'Quadrant_Dist_3D']
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]

            jogos_ordenados = jogos.sort_values(score_col, ascending=False)

            st.dataframe(
                jogos_ordenados[cols_padrao]
                .head(10)
                .style.format({
                    score_col: '{:.1%}',
                    'M_H': '{:.2f}',
                    'M_A': '{:.2f}',
                    'Quadrant_Dist_3D': '{:.2f}'
                })
                .background_gradient(subset=[score_col], cmap='RdYlGn')
                .background_gradient(subset=['M_H', 'M_A'], cmap='coolwarm'),
                use_container_width=True
            )
            st.write("---")

# ---------------- ESTRATÉGIAS AVANÇADAS 3D ----------------
def gerar_estrategias_3d_16_quadrantes(df):
    st.markdown("### 🎯 Estratégias 3D por Categoria")

    estrategias_3d = {
        'Fav Forte + Momentum': {
            'descricao': '**Favoritos Fortes com Momentum Positivo** - Alta aggression + handscore + momentum',
            'quadrantes': [1, 2, 3, 4],
            'momentum_min': 0.5,
            'estrategia': 'Apostar fortemente, especialmente contra underdogs com momentum negativo',
            'confianca': 'Muito Alta'
        },
        'Fav Moderado + Momentum': {
            'descricao': '**Favoritos Moderados em Ascensão** - Aggression positiva + momentum positivo', 
            'quadrantes': [5, 6, 7, 8],
            'momentum_min': 0.3,
            'estrategia': 'Buscar value, ótimos quando momentum confirma a tendência',
            'confianca': 'Alta'
        },
        'Under Moderado - Momentum': {
            'descricao': '**Underdogs Moderados em Decadência** - Aggression negativa + momentum negativo',
            'quadrantes': [9, 10, 11, 12],
            'momentum_max': -0.3,
            'estrategia': 'Apostar contra, risco elevado de não cobrir handicap',
            'confianca': 'Média-Alta'
        },
        'Under Forte - Momentum': {
            'descricao': '**Underdogs Fortes em Crise** - Aggression muito negativa + momentum negativo',
            'quadrantes': [13, 14, 15, 16], 
            'momentum_max': -0.5,
            'estrategia': 'Evitar completamente ou apostar contra em situações específicas',
            'confianca': 'Média'
        }
    }

    for categoria, info in estrategias_3d.items():
        st.write(f"**{categoria}**")
        st.write(f"📋 {info['descricao']}")
        st.write(f"🎯 Estratégia: {info['estrategia']}")
        st.write(f"📊 Confiança: {info['confianca']}")

        if 'momentum_min' in info:
            jogos_categoria = df[
                (df['Quadrante_Home'].isin(info['quadrantes']) | 
                 df['Quadrante_Away'].isin(info['quadrantes'])) &
                ((df['M_H'] >= info['momentum_min']) | (df['M_A'] >= info['momentum_min']))
            ]
        elif 'momentum_max' in info:
            jogos_categoria = df[
                (df['Quadrante_Home'].isin(info['quadrantes']) | 
                 df['Quadrante_Away'].isin(info['quadrantes'])) &
                ((df['M_H'] <= info['momentum_max']) | (df['M_A'] <= info['momentum_max']))
            ]
        else:
            jogos_categoria = df[
                df['Quadrante_Home'].isin(info['quadrantes']) | 
                df['Quadrante_Away'].isin(info['quadrantes'])
            ]

        if not jogos_categoria.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jogos Encontrados", len(jogos_categoria))
            with col2:
                avg_score = jogos_categoria['Quadrante_ML_Score_Main'].mean()
                st.metric("Score Médio", f"{avg_score:.1%}")
            with col3:
                high_value = len(jogos_categoria[jogos_categoria['Quadrante_ML_Score_Main'] >= 0.60])
                st.metric("Alto Valor", high_value)

        st.write("---")

# ---------------- FUNÇÕES LIVE / PROFIT ----------------
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
        st.error("❌ 'Score_Final_3D' não encontrado – gere o score antes de filtrar.")
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

# ---------------- EXECUÇÃO PRINCIPAL 3D CORRIGIDA ----------------
if not history.empty:
    modelo_home_cb, modelo_away_cb, games_today = treinar_modelo_inteligente(history, games_today)
    st.success("✅ Modelo 3D Inteligente (CatBoost) treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D")

# ---------------- EXIBIÇÃO DOS RESULTADOS 3D CORRIGIDOS ----------------
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML Inteligente")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_3d = games_today.copy()

    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
    ranking_3d = gerar_score_combinado_3d_16(ranking_3d)
    ranking_3d = update_real_time_data_3d(ranking_3d)

    st.markdown("## 📡 Live Score Monitor - Sistema 3D Inteligente")
    live_summary_3d = generate_live_summary_3d(ranking_3d)
    st.json(live_summary_3d)

    ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

    colunas_3d = [
        'League', 'Time',
        'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
        'Recomendacao', 'ML_Side', 'Asian_Line_Decimal', 'Cluster3D_Desc',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
        'Prob_Home', 'Prob_Away', 'ProbDiff', 'BetSignal_CB', 'ProbDiffThreshold_Usado',
        'Score_Final_3D', 'Classificacao_Potencial_3D',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        'M_H', 'M_A', 'Quadrant_Dist_3D', 'Momentum_Diff',
        'Media_Score_Home', 'Media_Score_Away', 
        'Regressao_Force_Home', 'Regressao_Force_Away',
        'eh_fav_forte_com_momentum', 'eh_under_forte_sem_momentum',
        'eh_forte_melhora_home', 'eh_forte_melhora_away',
        'score_confianca_composto',
        'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
    ]

    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    st.write(f"🎯 Exibindo {len(ranking_3d)} jogos ordenados por Score 3D Inteligente")

    st.dataframe(
        estilo_tabela_3d_quadrantes(ranking_3d[cols_finais_3d])
        .format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Home_Red': '{:.0f}',
            'Away_Red': '{:.0f}',
            'Profit_Quadrante': '{:.2f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Prob_Home': '{:.1%}',
            'Prob_Away': '{:.1%}',
            'ProbDiff': '{:.1%}',
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
        use_container_width=True,
        height=800
    )

    analisar_padroes_3d_quadrantes_16_dual(ranking_3d)
    gerar_estrategias_3d_16_quadrantes(ranking_3d)

else:
    st.error("""
    ❌ **Não foi possível gerar a tabela de confrontos 3D**
    
    **Possíveis causas:**
    - Dados de hoje vazios
    - Colunas do modelo ML não foram criadas
    - Erro no processamento dos dados
    """)

    if games_today.empty:
        st.warning("📭 games_today está vazio")
    else:
        st.info(f"📊 games_today tem {len(games_today)} linhas")
        st.info(f"🔍 Colunas: {list(games_today.columns)}")

# ---------------- RESUMO EXECUTIVO 3D ----------------
def resumo_3d_16_quadrantes_hoje(df):
    st.markdown("### 📋 Resumo Executivo - Sistema 3D Inteligente Hoje")

    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)

    alto_potencial_3d = len(df[df['Classificacao_Potencial_3D'] == '🌟 ALTO POTENCIAL 3D'])
    valor_solido_3d = len(df[df['Classificacao_Potencial_3D'] == '💼 VALOR SOLIDO 3D'])

    alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
    alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])

    fav_forte_momentum = len(df[df['eh_fav_forte_com_momentum'] == 1])
    forte_melhora = len(df[(df['eh_forte_melhora_home'] == 1) | (df['eh_forte_melhora_away'] == 1)])
    conflitos_value = len(df[(df['conflito_agg_regressao_home'] == 1) | (df['conflito_agg_regressao_away'] == 1)])

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
        st.metric("📊 Score Confiança Médio", f"{df['score_confianca_composto'].mean():.2f}")

    st.markdown("#### 📊 Distribuição de Recomendações 3D Inteligentes")
    if 'Recomendacao' in df.columns:
        dist_recomendacoes = df['Recomendacao'].value_counts()
        st.dataframe(dist_recomendacoes, use_container_width=True)

if not games_today.empty and 'Classificacao_Potencial_3D' in games_today.columns:
    resumo_3d_16_quadrantes_hoje(games_today)
