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

# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------
LIVESCORE_FOLDER = "LiveScore"


# ========================= FUNÇÕES BÁSICAS =========================

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


# ========================= ASIAN LINE =========================

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
            return -num  # Away → Home
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
        return -result  # Away → Home
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


# ========================= LIMPEZA & MOMENTUM (M / MT) =========================

def validar_e_limpar_base(df, nome="DF"):
    """Garante que todas as colunas essenciais existam e não tenham NaN."""
    df = df.copy()

    colunas_essenciais = [
        'League', 'Home', 'Away',
        'HandScore_Home', 'HandScore_Away',
        'Aggression_Home', 'Aggression_Away',
        'Goals_H_FT', 'Goals_A_FT',
        'Asian_Line'
    ]

    for col in colunas_essenciais:
        if col not in df.columns:
            st.warning(f"{nome}: Criando coluna ausente: {col}")
            df[col] = 0.0

    # Colunas numéricas principais
    colunas_numericas = [
        'HandScore_Home', 'HandScore_Away',
        'Aggression_Home', 'Aggression_Away',
        'Goals_H_FT', 'Goals_A_FT'
    ]

    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preencher NaNs com neutro
    df = df.fillna(0.0)

    st.success(f"🔒 {nome}: Dados validados e NaNs eliminados ({len(df)} linhas)")
    return df


def calcular_zscores_detalhados(df):
    """
    Calcula Z-scores a partir do HandScore:
    - M_H, M_A: Z-score do time em relação à liga (performance relativa)
    - MT_H, MT_A: Z-score do time em relação a si mesmo (consistência)
    """
    df = df.copy()

    st.info("📊 Calculando Z-scores (M / MT) a partir do HandScore...")

    # Garantir colunas de HandScore
    for col in ['HandScore_Home', 'HandScore_Away']:
        if col not in df.columns:
            st.error(f"❌ Coluna ausente: {col}. Preenchendo com 0.")
            df[col] = 0.0

    # 1. Z-SCORE POR LIGA (M_H, M_A)
    if 'League' in df.columns:
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

        st.success(f"✅ Z-score por liga calculado para {len(df)} jogos")
    else:
        st.warning("⚠️ Coluna League não encontrada")
        df['M_H'] = 0.0
        df['M_A'] = 0.0

    # 2. Z-SCORE POR TIME (MT_H, MT_A)
    if 'Home' in df.columns and 'Away' in df.columns:
        home_team_stats = df.groupby('Home').agg({
            'HandScore_Home': ['mean', 'std']
        }).round(3)
        home_team_stats.columns = ['HT_mean', 'HT_std']
        home_team_stats['HT_std'] = home_team_stats['HT_std'].replace(0, 1)

        away_team_stats = df.groupby('Away').agg({
            'HandScore_Away': ['mean', 'std']
        }).round(3)
        away_team_stats.columns = ['AT_mean', 'AT_std']
        away_team_stats['AT_std'] = away_team_stats['AT_std'].replace(0, 1)

        df = df.merge(home_team_stats, left_on='Home', right_index=True, how='left')
        df = df.merge(away_team_stats, left_on='Away', right_index=True, how='left')

        df['MT_H'] = (df['HandScore_Home'] - df['HT_mean']) / df['HT_std']
        df['MT_A'] = (df['HandScore_Away'] - df['AT_mean']) / df['AT_std']

        df['MT_H'] = np.clip(df['MT_H'], -5, 5)
        df['MT_A'] = np.clip(df['MT_A'], -5, 5)

        st.success(f"✅ Z-score por time calculado para {len(df)} jogos")

        df = df.drop(
            ['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std',
             'HT_mean', 'HT_std', 'AT_mean', 'AT_std'],
            axis=1, errors='ignore'
        )
    else:
        st.warning("⚠️ Colunas Home ou Away não encontradas")
        df['MT_H'] = 0.0
        df['MT_A'] = 0.0

    return df


def clean_features_for_training(X):
    """
    Remove infinitos, NaNs e valores extremos das features
    """
    X_clean = X.copy()

    # Converter para DataFrame se for numpy array
    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean, columns=X.columns if hasattr(X, 'columns') else range(X.shape[1]))

    # 1. Substituir infinitos por NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    # 2. Contar problemas antes da limpeza
    inf_count = (X_clean == np.inf).sum().sum() + (X_clean == -np.inf).sum().sum()
    nan_count = X_clean.isna().sum().sum()

    if inf_count > 0 or nan_count > 0:
        st.warning(f"⚠️ Encontrados {inf_count} infinitos e {nan_count} NaNs nas features")

    # 3. Preencher com mediana
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)

    # 4. Limitar outliers
    for col in X_clean.columns:
        if X_clean[col].dtype in [np.float64, np.float32]:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)

    # 5. Verificação final
    final_inf_count = (X_clean == np.inf).sum().sum() + (X_clean == -np.inf).sum().sum()
    final_nan_count = X_clean.isna().sum().sum()

    if final_inf_count > 0 or final_nan_count > 0:
        st.error(f"❌ Ainda existem {final_inf_count} infinitos e {final_nan_count} NaNs")
        X_clean = X_clean.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)

    st.success(f"✅ Features limpas: {X_clean.shape}")

    return X_clean


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
        st.info("ℹ️ Apenas 1 jogo encontrado - cluster único criado")
        return df

    n_clusters = min(max_clusters, max(2, n_samples // 3))

    st.info(f"🎯 Clusterização: {n_samples} amostras → {n_clusters} clusters")

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
        'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}',
        'Tamanho': '{:.0f}'
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
        cluster_descriptions[i] = classificar_cluster(centroid['dx'], centroid['dy'], centroid['dz'])

    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(cluster_descriptions)

    cluster_stats = df.groupby('Cluster3D_Label').agg({
        'dx': 'mean', 'dy': 'mean', 'dz': 'mean',
        'Cluster3D_Desc': 'first'
    }).round(3)

    st.markdown("### 📊 Estatísticas dos Clusters")
    st.dataframe(cluster_stats)

    return df


# ========================= CARREGAMENTO DE DADOS =========================

st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options) - 1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

games_today_raw = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today_raw = filter_leagues(games_today_raw)


@st.cache_data(ttl=3600)
def load_cached_data(selected_file_):
    games_today_local = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file_))
    games_today_local = filter_leagues(games_today_local)

    history_local = filter_leagues(load_all_games(GAMES_FOLDER))
    history_local = history_local.copy()

    return games_today_local, history_local


games_today, history = load_cached_data(selected_file)


# ========================= LIVE SCORE INTEGRATION =========================

def load_and_merge_livescore(games_today_, selected_date_str_):
    """Carrega e faz merge dos dados do Live Score"""
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str_}.csv")

    games_today_ = setup_livescore_columns(games_today_)

    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)

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
            return games_today_
        else:
            games_today_ = games_today_.merge(
                results_df,
                left_on='Id',
                right_on='Id',
                how='left',
                suffixes=('', '_RAW')
            )

            games_today_['Goals_H_Today'] = games_today_['home_goal']
            games_today_['Goals_A_Today'] = games_today_['away_goal']
            games_today_.loc[games_today_['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

            games_today_['Home_Red'] = games_today_['home_red']
            games_today_['Away_Red'] = games_today_['away_red']

            st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
            return games_today_
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str_}")
        return games_today_


games_today = load_and_merge_livescore(games_today, selected_date_str)

# History já veio filtrado; apenas garantir cópia
history = history.copy()

# ========================= ASIAN LINE DECIMAL =========================

history['Asian_Line_Decimal'] = history.get('Asian_Line', np.nan).apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today.get('Asian_Line', np.nan).apply(convert_asian_line_to_decimal)

history = history.dropna(subset=['Asian_Line_Decimal'])
st.info(f"📊 Histórico com Asian Line válida: {len(history)} jogos")

# ========================= FILTRO TEMPORAL (ANTI-LEAKAGE) =========================

if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

# ========================= LIMPEZA E MOMENTUM =========================

history = validar_e_limpar_base(history, "History")
games_today = validar_e_limpar_base(games_today, "GamesToday")

history = calcular_zscores_detalhados(history)
games_today = calcular_zscores_detalhados(games_today)

# ========================= TARGET AH HISTÓRICO =========================

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Target_AH_Home"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=False) > 0.5 else 0, axis=1
)

# ========================= SISTEMA 3D DE 16 QUADRANTES =========================

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
    """Classifica Aggression e HandScore em um dos 16 quadrantes"""
    if pd.isna(agg) or pd.isna(hs):
        return 0

    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id

    return 0


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


# ========================= REGRESSÃO À MÉDIA =========================

def calcular_regressao_media(df):
    """
    Calcula tendência de regressão à média baseada em:
    - M_H, M_A: Z-score do momentum na liga
    - MT_H, MT_A: Z-score do momentum do time
    """
    df = df.copy()

    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])

    df['Regressao_Force_Home'] = -np.sign(df['M_H']) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df['M_A']) * (df['Extremidade_Away'] ** 0.7)

    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))

    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] +
                              0.4 * (1 - df['Aggression_Home']))
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] +
                              0.4 * (1 - df['Aggression_Away']))

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


history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)


# ========================= DISTÂNCIAS 3D =========================

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

    df['Quadrant_Dist_3D'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

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

    df['Magnitude_3D'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return df


games_today = calcular_distancias_3d(games_today)
history = calcular_distancias_3d(history)


# ========================= PLOT 16 QUADRANTES 2D =========================

def plot_quadrantes_16(df, side="Home"):
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


# ========================= VISUALIZAÇÃO 3D INTERATIVA =========================

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
    magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

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


# ========================= FILTRO REGRESSÃO À MÉDIA =========================

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


# ========================= PLOT 3D FIXO =========================

def create_fixed_3d_plot(df_plot, n_to_show, selected_league_):
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
                f"<b>{row.get('Home', 'N/A')} vs {row.get('Away', 'N/A')}</b><br>"
                f"🏆 {row.get('League', 'N/A')}<br>"
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
    if selected_league_ != "⚽ Todas as ligas":
        titulo_3d += f" | {selected_league_}"

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


# ========================= CLUSTERIZAÇÃO 3D APÓS MOMENTUM =========================

history = aplicar_clusterizacao_3d(history)
games_today = aplicar_clusterizacao_3d(games_today)


# ========================= FEATURES INTELIGENTES & MODELO =========================

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


def treinar_modelo_inteligente(history_, games_today_):
    history_ = adicionar_features_inteligentes_ml(history_)
    games_today_ = adicionar_features_inteligentes_ml(games_today_)

    history_ = calcular_distancias_3d(history_)
    games_today_ = calcular_distancias_3d(games_today_)

    history_ = aplicar_clusterizacao_3d(history_)
    games_today_ = aplicar_clusterizacao_3d(games_today_)

    ligas_dummies = pd.get_dummies(history_['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history_['Cluster3D_Label'], prefix='C3D')

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

    available_3d = [f for f in features_3d if f in history_.columns]
    available_regressao = [f for f in features_regressao if f in history_.columns]
    available_inteligentes = [f for f in features_inteligentes if f in history_.columns]

    extras_3d = history_[available_3d].fillna(0)
    extras_regressao = history_[available_regressao].fillna(0)
    extras_inteligentes = history_[available_inteligentes].fillna(0)

    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, extras_regressao, extras_inteligentes], axis=1)

    X = clean_features_for_training(X)

    y_home = history_['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    ligas_today = pd.get_dummies(games_today_['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today_['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today_[available_3d].fillna(0)
    extras_regressao_today = games_today_[available_regressao].fillna(0)
    extras_inteligentes_today = games_today_[available_inteligentes].fillna(0)

    X_today = pd.concat([ligas_today, clusters_today, extras_today, extras_regressao_today, extras_inteligentes_today], axis=1)
    X_today = clean_features_for_training(X_today)

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today_['Prob_Home'] = proba_home
    games_today_['Prob_Away'] = proba_away
    games_today_['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today_['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today_['Quadrante_ML_Score_Home'] = games_today_['Prob_Home']
    games_today_['Quadrante_ML_Score_Away'] = games_today_['Prob_Away']
    games_today_['Quadrante_ML_Score_Main'] = games_today_['ML_Confidence']

    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features (Com Inteligência)")
    st.dataframe(importances.head(20).to_frame("Importância"), use_container_width=True)

    inteligentes_no_top = len([f for f in importances.head(15).index if any(keyword in f for keyword in ['eh_', 'conflito_', 'momentum_', 'padrao_', 'score_'])])
    st.info(f"🧠 Features Inteligentes no Top 15: {inteligentes_no_top}")

    st.success("✅ Modelo Inteligente treinado com sucesso!")
    return model_home, games_today_


# ========================= INDICADORES 3D & SCORING =========================

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


# ========================= LIVE / PROFIT =========================

def determine_handicap_result(row):
    """Determina se o HOME cobriu o handicap asiático (perspectiva do Home)."""
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
    """Verifica se a recomendação bateu com o resultado do handicap."""
    if pd.isna(rec) or handicap_result is None:
        return None

    rec = str(rec).upper()

    if any(k in rec for k in ['HOME', 'VALUE NO HOME', 'FAVORITO HOME', 'MODELO CONFIA HOME']):
        return handicap_result in ["HOME_COVERED", "HALF_HOME_COVERED"]

    if any(k in rec for k in ['AWAY', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
        return handicap_result in ["HOME_NOT_COVERED", "HALF_HOME_NOT_COVERED"]

    return None


def calculate_handicap_profit(rec, handicap_result, odd_home, odd_away, asian_line_decimal):
    """
    Calcula lucro unitário do handicap asiático (stake = 1, odds já líquidas),
    baseado apenas no resultado categórico (HOME_COVERED, HALF_HOME_COVERED, etc.).
    """
    if pd.isna(rec) or handicap_result is None:
        return 0.0

    rec = str(rec).upper()
    is_home_bet = any(k in rec for k in ['HOME', 'FAVORITO HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away_bet = any(k in rec for k in ['AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])

    if not (is_home_bet or is_away_bet):
        return 0.0

    odd = odd_home if is_home_bet else odd_away
    result = str(handicap_result).upper()

    if result == "PUSH":
        return 0.0

    if result == "HALF_HOME_COVERED":
        return odd / 2 if is_home_bet else -0.5
    if result == "HALF_HOME_NOT_COVERED":
        return -0.5 if is_home_bet else odd / 2

    if result == "HOME_COVERED":
        return odd if is_home_bet else -1.0
    if result == "HOME_NOT_COVERED":
        return -1.0 if is_home_bet else odd

    return 0.0


def update_real_time_data_3d(df):
    """
    Atualiza resultados e lucro considerando APENAS linhas com Score_Final_3D acima do limite.
    Usa odds asiáticas Odd_H_Asi / Odd_A_Asi.
    """
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
    """Gera resumo em tempo real para sistema 3D"""
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


# ========================= RESUMO EXECUTIVO 3D =========================
def resumo_3d_16_quadrantes_hoje(df):
    """Resumo executivo dos 16 quadrantes 3D de hoje"""
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

def inferir_probabilidades_ml_dual(df):
    """Aplica modelo ML para gerar scores Home/Away e a recomendação final"""
    df = df.copy()

    modelo_home_path = os.path.join(MODELS_FOLDER, "model_home.pkl")
    modelo_away_path = os.path.join(MODELS_FOLDER, "model_away.pkl")

    if not os.path.exists(modelo_home_path) or not os.path.exists(modelo_away_path):
        st.error("❌ Modelos ML Home/Away não encontrados na pasta Models/")
        df['Quadrante_ML_Score_Home'] = 0.5
        df['Quadrante_ML_Score_Away'] = 0.5
        df['Recomendacao'] = "HOME"  # neutro
        return df

    modelo_home = joblib.load(modelo_home_path)
    modelo_away = joblib.load(modelo_away_path)

    features = [
        col for col in df.columns
        if col not in ['Home', 'Away', 'League', 'Date',
                       'Recomendacao', 'Quadrante_ML_Score_Home',
                       'Quadrante_ML_Score_Away', 'Score_Final_3D']
    ]

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Previsões ML
    if hasattr(modelo_home, "predict_proba"):
        df['Quadrante_ML_Score_Home'] = modelo_home.predict_proba(X)[:, 1]
    else:
        df['Quadrante_ML_Score_Home'] = modelo_home.predict(X)

    if hasattr(modelo_away, "predict_proba"):
        df['Quadrante_ML_Score_Away'] = modelo_away.predict_proba(X)[:, 1]
    else:
        df['Quadrante_ML_Score_Away'] = modelo_away.predict(X)

    # Normalizar probabilidades
    soma_prob = df['Quadrante_ML_Score_Home'] + df['Quadrante_ML_Score_Away']
    df['Quadrante_ML_Score_Home'] = df['Quadrante_ML_Score_Home'] / soma_prob
    df['Quadrante_ML_Score_Away'] = df['Quadrante_ML_Score_Away'] / soma_prob

    # Indicador final de recomendação
    df['Recomendacao'] = np.where(
        df['Quadrante_ML_Score_Home'] >= df['Quadrante_ML_Score_Away'],
        'HOME',
        'AWAY'
    )

    return df




# ========================= EXECUÇÃO FINAL - EXIBIÇÃO DOS RESULTADOS =========================
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes (ML Inteligente)")

if games_today.empty:
    st.error("📭 Nenhum jogo disponível hoje. Verifique a pasta GamesDay.")
    st.stop()

ranking_3d = games_today.copy()

ranking_3d = inferir_probabilidades_ml_dual(ranking_3d)
ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
ranking_3d = gerar_score_combinado_3d_16(ranking_3d)
ranking_3d = update_real_time_data_3d(ranking_3d)
ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)


st.success(f"🎯 {len(ranking_3d)} jogos processados pelo Sistema 3D Inteligente")

# 📌 Resumo executivo
resumo_3d_16_quadrantes_hoje(ranking_3d)

# 📋 Tabela filtrável
num_show = st.slider(
    "Quantos jogos exibir na tabela?", 
    min_value=5,
    max_value=len(ranking_3d), 
    value=min(40, len(ranking_3d))
)

df_show = ranking_3d.head(num_show).copy()

st.markdown("### 📋 Lista de Recomendações - Ordenado por Score 3D")
st.dataframe(
    estilo_tabela_3d_quadrantes(df_show)
    .format({
        'Quadrante_ML_Score_Home': '{:.1%}',
        'Quadrante_ML_Score_Away': '{:.1%}',
        'Score_Final_3D': '{:.1f}',
        'Profit_Quadrante': '{:.2f}'
    }, na_rep="-"),
    use_container_width=True
)

if "Profit_Quadrante" in ranking_3d.columns and ranking_3d['Profit_Quadrante'].notna().any():
    st.success("📈 Monitorando lucro em tempo real!")
else:
    st.info("🕗 Aguardando gols / odds do LiveScore para lucro em tempo real...")

st.markdown("---")
st.markdown("🏁 Fim da execução — Sistema 3D Inteligente 🚀")

