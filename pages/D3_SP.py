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

def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
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
        n_init=20           # mais robusto
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

# Preparar dados para visualização 3D
df_plot = df_filtered.nlargest(n_to_show, "Quadrant_Dist_3D").reset_index(drop=True)

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
                f"🔥 MT_H: {row.get('MT_H', np.nan):.2f} | MT_A: {row.get('MT_A', np.nan):.2f}"
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
                eye=dict(x=1.8, y=1.8, z=1.2),  # POSIÇÃO FIXA DA CÂMERA
                up=dict(x=0, y=0, z=1),
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



# # ---------------- MODELO ML 3D PARA 16 QUADRANTES ----------------
# def treinar_modelo_3d_quadrantes_16_dual(history, games_today):
#     """
#     Treina modelo ML 3D para Home e Away com base nos 16 quadrantes + Momentum
#     Agora inclui projeções trigonométricas sin/cos.
#     """
#     # Garantir cálculo das distâncias 3D
#     history = calcular_distancias_3d(history)
#     games_today = calcular_distancias_3d(games_today)

#     # Features categóricas (quadrantes + liga)
#     quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
#     quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
#     ligas_dummies = pd.get_dummies(history['League'], prefix='League')
#     clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')  # 🆕 NOVO

#     # Features 3D contínuas (agora com sin/cos)
#     extras_3d = history[[
#         'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
#         'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
#         'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
#         'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
#         'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',  # 🆕 novos vetores compostos
#         'Vector_Sign',                               # 🆕 direção espacial
#         'Momentum_Diff', 'Magnitude_3D'
#        # 'M_H', 'M_A', 'MT_H', 'MT_A'
#     ]].fillna(0)

#     # Combinar todas as features
#     X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies,clusters_dummies, extras_3d], axis=1)

#     # Targets
#     y_home = history['Target_AH_Home']
#     y_away = 1 - y_home

#     # Modelos RandomForest dual
#     model_home = RandomForestClassifier(
#         n_estimators=500, max_depth=12, random_state=42,
#         class_weight='balanced_subsample', n_jobs=-1
#     )
#     model_away = RandomForestClassifier(
#         n_estimators=500, max_depth=12, random_state=42,
#         class_weight='balanced_subsample', n_jobs=-1
#     )

#     model_home.fit(X, y_home)
#     model_away.fit(X, y_away)

#     # Preparar dados de hoje com as mesmas features
#     qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
#     qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
#     ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
#     clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
#     extras_today = games_today[[
#         'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
#         'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
#         'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
#         'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
#         'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',  # 🆕 novos vetores compostos
#         'Vector_Sign',                               # 🆕 direção espacial
#         'Momentum_Diff', 'Magnitude_3D'
#         #'M_H', 'M_A', 'MT_H', 'MT_A'
#     ]].fillna(0)

#     X_today = pd.concat([qh_today, qa_today, ligas_today, clusters_today, extras_today], axis=1)

#     # Previsões
#     probas_home = model_home.predict_proba(X_today)[:, 1]
#     probas_away = model_away.predict_proba(X_today)[:, 1]

#     games_today['Quadrante_ML_Score_Home'] = probas_home
#     games_today['Quadrante_ML_Score_Away'] = probas_away
#     games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
#     games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

#     # Importância das features
#     importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
#     top_feats = importances.head(15)
#     st.markdown("### 🔍 Top Features mais importantes (Modelo 3D HOME)")
#     st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)

#     features_3d_no_top = [feat for feat in top_feats.index if any(k in feat for k in ['Sin', 'Cos', 'Dist_3D', 'Momentum'])]
#     st.info(f"📊 Features vetoriais 3D (sin/cos + momentum) no Top 15: {len(features_3d_no_top)}")

#     st.success("✅ Modelo 3D dual (Home/Away) atualizado com vetores sin/cos!")
#     return model_home, model_away, games_today


# def treinar_modelo_3d_quadrantes_16_dual(history, games_today):
#     """
#     Treina modelo ML 3D (Home/Away) sem usar Quadrantes fixos.
#     Mantém clusters + features vetoriais (sin/cos, momentum, magnitude, etc).
#     """
#     # Recalcular features 3D
#     history = calcular_distancias_3d(history)
#     games_today = calcular_distancias_3d(games_today)
#     history = aplicar_clusterizacao_3d(history)
#     games_today = aplicar_clusterizacao_3d(games_today)

#     # Features categóricas (liga + cluster)
#     ligas_dummies = pd.get_dummies(history['League'], prefix='League')
#     clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

#     # Features contínuas vetoriais
#     extras_3d = history[[
#         'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
#         'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
#         'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
#         'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
#         'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
#         'Vector_Sign', 'Magnitude_3D'
#     ]].fillna(0)

#     # Combinar todas as features
#     X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)

#     # Target
#     y_home = history['Target_AH_Home']


#     # Modelos RF dual
#     model_home = RandomForestClassifier(
#         n_estimators=500, max_depth=12, random_state=42,
#         class_weight='balanced_subsample', n_jobs=-1
#     )
   

#     model_home.fit(X, y_home)

#     # Prepara dados de hoje
#     ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
#     clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
#     extras_today = games_today[extras_3d.columns].fillna(0)

#     X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

#     # Previsões
#     probas_home = model_home.predict_proba(X_today)[:, 1]


#     games_today['Quadrante_ML_Score_Home'] = probas_home

#     games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home)
#     games_today['ML_Side'] = np.where(probas_home > 0.55, 'HOME', 'AWAY')

#     # Importância das features
#     importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
#     top_feats = importances.head(20)

#     st.markdown("### 🔍 Top Features (sem quadrantes)")
#     st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)

#     st.info(f"📊 Features vetoriais e clusters no Top 20: {len([f for f in top_feats.index if 'Sin' in f or 'Cos' in f or 'Cluster' in f])}")

#     st.success("✅ Modelo 3D treinado sem quadrantes fixos – apenas features vetoriais + clusters.")
#     return model_home, model_away, games_today

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import streamlit as st
import pandas as pd

def treinar_modelo_3d_quadrantes_16_dual(history, games_today):
    """
    Treina um único modelo ML 3D (lado Home) e deriva o lado Away por complemento.
    Remove redundância do modelo dual, mantendo consistência probabilística.
    """

    # ----------------------------
    # 🧩 Garantir features 3D e clusters
    # ----------------------------
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    # ----------------------------
    # 🧠 Feature Engineering
    # ----------------------------
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

    extras_3d = history[features_3d].fillna(0)
    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)

    # ----------------------------
    # 🎯 Target e Treinamento
    # ----------------------------
    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    # ----------------------------
    # 🔮 Previsões (complemento lógico)
    # ----------------------------
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d].fillna(0)

    X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)

    # ----------------------------
    # 📊 Importância de Features
    # ----------------------------
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features (Modelo Único – Home)")
    st.dataframe(importances.head(20).to_frame("Importância"), use_container_width=True)

    st.success("✅ Modelo 3D treinado apenas com lado HOME (sem redundância).")
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

    # 3. RECOMENDAÇÃO DE APOSTA 3D PARA 16 QUADRANTES
    def gerar_recomendacao_3d_16_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']
        momentum_h = row.get('M_H', 0)
        momentum_a = row.get('M_A', 0)

        # Padrões 3D específicos incorporando momentum
        if 'Fav Forte' in home_q and 'Under Forte' in away_q and momentum_h > 1.0:
            return f'💪 FAVORITO HOME SUPER FORTE (+Momentum) ({score_home:.1%})'
        elif 'Under Forte' in home_q and 'Fav Forte' in away_q and momentum_a > 1.0:
            return f'💪 FAVORITO AWAY SUPER FORTE (+Momentum) ({score_away:.1%})'
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
        elif 'Neutro' in home_q and score_away >= 0.58 and momentum_a > 0:
            return f'🔄 AWAY EM NEUTRO (+Momentum) ({score_away:.1%})'
        elif 'Neutro' in away_q and score_home >= 0.58 and momentum_h > 0:
            return f'🔄 HOME EM NEUTRO (+Momentum) ({score_home:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_3d_16_dual, axis=1)

    # 4. RANKING POR MELHOR PROBABILIDADE 3D
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- EXECUÇÃO PRINCIPAL 3D ----------------
# Executar treinamento 3D
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_3d_quadrantes_16_dual(history, games_today)
    st.success("✅ Modelo 3D dual com 16 quadrantes treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D")

# ---------------- ANÁLISE DE PADRÕES 3D PARA 16 QUADRANTES ----------------
def analisar_padroes_3d_quadrantes_16_dual(df):
    """Analisa padrões recorrentes nas combinações 3D de 16 quadrantes"""
    st.markdown("### 🔍 Análise de Padrões 3D por Combinação")

    # Padrões prioritários 3D para 16 quadrantes
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

    # Ordenar padrões por prioridade
    padroes_ordenados = sorted(padroes_3d.items(), key=lambda x: x[1]['prioridade'])

    for padrao, info in padroes_ordenados:
        # Buscar jogos que correspondem ao padrão 3D
        home_q, away_q = padrao.split(' vs ')[0], padrao.split(' vs ')[1]
        
        # Simplificar busca por quadrantes (remover condições de momentum do texto)
        home_q_base = home_q.split(' (')[0] if ' (' in home_q else home_q
        away_q_base = away_q.split(' (')[0] if ' (' in away_q else away_q

        jogos = df[
            (df['Quadrante_Home_Label'] == home_q_base) & 
            (df['Quadrante_Away_Label'] == away_q_base)
        ]

        # Aplicar filtros de momentum
        if 'momentum_min_home' in info:
            jogos = jogos[jogos['M_H'] >= info['momentum_min_home']]
        if 'momentum_max_home' in info:
            jogos = jogos[jogos['M_H'] <= info['momentum_max_home']]
        if 'momentum_min_away' in info:
            jogos = jogos[jogos['M_A'] >= info['momentum_min_away']]
        if 'momentum_max_away' in info:
            jogos = jogos[jogos['M_A'] <= info['momentum_max_away']]

        # Filtrar por score mínimo
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

            # Colunas para exibir
            cols_padrao = ['Ranking', 'Home', 'Away', 'League', score_col, 'M_H', 'M_A', 'Recomendacao', 'Quadrant_Dist_3D']
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]

            # Ordenar por score
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

# ---------------- ESTRATÉGIAS AVANÇADAS 3D PARA 16 QUADRANTES ----------------
def gerar_estrategias_3d_16_quadrantes(df):
    """Gera estratégias específicas baseadas nos 16 quadrantes 3D"""
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

        # Filtrar jogos da categoria
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

# ---------------- SISTEMA DE SCORING 3D PARA 16 QUADRANTES ----------------
def calcular_pontuacao_3d_quadrante_16(quadrante_id, momentum=0):
    """Calcula pontuação base 3D para cada quadrante (0-100) considerando momentum"""
    scores_base = {
        # Fav Forte: alta pontuação
        1: 85, 2: 80, 3: 75, 4: 70,
        # Fav Moderado: média-alta
        5: 70, 6: 65, 7: 60, 8: 55,
        # Under Moderado: média-baixa  
        9: 50, 10: 45, 11: 40, 12: 35,
        # Under Forte: baixa pontuação
        13: 35, 14: 30, 15: 25, 16: 20
    }
    
    base_score = scores_base.get(quadrante_id, 50)
    
    # Ajustar score base pelo momentum
    momentum_boost = momentum * 10  # +10 pontos por unidade de momentum
    adjusted_score = base_score + momentum_boost
    
    # Limitar entre 0-100
    return max(0, min(100, adjusted_score))

def gerar_score_combinado_3d_16(df):
    """Gera score combinado 3D considerando quadrantes e momentum"""
    df = df.copy()

    # Score base dos quadrantes ajustado pelo momentum
    df['Score_Base_Home'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Home'], x.get('M_H', 0)), axis=1
    )
    df['Score_Base_Away'] = df.apply(
        lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Away'], x.get('M_A', 0)), axis=1
    )

    # Score combinado (média ponderada)
    df['Score_Combinado_3D'] = (df['Score_Base_Home'] * 0.5 + df['Score_Base_Away'] * 0.3 + 
                               df['Quadrant_Dist_3D'] * 0.2)

    # Ajustar pelo ML Score 3D
    df['Score_Final_3D'] = df['Score_Combinado_3D'] * df['Quadrante_ML_Score_Main']

    # Classificar por potencial 3D
    conditions = [
        df['Score_Final_3D'] >= 60,
        df['Score_Final_3D'] >= 45, 
        df['Score_Final_3D'] >= 30,
        df['Score_Final_3D'] < 30
    ]
    choices = ['🌟 ALTO POTENCIAL 3D', '💼 VALOR SOLIDO 3D', '⚖️ NEUTRO 3D', '🔴 BAIXO POTENCIAL 3D']
    df['Classificacao_Potencial_3D'] = np.select(conditions, choices, default='⚖️ NEUTRO 3D')

    return df

# ---------------- EXIBIÇÃO DOS RESULTADOS 3D ----------------
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição 3D
    ranking_3d = games_today.copy()

    # Aplicar indicadores explicativos 3D
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)

    # Aplicar scoring combinado 3D
    ranking_3d = gerar_score_combinado_3d_16(ranking_3d)

    # ---------------- ATUALIZAR COM DADOS LIVE 3D ----------------
    def determine_handicap_result(row):
        """Determina se o HOME cobriu o handicap"""
        try:
            gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
            ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
            asian_line_decimal = row.get('Asian_Line_Decimal')
        except (ValueError, TypeError):
            return None

        if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_decimal):
            return None

        margin = gh - ga
        handicap_result = calc_handicap_result(margin, asian_line_decimal, invert=False)

        if handicap_result > 0.5:
            return "HOME_COVERED"
        elif handicap_result == 0.5:
            return "PUSH"
        else:
            return "HOME_NOT_COVERED"

    def check_handicap_recommendation_correct(rec, handicap_result):
        """Verifica se a recomendação estava correta"""
        if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
            return None

        rec = str(rec)

        if any(keyword in rec for keyword in ['HOME', 'Home', 'VALUE NO HOME', 'FAVORITO HOME']):
            return handicap_result == "HOME_COVERED"
        elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
            return handicap_result in ["HOME_NOT_COVERED", "PUSH"]

        return None

    def calculate_handicap_profit(rec, handicap_result, odds_row, asian_line_decimal):
        """Calcula profit para handicap asiático"""
        if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid' or pd.isna(asian_line_decimal):
            return 0

        rec = str(rec).upper()
        is_home_bet = any(k in rec for k in ['HOME', 'FAVORITO HOME', 'VALUE NO HOME'])
        is_away_bet = any(k in rec for k in ['AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])

        if not (is_home_bet or is_away_bet):
            return 0

        odd = odds_row.get('Odd_H_Asi', np.nan) if is_home_bet else odds_row.get('Odd_A_Asi', np.nan)
        if pd.isna(odd):
            return 0

        def split_line(line):
            frac = abs(line) % 1
            if frac == 0.25:
                base = math.floor(abs(line))
                base = base if line > 0 else -base
                return [base, base + (0.5 if line > 0 else -0.5)]
            elif frac == 0.75:
                base = math.floor(abs(line))
                base = base if line > 0 else -base
                return [base + (0.5 if line > 0 else -0.5), base + (1.0 if line > 0 else -1.0)]
            else:
                return [line]

        asian_line_for_eval = -asian_line_decimal if is_home_bet else asian_line_decimal
        lines = split_line(asian_line_for_eval)

        def single_profit(result):
            if result == "PUSH":
                return 0
            elif (is_home_bet and result == "HOME_COVERED") or (is_away_bet and result == "HOME_NOT_COVERED"):
                return odd
            elif (is_home_bet and result == "HOME_NOT_COVERED") or (is_away_bet and result == "HOME_COVERED"):
                return -1
            return 0

        if len(lines) == 2:
            p1 = single_profit(handicap_result)
            p2 = single_profit(handicap_result)
            return (p1 + p2) / 2
        else:
            return single_profit(handicap_result)

    def update_real_time_data_3d(df):
        """Atualiza todos os dados em tempo real para sistema 3D"""
        df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
        df['Quadrante_Correct'] = df.apply(
            lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
        )
        df['Profit_Quadrante'] = df.apply(
            lambda r: calculate_handicap_profit(r['Recomendacao'], r['Handicap_Result'], r, r['Asian_Line_Decimal']), axis=1
        )
        return df

    # Aplicar atualização em tempo real 3D
    ranking_3d = update_real_time_data_3d(ranking_3d)

    # ---------------- RESUMO LIVE 3D ----------------
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

    # Exibir resumo live 3D
    st.markdown("## 📡 Live Score Monitor - Sistema 3D")
    live_summary_3d = generate_live_summary_3d(ranking_3d)
    st.json(live_summary_3d)

    # Ordenar por score final 3D
    ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

    # Colunas para exibição 3D
    colunas_3d = [
        
        'League', 'Time',
        'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today','ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Score_Final_3D', 'Classificacao_Potencial_3D',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao',
        # Colunas 3D
        'M_H', 'M_A', 'Quadrant_Dist_3D', 'Momentum_Diff',
        # Colunas Live Score
        'Asian_Line_Decimal', 'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
    ]

    # Filtrar colunas existentes
    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    # Função de estilo para tabela 3D
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

        # Aplicar gradientes para colunas numéricas
        if 'Quadrante_ML_Score_Home' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')
        if 'Quadrante_ML_Score_Away' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')
        if 'Score_Final_3D' in df.columns:
            styler = styler.background_gradient(subset=['Score_Final_3D'], cmap='RdYlGn')
        if 'M_H' in df.columns:
            styler = styler.background_gradient(subset=['M_H', 'M_A'], cmap='coolwarm')

        return styler

    st.dataframe(
        estilo_tabela_3d_quadrantes(ranking_3d[cols_finais_3d].head(25))
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
            'Momentum_Diff': '{:.2f}'
        }, na_rep="-"),
        use_container_width=True
    )

    # ---------------- ANÁLISES ESPECÍFICAS 3D ----------------
    analisar_padroes_3d_quadrantes_16_dual(ranking_3d)
    gerar_estrategias_3d_16_quadrantes(ranking_3d)

else:
    st.info("⚠️ Aguardando dados para gerar ranking 3D de 16 quadrantes")

# ---------------- RESUMO EXECUTIVO 3D ----------------
def resumo_3d_16_quadrantes_hoje(df):
    """Resumo executivo dos 16 quadrantes 3D de hoje"""

    st.markdown("### 📋 Resumo Executivo - Sistema 3D Hoje")

    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)

    # Estatísticas de classificação 3D
    alto_potencial_3d = len(df[df['Classificacao_Potencial_3D'] == '🌟 ALTO POTENCIAL 3D'])
    valor_solido_3d = len(df[df['Classificacao_Potencial_3D'] == '💼 VALOR SOLIDO 3D'])

    alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
    alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])

    # Estatísticas de momentum
    momentum_positivo_home = len(df[df['M_H'] > 0.5])
    momentum_negativo_home = len(df[df['M_H'] < -0.5])
    momentum_positivo_away = len(df[df['M_A'] > 0.5])
    momentum_negativo_away = len(df[df['M_A'] < -0.5])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("🌟 Alto Potencial 3D", alto_potencial_3d)
    with col2:
        st.metric("📈 Momentum + Home", momentum_positivo_home)
        st.metric("📉 Momentum - Home", momentum_negativo_home)
    with col3:
        st.metric("📈 Momentum + Away", momentum_positivo_away)
        st.metric("📉 Momentum - Away", momentum_negativo_away)
    with col4:
        st.metric("💼 Valor Sólido 3D", valor_solido_3d)
        st.metric("🎯 Alto Valor", alto_valor_home + alto_valor_away)

    # Distribuição de recomendações 3D
    st.markdown("#### 📊 Distribuição de Recomendações 3D")
    if 'Recomendacao' in df.columns:
        dist_recomendacoes = df['Recomendacao'].value_counts()
        st.dataframe(dist_recomendacoes, use_container_width=True)

if not games_today.empty and 'Classificacao_Potencial_3D' in games_today.columns:
    resumo_3d_16_quadrantes_hoje(games_today)


########################################
#### 🧭 BLOCO DE VALIDAÇÃO DE ÂNGULOS E EIXOS
########################################
def diagnostico_vetorial(df, home_team, away_team):
    jogo = df[(df['Home'] == home_team) & (df['Away'] == away_team)]
    if jogo.empty:
        st.warning("⚠️ Jogo não encontrado.")
        return

    dx = jogo['Aggression_Home'].values[0] - jogo['Aggression_Away'].values[0]
    dy = jogo['M_H'].values[0] - jogo['M_A'].values[0]
    dz = jogo['MT_H'].values[0] - jogo['MT_A'].values[0]

    angle_XY = np.degrees(np.arctan2(dy, dx))
    angle_XZ = np.degrees(np.arctan2(dz, dx))
    angle_YZ = np.degrees(np.arctan2(dz, dy))

    st.markdown("### 🧮 Diagnóstico do Vetor 3D")
    st.write(f"**Home:** {home_team} | **Away:** {away_team}")
    st.write(f"dx (Aggression): {dx:.4f}")
    st.write(f"dy (M - Liga): {dy:.4f}")
    st.write(f"dz (MT - Time): {dz:.4f}")
    st.write(f"angle_XY (Agg × M): {angle_XY:.2f}°")
    st.write(f"angle_XZ (Agg × MT): {angle_XZ:.2f}°")
    st.write(f"angle_YZ (M × MT): {angle_YZ:.2f}°")

    # Quick check visual
    if abs(dz) < 1e-6:
        st.info("📉 MT_ ≈ 0 → vetor achatado no plano XY (sem componente vertical).")
    else:
        st.success("📈 MT_ ≠ 0 → componente vertical detectada (ângulo 3D ativo).")

# exemplo de uso:
diagnostico_vetorial(games_today, "Bahia", "Internacional RS")



st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML** implementado com sucesso!")
st.info("""
**Resumo das melhorias 3D:**
- 🔢 16 quadrantes com análise 3D completa
- 📊 Momentum integrado como terceira dimensão
- 🎯 Distâncias e ângulos 3D calculados
- 📈 Visualizações 3D interativas
- 🔍 Padrões específicos incorporando momentum
- 💡 Estratégias adaptadas para análise multidimensional
""")
