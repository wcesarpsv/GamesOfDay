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
EXCLUDED_LEAGUE_KEYWORDS = ["cup","coppa", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

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


def calculate_ah_home_target(margin, asian_line_str):
    """Calcula target AH Home diretamente da string original"""
    line_home = convert_asian_line_to_decimal(asian_line_str)
    if pd.isna(line_home) or pd.isna(margin):
        return np.nan
    return 1 if margin > line_home else 0


from sklearn.cluster import KMeans

def aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5):
    """
    Clusterização temporalmente segura:
    - Treina clusters apenas nos dados históricos
    - Aplica nos dados históricos e nos jogos de hoje
    """
    
    # 1. Preparar dados para clusterização (apenas históricos)
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    
    # Garantir que temos as colunas necessárias
    history_clean = history[required_cols].fillna(0)
    games_today_clean = games_today[required_cols].fillna(0)
    
    # 2. Calcular diferenças espaciais (apenas para clusterização)
    history_clean['dx'] = history_clean['Aggression_Home'] - history_clean['Aggression_Away']
    history_clean['dy'] = history_clean['M_H'] - history_clean['M_A']
    history_clean['dz'] = history_clean['MT_H'] - history_clean['MT_A']
    
    games_today_clean['dx'] = games_today_clean['Aggression_Home'] - games_today_clean['Aggression_Away']
    games_today_clean['dy'] = games_today_clean['M_H'] - games_today_clean['M_A']
    games_today_clean['dz'] = games_today_clean['MT_H'] - games_today_clean['MT_A']
    
    # 3. Treinar KMeans APENAS nos dados históricos
    X_train = history_clean[['dx', 'dy', 'dz']].values
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        init='k-means++',
        n_init=10
    )
    kmeans.fit(X_train)
    
    # 4. Aplicar clusters nos dados históricos
    history['Cluster3D_Label'] = kmeans.predict(history_clean[['dx', 'dy', 'dz']].values)
    
    # 5. Aplicar clusters nos jogos de hoje
    games_today['Cluster3D_Label'] = kmeans.predict(games_today_clean[['dx', 'dy', 'dz']].values)
    
    # 6. Adicionar descrição
    cluster_descriptions = {
        0: '⚡ Agressivos + Momentum Positivo',
        1: '💤 Reativos + Momentum Negativo', 
        2: '⚖️ Equilibrados',
        3: '🔥 Alta Variância',
        4: '🌪️ Caóticos / Transição'
    }
    
    history['Cluster3D_Desc'] = history['Cluster3D_Label'].map(cluster_descriptions).fillna('🌀 Outro')
    games_today['Cluster3D_Desc'] = games_today['Cluster3D_Label'].map(cluster_descriptions).fillna('🌀 Outro')
    
    return history, games_today



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
    lambda r: calculate_ah_home_target(r["Margin"], r["Asian_Line"]), axis=1
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
########################################


def adicionar_regressao_media_completa(df):
    df = df.copy()
    
    # ------------------------------------------------------------------
    # 1 Regressão à média no Momentum Time (MT)
    # ------------------------------------------------------------------
    df['MT_H_Abs'] = df['MT_H'].abs()
    df['MT_A_Abs'] = df['MT_A'].abs()
    
    df['Streak_Extremo_H'] = 0
    df['Streak_Extremo_A'] = 0
    df['Games_Above_Expected_H'] = 0
    df['Games_Above_Expected_A'] = 0

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel('K'))

    for team in all_teams:
        mask_h = df['Home'] == team
        mask_a = df['Away'] == team
        
        if mask_h.sum() > 5:
            mt = df.loc[mask_h, 'MT_H']
            extremo = mt.abs() > 1.5
            df.loc[mask_h, 'Streak_Extremo_H'] = extremo.groupby((~extremo).cumsum()).cumcount() + 1
            df.loc[mask_h, 'Games_Above_Expected_H'] = (mt > 1.8).cumsum()

        if mask_a.sum() > 5:
            mt = df.loc[mask_a, 'MT_A']
            extremo = mt.abs() > 1.5
            df.loc[mask_a, 'Streak_Extremo_A'] = extremo.groupby((~extremo).cumsum()).cumcount() + 1
            df.loc[mask_a, 'Games_Above_Expected_A'] = (mt > 1.8).cumsum()

    # Penalidade crescente quanto mais tempo o time está "quente demais"
    df['MT_Reversion_Score_H'] = np.where(
        df['MT_H'] > 1.8,
        -0.4 - 0.12 * df['Streak_Extremo_H'],
        np.where(df['MT_H'] < -1.8, 0.3 + 0.10 * df['Streak_Extremo_H'], 0)
    )
    df['MT_Reversion_Score_A'] = np.where(
        df['MT_A'] > 1.8,
        -0.4 - 0.12 * df['Streak_Extremo_A'],
        np.where(df['MT_A'] < -1.8, 0.3 + 0.10 * df['Streak_Extremo_A'], 0)
    )

    # ------------------------------------------------------------------
    # 2 HandScore extremo + MT contrário → forte sinal de regressão
    # ------------------------------------------------------------------
    df['HS_Reversion_Penalty_H'] = np.where(
        (df.get('HandScore_Home', 0) > 45) & (df['MT_H'] < -0.8), -0.7,
        np.where((df.get('HandScore_Home', 0) < -45) & (df['MT_H'] > 1.5), 0.6, 0)
    )
    df['HS_Reversion_Penalty_A'] = np.where(
        (df.get('HandScore_Away', 0) > 45) & (df['MT_A'] < -0.8), -0.7,
        np.where((df.get('HandScore_Away', 0) < -45) & (df['MT_A'] > 1.5), 0.6, 0)
    )

    # ------------------------------------------------------------------
    # 3 Bayesian shrinkage nas odds de abertura (opcional, mas recomendado)
    # ------------------------------------------------------------------
    if 'Imp_H_OP_Norm' in df.columns:
        shrinkage = 0.12
        df['Imp_H_Shrinked'] = (1 - shrinkage) * df['Imp_H_OP_Norm'] + shrinkage * (1/3)
        df['Imp_A_Shrinked'] = (1 - shrinkage) * df['Imp_A_OP_Norm'] + shrinkage * (1/3)

    return df


# ---------------- VISUALIZAÇÃO INTERATIVA 3D COM TAMANHO FIXO ----------------
import plotly.graph_objects as go

st.markdown("## 🎯 Visualização Interativa 3D – Tamanho Fixo")

# 🔽 Filtro multiselecionável de ligas
if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())
    
    selected_leagues = st.multiselect(
        "Selecione uma ou mais ligas para análise:",
        options=leagues,
        default=[],
        help="Escolha múltiplas ligas para comparar comportamentos entre campeonatos diferentes."
    )

    if selected_leagues:
        df_filtered = games_today[games_today["League"].isin(selected_leagues)].copy()
    else:
        df_filtered = games_today.copy()
else:
    st.warning("⚠️ Nenhuma coluna de 'League' encontrada — exibindo todos os jogos.")
    df_filtered = games_today.copy()

# Controle de número de confrontos
max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância 3D):", 10, min(max_n, 200), 40, step=5)




###############################

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


##########################################

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
###############
            # # CONFIGURAÇÃO DA CÂMERA (AGORA INTERATIVA)
            # camera=dict(
            #     eye=dict(x=cam_x, y=cam_y, z=cam_z),
            #     up=dict(x=up_x, y=up_y, z=up_z),
            #     center=dict(x=0, y=0, z=0)
            # )
            
###############
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


# Ajusta para múltiplas ligas (string amigável)
selected_league_label = ", ".join(selected_leagues) if selected_leagues else "⚽ Todas as ligas"
fig_3d_fixed = create_fixed_3d_plot(df_plot, n_to_show, selected_league_label)

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
# history = aplicar_clusterizacao_3d(history, n_clusters=5)
# games_today = aplicar_clusterizacao_3d(games_today, n_clusters=5)
# ✅ NOVO (seguro)
history, games_today = aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5)

# === ONDE VOCÊ VAI CHAMAR ===
# Depois de calcular_momentum_time e antes do treino do modelo:

history = adicionar_regressao_media_completa(history)
games_today = adicionar_regressao_media_completa(games_today)


def treinar_modelo_3d_clusters_single(history, games_today):
    """
    Treina o modelo 3D (Home) com possibilidade de incluir odds de abertura implícitas normalizadas
    e gera análise de viés de mercado (Market Bias Opening) com segurança de dados.
    """

    st.markdown("### ⚙️ Configuração do Treino 3D com Odds de Abertura")

    # Toggle no Streamlit
    use_opening_odds = st.checkbox("📊 Incluir Odds de Abertura no Treino", value=True)

    # ----------------------------
    # 🧩 Garantir features 3D e clusters
    # ----------------------------
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history, games_today = aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5)

    # ----------------------------
    # 🧠 Feature Engineering - COM SEGURANÇA
    # ----------------------------
    # Garantir que as colunas de liga existem
    if 'League' not in history.columns:
        history['League'] = 'Unknown'
    if 'League' not in games_today.columns:
        games_today['League'] = 'Unknown'
    
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    
    # Garantir que clusters existem
    if 'Cluster3D_Label' not in history.columns:
        history['Cluster3D_Label'] = 0
    if 'Cluster3D_Label' not in games_today.columns:
        games_today['Cluster3D_Label'] = 0
        
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    # Lista de features 3D base (garantir que existem)
    features_3d_base = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D',
        # NOVAS FEATURES DE REGRESSÃO À MÉDIA
        'MT_Reversion_Score_H', 'MT_Reversion_Score_A',
        'HS_Reversion_Penalty_H', 'HS_Reversion_Penalty_A',
        'Streak_Extremo_H', 'Streak_Extremo_A',
        'Games_Above_Expected_H', 'Games_Above_Expected_A'
    ]

    # 🆕 CRIAR COLUNAS FALTANTES DE FORMA SEGURA
    for feature in features_3d_base:
        if feature not in history.columns:
            history[feature] = 0.0
        if feature not in games_today.columns:
            games_today[feature] = 0.0

    # 🆕 CRIAR COLUNAS BAYESIANAS FALTANTES
    if 'Quadrante_Bayes_Score_H' not in history.columns:
        history['Quadrante_Bayes_Score_H'] = 0.5  # Valor neutro
    if 'Quadrante_Bayes_Score_H' not in games_today.columns:
        games_today['Quadrante_Bayes_Score_H'] = 0.5

    # Selecionar apenas as features que realmente existem
    features_3d_existentes = [f for f in features_3d_base + ['Quadrante_Bayes_Score_H'] if f in history.columns]
    extras_3d = history[features_3d_existentes].fillna(0)

    # ----------------------------
    # 🎯 Features de Odds Implícitas Normalizadas
    # ----------------------------
    odds_features = pd.DataFrame()
    if use_opening_odds:
        # 🆕 GARANTIR QUE COLUNAS DE ODDS EXISTEM
        for col in ['Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP','Odd_H','Odd_D','Odd_A']:
            if col not in history.columns:
                history[col] = 3.0  # Valor padrão neutro
            if col not in games_today.columns:
                games_today[col] = 3.0

        # Calcular probabilidades implícitas
        history['Imp_H_OP'] = 1 / history['Odd_H_OP']
        history['Imp_D_OP'] = 1 / history['Odd_D_OP']
        history['Imp_A_OP'] = 1 / history['Odd_A_OP']
        history[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']] = history[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].replace([np.inf, -np.inf], np.nan)

        sum_probs = history[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].sum(axis=1).replace(0, np.nan)
        history['Imp_H_OP_Norm'] = history['Imp_H_OP'] / sum_probs
        history['Imp_D_OP_Norm'] = history['Imp_D_OP'] / sum_probs
        history['Imp_A_OP_Norm'] = history['Imp_A_OP'] / sum_probs
        history['Diff_Odd_H'] = history['Odd_H_OP'] - history['Odd_H']
        history['Diff_Odd_D'] = history['Odd_D_OP'] - history['Odd_D']
        history['Diff_Odd_A'] = history['Odd_A_OP'] - history['Odd_A']

        # 🆕 CRIAR COLUNAS SHRINKED DE FORMA SEGURA
        history['Imp_H_Shrinked'] = history['Imp_H_OP_Norm']  # Fallback sem shrinkage
        history['Imp_A_Shrinked'] = history['Imp_A_OP_Norm']
        
        # Aplicar shrinkage se possível
        shrinkage = 0.12
        if 'Imp_H_OP_Norm' in history.columns:
            history['Imp_H_Shrinked'] = (1 - shrinkage) * history['Imp_H_OP_Norm'] + shrinkage * (1/3)
            history['Imp_A_Shrinked'] = (1 - shrinkage) * history['Imp_A_OP_Norm'] + shrinkage * (1/3)

        # Coletar features de odds
        odds_cols = ['Imp_H_OP_Norm', 'Imp_D_OP_Norm', 'Imp_A_OP_Norm','Diff_Odd_H','Diff_Odd_D','Diff_Odd_A','Imp_H_Shrinked','Imp_A_Shrinked']
        odds_cols_existentes = [col for col in odds_cols if col in history.columns]
        odds_features = history[odds_cols_existentes].fillna(0)

    # ----------------------------
    # 🧩 Montagem final do dataset
    # ----------------------------
    if use_opening_odds and not odds_features.empty:
        X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, odds_features], axis=1)
    else:
        X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)

    # Garantir que o target existe
    if 'Target_AH_Home' not in history.columns:
        st.error("❌ Target_AH_Home não encontrado no histórico. Verifique os dados.")
        return None, games_today
        
    y_home = history['Target_AH_Home'].astype(int)

    # ----------------------------
    # 🏗️ Modelo
    # ----------------------------
    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    # ----------------------------
    # 🔮 Previsões no dataset do dia
    # ----------------------------
    # Preparar dados do dia com as mesmas colunas
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d_existentes].fillna(0)

    if use_opening_odds:
        # Preparar odds do dia
        for col in ['Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP','Odd_H','Odd_D','Odd_A']:
            if col not in games_today.columns:
                games_today[col] = 3.0

        games_today['Imp_H_OP'] = 1 / games_today['Odd_H_OP']
        games_today['Imp_D_OP'] = 1 / games_today['Odd_D_OP']
        games_today['Imp_A_OP'] = 1 / games_today['Odd_A_OP']
        games_today[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']] = games_today[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].replace([np.inf, -np.inf], np.nan)

        sum_today = games_today[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].sum(axis=1).replace(0, np.nan)
        games_today['Imp_H_OP_Norm'] = games_today['Imp_H_OP'] / sum_today
        games_today['Imp_D_OP_Norm'] = games_today['Imp_D_OP'] / sum_today
        games_today['Imp_A_OP_Norm'] = games_today['Imp_A_OP'] / sum_today
        games_today['Diff_Odd_H'] = games_today['Odd_H_OP'] - games_today['Odd_H']
        games_today['Diff_Odd_D'] = games_today['Odd_D_OP'] - games_today['Odd_D']
        games_today['Diff_Odd_A'] = games_today['Odd_A_OP'] - games_today['Odd_A']

        # Shrinkage para dados do dia
        games_today['Imp_H_Shrinked'] = games_today['Imp_H_OP_Norm']
        games_today['Imp_A_Shrinked'] = games_today['Imp_A_OP_Norm']
        if 'Imp_H_OP_Norm' in games_today.columns:
            games_today['Imp_H_Shrinked'] = (1 - shrinkage) * games_today['Imp_H_OP_Norm'] + shrinkage * (1/3)
            games_today['Imp_A_Shrinked'] = (1 - shrinkage) * games_today['Imp_A_OP_Norm'] + shrinkage * (1/3)

        odds_today = games_today[odds_cols_existentes].fillna(0)
        X_today = pd.concat([ligas_today, clusters_today, extras_today, odds_today], axis=1)
    else:
        X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    # Garantir que X_today tem as mesmas colunas que X
    missing_cols = set(X.columns) - set(X_today.columns)
    for col in missing_cols:
        X_today[col] = 0
    
    X_today = X_today[X.columns]  # Reordenar colunas

    # ----------------------------
    # 📈 Previsões
    # ----------------------------
    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today['ML_Confidence']

    # ----------------------------
    # 📊 Avaliação rápida (cross-check)
    # ----------------------------
    accuracy = model_home.score(X, y_home)
    st.metric("Accuracy (Treino)", f"{accuracy:.2%}")
    st.write("📘 Features usadas:", len(X.columns))

    # ----------------------------
    # 🔍 Importância de Features
    # ----------------------------
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_feats = importances.head(25).to_frame("Importância")

    st.markdown("### 🔍 Top Features (Modelo Único – Home)")
    st.dataframe(top_feats, use_container_width=True)

    if use_opening_odds:
        odds_influentes = [f for f in top_feats.index if "Imp_" in f]
        if odds_influentes:
            st.success(f"💡 Variáveis de abertura influentes: {', '.join(odds_influentes)}")
        else:
            st.info("📊 As odds de abertura ainda não mostraram forte impacto.")

    # ============================================================
    # 🧩 Segurança final
    # ============================================================
    if "Quadrante_ML_Score_Home" not in games_today.columns:
        games_today["Quadrante_ML_Score_Home"] = np.nan
        games_today["Quadrante_ML_Score_Away"] = np.nan
        games_today["Quadrante_ML_Score_Main"] = np.nan
        games_today["ML_Side"] = "N/A"
        games_today["ML_Confidence"] = 0.0

    for col in ["League", "Home", "Away"]:
        if col not in games_today.columns:
            games_today[col] = "N/A"

    if games_today.empty:
        st.warning("⚠️ Nenhum jogo válido encontrado após o treino. Verifique o CSV e as odds.")
    else:
        st.success(f"✅ {len(games_today)} jogos processados e prontos para análise 3D.")

    st.success("✅ Modelo 3D treinado (HOME) – com análise de viés integrada.")
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
            # ADICIONAR ESTA CONDIÇÃO ANTES DO "else":
        elif score_home >= 0.75 and momentum_h >= -0.5:  # Score alto, momentum não muito negativo
            return f'🎯 VALUE HOME (Score Alto) ({score_home:.1%})'
        elif score_away >= 0.75 and momentum_a >= -0.5:
            return f'🎯 VALUE AWAY (Score Alto) ({score_away:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_3d_16_dual, axis=1)

    # 4. RANKING POR MELHOR PROBABILIDADE 3D
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- EXECUÇÃO PRINCIPAL 3D ----------------
# Executar treinamento 3D
if not history.empty:
    modelo_home, games_today = treinar_modelo_3d_clusters_single(history, games_today)
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

    


    ########### ---------------- LIVE SCORE MONITOR – SISTEMA 3D (1X2) ----------------

    def determine_match_result_1x2(row):
        """Determina o resultado 1X2 puro (sem handicap)."""
        try:
            gh = float(row['Goals_H_Today'])
            ga = float(row['Goals_A_Today'])
        except (ValueError, TypeError):
            return None
    
        if pd.isna(gh) or pd.isna(ga):
            return None
    
        if gh > ga:
            return "HOME_WIN"
        elif gh < ga:
            return "AWAY_WIN"
        else:
            return "DRAW"
    
    
    def check_recommendation_correct_1x2(recomendacao, match_result):
        """Verifica se a recomendação acertou o resultado 1X2."""
        if pd.isna(recomendacao) or match_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
            return None
    
        recomendacao_str = str(recomendacao).upper()
    
        is_home_bet = any(k in recomendacao_str for k in [
            'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
            'MODELO CONFIA HOME', 'H:', 'HOME)'
        ])
        is_away_bet = any(k in recomendacao_str for k in [
            'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
            'MODELO CONFIA AWAY', 'A:', 'AWAY)'
        ])
    
        if is_home_bet and match_result == "HOME_WIN":
            return True
        elif is_away_bet and match_result == "AWAY_WIN":
            return True
        else:
            return False
    
    
    def calculate_profit_1x2(recomendacao, match_result, odds_row):
        """Calcula o profit líquido (odds brutas - 1) para apostas 1X2."""
        if pd.isna(recomendacao) or match_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
            return 0
    
        recomendacao_str = str(recomendacao).upper()
    
        is_home_bet = any(k in recomendacao_str for k in [
            'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
            'MODELO CONFIA HOME', 'H:', 'HOME)'
        ])
        is_away_bet = any(k in recomendacao_str for k in [
            'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
            'MODELO CONFIA AWAY', 'A:', 'AWAY)'
        ])
    
        if is_home_bet:
            odd = odds_row.get('Odd_H', np.nan)
            won = match_result == "HOME_WIN"
        elif is_away_bet:
            odd = odds_row.get('Odd_A', np.nan)
            won = match_result == "AWAY_WIN"
        else:
            return 0
    
        if pd.isna(odd):
            return 0
    
        return (odd - 1) if won else -1
    
    
    def update_real_time_data_1x2(df):
        """
        Atualiza as métricas 1X2 no DataFrame.
        Retorna: Result_1x2, Quadrante_Correct_1x2, Profit_1x2
        """
        df = df.copy()
    
        df['Result_1x2'] = df.apply(determine_match_result_1x2, axis=1)
        df['Quadrante_Correct_1x2'] = df.apply(
            lambda r: check_recommendation_correct_1x2(
                r['Recomendacao'], r['Result_1x2']
            ), axis=1
        )
        df['Profit_1x2'] = df.apply(
            lambda r: calculate_profit_1x2(
                r['Recomendacao'], r['Result_1x2'], r
            ), axis=1
        )
    
        return df


    ##################################################################

    # ########### NOVA SEÇÃO - INDICADORES DE FORMA E VALOR ###########
    # # =========================================================
    # # 🧠 4 NOVAS INDICAÇÕES DE FORMA (HOME & AWAY)
    # # =========================================================
    # import numpy as np
    # import pandas as pd
    
    # st.subheader("🧩 Indicadores de Forma e Valor (MT_ + M_)")
    
    # def definir_indicacao_forma(df, prefix):
    #     """
    #     Define a indicação estratégica baseada no momentum da liga (M_)
    #     e no momentum próprio do time (MT_).
    #     """
    #     M_col = f"M_{prefix}"
    #     MT_col = f"MT_{prefix}"
    
    #     condicoes = [
    #         (df[MT_col] >= 0.5) & (df[M_col].between(-0.5, 0.5)),   # 🟩 Sustainable Form
    #         (df[MT_col].between(0, 0.5)) & (df[M_col] <= -0.5),     # 🟨 Undervalued Recovery
    #         (df[MT_col] >= 0.5) & (df[M_col] >= 0.5),               # 🟥 Overhyped Risk
    #         (df[MT_col] <= -0.5) & (df[M_col] <= -0.5),             # 🟦 Hidden Bounce
    #     ]
    #     resultados = [
    #         "🟩 Sustainable Form",
    #         "🟨 Undervalued Recovery",
    #         "🟥 Overhyped Risk",
    #         "🟦 Hidden Bounce",
    #     ]
    #     return np.select(condicoes, resultados, default="⚪ Neutra")
    
    # # Aplicar regras no DataFrame principal
    # ranking_3d["Indicacao_Forma_Home"] = definir_indicacao_forma(ranking_3d, "H")
    # ranking_3d["Indicacao_Forma_Away"] = definir_indicacao_forma(ranking_3d, "A")
    
    # # =========================================================
    # # 🏷️ CRIAR SELO FINAL COMBINANDO CLASSIFICAÇÃO + FORMA
    # # =========================================================
    # def criar_selo(row, side="Home"):
    #     val_col = f"Classificacao_Valor_{side}"
    #     forma_col = f"Indicacao_Forma_{side}"
    #     val = str(row.get(val_col, "") or "")
    #     forma = str(row.get(forma_col, "") or "")
    #     if val and forma and val != "nan" and forma != "nan":
    #         return f"{val} | {forma}"
    #     elif forma:
    #         return forma
    #     else:
    #         return val
    
    # # Garantir que colunas de valor existam
    # for c in ["Classificacao_Valor_Home", "Classificacao_Valor_Away", "Profit_Quadrante"]:
    #     if c not in ranking_3d.columns:
    #         ranking_3d[c] = np.nan
    
    # ranking_3d["Selo_Estrategia_Home"] = ranking_3d.apply(lambda r: criar_selo(r, "Home"), axis=1)
    # ranking_3d["Selo_Estrategia_Away"] = ranking_3d.apply(lambda r: criar_selo(r, "Away"), axis=1)
    
    # # =========================================================
    # # 🎨 VISUALIZAÇÃO NO STREAMLIT (TEMA ESCURO SEGURO)
    # # =========================================================
    # st.markdown("### 🎯 Recomendação de Estratégia por Time")
    
    # cols_show = [c for c in [
    #     "League", "Home", "Away",'Goals_H_Today','Goals_A_Today',
    #     "M_H", "MT_H", "Indicacao_Forma_Home",
    #     "Classificacao_Valor_Home", "Selo_Estrategia_Home",
    #     "M_A", "MT_A", "Indicacao_Forma_Away",
    #     "Classificacao_Valor_Away", "Selo_Estrategia_Away",
    #     "Profit_Quadrante"
    # ] if c in ranking_3d.columns]
    
    # st.dataframe(
    #     ranking_3d[cols_show]
    #     .style.format({
    #         "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}",
    #         "M_H": "{:.2f}", "MT_H": "{:.2f}",
    #         "M_A": "{:.2f}", "MT_A": "{:.2f}",
    #         "Profit_Quadrante": "{:.2f}"
    #     })
    #     # Cores otimizadas para fundo escuro
    #     .applymap(lambda v: "background-color: #006400; color: white" if "Sustainable" in str(v) else None, subset=["Indicacao_Forma_Home", "Indicacao_Forma_Away"])
    #     .applymap(lambda v: "background-color: #9ACD32; color: black" if "Undervalued" in str(v) else None, subset=["Indicacao_Forma_Home", "Indicacao_Forma_Away"])
    #     .applymap(lambda v: "background-color: #B22222; color: white" if "Overhyped" in str(v) else None, subset=["Indicacao_Forma_Home", "Indicacao_Forma_Away"])
    #     .applymap(lambda v: "background-color: #1E90FF; color: white" if "Hidden" in str(v) else None, subset=["Indicacao_Forma_Home", "Indicacao_Forma_Away"]),
    #     use_container_width=True
    # )
    
    # # =========================================================
    # # 💾 EXPORTAR RESULTADO COM NOVAS INDICAÇÕES
    # # =========================================================
    # csv_path = os.path.join(BASE_DIR, "GamesDay", f"Estrategia_Forma_{datetime.now().strftime('%Y-%m-%d')}.csv")
    # ranking_3d.to_csv(csv_path, index=False)
    
    # st.success(f"✅ Estratégias salvas com sucesso em: {csv_path}")
    # st.download_button(
    #     "📥 Baixar CSV com Estratégias",
    #     data=open(csv_path, "rb").read(),
    #     file_name=os.path.basename(csv_path),
    #     mime="text/csv"
    # )

    
    # ============================================================
    ##### NOVA LOGICA CORRETA PARA HANDICAP - SUBSTITUINDO A ANTERIOR
    
    def handicap_favorito_v9(margin, line):
        """
        Calcula handicap para FAVORITOS (linhas negativas)
        margin: gols_home - gols_away
        line: linha negativa (ex: -0.25, -1.25, etc)
        """
        line_abs = abs(line)
        
        # Linhas inteiras (-1, -2, etc)
        if line_abs.is_integer():
            if margin > line_abs:
                return 1      # Win
            elif margin == line_abs:
                return 0      # Push
            else:
                return -1     # Lose
        
        # Linha -0.25
        elif line == -0.25:
            if margin > 0:
                return 1      # Win
            elif margin == 0:
                return -0.5   # Half lose
            else:
                return -1     # Lose
        
        # Linha -0.50
        elif line == -0.50:
            if margin > 0:
                return 1      # Win
            else:
                return -1     # Lose
        
        # Linha -0.75
        elif line == -0.75:
            if margin >= 2:
                return 1      # Win by 2+
            elif margin == 1:
                return 0.5    # Half win
            else:
                return -1     # Lose
        
        # Linha -1.25
        elif line == -1.25:
            if margin >= 2:
                return 1      # Win by 2+
            elif margin == 1:
                return -0.5   # Half lose
            else:
                return -1     # Lose
        
        # Linha -1.50
        elif line == -1.50:
            if margin >= 2:
                return 1      # Win by 2+
            else:
                return -1     # Lose
        
        # Linha -1.75
        elif line == -1.75:
            if margin >= 3:
                return 1      # Win by 3+
            elif margin == 2:
                return 0.5    # Half win
            else:
                return -1     # Lose
        
        # Linha -2.00
        elif line == -2.00:
            if margin > 2:
                return 1      # Win by 3+
            elif margin == 2:
                return 0      # Push
            else:
                return -1     # Lose
        
        return np.nan
    
    def handicap_underdog_v9(margin, line):
        """
        Calcula handicap para UNDERDOGS (linhas positivas)
        margin: gols_home - gols_away  
        line: linha positiva (ex: +0.25, +1.25, etc)
        """
        # Linhas inteiras (0, +1, +2, etc)
        if line.is_integer():
            if margin >= -line:
                return 1      # Win ou empate
            elif margin == -(line + 1):
                return 0      # Push (perde por exatamente line+1)
            else:
                return -1     # Lose
        
        # Linha +0.25
        elif line == 0.25:
            if margin > 0:
                return 1      # Win
            elif margin == 0:
                return 0.5    # Half win
            else:
                return -1     # Lose
        
        # Linha +0.50
        elif line == 0.50:
            if margin >= 0:
                return 1      # Win ou Draw
            else:
                return -1     # Lose
        
        # Linha +0.75
        elif line == 0.75:
            if margin >= 0:
                return 1      # Win ou Draw
            elif margin == -1:
                return -0.5   # Half lose (lose by 1)
            else:
                return -1     # Lose
        
        # Linha +1.00
        elif line == 1.00:
            if margin >= -1:
                return 1      # Win, Draw ou Lose by 1
            else:
                return -1     # Lose by 2+
        
        # Linha +1.25
        elif line == 1.25:
            if margin >= -1:
                return 1      # Win, Draw ou Lose by 1
            elif margin == -2:
                return 0.5    # Half win (lose by 2)
            else:
                return -1     # Lose by 3+
        
        # Linha +1.50
        elif line == 1.50:
            if margin >= -1:
                return 1      # Win, Draw ou Lose by 1
            else:
                return -1     # Lose by 2+
        
        # Linha +1.75
        elif line == 1.75:
            if margin >= -1:
                return 1      # Win, Draw ou Lose by 1
            elif margin == -2:
                return -0.5   # Half lose (lose by 2)
            else:
                return -1     # Lose by 3+
        
        # Linha +2.00
        elif line == 2.00:
            if margin >= -2:
                return 1      # Win, Draw ou Lose by 1-2
            elif margin == -3:
                return 0      # Push (lose by 3)
            else:
                return -1     # Lose by 4+
        
        return np.nan
    
    def handicap_home_v9(row):
        """Calcula handicap para apostas no HOME"""
        margin = row['Goals_H_Today'] - row['Goals_A_Today']
        line = row['Asian_Line_Decimal']
        
        if line < 0:  # Home é favorito
            return handicap_favorito_v9(margin, line)
        else:  # Home é underdog
            return handicap_underdog_v9(margin, line)
    
    def handicap_away_v9(row):
        """Calcula handicap para apostas no AWAY"""
        margin = row['Goals_A_Today'] - row['Goals_H_Today']  
        line = -row['Asian_Line_Decimal']  # Inverte a linha
        
        if line < 0:  # Away é favorito
            return handicap_favorito_v9(margin, line)
        else:  # Away é underdog
            return handicap_underdog_v9(margin, line)
    
    # ---------------- SUBSTITUINDO AS FUNÇÕES ORIGINAIS DO LIVE SCORE MONITOR ----------------
    
    def determine_handicap_result_3d(row):
        """
        SUBSTITUÍDA: Nova função para determinar resultado do handicap usando lógica v9
        """
        try:
            gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
            ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
            asian_line = float(row['Asian_Line_Decimal'])
            recomendacao = str(row.get('Recomendacao', '')).upper()
        except (ValueError, TypeError):
            return None
    
        if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line):
            return None
    
        # Detectar lado da aposta
        is_home_bet = any(k in recomendacao for k in [
            'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
            'MODELO CONFIA HOME', 'H:', 'HOME)'
        ])
        is_away_bet = any(k in recomendacao for k in [
            'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
            'MODELO CONFIA AWAY', 'A:', 'AWAY)'
        ])
    
        if not is_home_bet and not is_away_bet:
            return None
    
        # Usar nova lógica v9
        if is_home_bet:
            outcome = handicap_home_v9(row)
        else:
            outcome = handicap_away_v9(row)
    
        # Mapear outcome para resultado descritivo
        if outcome == 1:
            return "FULL_WIN"
        elif outcome == 0.5:
            return "HALF_WIN" 
        elif outcome == 0:
            return "PUSH"
        elif outcome == -0.5:
            return "HALF_LOSS"
        elif outcome == -1:
            return "LOSS"
        else:
            return None
    
    def check_handicap_recommendation_correct_3d(recomendacao, handicap_result):
        """
        SUBSTITUÍDA: Nova função para verificar se recomendação estava correta
        """
        if pd.isna(recomendacao) or handicap_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
            return None
    
        # Considerar correto se for FULL_WIN ou HALF_WIN
        return handicap_result in ["FULL_WIN", "HALF_WIN"]
    
    def calculate_handicap_profit_3d(recomendacao, handicap_result, odds_row):
        """
        SUBSTITUÍDA: Nova função para calcular profit usando lógica v9
        """
        if pd.isna(recomendacao) or handicap_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
            return 0
    
        recomendacao_str = str(recomendacao).upper()
    
        is_home_bet = any(k in recomendacao_str for k in [
            'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
            'MODELO CONFIA HOME', 'H:', 'HOME)'
        ])
        is_away_bet = any(k in recomendacao_str for k in [
            'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 
            'MODELO CONFIA AWAY', 'A:', 'AWAY)'
        ])
    
        if is_home_bet:
            odd = odds_row.get('Odd_H_Asi', np.nan)
        elif is_away_bet:
            odd = odds_row.get('Odd_A_Asi', np.nan)
        else:
            return 0
    
        if pd.isna(odd):
            return 0
    
        # Calcular profit baseado no resultado
        if handicap_result == "FULL_WIN":
            return odd  # Odd líquida
        elif handicap_result == "HALF_WIN":
            return odd / 2 # Odd líquida
        elif handicap_result == "PUSH":
            return 0
        elif handicap_result == "HALF_LOSS":
            return -0.5
        elif handicap_result == "LOSS":
            return -1
        else:
            return 0
    
    def update_real_time_data_3d(df):
        """
        SUBSTITUÍDA: Nova função principal de atualização usando lógica v9
        """
        df = df.copy()
    
        # Determinar resultado do handicap
        df['Handicap_Result'] = df.apply(determine_handicap_result_3d, axis=1)
        
        # Calcular se a recomendação estava correta
        df['Quadrante_Correct'] = df.apply(
            lambda r: check_handicap_recommendation_correct_3d(
                r['Recomendacao'], r['Handicap_Result']
            ), axis=1
        )
        
        # Calcular profit
        df['Profit_Quadrante'] = df.apply(
            lambda r: calculate_handicap_profit_3d(
                r['Recomendacao'], r['Handicap_Result'], r
            ), axis=1
        )
    
        return df
    
    def generate_live_summary_3d(df):
        """
        SUBSTITUÍDA: Nova função de resumo usando lógica v9
        """
        finished_games = df[df['Handicap_Result'].notna()]
        
        if finished_games.empty:
            return {
                "Total Jogos": len(df),
                "Jogos Finalizados": 0,
                "Apostas Quadrante 3D": 0,
                "Acertos Quadrante 3D": 0,
                "Winrate Quadrante 3D": "0%",
                "Profit Quadrante 3D": "0.00u",
                "ROI Quadrante 3D": "0%",
                "Full Wins": 0,
                "Half Wins": 0, 
                "Pushes": 0,
                "Half Losses": 0,
                "Losses": 0
            }
        
        quadrante_bets = finished_games[finished_games['Quadrante_Correct'].notna()]
        total_bets = len(quadrante_bets)
        correct_bets = quadrante_bets['Quadrante_Correct'].sum()
        winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
        total_profit = quadrante_bets['Profit_Quadrante'].sum()
        roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
        
        # Estatísticas detalhadas dos outcomes
        full_wins = (finished_games['Handicap_Result'] == 'FULL_WIN').sum()
        half_wins = (finished_games['Handicap_Result'] == 'HALF_WIN').sum()
        pushes = (finished_games['Handicap_Result'] == 'PUSH').sum()
        half_losses = (finished_games['Handicap_Result'] == 'HALF_LOSS').sum()
        losses = (finished_games['Handicap_Result'] == 'LOSS').sum()
        
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": len(finished_games),
            "Apostas Quadrante 3D": total_bets,
            "Acertos Quadrante 3D": int(correct_bets),
            "Winrate Quadrante 3D": f"{winrate:.1f}%",
            "Profit Quadrante 3D": f"{total_profit:.2f}u",
            "ROI Quadrante 3D": f"{roi:.1f}%",
            "Full Wins": int(full_wins),
            "Half Wins": int(half_wins),
            "Pushes": int(pushes),
            "Half Losses": int(half_losses),
            "Losses": int(losses)
        }
    
    # ---------------- MANTENDO AS FUNÇÕES 1X2 ORIGINAIS (NÃO ALTERADAS) ----------------
    
    def determine_match_result_1x2(row):
        """MANTIDA: Determina o resultado 1X2 puro (sem handicap)"""
        try:
            gh = float(row['Goals_H_Today'])
            ga = float(row['Goals_A_Today'])
        except (ValueError, TypeError):
            return None
    
        if pd.isna(gh) or pd.isna(ga):
            return None
    
        if gh > ga:
            return "HOME_WIN"
        elif gh < ga:
            return "AWAY_WIN"
        else:
            return "DRAW"
    
    def check_recommendation_correct_1x2(recomendacao, match_result):
        """MANTIDA: Verifica se a recomendação acertou o resultado 1X2"""
        if pd.isna(recomendacao) or match_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
            return None
    
        recomendacao_str = str(recomendacao).upper()
    
        is_home_bet = any(k in recomendacao_str for k in [
            'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
            'MODELO CONFIA HOME', 'H:', 'HOME)'
        ])
        is_away_bet = any(k in recomendacao_str for k in [
            'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
            'MODELO CONFIA AWAY', 'A:', 'AWAY)'
        ])
    
        if is_home_bet and match_result == "HOME_WIN":
            return True
        elif is_away_bet and match_result == "AWAY_WIN":
            return True
        else:
            return False
    
    def calculate_profit_1x2(recomendacao, match_result, odds_row):
        """MANTIDA: Calcula o profit líquido (odds brutas - 1) para apostas 1X2"""
        if pd.isna(recomendacao) or match_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
            return 0
    
        recomendacao_str = str(recomendacao).upper()
    
        is_home_bet = any(k in recomendacao_str for k in [
            'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
            'MODELO CONFIA HOME', 'H:', 'HOME)'
        ])
        is_away_bet = any(k in recomendacao_str for k in [
            'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
            'MODELO CONFIA AWAY', 'A:', 'AWAY)'
        ])
    
        if is_home_bet:
            odd = odds_row.get('Odd_H', np.nan)
            won = match_result == "HOME_WIN"
        elif is_away_bet:
            odd = odds_row.get('Odd_A', np.nan)
            won = match_result == "AWAY_WIN"
        else:
            return 0
    
        if pd.isna(odd):
            return 0
    
        return (odd - 1) if won else -1
    
    def update_real_time_data_1x2(df):
        """MANTIDA: Atualiza as métricas 1X2 no DataFrame"""
        df = df.copy()
    
        df['Result_1x2'] = df.apply(determine_match_result_1x2, axis=1)
        df['Quadrante_Correct_1x2'] = df.apply(
            lambda r: check_recommendation_correct_1x2(
                r['Recomendacao'], r['Result_1x2']
            ), axis=1
        )
        df['Profit_1x2'] = df.apply(
            lambda r: calculate_profit_1x2(
                r['Recomendacao'], r['Result_1x2'], r
            ), axis=1
        )
    
        return df
    
    # ---------------- FUNÇÃO DE COMPARAÇÃO SISTEMAS (MANTIDA) ----------------
    
    def compare_systems_summary(df):
        """MANTIDA: Compara Sistema 3D Handicap x Sistema 1x2"""
        def calc(prefix):
            if prefix == "Quadrante":  # Handicap Asiático
                correct_col = "Quadrante_Correct"
                profit_col = "Profit_Quadrante"
            else:  # 1x2
                correct_col = f"Quadrante_Correct_{prefix}"
                profit_col = f"Profit_{prefix}"
    
            valid = df[correct_col].notna().sum()
            total_profit = df[profit_col].sum()
            roi = total_profit / valid if valid > 0 else 0
            acc = df[correct_col].mean(skipna=True)
            return {"Bets": valid, "Hit%": f"{acc:.1%}", "Profit": f"{total_profit:.2f}", "ROI": f"{roi:.1%}"}
    
        ah = calc("Quadrante")
        x12 = calc("1x2")
    
        resumo = pd.DataFrame({
            "Métrica": ["Apostas", "Taxa de Acerto", "Lucro Total", "ROI Médio"],
            "Sistema Asiático (AH)": [ah["Bets"], ah["Hit%"], ah["Profit"], ah["ROI"]],
            "Sistema 1x2": [x12["Bets"], x12["Hit%"], x12["Profit"], x12["ROI"]]
        })
    
        st.markdown("### ⚖️ Comparativo de Performance – AH vs 1x2")
        
        # Apenas exibir sem highlight para evitar o erro
        st.dataframe(resumo, use_container_width=True)
        
        # Adicionar análise textual simples
        ah_profit = float(ah["Profit"].replace('u', ''))
        x12_profit = float(x12["Profit"].replace('u', ''))
        
        if ah_profit > x12_profit:
            st.success(f"✅ **Sistema Asiático performou melhor**: {ah['Profit']} vs {x12['Profit']}")
        elif x12_profit > ah_profit:
            st.success(f"✅ **Sistema 1x2 performou melhor**: {x12['Profit']} vs {ah['Profit']}")
        else:
            st.info("⚖️ **Sistemas com performance similar**")
    

    # ================================================================
    # 📡 CHAMAR O LIVE SCORE MONITOR - SISTEMA ATUALIZADO V9
    # ================================================================
    
    # Aplicar atualização em tempo real 3D (Handicap Asiático)
    ranking_3d = update_real_time_data_3d(ranking_3d)
    
    # Aplicar atualização em tempo real 1x2
    ranking_3d = update_real_time_data_1x2(ranking_3d)
    
    # Exibir resumo live 3D
    st.markdown("## 📡 Live Score Monitor - Sistema 3D (Handicap Asiático)")
    live_summary_3d = generate_live_summary_3d(ranking_3d)
    st.json(live_summary_3d)
    
    # Exibir resumo 1x2
    st.markdown("## 📡 Live Score Monitor - Sistema 3D (1x2)")
    finished_1x2 = ranking_3d[ranking_3d['Result_1x2'].notna()]
    if not finished_1x2.empty:
        total_bets = finished_1x2['Quadrante_Correct_1x2'].notna().sum()
        correct_bets = finished_1x2['Quadrante_Correct_1x2'].sum()
        total_profit = finished_1x2['Profit_1x2'].sum()
        winrate = correct_bets / total_bets if total_bets > 0 else 0
        roi = total_profit / total_bets if total_bets > 0 else 0
    
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Apostas (1x2)", total_bets)
        with col2:
            st.metric("Winrate (1x2)", f"{winrate:.1%}")
        with col3:
            st.metric("Lucro Total (1x2)", f"{total_profit:.2f}u")
        with col4:
            st.metric("ROI (1x2)", f"{roi:.1%}")
    else:
        st.info("⚠️ Nenhum jogo finalizado ainda para o sistema 1x2.")
    
    # Exibir comparativo de sistemas
    compare_systems_summary(ranking_3d)


    #####################################################################

    # Ordenar por score final 3D
    ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

    # Colunas para exibição 3D
    colunas_3d = [
        
        'League', 'Time',
        'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'Recomendacao', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Score_Final_3D', 'Classificacao_Potencial_3D',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
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




# =====================================================
# 🧠 ML2 – MOVIMENTO DE MERCADO (Market Bias Model)
# =====================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def treinar_ml_movimento_mercado(history, games_today):
    """
    Treina modelo secundário para prever se o movimento de odds (open→close)
    indica corretamente o lado vencedor e detecta divergências modelo x mercado.
    """

    df = history.copy()

    # -------------------------------------------------
    # ⚙️ 1️⃣ Cria Outcome e linha de abertura (Home)
    # -------------------------------------------------
    if "Outcome_Home_FT" not in df.columns:
        if {"Goals_H_FT", "Goals_A_FT"}.issubset(df.columns):
            df["Outcome_Home_FT"] = np.sign(df["Goals_H_FT"] - df["Goals_A_FT"]).astype("Int64")
        else:
            st.warning("⚠️ ML2: sem FT no histórico — modelo ignorado.")
            return None, games_today

    for df_tmp in (df, games_today):
        if "Asian_Line_OP" in df_tmp.columns and "Asian_Line_OP_Decimal" not in df_tmp.columns:
            df_tmp["Asian_Line_OP_Decimal"] = df_tmp["Asian_Line_OP"].apply(convert_asian_line_to_decimal)

    # -------------------------------------------------
    # ⚙️ 2️⃣ Probabilidades implícitas normalizadas
    # -------------------------------------------------
    for prefix in ["OP", ""]:
        suffix = f"_{prefix}" if prefix else ""
        df[f"Imp_H{suffix}_Norm"] = 1 / df[f"Odd_H{suffix}"]
        df[f"Imp_D{suffix}_Norm"] = 1 / df[f"Odd_D{suffix}"]
        df[f"Imp_A{suffix}_Norm"] = 1 / df[f"Odd_A{suffix}"]
        soma = df[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]].sum(axis=1).replace(0, np.nan)
        df[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]] = (
            df[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]].div(soma, axis=0).fillna(0)
        )

    # -------------------------------------------------
    # ⚙️ 3️⃣ Cria variações (Δ)
    # -------------------------------------------------
    df["Δ_Imp_H"] = df["Imp_H_Norm"] - df["Imp_H_OP_Norm"]
    df["Δ_Imp_A"] = df["Imp_A_Norm"] - df["Imp_A_OP_Norm"]
    df["Δ_Spread_HA"] = df["Δ_Imp_H"] - df["Δ_Imp_A"]
    if "Asian_Line_OP_Decimal" in df.columns:
        df["Δ_Asian_Line"] = df["Asian_Line_Decimal"] - df["Asian_Line_OP_Decimal"]
    else:
        df["Δ_Asian_Line"] = 0.0

    # -------------------------------------------------
    # ⚙️ 4️⃣ Target: movimento certo?
    # -------------------------------------------------
    df = df.dropna(subset=["Δ_Imp_H", "Δ_Imp_A", "Outcome_Home_FT"])
    df["Target_Market_Correct"] = np.where(
        ((df["Δ_Imp_H"] > 0) & (df["Outcome_Home_FT"] == 1)) |
        ((df["Δ_Imp_H"] < 0) & (df["Outcome_Home_FT"] == -1)),
        1, 0
    )

    # -------------------------------------------------
    # ⚙️ 5️⃣ Features dinâmicas
    # -------------------------------------------------
    base_features = [
        "Δ_Imp_H", "Δ_Imp_A", "Δ_Spread_HA",
        "Imp_H_OP_Norm", "Imp_A_OP_Norm",
        "Imp_H_Norm", "Imp_A_Norm",
        "Δ_Asian_Line", "Asian_Line_Decimal",
        "M_H", "M_A", "MT_H", "MT_A", "Diff_Power"
    ]
    features = [f for f in base_features if f in df.columns]
    if not features:
        st.warning("⚠️ ML2: nenhuma feature encontrada.")
        return None, games_today

    X = df[features].fillna(0)
    y = df["Target_Market_Correct"]

    # -------------------------------------------------
    # ⚙️ 6️⃣ Treina RandomForest
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model_market = RandomForestClassifier(n_estimators=450, max_depth=10, class_weight='balanced', random_state=42)
    model_market.fit(X_train, y_train)

    y_pred = model_market.predict(X_test)
    y_prob = model_market.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.markdown(f"""
    ### 📊 ML2 – Market Movement Model
    - Accuracy: **{acc:.3f}**
    - ROC-AUC: **{auc:.3f}**
    - Amostras: {len(df)}
    """)

    # -------------------------------------------------
    # ⚙️ 7️⃣ Aplica aos jogos do dia
    # -------------------------------------------------
    if not games_today.empty:
        temp = games_today.copy()

        for prefix in ["OP", ""]:
            suffix = f"_{prefix}" if prefix else ""
            temp[f"Imp_H{suffix}_Norm"] = 1 / temp[f"Odd_H{suffix}"]
            temp[f"Imp_D{suffix}_Norm"] = 1 / temp[f"Odd_D{suffix}"]
            temp[f"Imp_A{suffix}_Norm"] = 1 / temp[f"Odd_A{suffix}"]
            soma = temp[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]].sum(axis=1).replace(0, np.nan)
            temp[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]] = (
                temp[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]].div(soma, axis=0).fillna(0)
            )

        temp["Δ_Imp_H"] = temp["Imp_H_Norm"] - temp["Imp_H_OP_Norm"]
        temp["Δ_Imp_A"] = temp["Imp_A_Norm"] - temp["Imp_A_OP_Norm"]
        temp["Δ_Spread_HA"] = temp["Δ_Imp_H"] - temp["Δ_Imp_A"]
        if "Asian_Line_OP_Decimal" in temp.columns:
            temp["Δ_Asian_Line"] = temp["Asian_Line_Decimal"] - temp["Asian_Line_OP_Decimal"]
        else:
            temp["Δ_Asian_Line"] = 0.0

        temp = temp.fillna(0)
        X_today = temp[[f for f in features if f in temp.columns]]
        preds = model_market.predict_proba(X_today)[:, 1]
        temp["Market_Pred_Confidence"] = preds
        temp["Market_Pred_Side"] = np.where(temp["Δ_Imp_H"] > 0, "HOME", "AWAY")

        # -------------------------------------------------
        # ⚙️ 8️⃣ Divergência modelo x mercado
        # -------------------------------------------------
        if "ML_Side" in temp.columns:
            temp["Market_Model_Divergence"] = np.where(
                temp["ML_Side"].str.upper() == temp["Market_Pred_Side"], 1, -1
            )
        else:
            temp["Market_Model_Divergence"] = np.nan

        temp["Value_Bet_MarketBias"] = np.where(
            (temp["Market_Model_Divergence"] == -1) & (temp["Market_Pred_Confidence"] >= 0.60),
            True, False
        )

        games_today = temp.copy()

    return model_market, games_today


modelo_mercado, games_today = treinar_ml_movimento_mercado(history, games_today)


# =====================================================
# 🧩 BLOCO FINAL – SINERGIA ML1 + ML2
# =====================================================

import plotly.express as px

# -------------------------------------------------
# ⚙️ 1️⃣ Cria Score_Model_Chosen com base no lado previsto pela ML1
# -------------------------------------------------
if {"Quadrante_ML_Score_Home", "Quadrante_ML_Score_Away", "ML_Side"}.issubset(games_today.columns):
    games_today["Score_Model_Chosen"] = np.where(
        games_today["ML_Side"].str.upper() == "HOME",
        games_today["Quadrante_ML_Score_Home"],
        games_today["Quadrante_ML_Score_Away"]
    )
else:
    st.warning("⚠️ Colunas de score 3D não encontradas para gerar Score_Model_Chosen.")
    games_today["Score_Model_Chosen"] = np.nan


# -------------------------------------------------
# ⚙️ 2️⃣ Função principal de fusão ML1+ML2
# -------------------------------------------------
def combinar_modelos_ml1_ml2(games_today, lim_conf_modelo=0.55, lim_conf_mercado=0.50):
    """
    Integra previsões da ML1 (modelo 3D) e ML2 (movimento de mercado),
    calculando sinergia e consenso probabilístico.
    """
    df = games_today.copy()

    # -------------------------------------------------
    # ⚙️ 3️⃣ Verificação mínima
    # -------------------------------------------------
    if not {"ML_Side", "Score_Model_Chosen", "Market_Pred_Side", "Market_Pred_Confidence"}.issubset(df.columns):
        st.warning("⚠️ Fusão ML1+ML2: colunas necessárias não encontradas.")
        return df

    # -------------------------------------------------
    # ⚙️ 4️⃣ Concordância e divergência
    # -------------------------------------------------
    df["ML_Agree_Market"] = np.where(
        (df["ML_Side"].str.upper() == df["Market_Pred_Side"]) &
        (df["Score_Model_Chosen"] >= lim_conf_modelo) &
        (df["Market_Pred_Confidence"] >= lim_conf_mercado),
        True, False
    )

    df["ML_Diverge_Market"] = np.where(
        (df["ML_Side"].str.upper() != df["Market_Pred_Side"]) &
        (df["Score_Model_Chosen"] >= lim_conf_modelo) &
        (df["Market_Pred_Confidence"] >= lim_conf_mercado),
        True, False
    )

    # -------------------------------------------------
    # ⚙️ 5️⃣ Consensus Score
    # -------------------------------------------------
    df["Consensus_Score"] = (
        df["Score_Model_Chosen"] * 0.5 + df["Market_Pred_Confidence"] * 0.5
    ).round(3)

    # -------------------------------------------------
    # ⚙️ 6️⃣ Classificação do sinal
    # -------------------------------------------------
    df["Consensus_Label"] = np.select(
        [
            df["ML_Agree_Market"],
            df["ML_Diverge_Market"]
        ],
        [
            "✅ Full Agreement",
            "⚠️ Divergente Value"
        ],
        default="❕ Indefinido"
    )

    # -------------------------------------------------
    # ⚙️ 7️⃣ Painel Streamlit – Tabela
    # -------------------------------------------------
    st.markdown("## 🔄 Sinergia entre Modelo 3D e Mercado (ML1 + ML2)")

    df_exibir = df[
        [
            "League", "Home", "Away",'Goals_H_Today','Goals_A_Today',
            "ML_Side", "Market_Pred_Side",
            "Score_Model_Chosen", "Market_Pred_Confidence",
            "Consensus_Score", "Consensus_Label"
        ]
    ].sort_values("Consensus_Score", ascending=False)

    st.dataframe(
        df_exibir.style.applymap(
            lambda v: "background-color:#1e90ff; color:white;" if v == "✅ Full Agreement" else
                      ("background-color:#ff8c00; color:white;" if v == "⚠️ Divergente Value" else None),
            subset=["Consensus_Label"]
        ).format({
            "Goals_H_Today": "{:.0f}",
            "Goals_A_Today": "{:.0f}",
            "Score_Model_Chosen": "{:.2f}",
            "Market_Pred_Confidence": "{:.2f}",
            "Consensus_Score": "{:.2f}"
        })
    )

    # -------------------------------------------------
    # ⚙️ 8️⃣ Estatísticas e gráfico de proporção
    # -------------------------------------------------
    total = len(df)
    n_agree = df["ML_Agree_Market"].sum()
    n_div = df["ML_Diverge_Market"].sum()

    st.markdown(f"""
    ### 📈 Estatísticas de Sinergia
    - Total de jogos: **{total}**
    - Convergentes (modelo + mercado): **{n_agree} ({n_agree/total:.1%})**
    - Divergentes (value bets potenciais): **{n_div} ({n_div/total:.1%})**
    """)

    # Gráfico de proporção
    counts = df["Consensus_Label"].value_counts().reset_index()
    counts.columns = ["Tipo", "Qtd"]
    fig = px.bar(
        counts, x="Tipo", y="Qtd",
        color="Tipo",
        color_discrete_map={
            "✅ Full Agreement": "#1e90ff",
            "⚠️ Divergente Value": "#ff8c00",
            "❕ Indefinido": "#808080"
        },
        title="Distribuição de Concordância ML1 vs ML2"
    )
    st.plotly_chart(fig, use_container_width=True)

    return df


# -------------------------------------------------
# ⚙️ 9️⃣ Execução final
# -------------------------------------------------
games_today = combinar_modelos_ml1_ml2(games_today)


# =====================================================
# 📊 ANÁLISE DE MOVIMENTO DE ODDS (Abertura → Fecho)
# =====================================================

if {"Odd_H", "Odd_D", "Odd_A", "Odd_H_OP", "Odd_D_OP", "Odd_A_OP"}.issubset(games_today.columns):

    # Diferenças absolutas e percentuais
    games_today["ΔOdd_H_%"] = ((games_today["Odd_H"] - games_today["Odd_H_OP"]) / games_today["Odd_H_OP"] * 100).round(2)
    games_today["ΔOdd_D_%"] = ((games_today["Odd_D"] - games_today["Odd_D_OP"]) / games_today["Odd_D_OP"] * 100).round(2)
    games_today["ΔOdd_A_%"] = ((games_today["Odd_A"] - games_today["Odd_A_OP"]) / games_today["Odd_A_OP"] * 100).round(2)

    # Label interpretativo
    def interpretar_movimento(r):
        movimentos = []
        if r["ΔOdd_H_%"] < -2:
            movimentos.append("📉 Home caiu")
        elif r["ΔOdd_H_%"] > 2:
            movimentos.append("📈 Home subiu")

        if r["ΔOdd_A_%"] < -2:
            movimentos.append("📉 Away caiu")
        elif r["ΔOdd_A_%"] > 2:
            movimentos.append("📈 Away subiu")

        if r["ΔOdd_D_%"] < -2:
            movimentos.append("📉 Draw caiu")
        elif r["ΔOdd_D_%"] > 2:
            movimentos.append("📈 Draw subiu")

        return ", ".join(movimentos) if movimentos else "⚖️ Estável"

    games_today["Market_Move_Label"] = games_today.apply(interpretar_movimento, axis=1)

    # Mostra tabela resumida no Streamlit
    st.markdown("## 📊 Movimento de Odds – Abertura → Fecho")
    st.dataframe(
        games_today[
            ["League", "Home", "Away", "ML_Side", "Market_Pred_Side",'Goals_H_Today','Goals_A_Today',
             "Odd_H_OP", "Odd_H", "ΔOdd_H_%",
             "Odd_D_OP", "Odd_D", "ΔOdd_D_%",
             "Odd_A_OP", "Odd_A", "ΔOdd_A_%",
             "Market_Move_Label"]
        ].sort_values("ΔOdd_H_%", ascending=True)
        .style.format({
            "Goals_H_Today": "{:.0f}",
            "Goals_A_Today": "{:.0f}",
            "Odd_H_OP": "{:.2f}", "Odd_H": "{:.2f}", "ΔOdd_H_%": "{:+.2f}%",
            "Odd_D_OP": "{:.2f}", "Odd_D": "{:.2f}", "ΔOdd_D_%": "{:+.2f}%",
            "Odd_A_OP": "{:.2f}", "Odd_A": "{:.2f}", "ΔOdd_A_%": "{:+.2f}%"
        })
        .applymap(lambda v: "background-color:#228B22; color:white" if isinstance(v, float) and v < -2 else None, subset=["ΔOdd_H_%", "ΔOdd_D_%", "ΔOdd_A_%"])
        .applymap(lambda v: "background-color:#B22222; color:white" if isinstance(v, float) and v > 2 else None, subset=["ΔOdd_H_%", "ΔOdd_D_%", "ΔOdd_A_%"])
    )
else:
    st.warning("⚠️ Colunas de odds de abertura/fecho não encontradas para análise de movimento.")



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
