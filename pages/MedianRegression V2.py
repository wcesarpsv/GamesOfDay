from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import math
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Análise de Quadrantes 3D - Bet Indicator", layout="wide")
st.title("🎯 Análise 3D de 16 Quadrantes - ML Avançado (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "coppa", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas do Live Score existam no DataFrame"""
    df = df.copy()
    for col in ['Goals_H_Today', 'Goals_A_Today', 'Home_Red', 'Away_Red']:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------- Helpers Básicos ----------------
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Desduplicar colunas de gols se vierem de merges anteriores
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

    Regras gerais:
    - Valores simples: float(value) e inverte sinal (Away → Home)
    - Splits: média dos dois lados, mantendo sinal do primeiro, depois inverte (Away → Home)
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
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result  # Inverte sinal (Away → Home)
    except ValueError:
        return np.nan


def calculate_ah_home_target(margin, asian_line_str):
    """Calcula target AH Home diretamente da string original"""
    line_home = convert_asian_line_to_decimal(asian_line_str)
    if pd.isna(line_home) or pd.isna(margin):
        return np.nan
    return 1 if margin > line_home else 0


def aplicar_clusterizacao_3d_segura(history: pd.DataFrame,
                                    games_today: pd.DataFrame,
                                    n_clusters: int = 5):
    """
    Clusterização temporalmente segura:
    - Treina clusters apenas nos dados históricos
    - Aplica nos dados históricos e nos jogos de hoje
    """
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']

    history_clean = history[required_cols].fillna(0).copy()
    games_today_clean = games_today[required_cols].fillna(0).copy()

    # Diferenças espaciais
    for df_name, df_clean in [('history', history_clean), ('games_today', games_today_clean)]:
        df_clean['dx'] = df_clean['Aggression_Home'] - df_clean['Aggression_Away']
        df_clean['dy'] = df_clean['M_H'] - df_clean['M_A']
        df_clean['dz'] = df_clean['MT_H'] - df_clean['MT_A']

    X_train = history_clean[['dx', 'dy', 'dz']].values

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        init='k-means++',
        n_init=10
    )
    kmeans.fit(X_train)

    history = history.copy()
    games_today = games_today.copy()

    history['Cluster3D_Label'] = kmeans.predict(history_clean[['dx', 'dy', 'dz']].values)
    games_today['Cluster3D_Label'] = kmeans.predict(games_today_clean[['dx', 'dy', 'dz']].values)

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

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")


@st.cache_data(ttl=3600)
def load_cached_data(selected_file: str):
    """Carrega e filtra games_today + histórico pesado com cache."""
    games_today_local = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today_local = filter_leagues(games_today_local)

    history_local = filter_leagues(load_all_games(GAMES_FOLDER))
    # Garante FT + Asian_Line no histórico base
    history_local = history_local.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

    return games_today_local, history_local


# Carregar via cache
games_today, history = load_cached_data(selected_file)


# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today: pd.DataFrame, selected_date_str: str) -> pd.DataFrame:
    """Carrega e faz merge dos dados do Live Score"""
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

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

        games_today = games_today.merge(
            results_df,
            on='Id',
            how='left',
            suffixes=('', '_RAW')
        )

        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

        games_today['Home_Red'] = games_today['home_red']
        games_today['Away_Red'] = games_today['away_red']

        st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
        return games_today
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today


# Aplicar LiveScore
games_today = load_and_merge_livescore(games_today, selected_date_str)

# ---------------- CONVERSÃO ASIAN LINE ----------------
history = history.copy()
games_today = games_today.copy()

history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

history = history.dropna(subset=['Asian_Line_Decimal'])
st.info(f"📊 Histórico com Asian Line válida: {len(history)} jogos")

# Filtro temporal anti-leakage
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
    for qid, config in QUADRANTES_16.items():
        if config['agg_min'] <= agg <= config['agg_max'] and config['hs_min'] <= hs <= config['hs_max']:
            return qid
    return 0


# Aplicar classificação
for df in [games_today, history]:
    df['Quadrante_Home'] = df.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    df['Quadrante_Away'] = df.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )


# ---------------- CÁLCULO DE DISTÂNCIAS 3D ----------------
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing}")
        for col in [
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Quadrant_Angle_XY', 'Quadrant_Angle_XZ', 'Quadrant_Angle_YZ',
            'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
            'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
            'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
            'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
            'Vector_Sign', 'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D'
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


# ---------------- VISUALIZAÇÃO 2D DOS 16 QUADRANTES ----------------
def plot_quadrantes_16(df: pd.DataFrame, side: str = "Home"):
    fig, ax = plt.subplots(figsize=(14, 10))

    cores_categorias = {
        'Fav Forte': 'lightcoral',
        'Fav Moderado': 'lightpink',
        'Under Moderado': 'lightblue',
        'Under Forte': 'lightsteelblue'
    }

    for qid in range(1, 17):
        mask = df[f'Quadrante_{side}'] == qid
        if mask.any():
            nome = QUADRANTES_16[qid]['nome']
            cat = " ".join(nome.split()[:2])
            cor = cores_categorias.get(cat, 'gray')
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(x, y, c=cor,
                       label=nome,
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


# ---------------- REGRESSÃO À MÉDIA COMPLETA ----------------
def adicionar_regressao_media_completa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['MT_H_Abs'] = df['MT_H'].abs()
    df['MT_A_Abs'] = df['MT_A'].abs()

    for col in ['Streak_Extremo_H', 'Streak_Extremo_A',
                'Games_Above_Expected_H', 'Games_Above_Expected_A']:
        if col not in df.columns:
            df[col] = 0

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

    df['HS_Reversion_Penalty_H'] = np.where(
        (df.get('HandScore_Home', 0) > 45) & (df['MT_H'] < -0.8), -0.7,
        np.where((df.get('HandScore_Home', 0) < -45) & (df['MT_H'] > 1.5), 0.6, 0)
    )
    df['HS_Reversion_Penalty_A'] = np.where(
        (df.get('HandScore_Away', 0) > 45) & (df['MT_A'] < -0.8), -0.7,
        np.where((df.get('HandScore_Away', 0) < -45) & (df['MT_A'] > 1.5), 0.6, 0)
    )

    if 'Imp_H_OP_Norm' in df.columns:
        shrinkage = 0.12
        df['Imp_H_Shrinked'] = (1 - shrinkage) * df['Imp_H_OP_Norm'] + shrinkage * (1/3)
        df['Imp_A_Shrinked'] = (1 - shrinkage) * df['Imp_A_OP_Norm'] + shrinkage * (1/3)

    return df


# ---------------- VISUALIZAÇÃO 3D FIXA + FILTRO ANGULAR ----------------
st.markdown("## 🎯 Visualização Interativa 3D – Tamanho Fixo")

if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())
    selected_leagues = st.multiselect(
        "Selecione uma ou mais ligas para análise:",
        options=leagues,
        default=[],
        help="Escolha múltiplas ligas para comparar comportamentos entre campeonatos diferentes."
    )
    df_filtered = games_today[games_today["League"].isin(selected_leagues)].copy() if selected_leagues else games_today.copy()
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
        step=5
    )
with col_ang2:
    angulo_xz_range = st.slider(
        "Ângulo XZ - Aggression × Momentum Time:",
        -180, 180, (-45, 45),
        step=5
    )
with col_ang3:
    magnitude_min = st.slider(
        "Magnitude Mínima 3D:",
        0.0, 5.0, 0.5, 0.1
    )

aplicar_filtro = st.button("🎯 Aplicar Filtros Angulares", type="primary")


def filtrar_por_angulo(df: pd.DataFrame,
                       angulo_xy_range,
                       angulo_xz_range,
                       magnitude_min: float) -> pd.DataFrame:
    df_filtrado = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df_filtrado.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para filtro angular: {missing}")
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

    final_mask = mask_xy & mask_xz & mask_mag
    df_filtrado = df_filtrado[final_mask].copy()

    df_filtrado['Angulo_XY'] = angulo_xy[final_mask]
    df_filtrado['Angulo_XZ'] = angulo_xz[final_mask]
    df_filtrado['Magnitude_3D_Filtro'] = magnitude[final_mask]

    return df_filtrado


# PREPARAR df_plot SEM DUPLICAR LÓGICA (FILTRO ANGULAR CORRIGIDO)
if aplicar_filtro:
    df_plot = filtrar_por_angulo(df_filtered, angulo_xy_range, angulo_xz_range, magnitude_min)
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
    df_plot = df_filtered.nlargest(n_to_show, "Quadrant_Dist_3D").reset_index(drop=True)


def create_fixed_3d_plot(df_plot: pd.DataFrame,
                         n_to_show: int,
                         selected_league: str):
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

    titulo_3d = f"Top {min(n_to_show, len(df_plot))} Distâncias 3D – Tamanho Fixo"
    if selected_league != "⚽ Todas as ligas":
        titulo_3d += f" | {selected_league}"

    fig_3d.update_layout(
        title=dict(text=titulo_3d, x=0.5, font=dict(size=16, color='white')),
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
                eye=dict(x=0.0, y=0.0, z=1.2),
                up=dict(x=0.3, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
    )
    return fig_3d


selected_league_label = ", ".join(selected_leagues) if 'selected_leagues' in locals() and selected_leagues else "⚽ Todas as ligas"
fig_3d_fixed = create_fixed_3d_plot(df_plot, n_to_show, selected_league_label)
st.plotly_chart(fig_3d_fixed, use_container_width=True)

st.markdown("""
### 🎯 Legenda do Espaço 3D Fixo

**Eixos com Ranges Fixos:**
- **X (Vermelho)**: Aggression → `-1.2` (Zebra Extrema) ↔ `+1.2` (Favorito Extremo)
- **Y (Verde)**: Momentum Liga → `-4.0` (Muito Negativo) ↔ `+4.0` (Muito Positivo)  
- **Z (Azul)**: Momentum Time → `-4.0` (Muito Negativo) ↔ `+4.0` (Muito Positivo)

**Referências Visuais:**
- 📍 **Plano Cinza**: Ponto neutro (Z=0)
- 🔵 **Bolas Azuis**: Times da Casa (Home)
- 🔴 **Losangos Vermelhos**: Visitantes (Away)
- ⚫ **Linhas Cinzas**: Conexões entre confrontos
""")

# Clusterização + Regressão à Média
history, games_today = aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5)
history = adicionar_regressao_media_completa(history)
games_today = adicionar_regressao_media_completa(games_today)


def treinar_modelo_3d_clusters_single(history: pd.DataFrame,
                                      games_today: pd.DataFrame):
    st.markdown("### ⚙️ Configuração do Treino 3D com Odds de Abertura")
    use_opening_odds = st.checkbox("📊 Incluir Odds de Abertura no Treino", value=True)

    history_local = calcular_distancias_3d(history)
    games_today_local = calcular_distancias_3d(games_today)
    history_local, games_today_local = aplicar_clusterizacao_3d_segura(history_local, games_today_local, n_clusters=5)

    for df in [history_local, games_today_local]:
        if 'League' not in df.columns:
            df['League'] = 'Unknown'

    ligas_dummies = pd.get_dummies(history_local['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history_local['Cluster3D_Label'], prefix='C3D')

    features_3d_base = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D',
        'MT_Reversion_Score_H', 'MT_Reversion_Score_A',
        'HS_Reversion_Penalty_H', 'HS_Reversion_Penalty_A',
        'Streak_Extremo_H', 'Streak_Extremo_A',
        'Games_Above_Expected_H', 'Games_Above_Expected_A'
    ]

    for feature in features_3d_base:
        for df in [history_local, games_today_local]:
            if feature not in df.columns:
                df[feature] = 0.0

    for df in [history_local, games_today_local]:
        if 'Quadrante_Bayes_Score_H' not in df.columns:
            df['Quadrante_Bayes_Score_H'] = 0.5

    features_3d_existentes = [f for f in features_3d_base + ['Quadrante_Bayes_Score_H'] if f in history_local.columns]
    extras_3d = history_local[features_3d_existentes].fillna(0)

    odds_features = pd.DataFrame()
    if use_opening_odds:
        for df in [history_local, games_today_local]:
            for col in ['Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP', 'Odd_H', 'Odd_D', 'Odd_A']:
                if col not in df.columns:
                    df[col] = 3.0

        for df_tmp, name in [(history_local, "history"), (games_today_local, "games_today")]:
            df_tmp['Imp_H_OP'] = 1 / df_tmp['Odd_H_OP']
            df_tmp['Imp_D_OP'] = 1 / df_tmp['Odd_D_OP']
            df_tmp['Imp_A_OP'] = 1 / df_tmp['Odd_A_OP']
            df_tmp[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']] = df_tmp[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].replace([np.inf, -np.inf], np.nan)

            sum_probs = df_tmp[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].sum(axis=1).replace(0, np.nan)
            df_tmp['Imp_H_OP_Norm'] = df_tmp['Imp_H_OP'] / sum_probs
            df_tmp['Imp_D_OP_Norm'] = df_tmp['Imp_D_OP'] / sum_probs
            df_tmp['Imp_A_OP_Norm'] = df_tmp['Imp_A_OP'] / sum_probs

        history_local['Diff_Odd_H'] = history_local['Odd_H_OP'] - history_local['Odd_H']
        history_local['Diff_Odd_D'] = history_local['Odd_D_OP'] - history_local['Odd_D']
        history_local['Diff_Odd_A'] = history_local['Odd_A_OP'] - history_local['Odd_A']

        shrinkage = 0.12
        history_local['Imp_H_Shrinked'] = (1 - shrinkage) * history_local['Imp_H_OP_Norm'] + shrinkage * (1/3)
        history_local['Imp_A_Shrinked'] = (1 - shrinkage) * history_local['Imp_A_OP_Norm'] + shrinkage * (1/3)

        odds_cols = ['Imp_H_OP_Norm', 'Imp_D_OP_Norm', 'Imp_A_OP_Norm',
                     'Diff_Odd_H', 'Diff_Odd_D', 'Diff_Odd_A',
                     'Imp_H_Shrinked', 'Imp_A_Shrinked']
        odds_cols_existentes = [c for c in odds_cols if c in history_local.columns]
        odds_features = history_local[odds_cols_existentes].fillna(0)
    else:
        odds_cols_existentes = []

    if use_opening_odds and not odds_features.empty:
        X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, odds_features], axis=1)
    else:
        X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)

    if 'Target_AH_Home' not in history_local.columns:
        st.error("❌ Target_AH_Home não encontrado no histórico. Verifique os dados.")
        return None, games_today_local

    y_home = history_local['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model_home.fit(X, y_home)

    ligas_today = pd.get_dummies(games_today_local['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today_local['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today_local[features_3d_existentes].fillna(0)

    if use_opening_odds:
        games_today_local['Imp_H_OP'] = 1 / games_today_local['Odd_H_OP']
        games_today_local['Imp_D_OP'] = 1 / games_today_local['Odd_D_OP']
        games_today_local['Imp_A_OP'] = 1 / games_today_local['Odd_A_OP']
        games_today_local[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']] = games_today_local[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].replace([np.inf, -np.inf], np.nan)

        sum_today = games_today_local[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].sum(axis=1).replace(0, np.nan)
        games_today_local['Imp_H_OP_Norm'] = games_today_local['Imp_H_OP'] / sum_today
        games_today_local['Imp_D_OP_Norm'] = games_today_local['Imp_D_OP'] / sum_today
        games_today_local['Imp_A_OP_Norm'] = games_today_local['Imp_A_OP'] / sum_today

        games_today_local['Diff_Odd_H'] = games_today_local['Odd_H_OP'] - games_today_local['Odd_H']
        games_today_local['Diff_Odd_D'] = games_today_local['Odd_D_OP'] - games_today_local['Odd_D']
        games_today_local['Diff_Odd_A'] = games_today_local['Odd_A_OP'] - games_today_local['Odd_A']

        shrinkage = 0.12
        games_today_local['Imp_H_Shrinked'] = (1 - shrinkage) * games_today_local['Imp_H_OP_Norm'] + shrinkage * (1/3)
        games_today_local['Imp_A_Shrinked'] = (1 - shrinkage) * games_today_local['Imp_A_OP_Norm'] + shrinkage * (1/3)

        odds_today = games_today_local[odds_cols_existentes].fillna(0)
        X_today = pd.concat([ligas_today, clusters_today, extras_today, odds_today], axis=1)
    else:
        X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    missing_cols = set(X.columns) - set(X_today.columns)
    for col in missing_cols:
        X_today[col] = 0
    X_today = X_today[X.columns]

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today_local['Prob_Home'] = proba_home
    games_today_local['Prob_Away'] = proba_away
    games_today_local['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today_local['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today_local['Quadrante_ML_Score_Home'] = games_today_local['Prob_Home']
    games_today_local['Quadrante_ML_Score_Away'] = games_today_local['Prob_Away']
    games_today_local['Quadrante_ML_Score_Main'] = games_today_local['ML_Confidence']

    accuracy = model_home.score(X, y_home)
    st.metric("Accuracy (Treino)", f"{accuracy:.2%}")
    st.write("📘 Features usadas:", len(X.columns))

    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_feats = importances.head(25).to_frame("Importância")

    st.markdown("### 🔍 Top Features (Modelo Único – Home)")
    st.dataframe(top_feats, use_container_width=True)

    if "Quadrante_ML_Score_Home" not in games_today_local.columns:
        games_today_local["Quadrante_ML_Score_Home"] = np.nan
        games_today_local["Quadrante_ML_Score_Away"] = np.nan
        games_today_local["Quadrante_ML_Score_Main"] = np.nan
        games_today_local["ML_Side"] = "N/A"
        games_today_local["ML_Confidence"] = 0.0

    for col in ["League", "Home", "Away"]:
        if col not in games_today_local.columns:
            games_today_local[col] = "N/A"

    if games_today_local.empty:
        st.warning("⚠️ Nenhum jogo válido encontrado após o treino. Verifique o CSV e as odds.")
    else:
        st.success(f"✅ {len(games_today_local)} jogos processados e prontos para análise 3D.")

    st.success("✅ Modelo 3D treinado (HOME) – com análise de viés integrada.")
    return model_home, games_today_local


def adicionar_indicadores_explicativos_3d_16_dual(df: pd.DataFrame) -> pd.DataFrame:
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
        elif score_home >= 0.75 and momentum_h >= -0.5:
            return f'🎯 VALUE HOME (Score Alto) ({score_home:.1%})'
        elif score_away >= 0.75 and momentum_a >= -0.5:
            return f'🎯 VALUE AWAY (Score Alto) ({score_away:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_3d_16_dual, axis=1)
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
    momentum_boost = momentum * 10
    adjusted_score = base_score + momentum_boost
    return max(0, min(100, adjusted_score))


def gerar_score_combinado_3d_16(df: pd.DataFrame) -> pd.DataFrame:
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


st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not history.empty:
    modelo_home, games_today = treinar_modelo_3d_clusters_single(history, games_today)
    st.success("✅ Modelo 3D dual com 16 quadrantes treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_3d = games_today.copy()
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
    ranking_3d = gerar_score_combinado_3d_16(ranking_3d)

    if 'Profit_Quadrante' not in ranking_3d.columns:
        ranking_3d['Profit_Quadrante'] = np.nan

    ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

    colunas_3d = [
        'League', 'Time',
        'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'Recomendacao', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
        'Score_Final_3D', 'Classificacao_Potencial_3D',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        'M_H', 'M_A', 'Quadrant_Dist_3D', 'Momentum_Diff',
        'Asian_Line_Decimal', 'Home_Red', 'Away_Red', 'Profit_Quadrante'
    ]
    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    def estilo_tabela_3d_quadrantes(df: pd.DataFrame):
        def cor_classificacao_3d(valor):
            v = str(valor)
            if any(tag in v for tag in [
                '🌟 ALTO POTENCIAL 3D', '💼 VALOR SOLIDO 3D',
                '🔴 BAIXO POTENCIAL 3D', '🏆 ALTO VALOR', '🔴 ALTO RISCO',
                'VALUE', 'EVITAR'
            ]):
                return 'font-weight: bold'
            return ''

        colunas_para_estilo = [c for c in [
            'Classificacao_Potencial_3D',
            'Classificacao_Valor_Home',
            'Classificacao_Valor_Away',
            'Recomendacao'
        ] if c in df.columns]

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

    st.dataframe(
        estilo_tabela_3d_quadrantes(ranking_3d[cols_finais_3d]).format({
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
else:
    ranking_3d = pd.DataFrame()
    st.info("⚠️ Aguardando dados para gerar ranking 3D de 16 quadrantes")


def resumo_3d_16_quadrantes_hoje(df: pd.DataFrame):
    st.markdown("### 📋 Resumo Executivo - Sistema 3D Hoje")
    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)
    alto_potencial_3d = len(df[df['Classificacao_Potencial_3D'] == '🌟 ALTO POTENCIAL 3D'])
    valor_solido_3d = len(df[df['Classificacao_Potencial_3D'] == '💼 VALOR SOLIDO 3D'])

    alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
    alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])

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

    st.markdown("#### 📊 Distribuição de Recomendações 3D")
    if 'Recomendacao' in df.columns:
        dist_recomendacoes = df['Recomendacao'].value_counts()
        st.dataframe(dist_recomendacoes, use_container_width=True)


if not games_today.empty and 'Classificacao_Potencial_3D' in games_today.columns:
    resumo_3d_16_quadrantes_hoje(games_today)


# =====================================================
# 🧠 ML2 – MOVIMENTO DE MERCADO (Market Bias Model)
# =====================================================
def treinar_ml_movimento_mercado(history: pd.DataFrame,
                                 games_today: pd.DataFrame):
    df = history.copy()

    if "Outcome_Home_FT" not in df.columns:
        if {"Goals_H_FT", "Goals_A_FT"}.issubset(df.columns):
            df["Outcome_Home_FT"] = np.sign(df["Goals_H_FT"] - df["Goals_A_FT"]).astype("Int64")
        else:
            st.warning("⚠️ ML2: sem FT no histórico — modelo ignorado.")
            return None, games_today

    for df_tmp in (df, games_today):
        if "Asian_Line_OP" in df_tmp.columns and "Asian_Line_OP_Decimal" not in df_tmp.columns:
            df_tmp["Asian_Line_OP_Decimal"] = df_tmp["Asian_Line_OP"].apply(convert_asian_line_to_decimal)

    for prefix in ["OP", ""]:
        suffix = f"_{prefix}" if prefix else ""
        for col in [f"Odd_H{suffix}", f"Odd_D{suffix}", f"Odd_A{suffix}"]:
            if col not in df.columns:
                st.warning(f"⚠️ ML2: coluna {col} ausente no histórico.")
                return None, games_today

        df[f"Imp_H{suffix}_Norm"] = 1 / df[f"Odd_H{suffix}"]
        df[f"Imp_D{suffix}_Norm"] = 1 / df[f"Odd_D{suffix}"]
        df[f"Imp_A{suffix}_Norm"] = 1 / df[f"Odd_A{suffix}"]
        soma = df[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]].sum(axis=1).replace(0, np.nan)
        df[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]] = (
            df[[f"Imp_H{suffix}_Norm", f"Imp_D{suffix}_Norm", f"Imp_A{suffix}_Norm"]].div(soma, axis=0).fillna(0)
        )

    df["Δ_Imp_H"] = df["Imp_H_Norm"] - df["Imp_H_OP_Norm"]
    df["Δ_Imp_A"] = df["Imp_A_Norm"] - df["Imp_A_OP_Norm"]
    df["Δ_Spread_HA"] = df["Δ_Imp_H"] - df["Δ_Imp_A"]
    if "Asian_Line_OP_Decimal" in df.columns:
        df["Δ_Asian_Line"] = df["Asian_Line_Decimal"] - df["Asian_Line_OP_Decimal"]
    else:
        df["Δ_Asian_Line"] = 0.0

    df = df.dropna(subset=["Δ_Imp_H", "Δ_Imp_A", "Outcome_Home_FT"])

    df["Target_Market_Correct"] = np.where(
        ((df["Δ_Imp_H"] > 0) & (df["Outcome_Home_FT"] == 1)) |
        ((df["Δ_Imp_H"] < 0) & (df["Outcome_Home_FT"] == -1)),
        1, 0
    )

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model_market = RandomForestClassifier(
        n_estimators=450, max_depth=10,
        class_weight='balanced',
        random_state=42
    )
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

    if not games_today.empty:
        temp = games_today.copy()

        for prefix in ["OP", ""]:
            suffix = f"_{prefix}" if prefix else ""
            for col in [f"Odd_H{suffix}", f"Odd_D{suffix}", f"Odd_A{suffix}"]:
                if col not in temp.columns:
                    temp[col] = 3.0

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
if {"Quadrante_ML_Score_Home", "Quadrante_ML_Score_Away", "ML_Side"}.issubset(games_today.columns):
    games_today["Score_Model_Chosen"] = np.where(
        games_today["ML_Side"].str.upper() == "HOME",
        games_today["Quadrante_ML_Score_Home"],
        games_today["Quadrante_ML_Score_Away"]
    )
else:
    st.warning("⚠️ Colunas de score 3D não encontradas para gerar Score_Model_Chosen.")
    games_today["Score_Model_Chosen"] = np.nan


def combinar_modelos_ml1_ml2(games_today: pd.DataFrame,
                             lim_conf_modelo: float = 0.55,
                             lim_conf_mercado: float = 0.50) -> pd.DataFrame:
    df = games_today.copy()
    required = {"ML_Side", "Score_Model_Chosen", "Market_Pred_Side", "Market_Pred_Confidence"}
    if not required.issubset(df.columns):
        st.warning("⚠️ Fusão ML1+ML2: colunas necessárias não encontradas.")
        return df

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

    df["Consensus_Score"] = (
        df["Score_Model_Chosen"] * 0.5 + df["Market_Pred_Confidence"] * 0.5
    ).round(3)

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

    st.markdown("## 🔄 Sinergia entre Modelo 3D e Mercado (ML1 + ML2)")

    cols_show = [
        "League", "Home", "Away", 'Goals_H_Today', 'Goals_A_Today',
        "ML_Side", "Market_Pred_Side",
        "Score_Model_Chosen", "Market_Pred_Confidence",
        "Consensus_Score", "Consensus_Label"
    ]
    cols_show = [c for c in cols_show if c in df.columns]

    df_exibir = df[cols_show].sort_values("Consensus_Score", ascending=False)

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
        }),
        use_container_width=True
    )

    total = len(df)
    n_agree = df["ML_Agree_Market"].sum()
    n_div = df["ML_Diverge_Market"].sum()

    st.markdown(f"""
    ### 📈 Estatísticas de Sinergia
    - Total de jogos: **{total}**
    - Convergentes (modelo + mercado): **{n_agree} ({n_agree/total:.1%})**
    - Divergentes (value bets potenciais): **{n_div} ({n_div/total:.1%})**
    """)

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


games_today = combinar_modelos_ml1_ml2(games_today)


# =====================================================
# 📊 ANÁLISE DE MOVIMENTO DE ODDS (Abertura → Fecho)
# =====================================================
if {"Odd_H", "Odd_D", "Odd_A", "Odd_H_OP", "Odd_D_OP", "Odd_A_OP"}.issubset(games_today.columns):

    games_today["ΔOdd_H_%"] = ((games_today["Odd_H"] - games_today["Odd_H_OP"]) / games_today["Odd_H_OP"] * 100).round(2)
    games_today["ΔOdd_D_%"] = ((games_today["Odd_D"] - games_today["Odd_D_OP"]) / games_today["Odd_D_OP"] * 100).round(2)
    games_today["ΔOdd_A_%"] = ((games_today["Odd_A"] - games_today["Odd_A_OP"]) / games_today["Odd_A_OP"] * 100).round(2)

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

    st.markdown("## 📊 Movimento de Odds – Abertura → Fecho")
    cols_move = [
        "League", "Home", "Away", "ML_Side", "Market_Pred_Side",
        'Goals_H_Today', 'Goals_A_Today',
        "Odd_H_OP", "Odd_H", "ΔOdd_H_%",
        "Odd_D_OP", "Odd_D", "ΔOdd_D_%",
        "Odd_A_OP", "Odd_A", "ΔOdd_A_%",
        "Market_Move_Label"
    ]
    cols_move = [c for c in cols_move if c in games_today.columns]

    st.dataframe(
        games_today[cols_move]
        .sort_values("ΔOdd_H_%", ascending=True)
        .style.format({
            "Goals_H_Today": "{:.0f}",
            "Goals_A_Today": "{:.0f}",
            "Odd_H_OP": "{:.2f}", "Odd_H": "{:.2f}", "ΔOdd_H_%": "{:+.2f}%",
            "Odd_D_OP": "{:.2f}", "Odd_D": "{:.2f}", "ΔOdd_D_%": "{:+.2f}%",
            "Odd_A_OP": "{:.2f}", "Odd_A": "{:.2f}", "ΔOdd_A_%": "{:+.2f}%"
        })
        .applymap(
            lambda v: "background-color:#228B22; color:white"
            if isinstance(v, float) and v < -2 else None,
            subset=["ΔOdd_H_%", "ΔOdd_D_%", "ΔOdd_A_%"]
        )
        .applymap(
            lambda v: "background-color:#B22222; color:white"
            if isinstance(v, float) and v > 2 else None,
            subset=["ΔOdd_H_%", "ΔOdd_D_%", "ΔOdd_A_%"]
        ),
        use_container_width=True
    )
else:
    st.warning("⚠️ Colunas de odds de abertura/fecho não encontradas para análise de movimento.")


st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML + Open/Close Market Movement** carregado com sucesso!")
st.info("""
**Resumo das melhorias 3D / Open-Close:**
- 🔢 16 quadrantes com análise 3D completa
- 📊 Momentum integrado como terceira dimensão
- 🎯 Distâncias e ângulos 3D com filtro angular corrigido
- 📈 Visualização 3D fixa para referência espacial
- 🧠 ML1 (3D Handicap) + ML2 (Movimento de Mercado) em sinergia
- 💹 Análise de movimento de odds Abertura → Fecho (Open/Close MR)
""")
