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
st.title("🎯 Análise 3D de 16 Quadrantes - ML Avançado (Home & Away)")

# ========================= CONFIGURAÇÕES GERAIS =========================
PAGE_PREFIX = "QuadrantesML_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ============================================================
# 🔧 LIVE SCORE – COLUNAS BÁSICAS
# ============================================================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas do Live Score existam no DataFrame."""
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

# ============================================================
# 🔧 HELPERS BÁSICOS – LOAD / PREPROCESS
# ============================================================
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

# ============================================================
# 🔧 ASIAN HANDICAP – CONVERSÃO E TARGET BÁSICO
# ============================================================
def convert_asian_line_to_decimal(value):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    Ex.: '0/0.5' (away +0.25) -> home -0.25
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Caso simples – número único
    if "/" not in value:
        try:
            num = float(value)
            return -num  # Inverte sinal (Away → Home)
        except ValueError:
            return np.nan

    # Caso duplo – média dos dois lados, mantendo sinal de origem
    try:
        parts = [float(p) for p in value.split("/")]
        avg = np.mean(parts)
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        # Inverter sinal no final (Away → Home)
        return -result
    except ValueError:
        return np.nan

def calculate_ah_home_target(margin, asian_line_str):
    """Calcula target AH Home diretamente da string original."""
    line_home = convert_asian_line_to_decimal(asian_line_str)
    if pd.isna(line_home) or pd.isna(margin):
        return np.nan
    return 1 if margin > line_home else 0

# ============================================================
# 🧮 RESULTADO & PROFIT HISTÓRICO PARA EV (AH) – HISTORY
# ============================================================
def determine_handicap_result_history(row, side="HOME"):
    """
    Determina o resultado do handicap asiático no HISTÓRICO,
    usando Goals_H_FT / Goals_A_FT e Asian_Line_Decimal.
    Retorna:
      - 'HOME_COVERED', 'AWAY_COVERED', 'HALF_WIN', 'HALF_LOSS', 'PUSH'
    """
    try:
        gh = float(row['Goals_H_FT'])
        ga = float(row['Goals_A_FT'])
        asian_line = float(row['Asian_Line_Decimal'])
    except (KeyError, TypeError, ValueError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line):
        return None

    frac = abs(asian_line % 1)
    is_quarter = frac in [0.25, 0.75]

    def single_result(gh, ga, line, side):
        if side == "HOME":
            adjusted = (gh + line) - ga
        else:
            adjusted = (ga - line) - gh
        if adjusted > 0:
            return 1.0
        elif adjusted == 0:
            return 0.5
        else:
            return 0.0

    if is_quarter:
        if asian_line > 0:
            line1 = math.floor(asian_line * 2) / 2
            line2 = line1 + 0.5
        else:
            line1 = math.ceil(asian_line * 2) / 2
            line2 = line1 - 0.5

        r1 = single_result(gh, ga, line1, side)
        r2 = single_result(gh, ga, line2, side)
        avg = (r1 + r2) / 2

        if avg == 1:
            return f"{side}_COVERED"
        elif avg == 0.75:
            return "HALF_WIN"
        elif avg == 0.5:
            return "PUSH"
        elif avg == 0.25:
            return "HALF_LOSS"
        else:
            return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"

    # Linhas padrão (0, .5, 1, 1.5, etc.)
    if side == "HOME":
        adjusted = (gh + asian_line) - ga
    else:
        adjusted = (ga - asian_line) - gh

    if adjusted > 0:
        return f"{side}_COVERED"
    elif adjusted < 0:
        return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"
    else:
        return "PUSH"

def calculate_handicap_profit_history(row, side="HOME"):
    """
    Calcula o profit líquido histórico para uma aposta fixa em HOME ou AWAY
    usando Asian_Line_Decimal e odds líquidas Odd_H_Asi / Odd_A_Asi.
    Regras (push = 0):
      - WIN       → +odd
      - HALF_WIN  → +odd/2
      - PUSH      → 0
      - HALF_LOSS → -0.5
      - LOSS      → -1
    """
    result = determine_handicap_result_history(row, side=side)
    if result is None:
        return 0.0

    if side == "HOME":
        odd = row.get("Odd_H_Asi", np.nan)
    else:
        odd = row.get("Odd_A_Asi", np.nan)

    if pd.isna(odd):
        return 0.0

    if (side == "HOME" and result == "HOME_COVERED") or \
       (side == "AWAY" and result == "AWAY_COVERED"):
        return float(odd)
    elif result == "HALF_WIN":
        return float(odd) / 2.0
    elif result == "HALF_LOSS":
        return -0.5
    elif result == "PUSH":
        return 0.0
    else:
        return -1.0

# ============================================================
# 🧩 PREPARAR TARGETS EV (HOME & AWAY)
# ============================================================
def preparar_targets_ev(history: pd.DataFrame) -> pd.DataFrame:
    """
    Cria:
      - Profit_Home_EV, Profit_Away_EV
      - Target_EV_Home, Target_EV_Away (1 se profit > 0, senão 0)
    """
    history = history.copy()
    required_cols = ["Goals_H_FT", "Goals_A_FT", "Asian_Line_Decimal", "Odd_H_Asi", "Odd_A_Asi"]
    missing = [c for c in required_cols if c not in history.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para EV: {missing}")
        return history

    history["Profit_Home_EV"] = history.apply(
        lambda r: calculate_handicap_profit_history(r, side="HOME"), axis=1
    )
    history["Profit_Away_EV"] = history.apply(
        lambda r: calculate_handicap_profit_history(r, side="AWAY"), axis=1
    )

    history["Target_EV_Home"] = (history["Profit_Home_EV"] > 0).astype(int)
    history["Target_EV_Away"] = (history["Profit_Away_EV"] > 0).astype(int)

    st.info(
        f"🎯 Targets EV criados: {history['Target_EV_Home'].sum()} jogos lucrativos Home, "
        f"{history['Target_EV_Away'].sum()} jogos lucrativos Away."
    )
    return history

# ============================================================
# 🧮 3D – DISTÂNCIAS / ÂNGULOS / FEATURES
# ============================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula distância 3D e ângulos usando Aggression, Momentum (liga) e Momentum (time)
    Versão neutra + features compostas (sin/cos combinados e sinal vetorial).
    """
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]

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

# ============================================================
# 🧩 CLUSTERIZAÇÃO 3D – GLOBAL E POR LIGA
# ============================================================
def aplicar_clusterizacao_3d(df, n_clusters=2, random_state=42):
    """
    Cria clusters espaciais com base em Aggression, Momentum Liga e Momentum Time.
    Retorna o DataFrame com a nova coluna 'Cluster3D_Label'.
    """
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init='k-means++',
        n_init=10
    )
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
    centroids['Cluster'] = range(n_clusters)

    st.markdown("### 🧭 Clusters 3D Criados (KMeans)")
    st.dataframe(centroids.style.format({'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}'}))

    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map({
        0: '⚡ Agressivos + Momentum Positivo',
        1: '💤 Reativos + Momentum Negativo',
        2: '⚖️ Equilibrados',
        3: '🔥 Alta Variância',
        4: '🌪️ Caóticos / Transição'
    }).fillna('🌀 Outro')

    return df

def aplicar_clusterizacao_3d_por_liga(df, n_clusters=4, random_state=42):
    df = df.copy()
    required_cols = ["Aggression_Home", "Aggression_Away", "M_H", "M_A", "MT_H", "MT_A", "League"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização por liga: {missing}")
        df["Cluster3D_Label"] = 0
        df["Cluster3D_Desc"] = "⚪ Equilibrado / Neutro"
        df["C3D_ZScore"] = 0
        df["C3D_Sin"] = 0
        df["C3D_Cos"] = 1
        return df

    df["dx"] = df["Aggression_Home"] - df["Aggression_Away"]
    df["dy"] = df["M_H"] - df["M_A"]
    df["dz"] = df["MT_H"] - df["MT_A"]

    df["Cluster3D_Label"] = np.nan
    ligas_processadas = 0

    for league, subdf in df.groupby("League"):
        subdf = subdf.dropna(subset=["dx", "dy", "dz"]).copy()
        if len(subdf) < n_clusters * 2:
            df.loc[subdf.index, "Cluster3D_Label"] = 0
            continue

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            subdf["Cluster3D_Label"] = kmeans.fit_predict(subdf[["dx", "dy", "dz"]])
            df.loc[subdf.index, "Cluster3D_Label"] = subdf["Cluster3D_Label"]
            ligas_processadas += 1
        except Exception as e:
            st.warning(f"⚠️ Falha ao clusterizar {league}: {e}")
            df.loc[subdf.index, "Cluster3D_Label"] = 0

    df["Cluster3D_Label"] = df["Cluster3D_Label"].fillna(0).astype(int)

    mean_c = df["Cluster3D_Label"].mean()
    std_c = df["Cluster3D_Label"].std(ddof=0) or 1
    df["C3D_ZScore"] = (df["Cluster3D_Label"] - mean_c) / std_c
    df["C3D_Sin"] = np.sin(df["Cluster3D_Label"])
    df["C3D_Cos"] = np.cos(df["Cluster3D_Label"])

    desc_map = {
        0: "⚪ Equilibrado / Neutro",
        1: "🟢 Fav Leve / Momentum Positivo",
        2: "🔵 Agressivo / Dominante",
        3: "🔴 Underdog / Momentum Negativo"
    }
    df["Cluster3D_Desc"] = df["Cluster3D_Label"].map(desc_map).fillna("🌀 Outro")

    st.info(f"✅ Clusterização 3D por liga concluída ({ligas_processadas} ligas processadas).")
    return df

# ============================================================
# 🔁 CARREGAMENTO COM CACHE
# ============================================================
@st.cache_data(ttl=3600)
def load_cached_data(selected_file: str):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)

    history = filter_leagues(load_all_games(GAMES_FOLDER))
    if not history.empty:
        history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

    return games_today, history

# ============================================================
# 🔁 LIVE SCORE – MERGE
# ============================================================
def load_and_merge_livescore(games_today: pd.DataFrame, selected_date_str: str) -> pd.DataFrame:
    """Carrega e faz merge dos dados do Live Score."""
    games_today = setup_livescore_columns(games_today)
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    if not os.path.exists(livescore_file):
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

    st.info(f"📡 LiveScore file found: {livescore_file}")
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
    missing_cols = [col for col in required_cols if col not in results_df.columns]

    if missing_cols:
        st.error(f"❌ LiveScore file missing columns: {missing_cols}")
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

    st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
    return games_today

# ============================================================
# 🎯 DEFINIÇÃO DOS 16 QUADRANTES (AGGRESSION x HANDSCORE)
# ============================================================
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
    9:  {"nome": "Under Moderado Neutro",        "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado",      "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",         "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte",   "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},

    # 🔴 QUADRANTE 13-16: FORTE UNDERDOG (-1.0 a -0.75)
    13: {"nome": "Under Forte Neutro",      "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",    "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",       "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    """Classifica Aggression e HandScore em um dos 16 quadrantes."""
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id
    return 0

# ============================================================
# 📈 PLOT 2D DOS 16 QUADRANTES
# ============================================================
def plot_quadrantes_16(df, side="Home"):
    fig, ax = plt.subplots(figsize=(14, 10))
    cores_categorias = {
        'Fav Forte': 'lightcoral',
        'Fav Moderado': 'lightpink',
        'Under Moderado': 'lightblue',
        'Under Forte': 'lightsteelblue'
    }

    for quadrante_id in range(1, 17):
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            nome = QUADRANTES_16[quadrante_id]['nome']
            categoria = " ".join(nome.split()[:2])
            cor = cores_categorias.get(categoria, 'gray')

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

# ============================================================
# 📊 VISUALIZAÇÃO 3D – TAMANHO FIXO
# ============================================================
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
                eye=dict(x=0.0, y=0.0, z=1.2),
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

# ============================================================
# 🧠 MODELO 3D – CLUSTERS + RF (HOME COVER)
# ============================================================
def treinar_modelo_3d_clusters_single(history: pd.DataFrame, games_today: pd.DataFrame):
    """
    Modelo principal de probabilidade de HOME cobrir o handicap,
    usando features 3D + clusters.
    """
    history = history.copy()
    games_today = games_today.copy()

    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)

    USE_CLUSTER_POR_LIGA = True
    if USE_CLUSTER_POR_LIGA:
        history = aplicar_clusterizacao_3d_por_liga(history, n_clusters=4)
        games_today = aplicar_clusterizacao_3d_por_liga(games_today, n_clusters=4)
    else:
        history = aplicar_clusterizacao_3d(history, n_clusters=4)
        games_today = aplicar_clusterizacao_3d(games_today, n_clusters=4)

    # Feature eng de clusters
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

    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D'
    ]
    features_cluster = ['Cluster3D_Label', 'C3D_ZScore', 'C3D_Sin', 'C3D_Cos']

    X = pd.concat([ligas_dummies, history[features_3d + features_cluster]], axis=1).fillna(0)
    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        max_features='log2',
        n_jobs=-1
    )
    model_home.fit(X, y_home)

    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(
        columns=ligas_dummies.columns, fill_value=0
    )
    X_today = pd.concat([ligas_today, games_today[features_3d + features_cluster]], axis=1).fillna(0)

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today['ML_Confidence']

    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features (Modelo 3D com Clusters Otimizados)")
    st.dataframe(importances.head(25).to_frame("Importância"), use_container_width=True)

    cluster_stats = (
        history.groupby("Cluster3D_Label")
        .agg(
            Jogos=("Target_AH_Home", "count"),
            WinRate=("Target_AH_Home", "mean"),
            Média_Dist3D=("Quadrant_Dist_3D", "mean"),
            Média_Magnitude=("Magnitude_3D", "mean")
        )
        .reset_index()
        .sort_values("Cluster3D_Label")
    )

    st.markdown("### 🧭 Análise de Distribuição e Performance por Cluster 3D")
    st.dataframe(
        cluster_stats.style.format({
            "WinRate": "{:.1%}",
            "Média_Dist3D": "{:.2f}",
            "Média_Magnitude": "{:.2f}"
        }).background_gradient(subset=["WinRate"], cmap="RdYlGn"),
        use_container_width=True
    )

    st.success("✅ Modelo 3D treinado e clusters analisados com sucesso!")
    return model_home, games_today

# ============================================================
# 🧠 MODELO EV 3D – APRENDER ONDE HÁ VALOR (HOME & AWAY)
# ============================================================
def treinar_modelo_ev_3d(history: pd.DataFrame, games_today: pd.DataFrame):
    """
    Treina dois modelos:
      - EV_Home: probabilidade de apostar Home ser EV+
      - EV_Away: probabilidade de apostar Away ser EV+
    Usa:
      - Features 3D (distâncias, ângulos, magnitude)
      - Clusters 3D
      - Asian_Line_Decimal
      - Odds líquidas Odd_H_Asi / Odd_A_Asi
    """
    history = history.copy()
    games_today = games_today.copy()

    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)

    history = aplicar_clusterizacao_3d_por_liga(history, n_clusters=4)
    games_today = aplicar_clusterizacao_3d_por_liga(games_today, n_clusters=4)

    if "Target_EV_Home" not in history.columns or "Target_EV_Away" not in history.columns:
        st.error("❌ Target_EV_Home / Target_EV_Away não encontrados. Certifique-se que 'preparar_targets_ev(history)' foi chamado.")
        return None, None, games_today

    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D'
    ]
    features_cluster = ['Cluster3D_Label', 'C3D_ZScore', 'C3D_Sin', 'C3D_Cos']
    features_market = ['Asian_Line_Decimal', 'Odd_H_Asi', 'Odd_A_Asi']

    all_features = features_3d + features_cluster + features_market

    X_ev = pd.concat([ligas_dummies, history[all_features]], axis=1).fillna(0)
    y_ev_home = history["Target_EV_Home"].astype(int)
    y_ev_away = history["Target_EV_Away"].astype(int)

    base_params = dict(
        n_estimators=400,
        max_depth=10,
        random_state=42,
        class_weight='balanced_subsample',
        max_features='sqrt',
        n_jobs=-1
    )

    model_ev_home = RandomForestClassifier(**base_params)
    model_ev_away = RandomForestClassifier(**base_params)

    model_ev_home.fit(X_ev, y_ev_home)
    model_ev_away.fit(X_ev, y_ev_away)

    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(
        columns=ligas_dummies.columns,
        fill_value=0
    )
    X_today_ev = pd.concat([ligas_today, games_today[all_features]], axis=1).fillna(0)

    prob_value_home = model_ev_home.predict_proba(X_today_ev)[:, 1]
    prob_value_away = model_ev_away.predict_proba(X_today_ev)[:, 1]

    games_today["EV_Prob_Home"] = prob_value_home
    games_today["EV_Prob_Away"] = prob_value_away

    games_today["EV_Edge_Home"] = games_today["EV_Prob_Home"] * games_today["Odd_H_Asi"]
    games_today["EV_Edge_Away"] = games_today["EV_Prob_Away"] * games_today["Odd_A_Asi"]

    games_today["EV_Side"] = np.where(
        games_today["EV_Edge_Home"] > games_today["EV_Edge_Away"], "HOME", "AWAY"
    )
    games_today["EV_Edge_Main"] = games_today[["EV_Edge_Home", "EV_Edge_Away"]].max(axis=1)

    st.markdown("### 🔎 Top Features do Modelo EV (Home)")
    importances_ev = pd.Series(model_ev_home.feature_importances_, index=X_ev.columns).sort_values(ascending=False)
    st.dataframe(importances_ev.head(20).to_frame("Importância"), use_container_width=True)

    st.success("✅ Modelo EV 3D treinado com sucesso (Home & Away).")
    return model_ev_home, model_ev_away, games_today

# ============================================================
# 🔥 FUNÇÕES DE RECOMENDAÇÃO / SCORING / LIVE (3D + AH + 1X2)
# ============================================================
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

def gerar_score_combinado_3D_16(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------- Live 1x2 ----------
def determine_match_result_1x2(row):
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

def update_real_time_data_1x2(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------- Live Handicap 3D ----------
def determine_handicap_result_3d(row):
    """
    Determina o resultado do handicap asiático com base no lado recomendado.
    Suporta half-win / half-loss.
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

    recomendacao_str = recomendacao
    is_home_bet = any(k in recomendacao_str for k in [
        'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
        'MODELO CONFIA HOME', 'H:', 'HOME)'
    ])
    is_away_bet = any(k in recomendacao_str for k in [
        'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
        'MODELO CONFIA AWAY', 'A:', 'AWAY)'
    ])

    if not is_home_bet and not is_away_bet:
        return None

    side = "HOME" if is_home_bet else "AWAY"

    frac = abs(asian_line % 1)
    is_quarter = frac in [0.25, 0.75]

    def single_result(gh, ga, line, side):
        if side == "HOME":
            adjusted = (gh + line) - ga
        else:
            adjusted = (ga - line) - gh
        if adjusted > 0:
            return 1.0
        elif adjusted == 0:
            return 0.5
        else:
            return 0.0

    if is_quarter:
        if asian_line > 0:
            line1 = math.floor(asian_line * 2) / 2
            line2 = line1 + 0.5
        else:
            line1 = math.ceil(asian_line * 2) / 2
            line2 = line1 - 0.5

        r1 = single_result(gh, ga, line1, side)
        r2 = single_result(gh, ga, line2, side)
        avg = (r1 + r2) / 2

        if avg == 1:
            return f"{side}_COVERED"
        elif avg == 0.75:
            return "HALF_WIN"
        elif avg == 0.5:
            return "PUSH"
        elif avg == 0.25:
            return "HALF_LOSS"
        else:
            return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"

    if side == "HOME":
        adjusted = (gh + asian_line) - ga
    else:
        adjusted = (ga - asian_line) - gh

    if adjusted > 0:
        return f"{side}_COVERED"
    elif adjusted < 0:
        return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"
    else:
        return "PUSH"

def check_handicap_recommendation_correct_3d(recomendacao, handicap_result):
    if pd.isna(recomendacao) or handicap_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
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

    if is_home_bet and handicap_result in ["HOME_COVERED", "HALF_WIN"]:
        return True
    elif is_away_bet and handicap_result in ["AWAY_COVERED", "HALF_WIN"]:
        return True
    elif handicap_result == "PUSH":
        return None
    else:
        return False

def calculate_handicap_profit_3d(recomendacao, handicap_result, odds_row):
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

    if (is_home_bet and handicap_result == "HOME_COVERED") or \
       (is_away_bet and handicap_result == "AWAY_COVERED"):
        return odd
    elif handicap_result == "HALF_WIN":
        return odd / 2
    elif handicap_result == "HALF_LOSS":
        return -0.5
    elif handicap_result == "PUSH":
        return 0
    else:
        return -1

def update_real_time_data_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Handicap_Result'] = df.apply(determine_handicap_result_3d, axis=1)
    df['Quadrante_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct_3d(
            r['Recomendacao'], r['Handicap_Result']
        ), axis=1
    )
    df['Profit_Quadrante'] = df.apply(
        lambda r: calculate_handicap_profit_3d(
            r['Recomendacao'], r['Handicap_Result'], r
        ), axis=1
    )
    return df

def generate_live_summary_3d(df: pd.DataFrame):
    finished_games = df[df['Handicap_Result'].notna()]
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

def calcular_confiabilidade_por_liga_cluster(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    possible_league_cols = ["League", "Leagues", "Liga", "League_Name"]
    league_col = next((c for c in possible_league_cols if c in df.columns), None)
    if league_col is None:
        st.error("❌ Nenhuma coluna de liga encontrada no DataFrame.")
        return pd.DataFrame(columns=["League", "Liga_Cluster_Dom", "Liga_Cluster_WinRate", "Liga_Confiabilidade_Label"])
    df.rename(columns={league_col: "League"}, inplace=True)

    if "Cluster3D_Label" not in df.columns:
        st.warning("⚠️ Coluna 'Cluster3D_Label' não encontrada — atribuindo cluster neutro (0).")
        df["Cluster3D_Label"] = 0

    if "Target_AH_Home" not in df.columns:
        st.warning("⚠️ Coluna 'Target_AH_Home' não encontrada — criando temporariamente com zeros.")
        df["Target_AH_Home"] = 0

    liga_cluster_stats = (
        df.groupby(["League", "Cluster3D_Label"], dropna=False)
          .agg(
              Jogos=("Target_AH_Home", "count"),
              WinRate=("Target_AH_Home", "mean")
          )
          .reset_index()
    )

    if liga_cluster_stats.empty:
        st.warning("⚠️ Nenhum dado suficiente para calcular confiabilidade por liga.")
        return pd.DataFrame(columns=["League", "Liga_Cluster_Dom", "Liga_Cluster_WinRate", "Liga_Confiabilidade_Label"])

    liga_dominante = liga_cluster_stats.loc[
        liga_cluster_stats.groupby("League")["WinRate"].idxmax()
    ].reset_index(drop=True)

    def rotular_confiabilidade(wr):
        if wr >= 0.63:
            return "🟢 Confiável"
        elif wr >= 0.56:
            return "🟡 Moderada"
        else:
            return "🔴 Instável"

    liga_dominante["Liga_Confiabilidade_Label"] = liga_dominante["WinRate"].apply(rotular_confiabilidade)
    liga_dominante.rename(columns={
        "Cluster3D_Label": "Liga_Cluster_Dom",
        "WinRate": "Liga_Cluster_WinRate"
    }, inplace=True)

    liga_dominante["Liga_Cluster_WinRate"] = liga_dominante["Liga_Cluster_WinRate"].fillna(0.58)
    liga_dominante["Liga_Confiabilidade_Label"] = liga_dominante["Liga_Confiabilidade_Label"].fillna("🔴 Liga Nova")

    return liga_dominante[["League", "Liga_Cluster_Dom", "Liga_Cluster_WinRate", "Liga_Confiabilidade_Label"]]

def analisar_performance_por_tipo_recomendacao(df: pd.DataFrame):
    if df.empty or 'Recomendacao' not in df.columns:
        st.info("⚠️ Nenhuma recomendação disponível para análise.")
        return pd.DataFrame()

    df_valid = df[df['Quadrante_Correct'].notna()].copy()
    if df_valid.empty:
        st.info("⚠️ Nenhuma aposta finalizada ainda para análise de performance.")
        return pd.DataFrame()

    df_valid['Tipo_Recomendacao'] = (
        df_valid['Recomendacao']
        .str.extract(r'([📈🔻💪🎯⚖️].*?)\s*\(')[0]
        .str.strip()
    )

    resumo = (
        df_valid.groupby('Tipo_Recomendacao', dropna=False)
        .agg(
            Apostas=('Quadrante_Correct', 'count'),
            Acertos=('Quadrante_Correct', 'sum'),
            Winrate_Médio=('Quadrante_Correct', 'mean'),
            ROI_Médio=('Profit_Quadrante', lambda x: x.sum() / len(x) if len(x) > 0 else 0),
            Lucro_Total=('Profit_Quadrante', 'sum')
        )
        .reset_index()
        .sort_values('ROI_Médio', ascending=False)
    )

    st.dataframe(
        resumo.style.format({
            'Winrate_Médio': '{:.1%}',
            'ROI_Médio': '{:.1%}',
            'Lucro_Total': '{:.2f}',
            'Apostas': '{:.0f}',
            'Acertos': '{:.0f}'
        })
        .background_gradient(subset=['Winrate_Médio', 'ROI_Médio'], cmap='RdYlGn'),
        use_container_width=True
    )

    return resumo

# ============================================================
# ======================  MAIN FLOW  =========================
# ============================================================
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

if not os.path.exists(GAMES_FOLDER):
    st.error(f"Pasta '{GAMES_FOLDER}' não encontrada.")
    st.stop()

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

games_today, history = load_cached_data(selected_file)
games_today = load_and_merge_livescore(games_today, selected_date_str)

# CONVERSÃO ASIAN LINE
if not history.empty:
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
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

    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(
        lambda r: 1 if r["Margin"] > r["Asian_Line_Decimal"] else 0, axis=1
    )
    history = preparar_targets_ev(history)

# Asian line nos jogos de hoje
if 'Asian_Line' in games_today.columns:
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
else:
    games_today['Asian_Line_Decimal'] = np.nan

# CLASSIFICAÇÃO DOS QUADRANTES
games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)
if not history.empty:
    history['Quadrante_Home'] = history.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    history['Quadrante_Away'] = history.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )

# 3D FEATURES NOS JOGOS DO DIA
games_today = calcular_distancias_3d(games_today)

# VISUALIZAÇÃO 2D
st.markdown("### 📈 Visualização dos 16 Quadrantes (2D)")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

# FILTROS PARA 3D
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
    missing_cols = [c for c in required_cols if c not in df_filtrado.columns]
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

    mask_total = mask_xy & mask_xz & mask_mag
    df_filtrado = df_filtrado[mask_total].copy()
    df_filtrado['Angulo_XY'] = angulo_xy[mask_total]
    df_filtrado['Angulo_XZ'] = angulo_xz[mask_total]
    df_filtrado['Magnitude_3D_Filtro'] = magnitude[mask_total]
    return df_filtrado

df_plot = df_filtered.copy()
if aplicar_filtro:
    df_plot = filtrar_por_angulo(df_plot, angulo_xy_range, angulo_xz_range, magnitude_min)
    st.success(f"✅ Filtro aplicado! {len(df_plot)} jogos encontrados com os critérios angulares.")
    if not df_plot.empty:
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            st.metric("Ângulo XY Médio", f"{df_plot['Angulo_XY'].mean():.1f}°")
        with colf2:
            st.metric("Ângulo XZ Médio", f"{df_plot['Angulo_XZ'].mean():.1f}°")
        with colf3:
            st.metric("Magnitude Média", f"{df_plot['Magnitude_3D_Filtro'].mean():.2f}")
else:
    df_plot = df_plot.nlargest(n_to_show, "Quadrant_Dist_3D")

df_plot = df_plot.reset_index(drop=True)
fig_3d_fixed = create_fixed_3d_plot(df_plot, n_to_show, selected_league if 'selected_league' in locals() else "⚽ Todas as ligas")
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

# ============================================================
# 🔥 TREINO DOS MODELOS (3D + EV)
# ============================================================
if not history.empty:
    modelo_home, games_today = treinar_modelo_3d_clusters_single(history, games_today)
    model_ev_home, model_ev_away, games_today = treinar_modelo_ev_3d(history, games_today)
    st.success("✅ Modelo 3D dual com 16 quadrantes + EV treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D/EV")

# ============================================================
# 🏆 RANKING 3D / LIVE / ESTRATÉGIAS
# ============================================================
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_3d = games_today.copy()
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
    ranking_3d = gerar_score_combinado_3D_16(ranking_3d)

    # Live Handicap
    ranking_3d = update_real_time_data_3d(ranking_3d)

    # Confiabilidade por liga (cluster)
    try:
        liga_conf_cluster = calcular_confiabilidade_por_liga_cluster(history)
        if "League" in ranking_3d.columns:
            ranking_3d = ranking_3d.merge(liga_conf_cluster, on="League", how="left")
            st.success("✅ Colunas de confiabilidade 3D por liga adicionadas à tabela principal.")
        else:
            st.warning("⚠️ Coluna 'League' não encontrada em ranking_3d.")
    except Exception as e:
        st.error(f"Erro ao calcular confiabilidade 3D por liga: {e}")

    # Live summary 3D
    st.markdown("## 📡 Live Score Monitor - Sistema 3D")
    live_summary_3d = generate_live_summary_3d(ranking_3d)
    st.json(live_summary_3d)

    # Live 1x2
    st.markdown("## 📡 Live Score Monitor - Sistema 3D (1x2)")
    ranking_3d = update_real_time_data_1x2(ranking_3d)
    finished_1x2 = ranking_3d[ranking_3d['Result_1x2'].notna()]
    if not finished_1x2.empty:
        total_bets = finished_1x2['Quadrante_Correct_1x2'].notna().sum()
        correct_bets = finished_1x2['Quadrante_Correct_1x2'].sum()
        total_profit = finished_1x2['Profit_1x2'].sum()
        winrate = correct_bets / total_bets if total_bets > 0 else 0
        roi = total_profit / total_bets if total_bets > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Apostas (1x2)", total_bets)
        with c2:
            st.metric("Winrate (1x2)", f"{winrate:.1%}")
        with c3:
            st.metric("Lucro Total (1x2)", f"{total_profit:.2f}u")
        with c4:
            st.metric("ROI (1x2)", f"{roi:.1%}")
    else:
        st.info("⚠️ Nenhum jogo finalizado ainda para o sistema 1x2.")

    # Comparativo AH vs 1x2 (texto/tabela simplificada)
    def compare_systems_summary(df):
        def calc(prefix):
            if prefix == "Quadrante":
                correct_col = "Quadrante_Correct"
                profit_col = "Profit_Quadrante"
            else:
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
        # Estilo apenas em colunas numéricas
        resumo_styled = resumo.copy()
        numeric_cols = resumo.select_dtypes(include=['float64', 'int64']).columns
        
        resumo_styled = resumo.style.apply(
            lambda row: [
                'background-color: lightgreen' if col in numeric_cols and val == row[numeric_cols].max()
                else 'background-color: #ffb3b3' if col in numeric_cols and val == row[numeric_cols].min()
                else ''
                for col, val in row.iteritems()
            ],
            axis=1
        )
        
        st.dataframe(resumo_styled, use_container_width=True)


    # compare_systems_summary(ranking_3d)

    ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

    colunas_3d = [
        'League', "Liga_Confiabilidade_Label", 'Time',
        'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'Recomendacao', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
        'Score_Final_3D', 'Classificacao_Potencial_3D',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        'Quadrant_Dist_3D', 'Momentum_Diff',
        'Asian_Line_Decimal', 'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante',
        'EV_Side', 'EV_Edge_Home', 'EV_Edge_Away', 'EV_Edge_Main',
        'EV_Prob_Home', 'EV_Prob_Away'
    ]
    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    def estilo_tabela_3d_quadrantes(df):
        def cor_classificacao_3d(valor):
            s = str(valor)
            if '🌟 ALTO POTENCIAL 3D' in s: return 'font-weight: bold'
            if '💼 VALOR SOLIDO 3D' in s: return 'font-weight: bold'
            if '🔴 BAIXO POTENCIAL 3D' in s: return 'font-weight: bold'
            if '🏆 ALTO VALOR' in s: return 'font-weight: bold'
            if '🔴 ALTO RISCO' in s: return 'font-weight: bold'
            if 'VALUE' in s: return 'font-weight: bold'
            if 'EVITAR' in s: return 'font-weight: bold'
            return ''

        colunas_para_estilo = [c for c in [
            'Classificacao_Potencial_3D', 'Classificacao_Valor_Home',
            'Classificacao_Valor_Away', 'Recomendacao'
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
        if 'EV_Edge_Main' in df.columns:
            styler = styler.background_gradient(subset=['EV_Edge_Main'], cmap='RdYlGn')

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
            'Quadrant_Dist_3D': '{:.2f}',
            'Momentum_Diff': '{:.2f}',
            'EV_Edge_Home': '{:.2f}',
            'EV_Edge_Away': '{:.2f}',
            'EV_Edge_Main': '{:.2f}',
            'EV_Prob_Home': '{:.1%}',
            'EV_Prob_Away': '{:.1%}'
        }, na_rep="-"),
        use_container_width=True
    )

    st.markdown("## 📊 Performance por Tipo de Recomendação (3D – Agrupada)")
    try:
        perf_recomendacoes_agrupadas = analisar_performance_por_tipo_recomendacao(ranking_3d)
    except Exception as e:
        st.error(f"Erro ao calcular performance agrupada: {e}")

else:
    st.info("⚠️ Aguardando dados para gerar ranking 3D de 16 quadrantes")

st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML + EV** carregado e pronto para uso.")
st.info("""
**Resumo das funcionalidades:**
- 🔢 16 quadrantes com análise 3D completa (Aggression x HandScore x Momentum)
- 📊 Momentum integrado como terceira dimensão
- 🎯 Distâncias e ângulos 3D calculados
- 📈 Visualizações 2D e 3D interativas
- 🧠 Modelo 3D (Home Cover) com clusters por liga
- 💰 Modelo EV 3D (Home & Away) usando odds asiáticas líquidas
- 📡 Monitor Live para Handicap Asiático e 1x2
- ⚖️ Comparativo AH x 1x2 e análise por tipo de recomendação
""")
