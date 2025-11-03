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
import plotly.graph_objects as go
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
    """Converte handicaps asiáticos para decimal"""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if "/" not in value:
        try:
            num = float(value)
            return -num
        except ValueError:
            return np.nan

    try:
        parts = [float(p) for p in value.split("/")]
        avg = np.mean(parts)
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result
    except ValueError:
        return np.nan

def calculate_ah_home_target(margin, asian_line_str):
    """Calcula target AH Home"""
    line_home = convert_asian_line_to_decimal(asian_line_str)
    if pd.isna(line_home) or pd.isna(margin):
        return np.nan
    return 1 if margin > line_home else 0

# ---------------- CLUSTERIZAÇÃO 3D ----------------
def aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5):
    """Clusterização temporalmente segura"""
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    
    history_clean = history[required_cols].fillna(0)
    games_today_clean = games_today[required_cols].fillna(0)
    
    history_clean['dx'] = history_clean['Aggression_Home'] - history_clean['Aggression_Away']
    history_clean['dy'] = history_clean['M_H'] - history_clean['M_A']
    history_clean['dz'] = history_clean['MT_H'] - history_clean['MT_A']
    
    games_today_clean['dx'] = games_today_clean['Aggression_Home'] - games_today_clean['Aggression_Away']
    games_today_clean['dy'] = games_today_clean['M_H'] - games_today_clean['M_A']
    games_today_clean['dz'] = games_today_clean['MT_H'] - games_today_clean['MT_A']
    
    X_train = history_clean[['dx', 'dy', 'dz']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=10)
    kmeans.fit(X_train)
    
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
def load_cached_data(selected_file):
    """Cache apenas dos dados pesados"""
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
    
    return games_today, history

games_today, history = load_cached_data(selected_file)

# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today, selected_date_str):
    """Carrega e faz merge dos dados do Live Score"""
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

        required_cols = ['Id', 'status', 'home_goal', 'away_goal', 'home_red', 'away_red']
        missing_cols = [col for col in required_cols if col not in results_df.columns]

        if missing_cols:
            st.error(f"❌ LiveScore file missing columns: {missing_cols}")
            return games_today
        else:
            games_today = games_today.merge(results_df, left_on='Id', right_on='Id', how='left', suffixes=('', '_RAW'))
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

games_today = load_and_merge_livescore(games_today, selected_date_str)

# ---------------- PROCESSAMENTO DOS DADOS ----------------
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
history = history.dropna(subset=['Asian_Line_Decimal'])

if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Target_AH_Home"] = history.apply(lambda r: calculate_ah_home_target(r["Margin"], r["Asian_Line"]), axis=1)

# ---------------- SISTEMA 3D DE 16 QUADRANTES ----------------
st.markdown("## 🎯 Sistema 3D de 16 Quadrantes")

QUADRANTES_16 = {
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro", "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},
    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro", "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},
    9: {"nome": "Under Moderado Neutro", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},
    13: {"nome": "Under Forte Neutro", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
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

games_today['Quadrante_Home'] = games_today.apply(lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1)
games_today['Quadrante_Away'] = games_today.apply(lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1)
history['Quadrante_Home'] = history.apply(lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1)
history['Quadrante_Away'] = history.apply(lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1)

# ---------------- CÁLCULO MOMENTUM TIME ----------------
def calcular_momentum_time(df, window=6):
    """Calcula o Momentum do Time (MT_H / MT_A)"""
    df = df.copy()
    
    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = zscore

        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = zscore

    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)
    return df

history = calcular_momentum_time(history)
games_today = calcular_momentum_time(games_today)

# ---------------- CÁLCULO DISTÂNCIAS 3D ----------------
def calcular_distancias_3d(df):
    """Calcula distância 3D e ângulos"""
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Quadrant_Angle_XY', 'Quadrant_Angle_XZ', 
                   'Quadrant_Angle_YZ', 'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Sin_XZ', 
                   'Quadrant_Cos_XZ', 'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ', 'Quadrant_Sin_Combo', 
                   'Quadrant_Cos_Combo', 'Vector_Sign', 'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D']:
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

# ---------------- VISUALIZAÇÃO 2D ----------------
def plot_quadrantes_16(df, side="Home"):
    """Plot dos 16 quadrantes"""
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
            categoria = QUADRANTES_16[quadrante_id]['nome'].split()[0] + ' ' + QUADRANTES_16[quadrante_id]['nome'].split()[1]
            cor = cores_categorias.get(categoria, 'gray')
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(x, y, c=cor, label=QUADRANTES_16[quadrante_id]['nome'], alpha=0.7, s=50)

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

# ---------------- VISUALIZAÇÃO 3D INTERATIVA ----------------
st.markdown("## 🎯 Visualização Interativa 3D")

if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())
    selected_leagues = st.multiselect("Selecione ligas:", options=leagues, default=[])
    df_filtered = games_today[games_today["League"].isin(selected_leagues)].copy() if selected_leagues else games_today.copy()
else:
    df_filtered = games_today.copy()

max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir:", 10, min(max_n, 200), 40, step=5)
df_plot = df_filtered.nlargest(n_to_show, "Quadrant_Dist_3D").reset_index(drop=True)

def create_3d_plot(df_plot, n_to_show, selected_league):
    """Cria gráfico 3D interativo"""
    fig_3d = go.Figure()

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
            x=[xh, xa], y=[yh, ya], z=[zh, za],
            mode='lines+markers',
            line=dict(color='gray', width=4),
            marker=dict(size=5),
            hoverinfo='text',
            hovertext=f"<b>{row.get('Home','N/A')} vs {row.get('Away','N/A')}</b><br>"
                     f"🏆 {row.get('League','N/A')}<br>"
                     f"📍 Agg_H: {xh:.2f} | Agg_A: {xa:.2f}",
            showlegend=False
        ))

    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Home"], y=df_plot["M_H"], z=df_plot["MT_H"],
        mode='markers+text', name='Home',
        marker=dict(color='royalblue', size=10, opacity=0.9, symbol='circle', line=dict(color='darkblue', width=2)),
        text=df_plot["Home"], textposition="top center", hoverinfo='skip'
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=df_plot["Aggression_Away"], y=df_plot["M_A"], z=df_plot["MT_A"],
        mode='markers+text', name='Away',
        marker=dict(color='orangered', size=10, opacity=0.9, symbol='diamond', line=dict(color='darkred', width=2)),
        text=df_plot["Away"], textposition="top center", hoverinfo='skip'
    ))

    fig_3d.update_layout(
        title=dict(text=f"Top {n_to_show} Distâncias 3D", x=0.5, font=dict(size=16, color='white')),
        scene=dict(
            xaxis=dict(title='Aggression', range=[-1.2, 1.2]),
            yaxis=dict(title='Momentum Liga', range=[-4.0, 4.0]),
            zaxis=dict(title='Momentum Time', range=[-4.0, 4.0]),
            aspectmode="cube",
            camera=dict(eye=dict(x=0.0, y=0.0, z=1.2), up=dict(x=0.3, y=0, z=1), center=dict(x=0, y=0, z=0))
        ),
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig_3d

selected_league_label = ", ".join(selected_leagues) if selected_leagues else "⚽ Todas as ligas"
fig_3d = create_3d_plot(df_plot, n_to_show, selected_league_label)
st.plotly_chart(fig_3d, use_container_width=True)

# ---------------- MODELO ML 3D COMPLETO ----------------
def treinar_modelo_3d_clusters_single(history, games_today):
    """Treina o modelo 3D completo"""
    st.markdown("### ⚙️ Configuração do Treino 3D")
    
    # Aplicar clusterização
    history, games_today = aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5)
    
    # Feature Engineering
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    features_3d = ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
                  'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ', 'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
                  'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo', 'Vector_Sign', 'Magnitude_3D']

    extras_3d = history[features_3d].fillna(0)
    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)
    y_home = history['Target_AH_Home'].astype(int)

    # Modelo
    model_home = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, 
                                      class_weight='balanced_subsample', n_jobs=-1)
    model_home.fit(X, y_home)

    # Previsões
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
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today['ML_Confidence']

    # Avaliação
    accuracy = model_home.score(X, y_home)
    st.metric("Accuracy (Treino)", f"{accuracy:.2%}")
    
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_feats = importances.head(25).to_frame("Importância")
    st.markdown("### 🔍 Top Features")
    st.dataframe(top_feats, use_container_width=True)

    st.success("✅ Modelo 3D treinado com sucesso!")
    return model_home, games_today

# Executar treinamento
if not history.empty:
    modelo_home, games_today = treinar_modelo_3d_clusters_single(history, games_today)
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo 3D")

# ---------------- SISTEMA DE INDICAÇÕES 3D ----------------
def adicionar_indicadores_explicativos_3d_16_dual(df):
    """Adiciona classificações e recomendações"""
    df = df.copy()

    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))

    # Classificação de Valor para HOME
    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    # Classificação de Valor para AWAY
    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

    # Recomendações
    def gerar_recomendacao(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']
        
        if 'Fav Forte' in home_q and 'Under Forte' in away_q:
            return f'💪 FAVORITO HOME FORTE ({score_home:.1%})'
        elif 'Under Forte' in home_q and 'Fav Forte' in away_q:
            return f'💪 FAVORITO AWAY FORTE ({score_away:.1%})'
        elif ml_side == 'HOME' and score_home >= 0.60:
            return f'📈 MODELO CONFIA HOME ({score_home:.1%})'
        elif ml_side == 'AWAY' and score_away >= 0.60:
            return f'📈 MODELO CONFIA AWAY ({score_away:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao, axis=1)
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- SCORING 3D ----------------
def calcular_pontuacao_3d_quadrante_16(quadrante_id, momentum=0):
    """Calcula pontuação base 3D para cada quadrante"""
    scores_base = {
        1: 85, 2: 80, 3: 75, 4: 70, 5: 70, 6: 65, 7: 60, 8: 55,
        9: 50, 10: 45, 11: 40, 12: 35, 13: 35, 14: 30, 15: 25, 16: 20
    }
    
    base_score = scores_base.get(quadrante_id, 50)
    momentum_boost = momentum * 10
    adjusted_score = base_score + momentum_boost
    return max(0, min(100, adjusted_score))

def gerar_score_combinado_3d_16(df):
    """Gera score combinado 3D"""
    df = df.copy()

    df['Score_Base_Home'] = df.apply(lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Home'], x.get('M_H', 0)), axis=1)
    df['Score_Base_Away'] = df.apply(lambda x: calcular_pontuacao_3d_quadrante_16(x['Quadrante_Away'], x.get('M_A', 0)), axis=1)

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

# ---------------- EXIBIÇÃO FINAL ----------------
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_3d = games_today.copy()
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(ranking_3d)
    ranking_3d = gerar_score_combinado_3d_16(ranking_3d)
    ranking_3d = ranking_3d.sort_values('Score_Final_3D', ascending=False)

    # Colunas para exibição
    colunas_3d = [
        'League', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'Recomendacao', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label', 'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Score_Final_3D', 'Classificacao_Potencial_3D', 'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        'M_H', 'M_A', 'MT_H', 'MT_A', 'Quadrant_Dist_3D', 'Asian_Line_Decimal'
    ]

    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    def estilo_tabela_3d_quadrantes(df):
        def cor_classificacao_3d(valor):
            if '🌟 ALTO POTENCIAL 3D' in str(valor): return 'font-weight: bold'
            elif '💼 VALOR SOLIDO 3D' in str(valor): return 'font-weight: bold'
            elif '🔴 BAIXO POTENCIAL 3D' in str(valor): return 'font-weight: bold'
            elif '🏆 ALTO VALOR' in str(valor): return 'font-weight: bold'
            elif '🔴 ALTO RISCO' in str(valor): return 'font-weight: bold'
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

    st.dataframe(
        estilo_tabela_3d_quadrantes(ranking_3d[cols_finais_3d])
        .format({
            'Goals_H_Today': '{:.0f}', 'Goals_A_Today': '{:.0f}', 'Asian_Line_Decimal': '{:.2f}',
            'Quadrante_ML_Score_Home': '{:.1%}', 'Quadrante_ML_Score_Away': '{:.1%}', 
            'Score_Final_3D': '{:.1f}', 'M_H': '{:.2f}', 'M_A': '{:.2f}', 
            'MT_H': '{:.2f}', 'MT_A': '{:.2f}', 'Quadrant_Dist_3D': '{:.2f}'
        }, na_rep="-"),
        use_container_width=True
    )

else:
    st.info("⚠️ Aguardando dados para gerar ranking 3D de 16 quadrantes")

st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML** implementado com sucesso!")
