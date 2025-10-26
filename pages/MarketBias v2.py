# =====================================================
# 🎯 SISTEMA 3D COM CLUSTERS – Market Bias Edition
# =====================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from datetime import datetime
import math
import sys

# =====================================================
# 🧼 HOTFIXES E CONFIGURAÇÕES INICIAIS
# =====================================================
if hasattr(st, 'cache_data'):
    st.cache_data.clear()
if hasattr(st, 'cache_resource'):
    st.cache_resource.clear()

module_suffix = '_page'
for module_name in list(sys.modules.keys()):
    if module_name.endswith(module_suffix):
        del sys.modules[module_name]

st.set_page_config(page_title="Sistema 3D Clusters - Market Bias", layout="wide")
st.title("🎯 Sistema 3D com Clusters – Market Bias Edition")

PAGE_PREFIX = "Clusters3D_ML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# =====================================================
# 🧩 FUNÇÕES BÁSICAS
# =====================================================
def preprocess_df(df):
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

# =====================================================
# ⚙️ SETUP DE LIVESCORE
# =====================================================
def setup_livescore_columns(df):
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

# =====================================================
# 🧮 CONVERSÕES ASIAN LINE
# =====================================================
def convert_asian_line_to_decimal(line_str):
    if pd.isna(line_str) or line_str == "":
        return None
    try:
        line_str = str(line_str).strip()
        if "/" not in line_str:
            return float(line_str)
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
    except (ValueError, TypeError):
        return None

def calc_handicap_result(margin, asian_line_str, invert=False):
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

# =====================================================
# 🎯 CLUSTERIZAÇÃO 3D (KMEANS)
# =====================================================
def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    df = df.copy()
    req_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    if any(c not in df.columns for c in req_cols):
        st.warning("⚠️ Colunas ausentes para clusterização 3D.")
        df['Cluster3D_Label'] = -1
        df['Cluster3D_Desc'] = '🌀 Dados Insuficientes'
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    mapping = {
        0: '🏠 Home Domina Confronto',
        1: '🚗 Away Domina Confronto',
        2: '⚖️ Confronto Equilibrado',
        3: '🎭 Home Imprevisível',
        4: '🌪️ Home Instável'
    }
    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(mapping).fillna('🌀 Caso Atípico')
    return df

# =====================================================
# ⚙️ CÁLCULOS DE MOMENTUM E REGRESSÃO
# =====================================================
def calcular_momentum_time(df, window=6):
    df = df.copy()
    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())
    for team in all_teams:
        mask_home = df['Home'] == team
        mask_away = df['Away'] == team

        if mask_home.sum() > 2:
            s = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = z

        if mask_away.sum() > 2:
            s = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = z

    return df.fillna(0)

def calcular_regressao_media(df):
    df = df.copy()
    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])
    df['Regressao_Force_Home'] = -np.sign(df['M_H']) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df['M_A']) * (df['Extremidade_Away'] ** 0.7)
    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))
    return df

# =====================================================
# 💡 MARKET OPENING BIAS
# =====================================================
def calcular_market_opening_bias(df):
    df = df.copy()
    open_cols = ['Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP']

    if not all(col in df.columns for col in open_cols):
        st.warning("⚠️ Odds de abertura não encontradas — usando odds de fechamento.")
        df['Odd_H_OP'] = df.get('Odd_H')
        df['Odd_D_OP'] = df.get('Odd_D')
        df['Odd_A_OP'] = df.get('Odd_A')

    for prefix in ['H', 'D', 'A']:
        df[f'Impl_{prefix}_Open'] = 1 / df[f'Odd_{prefix}_OP']
        df[f'Impl_{prefix}_Close'] = 1 / df[f'Odd_{prefix}']

    df['Impl_Sum_Open'] = df[['Impl_H_Open', 'Impl_D_Open', 'Impl_A_Open']].sum(axis=1)
    df['Impl_Sum_Close'] = df[['Impl_H_Close', 'Impl_D_Close', 'Impl_A_Close']].sum(axis=1)
    for prefix in ['H', 'D', 'A']:
        df[f'Impl_{prefix}_Open'] /= df['Impl_Sum_Open']
        df[f'Impl_{prefix}_Close'] /= df['Impl_Sum_Close']

    if 'Prob_Home' in df.columns:
        df['Market_Error_Open_H'] = df['Prob_Home'] - df['Impl_H_Open']
        df['Market_Error_Open_A'] = (1 - df['Prob_Home']) - df['Impl_A_Open']
    else:
        df['Market_Error_Open_H'] = np.nan
        df['Market_Error_Open_A'] = np.nan

    df['Bias_Open_H'] = df['Market_Error_Open_H'] * 100
    df['Bias_Open_A'] = df['Market_Error_Open_A'] * 100
    return df

# =====================================================
# 📈 TREINAMENTO + APLICAÇÃO
# =====================================================
st.info("📊 Calculando Momentum, Regressão e Viés da Bookie...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("Nenhum arquivo CSV encontrado em GamesDay.")
    st.stop()

selected_file = st.selectbox("Selecione o arquivo de jogos:", files, index=len(files) - 1)
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# Pipeline
history = calcular_momentum_time(history)
games_today = calcular_momentum_time(games_today)
history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)

# Dummy model just to generate Prob_Home
games_today['Prob_Home'] = np.clip(np.random.normal(0.5, 0.1, len(games_today)), 0, 1)
games_today['Prob_Away'] = 1 - games_today['Prob_Home']

# Bias
games_today = calcular_market_opening_bias(games_today)
st.success("✅ Market Opening Bias calculado com sucesso!")

# =====================================================
# 🧭 ANÁLISE DE VIÉS DA BOOKIE
# =====================================================
st.markdown("## 🧭 Análise de Viés da Bookie – Odds de Abertura")
mean_bias_home = games_today['Bias_Open_H'].mean()
mean_bias_away = games_today['Bias_Open_A'].mean()
st.metric("📊 Média de Viés (Home)", f"{mean_bias_home:+.2f}%")
st.metric("📊 Média de Viés (Away)", f"{mean_bias_away:+.2f}%")

if 'League' in games_today.columns:
    league_bias = (
        games_today.groupby('League')[['Bias_Open_H', 'Bias_Open_A']]
        .mean().sort_values('Bias_Open_H', ascending=False)
    )
    st.markdown("### ⚽ Viés Médio por Liga")
    st.dataframe(league_bias.style.format('{:+.2f}'), use_container_width=True)

top_bias = games_today[['League', 'Home', 'Away', 'Bias_Open_H', 'Bias_Open_A', 'Prob_Home', 'Impl_H_Open']].copy()
top_bias['Abs_Bias'] = top_bias['Bias_Open_H'].abs()
top5 = top_bias.nlargest(5, 'Abs_Bias')
st.markdown("### 🔍 Top 5 Divergências (Bookie vs Modelo)")
st.dataframe(
    top5.style.format({
        'Bias_Open_H': '{:+.2f}',
        'Prob_Home': '{:.2%}',
        'Impl_H_Open': '{:.2%}'
    }),
    use_container_width=True
)
