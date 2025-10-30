############ Bloco A - Imports e Configurações Base ################
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import math
import plotly.graph_objects as go
import sys

# Configurações base
PAGE_PREFIX = "Clusters3D_ML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy", "coppa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

############ Bloco B - Funções de Helpers Base ################
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

############ Bloco C - Funções Asian Line ################
def convert_asian_line_to_decimal(value):
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

############ Bloco D - Sistema de Clusterização 3D ################
def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        df['Cluster3D_Desc'] = '🌀 Dados Insuficientes'
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    if len(X_cluster) < n_clusters:
        st.warning(f"⚠️ Dados insuficientes para {n_clusters} clusters. Ajustando...")
        n_clusters = max(2, len(X_cluster) // 2)

    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            init='k-means++',
            n_init=10
        )
        df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)
        
        centroids = kmeans.cluster_centers_
        classificacoes_clusters = classificar_clusters_dinamicamente(centroids)
        df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(classificacoes_clusters).fillna('🌀 Caso Atípico')
        
    except Exception as e:
        st.error(f"❌ Erro na clusterização: {e}")
        df['Cluster3D_Label'] = -1
        df['Cluster3D_Desc'] = '🌀 Erro na Clusterização'
    
    return df

def classificar_clusters_dinamicamente(centroids):
    classificacoes = {}
    
    THRESHOLD_ALTO = 0.3
    THRESHOLD_BAIXO = -0.3
    THRESHOLD_EQUILIBRADO = 0.15
    
    for cluster_id, centroid in enumerate(centroids):
        dx, dy, dz = centroid
        
        if dx > THRESHOLD_ALTO and dy > THRESHOLD_ALTO and dz > THRESHOLD_ALTO:
            classificacao = '🏠 Home Domina Totalmente'
        elif dx > THRESHOLD_ALTO and dy > THRESHOLD_ALTO:
            classificacao = '🏠 Home Domina (Liga Forte)'
        elif dx > THRESHOLD_ALTO and dz > THRESHOLD_ALTO:
            classificacao = '🏠 Home Domina (Time Forte)'
        elif dx > THRESHOLD_ALTO:
            classificacao = '📈 Home Agressivo'
        elif dx < THRESHOLD_BAIXO and dy < THRESHOLD_BAIXO and dz < THRESHOLD_BAIXO:
            classificacao = '🚗 Away Domina Totalmente'
        elif dx < THRESHOLD_BAIXO and dy < THRESHOLD_BAIXO:
            classificacao = '🚗 Away Domina (Liga Forte)'
        elif dx < THRESHOLD_BAIXO and dz < THRESHOLD_BAIXO:
            classificacao = '🚗 Away Domina (Time Forte)'
        elif dx < THRESHOLD_BAIXO:
            classificacao = '📉 Away Agressivo'
        elif abs(dx) <= THRESHOLD_EQUILIBRADO and abs(dy) <= 1.0 and abs(dz) <= 1.0:
            classificacao = '⚖️ Confronto Equilibrado'
        elif abs(dy) > 2.0 or abs(dz) > 2.0:
            if dx > 0:
                classificacao = '🎭 Home Imprevisível'
            else:
                classificacao = '🌪️ Away Imprevisível'
        elif (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
            classificacao = '🔄 Sinais Contraditórios'
        else:
            if dx > 0.1:
                classificacao = '📊 Home Leve Vantagem'
            elif dx < -0.1:
                classificacao = '📊 Away Leve Vantagem'
            else:
                classificacao = '⚖️ Equilíbrio Neutro'
        
        classificacoes[cluster_id] = classificacao
    
    return classificacoes

############ Bloco E - Cálculo de Momentum e Regressão ################
def calcular_momentum_time(df, window=6):
    df = df.copy()

    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    if 'HandScore_Home' not in df.columns or 'HandScore_Away' not in df.columns:
        st.warning("⚠️ Colunas HandScore não encontradas - preenchendo MT com 0")
        df['MT_H'] = 0
        df['MT_A'] = 0
        return df

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:
            try:
                series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
                if series.std(ddof=0) != 0:
                    zscore = (series - series.mean()) / series.std(ddof=0)
                else:
                    zscore = series * 0
                df.loc[mask_home, 'MT_H'] = zscore
            except Exception as e:
                df.loc[mask_home, 'MT_H'] = 0

        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            try:
                series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
                if series.std(ddof=0) != 0:
                    zscore = (series - series.mean()) / series.std(ddof=0)
                else:
                    zscore = series * 0
                df.loc[mask_away, 'MT_A'] = zscore
            except Exception as e:
                df.loc[mask_away, 'MT_A'] = 0

    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)

    return df

def calcular_regressao_media(df):
    df = df.copy()
    
    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])
    
    df['Regressao_Force_Home'] = -np.sign(df['M_H']) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df['M_A']) * (df['Extremidade_Away'] ** 0.7)
    
    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))
    
    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] + 0.4 * (1 - df['Aggression_Home']))
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] + 0.4 * (1 - df['Aggression_Away']))
    
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

############ Bloco F - Visualização 3D com Clusters ################
def create_3d_plot_with_clusters(df_plot, n_to_show, selected_league):
    if df_plot.empty:
        st.warning("📭 DataFrame vazio - não é possível criar gráfico")
        return go.Figure()
    
    fig_3d = go.Figure()

    cluster_cores = {
        '🏠 Home Domina Totalmente': 'blue',
        '🏠 Home Domina (Liga Forte)': 'darkblue',
        '🏠 Home Domina (Time Forte)': 'lightblue',
        '📈 Home Agressivo': 'cyan',
        '🚗 Away Domina Totalmente': 'red',
        '🚗 Away Domina (Liga Forte)': 'darkred', 
        '🚗 Away Domina (Time Forte)': 'lightcoral',
        '📉 Away Agressivo': 'orange',
        '⚖️ Confronto Equilibrado': 'green',
        '🎭 Home Imprevisível': 'purple',
        '🌪️ Away Imprevisível': 'magenta',
        '🔄 Sinais Contraditórios': 'yellow',
        '📊 Home Leve Vantagem': 'lightgreen',
        '📊 Away Leve Vantagem': 'lightsalmon',
        '⚖️ Equilíbrio Neutro': 'gray',
        '🌀 Caso Atípico': 'lightgray',
        '🌀 Dados Insuficientes': 'white',
        '🌀 Erro na Clusterização': 'black'
    }

    clusters_presentes = df_plot['Cluster3D_Desc'].unique()
    
    for cluster_name in clusters_presentes:
        cluster_data = df_plot[df_plot['Cluster3D_Desc'] == cluster_name]
        
        if cluster_data.empty:
            continue
            
        color = cluster_cores.get(cluster_name, 'gray')
        
        for idx, row in cluster_data.iterrows():
            try:
                xh = row.get('Aggression_Home', 0) or 0
                xa = row.get('Aggression_Away', 0) or 0
                yh = row.get('M_H', 0) or 0
                ya = row.get('M_A', 0) or 0  
                zh = row.get('MT_H', 0) or 0
                za = row.get('MT_A', 0) or 0
                
                xh = float(xh) if not pd.isna(xh) else 0.0
                xa = float(xa) if not pd.isna(xa) else 0.0
                yh = float(yh) if not pd.isna(yh) else 0.0
                ya = float(ya) if not pd.isna(ya) else 0.0
                zh = float(zh) if not pd.isna(zh) else 0.0
                za = float(za) if not pd.isna(za) else 0.0
                
                points_different = (xh != xa) or (yh != ya) or (zh != za)
                
                if points_different:
                    fig_3d.add_trace(go.Scatter3d(
                        x=[xh, xa],
                        y=[yh, ya], 
                        z=[zh, za],
                        mode='lines',
                        line=dict(color=color, width=4),
                        name=f'Linha {cluster_name}',
                        showlegend=False,
                        hoverinfo='text',
                        text=f"{row.get('Home', 'Home')} vs {row.get('Away', 'Away')}<br>{cluster_name}",
                        hovertemplate='<b>%{text}</b><extra></extra>'
                    ))
                    
            except Exception as e:
                continue

    for cluster_name in clusters_presentes:
        cluster_data = df_plot[df_plot['Cluster3D_Desc'] == cluster_name]
        
        if cluster_data.empty:
            continue
            
        color = cluster_cores.get(cluster_name, 'gray')
        
        home_valid = cluster_data[
            cluster_data[['Aggression_Home', 'M_H', 'MT_H']].notna().all(axis=1)
        ]
        
        away_valid = cluster_data[
            cluster_data[['Aggression_Away', 'M_A', 'MT_A']].notna().all(axis=1)
        ]

        if not home_valid.empty:
            fig_3d.add_trace(go.Scatter3d(
                x=home_valid['Aggression_Home'],
                y=home_valid['M_H'],
                z=home_valid['MT_H'],
                mode='markers',
                name=f'{cluster_name} - Home',
                marker=dict(
                    color=color,
                    size=10,
                    symbol='circle',
                    opacity=0.9,
                    line=dict(color='white', width=2)
                ),
                text=home_valid.apply(
                    lambda r: f"<b>{r.get('Home', 'Home')}</b><br>"
                             f"Cluster: {cluster_name}<br>"
                             f"vs {r.get('Away', 'Away')}<br>"
                             f"Agg: {r.get('Aggression_Home', 0):.2f}<br>"
                             f"M_Liga: {r.get('M_H', 0):.2f}<br>"
                             f"M_Time: {r.get('MT_H', 0):.2f}", 
                    axis=1
                ),
                hovertemplate='%{text}<extra></extra>'
            ))

        if not away_valid.empty:
            fig_3d.add_trace(go.Scatter3d(
                x=away_valid['Aggression_Away'],
                y=away_valid['M_A'],
                z=away_valid['MT_A'],
                mode='markers',
                name=f'{cluster_name} - Away',
                marker=dict(
                    color=color,
                    size=10,
                    symbol='diamond',
                    opacity=0.9,
                    line=dict(color='white', width=2)
                ),
                text=away_valid.apply(
                    lambda r: f"<b>{r.get('Away', 'Away')}</b><br>"
                             f"Cluster: {cluster_name}<br>" 
                             f"vs {r.get('Home', 'Home')}<br>"
                             f"Agg: {r.get('Aggression_Away', 0):.2f}<br>"
                             f"M_Liga: {r.get('M_A', 0):.2f}<br>"
                             f"M_Time: {r.get('MT_A', 0):.2f}",
                    axis=1
                ),
                hovertemplate='%{text}<extra></extra>'
            ))

    if len(fig_3d.data) == 0:
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=1, opacity=0),
            showlegend=False
        ))

    all_x = pd.concat([df_plot['Aggression_Home'], df_plot['Aggression_Away']]).dropna()
    all_y = pd.concat([df_plot['M_H'], df_plot['M_A']]).dropna()  
    all_z = pd.concat([df_plot['MT_H'], df_plot['MT_A']]).dropna()
    
    x_range = [all_x.min() - 0.1, all_x.max() + 0.1] if len(all_x) > 0 else [-1, 1]
    y_range = [all_y.min() - 0.5, all_y.max() + 0.5] if len(all_y) > 0 else [-3, 3]
    z_range = [all_z.min() - 0.5, all_z.max() + 0.5] if len(all_z) > 0 else [-3, 3]

    titulo_3d = f"Top {n_to_show} Confrontos - Visualização 3D por Clusters"
    if selected_league != "⚽ Todas as ligas":
        titulo_3d += f" | {selected_league}"

    fig_3d.update_layout(
        title=dict(text=titulo_3d, x=0.5, font=dict(size=16, color='white')),
        scene=dict(
            xaxis=dict(title='Aggression (-1 zebra ↔ +1 favorito)', range=x_range),
            yaxis=dict(title='Momentum (Liga)', range=y_range),
            zaxis=dict(title='Momentum (Time)', range=z_range),
            aspectmode="cube"
        ),
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig_3d

############ Bloco G - Carregamento de Dados e Cache ################
@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
    
    return games_today, history

def load_and_merge_livescore(games_today, selected_date_str):
    games_today = setup_livescore_columns(games_today)
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    if os.path.exists(livescore_file):
        results_df = pd.read_csv(livescore_file)
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

        required_cols = ['Id', 'status', 'home_goal', 'away_goal', 'home_red', 'away_red']
        missing_cols = [col for col in required_cols if col not in results_df.columns]

        if not missing_cols:
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
    
    return games_today

############ Bloco H - Cálculo de Distâncias 3D ################
def calcular_distancias_3d(df):
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
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

############ Bloco I - Sistema ML com Clusters ################
def treinar_modelo_com_clusters(history, games_today):
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(
        lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=False) > 0.5 else 0, axis=1
    )

    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    features_3d = ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
                  'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ', 'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
                  'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo', 'Vector_Sign', 'Magnitude_3D']
    
    features_regressao = ['Media_Score_Home', 'Media_Score_Away', 'Regressao_Force_Home', 
                         'Regressao_Force_Away', 'Extremidade_Home', 'Extremidade_Away']

    extras_3d = history[features_3d].fillna(0)
    extras_regressao = history[features_regressao].fillna(0)

    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, extras_regressao], axis=1)
    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d].fillna(0)
    extras_regressao_today = games_today[features_regressao].fillna(0)

    X_today = pd.concat([ligas_today, clusters_today, extras_today, extras_regressao_today], axis=1)

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Cluster_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Cluster_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Cluster_ML_Score_Main'] = games_today['ML_Confidence']

    return model_home, games_today

############ Bloco J - Sistema de Recomendações com Clusters ################
def adicionar_indicadores_explicativos_clusters(df):
    df = df.copy()

    conditions_home = [
        df['Cluster_ML_Score_Home'] >= 0.65,
        df['Cluster_ML_Score_Home'] >= 0.58,
        df['Cluster_ML_Score_Home'] >= 0.52,
        df['Cluster_ML_Score_Home'] >= 0.48,
        df['Cluster_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    conditions_away = [
        df['Cluster_ML_Score_Away'] >= 0.65,
        df['Cluster_ML_Score_Away'] >= 0.58,
        df['Cluster_ML_Score_Away'] >= 0.52,
        df['Cluster_ML_Score_Away'] >= 0.48,
        df['Cluster_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

    def gerar_recomendacao_clusters(row):
        try:
            cluster = row.get('Cluster3D_Desc', '🌀 Caso Atípico')
            score_home = row.get('Cluster_ML_Score_Home', 0.5)
            score_away = row.get('Cluster_ML_Score_Away', 0.5)
            ml_side = row.get('ML_Side', 'HOME')
            tendencia_h = row.get('Tendencia_Home', '⚖️ ESTÁVEL')
            tendencia_a = row.get('Tendencia_Away', '⚖️ ESTÁVEL')
            
            if any(term in cluster for term in ['Home Domina', 'Home Agressivo', 'Home Vantagem']):
                if score_home >= 0.65 and any(term in tendencia_h for term in ['MELHORA', 'FORTE']):
                    return f'💪 HOME DOMINANTE + Melhora ({score_home:.1%})'
                elif score_home >= 0.58:
                    return f'🎯 HOME DOMINANTE ({score_home:.1%})'
                elif score_home >= 0.52:
                    return f'📈 HOME com Vantagem ({score_home:.1%})'
                else:
                    return f'⚖️ HOME favorecido mas cuidado ({score_home:.1%})'

            elif any(term in cluster for term in ['Away Domina', 'Away Agressivo', 'Away Vantagem']):
                if score_away >= 0.65 and any(term in tendencia_a for term in ['MELHORA', 'FORTE']):
                    return f'💪 AWAY DOMINANTE + Melhora ({score_away:.1%})'
                elif score_away >= 0.58:
                    return f'🎯 AWAY DOMINANTE ({score_away:.1%})'
                elif score_away >= 0.52:
                    return f'📈 AWAY com Vantagem ({score_away:.1%})'
                else:
                    return f'⚖️ AWAY favorecido mas cuidado ({score_away:.1%})'

            elif any(term in cluster for term in ['Equilibrado', 'Equilíbrio', 'Neutro']):
                if ml_side == 'HOME' and score_home >= 0.55:
                    return f'📈 VALUE NO HOME (Equilibrado) ({score_home:.1%})'
                elif ml_side == 'AWAY' and score_away >= 0.55:
                    return f'📈 VALUE NO AWAY (Equilibrado) ({score_away:.1%})'
                else:
                    return f'⚖️ CONFRONTO EQUILIBRADO (H:{score_home:.1%} A:{score_away:.1%})'

            elif any(term in cluster for term in ['Imprevisível', 'Instável', 'Contraditório']):
                if 'MELHORA' in tendencia_h and score_home >= 0.55:
                    return f'🎲 IMPREVISÍVEL mas Home Melhorando ({score_home:.1%})'
                elif 'MELHORA' in tendencia_a and score_away >= 0.55:
                    return f'🎲 IMPREVISÍVEL mas Away Melhorando ({score_away:.1%})'
                else:
                    return f'🎲 JOGO IMPREVISÍVEL - Cautela (H:{score_home:.1%} A:{score_away:.1%})'

            else:
                if score_home >= 0.70:
                    return f'🏆 HOME FORTE ({score_home:.1%})'
                elif score_away >= 0.70:
                    return f'🏆 AWAY FORTE ({score_away:.1%})'
                elif score_home >= 0.60:
                    return f'✅ HOME com Valor ({score_home:.1%})'
                elif score_away >= 0.60:
                    return f'✅ AWAY com Valor ({score_away:.1%})'
                elif abs(score_home - score_away) < 0.1:
                    return f'⚖️ CONFRONTO EQUILIBRADO (H:{score_home:.1%} A:{score_away:.1%})'
                else:
                    return f'🔍 ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'
                    
        except Exception as e:
            return f'❌ ERRO: {str(e)}'

    df['Recomendacao'] = df.apply(gerar_recomendacao_clusters, axis=1)

    df['Score_Final_Clusters'] = (
        df['Cluster_ML_Score_Main'] * 0.6 + 
        df.get('Media_Score_Home', 0.5) * 0.2 + 
        df.get('Media_Score_Away', 0.5) * 0.2
    ) * 100

    conditions_potencial = [
        df['Score_Final_Clusters'] >= 70,
        df['Score_Final_Clusters'] >= 60,
        df['Score_Final_Clusters'] >= 50,
        df['Score_Final_Clusters'] >= 40,
        df['Score_Final_Clusters'] < 40
    ]
    choices_potencial = ['🌟🌟🌟 POTENCIAL MÁXIMO', '🌟🌟 ALTO POTENCIAL', '🌟 POTENCIAL MODERADO', '⚖️ POTENCIAL BAIXO', '🔴 RISCO ALTO']
    df['Classificacao_Potencial'] = np.select(conditions_potencial, choices_potencial, default='🌟 POTENCIAL MODERADO')

    df['Ranking'] = df['Score_Final_Clusters'].rank(ascending=False, method='dense').astype(int)

    return df



############ Bloco L - Sistema Live Score com Clusters ################
# ---------------- LIVE SCORE COM CLUSTERS ----------------
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
    if pd.isna(rec) or handicap_result is None or 'EVITAR' in str(rec):
        return None

    rec = str(rec)

    if any(keyword in rec for keyword in ['HOME', 'Home', 'DOMINANTE']):
        return handicap_result == "HOME_COVERED"
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'DOMINANTE']):
        return handicap_result in ["HOME_NOT_COVERED", "PUSH"]

    return None

def update_real_time_data_clusters(df):
    """Atualiza todos os dados em tempo real para sistema com clusters"""
    df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
    df['Cluster_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
    )
    return df

def generate_live_summary_clusters(df):
    """Gera resumo em tempo real para sistema com clusters"""
    finished_games = df.dropna(subset=['Handicap_Result'])

    if finished_games.empty:
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": 0,
            "Recomendações Cluster": 0,
            "Acertos Cluster": 0,
            "Winrate Cluster": "0%"
        }

    cluster_recomendados = finished_games[finished_games['Cluster_Correct'].notna()]
    total_recomendados = len(cluster_recomendados)
    correct_recomendados = cluster_recomendados['Cluster_Correct'].sum()
    winrate = (correct_recomendados / total_recomendados) * 100 if total_recomendados > 0 else 0

    return {
        "Total Jogos": len(df),
        "Jogos Finalizados": len(finished_games),
        "Recomendações Cluster": total_recomendados,
        "Acertos Cluster": int(correct_recomendados),
        "Winrate Cluster": f"{winrate:.1f}%"
    }




############ Bloco K - Sistema Live Score com Clusters ################
def determine_handicap_result(row):
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

def update_real_time_data_clusters(df):
    df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
    return df

############ Bloco L - Sistema Live Score com Clusters ################
# ---------------- LIVE SCORE COM CLUSTERS ----------------
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
    if pd.isna(rec) or handicap_result is None or 'EVITAR' in str(rec):
        return None

    rec = str(rec)

    if any(keyword in rec for keyword in ['HOME', 'Home', 'DOMINANTE']):
        return handicap_result == "HOME_COVERED"
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'DOMINANTE']):
        return handicap_result in ["HOME_NOT_COVERED", "PUSH"]

    return None

def update_real_time_data_clusters(df):
    """Atualiza todos os dados em tempo real para sistema com clusters"""
    df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
    df['Cluster_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
    )
    return df

def generate_live_summary_clusters(df):
    """Gera resumo em tempo real para sistema com clusters"""
    finished_games = df.dropna(subset=['Handicap_Result'])

    if finished_games.empty:
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": 0,
            "Recomendações Cluster": 0,
            "Acertos Cluster": 0,
            "Winrate Cluster": "0%"
        }

    cluster_recomendados = finished_games[finished_games['Cluster_Correct'].notna()]
    total_recomendados = len(cluster_recomendados)
    correct_recomendados = cluster_recomendados['Cluster_Correct'].sum()
    winrate = (correct_recomendados / total_recomendados) * 100 if total_recomendados > 0 else 0

    return {
        "Total Jogos": len(df),
        "Jogos Finalizados": len(finished_games),
        "Recomendações Cluster": total_recomendados,
        "Acertos Cluster": int(correct_recomendados),
        "Winrate Cluster": f"{winrate:.1f}%"
    }

############ Bloco M - Estilo da Tabela ################
def estilo_tabela_clusters(df):
    def cor_classificacao(valor):
        if '🌟🌟🌟' in str(valor): return 'font-weight: bold'
        elif '🌟🌟' in str(valor): return 'font-weight: bold'
        elif '🌟' in str(valor): return 'font-weight: bold'
        elif '🔴' in str(valor): return 'font-weight: bold'
        elif '🏆' in str(valor): return 'font-weight: bold'
        else: return ''

    colunas_para_estilo = []
    for col in ['Classificacao_Potencial', 'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao']:
        if col in df.columns:
            colunas_para_estilo.append(col)

    styler = df.style
    if colunas_para_estilo:
        styler = styler.applymap(cor_classificacao, subset=colunas_para_estilo)

    if 'Cluster_ML_Score_Home' in df.columns:
        styler = styler.background_gradient(subset=['Cluster_ML_Score_Home'], cmap='RdYlGn')
    if 'Cluster_ML_Score_Away' in df.columns:
        styler = styler.background_gradient(subset=['Cluster_ML_Score_Away'], cmap='RdYlGn')
    if 'Score_Final_Clusters' in df.columns:
        styler = styler.background_gradient(subset=['Score_Final_Clusters'], cmap='RdYlGn')
    if 'M_H' in df.columns:
        styler = styler.background_gradient(subset=['M_H', 'M_A'], cmap='coolwarm')

    return styler

############ Bloco N - Execução Principal ################
st.info("📂 Carregando dados para análise 3D com clusters...")

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

history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

history = history.dropna(subset=['Asian_Line_Decimal'])

if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

history = calcular_momentum_time(history)
games_today = calcular_momentum_time(games_today)
history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)

if not history.empty:
    try:
        modelo_home, games_today = treinar_modelo_com_clusters(history, games_today)
        st.success("✅ Modelo 3D com Clusters treinado com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro no treinamento do modelo: {e}")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

st.markdown("## 🎯 Visualização 3D com Clusters")

col1, col2 = st.columns([2, 1])
with col1:
    if "League" in games_today.columns and not games_today["League"].isna().all():
        leagues = sorted(games_today["League"].dropna().unique())
        selected_league = st.selectbox(
            "Selecione a liga para análise:",
            options=["⚽ Todas as ligas"] + leagues,
            index=0
        )
    else:
        selected_league = "⚽ Todas as ligas"

with col2:
    max_n = len(games_today)
    n_to_show = st.slider("Jogos para exibir:", 10, min(max_n, 100), 30, step=5)

if selected_league != "⚽ Todas as ligas":
    df_filtered = games_today[games_today["League"] == selected_league].copy()
else:
    df_filtered = games_today.copy()

df_filtered = aplicar_clusterizacao_3d(df_filtered)

clusters_disponiveis = df_filtered['Cluster3D_Desc'].unique() if 'Cluster3D_Desc' in df_filtered.columns else []
if len(clusters_disponiveis) > 0:
    cluster_selecionado = st.selectbox(
        "Filtrar por tipo de confronto:",
        options=["🎯 Todos os clusters"] + list(clusters_disponiveis),
        index=0
    )

    if cluster_selecionado != "🎯 Todos os clusters":
        df_plot = df_filtered[df_filtered['Cluster3D_Desc'] == cluster_selecionado].copy()
    else:
        df_plot = df_filtered.copy()
else:
    df_plot = df_filtered.copy()

df_plot = df_plot.head(n_to_show)

if not df_plot.empty:
    try:
        fig_3d_clusters = create_3d_plot_with_clusters(df_plot, n_to_show, selected_league)
        st.plotly_chart(fig_3d_clusters, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico 3D: {e}")

st.markdown("## 🏆 Melhores Oportunidades - Sistema Clusters 3D")

if not games_today.empty and 'Cluster_ML_Score_Home' in games_today.columns:
    ranking_clusters = games_today.copy()
    ranking_clusters = adicionar_indicadores_explicativos_clusters(ranking_clusters)
    ranking_clusters = update_real_time_data_clusters(ranking_clusters)  # ✅ ATUALIZA LIVE SCORE
    ranking_clusters = ranking_clusters.sort_values('Score_Final_Clusters', ascending=False)

    # ---------------- EXIBIR RESUMO LIVE ----------------
    st.markdown("### 📡 Live Score Monitor")
    live_summary = generate_live_summary_clusters(ranking_clusters)
    st.json(live_summary)
    
    colunas_principais = [
        'League','Time' ,'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'ML_Side',
        'Recomendacao', 'Cluster_ML_Score_Home', 'Cluster_ML_Score_Away',
        'Score_Final_Clusters', 'Classificacao_Potencial', 'Cluster3D_Desc',
        'Tendencia_Home', 'Tendencia_Away',
        # Live Score
        'Asian_Line_Decimal', 'Handicap_Result', 'Cluster_Correct'
    ]
    
    cols_finais = [c for c in colunas_principais if c in ranking_clusters.columns]
    
    styler = estilo_tabela_clusters(ranking_clusters[cols_finais])

    st.dataframe(
        styler.format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Cluster_ML_Score_Home': '{:.1%}',
            'Cluster_ML_Score_Away': '{:.1%}',
            'Score_Final_Clusters': '{:.1f}',
            'M_H': '{:.2f}',
            'M_A': '{:.2f}'
        }, na_rep="-"),
        use_container_width=True,
        height=600
    )
else:
    st.error("❌ Não foi possível gerar a tabela de confrontos")

st.success("🎯 **Sistema 3D com Clusters ML** implementado com sucesso!")
