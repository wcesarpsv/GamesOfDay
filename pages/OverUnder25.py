# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import math

# ==========================================================
# CONFIGURAÇÕES BÁSICAS
# ==========================================================
st.set_page_config(page_title="Análise Over/Under 2.5 Gols - Bet Indicator", layout="wide")
st.title("🎯 Previsão Over/Under 2.5 Gols - ML + WG GAP")

PAGE_PREFIX = "OverUnderML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "coppa", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ==========================================================
# LIVE SCORE – COLUNAS
# ==========================================================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas do Live Score existam no DataFrame"""
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

# ==========================================================
# HELPERS BÁSICOS
# ==========================================================
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

# ==========================================================
# CRIAÇÃO DO TARGET OVER/UNDER 2.5 (MAIS ROBUSTA)
# ==========================================================
def create_over_under_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria target para Over/Under 2.5 gols
    Target_Over = 1 se total de gols > 2.5, 0 caso contrário
    """
    df = df.copy()
    
    # Verificar se as colunas necessárias existem
    required_cols = ['Goals_H_FT', 'Goals_A_FT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para criar target Over/Under: {missing_cols}")
        st.info("🔍 Procurando colunas alternativas de gols...")
        
        # Procurar colunas alternativas que possam conter dados de gols
        possible_goal_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['goal', 'gol', 'score', 'ft']):
                possible_goal_cols.append(col)
        
        if possible_goal_cols:
            st.info(f"Colunas potenciais de gols encontradas: {possible_goal_cols}")
            
            # Tentar usar as primeiras colunas encontradas como alternativa
            if len(possible_goal_cols) >= 2:
                df['Goals_H_FT'] = df[possible_goal_cols[0]]
                df['Goals_A_FT'] = df[possible_goal_cols[1]]
                st.info(f"Usando colunas alternativas: {possible_goal_cols[0]} e {possible_goal_cols[1]}")
            else:
                # Criar colunas vazias como fallback
                st.warning("Colunas alternativas insuficientes, criando target vazio")
                df["Target_Over"] = 0
                return df
        else:
            # Criar colunas vazias para evitar erro
            st.warning("Nenhuma coluna alternativa encontrada, criando target vazio")
            for col in missing_cols:
                df[col] = 0
            df["Target_Over"] = 0
            return df
    
    # Verificar se temos dados válidos nas colunas de gols
    valid_goals_mask = (~df['Goals_H_FT'].isna()) & (~df['Goals_A_FT'].isna())
    valid_count = valid_goals_mask.sum()
    
    if valid_count == 0:
        st.warning("⚠️ Nenhum dado válido encontrado nas colunas de gols")
        df["Target_Over"] = 0
        return df
    
    st.info(f"✅ Encontrados {valid_count} jogos com dados de gols válidos")
    
    # Calcular total de gols
    df["Total_Goals"] = df["Goals_H_FT"] + df["Goals_A_FT"]
    
    # Criar target binário
    df["Target_Over"] = (df["Total_Goals"] > 2.5).astype(int)
    
    # Estatísticas
    total_games = len(df)
    over_games = df["Target_Over"].sum()
    under_games = total_games - over_games
    
    st.info(f"🎯 Total analisado: {total_games} jogos")
    st.info(f"🔴 Jogos Over 2.5: {over_games} ({over_games/total_games:.1%})")
    st.info(f"🔵 Jogos Under 2.5: {under_games} ({under_games/total_games:.1%})")
    
    return df
# ==========================================================
# FEATURES ESPECÍFICAS PARA OVER/UNDER
# ==========================================================
def create_over_under_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features específicas para previsão de Over/Under 2.5
    """
    df = df.copy()
    
    # Features básicas de ataque
    basic_features = []
    
    # Usar OverScore se disponível - SEMPRE calcular avg aqui também
    if 'OverScore_Home' in df.columns and 'OverScore_Away' in df.columns:
        df['OverScore_Avg'] = (df['OverScore_Home'] + df['OverScore_Away']) / 2
        df['OverScore_Diff'] = df['OverScore_Home'] - df['OverScore_Away']
        df['OverScore_Max'] = df[['OverScore_Home', 'OverScore_Away']].max(axis=1)
        df['OverScore_Min'] = df[['OverScore_Home', 'OverScore_Away']].min(axis=1)
        basic_features.extend(['OverScore_Avg', 'OverScore_Diff', 'OverScore_Max', 'OverScore_Min'])
    
    # Features de média de gols
    if 'Goals_H_FT' in df.columns and 'Goals_A_FT' in df.columns:
        df['Avg_Goals_Home'] = df['Goals_H_FT']
        df['Avg_Goals_Away'] = df['Goals_A_FT']
        df['Total_Goals_Expected'] = df['Avg_Goals_Home'] + df['Avg_Goals_Away']
        basic_features.extend(['Avg_Goals_Home', 'Avg_Goals_Away', 'Total_Goals_Expected'])
    
    # Features de odds (se disponíveis)
    odds_features = []
    if 'Odd_Over25' in df.columns and 'Odd_Under25' in df.columns:
        df['Over25_Probability'] = 1 / df['Odd_Over25']
        df['Under25_Probability'] = 1 / df['Odd_Under25']
        df['Over_Under_Ratio'] = df['Odd_Under25'] / df['Odd_Over25']
        odds_features.extend(['Over25_Probability', 'Under25_Probability', 'Over_Under_Ratio'])
    
    # Features de forma ofensiva/defensiva
    form_features = []
    if 'Aggression_Home' in df.columns and 'Aggression_Away' in df.columns:
        df['Aggression_Total'] = df['Aggression_Home'] + df['Aggression_Away']
        df['Aggression_Diff'] = df['Aggression_Home'] - df['Aggression_Away']
        form_features.extend(['Aggression_Total', 'Aggression_Diff'])
    
    # Combinar todas as features
    all_features = basic_features + odds_features + form_features
    available_features = [f for f in all_features if f in df.columns]
    
    st.info(f"📋 Features Over/Under disponíveis: {len(available_features)}/{len(all_features)}")
    
    return df[available_features].fillna(0)

# ==========================================================
# WEIGHTED GOALS ADAPTADO PARA OVER/UNDER
# ==========================================================
def adicionar_weighted_goals_over_under(df: pd.DataFrame) -> pd.DataFrame:
    """
    WG adaptado para Over/Under - foco em características ofensivas
    """
    df_temp = df.copy()

    for col in ['Goals_H_FT', 'Goals_A_FT']:
        if col not in df_temp.columns:
            df_temp[col] = np.nan

    def wg_offensive_home(row):
        gh = row.get('Goals_H_FT', np.nan)
        if pd.isna(gh):
            return np.nan
        
        # Usar OverScore se disponível como peso
        over_weight = row.get('OverScore_Home', 50) / 100  # Normalizar para 0-1
        return gh * (0.5 + over_weight * 0.5)

    def wg_offensive_away(row):
        ga = row.get('Goals_A_FT', np.nan)
        if pd.isna(ga):
            return np.nan
        
        over_weight = row.get('OverScore_Away', 50) / 100
        return ga * (0.5 + over_weight * 0.5)

    df_temp['WG_Offensive_Home'] = df_temp.apply(wg_offensive_home, axis=1)
    df_temp['WG_Offensive_Away'] = df_temp.apply(wg_offensive_away, axis=1)

    return df_temp

def calcular_rolling_offensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcular médias móveis para features ofensivas
    """
    df_temp = df.copy()

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp = df_temp.sort_values('Date')

    for col in ['WG_Offensive_Home', 'WG_Offensive_Away']:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    # Rolling features para times como home
    df_temp['WG_Offensive_Home_Team'] = df_temp.groupby('Home')['WG_Offensive_Home'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Rolling features para times como away  
    df_temp['WG_Offensive_Away_Team'] = df_temp.groupby('Away')['WG_Offensive_Away'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Diferenciais ofensivos
    df_temp['WG_Offensive_Diff'] = df_temp['WG_Offensive_Home_Team'] - df_temp['WG_Offensive_Away_Team']
    df_temp['WG_Offensive_Total'] = df_temp['WG_Offensive_Home_Team'] + df_temp['WG_Offensive_Away_Team']

    return df_temp

def enrich_games_today_with_offensive_features(games_today: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquecer jogos de hoje com WG + OverScore últimos valores do histórico
    SEM ERROS mesmo quando colunas não existirem em algumas ligas
    """
    if history.empty or games_today.empty:
        return games_today

    hist = history.copy()

    # Garantir colunas mínimas no histórico
    base_cols = [
        'WG_Offensive_Home_Team', 'WG_Offensive_Away_Team',
        'OverScore_Home', 'OverScore_Away'
    ]
    for col in base_cols:
        if col not in hist.columns:
            hist[col] = 0.0

    # Agrupar últimos valores por time
    last = hist.groupby('Home')[base_cols].last().reset_index()
    last = last.rename(columns={
        'Home': 'Team',
        'WG_Offensive_Home_Team': 'WG_H_Last',
        'WG_Offensive_Away_Team': 'WG_A_Last',
        'OverScore_Home': 'Over_H_Last',
        'OverScore_Away': 'Over_A_Last'
    })

    # Merge HOME
    games_today = games_today.merge(
        last, left_on='Home', right_on='Team', how='left'
    ).drop('Team', axis=1)

    # Merge AWAY
    games_today = games_today.merge(
        last, left_on='Away', right_on='Team', how='left', suffixes=('_HOME', '_AWAY')
    ).drop('Team', axis=1)

    # Criar colunas faltantes com 0 para evitar KeyError
    required_cols = [
        'WG_H_Last_HOME', 'WG_A_Last_HOME', 'Over_H_Last_HOME', 'Over_A_Last_HOME',
        'WG_H_Last_AWAY', 'WG_A_Last_AWAY', 'Over_H_Last_AWAY', 'Over_A_Last_AWAY'
    ]
    for col in required_cols:
        if col not in games_today.columns:
            games_today[col] = 0.0
        else:
            games_today[col] = games_today[col].fillna(0.0)

    # Consolidar finais
    games_today['WG_Offensive_Home_Last'] = games_today['WG_H_Last_HOME']
    games_today['WG_Offensive_Away_Last'] = games_today['WG_H_Last_AWAY']

    games_today['WG_Offensive_Diff_Last'] = (
        games_today['WG_Offensive_Home_Last'] - games_today['WG_Offensive_Away_Last']
    )

    games_today['WG_Offensive_Total_Last'] = (
        games_today['WG_Offensive_Home_Last'] + games_today['WG_Offensive_Away_Last']
    )

    # OverScore média final para o modelo
    games_today['OverScore_Avg_Last'] = (
        games_today['Over_H_Last_HOME'] + games_today['Over_H_Last_AWAY']
    ) / 2

    return games_today



# ==========================================================
# PARÂMETROS POR LIGA PARA OVER/UNDER
# ==========================================================
@st.cache_data(ttl=7*24*3600)
def calcular_parametros_liga_over_under(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula parâmetros específicos por liga para Over/Under
    """
    if history.empty:
        return pd.DataFrame()
    
    df = history.copy()
    
    # Garantir que temos as colunas necessárias
    required_cols = ['League', 'Goals_H_FT', 'Goals_A_FT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo de parâmetros Over/Under: {missing_cols}")
        return pd.DataFrame()
    
    # Filtrar apenas linhas com dados válidos
    df = df.dropna(subset=['Goals_H_FT', 'Goals_A_FT']).copy()
    
    if df.empty:
        st.warning("⚠️ Nenhum dado válido após remover NaNs")
        return pd.DataFrame()
    
    # Calcular estatísticas por liga
    liga_stats = df.groupby('League').agg({
        'Goals_H_FT': ['count', 'mean'],
        'Goals_A_FT': 'mean'
    }).round(3)
    
    # Flatten column names
    liga_stats.columns = [
        'Jogos_Total', 'Gols_Media_Casa', 
        'Gols_Media_Fora'
    ]
    
    # Calcular estatísticas Over/Under
    df['Total_Goals'] = df['Goals_H_FT'] + df['Goals_A_FT']
    over_stats = df.groupby('League')['Total_Goals'].agg([
        ('Over_Rate', lambda x: (x > 2.5).mean()),
        ('Avg_Total_Goals', 'mean'),
        ('Goal_Variance', 'std')
    ]).round(3)
    
    # Combinar estatísticas
    liga_stats = liga_stats.merge(over_stats, on='League', how='left')
    
    # Calcular tendência ofensiva da liga
    liga_stats['Offensive_Rating'] = (
        liga_stats['Gols_Media_Casa'] + liga_stats['Gols_Media_Fora']
    ).round(2)
    
    # Filtrar ligas com poucos jogos
    liga_stats = liga_stats[liga_stats['Jogos_Total'] >= 5].copy()
    
    st.success(f"✅ Parâmetros Over/Under calculados para {len(liga_stats)} ligas")
    
    return liga_stats.reset_index()

# ==========================================================
# MODELO PARA OVER/UNDER
# ==========================================================
def train_over_under_model(X, y, feature_names):
    """
    Treina modelo específico para Over/Under 2.5
    """
    st.info("🤖 Treinando modelo Over/Under 2.5...")
    X_clean = clean_features_for_training(X)
    y_clean = y.copy()

    if hasattr(y_clean, 'isna') and y_clean.isna().any():
        st.warning(f"⚠️ Encontrados {y_clean.isna().sum()} NaNs no target - removendo")
        valid_mask = ~y_clean.isna()
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    try:
        scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='accuracy')
        st.write(f"📊 Validação Cruzada Over/Under: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        if scores.mean() < 0.55:
            st.warning("⚠️ Modelo abaixo do esperado - verificar qualidade dos dados")
        elif scores.mean() > 0.65:
            st.success("🎯 Modelo Over/Under com boa performance!")
    except Exception as e:
        st.warning(f"⚠️ Validação cruzada falhou: {e}")

    model.fit(X_clean, y_clean)

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    st.write("🔍 **Top Features mais importantes para Over/Under:**")
    st.dataframe(importances.head(10).to_frame("Importância"))

    return model

# ==========================================================
# LIMPEZA DAS FEATURES (MANTIDA DO CÓDIGO ORIGINAL)
# ==========================================================
def clean_features_for_training(X):
    X_clean = X.copy()
    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean)
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    inf_count = (X_clean == np.inf).sum().sum() + (X_clean == -np.inf).sum().sum()
    nan_count = X_clean.isna().sum().sum()
    if inf_count > 0 or nan_count > 0:
        st.warning(f"⚠️ Encontrados {inf_count} infinitos e {nan_count} NaNs nas features")
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)
    for col in X_clean.columns:
        if X_clean[col].dtype in [np.float64, np.float32]:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)
    final_inf_count = (X_clean == np.inf).sum().sum() + (X_clean == -np.inf).sum().sum()
    final_nan_count = X_clean.isna().sum().sum()
    if final_inf_count > 0 or final_nan_count > 0:
        st.error(f"❌ Ainda existem {final_inf_count} infinitos e {final_nan_count} NaNs após limpeza")
        X_clean = X_clean.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
    st.success(f"✅ Features limpas: {X_clean.shape}")
    return X_clean

# ==========================================================
# LOAD + FILTER HISTORY (TIME-SAFE) - MANTIDO
# ==========================================================
def load_and_filter_history(selected_date_str: str) -> pd.DataFrame:
    """Carrega histórico APENAS com jogos anteriores à data selecionada"""
    st.info("📊 Carregando histórico com filtro temporal correto...")

    history = filter_leagues(load_all_games(GAMES_FOLDER))

    if history.empty:
        st.warning("⚠️ Histórico vazio")
        return history

    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        selected_date = pd.to_datetime(selected_date_str)
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📅 Histórico filtrado: {len(history)} jogos anteriores a {selected_date_str}")

    # Preencher quaisquer NaNs com 0 APÓS o load
    history = history.fillna(0)

    st.success(f"✅ Histórico processado: {len(history)} jogos")
    return history

# ==========================================================
# GRÁFICO PARA OVER/UNDER
# ==========================================================
def plot_over_under_analysis(games_today: pd.DataFrame):
    """
    Gráfico específico para análise Over/Under
    """
    if games_today.empty:
        st.info("Sem jogos para exibir no gráfico Over/Under.")
        return

    required_cols = ['League', 'Home', 'Away', 'Prob_Over', 'WG_Offensive_Total_Last']
    missing = [c for c in required_cols if c not in games_today.columns]
    if missing:
        st.warning(f"Não é possível gerar o gráfico Over/Under. Faltam colunas: {missing}")
        return

    st.markdown("## 📊 Análise Over/Under 2.5 - Probabilidade vs Potencial Ofensivo")

    # Preparar dados
    df_plot = games_today.copy()
    df_plot = df_plot.sort_values('Prob_Over', ascending=False).head(20)

    # Criar gráfico
    fig = go.Figure()

    # Adicionar barras para probabilidade Over
    fig.add_trace(go.Bar(
        x=df_plot['Home'] + ' vs ' + df_plot['Away'],
        y=df_plot['Prob_Over'],
        name='Probabilidade Over',
        marker_color='red',
        opacity=0.7
    ))

    # Adicionar linha para potencial ofensivo
    fig.add_trace(go.Scatter(
        x=df_plot['Home'] + ' vs ' + df_plot['Away'],
        y=df_plot['WG_Offensive_Total_Last'],
        name='Potencial Ofensivo (WG)',
        line=dict(color='blue', width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Probabilidade Over 2.5 vs Potencial Ofensivo',
        xaxis_title='Jogos',
        yaxis_title='Probabilidade Over',
        yaxis2=dict(
            title='Potencial Ofensivo',
            overlaying='y',
            side='right'
        ),
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# CARREGAR DADOS (GAMESDAY + HISTORY + LIVESCORE) - MANTIDO
# ==========================================================
st.info("📂 Carregando dados para análise Over/Under 2.5...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Função de cache para carregar dados
@st.cache_data(ttl=3600)
def load_cached_data_over_under(selected_file: str):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    history = load_and_filter_history(selected_date_str)
    return games_today, history, selected_date_str

games_today, history, selected_date_str = load_cached_data_over_under(selected_file)

# Carregar livescore
def load_and_merge_livescore(games_today: pd.DataFrame, selected_date_str: str) -> pd.DataFrame:
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

        required_cols = ['Id', 'status', 'home_goal', 'away_goal']
        missing_cols = [col for col in required_cols if col not in results_df.columns]

        if missing_cols:
            st.error(f"❌ LiveScore file missing columns: {missing_cols}")
            return games_today

        games_today = games_today.merge(
            results_df[required_cols],
            left_on='Id',
            right_on='Id',
            how='left',
            suffixes=('', '_RAW')
        )

        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

        st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
        return games_today
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

games_today = load_and_merge_livescore(games_today, selected_date_str)

# ==========================================================
# ==========================================================
# PROCESSAMENTO PARA OVER/UNDER (CORRIGIDO)
# ==========================================================

# 1. Calcular parâmetros por liga
if not history.empty:
    liga_params_ou = calcular_parametros_liga_over_under(history)
    
    if not liga_params_ou.empty:
        st.markdown("### 📊 Parâmetros Over/Under por Liga")
        display_params = liga_params_ou.sort_values('Over_Rate', ascending=False)[[
            'League', 'Jogos_Total', 'Over_Rate', 'Avg_Total_Goals', 'Offensive_Rating'
        ]].head(15)
        st.dataframe(display_params)

# ==========================================================
# 2️⃣ PROCESSAMENTO PARA OVER/UNDER (CORRIGIDO 100%)
# ==========================================================

if not history.empty:

    # 🔹 Criar target Over/Under (HISTÓRICO ORIGINAL)
    history_ou = create_over_under_target(history)

    # ❗ Garantir que Target_Over existe
    if 'Target_Over' not in history_ou.columns:
        st.error("❌ Target_Over não foi criado no histórico. Conferir colunas de gols!")
        st.write("Colunas disponíveis no histórico:", list(history_ou.columns))
        st.stop()

    st.success("🎯 Target_Over criado com sucesso no histórico")

    # 🔹 Adicionar features ofensivas (NOVA CÓPIA — NÃO sobrescrever history_ou)
    history_enriched = adicionar_weighted_goals_over_under(history_ou)
    history_enriched = calcular_rolling_offensive_features(history_enriched)

    # 🔹 Gerar features SOMENTE para modelo
    X_hist_ou = create_over_under_features(history_enriched)
    y_ou = history_ou['Target_Over']

    # ❗ Verificação final de validade
    if X_hist_ou.empty or y_ou.empty:
        st.error("❌ Dados insuficientes para treinar modelo Over/Under")
        st.stop()

    # 🔹 Treinar Modelo
    model_ou = train_over_under_model(X_hist_ou, y_ou, X_hist_ou.columns)

    # ==========================================================
    # 3️⃣ PREVISÕES PARA OS JOGOS DE HOJE
    # ==========================================================
    # 3️⃣ PREVISÕES PARA OS JOGOS DE HOJE
    if not games_today.empty:
    
        # 🔹 Enriquecer dados dos jogos de hoje
        games_enriched = adicionar_weighted_goals_over_under(games_today.copy())
        games_enriched = enrich_games_today_with_offensive_features(
            games_enriched, history_enriched
        )
    
        # 🔹 Garantir que TODAS features do treino existem nos jogos de hoje
        for col in feature_list:
            if col not in games_enriched.columns:
                games_enriched[col] = 0.0
    
        # 🔹 Garantir a ordem correta de features
        X_today_ou = games_enriched[feature_list]
    
        # 🔹 Predição
        proba_over = model_ou.predict_proba(X_today_ou)[:, 1]
        games_today['Prob_Over'] = proba_over
        games_today['Pred_Over'] = (proba_over > 0.5).astype(int)
    
        # 🔹 Confiança
        games_today['OU_Confidence'] = np.abs(proba_over - 0.5) * 2
    
        # 🔹 Sinal final
        games_today['OU_Signal'] = np.where(
            games_today['Prob_Over'] > 0.5, 
            'OVER', 
            'UNDER'
        )
    
        games_today['OU_Approved'] = games_today['OU_Confidence'] > 0.1



# ==========================================================
# DASHBOARD OVER/UNDER
# ==========================================================
if not games_today.empty:
    # Gráfico de análise
    plot_over_under_analysis(games_today)
    
    # Ranking por probabilidade Over
    st.markdown("## 🏆 Melhores Oportunidades Over/Under 2.5")
    
    # Ordenar por probabilidade Over
    ranking_ou = games_today.sort_values('Prob_Over', ascending=False).copy()
    
    # Colunas para display
    cols_ou = [
        'League', 'Home', 'Away', 
        'OverScore_Home', 'OverScore_Away', 'OverScore_Avg',
        'WG_Offensive_Total_Last', 'WG_Offensive_Diff_Last',
        'Prob_Over', 'OU_Confidence', 'OU_Signal', 'OU_Approved'
    ]
    
    # Filtrar colunas disponíveis
    cols_ou = [c for c in cols_ou if c in ranking_ou.columns]
    
    st.dataframe(ranking_ou[cols_ou].head(25))
    
    # Sinais aprovados
    aprovados_ou = ranking_ou[ranking_ou['OU_Approved']].copy()
    if not aprovados_ou.empty:
        st.markdown("### ✅ Sinais Over/Under Aprovados")
        
        cols_aprov_ou = [
            'League', 'Home', 'Away', 
            'OverScore_Avg', 'Prob_Over', 'OU_Signal', 'OU_Confidence'
        ]
        cols_aprov_ou = [c for c in cols_aprov_ou if c in aprovados_ou.columns]
        
        st.dataframe(aprovados_ou[cols_aprov_ou].head(20))
        
        # Estatísticas dos sinais
        col1, col2, col3 = st.columns(3)
        with col1:
            over_count = len(aprovados_ou[aprovados_ou['OU_Signal'] == 'OVER'])
            st.metric("Sinais OVER", over_count)
        with col2:
            under_count = len(aprovados_ou[aprovados_ou['OU_Signal'] == 'UNDER'])
            st.metric("Sinais UNDER", under_count)
        with col3:
            avg_confidence = aprovados_ou['OU_Confidence'].mean()
            st.metric("Confiança Média", f"{avg_confidence:.2f}")
    else:
        st.info("Nenhum sinal Over/Under aprovado para hoje.")

# ==========================================================
# ANÁLISE DAS FEATURES OVERSCORE (SIMPLIFICADO)
# ==========================================================
if not games_today.empty and 'OverScore_Home' in games_today.columns:
    st.markdown("## 📈 Análise das Features OverScore")
    
    # AGORA OverScore_Avg DEVE EXISTIR (foi criado em create_over_under_features)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("OverScore Home Médio", f"{games_today['OverScore_Home'].mean():.1f}")
        st.metric("OverScore Away Médio", f"{games_today['OverScore_Away'].mean():.1f}")
    
    with col2:
        st.metric("OverScore Avg Médio", f"{games_today['OverScore_Avg'].mean():.1f}")
        st.metric("Maior OverScore", f"{games_today['OverScore_Avg'].max():.1f}")
    
    # Correlação com probabilidade Over
    if 'Prob_Over' in games_today.columns:
        corr_home = games_today['OverScore_Home'].corr(games_today['Prob_Over'])
        corr_away = games_today['OverScore_Away'].corr(games_today['Prob_Over'])
        corr_avg = games_today['OverScore_Avg'].corr(games_today['Prob_Over'])
        
        st.write(f"**Correlações:**")
        st.write(f"- OverScore Home vs Prob Over: {corr_home:.3f}")
        st.write(f"- OverScore Away vs Prob Over: {corr_away:.3f}")
        st.write(f"- OverScore Avg vs Prob Over: {corr_avg:.3f}")

st.success("✅ Análise Over/Under 2.5 concluída!")
