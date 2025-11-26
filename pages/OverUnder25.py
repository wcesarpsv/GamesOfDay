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
st.set_page_config(page_title="Sistema Over/Under - Análise Preditiva", layout="wide")
st.title("⚽ Sistema de Previsão Over/Under 2.5")

PAGE_PREFIX = "OverUnder_ML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "coppa", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ==========================================================
# HELPERS BÁSICOS
# ==========================================================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renomear colunas comuns para padronização
    column_mapping = {
        'Goals_H_FT_x': 'Goals_H_FT', 'Goals_A_FT_x': 'Goals_A_FT',
        'Goals_H_FT_y': 'Goals_H_FT', 'Goals_A_FT_y': 'Goals_A_FT',
        'HomeTeam': 'Home', 'AwayTeam': 'Away',
        'home_team': 'Home', 'away_team': 'Away'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
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
# CONVERSÃO OVERLINE
# ==========================================================
def convert_overline_to_decimal(overline_str):
    """
    Converte linhas de Over/Under no formato string para decimal
    """
    if pd.isna(overline_str) or overline_str == "":
        return 2.5  # Default mais comum
    
    overline_str = str(overline_str).strip()
    
    # Mapeamento de linhas split
    split_lines = {
        '2.5/3': 2.75,
        '2/2.5': 2.25,
        '1.5/2': 1.75,
        '3/3.5': 3.25,
        '3.5/4': 3.75,
        '2.5/3.0': 2.75,
        '2.0/2.5': 2.25,
        '1.5': 1.5,
        '2': 2.0,
        '2.5': 2.5,
        '3': 3.0,
        '3.5': 3.5,
        '4': 4.0,
        '4.5': 4.5
    }
    
    if overline_str in split_lines:
        return split_lines[overline_str]
    
    # Tentar converter para float
    try:
        return float(overline_str)
    except ValueError:
        return 2.5  # Fallback

# ==========================================================
# TARGETS OVER/UNDER
# ==========================================================
def create_over_under_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria targets para Over/Under 2.5 e outras linhas
    """
    df = df.copy()
    
    # Verificar se temos colunas de gols
    goals_h_col = next((col for col in ['Goals_H_FT', 'home_goal', 'Goals_H'] if col in df.columns), None)
    goals_a_col = next((col for col in ['Goals_A_FT', 'away_goal', 'Goals_A'] if col in df.columns), None)
    
    if not goals_h_col or not goals_a_col:
        st.error("❌ Colunas de gols não encontradas no DataFrame")
        return df
    
    # Total de gols do jogo
    df['Total_Goals'] = df[goals_h_col] + df[goals_a_col]
    
    # Converter OverLine para análise
    if 'OverLine' in df.columns:
        df['OverLine_Decimal'] = df['OverLine'].apply(convert_overline_to_decimal)
    
    # Targets principais
    df['Over_2.5'] = (df['Total_Goals'] > 2.5).astype(int)
    df['Under_2.5'] = (df['Total_Goals'] < 2.5).astype(int)
    
    # Targets adicionais para análise
    df['Over_1.5'] = (df['Total_Goals'] > 1.5).astype(int)
    df['Over_3.5'] = (df['Total_Goals'] > 3.5).astype(int)
    df['BTTS_Yes'] = ((df[goals_h_col] > 0) & (df[goals_a_col] > 0)).astype(int)
    
    return df

# ==========================================================
# PARÂMETROS POR LIGA
# ==========================================================
@st.cache_data(ttl=7*24*3600)
def calcular_parametros_liga_avancado(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula parâmetros específicos por liga para Over/Under
    """
    if history.empty:
        return pd.DataFrame()
    
    df = history.copy()
    
    # Encontrar colunas de gols
    goals_h_col = next((col for col in ['Goals_H_FT', 'home_goal', 'Goals_H'] if col in df.columns), None)
    goals_a_col = next((col for col in ['Goals_A_FT', 'away_goal', 'Goals_A'] if col in df.columns), None)
    
    if not goals_h_col or not goals_a_col or 'League' not in df.columns:
        st.warning("⚠️ Colunas necessárias não encontradas para cálculo de parâmetros")
        return pd.DataFrame()
    
    # Calcular estatísticas por liga
    liga_stats = df.groupby('League').agg({
        goals_h_col: ['count', 'mean'],
        goals_a_col: 'mean'
    }).round(3)
    
    # Flatten column names
    liga_stats.columns = ['Jogos_Total', 'Gols_Media_Casa', 'Gols_Media_Fora']
    
    # Calcular métricas para Over/Under
    liga_stats['Base_Goals_Liga'] = (liga_stats['Gols_Media_Casa'] + liga_stats['Gols_Media_Fora']).round(2)
    liga_stats['Over_2.5_Rate'] = df.groupby('League').apply(
        lambda x: (x[goals_h_col] + x[goals_a_col] > 2.5).mean()
    ).round(3)
    
    liga_stats['BTTS_Rate'] = df.groupby('League').apply(
        lambda x: ((x[goals_h_col] > 0) & (x[goals_a_col] > 0)).mean()
    ).round(3)
    
    # Filtrar ligas com poucos jogos
    liga_stats = liga_stats[liga_stats['Jogos_Total'] >= 5].copy()
    
    st.success(f"✅ Parâmetros calculados para {len(liga_stats)} ligas")
    
    return liga_stats.reset_index()

# ==========================================================
# WEIGHTED GOALS PARA OVER/UNDER
# ==========================================================
def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    WG_Home / WG_Away ofensivos para Over/Under
    """
    df_temp = df.copy()

    # Encontrar colunas de gols
    goals_h_col = next((col for col in ['Goals_H_FT', 'home_goal', 'Goals_H'] if col in df_temp.columns), None)
    goals_a_col = next((col for col in ['Goals_A_FT', 'away_goal', 'Goals_A'] if col in df_temp.columns), None)

    if not goals_h_col or not goals_a_col:
        df_temp['WG_Home'] = 0.0
        df_temp['WG_Away'] = 0.0
        return df_temp

    def odds_to_probs_1x2(row):
        try:
            odd_h = float(row.get('Odd_H', 0))
            odd_d = float(row.get('Odd_D', 0))
            odd_a = float(row.get('Odd_A', 0))
        except Exception:
            return 0.33, 0.34, 0.33

        if odd_h <= 1.01 or odd_a <= 1.01:
            return 0.33, 0.34, 0.33

        inv_h = 1.0 / odd_h if odd_h > 0 else 0.0
        inv_d = 1.0 / odd_d if odd_d > 0 else 0.0
        inv_a = 1.0 / odd_a if odd_a > 0 else 0.0

        total = inv_h + inv_d + inv_a
        if total <= 0:
            return 0.33, 0.34, 0.33

        p_h = inv_h / total
        p_d = inv_d / total
        p_a = inv_a / total
        return p_h, p_d, p_a

    def wg_home(row):
        gh = row.get(goals_h_col, np.nan)
        ga = row.get(goals_a_col, np.nan)
        if pd.isna(gh) or pd.isna(ga):
            return np.nan

        p_h, p_d, p_a = odds_to_probs_1x2(row)
        weight_for = (1 - p_h) + 0.5 * p_d
        weight_against = p_h + 0.5 * p_d
        return gh * weight_for - ga * weight_against

    def wg_away(row):
        gh = row.get(goals_h_col, np.nan)
        ga = row.get(goals_a_col, np.nan)
        if pd.isna(gh) or pd.isna(ga):
            return np.nan

        p_h, p_d, p_a = odds_to_probs_1x2(row)
        weight_for = (1 - p_a) + 0.5 * p_d
        weight_against = p_a + 0.5 * p_d
        return ga * weight_for - gh * weight_against

    df_temp['WG_Home'] = df_temp.apply(wg_home, axis=1)
    df_temp['WG_Away'] = df_temp.apply(wg_away, axis=1)

    return df_temp

def adicionar_weighted_goals_defensivos(df: pd.DataFrame, liga_params: pd.DataFrame) -> pd.DataFrame:
    """
    WG Defensivos para Over/Under
    """
    df_temp = df.copy()
    
    # Encontrar colunas de gols
    goals_h_col = next((col for col in ['Goals_H_FT', 'home_goal', 'Goals_H'] if col in df_temp.columns), None)
    goals_a_col = next((col for col in ['Goals_A_FT', 'away_goal', 'Goals_A'] if col in df_temp.columns), None)

    if not goals_h_col or not goals_a_col:
        df_temp['WG_Def_Home'] = 0.0
        df_temp['WG_Def_Away'] = 0.0
        return df_temp
    
    # Inicializar parâmetros
    df_temp['Base_Goals_Liga'] = 2.5
    
    # Se temos parâmetros por liga
    if liga_params is not None and not liga_params.empty and 'League' in df_temp.columns:
        df_temp = df_temp.merge(
            liga_params[['League', 'Base_Goals_Liga']],
            on='League',
            how='left',
            suffixes=('', '_y')
        )
        if 'Base_Goals_Liga_y' in df_temp.columns:
            df_temp['Base_Goals_Liga'] = df_temp['Base_Goals_Liga_y']
            df_temp = df_temp.drop(['Base_Goals_Liga_y'], axis=1)
    
    # Cálculo defensivo simplificado
    df_temp['WG_Def_Home'] = (df_temp['Base_Goals_Liga'] / 2) - df_temp[goals_a_col]
    df_temp['WG_Def_Away'] = (df_temp['Base_Goals_Liga'] / 2) - df_temp[goals_h_col]

    return df_temp

def calcular_rolling_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling features específicas para Over/Under
    """
    df_temp = df.copy()

    # Encontrar colunas de times
    home_col = next((col for col in ['Home', 'HomeTeam', 'home_team'] if col in df_temp.columns), None)
    away_col = next((col for col in ['Away', 'AwayTeam', 'away_team'] if col in df_temp.columns), None)
    goals_h_col = next((col for col in ['Goals_H_FT', 'home_goal', 'Goals_H'] if col in df_temp.columns), None)
    goals_a_col = next((col for col in ['Goals_A_FT', 'away_goal', 'Goals_A'] if col in df_temp.columns), None)

    if not home_col or not away_col or not goals_h_col or not goals_a_col:
        st.warning("⚠️ Colunas de times ou gols não encontradas para rolling features")
        return df_temp

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp = df_temp.sort_values('Date')

    # Rolling de gols marcados e sofridos
    df_temp['Goals_Scored_Home_5G'] = df_temp.groupby(home_col)[goals_h_col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df_temp['Goals_Conceded_Home_5G'] = df_temp.groupby(home_col)[goals_a_col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    df_temp['Goals_Scored_Away_5G'] = df_temp.groupby(away_col)[goals_a_col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df_temp['Goals_Conceded_Away_5G'] = df_temp.groupby(away_col)[goals_h_col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Rolling de Over/Under
    df_temp['Home_Over_Rate_10G'] = df_temp.groupby(home_col).apply(
        lambda x: (x[goals_h_col] + x[goals_a_col] > 2.5).shift(1).rolling(10, min_periods=3).mean()
    ).reset_index(level=0, drop=True)
    
    df_temp['Away_Over_Rate_10G'] = df_temp.groupby(away_col).apply(
        lambda x: (x[goals_h_col] + x[goals_a_col] > 2.5).shift(1).rolling(10, min_periods=3).mean()
    ).reset_index(level=0, drop=True)

    # Expected total goals
    df_temp['Expected_Total_Goals'] = (
        df_temp['Goals_Scored_Home_5G'] + df_temp['Goals_Scored_Away_5G'] +
        df_temp['Goals_Conceded_Home_5G'] + df_temp['Goals_Conceded_Away_5G']
    ) / 2

    return df_temp

# ==========================================================
# FEATURES PARA OVER/UNDER
# ==========================================================
def create_over_under_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features específicas para previsão de Over/Under
    """
    df = df.copy()

    # Features básicas de ataque e defesa
    basic_features = [
        'Goals_Scored_Home_5G', 'Goals_Conceded_Home_5G',
        'Goals_Scored_Away_5G', 'Goals_Conceded_Away_5G',
        'Expected_Total_Goals'
    ]

    # Features de tendência
    trend_features = [
        'Home_Over_Rate_10G', 'Away_Over_Rate_10G'
    ]

    # Features de WG
    wg_features = [
        'WG_Home', 'WG_Away', 'WG_Def_Home', 'WG_Def_Away'
    ]

    # Features de mercado (se disponíveis)
    market_features = []
    if 'Odd_Over25' in df.columns and 'Odd_Under25' in df.columns:
        # Probabilidades implícitas do mercado
        prob_over = 1 / df['Odd_Over25']
        prob_under = 1 / df['Odd_Under25']
        total_prob = prob_over + prob_under
        
        df['Market_Over_Prob'] = prob_over / total_prob
        df['Market_Under_Prob'] = prob_under / total_prob
        df['Market_Bias'] = df['Market_Over_Prob'] - 0.5
        
        market_features = ['Market_Over_Prob', 'Market_Under_Prob', 'Market_Bias']

    # Features de liga
    league_features = []
    if 'Base_Goals_Liga' in df.columns:
        league_features = ['Base_Goals_Liga']

    # Combinar todas as features
    all_features = basic_features + trend_features + wg_features + market_features + league_features
    available_features = [f for f in all_features if f in df.columns]

    st.info(f"📋 Features Over/Under disponíveis: {len(available_features)}/{len(all_features)}")

    return df[available_features].fillna(0)

# ==========================================================
# MODELO MACHINE LEARNING
# ==========================================================
def clean_features_for_training(X):
    X_clean = X.copy()
    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean)
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
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
    
    X_clean = X_clean.fillna(0)
    X_clean = X_clean.replace([np.inf, -np.inf], 0)
    
    return X_clean

def train_over_under_model(X, y, feature_names):
    st.info("🤖 Treinando modelo Over/Under...")
    X_clean = clean_features_for_training(X)
    y_clean = y.copy()

    if hasattr(y_clean, 'isna') and y_clean.isna().any():
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
            st.warning("⚠️ Modelo abaixo do esperado - verificar features")
        elif scores.mean() > 0.65:
            st.success("🎯 Modelo com excelente performance!")
            
    except Exception as e:
        st.warning(f"⚠️ Validação cruzada falhou: {e}")

    model.fit(X_clean, y_clean)

    # Importância das features
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    st.write("🔍 **Top Features para Over/Under:**")
    st.dataframe(importances.head(10).to_frame("Importância"))

    return model

# ==========================================================
# PREDIÇÃO E ANÁLISE DE VALOR
# ==========================================================
def predict_over_under(games_today: pd.DataFrame, model, features: list):
    """
    Faz previsões de Over/Under e calcula valor
    """
    # Garantir mesmas features do treino
    X_today = games_today.reindex(columns=features, fill_value=0)
    
    # Prever probabilidades
    games_today['Model_Over_Prob'] = model.predict_proba(X_today)[:, 1]
    games_today['Model_Under_Prob'] = 1 - games_today['Model_Over_Prob']
    
    # Calcular valor vs odds
    if all(col in games_today.columns for col in ['Odd_Over25', 'Odd_Under25']):
        games_today['Over_Value'] = (games_today['Model_Over_Prob'] * games_today['Odd_Over25']) - 1
        games_today['Under_Value'] = (games_today['Model_Under_Prob'] * games_today['Odd_Under25']) - 1
        
        # Recomendações baseadas em valor
        games_today['Over_Recommended'] = (games_today['Over_Value'] > 0.05) & (games_today['Model_Over_Prob'] > 0.4)
        games_today['Under_Recommended'] = (games_today['Under_Value'] > 0.05) & (games_today['Model_Under_Prob'] > 0.4)
        
        # Força da recomendação
        games_today['Recommendation_Strength'] = np.maximum(
            games_today['Over_Value'], games_today['Under_Value']
        )
        
        # Confidence do modelo
        games_today['Model_Confidence'] = np.abs(games_today['Model_Over_Prob'] - 0.5) * 2
    
    return games_today

# ==========================================================
# VISUALIZAÇÕES
# ==========================================================
def plot_over_under_analysis(games_today: pd.DataFrame):
    """
    Gráficos específicos para Over/Under
    """
    if games_today.empty:
        return
    
    # Encontrar colunas de identificação
    home_col = next((col for col in ['Home', 'HomeTeam', 'home_team'] if col in games_today.columns), 'Team_Home')
    away_col = next((col for col in ['Away', 'AwayTeam', 'away_team'] if col in games_today.columns), 'Team_Away')
    
    # Criar coluna de identificação do jogo
    games_today['Match_Label'] = games_today[home_col].astype(str) + ' vs ' + games_today[away_col].astype(str)
    
    # Gráfico 1: Probabilidades do Modelo vs Mercado
    if all(col in games_today.columns for col in ['Model_Over_Prob', 'Market_Over_Prob']):
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=games_today['Market_Over_Prob'],
            y=games_today['Model_Over_Prob'],
            mode='markers',
            text=games_today['Match_Label'],
            marker=dict(
                size=10,
                color=games_today.get('Recommendation_Strength', 0),
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Valor")
            ),
            name='Jogos'
        ))
        
        # Linha de igualdade
        fig1.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Mercado = Modelo'
        ))
        
        fig1.update_layout(
            title='Modelo vs Mercado - Probabilidades Over 2.5',
            xaxis_title='Probabilidade do Mercado',
            yaxis_title='Probabilidade do Modelo',
            height=500
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Expected Goals vs Probabilidade
    if 'Expected_Total_Goals' in games_today.columns:
        fig2 = go.Figure()
        
        colors = ['red' if x < 2.5 else 'green' for x in games_today['Expected_Total_Goals']]
        
        fig2.add_trace(go.Scatter(
            x=games_today['Expected_Total_Goals'],
            y=games_today['Model_Over_Prob'],
            mode='markers',
            text=games_today['Match_Label'],
            marker=dict(
                size=8,
                color=colors,
                opacity=0.7
            ),
            name='Jogos'
        ))
        
        fig2.add_vline(x=2.5, line_dash="dash", line_color="blue", annotation_text="Linha 2.5")
        fig2.add_hline(y=0.5, line_dash="dash", line_color="blue", annotation_text="50%")
        
        fig2.update_layout(
            title='Expected Goals vs Probabilidade Over',
            xaxis_title='Expected Total Goals',
            yaxis_title='Probabilidade Over 2.5',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

def display_over_under_recommendations(games_today: pd.DataFrame):
    """
    Mostra recomendações de Over/Under
    """
    st.markdown("## 🎯 Recomendações Over/Under 2.5")
    
    # Encontrar colunas de identificação
    home_col = next((col for col in ['Home', 'HomeTeam', 'home_team'] if col in games_today.columns), 'Home')
    away_col = next((col for col in ['Away', 'AwayTeam', 'away_team'] if col in games_today.columns), 'Away')
    league_col = next((col for col in ['League', 'league'] if col in games_today.columns), 'League')
    
    # Colunas para display
    recommendation_columns = [
        league_col, home_col, away_col, 
        'Model_Over_Prob', 'Model_Under_Prob',
        'Odd_Over25', 'Odd_Under25',
        'Over_Value', 'Under_Value',
        'Over_Recommended', 'Under_Recommended',
        'Model_Confidence', 'Expected_Total_Goals'
    ]
    
    available_columns = [c for c in recommendation_columns if c in games_today.columns]
    
    if not available_columns:
        st.warning("Não há dados suficientes para recomendações")
        return
    
    # Separar recomendações
    over_recommended = games_today[games_today['Over_Recommended']].copy()
    under_recommended = games_today[games_today['Under_Recommended']].copy()
    
    # Display Over Recomendados
    if not over_recommended.empty:
        st.subheader("✅ OVER 2.5 Recomendados")
        over_recommended = over_recommended.sort_values('Over_Value', ascending=False)
        st.dataframe(over_recommended[available_columns].head(15))
    
    # Display Under Recomendados
    if not under_recommended.empty:
        st.subheader("✅ UNDER 2.5 Recomendados")
        under_recommended = under_recommended.sort_values('Under_Value', ascending=False)
        st.dataframe(under_recommended[available_columns].head(15))
    
    # Estatísticas
    if not over_recommended.empty or not under_recommended.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Over Recomendados", len(over_recommended))
        with col2:
            st.metric("Under Recomendados", len(under_recommended))
        with col3:
            avg_over_value = over_recommended['Over_Value'].mean() if not over_recommended.empty else 0
            st.metric("Valor Médio Over", f"{avg_over_value:.1%}")
        with col4:
            avg_under_value = under_recommended['Under_Value'].mean() if not under_recommended.empty else 0
            st.metric("Valor Médio Under", f"{avg_under_value:.1%}")
    
    # Todos os jogos para análise
    st.subheader("📊 Todos os Jogos - Análise Completa")
    all_games_sorted = games_today.sort_values('Recommendation_Strength', ascending=False)
    st.dataframe(all_games_sorted[available_columns].head(25))

# ==========================================================
# CARREGAMENTO DE DADOS
# ==========================================================
@st.cache_data(ttl=3600)
def load_cached_data(selected_file: str):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    
    # Carregar histórico
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        selected_date = pd.to_datetime(selected_date_str)
        history = history[history["Date"] < selected_date].copy()
    
    return games_today, history, selected_date_str

# ==========================================================
# FLUXO PRINCIPAL
# ==========================================================
def main():
    st.sidebar.title("⚽ Configurações Over/Under")
    
    # Carregar dados
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
    if not files:
        st.error("❌ Nenhum arquivo encontrado na pasta GamesDay")
        return
    
    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.sidebar.selectbox("Selecione a data:", options, index=len(options)-1)
    
    with st.spinner("Carregando dados..."):
        games_today, history, selected_date_str = load_cached_data(selected_file)
    
    st.success(f"📅 Análise para: {selected_date_str}")
    
    # Mostrar estrutura dos dados
    st.sidebar.subheader("📋 Estrutura dos Dados")
    st.sidebar.write(f"Jogos hoje: {len(games_today)}")
    st.sidebar.write(f"Histórico: {len(history)}")
    st.sidebar.write("Colunas disponíveis:")
    st.sidebar.write(list(games_today.columns) if not games_today.empty else "Nenhuma coluna")
    
    if history.empty or games_today.empty:
        st.error("❌ Dados insuficientes para análise")
        return
    
    # Verificar colunas necessárias
    required_columns = ['Odd_Over25', 'OverLine', 'Odd_Under25']
    missing_columns = [col for col in required_columns if col not in history.columns]
    
    if missing_columns:
        st.error(f"❌ Colunas de Over/Under faltando: {missing_columns}")
        st.info("Colunas disponíveis no histórico:")
        st.write(list(history.columns))
        return
    
    # ==========================================================
    # PROCESSAMENTO DO HISTÓRICO
    # ==========================================================
    st.markdown("## 🔄 Processando Dados Históricos")
    
    with st.spinner("Calculando parâmetros por liga..."):
        liga_params = calcular_parametros_liga_avancado(history)
    
    with st.spinner("Calculando métricas de gols..."):
        # Targets
        history = create_over_under_targets(history)
        
        # Weighted Goals
        history = adicionar_weighted_goals(history)
        history = adicionar_weighted_goals_defensivos(history, liga_params)
        
        # Rolling features
        history = calcular_rolling_goal_features(history)
        
        # Features finais
        X_hist = create_over_under_features(history)
        y_hist = history['Over_2.5']
    
    # ==========================================================
    # TREINAMENTO DO MODELO
    # ==========================================================
    if len(history) > 50:
        with st.spinner("Treinando modelo Over/Under..."):
            model = train_over_under_model(X_hist, y_hist, X_hist.columns.tolist())
        
        # ==========================================================
        # PREDIÇÃO PARA JOGOS DE HOJE
        # ==========================================================
        st.markdown("## 🔮 Previsões para os Jogos de Hoje")
        
        with st.spinner("Processando jogos de hoje..."):
            # Aplicar mesmo processamento aos jogos de hoje
            games_today = create_over_under_targets(games_today)
            games_today = adicionar_weighted_goals(games_today)
            games_today = adicionar_weighted_goals_defensivos(games_today, liga_params)
            games_today = calcular_rolling_goal_features(games_today)
            games_today = create_over_under_features(games_today)
            
            # Fazer previsões
            games_today = predict_over_under(games_today, model, X_hist.columns.tolist())
        
        # ==========================================================
        # RESULTADOS E VISUALIZAÇÕES
        # ==========================================================
        plot_over_under_analysis(games_today)
        display_over_under_recommendations(games_today)
        
    else:
        st.warning("⚠️ Histórico insuficiente para treinar o modelo (mínimo 50 jogos)")

if __name__ == "__main__":
    main()
