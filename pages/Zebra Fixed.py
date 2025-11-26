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
st.set_page_config(page_title="Análise de Quadrantes 3D - Bet Indicator", layout="wide")
st.title("🎯 Análise 3D de 16 Quadrantes - ML + WG GAP")

PAGE_PREFIX = "QuadrantesML_3D"
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
# ASIAN LINE → DECIMAL (HOME)
# ==========================================================
def convert_asian_line_to_decimal_corrigido(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    """
    if pd.isna(line_str) or line_str == "":
        return None

    line_str = str(line_str).strip()

    if line_str in ("0", "0.0"):
        return 0.0

    common_splits = {
        '0/0.5': -0.25,
        '0.5/1': -0.75,
        '1/1.5': -1.25,
        '1.5/2': -1.75,
        '2/2.5': -2.25,
        '2.5/3': -2.75,
        '3/3.5': -3.25,
        '0/-0.5': 0.25,
        '-0.5/-1': 0.75,
        '-1/-1.5': 1.25,
        '-1.5/-2': 1.75,
        '-2/-2.5': 2.25,
        '-2.5/-3': 2.75,
        '-3/-3.5': 3.25,
        '0.75': -0.75,
        '-0.75': 0.75,
        '0.25': -0.25,
        '-0.25': 0.25,
    }

    if line_str in common_splits:
        return common_splits[line_str]

    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num
        except ValueError:
            return None

    try:
        parts = [float(p) for p in line_str.split("/")]
        avg = sum(parts) / len(parts)
        first_part = parts[0]
        if first_part < 0:
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result
    except (ValueError, TypeError):
        st.warning(f"⚠️ Split handicap não reconhecido: {line_str}")
        return None

# ==========================================================
# AVALIAÇÃO DO RESULTADO DO HANDICAP
# ==========================================================
def _single_leg_home(margin, line):
    adj = margin + line
    # linhas inteiras / meia
    if abs(line * 2) % 2 == 0:
        if adj > 0:
            return 1.0
        elif abs(adj) < 1e-9:
            return 0.5
        else:
            return 0.0
    else:
        # linhas de quarto
        return 1.0 if adj > 0 else 0.0

def calc_handicap_result_corrigido(margin, asian_line_decimal):
    if pd.isna(margin) or pd.isna(asian_line_decimal):
        return np.nan

    line = float(asian_line_decimal)

    # split lines (ex: -0.75, +0.25, etc)
    if abs(line * 2) % 1 != 0:
        sign = 1 if line > 0 else -1
        base = abs(line)
        lower = math.floor(base * 2) / 2.0
        upper = math.ceil(base * 2) / 2.0
        l1 = sign * lower
        l2 = sign * upper
        r1 = _single_leg_home(margin, l1)
        r2 = _single_leg_home(margin, l2)
        return 0.5 * (r1 + r2)
    else:
        return _single_leg_home(margin, line)

# ==========================================================
# Z-SCORES (M_H, M_A, MT_H, MT_A)
# ==========================================================
def calcular_zscores_detalhados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    st.info("📊 Calculando Z-scores a partir do HandScore...")

    if 'League' in df.columns and 'HandScore_Home' in df.columns and 'HandScore_Away' in df.columns:
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
        st.warning("⚠️ Colunas League ou HandScore não encontradas")
        df['M_H'] = 0
        df['M_A'] = 0

    if 'Home' in df.columns and 'Away' in df.columns:
        home_team_stats = df.groupby('Home').agg({'HandScore_Home': ['mean', 'std']}).round(3)
        home_team_stats.columns = ['HT_mean', 'HT_std']
        away_team_stats = df.groupby('Away').agg({'HandScore_Away': ['mean', 'std']}).round(3)
        away_team_stats.columns = ['AT_mean', 'AT_std']
        home_team_stats['HT_std'] = home_team_stats['HT_std'].replace(0, 1)
        away_team_stats['AT_std'] = away_team_stats['AT_std'].replace(0, 1)
        df = df.merge(home_team_stats, left_on='Home', right_index=True, how='left')
        df = df.merge(away_team_stats, left_on='Away', right_index=True, how='left')
        df['MT_H'] = (df['HandScore_Home'] - df['HT_mean']) / df['HT_std']
        df['MT_A'] = (df['HandScore_Away'] - df['AT_mean']) / df['AT_std']
        df['MT_H'] = np.clip(df['MT_H'], -5, 5)
        df['MT_A'] = np.clip(df['MT_A'], -5, 5)
        st.success(f"✅ Z-score por time calculado para {len(df)} jogos")
        df = df.drop(['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std',
                      'HT_mean', 'HT_std', 'AT_mean', 'AT_std'], axis=1, errors='ignore')
    else:
        st.warning("⚠️ Colunas Home ou Away não encontradas")
        df['MT_H'] = 0
        df['MT_A'] = 0

    return df

# ==========================================================
# LIMPEZA DAS FEATURES
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
# LOAD + FILTER HISTORY (TIME-SAFE)
# ==========================================================
def load_and_filter_history(selected_date_str: str) -> pd.DataFrame:
    """Carrega histórico APENAS com jogos anteriores à data selecionada - sem perder linhas"""
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

    # Conversão de Asian Line
    if "Asian_Line" in history.columns:
        history["Asian_Line_Decimal"] = history["Asian_Line"].apply(convert_asian_line_to_decimal_corrigido)
    else:
        history["Asian_Line_Decimal"] = np.nan

    # Preencher quaisquer NaNs com 0 APÓS o load
    history = history.fillna(0)

    st.success(f"✅ Histórico processado: {len(history)} jogos (sem drop de linhas)")
    return history

# ==========================================================
# CRIAÇÃO DO TARGET (AH_HOME + AH_AWAY + ZEBRA)
# ==========================================================
def create_better_target_corrigido(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Margin"] = df["Goals_H_FT"] - df["Goals_A_FT"]
    df["AH_Result"] = df.apply(
        lambda r: calc_handicap_result_corrigido(
            r["Margin"], r["Asian_Line_Decimal"]
        ),
        axis=1
    )

    total = len(df)
    # remove push (0.5 = push)
    df = df[df["AH_Result"] != 0.5].copy()
    clean = len(df)

    # Targets
    df["Target_AH_Home"] = (df["AH_Result"] > 0.5).astype(int)
    df["Target_AH_Away"] = (df["AH_Result"] < 0.5).astype(int)

    df["Expected_Favorite"] = np.where(
        df["Asian_Line_Decimal"] < 0,
        "HOME",
        np.where(df["Asian_Line_Decimal"] > 0, "AWAY", "NONE")
    )

    df["Zebra"] = np.where(
        (
            (df["Expected_Favorite"] == "HOME") & (df["Target_AH_Home"] == 0)
        ) |
        (
            (df["Expected_Favorite"] == "AWAY") & (df["Target_AH_Home"] == 1)
        ),
        1,
        0
    )

    win_rate_home = df["Target_AH_Home"].mean() if len(df) > 0 else 0.0
    win_rate_away = df["Target_AH_Away"].mean() if len(df) > 0 else 0.0
    zebra_rate = df["Zebra"].mean() if len(df) > 0 else 0.0

    st.info(f"🎯 Total analisado: {total} jogos")
    st.info(f"🗑️ Excluídos por Push puro: {total-clean} jogos ({(total-clean)/total:.1%})")
    st.info(f"📊 Treino com: {clean} jogos restantes")
    st.info(f"🏠 Win rate HOME cobrindo: {win_rate_home:.1%}")
    st.info(f"🌍 Win rate AWAY cobrindo: {win_rate_away:.1%}")
    st.info(f"🦓 Taxa de Zebra (favorito falhou): {zebra_rate:.1%}")
    return df

# ==========================================================
# PARÂMETROS POR LIGA (BASE GOLS + PESO ASIAN) - ATUALIZADO!
# ==========================================================
@st.cache_data(ttl=7*24*3600)
def calcular_parametros_liga_avancado(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula parâmetros específicos por liga de forma robusta
    """
    if history.empty:
        return pd.DataFrame()
    
    df = history.copy()
    
    # Garantir que temos as colunas necessárias
    required_cols = ['League', 'Goals_H_FT', 'Goals_A_FT', 'Asian_Line_Decimal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo de parâmetros: {missing_cols}")
        return pd.DataFrame()
    
    # Calcular estatísticas por liga
    liga_stats = df.groupby('League').agg({
        'Goals_H_FT': ['count', 'mean'],
        'Goals_A_FT': 'mean',
        'Asian_Line_Decimal': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    liga_stats.columns = [
        'Jogos_Total', 'Gols_Media_Casa', 
        'Gols_Media_Fora', 'Asian_Line_Media', 'Asian_Line_Std'
    ]
    
    # Calcular Base_Goals_Liga (gols totais médios)
    liga_stats['Base_Goals_Liga'] = (
        liga_stats['Gols_Media_Casa'] + liga_stats['Gols_Media_Fora']
    ).round(2)
    
    # Calcular Asian_Weight_Liga baseado na variabilidade do handicap
    # Ligas com handicaps mais variáveis → peso maior
    if not liga_stats['Asian_Line_Std'].isna().all():
        asi_std_min = liga_stats['Asian_Line_Std'].min()
        asi_std_max = liga_stats['Asian_Line_Std'].max()
        
        if asi_std_max > asi_std_min:
            # Normalizar entre 0.4 e 0.8 baseado na variabilidade
            liga_stats['Asian_Weight_Liga'] = 0.4 + 0.4 * (
                (liga_stats['Asian_Line_Std'] - asi_std_min) / 
                (asi_std_max - asi_std_min)
            )
        else:
            liga_stats['Asian_Weight_Liga'] = 0.6  # default se não há variação
    else:
        liga_stats['Asian_Weight_Liga'] = 0.6
    
    liga_stats['Asian_Weight_Liga'] = liga_stats['Asian_Weight_Liga'].round(3)
    
    # Filtrar ligas com poucos jogos (menos confiáveis)
    liga_stats = liga_stats[liga_stats['Jogos_Total'] >= 10].copy()
    
    st.success(f"✅ Parâmetros calculados para {len(liga_stats)} ligas")
    
    return liga_stats.reset_index()

def mostrar_parametros_ligas(liga_params: pd.DataFrame):
    """Mostra os parâmetros calculados por liga"""
    if liga_params.empty:
        return
    
    st.markdown("### 📊 Parâmetros por Liga Calculados")
    
    # Ordenar por número de jogos para confiabilidade
    display_params = liga_params.sort_values('Jogos_Total', ascending=False)[[
        'League', 'Jogos_Total', 'Base_Goals_Liga', 'Asian_Weight_Liga',
        'Gols_Media_Casa', 'Gols_Media_Fora', 'Asian_Line_Media'
    ]]
    
    st.dataframe(display_params.head(15))
    
    # Estatísticas sumarizadas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ligas Analisadas", len(liga_params))
    with col2:
        st.metric("Avg Base Goals", f"{liga_params['Base_Goals_Liga'].mean():.2f}")
    with col3:
        st.metric("Avg Asian Weight", f"{liga_params['Asian_Weight_Liga'].mean():.3f}")

# ==========================================================
# WEIGHTED GOALS (OFENSIVO / DEFENSIVO / AH / ROLLING) - ATUALIZADO!
# ==========================================================
def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    WG_Home / WG_Away ofensivos baseado em:
      - Gols marcados / sofridos (FULL TIME)
      - Odds 1x2 (Odd_H, Odd_D, Odd_A)
    """
    df_temp = df.copy()

    for col in ['Goals_H_FT', 'Goals_A_FT']:
        if col not in df_temp.columns:
            df_temp[col] = np.nan

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
        gh = row.get('Goals_H_FT', np.nan)
        ga = row.get('Goals_A_FT', np.nan)
        if pd.isna(gh) or pd.isna(ga):
            return np.nan

        p_h, p_d, p_a = odds_to_probs_1x2(row)
        weight_for = (1 - p_h) + 0.5 * p_d
        weight_against = p_h + 0.5 * p_d
        return gh * weight_for - ga * weight_against

    def wg_away(row):
        gh = row.get('Goals_H_FT', np.nan)
        ga = row.get('Goals_A_FT', np.nan)
        if pd.isna(gh) or pd.isna(ga):
            return np.nan

        p_h, p_d, p_a = odds_to_probs_1x2(row)
        weight_for = (1 - p_a) + 0.5 * p_d
        weight_against = p_a + 0.5 * p_d
        return ga * weight_for - gh * weight_against

    df_temp['WG_Home'] = df_temp.apply(wg_home, axis=1)
    df_temp['WG_Away'] = df_temp.apply(wg_away, axis=1)

    return df_temp

def adicionar_weighted_goals_defensivos_corrigido(df: pd.DataFrame, liga_params: pd.DataFrame) -> pd.DataFrame:
    """
    Versão corrigida usando parâmetros específicos por liga
    """
    df_temp = df.copy()
    
    # Garantir colunas básicas
    df_temp['Goals_H_FT'] = df_temp.get('Goals_H_FT', df_temp.get('Goals_H_Today', 0))
    df_temp['Goals_A_FT'] = df_temp.get('Goals_A_FT', df_temp.get('Goals_A_Today', 0))
    
    # Se não temos parâmetros por liga, calcular defaults globais
    if liga_params is None or liga_params.empty:
        st.warning("⚠️ Sem parâmetros por liga - usando defaults globais")
        global_base_goals = df_temp['Goals_H_FT'].mean() + df_temp['Goals_A_FT'].mean() if not df_temp.empty else 2.5
        global_asian_weight = 0.6
        
        df_temp['Base_Goals_Liga'] = global_base_goals
        df_temp['Asian_Weight_Liga'] = global_asian_weight
    else:
        # Merge com parâmetros por liga
        df_temp = df_temp.merge(
            liga_params[['League', 'Base_Goals_Liga', 'Asian_Weight_Liga']],
            on='League',
            how='left'
        )
        
        # Preencher missing values com médias globais
        global_base = liga_params['Base_Goals_Liga'].mean() if not liga_params.empty else 2.5
        global_asian = liga_params['Asian_Weight_Liga'].mean() if not liga_params.empty else 0.6
        
        df_temp['Base_Goals_Liga'] = df_temp['Base_Goals_Liga'].fillna(global_base)
        df_temp['Asian_Weight_Liga'] = df_temp['Asian_Weight_Liga'].fillna(global_asian)
    
    # Cálculo dos expected goals
    if 'Asian_Line_Decimal' in df_temp.columns:
        df_temp['xGF_H'] = (df_temp['Base_Goals_Liga'] / 2) + df_temp['Asian_Line_Decimal'] * df_temp['Asian_Weight_Liga']
        df_temp['xGF_A'] = (df_temp['Base_Goals_Liga'] / 2) - df_temp['Asian_Line_Decimal'] * df_temp['Asian_Weight_Liga']
        
        df_temp['xGA_H'] = df_temp['xGF_A']  # Gols esperados sofridos pelo Home
        df_temp['xGA_A'] = df_temp['xGF_H']  # Gols esperados sofridos pelo Away
        
        # WG Defensivo = quanto a defesa performou MELHOR que o esperado
        df_temp['WG_Def_Home'] = df_temp['xGA_H'] - df_temp['Goals_A_FT']  # Positive = boa defesa
        df_temp['WG_Def_Away'] = df_temp['xGA_A'] - df_temp['Goals_H_FT']  # Positive = boa defesa
    else:
        df_temp['WG_Def_Home'] = 0.0
        df_temp['WG_Def_Away'] = 0.0
    
    # Limpeza
    cols_to_drop = ['xGF_H', 'xGF_A', 'xGA_H', 'xGA_A']
    df_temp.drop(columns=[c for c in cols_to_drop if c in df_temp.columns], inplace=True)
    
    return df_temp

def adicionar_weighted_goals_ah(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()
    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_AH_Home'] = 0.0
        df_temp['WG_AH_Away'] = 0.0
        return df_temp

    for col in ['WG_Home', 'WG_Away']:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    fator = 1 + df_temp['Asian_Line_Decimal'].abs()
    df_temp['WG_AH_Home'] = df_temp['WG_Home'] * fator
    df_temp['WG_AH_Away'] = df_temp['WG_Away'] * fator

    return df_temp

def adicionar_weighted_goals_ah_defensivos(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()
    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_AH_Def_Home'] = 0.0
        df_temp['WG_AH_Def_Away'] = 0.0
        return df_temp

    for col in ['WG_Def_Home', 'WG_Def_Away']:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    fator = 1 + df_temp['Asian_Line_Decimal'].abs()
    df_temp['WG_AH_Def_Home'] = df_temp['WG_Def_Home'] * fator
    df_temp['WG_AH_Def_Away'] = df_temp['WG_Def_Away'] * fator

    return df_temp

def calcular_metricas_completas(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()

    for col in [
        'WG_Home', 'WG_Away',
        'WG_Def_Home', 'WG_Def_Away'
    ]:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    df_temp['WG_Balance_Home'] = df_temp['WG_Home'] + df_temp['WG_Def_Home']
    df_temp['WG_Balance_Away'] = df_temp['WG_Away'] + df_temp['WG_Def_Away']

    df_temp['WG_Total_Home'] = df_temp['WG_Balance_Home']
    df_temp['WG_Total_Away'] = df_temp['WG_Balance_Away']

    df_temp['WG_Net_Home'] = df_temp['WG_Home'] - df_temp['WG_Def_Away']
    df_temp['WG_Net_Away'] = df_temp['WG_Away'] - df_temp['WG_Def_Home']

    return df_temp

def calcular_rolling_wg_features_completo(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp = df_temp.sort_values('Date')

    for col in [
        'WG_Home', 'WG_Away',
        'WG_AH_Home', 'WG_AH_Away',
        'WG_Def_Home', 'WG_Def_Away',
        'WG_AH_Def_Home', 'WG_AH_Def_Away',
        'WG_Balance_Home', 'WG_Balance_Away',
        'WG_Total_Home', 'WG_Total_Away',
        'WG_Net_Home', 'WG_Net_Away'
    ]:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    roll = lambda x: x.shift(1).rolling(3, min_periods=1).mean()

    df_temp['WG_Home_Team'] = df_temp.groupby('Home')['WG_Home'].transform(roll)
    df_temp['WG_Away_Team'] = df_temp.groupby('Away')['WG_Away'].transform(roll)

    df_temp['WG_AH_Home_Team'] = df_temp.groupby('Home')['WG_AH_Home'].transform(roll)
    df_temp['WG_AH_Away_Team'] = df_temp.groupby('Away')['WG_AH_Away'].transform(roll)

    df_temp['WG_Def_Home_Team'] = df_temp.groupby('Home')['WG_Def_Home'].transform(roll)
    df_temp['WG_Def_Away_Team'] = df_temp.groupby('Away')['WG_Def_Away'].transform(roll)

    df_temp['WG_AH_Def_Home_Team'] = df_temp.groupby('Home')['WG_AH_Def_Home'].transform(roll)
    df_temp['WG_AH_Def_Away_Team'] = df_temp.groupby('Away')['WG_AH_Def_Away'].transform(roll)

    df_temp['WG_Balance_Home_Team'] = df_temp.groupby('Home')['WG_Balance_Home'].transform(roll)
    df_temp['WG_Balance_Away_Team'] = df_temp.groupby('Away')['WG_Balance_Away'].transform(roll)

    df_temp['WG_Total_Home_Team'] = df_temp.groupby('Home')['WG_Total_Home'].transform(roll)
    df_temp['WG_Total_Away_Team'] = df_temp.groupby('Away')['WG_Total_Away'].transform(roll)

    df_temp['WG_Net_Home_Team'] = df_temp.groupby('Home')['WG_Net_Home'].transform(roll)
    df_temp['WG_Net_Away_Team'] = df_temp.groupby('Away')['WG_Net_Away'].transform(roll)

    df_temp['WG_Diff'] = df_temp['WG_Home_Team'] - df_temp['WG_Away_Team']
    df_temp['WG_AH_Diff'] = df_temp['WG_AH_Home_Team'] - df_temp['WG_AH_Away_Team']
    df_temp['WG_Def_Diff'] = df_temp['WG_Def_Home_Team'] - df_temp['WG_Def_Away_Team']
    df_temp['WG_Balance_Diff'] = df_temp['WG_Balance_Home_Team'] - df_temp['WG_Balance_Away_Team']
    df_temp['WG_Net_Diff'] = df_temp['WG_Net_Home_Team'] - df_temp['WG_Net_Away_Team']

    df_temp['WG_Confidence'] = (
        df_temp['WG_Home_Team'].notna().astype(int) +
        df_temp['WG_Away_Team'].notna().astype(int) +
        df_temp['WG_Def_Home_Team'].notna().astype(int) +
        df_temp['WG_Def_Away_Team'].notna().astype(int)
    )

    return df_temp

def enrich_games_today_with_wg_completo(games_today: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or games_today.empty:
        return games_today

    last_wg_home = history.groupby('Home').agg({
        'WG_Home_Team': 'last',
        'WG_AH_Home_Team': 'last',
        'WG_Def_Home_Team': 'last',
        'WG_AH_Def_Home_Team': 'last',
        'WG_Balance_Home_Team': 'last',
        'WG_Total_Home_Team': 'last',
        'WG_Net_Home_Team': 'last'
    }).reset_index().rename(columns={
        'Home': 'Team',
        'WG_Home_Team': 'WG_Home_Team_Last',
        'WG_AH_Home_Team': 'WG_AH_Home_Team_Last',
        'WG_Def_Home_Team': 'WG_Def_Home_Team_Last',
        'WG_AH_Def_Home_Team': 'WG_AH_Def_Home_Team_Last',
        'WG_Balance_Home_Team': 'WG_Balance_Home_Team_Last',
        'WG_Total_Home_Team': 'WG_Total_Home_Team_Last',
        'WG_Net_Home_Team': 'WG_Net_Home_Team_Last'
    })

    last_wg_away = history.groupby('Away').agg({
        'WG_Away_Team': 'last',
        'WG_AH_Away_Team': 'last',
        'WG_Def_Away_Team': 'last',
        'WG_AH_Def_Away_Team': 'last',
        'WG_Balance_Away_Team': 'last',
        'WG_Total_Away_Team': 'last',
        'WG_Net_Away_Team': 'last'
    }).reset_index().rename(columns={
        'Away': 'Team',
        'WG_Away_Team': 'WG_Away_Team_Last',
        'WG_AH_Away_Team': 'WG_AH_Away_Team_Last',
        'WG_Def_Away_Team': 'WG_Def_Away_Team_Last',
        'WG_AH_Def_Away_Team': 'WG_AH_Def_Away_Team_Last',
        'WG_Balance_Away_Team': 'WG_Balance_Away_Team_Last',
        'WG_Total_Away_Team': 'WG_Total_Away_Team_Last',
        'WG_Net_Away_Team': 'WG_Net_Away_Team_Last'
    })

    games_today = games_today.merge(
        last_wg_home, left_on='Home', right_on='Team', how='left'
    ).drop('Team', axis=1)

    games_today = games_today.merge(
        last_wg_away, left_on='Away', right_on='Team', how='left'
    ).drop('Team', axis=1)

    wg_cols = [
        'WG_Home_Team_Last', 'WG_AH_Home_Team_Last', 'WG_Def_Home_Team_Last', 'WG_AH_Def_Home_Team_Last',
        'WG_Balance_Home_Team_Last', 'WG_Total_Home_Team_Last', 'WG_Net_Home_Team_Last',
        'WG_Away_Team_Last', 'WG_AH_Away_Team_Last', 'WG_Def_Away_Team_Last', 'WG_AH_Def_Away_Team_Last',
        'WG_Balance_Away_Team_Last', 'WG_Total_Away_Team_Last', 'WG_Net_Away_Team_Last'
    ]

    for col in wg_cols:
        if col in games_today.columns:
            games_today[col] = games_today[col].fillna(0.0)
        else:
            games_today[col] = 0.0

    games_today['WG_Diff'] = games_today['WG_Home_Team_Last'] - games_today['WG_Away_Team_Last']
    games_today['WG_AH_Diff'] = games_today['WG_AH_Home_Team_Last'] - games_today['WG_AH_Away_Team_Last']
    games_today['WG_Def_Diff'] = games_today['WG_Def_Home_Team_Last'] - games_today['WG_Def_Away_Team_Last']
    games_today['WG_Balance_Diff'] = games_today['WG_Balance_Home_Team_Last'] - games_today['WG_Balance_Away_Team_Last']
    games_today['WG_Net_Diff'] = games_today['WG_Net_Home_Team_Last'] - games_today['WG_Net_Away_Team_Last']

    games_today['WG_Confidence'] = (
        games_today['WG_Home_Team_Last'].notna().astype(int) +
        games_today['WG_Away_Team_Last'].notna().astype(int) +
        games_today['WG_Def_Home_Team_Last'].notna().astype(int) +
        games_today['WG_Def_Away_Team_Last'].notna().astype(int)
    )

    return games_today

# ==========================================================
# FEATURE SET FINAL (INCLUDING WG + GES + 3D)
# ==========================================================
def create_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    basic_features = [
        'Aggression_Home', 'Aggression_Away',
        'M_H', 'M_A', 'MT_H', 'MT_A'
    ]

    if 'Aggression_Home' in df.columns and 'Aggression_Away' in df.columns:
        df['Aggression_Diff'] = df['Aggression_Home'] - df['Aggression_Away']
        df['Aggression_Total'] = df['Aggression_Home'] + df['Aggression_Away']
    if 'M_H' in df.columns and 'M_A' in df.columns:
        df['M_Total'] = df['M_H'] + df['M_A']
        df['Momentum_Advantage'] = (df['M_H'] - df['M_A'])
    if 'MT_H' in df.columns and 'MT_A' in df.columns:
        df['MT_Total'] = df['MT_H'] + df['MT_A']
        if 'Momentum_Advantage' in df.columns:
            df['Momentum_Advantage'] = df['Momentum_Advantage'] + (df['MT_H'] - df['MT_A'])
        else:
            df['Momentum_Advantage'] = (df['MT_H'] - df['MT_A'])

    derived_features = [
        'Aggression_Diff', 'M_Total', 'MT_Total',
        'Momentum_Advantage', 'Aggression_Total'
    ]

    vector_features = [
        'Quadrant_Dist_3D', 'Momentum_Diff', 'Magnitude_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ'
    ]

    wg_features = [
        'WG_Home_Team_Last', 'WG_Away_Team_Last', 'WG_Diff',
        'WG_AH_Home_Team_Last', 'WG_AH_Away_Team_Last', 'WG_AH_Diff',
        'WG_Def_Home_Team_Last', 'WG_Def_Away_Team_Last', 'WG_Def_Diff',
        'WG_Balance_Home_Team_Last', 'WG_Balance_Away_Team_Last', 'WG_Balance_Diff',
        'WG_Net_Home_Team_Last', 'WG_Net_Away_Team_Last', 'WG_Net_Diff',
        'WG_Confidence',
        # 🚀 novo gap 2D
        'WG_Dist_2D'
    ]

    ges_features = [
        'GES_Of_H_Roll', 'GES_Of_A_Roll', 'GES_Of_Diff',
        'GES_Def_H_Roll', 'GES_Def_A_Roll', 'GES_Def_Diff',
        'GES_Total_Diff'
    ]

    all_features = basic_features + derived_features + vector_features + wg_features + ges_features
    available_features = [f for f in all_features if f in df.columns]

    st.info(f"📋 Features disponíveis para ML: {len(available_features)}/{len(all_features)}")

    trig_features = [f for f in available_features if 'Sin' in f or 'Cos' in f]
    if trig_features:
        st.success(f"✅ Features trigonométricas incluídas: {len(trig_features)}")

    return df[available_features].fillna(0)

# ==========================================================
# TREINO DO MODELO
# ==========================================================
def train_improved_model(X, y, feature_names):
    st.info("🤖 Treinando modelo otimizado...")
    X_clean = clean_features_for_training(X)
    y_clean = y.copy()

    if hasattr(y_clean, 'isna') and y_clean.isna().any():
        st.warning(f"⚠️ Encontrados {y_clean.isna().sum()} NaNs no target - removendo")
        valid_mask = ~y_clean.isna()
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    try:
        scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='accuracy')
        st.write(f"📊 Validação Cruzada: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        if scores.mean() < 0.55:
            st.warning("⚠️ Modelo abaixo do esperado - verificar qualidade dos dados")
        elif scores.mean() > 0.65:
            st.success("🎯 Modelo com boa performance!")
    except Exception as e:
        st.warning(f"⚠️ Validação cruzada falhou: {e}")

    model.fit(X_clean, y_clean)

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    st.write("🔍 **Top Features mais importantes:**")
    st.dataframe(importances.head(10).to_frame("Importância"))

    return model

# ==========================================================
# GOAL EFFICIENCY SCORE (GES)
# ==========================================================
def adicionar_goal_efficiency_score(df: pd.DataFrame, liga_params: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Goals_H_FT' not in df.columns:
        df['Goals_H_FT'] = df.get('Goals_H_Today', np.nan)
    if 'Goals_A_FT' not in df.columns:
        df['Goals_A_FT'] = df.get('Goals_A_Today', np.nan)

    if liga_params is None or liga_params.empty:
        st.warning("⚠️ Liga params ausente para GES - usando defaults")
        df['GES_H'] = df['Goals_H_FT'] - 1.25
        df['GES_A'] = df['Goals_A_FT'] - 1.25
        return df

    df = df.merge(
        liga_params[['League', 'Base_Goals_Liga', 'Asian_Weight_Liga']],
        on='League',
        how='left'
    )

    df['Base_Goals_Liga'] = df['Base_Goals_Liga'].fillna(2.5)
    df['Asian_Weight_Liga'] = df['Asian_Weight_Liga'].fillna(0.6)

    df['xGF_H'] = (df['Base_Goals_Liga'] / 2) + df['Asian_Line_Decimal'] * df['Asian_Weight_Liga']
    df['xGF_A'] = (df['Base_Goals_Liga'] / 2) - df['Asian_Line_Decimal'] * df['Asian_Weight_Liga']

    df['GES_Of_H'] = df['Goals_H_FT'] - df['xGF_H']
    df['GES_Of_A'] = df['Goals_A_FT'] - df['xGF_A']

    df['GES_Def_H'] = df['xGF_A'] - df['Goals_A_FT']
    df['GES_Def_A'] = df['xGF_H'] - df['Goals_H_FT']

    for col in ['GES_Of_H', 'GES_Of_A', 'GES_Def_H', 'GES_Def_A']:
        liga_mean = df.groupby('League')[col].transform('mean')
        liga_std = df.groupby('League')[col].transform('std').replace(0, 1)
        df[col + '_Norm'] = (df[col] - liga_mean) / liga_std
        df[col + '_Norm'] = df[col + '_Norm'].clip(-5, 5)

    df['GES_Of_H_Norm'] = df['GES_Of_H_Norm'].fillna(0)
    df['GES_Of_A_Norm'] = df['GES_Of_A_Norm'].fillna(0)
    df['GES_Def_H_Norm'] = df['GES_Def_H_Norm'].fillna(0)
    df['GES_Def_A_Norm'] = df['GES_Def_A_Norm'].fillna(0)

    return df

def calcular_rolling_ges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

    df['GES_Of_H_Roll'] = df.groupby('Home')['GES_Of_H_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['GES_Def_H_Roll'] = df.groupby('Home')['GES_Def_H_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    df['GES_Of_A_Roll'] = df.groupby('Away')['GES_Of_A_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['GES_Def_A_Roll'] = df.groupby('Away')['GES_Def_A_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    df[['GES_Of_H_Roll', 'GES_Of_A_Roll',
        'GES_Def_H_Roll', 'GES_Def_A_Roll']] = df[[
        'GES_Of_H_Roll', 'GES_Of_A_Roll',
        'GES_Def_H_Roll', 'GES_Def_A_Roll'
    ]].fillna(0)

    df['GES_Of_Diff'] = df['GES_Of_H_Roll'] - df['GES_Of_A_Roll']
    df['GES_Def_Diff'] = df['GES_Def_H_Roll'] - df['GES_Def_A_Roll']
    df['GES_Total_Diff'] = (df['GES_Of_Diff'] * 0.60) + (df['GES_Def_Diff'] * 0.40)

    return df

# ==========================================================
# GRÁFICO WG X WG_DEF (2D)
# ==========================================================
def plot_wg_vs_wgdef_scatter_interactive(games_today: pd.DataFrame):
    if games_today.empty:
        st.info("Sem jogos para exibir no gráfico WG x WG_Def.")
        return

    required_cols = [
        'League', 'Home', 'Away',
        'WG_Home_Team_Last', 'WG_Away_Team_Last',
        'WG_Def_Home_Team_Last', 'WG_Def_Away_Team_Last'
    ]
    missing = [c for c in required_cols if c not in games_today.columns]
    if missing:
        st.warning(f"Não é possível gerar o gráfico WG x WG_Def. Faltam colunas: {missing}")
        return

    st.markdown("## 📊 Mapa 2D – WG Ofensivo x WG Defensivo (Histórico por Time)")

    ligas_disponiveis = sorted(games_today['League'].dropna().unique().tolist())

    if 'wg_ligas_aplicadas' not in st.session_state:
        st.session_state['wg_ligas_aplicadas'] = ligas_disponiveis

    defaults_filtrados = [
        l for l in st.session_state['wg_ligas_aplicadas']
        if l in ligas_disponiveis
    ]
    if not defaults_filtrados:
        defaults_filtrados = ligas_disponiveis

    ligas_temp = st.multiselect(
        "Selecione as ligas (confirme depois em 'Aplicar filtros')",
        options=ligas_disponiveis,
        default=defaults_filtrados
    )

    if st.button("Aplicar filtros de ligas (WG)", type="primary"):
        if ligas_temp:
            st.session_state['wg_ligas_aplicadas'] = ligas_temp

    ligas_usadas = st.session_state['wg_ligas_aplicadas']

    max_jogos = st.slider(
        "Quantidade máxima de jogos exibidos no gráfico:",
        min_value=5,
        max_value=50,
        value=30,
        step=1
    )

    df_plot = games_today[games_today['League'].isin(ligas_usadas)].copy()

    if df_plot.empty:
        st.info("Nenhum jogo encontrado para as ligas selecionadas.")
        return

    df_plot['WG_Diff_Grafico'] = (
        df_plot['WG_Home_Team_Last'] - df_plot['WG_Away_Team_Last']
    ).abs()
    df_plot = df_plot.sort_values('WG_Diff_Grafico', ascending=False).head(max_jogos)

    def insight_wg(wg_of, wg_def):
        if wg_of > 0 and wg_def > 0:
            return "Ataque forte & defesa sólida"
        elif wg_of > 0 and wg_def <= 0:
            return "Ataque forte & defesa vulnerável"
        elif wg_of <= 0 and wg_def > 0:
            return "Ataque fraco & defesa sólida"
        else:
            return "Ataque fraco & defesa vulnerável"

    home_x = df_plot['WG_Home_Team_Last'].values
    home_y = df_plot['WG_Def_Home_Team_Last'].values

    away_x = df_plot['WG_Away_Team_Last'].values
    away_y = df_plot['WG_Def_Away_Team_Last'].values

    home_customdata = np.stack([
        df_plot['Home'].astype(str).values,
        df_plot['Away'].astype(str).values,
        df_plot['League'].astype(str).values,
        df_plot['WG_Home_Team_Last'].values,
        df_plot['WG_Def_Home_Team_Last'].values,
        df_plot.apply(lambda r: insight_wg(
            r['WG_Home_Team_Last'], r['WG_Def_Home_Team_Last']
        ), axis=1).astype(str).values
    ], axis=-1)

    away_customdata = np.stack([
        df_plot['Home'].astype(str).values,
        df_plot['Away'].astype(str).values,
        df_plot['League'].astype(str).values,
        df_plot['WG_Away_Team_Last'].values,
        df_plot['WG_Def_Away_Team_Last'].values,
        df_plot.apply(lambda r: insight_wg(
            r['WG_Away_Team_Last'], r['WG_Def_Away_Team_Last']
        ), axis=1).astype(str).values
    ], axis=-1)

    line_traces = []
    for _, r in df_plot.iterrows():
        line_traces.append(
            go.Scatter(
                x=[r['WG_Home_Team_Last'], r['WG_Away_Team_Last']],
                y=[r['WG_Def_Home_Team_Last'], r['WG_Def_Away_Team_Last']],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.7)", width=1),
                hoverinfo="skip",
                showlegend=False
            )
        )

    home_text = df_plot['Home'].astype(str).values
    away_text = df_plot['Away'].astype(str).values

    trace_home = go.Scatter(
        x=home_x, y=home_y,
        mode="markers+text",
        name="Home",
        text=home_text,
        textposition="top right",
        marker=dict(size=9, color="blue"),
        customdata=home_customdata,
        hovertemplate=(
            "<b>%{customdata[0]} x %{customdata[1]}</b><br>"
            "Liga: %{customdata[2]}<br>"
            "WG Ofensivo: %{customdata[3]:.3f}<br>"
            "WG Defensivo: %{customdata[4]:.3f}<br>"
            "Insight: %{customdata[5]}<extra></extra>"
        )
    )

    trace_away = go.Scatter(
        x=away_x, y=away_y,
        mode="markers+text",
        name="Away",
        text=away_text,
        textposition="bottom left",
        marker=dict(size=9, color="orange"),
        customdata=away_customdata,
        hovertemplate=(
            "<b>%{customdata[0]} x %{customdata[1]}</b><br>"
            "Liga: %{customdata[2]}<br>"
            "WG Ofensivo: %{customdata[3]:.3f}<br>"
            "WG Defensivo: %{customdata[4]:.3f}<br>"
            "Insight: %{customdata[5]}<extra></extra>"
        )
    )

    fig = go.Figure()

    for lt in line_traces:
        fig.add_trace(lt)

    fig.add_trace(trace_home)
    fig.add_trace(trace_away)

    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"))

    fig.update_layout(
        title="WG x WG_Def – Comparação Home (azul) vs Away (laranja)",
        xaxis_title="WG Ofensivo (Rolling histórico)",
        yaxis_title="WG Defensivo (Rolling histórico)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)")

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# LOAD CACHED DATA + LIVESCORE MERGE
# ==========================================================
@st.cache_data(ttl=3600)
def load_cached_data(selected_file: str):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    history = load_and_filter_history(selected_date_str)
    return games_today, history, selected_date_str

def load_and_merge_livescore(games_today: pd.DataFrame, selected_date_str: str) -> pd.DataFrame:
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

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
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

# ==========================================================
# QUADRANTES 16 (AGGRESSION X HANDSCORE)
# ==========================================================
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
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id
    return 0

# ==========================================================
# DISTÂNCIA 3D ENTRE HOME E AWAY (AGG, M_H, MT_H)
# ==========================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
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
            'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D'
        ]:
            df[col] = np.nan
        return df

    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    df['Quadrant_Dist_3D'] = np.sqrt(
        (dx)**2 * 1.5 + (dy/3.5)**2 * 2.0 + (dz/3.5)**2 * 1.8
    ) * 10

    df['Quadrant_Angle_XY'] = np.degrees(np.arctan2(dy, dx))
    df['Quadrant_Angle_XZ'] = np.degrees(np.arctan2(dz, dx))
    df['Quadrant_Angle_YZ'] = np.degrees(np.arctan2(dz, dy))

    angle_xy = np.arctan2(dy, dx)
    angle_xz = np.arctan2(dz, dx)
    angle_yz = np.arctan2(dz, dy)

    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Quadrant_Sin_XZ'] = np.sin(angle_xz)
    df['Quadrant_Cos_XZ'] = np.cos(angle_xz)
    df['Quadrant_Sin_YZ'] = np.sin(angle_yz)
    df['Quadrant_Cos_YZ'] = np.cos(angle_yz)

    df['Quadrant_Separation_3D'] = (
        0.4 * (60 * dx) + 0.35 * (20 * dy) + 0.25 * (20 * dz)
    )

    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    trig_cols = ['Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Sin_XZ',
                 'Quadrant_Cos_XZ', 'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ']
    created_trig = [col for col in trig_cols if col in df.columns]
    st.success(f"✅ Features trigonométricas calculadas: {len(created_trig)}/6")

    return df

# ==========================================================
# THRESHOLD DINÂMICO POR HANDICAP (ML)
# ==========================================================
def min_confidence_by_line(line):
    try:
        if pd.isna(line):
            return 0.60
        abs_line = abs(float(line))
    except Exception:
        return 0.60

    if abs_line >= 1.50:
        return 0.60
    if abs_line >= 1.00:
        return 0.58
    if abs_line >= 0.75:
        return 0.56
    if abs_line >= 0.50:
        return 0.54
    if abs_line >= 0.25:
        return 0.52
    return 0.50

# ==========================================================
# NOVO: DISTÂNCIA 2D NO PLANO WG (WG_Dist_2D)
# ==========================================================
def calcular_distancia_wg_2d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    req_cols = [
        'WG_Home_Team_Last', 'WG_Away_Team_Last',
        'WG_Def_Home_Team_Last', 'WG_Def_Away_Team_Last'
    ]
    if any(c not in df.columns for c in req_cols):
        df['WG_Dist_2D'] = np.nan
        return df

    dx = df['WG_Home_Team_Last'] - df['WG_Away_Team_Last']
    dy = df['WG_Def_Home_Team_Last'] - df['WG_Def_Away_Team_Last']

    df['WG_Dist_2D'] = np.sqrt(dx**2 + dy**2) * 10  # escala visual
    return df

def calcular_distancia_wg_2d_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versão para o histórico: usa WG_Home_Team / WG_Away_Team e WG_Def_Home_Team / WG_Def_Away_Team.
    """
    df = df.copy()
    req_cols = [
        'WG_Home_Team', 'WG_Away_Team',
        'WG_Def_Home_Team', 'WG_Def_Away_Team'
    ]
    if any(c not in df.columns for c in req_cols):
        df['WG_Dist_2D'] = np.nan
        return df

    dx = df['WG_Home_Team'] - df['WG_Away_Team']
    dy = df['WG_Def_Home_Team'] - df['WG_Def_Away_Team']

    df['WG_Dist_2D'] = np.sqrt(dx**2 + dy**2) * 10
    return df

# ==========================================================
# NOVO: SINAL FINAL WG GAP + ML (OPÇÃO B)
# ==========================================================
def gerar_sinal_wg_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    min_gap = 6.0
    df['WG_Gap_OK'] = df['WG_Dist_2D'] >= min_gap

    df['WG_Side'] = np.where(
        df['WG_Home_Team_Last'] > df['WG_Away_Team_Last'],
        'HOME',
        'AWAY'
    )

    if 'Bet_Side' in df.columns and 'Bet_Approved' in df.columns:
        # ML manda, WG pode ajustar o lado quando gap forte + aposta aprovada
        df['Final_Side'] = df['Bet_Side']
        df['Final_Approved'] = df['Bet_Approved']

        mask = df['WG_Gap_OK'] & df['Bet_Approved']
        df.loc[mask, 'Final_Side'] = df.loc[mask, 'WG_Side']
    else:
        df['Final_Side'] = df['WG_Side']
        df['Final_Approved'] = df['WG_Gap_OK']

    return df

# ==========================================================
# CARREGAR DADOS (GAMESDAY + HISTORY + LIVESCORE)
# ==========================================================
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

games_today, history, selected_date_str = load_cached_data(selected_file)

if 'Asian_Line' in games_today.columns:
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)
else:
    games_today['Asian_Line_Decimal'] = np.nan

games_today = load_and_merge_livescore(games_today, selected_date_str)

# Quadrantes
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

# Z-scores
st.markdown("## 📊 Calculando Z-scores a partir do HandScore")
if not history.empty:
    st.subheader("Para Histórico")
    history = calcular_zscores_detalhados(history)
if not games_today.empty:
    st.subheader("Para Jogos de Hoje")
    games_today = calcular_zscores_detalhados(games_today)

# Distâncias 3D
games_today = calcular_distancias_3d(games_today)
if not history.empty:
    history = calcular_distancias_3d(history)

# ==========================================================
# GES + WG NO HISTÓRICO + ENRICH NOS JOGOS DE HOJE - ATUALIZADO!
# ==========================================================
if not history.empty:
    # NOVO: Calcular parâmetros por liga de forma avançada
    liga_params = calcular_parametros_liga_avancado(history)
    
    # Mostrar parâmetros calculados
    if not liga_params.empty:
        mostrar_parametros_ligas(liga_params)

    history = adicionar_goal_efficiency_score(history, liga_params)
    history = calcular_rolling_ges(history)

    history = adicionar_weighted_goals(history)
    # NOVO: Usar função corrigida com parâmetros por liga
    history = adicionar_weighted_goals_defensivos_corrigido(history, liga_params)
    history = adicionar_weighted_goals_ah(history)
    history = adicionar_weighted_goals_ah_defensivos(history)
    history = calcular_metricas_completas(history)
    history = calcular_rolling_wg_features_completo(history)
    history = calcular_distancia_wg_2d_history(history)

    if not games_today.empty:
        games_today = adicionar_goal_efficiency_score(games_today, liga_params)
        games_today = calcular_rolling_ges(games_today)
        games_today = adicionar_weighted_goals(games_today)
        # NOVO: Usar função corrigida com parâmetros por liga
        games_today = adicionar_weighted_goals_defensivos_corrigido(games_today, liga_params)
        games_today = adicionar_weighted_goals_ah(games_today)
        games_today = adicionar_weighted_goals_ah_defensivos(games_today)
        games_today = calcular_metricas_completas(games_today)
        games_today = enrich_games_today_with_wg_completo(games_today, history)
        games_today = calcular_distancia_wg_2d(games_today)

# ==========================================================
# TREINO ML (DUAL) E PREDIÇÃO PARA OS JOGOS DE HOJE
# ==========================================================
model_home = None
model_away = None

if not history.empty:
    history_ml = create_better_target_corrigido(history)
    if len(history_ml) > 50:
        X_hist = create_robust_features(history_ml)

        # HOME
        y_home = history_ml['Target_AH_Home']
        model_home = train_improved_model(X_hist, y_home, X_hist.columns)

        # AWAY
        y_away = history_ml['Target_AH_Away']
        model_away = train_improved_model(X_hist, y_away, X_hist.columns)
    else:
        st.warning("Histórico insuficiente para treinar o modelo.")

if model_home is not None and model_away is not None and not games_today.empty:
    X_today = create_robust_features(games_today)

    # 🔐 GARANTIR MESMAS FEATURES DO TREINO (CORRIGE ERRO DE FEATURE NAMES)
    required_features = model_home.feature_names_in_
    X_today = X_today.reindex(columns=required_features, fill_value=0)

    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = model_away.predict_proba(X_today)[:, 1]

    games_today['Prob_Home_Cover'] = proba_home
    games_today['Prob_Away_Cover'] = proba_away

    games_today['Bet_Side'] = np.where(
        games_today['Prob_Home_Cover'] >= games_today['Prob_Away_Cover'],
        'HOME',
        'AWAY'
    )

    games_today['Bet_Confidence'] = np.maximum(
        games_today['Prob_Home_Cover'], games_today['Prob_Away_Cover']
    )

    games_today['Min_Conf_Required'] = games_today['Asian_Line_Decimal'].apply(min_confidence_by_line)
    games_today['Bet_Approved'] = games_today['Bet_Confidence'] >= games_today['Min_Conf_Required']

else:
    if not games_today.empty:
        games_today['Prob_Home_Cover'] = np.nan
        games_today['Prob_Away_Cover'] = np.nan
        games_today['Bet_Side'] = 'NONE'
        games_today['Bet_Confidence'] = 0.0
        games_today['Min_Conf_Required'] = 0.55
        games_today['Bet_Approved'] = False

# ==========================================================
# WG_Dist_2D + SINAL WG GAP (FINAL SIDE)
# ==========================================================
if not games_today.empty:
    if 'WG_Dist_2D' not in games_today.columns:
        games_today = calcular_distancia_wg_2d(games_today)
    games_today = gerar_sinal_wg_gap(games_today)

# ==========================================================
# DASHBOARD VISUAL
# ==========================================================
if not games_today.empty:
    # Gráfico WG
    plot_wg_vs_wgdef_scatter_interactive(games_today)

    # Ranking pelos maiores GAPS WG
    st.markdown("## 🏆 Melhores Confrontos por GAP WG (Ofensivo + Defensivo)")

    ranking = games_today.sort_values('WG_Dist_2D', ascending=False).copy()

    cols_rank = [
        'League', 'Home', 'Away', "Asian_Line_Decimal",
        'WG_Dist_2D',
        'WG_Home_Team_Last', 'WG_Away_Team_Last',
        'WG_Def_Home_Team_Last', 'WG_Def_Away_Team_Last',
        'WG_Diff', 'WG_Def_Diff',
        'M_H', 'M_A',
        'Prob_Home_Cover', 'Prob_Away_Cover',
        'Bet_Side', 'Bet_Confidence', 'Bet_Approved',
        'WG_Side', 'WG_Gap_OK',
        'Final_Side', 'Final_Approved'
    ]
    cols_rank = [c for c in cols_rank if c in ranking.columns]

    st.dataframe(ranking[cols_rank].head(25))

    # Tabela só com sinais aprovados
    aprovados = ranking[ranking['Final_Approved']].copy()
    if not aprovados.empty:
        st.markdown("### ✅ Sinais Aprovados (WG GAP + ML)")
        cols_aprov = [
            'League', 'Home', 'Away',
            'Asian_Line', 'Asian_Line_Decimal',
            'WG_Dist_2D',
            'Final_Side', 'Bet_Confidence'
        ]
        cols_aprov = [c for c in cols_aprov if c in aprovados.columns]
        st.dataframe(aprovados[cols_aprov].head(30))
    else:
        st.info("Nenhum sinal aprovado pelo filtro WG GAP + ML para hoje.")
