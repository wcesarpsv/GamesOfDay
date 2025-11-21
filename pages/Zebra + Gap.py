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

def _single_leg_home(margin, line):
    adj = margin + line
    if abs(line * 2) % 2 == 0:
        if adj > 0:
            return 1.0
        elif abs(adj) < 1e-9:
            return 0.5
        else:
            return 0.0
    else:
        return 1.0 if adj > 0 else 0.0

def calc_handicap_result_corrigido(margin, asian_line_decimal):
    if pd.isna(margin) or pd.isna(asian_line_decimal):
        return np.nan

    line = float(asian_line_decimal)

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

def calcular_zscores_detalhados(df):
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

# ---------------- CORREÇÕES CRÍTICAS PARA ML ----------------
def load_and_filter_history(selected_date_str):
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

    # Preencher quaisquer NaNs com 0 APÓS o load, conforme solicitado
    history = history.fillna(0)

    st.success(f"✅ Histórico processado: {len(history)} jogos (sem drop de linhas)")
    return history

def create_better_target_corrigido(df):
    df = df.copy()
    df["Margin"] = df["Goals_H_FT"] - df["Goals_A_FT"]
    df["AH_Result"] = df.apply(
        lambda r: calc_handicap_result_corrigido(
            r["Margin"], r["Asian_Line_Decimal"]
        ),
        axis=1
    )
    total = len(df)
    df = df[df["AH_Result"] != 0.5].copy()
    clean = len(df)
    df["Target_AH_Home"] = (df["AH_Result"] > 0.5).astype(int)
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
    win_rate = df["Target_AH_Home"].mean() if len(df) > 0 else 0.0
    zebra_rate = df["Zebra"].mean() if len(df) > 0 else 0.0
    st.info(f"🎯 Total analisado: {total} jogos")
    st.info(f"🗑️ Excluídos por Push puro: {total-clean} jogos ({(total-clean)/total:.1%})")
    st.info(f"📊 Treino com: {clean} jogos restantes")
    st.info(f"🏠 Win rate HOME cobrindo: {win_rate:.1%}")
    st.info(f"🦓 Taxa de Zebra (favorito falhou): {zebra_rate:.1%}")
    return df

# ---------------- WEIGHTED GOALS FEATURES (WG NOVO, COERENTE) ----------------
def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    WG_Home / WG_Away ofensivos baseado em:
      - Gols marcados / sofridos (FULL TIME)
      - Odds 1x2 (Odd_H, Odd_D, Odd_A)

    Ideia:
      - Marcar gols quando você era ZEBRA vale MAIS
      - Sofrer gols quando você era FAVORITO pesa MAIS contra
      - Usa probabilidades implícitas das odds 1x2
    """
    df_temp = df.copy()

    # Garante colunas numéricas de gols FT (NUNCA usa Goals_Today)
    for col in ['Goals_H_FT', 'Goals_A_FT']:
        if col not in df_temp.columns:
            df_temp[col] = np.nan

    # Converte odds 1x2 em probabilidades de mercado (com empate)
    def odds_to_probs_1x2(row):
        try:
            odd_h = float(row.get('Odd_H', 0))
            odd_d = float(row.get('Odd_D', 0))
            odd_a = float(row.get('Odd_A', 0))
        except Exception:
            return 0.33, 0.34, 0.33  # fallback neutro

        # Se odds inválidas, fallback
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

        # Quanto mais FAVORITO, menor peso pros gols marcados e MAIOR peso pros sofridos
        weight_for = (1 - p_h) + 0.5 * p_d      # marcar como zebra vale mais
        weight_against = p_h + 0.5 * p_d        # sofrer sendo favorito dói mais

        return gh * weight_for - ga * weight_against

    def wg_away(row):
        gh = row.get('Goals_H_FT', np.nan)
        ga = row.get('Goals_A_FT', np.nan)
        if pd.isna(gh) or pd.isna(ga):
            return np.nan

        p_h, p_d, p_a = odds_to_probs_1x2(row)

        # Agora do ponto de vista do visitante
        weight_for = (1 - p_a) + 0.5 * p_d      # marcar como zebra fora vale mais
        weight_against = p_a + 0.5 * p_d        # sofrer sendo favorito fora dói mais

        return ga * weight_for - gh * weight_against

    df_temp['WG_Home'] = df_temp.apply(wg_home, axis=1)
    df_temp['WG_Away'] = df_temp.apply(wg_away, axis=1)

    return df_temp


def adicionar_weighted_goals_defensivos(df: pd.DataFrame, liga_params: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    WG_Def_Home / WG_Def_Away:
      - Mede quanto a defesa sofreu em relação ao ESPERADO (xGA)
      - xGA vem de:
          * Média de gols da liga (Base_Goals_Liga)
          * Linha asiática (Asian_Line_Decimal)
          * Peso asiático da liga (Asian_Weight_Liga)
      - Defesa boa => valor POSITIVO (sofreu menos do que o esperado)
    """
    df_temp = df.copy()

    # Garantir colunas de gols FT (não usa Today)
    for col in ['Goals_H_FT', 'Goals_A_FT']:
        if col not in df_temp.columns:
            df_temp[col] = np.nan

    default_base_goals = 2.5
    default_asian_weight = 0.6

    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_Def_Home'] = 0.0
        df_temp['WG_Def_Away'] = 0.0
        return df_temp

    # Liga params traz Base_Goals_Liga e Asian_Weight_Liga
    # 🔒 Garante colunas, mesmo se merge falhar!
    df_temp['Base_Goals_Liga'] = default_base_goals
    df_temp['Asian_Weight_Liga'] = default_asian_weight
    
    if liga_params is not None and not liga_params.empty and 'League' in df_temp.columns:
        df_temp = df_temp.merge(
            liga_params[['League', 'Base_Goals_Liga', 'Asian_Weight_Liga', 'Jogos_Liga']],
            on='League',
            how='left',
            suffixes=('', '_m')
        )
    
        # Preenche valores faltantes
        df_temp['Base_Goals_Liga'] = df_temp['Base_Goals_Liga'].fillna(df_temp['Base_Goals_Liga_m'])
        df_temp['Asian_Weight_Liga'] = df_temp['Asian_Weight_Liga'].fillna(df_temp['Asian_Weight_Liga_m'])
    
    # Valores finais
    df_temp['Base_Goals_Usado'] = df_temp['Base_Goals_Liga'].fillna(default_base_goals)
    df_temp['Asian_Weight_Usado'] = df_temp['Asian_Weight_Liga'].fillna(default_asian_weight)

    else:
        df_temp['Base_Goals_Usado'] = default_base_goals
        df_temp['Asian_Weight_Usado'] = default_asian_weight

    # xGF por time via média da liga + linha asiática
    df_temp['xGF_H'] = (df_temp['Base_Goals_Usado'] / 2) + df_temp['Asian_Line_Decimal'] * df_temp['Asian_Weight_Usado']
    df_temp['xGF_A'] = (df_temp['Base_Goals_Usado'] / 2) - df_temp['Asian_Line_Decimal'] * df_temp['Asian_Weight_Usado']

    # xGA é o xGF do adversário
    df_temp['xGA_H'] = df_temp['xGF_A']
    df_temp['xGA_A'] = df_temp['xGF_H']

    # Gols sofridos reais
    df_temp['GA_H'] = df_temp['Goals_A_FT'].fillna(0)
    df_temp['GA_A'] = df_temp['Goals_H_FT'].fillna(0)

    # Defesa boa => sofreu MENOS que o esperado => xGA - GA > 0
    df_temp['WG_Def_Home'] = df_temp['xGA_H'] - df_temp['GA_H']
    df_temp['WG_Def_Away'] = df_temp['xGA_A'] - df_temp['GA_A']

    df_temp.drop(
        columns=[
            'xGF_H', 'xGF_A', 'xGA_H', 'xGA_A', 'GA_H', 'GA_A',
            'Base_Goals_Usado', 'Asian_Weight_Usado'
        ],
        inplace=True,
        errors='ignore'
    )

    return df_temp


def adicionar_weighted_goals_ah(df: pd.DataFrame) -> pd.DataFrame:
    """
    WG_AH_Home / WG_AH_Away:
      - Ajusta WG ofensivo pela dificuldade da linha asiática
      - Quanto maior o handicap absoluto, maior o peso
    """
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
    """
    WG_AH_Def_Home / WG_AH_Def_Away:
      - Ajusta WG defensivo pela dificuldade da linha (mesma lógica do ofensivo)
    """
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
    """
    Métricas combinando ataque/defesa para cada jogo (nível match):
      - Balance: ataque + defesa
      - Total: mesma coisa (mantido p/ compatibilidade)
      - Net: ataque próprio - defesa adversária
    """
    df_temp = df.copy()

    for col in [
        'WG_Home', 'WG_Away',
        'WG_Def_Home', 'WG_Def_Away'
    ]:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    # Balance (ataque + defesa)
    df_temp['WG_Balance_Home'] = df_temp['WG_Home'] + df_temp['WG_Def_Home']
    df_temp['WG_Balance_Away'] = df_temp['WG_Away'] + df_temp['WG_Def_Away']

    # Total (igual ao balance – mantido para não quebrar nada)
    df_temp['WG_Total_Home'] = df_temp['WG_Balance_Home']
    df_temp['WG_Total_Away'] = df_temp['WG_Balance_Away']

    # Net: ataque próprio - defesa adversária
    df_temp['WG_Net_Home'] = df_temp['WG_Home'] - df_temp['WG_Def_Away']
    df_temp['WG_Net_Away'] = df_temp['WG_Away'] - df_temp['WG_Def_Home']

    return df_temp


def calcular_rolling_wg_features_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling de WG ofensivo, defensivo e derivados (3 jogos) com shift(1)
    → time-safe (só usa o que já aconteceu ANTES do jogo)
    → separado por time em casa e fora
    """
    df_temp = df.copy()

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp = df_temp.sort_values('Date')

    # Garante colunas base
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

    # Rolling de 3 jogos, com shift(1)
    roll = lambda x: x.shift(1).rolling(3, min_periods=1).mean()

    # HOME / AWAY separados
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

    # Diffs entre as médias históricas dos dois times
    df_temp['WG_Diff'] = df_temp['WG_Home_Team'] - df_temp['WG_Away_Team']
    df_temp['WG_AH_Diff'] = df_temp['WG_AH_Home_Team'] - df_temp['WG_AH_Away_Team']
    df_temp['WG_Def_Diff'] = df_temp['WG_Def_Home_Team'] - df_temp['WG_Def_Away_Team']
    df_temp['WG_Balance_Diff'] = df_temp['WG_Balance_Home_Team'] - df_temp['WG_Balance_Away_Team']
    df_temp['WG_Net_Diff'] = df_temp['WG_Net_Home_Team'] - df_temp['WG_Net_Away_Team']

    # Confiança: quantas peças de WG temos históricos para esse jogo
    df_temp['WG_Confidence'] = (
        df_temp['WG_Home_Team'].notna().astype(int) +
        df_temp['WG_Away_Team'].notna().astype(int) +
        df_temp['WG_Def_Home_Team'].notna().astype(int) +
        df_temp['WG_Def_Away_Team'].notna().astype(int)
    )

    return df_temp


def enrich_games_today_with_wg_completo(games_today, history):
    """
    Para os jogos de hoje:
      - Puxa o ÚLTIMO valor de WG rolling (Ofensivo, Defensivo, AH, Net, etc.) de cada time
      - Calcula os diffs Home x Away só com base NO HISTÓRICO (sem gols de hoje)
    """
    if history.empty or games_today.empty:
        return games_today

    # Últimos valores por time como mandante
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

    # Últimos valores por time como visitante
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

    # Merge para HOME
    games_today = games_today.merge(
        last_wg_home, left_on='Home', right_on='Team', how='left'
    ).drop('Team', axis=1)

    # Merge para AWAY
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

    # Diffs só com histórico
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



@st.cache_data(ttl=7*24*3600)
def calcular_parametros_liga(history: pd.DataFrame) -> pd.DataFrame:
    """
    Parâmetros por liga:
    - Base_Goals_Liga
    - Asian_Weight_Liga
    - Jogos_Liga
    """
    df = history.copy()
    if 'Goals_H_FT' not in df.columns or 'Goals_A_FT' not in df.columns:
        return pd.DataFrame()

    df['Gols_Total'] = df['Goals_H_FT'].fillna(0) + df['Goals_A_FT'].fillna(0)

    liga_stats = df.groupby('League').agg(
        Jogos_Liga=('League', 'size'),
        Gols_Medios_Liga=('Gols_Total', 'mean'),
        Asi_Mean_Liga=('Asian_Line_Decimal', lambda x: x.abs().mean())
    ).reset_index()

    # Defaults globais
    gols_global = df['Gols_Total'].mean() if not df.empty else 2.5
    if 'Asian_Line_Decimal' in df.columns:
        asi_global = df['Asian_Line_Decimal'].abs().mean()
    else:
        asi_global = 0.6

    liga_stats['Gols_Medios_Liga'] = liga_stats['Gols_Medios_Liga'].fillna(gols_global)
    liga_stats['Asi_Mean_Liga'] = liga_stats['Asi_Mean_Liga'].fillna(asi_global)

    # Transformar Asi_Mean_Liga em peso (0.4 a 0.8 por exemplo)
    if not liga_stats['Asi_Mean_Liga'].isna().all():
        asi_min = liga_stats['Asi_Mean_Liga'].min()
        asi_max = liga_stats['Asi_Mean_Liga'].max()
        if asi_max > asi_min:
            liga_stats['Asian_Weight_Liga'] = 0.4 + 0.4 * (
                (liga_stats['Asi_Mean_Liga'] - asi_min) / (asi_max - asi_min)
            )
        else:
            liga_stats['Asian_Weight_Liga'] = 0.6
    else:
        liga_stats['Asian_Weight_Liga'] = 0.6

    liga_stats.rename(columns={
        'Gols_Medios_Liga': 'Base_Goals_Liga',
    }, inplace=True)

    return liga_stats

def create_robust_features(df):
    """
    Monta o vetor final de features usadas na ML (inclui WG).
    """
    df = df.copy()

    # Básicas
    basic_features = [
        'Aggression_Home', 'Aggression_Away',
        'M_H', 'M_A', 'MT_H', 'MT_A'
    ]

    # Derivadas de Agg/M/MT
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

    # Vetoriais 3D
    vector_features = [
        'Quadrant_Dist_3D', 'Momentum_Diff', 'Magnitude_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ'
    ]

    # WG (histórico trazido para jogos de hoje como *_Last)
    wg_features = [
        'WG_Home_Team_Last', 'WG_Away_Team_Last', 'WG_Diff',
        'WG_AH_Home_Team_Last', 'WG_AH_Away_Team_Last', 'WG_AH_Diff',
        'WG_Def_Home_Team_Last', 'WG_Def_Away_Team_Last', 'WG_Def_Diff',
        'WG_Balance_Home_Team_Last', 'WG_Balance_Away_Team_Last', 'WG_Balance_Diff',
        'WG_Net_Home_Team_Last', 'WG_Net_Away_Team_Last', 'WG_Net_Diff',
        'WG_Confidence'
    ]
  #########################
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



def adicionar_goal_efficiency_score(df: pd.DataFrame, liga_params: pd.DataFrame) -> pd.DataFrame:
    """
    GES = Goal Efficiency Score
    Eficiência ofensiva e defensiva em relação ao esperado
    Normalizado por liga para evitar assimetria na distribuição
    """
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

    # xGF esperado por mercado + média da liga
    df['xGF_H'] = (df['Base_Goals_Liga'] / 2) + df['Asian_Line_Decimal'] * df['Asian_Weight_Liga']
    df['xGF_A'] = (df['Base_Goals_Liga'] / 2) - df['Asian_Line_Decimal'] * df['Asian_Weight_Liga']

    # Eficiência ofensiva
    df['GES_Of_H'] = df['Goals_H_FT'] - df['xGF_H']
    df['GES_Of_A'] = df['Goals_A_FT'] - df['xGF_A']

    # Eficiência defensiva
    df['GES_Def_H'] = df['xGF_A'] - df['Goals_A_FT']
    df['GES_Def_A'] = df['xGF_H'] - df['Goals_H_FT']

    # Normalização por liga (evita assimetria crítica)
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

    # HOME – apenas jogos como mandante
    df['GES_Of_H_Roll'] = df.groupby('Home')['GES_Of_H_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['GES_Def_H_Roll'] = df.groupby('Home')['GES_Def_H_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # AWAY – apenas jogos como visitante
    df['GES_Of_A_Roll'] = df.groupby('Away')['GES_Of_A_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['GES_Def_A_Roll'] = df.groupby('Away')['GES_Def_A_Norm'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Fallback para evitar NaN no início de campeonato
    df[['GES_Of_H_Roll', 'GES_Of_A_Roll',
        'GES_Def_H_Roll', 'GES_Def_A_Roll']] = df[[
        'GES_Of_H_Roll', 'GES_Of_A_Roll',
        'GES_Def_H_Roll', 'GES_Def_A_Roll'
    ]].fillna(0)

    # Diferenças finais com peso ofensivo 60% + defensivo 40%
    df['GES_Of_Diff'] = df['GES_Of_H_Roll'] - df['GES_Of_A_Roll']
    df['GES_Def_Diff'] = df['GES_Def_H_Roll'] - df['GES_Def_A_Roll']

    df['GES_Total_Diff'] = (df['GES_Of_Diff'] * 0.60) + (df['GES_Def_Diff'] * 0.40)

    return df




##########################################################

def plot_wg_vs_wgdef_scatter_interactive(games_today: pd.DataFrame):
    """
    Gráfico 2D interativo (Plotly) mostrando WG Ofensivo x WG Defensivo
    para Home (laranja) e Away (azul), ligados por linha cinza.

    - Filtro de ligas com botão "Aplicar filtros" para evitar recálculo a cada clique
    - Slider para quantidade máxima de jogos
    - Tooltip com: Time, Liga, WG_Of e WG_Def + insight textual
    """

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

    # ---------- Estado de filtros em session_state ----------
    ligas_disponiveis = sorted(games_today['League'].dropna().unique().tolist())
    
    if 'wg_ligas_aplicadas' not in st.session_state:
        st.session_state['wg_ligas_aplicadas'] = ligas_disponiveis
    
    # 🔒 Filtra para garantir que defaults sempre existem nas opções
    defaults_filtrados = [
        l for l in st.session_state['wg_ligas_aplicadas']
        if l in ligas_disponiveis
    ]
    
    # Se todas as ligas default sumiram → seleciona todas as atuais
    if not defaults_filtrados:
        defaults_filtrados = ligas_disponiveis
    
    ligas_temp = st.multiselect(
        "Selecione as ligas (confirme depois em 'Aplicar filtros')",
        options=ligas_disponiveis,
        default=defaults_filtrados
    )


    if st.button("Aplicar filtros de ligas (WG)", type="primary"):
        # Atualiza apenas quando o usuário clica
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

    # ---------- Filtragem e ordenação ----------
    df_plot = games_today[games_today['League'].isin(ligas_usadas)].copy()

    if df_plot.empty:
        st.info("Nenhum jogo encontrado para as ligas selecionadas.")
        return

    # Maior assimetria ofensiva primeiro
    df_plot['WG_Diff_Grafico'] = (
        df_plot['WG_Home_Team_Last'] - df_plot['WG_Away_Team_Last']
    ).abs()
    df_plot = df_plot.sort_values('WG_Diff_Grafico', ascending=False).head(max_jogos)

        # ---------- Helpers ----------
    def insight_wg(wg_of, wg_def):
        if wg_of > 0 and wg_def > 0:
            return "Ataque forte & defesa sólida"
        elif wg_of > 0 and wg_def <= 0:
            return "Ataque forte & defesa vulnerável"
        elif wg_of <= 0 and wg_def > 0:
            return "Ataque fraco & defesa sólida"
        else:
            return "Ataque fraco & defesa vulnerável"

    # Coordenadas dos pontos HOME e AWAY
    home_x = df_plot['WG_Home_Team_Last'].values
    home_y = df_plot['WG_Def_Home_Team_Last'].values
    
    away_x = df_plot['WG_Away_Team_Last'].values
    away_y = df_plot['WG_Def_Away_Team_Last'].values


    # ---------- customdata para HOME ----------
    home_customdata = np.stack([
        df_plot['Home'].astype(str).values,        # customdata[0] - Home
        df_plot['Away'].astype(str).values,        # customdata[1] - Away
        df_plot['League'].astype(str).values,      # customdata[2] - Liga
        df_plot['WG_Home_Team_Last'].values,       # customdata[3] - WG Of Home
        df_plot['WG_Def_Home_Team_Last'].values,   # customdata[4] - WG Def Home
        df_plot.apply(lambda r: insight_wg(
            r['WG_Home_Team_Last'], r['WG_Def_Home_Team_Last']
        ), axis=1).astype(str).values              # customdata[5] - Insight Home
    ], axis=-1)

    # ---------- customdata para AWAY ----------
    away_customdata = np.stack([
        df_plot['Home'].astype(str).values,        # customdata[0] - Home
        df_plot['Away'].astype(str).values,        # customdata[1] - Away
        df_plot['League'].astype(str).values,      # customdata[2] - Liga
        df_plot['WG_Away_Team_Last'].values,       # customdata[3] - WG Of Away
        df_plot['WG_Def_Away_Team_Last'].values,   # customdata[4] - WG Def Away
        df_plot.apply(lambda r: insight_wg(
            r['WG_Away_Team_Last'], r['WG_Def_Away_Team_Last']
        ), axis=1).astype(str).values              # customdata[5] - Insight Away
    ], axis=-1)

    # ---------- Construção das linhas HOME ↔ AWAY ----------
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

    # ---------- Traces HOME e AWAY ----------
    home_text = df_plot['Home'].astype(str).values
    away_text = df_plot['Away'].astype(str).values

    trace_home = go.Scatter(
        x=home_x, y=home_y,
        mode="markers+text",
        name="Home",
        text=home_text,
        textposition="top right",
        marker=dict(size=9, color="orange"),
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
        marker=dict(size=9, color="blue"),
        customdata=away_customdata,
        hovertemplate=(
            "<b>%{customdata[0]} x %{customdata[1]}</b><br>"
            "Liga: %{customdata[2]}<br>"
            "WG Ofensivo: %{customdata[3]:.3f}<br>"
            "WG Defensivo: %{customdata[4]:.3f}<br>"
            "Insight: %{customdata[5]}<extra></extra>"
        )
    )

    # ---------- Montagem da figura ----------

    fig = go.Figure()

    # linhas primeiro (fica por baixo)
    for lt in line_traces:
        fig.add_trace(lt)

    fig.add_trace(trace_home)
    fig.add_trace(trace_away)

    # Eixos 0 / 0
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"))


    fig.update_layout(
        title="WG x WG_Def – Comparação Home (laranja) vs Away (azul)",
        xaxis_title="WG Ofensivo (Rolling histórico)",
        yaxis_title="WG Defensivo (Rolling histórico)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.2)")

    st.plotly_chart(fig, use_container_width=True)


######################################################





@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    # Extrair data do arquivo
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    history = load_and_filter_history(selected_date_str)
    return games_today, history, selected_date_str

def load_and_merge_livescore(games_today, selected_date_str):
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

# ---------------- Carregar Dados ----------------
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

games_today, history, selected_date_str = load_cached_data(selected_file)

# Converter Asian_Line de games_today
if 'Asian_Line' in games_today.columns:
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)
else:
    games_today['Asian_Line_Decimal'] = np.nan

games_today = load_and_merge_livescore(games_today, selected_date_str)

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
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id
    return 0

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

# ---------------- CÁLCULO DE Z-SCORES ----------------
st.markdown("## 📊 Calculando Z-scores a partir do HandScore")

if not history.empty:
    st.subheader("Para Histórico")
    history = calcular_zscores_detalhados(history)

if not games_today.empty:
    st.subheader("Para Jogos de Hoje")
    games_today = calcular_zscores_detalhados(games_today)

def calcular_distancias_3d(df):
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

games_today = calcular_distancias_3d(games_today)
if not history.empty:
    history = calcular_distancias_3d(history)

# ---------------- GES + WG NO HISTÓRICO + ENRICH NOS JOGOS DE HOJE ----------------
if not history.empty:

    # Primeiro obtém parâmetros por liga
    liga_params = calcular_parametros_liga(history)

    # --- GES primeiro ---
    history = adicionar_goal_efficiency_score(history, liga_params)
    history = calcular_rolling_ges(history)

    # --- SÓ depois o WG (porque precisa do Base_Goals_Liga dos liga_params!) ---
    history = adicionar_weighted_goals(history)
    history = adicionar_weighted_goals_defensivos(history, liga_params)
    history = adicionar_weighted_goals_ah(history)
    history = adicionar_weighted_goals_ah_defensivos(history)
    history = calcular_metricas_completas(history)
    history = calcular_rolling_wg_features_completo(history)

    # Enriquecer os jogos de hoje com ambos
    if not games_today.empty:
        games_today = adicionar_goal_efficiency_score(games_today, liga_params)
        games_today = calcular_rolling_ges(games_today)
        games_today = enrich_games_today_with_wg_completo(games_today, history)



# ---------------- THRESHOLD DINÂMICO POR HANDICAP ----------------
def min_confidence_by_line(line):
    """
    Threshold base da confiança mínima em função da linha asiática.
    """
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

def calcular_estatisticas_similares_e_ajustar_confianca(games_today, history_clean):
    """
    Para cada jogo de hoje:
      - Procura jogos históricos com linha semelhante
      - Calcula WinRate da SIDE escolhida pela ML nesses jogos
      - Gera N_Similares, WinRate_Similares, Confidence_Adjustment
      - Ajusta Min_Conf_Required por liga (Jogos_Liga)
      - Atualiza Bet_Confidence_Adjusted e Bet_Approved
    """
    if games_today.empty or history_clean.empty or 'Asian_Line_Decimal' not in games_today.columns:
        games_today['N_Similares'] = 0
        games_today['WinRate_Similares'] = np.nan
        games_today['Confidence_Adjustment'] = 0.0
        games_today['Bet_Confidence_Adjusted'] = games_today.get('Bet_Confidence', 0.5)
        games_today['Min_Conf_Required_Liga'] = games_today.get('Min_Conf_Required', 0.55)
        games_today['Bet_Approved'] = games_today['Bet_Confidence_Adjusted'] >= games_today['Min_Conf_Required_Liga']
        return games_today

    hist = history_clean.copy()
    if 'Asian_Line_Decimal' not in hist.columns:
        hist['Asian_Line_Decimal'] = np.nan

    league_game_counts = hist.groupby('League').size().to_dict() if 'League' in hist.columns else {}

    def _similar_stats(row):
        line = row.get('Asian_Line_Decimal', np.nan)
        league = row.get('League', None)
        bet_side = row.get('Bet_Side', 'HOME')
        if pd.isna(line):
            return 0, np.nan

        mask_line_global = hist['Asian_Line_Decimal'].between(line - 0.25, line + 0.25)
        subset_global = hist[mask_line_global]
        subset_liga = subset_global[subset_global['League'] == league] if league is not None and 'League' in hist.columns else pd.DataFrame()

        if bet_side == 'HOME':
            serie_liga = subset_liga['Target_AH_Home'] if not subset_liga.empty else pd.Series(dtype=float)
            serie_global = subset_global['Target_AH_Home']
        else:
            serie_liga = 1 - subset_liga['Target_AH_Home'] if not subset_liga.empty else pd.Series(dtype=float)
            serie_global = 1 - subset_global['Target_AH_Home']

        n_liga = len(serie_liga)
        n_global = len(serie_global)
        wr_liga = serie_liga.mean() if n_liga > 0 else np.nan
        wr_global = serie_global.mean() if n_global > 0 else np.nan

        if np.isnan(wr_global):
            wr_global = 0.5

        # Peso da liga vs global
        if n_liga >= 20:
            peso_liga = 1.0
        elif n_liga >= 10:
            peso_liga = 0.7
        elif n_liga >= 5:
            peso_liga = 0.5
        elif n_liga >= 1:
            peso_liga = 0.25
        else:
            peso_liga = 0.0

        if np.isnan(wr_liga):
            peso_liga = 0.0

        wr_comb = wr_liga * peso_liga + wr_global * (1 - peso_liga)
        n_eff = n_liga if n_liga > 0 else n_global

        return int(n_eff), wr_comb

    sim_df = games_today.apply(_similar_stats, axis=1, result_type='expand')
    games_today['N_Similares'] = sim_df[0].astype(int)
    games_today['WinRate_Similares'] = sim_df[1]

    def ajuste_conf(n):
        """
        Ajuste na confiança em função do volume de jogos similares.
        """
        if n >= 100:
            return 0.04
        if n >= 40:
            return 0.02
        if n >= 16:
            return 0.00
        if n >= 6:
            return -0.03
        if n >= 1:
            return -0.06
        return -0.10

    games_today['Confidence_Adjustment'] = games_today['N_Similares'].apply(ajuste_conf)

    if 'Bet_Confidence' in games_today.columns:
        games_today['Bet_Confidence_Adjusted'] = (
            games_today['Bet_Confidence'] + games_today['Confidence_Adjustment']
        ).clip(0.0, 1.0)
    else:
        games_today['Bet_Confidence_Adjusted'] = 0.5

    def ajuste_min_conf_por_liga(row):
        """
        Liga com menos jogos = exigir um pouco mais de confiança.
        """
        base = row.get('Min_Conf_Required', 0.55)
        league = row.get('League', None)
        if league is None:
            return base
        n_jogos = league_game_counts.get(league, 0)

        if n_jogos >= 200:
            delta = 0.0
        elif n_jogos >= 100:
            delta = 0.01
        elif n_jogos >= 50:
            delta = 0.015
        elif n_jogos >= 10:
            delta = 0.02
        else:
            delta = 0.03

        return float(np.clip(base + delta, 0.50, 0.70))

    games_today['Min_Conf_Required_Liga'] = games_today.apply(ajuste_min_conf_por_liga, axis=1)
    games_today['Bet_Approved'] = games_today['Bet_Confidence_Adjusted'] >= games_today['Min_Conf_Required_Liga']

    return games_today

def treinar_modelo_3d_quadrantes_16_corrigido(history, games_today):
    st.markdown("## 🤖 TREINAMENTO DO MODELO ML (CORRIGIDO)")
    if history.empty:
        st.error("❌ Histórico vazio - não é possível treinar modelo")
        return None, None, games_today

    history_clean = create_better_target_corrigido(history)

    if history_clean.empty:
        st.error("❌ Nenhum jogo válido após criação do target (sem push)")
        return None, None, games_today

    X_history = create_robust_features(history_clean)

    if X_history.empty:
        st.error("❌ Nenhuma feature disponível para treinamento")
        return None, None, games_today

    y_home = history_clean["Target_AH_Home"]
    y_away = 1 - y_home

    st.success(f"✅ Dados de treino: {X_history.shape[0]} amostras, {X_history.shape[1]} features")

    # Modelo HOME
    st.subheader("Modelo HOME (Home cobre AH)")
    model_home = train_improved_model(X_history, y_home, X_history.columns.tolist())

    # Modelo AWAY
    st.subheader("Modelo AWAY (Away cobre AH)")
    model_away = train_improved_model(X_history, y_away, X_history.columns.tolist())

    # ---------------- PREVISÕES PARA JOGOS DE HOJE ----------------
    if not games_today.empty:
        # Expected Favorite pelo handicap
        if 'Asian_Line_Decimal' in games_today.columns:
            games_today["Expected_Favorite"] = np.where(
                games_today["Asian_Line_Decimal"] < 0,
                "HOME",
                np.where(games_today["Asian_Line_Decimal"] > 0, "AWAY", "NONE")
            )

        X_today = create_robust_features(games_today)

        missing_cols = set(X_history.columns) - set(X_today.columns)
        for col in missing_cols:
            X_today[col] = 0
        X_today = X_today[X_history.columns]

        probas_home = model_home.predict_proba(X_today)[:, 1]
        probas_away = model_away.predict_proba(X_today)[:, 1]

        games_today['Quadrante_ML_Score_Home'] = probas_home
        games_today['Quadrante_ML_Score_Away'] = probas_away
        games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)

        games_today['Bet_Side'] = np.where(
            probas_home >= probas_away,
            'HOME',
            'AWAY'
        )

        games_today['Bet_Confidence'] = games_today['Quadrante_ML_Score_Main']

        # Threshold base pela linha
        if 'Asian_Line_Decimal' in games_today.columns:
            games_today['Min_Conf_Required'] = games_today['Asian_Line_Decimal'].apply(min_confidence_by_line)
        else:
            games_today['Min_Conf_Required'] = 0.60

        # Ajuste de confiança por jogos similares + threshold por liga
        games_today = calcular_estatisticas_similares_e_ajustar_confianca(games_today, history_clean)

        # Marca se é Zebra (ML contra o favorito de mercado)
        if 'Expected_Favorite' in games_today.columns:
            games_today['Is_Zebra_Bet'] = np.where(
                (games_today['Bet_Approved']) &
                (games_today['Expected_Favorite'].isin(['HOME', 'AWAY'])) &
                (games_today['Bet_Side'] != games_today['Expected_Favorite']),
                1,
                0
            )
        else:
            games_today['Is_Zebra_Bet'] = 0

        games_today['Bet_Label'] = np.where(
            ~games_today['Bet_Approved'],
            'NO BET',
            np.where(games_today['Bet_Side'] == 'HOME', 'BET HOME', 'BET AWAY')
        )

        st.success(f"✅ Previsões e lógica de aposta geradas para {len(games_today)} jogos de hoje")

        aprovados = games_today['Bet_Approved'].sum()
        zebras = games_today['Is_Zebra_Bet'].sum()
        st.info(f"Apostas aprovadas hoje: {aprovados} | Zebras agressivas sinalizadas: {zebras}")

    return model_home, model_away, games_today

# ---------------- EXECUTAR TREINO ----------------
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_3d_quadrantes_16_corrigido(history, games_today)

    if modelo_home is not None:
        st.success("✅ Modelo 3D corrigido treinado com sucesso!")

        if 'Quadrante_ML_Score_Main' in games_today.columns:
            avg_score = games_today['Quadrante_ML_Score_Main'].mean()
            high_confidence = len(games_today[games_today['Bet_Approved'] == True])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Score Médio", f"{avg_score:.1%}")
            with col2:
                st.metric("🎯 Apostas aprovadas (Home/Away)", high_confidence)
    else:
        st.error("❌ Falha no treinamento do modelo")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

# ---------------- INDICADORES EXPLICATIVOS 3D + CARDS ----------------
def adicionar_indicadores_explicativos_3d_16_dual(df):
    if df.empty:
        return df

    df = df.copy()

    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(
        lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro')
    )
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(
        lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro')
    )

    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_home = ['ALTO VALOR', 'BOM VALOR', 'NEUTRO', 'CAUTELA', 'ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='NEUTRO')

    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choices_away = ['ALTO VALOR', 'BOM VALOR', 'NEUTRO', 'CAUTELA', 'ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='NEUTRO')

    def gerar_recomendacao_3d_16_dual(row):
        home_q = row.get('Quadrante_Home_Label', 'Neutro')
        away_q = row.get('Quadrante_Away_Label', 'Neutro')
        score_home = row.get('Quadrante_ML_Score_Home', 0.5)
        score_away = row.get('Quadrante_ML_Score_Away', 0.5)
        bet_side = row.get('Bet_Side', 'HOME')
        bet_conf = row.get('Bet_Confidence_Adjusted', row.get('Bet_Confidence', 0.5))
        bet_approved = bool(row.get('Bet_Approved', False))
        momentum_h = row.get('M_H', 0)
        momentum_a = row.get('M_A', 0)
        expected_fav = row.get('Expected_Favorite', 'NONE')
        is_zebra = int(row.get('Is_Zebra_Bet', 0))

        if not bet_approved:
            return f'NO BET (H:{score_home:.1%} A:{score_away:.1%})'

        if is_zebra and expected_fav in ['HOME', 'AWAY']:
            return f'ZEBRA contra {expected_fav} ({bet_side}, {bet_conf:.1%})'

        if 'Fav Forte' in home_q and 'Under Forte' in away_q and momentum_h > 1.0 and bet_side == 'HOME':
            return f'Favorito HOME muito forte (+Momentum, {bet_conf:.1%})'
        if 'Under Forte' in home_q and 'Fav Forte' in away_q and momentum_a > 1.0 and bet_side == 'AWAY':
            return f'Favorito AWAY muito forte (+Momentum, {bet_conf:.1%})'

        if bet_side == 'HOME' and bet_conf >= 0.60 and momentum_h > 0:
            return f'ML confia em HOME (+Momentum, {bet_conf:.1%})'
        if bet_side == 'AWAY' and bet_conf >= 0.60 and momentum_a > 0:
            return f'ML confia em AWAY (+Momentum, {bet_conf:.1%})'

        if momentum_h < -1.0 and bet_side == 'AWAY' and bet_conf >= 0.55:
            return f'HOME em má fase → aposta AWAY ({bet_conf:.1%})'
        if momentum_a < -1.0 and bet_side == 'HOME' and bet_conf >= 0.55:
            return f'AWAY em má fase → aposta HOME ({bet_conf:.1%})'

        return f'Analisar (Bet:{bet_side}, {bet_conf:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_3d_16_dual, axis=1)

    df['Ranking'] = df['Bet_Confidence_Adjusted'].rank(ascending=False, method='dense').astype(int)

    return df


# ---------------- GRÁFICO 2D INTERATIVO WG x WG_Def (ACIMA DOS PICKS) ----------------
if not games_today.empty:
    plot_wg_vs_wgdef_scatter_interactive(games_today)



st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(games_today)
    ranking_3d = ranking_3d.sort_values('Bet_Confidence_Adjusted', ascending=False)

    colunas_3d = [
        'Ranking', 'League', 'Time', 'Home', 'Away',
        'Goals_H_Today', 'Goals_A_Today',
        'Bet_Label', 'Bet_Side', 'Bet_Confidence', 'Bet_Confidence_Adjusted', 'Bet_Approved',
        'N_Similares', 'WinRate_Similares',
        'Expected_Favorite', 'Is_Zebra_Bet',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
        'Min_Conf_Required', 'Min_Conf_Required_Liga',
        'Recomendacao',
        'M_H', 'M_A', 'Quadrant_Dist_3D', 'Momentum_Diff',
        'WG_Diff', 'WG_AH_Diff', 'WG_Def_Diff', 'WG_Balance_Diff', 'WG_Net_Diff',
        'Asian_Line', 'Asian_Line_Decimal'
    ]

    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    def estilo_tabela_3d_quadrantes(df):
        prob_cols = [c for c in [
            'Quadrante_ML_Score_Home',
            'Quadrante_ML_Score_Away',
            'Bet_Confidence',
            'Bet_Confidence_Adjusted',
            'Min_Conf_Required',
            'Min_Conf_Required_Liga',
            'WinRate_Similares'
        ] if c in df.columns]

        styler = df.style
        if prob_cols:
            styler = styler.background_gradient(subset=prob_cols, cmap='RdYlGn')
        return styler

    st.dataframe(
        estilo_tabela_3d_quadrantes(ranking_3d[cols_finais_3d]).format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Bet_Confidence': '{:.1%}',
            'Bet_Confidence_Adjusted': '{:.1%}',
            'Min_Conf_Required': '{:.1%}',
            'Min_Conf_Required_Liga': '{:.1%}',
            'M_H': '{:.2f}',
            'M_A': '{:.2f}',
            'Quadrant_Dist_3D': '{:.2f}',
            'Momentum_Diff': '{:.2f}',
            'WG_Diff': '{:.2f}',
            'WG_AH_Diff': '{:.2f}',
            'WG_Def_Diff': '{:.2f}',
            'WG_Balance_Diff': '{:.2f}',
            'WG_Net_Diff': '{:.2f}',
            'WinRate_Similares': '{:.1%}'
        }, na_rep="-"),
        use_container_width=True,
        height=600
    )

    st.markdown("## 🎴 Cards de Picks (Apostas aprovadas pela ML)")

    aprovados = ranking_3d[ranking_3d['Bet_Approved'] == True].copy()
    aprovados = aprovados.sort_values('Bet_Confidence_Adjusted', ascending=False)

    if aprovados.empty:
        st.info("Nenhuma aposta aprovado pela estratégia hoje.")
    else:
        for _, row in aprovados.head(20).iterrows():
            titulo = f"{row.get('League', '')}: {row.get('Home', '')} vs {row.get('Away', '')}"
            with st.expander(titulo):
                linha = row.get('Asian_Line_Decimal', np.nan)
                try:
                    linha_str = f"{linha:+.2f}"
                except Exception:
                    linha_str = str(linha)

                st.write(f"Aposta sugerida: **{row.get('Bet_Label', 'NO BET')}**")
                st.write(f"Lado da aposta (ML): **{row.get('Bet_Side', '')}** na linha {linha_str}")

                st.write(
                    f"Confiança ML (ajustada): **{row.get('Bet_Confidence_Adjusted', 0):.1%}** "
                    f"(Base: {row.get('Bet_Confidence', 0):.1%} | "
                    f"Histórico similar: {row.get('WinRate_Similares', 0.5):.1%} em {int(row.get('N_Similares', 0))} jogos)"
                )

                st.write(f"Threshold mínimo p/ essa linha/lig: **{row.get('Min_Conf_Required_Liga', 0):.1%}**")
                st.write(f"Favorito da casa (linha): **{row.get('Expected_Favorite', 'NONE')}**")

                zebra_txt = "Sim, ML contra o favorito da casa" if row.get('Is_Zebra_Bet', 0) == 1 else "Não"
                st.write(f"Zebra agressiva: **{zebra_txt}**")

                st.write(f"Quadrante HOME: {row.get('Quadrante_Home_Label', 'Neutro')}")
                st.write(f"Quadrante AWAY: {row.get('Quadrante_Away_Label', 'Neutro')}")

                st.write(f"Recomendação: {row.get('Recomendacao', '')}")

else:
    st.info("⚠️ Aguardando dados para gerar ranking 3D")

# ---------------- RESUMO EXECUTIVO 3D ----------------
def resumo_3d_16_quadrantes_hoje(df):
    st.markdown("### 📋 Resumo Executivo - Sistema 3D Hoje")

    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)
    momentum_positivo_home = len(df[df['M_H'] > 0.5]) if 'M_H' in df.columns else 0
    momentum_negativo_home = len(df[df['M_H'] < -0.5]) if 'M_H' in df.columns else 0
    momentum_positivo_away = len(df[df['M_A'] > 0.5]) if 'M_A' in df.columns else 0
    momentum_negativo_away = len(df[df['M_A'] < -0.5]) if 'M_A' in df.columns else 0
    zebra_alta = len(df[df['Is_Zebra_Bet'] == 1]) if 'Is_Zebra_Bet' in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("📈 Momentum + Home", momentum_positivo_home)
    with col2:
        st.metric("📉 Momentum - Home", momentum_negativo_home)
        st.metric("📈 Momentum + Away", momentum_positivo_away)
    with col3:
        st.metric("📉 Momentum - Away", momentum_negativo_away)
        st.metric("🦓 Zebras Sinalizadas", zebra_alta)
    with col4:
        aprovadas = len(df[df['Bet_Approved'] == True]) if 'Bet_Approved' in df.columns else 0
        st.metric("🎯 Apostas Aprovadas", aprovadas)
        media_conf = df['Bet_Confidence_Adjusted'].mean() if 'Bet_Confidence_Adjusted' in df.columns else 0
        st.metric("Confiança Média", f"{media_conf:.1%}")

if not games_today.empty:
    resumo_3d_16_quadrantes_hoje(games_today)

st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML CORRIGIDO + ZEBRA + WG + Amostra Histórica** implementado com sucesso!")
