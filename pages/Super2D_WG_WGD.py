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

st.set_page_config(page_title="Análise de Quadrantes - Bet Indicator", layout="wide")
st.title("🎯 Análise de Quadrantes - ML Avançado (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "coppa", "uefa", "afc", "sudamericana", "copa", "trophy"]

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

def convert_asian_line(line_str):
    """Converte string de linha asiática em média numérica"""
    try:
        if pd.isna(line_str) or line_str == "":
            return None
        line_str = str(line_str).strip()
        if "/" not in line_str:
            val = float(line_str)
            return 0.0 if abs(val) < 1e-10 else val
        parts = [float(x) for x in line_str.split("/")]
        avg = sum(parts) / len(parts)
        return 0.0 if abs(avg) < 1e-10 else avg
    except:
        return None

def calc_handicap_result(margin, asian_line_str, invert=False):
    """Retorna média de pontos por linha (1 win, 0.5 push, 0 loss)"""
    if pd.isna(asian_line_str):
        return np.nan

    try:
        parts = [float(x) for x in str(asian_line_str).split('/')]
    except:
        return np.nan

    results = []
    for line in parts:
        if margin > line:
            results.append(1.0)   # Home cobre
        elif margin == line:
            results.append(0.5)   # Push
        else:
            results.append(0.0)   # Home não cobre

    return np.mean(results)

def convert_asian_line_to_decimal(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    """
    if pd.isna(line_str) or line_str == "":
        return None

    try:
        line_str = str(line_str).strip()

        if "/" not in line_str:
            num = float(line_str)
            return -num

        parts = [float(p) for p in line_str.split("/")]
        avg = np.mean(parts)

        if str(line_str).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)

        return -result

    except (ValueError, TypeError):
        return None

# ---------------- WEIGHTED GOALS FEATURES (OFENSIVAS + DEFENSIVAS) ----------------
def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Weighted Goals (WG) para o dataframe
    Penaliza marcar menos do que o esperado
    Premia marcar mais do que o mercado esperava
    """
    df_temp = df.copy()

    for col in ['WG_Home', 'WG_Away']:
        if col not in df_temp.columns:
            df_temp[col] = 0.0

    def odds_to_market_probs(row):
        try:
            odd_h = float(row.get('Odd_H', 0))
            odd_a = float(row.get('Odd_A', 0))

            if odd_h <= 0 or odd_a <= 0:
                return 0.50, 0.50

            inv_h = 1 / odd_h
            inv_a = 1 / odd_a
            total = inv_h + inv_a
            return inv_h / total, inv_a / total

        except:
            return 0.50, 0.50

    def wg_home(row):
        p_h, p_a = odds_to_market_probs(row)
        goals_h = row.get('Goals_H_FT', 0)
        goals_a = row.get('Goals_A_FT', 0)
        return (goals_h * (1 - p_h)) - (goals_a * p_h)

    def wg_away(row):
        p_h, p_a = odds_to_market_probs(row)
        goals_h = row.get('Goals_H_FT', 0)
        goals_a = row.get('Goals_A_FT', 0)
        return (goals_a * (1 - p_a)) - (goals_h * p_a)

    df_temp['WG_Home'] = df_temp.apply(wg_home, axis=1)
    df_temp['WG_Away'] = df_temp.apply(wg_away, axis=1)

    return df_temp

def adicionar_weighted_goals_defensivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOVO cálculo WG_def:
    Usa xGoals baseados em odds e Asian Line
    WG_Def = xGA - GA (defesa melhor = positivo)
    """
    df_temp = df.copy()

    # Se não tiver Asian_Line_Decimal, não tem como calcular: retorna 0
    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_Def_Home'] = 0.0
        df_temp['WG_Def_Away'] = 0.0
        return df_temp

    # Parâmetros do modelo
    base_goals = 2.5
    asian_weight = 0.6

    # Calcular xGF home e away, ajustado pela força do handicap
    df_temp['xGF_H'] = base_goals / 2 + df_temp['Asian_Line_Decimal'] * asian_weight
    df_temp['xGF_A'] = base_goals / 2 - df_temp['Asian_Line_Decimal'] * asian_weight

    # xGA é o xGF do adversário
    df_temp['xGA_H'] = df_temp['xGF_A']
    df_temp['xGA_A'] = df_temp['xGF_H']

    # Gols sofridos (reais) – tratar ausência de colunas em games_today
    if 'Goals_A_FT' in df_temp.columns:
        df_temp['GA_H'] = df_temp['Goals_A_FT'].fillna(0)
    else:
        df_temp['GA_H'] = 0

    if 'Goals_H_FT' in df_temp.columns:
        df_temp['GA_A'] = df_temp['Goals_H_FT'].fillna(0)
    else:
        df_temp['GA_A'] = 0

    # Weighted Defensive Performance
    df_temp['WG_Def_Home'] = df_temp['xGA_H'] - df_temp['GA_H']
    df_temp['WG_Def_Away'] = df_temp['xGA_A'] - df_temp['GA_A']

    # Limpeza
    df_temp.drop(columns=['xGF_H', 'xGF_A', 'xGA_H', 'xGA_A', 'GA_H', 'GA_A'], inplace=True)

    return df_temp

def adicionar_weighted_goals_ah(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta o WG com base na dificuldade do handicap do mercado.
    Handicaps altos = mercado espera goleada
    • Se superar -> WG deve pesar mais
    • Se frustrar -> WG deve punir fortemente
    """
    df_temp = df.copy()

    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_AH_Home'] = 0.0
        df_temp['WG_AH_Away'] = 0.0
        return df_temp

    df_temp['WG_AH_Home'] = df_temp['WG_Home'] * (1 + df_temp['Asian_Line_Decimal'].abs())
    df_temp['WG_AH_Away'] = df_temp['WG_Away'] * (1 + df_temp['Asian_Line_Decimal'].abs())

    return df_temp

def adicionar_weighted_goals_ah_defensivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta WG defensivo com base no handicap
    """
    df_temp = df.copy()

    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_AH_Def_Home'] = 0.0
        df_temp['WG_AH_Def_Away'] = 0.0
        return df_temp

    # Para defesa: handicap alto = maior desafio defensivo
    df_temp['WG_AH_Def_Home'] = df_temp['WG_Def_Home'] * (1 + df_temp['Asian_Line_Decimal'].abs())
    df_temp['WG_AH_Def_Away'] = df_temp['WG_Def_Away'] * (1 + df_temp['Asian_Line_Decimal'].abs())

    return df_temp

def calcular_metricas_completas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria métricas que combinam ataque e defesa
    """
    df_temp = df.copy()
    
    # Balance Offensive/Defensive
    df_temp['WG_Balance_Home'] = df_temp['WG_Home'] + df_temp['WG_Def_Home']  # Soma porque ambos positivos são bons
    df_temp['WG_Balance_Away'] = df_temp['WG_Away'] + df_temp['WG_Def_Away']
    
    # Performance Total (ataque + defesa)
    df_temp['WG_Total_Home'] = df_temp['WG_Home'] + df_temp['WG_Def_Home']
    df_temp['WG_Total_Away'] = df_temp['WG_Away'] + df_temp['WG_Def_Away']
    
    # Net Performance (ataque - defesa oponente)
    df_temp['WG_Net_Home'] = df_temp['WG_Home'] - df_temp['WG_Def_Away']
    df_temp['WG_Net_Away'] = df_temp['WG_Away'] - df_temp['WG_Def_Home']
    
    return df_temp

def calcular_rolling_wg_features_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features rolling dos Weighted Goals (incluindo defesa)
    """
    df_temp = df.copy()

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp = df_temp.sort_values('Date')

    # Features ofensivas existentes
    df_temp['WG_Home_Team'] = df_temp.groupby('Home')['WG_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_Away_Team'] = df_temp.groupby('Away')['WG_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    df_temp['WG_AH_Home_Team'] = df_temp.groupby('Home')['WG_AH_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_AH_Away_Team'] = df_temp.groupby('Away')['WG_AH_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # NOVAS: Features defensivas
    df_temp['WG_Def_Home_Team'] = df_temp.groupby('Home')['WG_Def_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_Def_Away_Team'] = df_temp.groupby('Away')['WG_Def_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    df_temp['WG_AH_Def_Home_Team'] = df_temp.groupby('Home')['WG_AH_Def_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_AH_Def_Away_Team'] = df_temp.groupby('Away')['WG_AH_Def_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Métricas compostas
    df_temp['WG_Balance_Home_Team'] = df_temp.groupby('Home')['WG_Balance_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_Balance_Away_Team'] = df_temp.groupby('Away')['WG_Balance_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    df_temp['WG_Total_Home_Team'] = df_temp.groupby('Home')['WG_Total_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_Total_Away_Team'] = df_temp.groupby('Away')['WG_Total_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    df_temp['WG_Net_Home_Team'] = df_temp.groupby('Home')['WG_Net_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df_temp['WG_Net_Away_Team'] = df_temp.groupby('Away')['WG_Net_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Diffs atualizados
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

def enrich_games_today_with_wg_completo(games_today, history):
    """
    Enriquece os jogos de hoje com TODAS as médias rolling do histórico
    """
    # Features ofensivas/defensivas de histórico (último valor por time)
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

    # Merge Home
    games_today = games_today.merge(
        last_wg_home, left_on='Home', right_on='Team', how='left'
    ).drop('Team', axis=1)

    # Merge Away
    games_today = games_today.merge(
        last_wg_away, left_on='Away', right_on='Team', how='left'
    ).drop('Team', axis=1)

    # Preencher NaN
    wg_cols = [
        'WG_Home_Team_Last', 'WG_AH_Home_Team_Last', 'WG_Def_Home_Team_Last', 'WG_AH_Def_Home_Team_Last',
        'WG_Balance_Home_Team_Last', 'WG_Total_Home_Team_Last', 'WG_Net_Home_Team_Last',
        'WG_Away_Team_Last', 'WG_AH_Away_Team_Last', 'WG_Def_Away_Team_Last', 'WG_AH_Def_Away_Team_Last',
        'WG_Balance_Away_Team_Last', 'WG_Total_Away_Team_Last', 'WG_Net_Away_Team_Last'
    ]
    
    for col in wg_cols:
        if col in games_today.columns:
            games_today[col] = games_today[col].fillna(0)
        else:
            games_today[col] = 0

    # Calcular diffs com base no histórico (não nos gols de hoje)
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

# ---------------- Carregar Dados ----------------
st.info("📂 Carregando dados para análise de quadrantes...")

if not os.path.exists(GAMES_FOLDER):
    st.warning("Pasta GamesDay não encontrada.")
    st.stop()

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today, selected_date_str):
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    games_today = setup_livescore_columns(games_today)

    if os.path.exists(livescore_file):
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
        else:
            games_today = games_today.merge(
                results_df,
                left_on='Id',
                right_on='Id',
                how='left',
                suffixes=('', '_RAW')
            )

            games_today['Goals_H_Today'] = games_today['home_goal']
            games_today['Goals_A_Today'] = games_today['away_goal']
            if 'status' in games_today.columns:
                games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

            games_today['Home_Red'] = games_today['home_red']
            games_today['Away_Red'] = games_today['away_red']

            st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
            return games_today
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

games_today = load_and_merge_livescore(games_today, selected_date_str)

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
if not history.empty:
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# ---------------- CONVERSÃO ASIAN LINE ----------------
if not history.empty:
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal) if 'Asian_Line' in games_today.columns else np.nan

if not history.empty:
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

# ---------------- APLICAR TODAS AS FEATURES WG (OFENSIVAS + DEFENSIVAS) ----------------
st.info("🧮 Calculando features completas de Weighted Goals...")

if not history.empty:
    history = adicionar_weighted_goals(history)
    history = adicionar_weighted_goals_defensivos(history)  # NOVO
    history = adicionar_weighted_goals_ah(history)
    history = adicionar_weighted_goals_ah_defensivos(history)  # NOVO
    history = calcular_metricas_completas(history)  # NOVO
    history = calcular_rolling_wg_features_completo(history)  # ATUALIZADO

games_today = adicionar_weighted_goals(games_today)
games_today = adicionar_weighted_goals_defensivos(games_today)  # NOVO
games_today = adicionar_weighted_goals_ah(games_today)
games_today = adicionar_weighted_goals_ah_defensivos(games_today)  # NOVO
games_today = calcular_metricas_completas(games_today)  # NOVO

if not history.empty:
    games_today = enrich_games_today_with_wg_completo(games_today, history)  # ATUALIZADO

st.success(f"✅ Weighted Goals completos calculados: {len(history) if not history.empty else 0} jogos históricos processados")

# Targets AH históricos
if not history.empty:
    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(
        lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"]) > 0.5 else 0, 
        axis=1
    )

# ---------------- SISTEMA DE 8 QUADRANTES ----------------
st.markdown("## 🎯 Sistema de 8 Quadrantes")

QUADRANTES_8 = {
    1: {"nome": "Underdog Value Forte",      "agg_max": -0.5, "hs_min": 30},
    2: {"nome": "Underdog Value",            "agg_max": 0,    "hs_min": 15},
    3: {"nome": "Favorite Reliable Forte",   "agg_min": 0.5,  "hs_min": 30},
    4: {"nome": "Favorite Reliable",         "agg_min": 0,    "hs_min": 15},
    5: {"nome": "Market Overrates Forte",    "agg_min": 0.5,  "hs_max": -30},
    6: {"nome": "Market Overrates",          "agg_min": 0,    "hs_max": -15},
    7: {"nome": "Weak Underdog Forte",       "agg_max": -0.5, "hs_max": -30},
    8: {"nome": "Weak Underdog",             "agg_max": 0,    "hs_max": -15}
}

def classificar_quadrante(agg, hs):
    """Classifica Aggression e HandScore em um dos 8 quadrantes"""
    if pd.isna(agg) or pd.isna(hs):
        return 0

    for quadrante_id, config in QUADRANTES_8.items():
        agg_ok = True
        hs_ok = True

        if 'agg_min' in config and agg < config['agg_min']:
            agg_ok = False
        if 'agg_max' in config and agg > config['agg_max']:
            agg_ok = False

        if 'hs_min' in config and hs < config['hs_min']:
            hs_ok = False
        if 'hs_max' in config and hs > config['hs_max']:
            hs_ok = False

        if agg_ok and hs_ok:
            return quadrante_id

    return 0

games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)

if not history.empty:
    history['Quadrante_Home'] = history.apply(
        lambda x: classificar_quadrante(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    history['Quadrante_Away'] = history.apply(
        lambda x: classificar_quadrante(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )

def calcular_distancias_quadrantes(df):
    """Calcula distância, separação média e ângulo entre os pontos Home e Away."""
    df = df.copy()
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away']):
        dx = df['Aggression_Home'] - df['Aggression_Away']
        dy = df['HandScore_Home'] - df['HandScore_Away']
        df['Quadrant_Dist'] = np.sqrt(dx**2 + (dy/60)**2 * 2.5) * 10
        df['Quadrant_Separation'] = 0.5 * (dy + 60 * dx)
        df['Quadrant_Angle_Geometric'] = np.degrees(np.arctan2(dy, dx))
        df['Quadrant_Angle_Normalized'] = np.degrees(np.arctan2((dy / 60), dx))
    else:
        st.warning("⚠️ Colunas Aggression/HandScore não encontradas para calcular as distâncias.")
        df['Quadrant_Dist'] = np.nan
        df['Quadrant_Separation'] = np.nan
        df['Quadrant_Angle_Geometric'] = np.nan
    return df

games_today = calcular_distancias_quadrantes(games_today)
if not history.empty:
    history = calcular_distancias_quadrantes(history)

# ---------------- VISUALIZAÇÃO DAS NOVAS FEATURES ----------------
st.markdown("## 📊 Análise das Weighted Goals Features Completas")

if not games_today.empty and 'WG_Diff' in games_today.columns:
    col1_m, col2_m, col3_m, col4_m = st.columns(4)

    with col1_m:
        st.metric("Média WG_Diff", f"{games_today['WG_Diff'].mean():.3f}")
    with col2_m:
        st.metric("Média WG_Def_Diff", f"{games_today['WG_Def_Diff'].mean():.3f}")
    with col3_m:
        st.metric("Média WG_Balance_Diff", f"{games_today['WG_Balance_Diff'].mean():.3f}")
    with col4_m:
        st.metric("Confiança Média WG", f"{games_today['WG_Confidence'].mean():.1f}")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    valid_data = games_today.dropna(subset=['WG_Diff', 'WG_Def_Diff'])
    
    if len(valid_data) > 0:
        # WG Ofensivo vs Defensivo
        if 'Quadrante_ML_Score_Main' in valid_data.columns:
            scatter1 = ax1.scatter(
                valid_data['WG_Diff'], valid_data['WG_Def_Diff'], 
                c=valid_data['Quadrante_ML_Score_Main'],
                cmap='RdYlGn', alpha=0.7, s=50
            )
            plt.colorbar(scatter1, ax=ax1, label='Score ML')
        else:
            ax1.scatter(valid_data['WG_Diff'], valid_data['WG_Def_Diff'], 
                        alpha=0.7, s=50, color='blue')
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('WG_Diff (Ataque Home - Away)')
        ax1.set_ylabel('WG_Def_Diff (Defesa Home - Away)')
        ax1.set_title('Ataque vs Defesa')
        
        # Balance vs Net
        valid_data_balance = games_today.dropna(subset=['WG_Balance_Diff', 'WG_Net_Diff'])
        if len(valid_data_balance) > 0:
            if 'Quadrante_ML_Score_Main' in valid_data_balance.columns:
                scatter2 = ax2.scatter(
                    valid_data_balance['WG_Balance_Diff'], valid_data_balance['WG_Net_Diff'],
                    c=valid_data_balance['Quadrante_ML_Score_Main'],
                    cmap='RdYlGn', alpha=0.7, s=50
                )
                plt.colorbar(scatter2, ax=ax2, label='Score ML')
            else:
                ax2.scatter(
                    valid_data_balance['WG_Balance_Diff'], valid_data_balance['WG_Net_Diff'],
                    alpha=0.7, s=50, color='green'
                )
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('WG_Balance_Diff')
            ax2.set_ylabel('WG_Net_Diff')
            ax2.set_title('Balance vs Net Performance')
        else:
            ax2.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Balance vs Net Performance')
        
        # Distribuição WG Defensivo
        ax3.hist(
            games_today['WG_Def_Diff'].dropna(), 
            bins=min(20, len(games_today)), 
            alpha=0.7, color='skyblue', edgecolor='black'
        )
        if len(games_today['WG_Def_Diff'].dropna()) > 0:
            ax3.axvline(
                x=games_today['WG_Def_Diff'].mean(), 
                color='red', linestyle='--', 
                label=f'Média: {games_today["WG_Def_Diff"].mean():.3f}'
            )
        ax3.set_xlabel('WG_Def_Diff')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição WG Defensivo')
        ax3.legend()
        
        # Correlação entre features
        corr_features = games_today[['WG_Diff', 'WG_Def_Diff', 'WG_Balance_Diff', 'WG_Net_Diff']].corr()
        im = ax4.imshow(corr_features, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_features.columns)))
        ax4.set_yticks(range(len(corr_features.columns)))
        ax4.set_xticklabels(corr_features.columns, rotation=45)
        ax4.set_yticklabels(corr_features.columns)
        ax4.set_title('Correlação entre Features WG')
        
        for i in range(len(corr_features.columns)):
            for j in range(len(corr_features.columns)):
                ax4.text(
                    j, i, f'{corr_features.iloc[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(corr_features.iloc[i, j]) > 0.5 else 'black'
                )
        
        plt.colorbar(im, ax=ax4)
    else:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(
                0.5, 0.5, 'Dados insuficientes para visualização', 
                ha='center', va='center', transform=ax.transAxes
            )
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("### 🏆 Top Jogos por Weighted Goals Balance")
if not games_today.empty and 'WG_Balance_Diff' in games_today.columns:
    top_wg = games_today.nlargest(10, 'WG_Balance_Diff')[[
        c for c in ['Home', 'Away', 'League', 'WG_Diff', 'WG_Def_Diff', 
                    'WG_Balance_Diff', 'WG_Net_Diff', 'WG_Confidence'] 
        if c in games_today.columns
    ]]
    st.dataframe(
        top_wg.style.format({
            'WG_Diff': '{:.3f}',
            'WG_Def_Diff': '{:.3f}',
            'WG_Balance_Diff': '{:.3f}',
            'WG_Net_Diff': '{:.3f}'
        }),
        use_container_width=True
    )

# ---------------- VISUALIZAÇÃO INTERATIVA ----------------
st.markdown("## 🎯 Visualização Interativa – Distância entre Times (Home × Away)")

if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())
    selected_league = st.selectbox(
        "Selecione a liga para análise:",
        options=["⚽ Todas as ligas"] + list(leagues),
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
if max_n == 0:
    n_to_show = 0
else:
    n_to_show = st.slider("Quantos confrontos exibir (Top por distância):", 10, min(max_n, 200), 40, step=5)

angle_min, angle_max = st.slider(
    "Filtrar por Ângulo (posição Home vs Away):",
    min_value=-180, max_value=180, value=(-180, 180), step=5,
    help="Ângulos positivos → Home acima | Ângulos negativos → Away acima"
)

use_combined_filter = st.checkbox(
    "Usar filtro combinado (Distância + Ângulo)",
    value=True,
    help="Se desmarcado, exibirá apenas confrontos dentro do intervalo de ângulo, ignorando o filtro de distância."
)

if "Quadrant_Dist" not in df_filtered.columns:
    df_filtered = calcular_distancias_quadrantes(df_filtered)

df_angle = df_filtered[
    (df_filtered['Quadrant_Angle_Normalized'] >= angle_min) &
    (df_filtered['Quadrant_Angle_Normalized'] <= angle_max)
]

if use_combined_filter:
    df_plot = df_angle.nlargest(n_to_show, "Quadrant_Dist").reset_index(drop=True)
else:
    df_plot = df_angle.reset_index(drop=True)

fig_int = go.Figure()

for _, row in df_plot.iterrows():
    xh, xa = row["Aggression_Home"], row["Aggression_Away"]
    yh, ya = row["HandScore_Home"], row["HandScore_Away"]

    fig_int.add_trace(go.Scatter(
        x=[xh, xa],
        y=[yh, ya],
        mode="lines+markers",
        line=dict(color="gray", width=1),
        marker=dict(size=5),
        hoverinfo="text",
        hovertext=(
            f"<b>{row['Home']} vs {row['Away']}</b><br>"
            f"🏆 {row.get('League','N/A')}<br>"
            f"📏 Distância: {row['Quadrant_Dist']:.2f}<br>"
            f"📐 Ângulo: {row['Quadrant_Angle_Normalized']:.1f}°<br>"
            f"🎯 WG_Diff: {row.get('WG_Diff', np.nan):.3f}<br>"
            f"🛡️ WG_Def_Diff: {row.get('WG_Def_Diff', np.nan):.3f}"
        ),
        showlegend=False
    ))

fig_int.add_trace(go.Scatter(
    x=df_plot["Aggression_Home"],
    y=df_plot["HandScore_Home"],
    mode="markers+text",
    name="Home",
    marker=dict(color="royalblue", size=8, opacity=0.8),
    text=df_plot["Home"],
    textposition="top center",
    hoverinfo="skip"
))

fig_int.add_trace(go.Scatter(
    x=df_plot["Aggression_Away"],
    y=df_plot["HandScore_Away"],
    mode="markers+text",
    name="Away",
    marker=dict(color="orangered", size=8, opacity=0.8),
    text=df_plot["Away"],
    textposition="top center",
    hoverinfo="skip"
))

fig_int.add_trace(go.Scatter(
    x=[-1, 1], y=[0, 0],
    mode="lines", line=dict(color="limegreen", width=2, dash="dash"), name="Eixo X"
))
fig_int.add_trace(go.Scatter(
    x=[0, 0], y=[-60, 60],
    mode="lines", line=dict(color="limegreen", width=2, dash="dash"), name="Eixo Y"
))

titulo = f"Confrontos – Aggression × HandScore"
if use_combined_filter and n_to_show > 0:
    titulo += f" | Top {n_to_show} Distâncias"
if "League" in games_today.columns and selected_league != "⚽ Todas as ligas":
    titulo += f" | {selected_league}"

fig_int.update_layout(
    title=titulo,
    xaxis_title="Aggression (-1 zebra ↔ +1 favorito)",
    yaxis_title="HandScore (-60 ↔ +60)",
    template="plotly_white",
    height=700,
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_int, use_container_width=True)

# ---------------- VISUALIZAÇÃO DOS QUADRANTES ----------------
def plot_quadrantes_avancado(df, side="Home"):
    """Plot dos 8 quadrantes com cores e anotações"""
    fig, ax = plt.subplots(figsize=(10, 8))

    cores_quadrantes = {
        1: 'lightgreen',
        2: 'green',
        3: 'lightcoral',
        4: 'red',
        5: 'lightyellow',
        6: 'yellow',
        7: 'lightgray',
        8: 'gray',
        0: 'black'
    }

    col_quadrante = f'Quadrante_{side}'
    col_agg = f'Aggression_{side}'
    col_hs = f'HandScore_{side}'

    if col_quadrante not in df.columns or col_agg not in df.columns or col_hs not in df.columns:
        ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax.transAxes)
        return fig

    for quadrante_id in range(9):
        mask = df[col_quadrante] == quadrante_id
        if mask.any():
            x = df.loc[mask, col_agg]
            y = df.loc[mask, col_hs]
            ax.scatter(
                x, y, 
                c=cores_quadrantes.get(quadrante_id, 'black'), 
                label=QUADRANTES_8.get(quadrante_id, {}).get('nome', 'Neutro'),
                alpha=0.7, s=50
            )

    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(x=-0.5, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=15, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=30, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=-15, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=-30, color='black', linestyle='--', alpha=0.3)

    ax.text(-0.75, 45, "Underdog\nValue Forte", ha='center', fontsize=9, weight='bold')
    ax.text(-0.25, 22, "Underdog\nValue", ha='center', fontsize=9)
    ax.text(0.75, 45, "Favorite\nReliable Forte", ha='center', fontsize=9, weight='bold')
    ax.text(0.25, 22, "Favorite\nReliable", ha='center', fontsize=9)
    ax.text(0.75, -45, "Market\nOverrates Forte", ha='center', fontsize=9, weight='bold')
    ax.text(0.25, -22, "Market\nOverrates", ha='center', fontsize=9)
    ax.text(-0.75, -45, "Weak\nUnderdog Forte", ha='center', fontsize=9, weight='bold')
    ax.text(-0.25, -22, "Weak\nUnderdog", ha='center', fontsize=9)

    ax.set_xlabel(f'Aggression_{side} (-1 zebra ↔ +1 favorito)')
    ax.set_ylabel(f'HandScore_{side} (-60 a +60)')
    ax.set_title(f'8 Quadrantes - {side}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

st.markdown("### 📈 Visualização dos Quadrantes")
col_q1, col_q2 = st.columns(2)
with col_q1:
    st.pyplot(plot_quadrantes_avancado(games_today, "Home"))
with col_q2:
    st.pyplot(plot_quadrantes_avancado(games_today, "Away"))

# ---------------- FUNÇÕES DE HANDICAP ----------------
def determine_handicap_result(row):
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
        asian_line_home = row['Asian_Line_Decimal']
        recomendacao = str(row.get('Recomendacao', '')).upper()
    except (ValueError, TypeError, KeyError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_home):
        return None

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

    if is_home_bet:
        asian_line = asian_line_home
    else:
        asian_line = -asian_line_home

    side = "HOME" if is_home_bet else "AWAY"

    frac = abs(asian_line % 1)
    is_quarter = frac in [0.25, 0.75]

    def single_result(gh, ga, line, side):
        if side == "HOME":
            adjusted = (gh + line) - ga
        else:
            adjusted = (ga + line) - gh

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
        adjusted = (ga + asian_line) - gh

    if adjusted > 0:
        return f"{side}_COVERED"
    elif adjusted < 0:
        return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"
    else:
        return "PUSH"

def check_handicap_recommendation_correct(rec, handicap_result):
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
        return None

    rec = str(rec)
    # (mantido como estava – ainda podemos refinar depois)
    return None

def calculate_handicap_profit(rec, handicap_result, odds_row, asian_line_decimal):
    # Mantido como no seu código original (podemos alinhar depois)
    return 0

# ---------------- TREINAMENTO ML DUAL COM TODAS AS FEATURES ----------------
def treinar_modelo_quadrantes_dual_completo(history, games_today):
    """
    Treina modelo ML para Home e Away com base nos quadrantes,
    ligas, métricas de distância E Weighted Goals completos.
    """
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    if len(history) < 10:
        st.warning("⚠️ Histórico insuficiente para treinar o modelo")
        games_today['Quadrante_ML_Score_Home'] = 0.5
        games_today['Quadrante_ML_Score_Away'] = 0.5
        games_today['Quadrante_ML_Score_Main'] = 0.5
        games_today['ML_Side'] = 'HOME'
        return None, None, games_today

    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League') if 'League' in history.columns else pd.DataFrame(index=history.index)

    extras = history[[
        c for c in ['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle_Geometric', 'Quadrant_Angle_Normalized']
        if c in history.columns
    ]].fillna(0)

    wg_features = history[[
        c for c in ['WG_Diff', 'WG_AH_Diff', 'WG_Def_Diff', 'WG_Balance_Diff', 'WG_Net_Diff', 'WG_Confidence']
        if c in history.columns
    ]].fillna(0)

    X = pd.concat([ligas_dummies, extras, wg_features, quadrantes_home, quadrantes_away], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    y_home = history['Target_AH_Home']
    y_away = 1 - y_home

    if y_home.nunique() < 2:
        st.warning("⚠️ Dados de target insuficientes para treinamento")
        games_today['Quadrante_ML_Score_Home'] = 0.5
        games_today['Quadrante_ML_Score_Away'] = 0.5
        games_today['Quadrante_ML_Score_Main'] = 0.5
        games_today['ML_Side'] = 'HOME'
        return None, None, games_today

    try:
        model_home = RandomForestClassifier(
            n_estimators=min(100, len(history)),
            max_depth=8, 
            random_state=42, 
            class_weight='balanced_subsample', 
            n_jobs=-1
        )
        model_away = RandomForestClassifier(
            n_estimators=min(100, len(history)),
            max_depth=8,
            random_state=42, 
            class_weight='balanced_subsample', 
            n_jobs=-1
        )

        model_home.fit(X, y_home)
        model_away.fit(X, y_away)

        X_today = pd.DataFrame(0, index=games_today.index, columns=X.columns)
        
        for col in X.columns:
            if col in games_today.columns:
                X_today[col] = games_today[col].fillna(0)
            elif col.startswith('QH_'):
                quadrante_num = int(col.split('_')[1])
                X_today[col] = (games_today['Quadrante_Home'] == quadrante_num).astype(int)
            elif col.startswith('QA_'):
                quadrante_num = int(col.split('_')[1])
                X_today[col] = (games_today['Quadrante_Away'] == quadrante_num).astype(int)
            elif col.startswith('League_') and 'League' in games_today.columns:
                league_name = col[7:]
                X_today[col] = (games_today['League'] == league_name).astype(int)

        wg_cols = ['WG_Diff', 'WG_AH_Diff', 'WG_Def_Diff', 'WG_Balance_Diff', 'WG_Net_Diff', 'WG_Confidence']
        for col in wg_cols:
            if col in games_today.columns and col in X_today.columns:
                X_today[col] = games_today[col].fillna(0)

        dist_cols = ['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle_Geometric', 'Quadrant_Angle_Normalized']
        for col in dist_cols:
            if col in games_today.columns and col in X_today.columns:
                X_today[col] = games_today[col].fillna(0)

        probas_home = model_home.predict_proba(X_today)[:, 1]
        probas_away = model_away.predict_proba(X_today)[:, 1]

        games_today['Quadrante_ML_Score_Home'] = probas_home
        games_today['Quadrante_ML_Score_Away'] = probas_away
        games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
        games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

        try:
            importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
            top_feats = importances.head(20)
            st.markdown("### 🔍 Top Features mais importantes (Modelo HOME - Completo)")
            st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível calcular importâncias: {e}")

        st.success("✅ Modelo dual completo (com features defensivas) treinado com sucesso!")
        return model_home, model_away, games_today
        
    except Exception as e:
        st.error(f"❌ Erro no treinamento do modelo: {e}")
        games_today['Quadrante_ML_Score_Home'] = 0.5
        games_today['Quadrante_ML_Score_Away'] = 0.5
        games_today['Quadrante_ML_Score_Main'] = 0.5
        games_today['ML_Side'] = 'HOME'
        return None, None, games_today

# ---------------- SISTEMA DE INDICAÇÕES EXPLÍCITAS DUAL ----------------
def adicionar_indicadores_explicativos_dual(df):
    """Adiciona classificações e recomendações explícitas para Home e Away"""
    df = df.copy()

    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.60,
        df['Quadrante_ML_Score_Home'] >= 0.55,
        df['Quadrante_ML_Score_Home'] >= 0.50,
        df['Quadrante_ML_Score_Home'] >= 0.45,
        df['Quadrante_ML_Score_Home'] < 0.45
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.60,
        df['Quadrante_ML_Score_Away'] >= 0.55,
        df['Quadrante_ML_Score_Away'] >= 0.50,
        df['Quadrante_ML_Score_Away'] >= 0.45,
        df['Quadrante_ML_Score_Away'] < 0.45
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

    def gerar_recomendacao_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']

        if home_q == 'Underdog Value' and away_q == 'Market Overrates':
            return f'🎯 VALUE NO HOME ({score_home:.1%})'
        elif home_q == 'Market Overrates' and away_q == 'Underdog Value':
            return f'🎯 VALUE NO AWAY ({score_away:.1%})'
        elif home_q == 'Favorite Reliable' and away_q == 'Weak Underdog':
            return f'💪 FAVORITO HOME ({score_home:.1%})'
        elif home_q == 'Weak Underdog' and away_q == 'Favorite Reliable':
            return f'💪 FAVORITO AWAY ({score_away:.1%})'
        elif ml_side == 'HOME' and score_home >= 0.55:
            return f'📈 MODELO CONFIA HOME ({score_home:.1%})'
        elif ml_side == 'AWAY' and score_away >= 0.55:
            return f'📈 MODELO CONFIA AWAY ({score_away:.1%})'
        elif 'Market Overrates' in home_q and score_away >= 0.55:
            return f'🔴 HOME SUPERAVALIADO → AWAY ({score_away:.1%})'
        elif 'Market Overrates' in away_q and score_home >= 0.55:
            return f'🔴 AWAY SUPERAVALIADO → HOME ({score_home:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_dual, axis=1)
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

def estilo_tabela_quadrantes_dual(df):
    def cor_classificacao(valor):
        if '🏆 ALTO VALOR' in str(valor): return 'font-weight: bold'
        elif '✅ BOM VALOR' in str(valor): return 'font-weight: bold' 
        elif '🔴 ALTO RISCO' in str(valor): return 'font-weight: bold'
        elif 'VALUE' in str(valor): return 'font-weight: bold'
        elif 'EVITAR' in str(valor): return 'font-weight: bold'
        elif 'SUPERAVALIADO' in str(valor): return 'font-weight: bold'
        else: return ''

    colunas_para_estilo = []
    if 'Classificacao_Valor_Home' in df.columns:
        colunas_para_estilo.append('Classificacao_Valor_Home')
    if 'Classificacao_Valor_Away' in df.columns:
        colunas_para_estilo.append('Classificacao_Valor_Away')
    if 'Recomendacao' in df.columns:
        colunas_para_estilo.append('Recomendacao')

    styler = df.style
    if colunas_para_estilo:
        styler = styler.applymap(cor_classificacao, subset=colunas_para_estilo)

    if 'Quadrante_ML_Score_Home' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')
    if 'Quadrante_ML_Score_Away' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')
    if 'Quadrante_ML_Score_Main' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Main'], cmap='RdYlGn')

    return styler

# ---------------- EXECUÇÃO PRINCIPAL ----------------
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_quadrantes_dual_completo(history, games_today)
    
    if modelo_home is not None and modelo_away is not None:
        st.success("✅ Modelo dual completo (Home/Away) treinado com sucesso!")
    else:
        st.warning("⚠️ Modelo não foi treinado - usando valores padrão")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")
    games_today['Quadrante_ML_Score_Home'] = 0.5
    games_today['Quadrante_ML_Score_Away'] = 0.5
    games_today['Quadrante_ML_Score_Main'] = 0.5
    games_today['ML_Side'] = 'HOME'

# ---------------- EXIBIÇÃO DOS RESULTADOS DUAL ----------------
st.markdown("## 🏆 Melhores Confrontos por Quadrantes ML (Home & Away)")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_quadrantes = games_today.copy()
    ranking_quadrantes['Quadrante_Home_Label'] = ranking_quadrantes['Quadrante_Home'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    ranking_quadrantes['Quadrante_Away_Label'] = ranking_quadrantes['Quadrante_Away'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )

    ranking_quadrantes = adicionar_indicadores_explicativos_dual(ranking_quadrantes)

    def update_real_time_data(df):
        df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
        df['Quadrante_Correct'] = df.apply(
            lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
        )
        df['Profit_Quadrante'] = df.apply(
            lambda r: calculate_handicap_profit(r['Recomendacao'], r['Handicap_Result'], r, r.get('Asian_Line_Decimal', np.nan)), axis=1
        )
        return df

    ranking_quadrantes = update_real_time_data(ranking_quadrantes)

    def generate_live_summary(df):
        finished_games = df.dropna(subset=['Handicap_Result'])

        if finished_games.empty:
            return {
                "Total Jogos": len(df),
                "Jogos Finalizados": 0,
                "Apostas Quadrante": 0,
                "Acertos Quadrante": 0,
                "Winrate Quadrante": "0%",
                "Profit Quadrante": 0,
                "ROI Quadrante": "0%"
            }

        quadrante_bets = finished_games[finished_games['Quadrante_Correct'].notna()]
        total_bets = len(quadrante_bets)
        correct_bets = quadrante_bets['Quadrante_Correct'].sum() if not quadrante_bets.empty else 0
        winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
        total_profit = quadrante_bets['Profit_Quadrante'].sum()
        roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": len(finished_games),
            "Apostas Quadrante": total_bets,
            "Acertos Quadrante": int(correct_bets),
            "Winrate Quadrante": f"{winrate:.1f}%",
            "Profit Quadrante": f"{total_profit:.2f}u",
            "ROI Quadrante": f"{roi:.1f}%"
        }

    st.markdown("## 📡 Live Score Monitor")
    live_summary = generate_live_summary(ranking_quadrantes)
    st.json(live_summary)

    if 'Quadrante_ML_Score_Main' in ranking_quadrantes.columns:
        ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score_Main', ascending=False)
    else:
        ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score_Home', ascending=False)

    colunas_possiveis = [
        'League', 'Time', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'ML_Side', 'Recomendacao',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 'Quadrante_ML_Score_Main', 
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        'WG_Diff', 'WG_Def_Diff', 'WG_Balance_Diff', 'WG_Net_Diff', 'WG_Confidence',
        'Asian_Line_Decimal', 'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
    ]

    cols_finais = [c for c in colunas_possiveis if c in ranking_quadrantes.columns]

    st.dataframe(
        estilo_tabela_quadrantes_dual(ranking_quadrantes[cols_finais])
        .format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Home_Red': '{:.0f}',
            'Away_Red': '{:.0f}',
            'Profit_Quadrante': '{:.2f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Quadrante_ML_Score_Main': '{:.1%}',
            'WG_Diff': '{:.3f}',
            'WG_Def_Diff': '{:.3f}',
            'WG_Balance_Diff': '{:.3f}',
            'WG_Net_Diff': '{:.3f}',
            'WG_Confidence': '{:.0f}'
        }, na_rep="-"),
        use_container_width=True
    )

else:
    st.info("⚠️ Aguardando dados para gerar ranking dual")

# ---------------- ÍNDICE DE CONVERGÊNCIA TOTAL ----------------
def calc_convergencia(row):
    try:
        score_home = float(row.get('Quadrante_ML_Score_Home', 0))
        score_away = float(row.get('Quadrante_ML_Score_Away', 0))
        dist = float(row.get('Quadrant_Dist', 0))
        ml_side = "HOME" if score_home > score_away else "AWAY"
        diff = abs(score_home - score_away)
    except Exception:
        return 0.0

    w_ml = min(diff * 2, 1.0)
    w_dist = min(dist / 0.8, 1.0)

    home_q = str(row.get('Quadrante_Home_Label', ''))
    away_q = str(row.get('Quadrante_Away_Label', ''))

    padrao_favoravel = (
        ('Underdog Value' in home_q and ml_side == 'HOME') or
        ('Market Overrates' in away_q and ml_side == 'HOME') or
        ('Favorite Reliable' in home_q and ml_side == 'HOME') or
        ('Weak Underdog' in away_q and ml_side == 'AWAY')
    )
    w_pattern = 1.0 if padrao_favoravel else 0.0

    confidence_score = round((0.5 * w_ml + 0.3 * w_dist + 0.2 * w_pattern), 3)
    return confidence_score

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_quadrantes['Confidence_Score'] = ranking_quadrantes.apply(calc_convergencia, axis=1)

    st.markdown("### 🥇 Gold Matches – Convergência Máxima")
    gold_matches = ranking_quadrantes[ranking_quadrantes['Confidence_Score'] >= 0.75]

    if not gold_matches.empty:
        st.dataframe(
            gold_matches[['League', 'Home', 'Away', 'Recomendacao', 
                          'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 'Confidence_Score',
                          'WG_Balance_Diff', 'WG_Net_Diff']]
            .sort_values('Confidence_Score', ascending=False)
            .style.format({
                'Quadrante_ML_Score_Home': '{:.1%}',
                'Quadrante_ML_Score_Away': '{:.1%}',
                'Confidence_Score': '{:.2f}',
                'WG_Balance_Diff': '{:.3f}',
                'WG_Net_Diff': '{:.3f}'
            })
            .background_gradient(subset=['Confidence_Score'], cmap='YlGn'),
            use_container_width=True
        )
    else:
        st.info("Nenhum confronto atingiu nível de convergência 🥇 Gold hoje.")

# ---------------- RESUMO EXECUTIVO DUAL ----------------
def resumo_quadrantes_hoje_dual(df):
    st.markdown("### 📋 Resumo Executivo - Quadrantes Hoje (Dual)")

    if df.empty:
        st.info("Nenhum dado disponível para resumo")
        return

    total_jogos = len(df)
    alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
    bom_valor_home = len(df[df['Classificacao_Valor_Home'] == '✅ BOM VALOR'])
    alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])
    bom_valor_away = len(df[df['Classificacao_Valor_Away'] == '✅ BOM VALOR'])

    home_recomendado = len(df[df['ML_Side'] == 'HOME'])
    away_recomendado = len(df[df['ML_Side'] == 'AWAY'])

    col1_r, col2_r, col3_r, col4_r = st.columns(4)

    with col1_r:
        st.metric("Total Jogos", total_jogos)
    with col2_r:
        st.metric("🎯 Alto Valor Home", alto_valor_home)
    with col3_r:
        st.metric("🎯 Alto Valor Away", alto_valor_away)
    with col4_r:
        st.metric("📊 Home vs Away", f"{home_recomendado} : {away_recomendado}")

    st.markdown("#### 📊 Distribuição de Recomendações")
    dist_recomendacoes = df['Recomendacao'].value_counts()
    st.dataframe(dist_recomendacoes, use_container_width=True)

if not games_today.empty and 'Classificacao_Valor_Home' in games_today.columns:
    resumo_quadrantes_hoje_dual(games_today)

# ======================== 🎯 VALUE BETS – FILTRADAS ========================
st.markdown("## 💰 Value Bets Confirmadas – Zona Ideal")

if 'ML_Side' in locals() or ('ML_Side' in games_today.columns):
    df_value = ranking_quadrantes.copy()

    df_value = df_value[
        (df_value['Quadrante_ML_Score_Main'] >= 0.55) &
        (df_value['WG_Def_Diff'] > 0) &
        (df_value['WG_Balance_Diff'] > 0.50)
    ]

    df_value = df_value[
        ((df_value['ML_Side'] == 'HOME') & (df_value['Asian_Line_Decimal'] >= -0.50)) |
        ((df_value['ML_Side'] == 'AWAY') & (df_value['Asian_Line_Decimal'] <= +0.50))
    ]

    if df_value.empty:
        st.info("⚠️ Nenhuma aposta atingiu o nível ideal de valor hoje.")
    else:
        df_value = df_value.sort_values('Quadrante_ML_Score_Main', ascending=False)

        col_visu = [
            'League', 'Home', 'Away', 'ML_Side', 'Recomendacao',
            'Quadrante_ML_Score_Main',
            'WG_Balance_Diff', 'WG_Def_Diff',
            'Asian_Line_Decimal', 'Quadrant_Dist',
            'Confidence_Score'
        ]
        col_visu = [c for c in col_visu if c in df_value.columns]

        st.dataframe(
            df_value[col_visu]
            .style.format({
                'Quadrante_ML_Score_Main': '{:.1%}',
                'WG_Balance_Diff': '{:.3f}',
                'WG_Def_Diff': '{:.3f}',
                'Asian_Line_Decimal': '{:.2f}',
                'Quadrant_Dist': '{:.2f}',
                'Confidence_Score': '{:.2f}'
            })
            .background_gradient(subset=['Quadrante_ML_Score_Main'], cmap='RdYlGn'),
            use_container_width=True
        )

st.markdown("---")
st.success("🎯 **Análise de Quadrantes ML Dual Completa** - Sistema avançado com features ofensivas e defensivas de Weighted Goals para identificação de value bets!")
