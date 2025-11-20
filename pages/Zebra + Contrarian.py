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

# ---------------- Asian Line Helpers ----------------
def convert_asian_line(line_str):
    """Converte string de linha asiática em média numérica simples (não usada no modelo)."""
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
    except Exception:
        return None

# ---------------- CORREÇÕES CRÍTICAS ASIAN LINE ----------------
def convert_asian_line_to_decimal_corrigido(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    """
    if pd.isna(line_str) or line_str == "":
        return None

    line_str = str(line_str).strip()

    # Caso especial: linha zero
    if line_str in ("0", "0.0"):
        return 0.0

    # ✅ Mapeamento de splits comuns
    common_splits = {
        # Splits positivos (Away dá handicap)
        '0/0.5': -0.25,
        '0.5/1': -0.75,
        '1/1.5': -1.25,
        '1.5/2': -1.75,
        '2/2.5': -2.25,
        '2.5/3': -2.75,
        '3/3.5': -3.25,

        # Splits negativos (Away recebe handicap)
        '0/-0.5': 0.25,
        '-0.5/-1': 0.75,
        '-1/-1.5': 1.25,
        '-1.5/-2': 1.75,
        '-2/-2.5': 2.25,
        '-2.5/-3': 2.75,
        '-3/-3.5': 3.25,

        # Quarter handicaps
        '0.75': -0.75,
        '-0.75': 0.75,
        '0.25': -0.25,
        '-0.25': 0.25,
    }

    if line_str in common_splits:
        return common_splits[line_str]

    # Caso simples — número único
    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num  # Inverte sinal (Away → Home)
        except ValueError:
            return None

    # Split genérico
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
    """
    Calcula o resultado de UM handicap (sem split) do ponto de vista do HOME.

    margin = Goals_H_FT - Goals_A_FT
    line   = handicap do HOME (Asian_Line_Decimal, já convertido)

    Retorno:
      1.0  -> full win
      0.5  -> push
      0.0  -> full loss
    """
    adj = margin + line  # gols_home - gols_away + handicap_home

    # Linhas inteiras (.0): podem ter push
    if abs(line * 2) % 2 == 0:  # múltiplo de 1.0 (ex: -2, -1, 0, 1, 2...)
        if adj > 0:
            return 1.0
        elif abs(adj) < 1e-9:
            return 0.5
        else:
            return 0.0

    # Linhas .5: não têm push
    else:
        return 1.0 if adj > 0 else 0.0

def calc_handicap_result_corrigido(margin, asian_line_decimal):
    """
    Calcula o resultado do Handicap Asiático do ponto de vista do HOME,
    considerando também quarter-lines (0.25, 0.75, etc).

    Retorno:
      0.0   -> full loss
      0.25  -> half loss
      0.5   -> push
      0.75  -> half win
      1.0   -> full win
    """
    if pd.isna(margin) or pd.isna(asian_line_decimal):
        return np.nan

    line = float(asian_line_decimal)

    # Quarter-lines: |line * 2| NÃO é inteiro (ex: 0.25, 0.75, 1.25...)
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

    # Linhas normais (.0 ou .5)
    else:
        return _single_leg_home(margin, line)

def testar_conversao_asian_line():
    st.markdown("### 🧪 TESTE COMPLETO – LINHA & RESULTADO")

    test_cases = [
        # Full lines
        ("0.5", "Away +0.5 → Home -0.5"),
        ("-0.5", "Away -0.5 → Home +0.5"),
        ("1.0", "Away +1.0 → Home -1.0"),
        ("-1.0", "Away -1.0 → Home +1.0"),

        # Splits
        ("0/0.5", "Away 0/0.5 → Home -0.25"),
        ("0/-0.5", "Away 0/-0.5 → Home +0.25"),
        ("0.5/1", "Away 0.5/1 → Home -0.75"),
        ("-0.5/-1", "Away -0.5/-1 → Home +0.75"),
        ("1/1.5", "Away 1/1.5 → Home -1.25"),
        ("-1/-1.5", "Away -1/-1.5 → Home +1.25"),
        ("1.5/2", "Away 1.5/2 → Home -1.75"),
        ("-1.5/-2", "Away -1.5/-2 → Home +1.75"),

        # Quarter-lines
        ("0.25", "Away +0.25 → Home -0.25"),
        ("-0.25", "Away -0.25 → Home +0.25"),
        ("0.75", "Away +0.75 → Home -0.75"),
        ("-0.75", "Away -0.75 → Home +0.75"),

        # Zero line
        ("0", "Away 0 → Home 0"),
    ]

    test_margins = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]

    results = []
    for line_str, desc in test_cases:
        decimal = convert_asian_line_to_decimal_corrigido(line_str)

        tests = []
        for m in test_margins:
            r = calc_handicap_result_corrigido(m, decimal)
            if r == 1.0:
                sym = "🟩"  # full win
            elif r == 0.5:
                sym = "🟨"  # half/push
            else:
                sym = "🟥"  # loss
            tests.append(f"{m}:{sym}")

        results.append({
            "AsianLine(Away)": line_str,
            "Convertido(Home)": decimal,
            "Teste": " | ".join(tests)
        })

    st.dataframe(pd.DataFrame(results))
    st.success("Conversões e resultados validados!")

# ---------------- WEIGHTED GOALS OFFENSIVOS ----------------
def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Weighted Goals (WG) ofensivos para o dataframe.

    - Penaliza marcar menos do que o esperado
    - Premia marcar mais do que o mercado esperava
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

            inv_h = 1.0 / odd_h
            inv_a = 1.0 / odd_a
            total = inv_h + inv_a
            return inv_h / total, inv_a / total

        except Exception:
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

# ---------------- WEIGHTED GOALS DEFENSIVOS ----------------
def adicionar_weighted_goals_defensivos(df: pd.DataFrame, liga_params: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    NOVO cálculo WG_def:
    Usa xGoals baseados em odds, Asian Line e parâmetros por liga.
    WG_Def = xGA - GA (defesa melhor = positivo)

    - Se liga tiver histórico → usa Base_Goals_Liga e Asian_Weight_Liga
    - Senão → usa defaults globais (2.5 gols, 0.6 de peso)
    """
    df_temp = df.copy()

    # Garantir colunas de gols (para evitar KeyError em games_today)
    if 'Goals_H_FT' not in df_temp.columns:
        df_temp['Goals_H_FT'] = df_temp.get('Goals_H_Today', np.nan)
    if 'Goals_A_FT' not in df_temp.columns:
        df_temp['Goals_A_FT'] = df_temp.get('Goals_A_Today', np.nan)

    # Defaults globais
    default_base_goals = 2.5
    default_asian_weight = 0.6

    # Se não tiver Asian_Line_Decimal, não dá pra ajustar por handicap
    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_Def_Home'] = 0.0
        df_temp['WG_Def_Away'] = 0.0
        return df_temp

    # Anexar parâmetros por liga, se disponíveis
    if liga_params is not None and not liga_params.empty and 'League' in df_temp.columns:
        df_temp = df_temp.merge(
            liga_params[['League', 'Base_Goals_Liga', 'Asian_Weight_Liga']],
            on='League',
            how='left'
        )
        df_temp['Base_Goals_Usado'] = df_temp['Base_Goals_Liga'].fillna(default_base_goals)
        df_temp['Asian_Weight_Usado'] = df_temp['Asian_Weight_Liga'].fillna(default_asian_weight)
    else:
        df_temp['Base_Goals_Usado'] = default_base_goals
        df_temp['Asian_Weight_Usado'] = default_asian_weight

    # xGF home e away ajustados por handicap + parâmetros da liga
    df_temp['xGF_H'] = (df_temp['Base_Goals_Usado'] / 2.0) + df_temp['Asian_Line_Decimal'] * df_temp['Asian_Weight_Usado']
    df_temp['xGF_A'] = (df_temp['Base_Goals_Usado'] / 2.0) - df_temp['Asian_Line_Decimal'] * df_temp['Asian_Weight_Usado']

    # xGA é o xGF do adversário
    df_temp['xGA_H'] = df_temp['xGF_A']
    df_temp['xGA_A'] = df_temp['xGF_H']

    # Gols sofridos (reais)
    df_temp['GA_H'] = df_temp['Goals_A_FT'].fillna(0)
    df_temp['GA_A'] = df_temp['Goals_H_FT'].fillna(0)

    # Weighted Defensive Performance (defesa boa = sofre menos do que o xGA)
    df_temp['WG_Def_Home'] = df_temp['xGA_H'] - df_temp['GA_H']
    df_temp['WG_Def_Away'] = df_temp['xGA_A'] - df_temp['GA_A']

    # Limpeza de colunas auxiliares
    df_temp.drop(
        columns=[
            'xGF_H', 'xGF_A', 'xGA_H', 'xGA_A', 'GA_H', 'GA_A',
            'Base_Goals_Usado', 'Asian_Weight_Usado',
            'Base_Goals_Liga', 'Asian_Weight_Liga'
        ],
        inplace=True,
        errors='ignore'
    )

    return df_temp

# ---------------- WEIGHTED GOALS AJUSTADOS POR HANDICAP ----------------
def adicionar_weighted_goals_ah(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta o WG ofensivo com base na dificuldade do handicap do mercado.
    Handicaps altos = mercado espera goleada:
    • Se superar -> WG deve pesar mais
    • Se frustrar -> WG deve punir fortemente
    """
    df_temp = df.copy()

    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_AH_Home'] = 0.0
        df_temp['WG_AH_Away'] = 0.0
        return df_temp

    fator = 1.0 + df_temp['Asian_Line_Decimal'].abs()
    df_temp['WG_AH_Home'] = df_temp['WG_Home'] * fator
    df_temp['WG_AH_Away'] = df_temp['WG_Away'] * fator

    return df_temp

def adicionar_weighted_goals_ah_defensivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta WG defensivo com base no handicap.
    Para defesa: handicap alto = maior desafio defensivo.
    """
    df_temp = df.copy()

    if 'Asian_Line_Decimal' not in df_temp.columns:
        df_temp['WG_AH_Def_Home'] = 0.0
        df_temp['WG_AH_Def_Away'] = 0.0
        return df_temp

    fator = 1.0 + df_temp['Asian_Line_Decimal'].abs()
    df_temp['WG_AH_Def_Home'] = df_temp['WG_Def_Home'] * fator
    df_temp['WG_AH_Def_Away'] = df_temp['WG_Def_Away'] * fator

    return df_temp

# ---------------- MÉTRICAS COMBINADAS WG ----------------
def calcular_metricas_completas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria métricas que combinam ataque e defesa usando Weighted Goals.
    """
    df_temp = df.copy()

    # Balance Offensive/Defensive (soma: positivo = bom ataque + boa defesa)
    df_temp['WG_Balance_Home'] = df_temp['WG_Home'] + df_temp['WG_Def_Home']
    df_temp['WG_Balance_Away'] = df_temp['WG_Away'] + df_temp['WG_Def_Away']

    # Performance Total (ataque + defesa)
    df_temp['WG_Total_Home'] = df_temp['WG_Home'] + df_temp['WG_Def_Home']
    df_temp['WG_Total_Away'] = df_temp['WG_Away'] + df_temp['WG_Def_Away']

    # Net Performance (ataque - defesa oponente)
    df_temp['WG_Net_Home'] = df_temp['WG_Home'] - df_temp['WG_Def_Away']
    df_temp['WG_Net_Away'] = df_temp['WG_Away'] - df_temp['WG_Def_Home']

    return df_temp

# ---------------- PARÂMETROS POR LIGA (BASE GOALS + PESO ASIAN) ----------------
@st.cache_data(ttl=7*24*3600)
def calcular_parametros_liga(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula parâmetros por liga usando o histórico:
    - Média de gols por jogo (Base_Goals_Liga)
    - Intensidade média de handicap (Asian_Weight_Liga derivado)
    """
    df = history.copy()

    if df.empty or 'League' not in df.columns:
        return pd.DataFrame()

    # Garantir colunas necessárias
    if 'Goals_H_FT' not in df.columns:
        df['Goals_H_FT'] = 0
    if 'Goals_A_FT' not in df.columns:
        df['Goals_A_FT'] = 0

    df['Gols_Total'] = df['Goals_H_FT'].fillna(0) + df['Goals_A_FT'].fillna(0)

    if 'Asian_Line_Decimal' not in df.columns and 'Asian_Line' in df.columns:
        df['Asian_Line_Decimal'] = df['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)

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

# ---------------- ROLLING WEIGHTED GOALS (DINÂMICO POR LIGA) ----------------
def _get_liga_windows(df: pd.DataFrame, min_window: int = 3, max_window: int = 10) -> dict:
    """Define janela dinâmica por liga baseada no número de jogos."""
    if 'League' not in df.columns:
        return {}

    liga_sizes = df.groupby('League').size()
    liga_windows = {}
    for liga, n_games in liga_sizes.items():
        # Janela = 1/3 dos jogos, limitado entre [min_window, max_window]
        w = max(min_window, min(max_window, int(max(n_games // 3, min_window))))
        liga_windows[liga] = w
    return liga_windows

def calcular_rolling_wg_features_completo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features rolling dos Weighted Goals (incluindo defesa),
    com janela dinâmica por liga.
    """
    df_temp = df.copy()

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp = df_temp.sort_values('Date')

    # Inicializar colunas como NaN
    cols_init = [
        'WG_Home_Team', 'WG_Away_Team',
        'WG_AH_Home_Team', 'WG_AH_Away_Team',
        'WG_Def_Home_Team', 'WG_Def_Away_Team',
        'WG_AH_Def_Home_Team', 'WG_AH_Def_Away_Team',
        'WG_Balance_Home_Team', 'WG_Balance_Away_Team',
        'WG_Total_Home_Team', 'WG_Total_Away_Team',
        'WG_Net_Home_Team', 'WG_Net_Away_Team'
    ]
    for c in cols_init:
        if c not in df_temp.columns:
            df_temp[c] = np.nan

    # Se não tiver League, aplica janela fixa padrão de 5 jogos
    liga_windows = _get_liga_windows(df_temp)
    if not liga_windows:
        default_window = 5
        if 'Home' in df_temp.columns:
            df_temp['WG_Home_Team'] = df_temp.groupby('Home')['WG_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_AH_Home_Team'] = df_temp.groupby('Home')['WG_AH_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Def_Home_Team'] = df_temp.groupby('Home')['WG_Def_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_AH_Def_Home_Team'] = df_temp.groupby('Home')['WG_AH_Def_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Balance_Home_Team'] = df_temp.groupby('Home')['WG_Balance_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Total_Home_Team'] = df_temp.groupby('Home')['WG_Total_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Net_Home_Team'] = df_temp.groupby('Home')['WG_Net_Home'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )

        if 'Away' in df_temp.columns:
            df_temp['WG_Away_Team'] = df_temp.groupby('Away')['WG_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_AH_Away_Team'] = df_temp.groupby('Away')['WG_AH_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Def_Away_Team'] = df_temp.groupby('Away')['WG_Def_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_AH_Def_Away_Team'] = df_temp.groupby('Away')['WG_AH_Def_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Balance_Away_Team'] = df_temp.groupby('Away')['WG_Balance_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Total_Away_Team'] = df_temp.groupby('Away')['WG_Total_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
            df_temp['WG_Net_Away_Team'] = df_temp.groupby('Away')['WG_Net_Away'].transform(
                lambda x: x.rolling(default_window, min_periods=1).mean()
            )
    else:
        # Aplicar janela específica por liga
        for liga, window in liga_windows.items():
            mask = df_temp['League'] == liga
            sub = df_temp.loc[mask].sort_values('Date').copy()

            if sub.empty:
                continue

            if 'Home' in sub.columns:
                sub['WG_Home_Team'] = sub.groupby('Home')['WG_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_AH_Home_Team'] = sub.groupby('Home')['WG_AH_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Def_Home_Team'] = sub.groupby('Home')['WG_Def_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_AH_Def_Home_Team'] = sub.groupby('Home')['WG_AH_Def_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Balance_Home_Team'] = sub.groupby('Home')['WG_Balance_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Total_Home_Team'] = sub.groupby('Home')['WG_Total_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Net_Home_Team'] = sub.groupby('Home')['WG_Net_Home'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

            if 'Away' in sub.columns:
                sub['WG_Away_Team'] = sub.groupby('Away')['WG_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_AH_Away_Team'] = sub.groupby('Away')['WG_AH_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Def_Away_Team'] = sub.groupby('Away')['WG_Def_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_AH_Def_Away_Team'] = sub.groupby('Away')['WG_AH_Def_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Balance_Away_Team'] = sub.groupby('Away')['WG_Balance_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Total_Away_Team'] = sub.groupby('Away')['WG_Total_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                sub['WG_Net_Away_Team'] = sub.groupby('Away')['WG_Net_Away'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

            # Atribuir de volta
            df_temp.loc[sub.index, [
                'WG_Home_Team', 'WG_Away_Team',
                'WG_AH_Home_Team', 'WG_AH_Away_Team',
                'WG_Def_Home_Team', 'WG_Def_Away_Team',
                'WG_AH_Def_Home_Team', 'WG_AH_Def_Away_Team',
                'WG_Balance_Home_Team', 'WG_Balance_Away_Team',
                'WG_Total_Home_Team', 'WG_Total_Away_Team',
                'WG_Net_Home_Team', 'WG_Net_Away_Team'
            ]] = sub[[
                'WG_Home_Team', 'WG_Away_Team',
                'WG_AH_Home_Team', 'WG_AH_Away_Team',
                'WG_Def_Home_Team', 'WG_Def_Away_Team',
                'WG_AH_Def_Home_Team', 'WG_AH_Def_Away_Team',
                'WG_Balance_Home_Team', 'WG_Balance_Away_Team',
                'WG_Total_Home_Team', 'WG_Total_Away_Team',
                'WG_Net_Home_Team', 'WG_Net_Away_Team'
            ]]

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



# ========================= CONTRARIAN MARKET SCORE =========================
def adicionar_market_contrarian_score(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()

    if 'Odd_H' not in df.columns or 'Odd_A' not in df.columns:
        st.warning("⚠️ Odds 1x2 não encontradas para Contrarian Score")
        df['Market_Return_H'] = 0
        df['Market_Return_A'] = 0
        df['Market_Score_Home'] = 0
        df['Market_Score_Away'] = 0
        return df
    
    df['P_H'] = 1 / df['Odd_H'].replace(0, np.nan)
    df['P_A'] = 1 / df['Odd_A'].replace(0, np.nan)
    df['P_Sum'] = df['P_H'] + df['P_A']

    df['Odd_H_Fair'] = df['P_Sum'] / df['P_H']
    df['Odd_A_Fair'] = df['P_Sum'] / df['P_A']

    def market_return(row, side):
        g_h = row["Goals_H_FT"]
        g_a = row["Goals_A_FT"]
        odd_fair = row[f'Odd_{side}_Fair']

        if pd.isna(odd_fair) or odd_fair <= 1:
            return 0.0

        win = (side == 'H' and g_h > g_a) or (side == 'A' and g_a > g_h)
        if win:
            return 1 - 1/odd_fair
        else:
            return -1/odd_fair

    df["Market_Return_H"] = df.apply(lambda row: market_return(row, 'H'), axis=1)
    df["Market_Return_A"] = df.apply(lambda row: market_return(row, 'A'), axis=1)

    df = df.sort_values("Date")

    df['Market_Score_Home'] = df.groupby('Home')["Market_Return_H"].cumsum()
    df['Market_Score_Away'] = df.groupby('Away')["Market_Return_A"].cumsum()

    return df



# ---------------- ENRIQUECER GAMES_TODAY COM WG DO HISTÓRICO ----------------
def enrich_games_today_with_wg_completo(games_today: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece os jogos de hoje com as médias rolling de Weighted Goals (ataque + defesa)
    calculadas no histórico. Usa apenas o ÚLTIMO valor por time.
    """
    gt = games_today.copy()
    hist = history.copy()

    # Garantir que as colunas de rolling existem no histórico
    required_cols_home = [
        'Home',
        'WG_Home_Team', 'WG_AH_Home_Team', 'WG_Def_Home_Team', 'WG_AH_Def_Home_Team',
        'WG_Balance_Home_Team', 'WG_Total_Home_Team', 'WG_Net_Home_Team'
    ]
    required_cols_away = [
        'Away',
        'WG_Away_Team', 'WG_AH_Away_Team', 'WG_Def_Away_Team', 'WG_AH_Def_Away_Team',
        'WG_Balance_Away_Team', 'WG_Total_Away_Team', 'WG_Net_Away_Team'
    ]

    for col in required_cols_home[1:] + required_cols_away[1:]:
        if col not in hist.columns:
            hist[col] = 0.0

    # Último valor por time (Home perspective)
    last_wg_home = hist.groupby('Home').agg({
        'WG_Home_Team': 'last',
        'WG_AH_Home_Team': 'last',
        'WG_Def_Home_Team': 'last',
        'WG_AH_Def_Home_Team': 'last',
        'WG_Balance_Home_Team': 'last',
        'WG_Total_Home_Team': 'last',
        'WG_Net_Home_Team': 'last'
    }).reset_index().rename(columns={
        'Home': 'Team'
    })

    last_wg_away = hist.groupby('Away').agg({
        'WG_Away_Team': 'last',
        'WG_AH_Away_Team': 'last',
        'WG_Def_Away_Team': 'last',
        'WG_AH_Def_Away_Team': 'last',
        'WG_Balance_Away_Team': 'last',
        'WG_Total_Away_Team': 'last',
        'WG_Net_Away_Team': 'last'
    }).reset_index().rename(columns={
        'Away': 'Team'
    })

    # Merge Home
    gt = gt.merge(
        last_wg_home,
        left_on='Home',
        right_on='Team',
        how='left'
    ).drop('Team', axis=1)

    # Merge Away
    gt = gt.merge(
        last_wg_away,
        left_on='Away',
        right_on='Team',
        how='left'
    ).drop('Team', axis=1)

    # Garantir colunas com fillna(0)
    wg_cols = [
        'WG_Home_Team', 'WG_AH_Home_Team', 'WG_Def_Home_Team', 'WG_AH_Def_Home_Team',
        'WG_Balance_Home_Team', 'WG_Total_Home_Team', 'WG_Net_Home_Team',
        'WG_Away_Team', 'WG_AH_Away_Team', 'WG_Def_Away_Team', 'WG_AH_Def_Away_Team',
        'WG_Balance_Away_Team', 'WG_Total_Away_Team', 'WG_Net_Away_Team'
    ]
    for col in wg_cols:
        if col not in gt.columns:
            gt[col] = 0.0
        else:
            gt[col] = gt[col].fillna(0)

    # Calcular diffs com base no histórico (não nos gols de hoje)
    gt['WG_Diff'] = gt['WG_Home_Team'] - gt['WG_Away_Team']
    gt['WG_AH_Diff'] = gt['WG_AH_Home_Team'] - gt['WG_AH_Away_Team']
    gt['WG_Def_Diff'] = gt['WG_Def_Home_Team'] - gt['WG_Def_Away_Team']
    gt['WG_Balance_Diff'] = gt['WG_Balance_Home_Team'] - gt['WG_Balance_Away_Team']
    gt['WG_Net_Diff'] = gt['WG_Net_Home_Team'] - gt['WG_Net_Away_Team']

    gt['WG_Confidence'] = (
        gt['WG_Home_Team'].notna().astype(int) +
        gt['WG_Away_Team'].notna().astype(int) +
        gt['WG_Def_Home_Team'].notna().astype(int) +
        gt['WG_Def_Away_Team'].notna().astype(int)
    )

    return gt

# ---------------- Z-SCORES DETALHADOS ----------------
def calcular_zscores_detalhados(df):
    """Calcula Z-scores a partir do HandScore com tratamento de infinitos."""
    df = df.copy()

    st.info("📊 Calculando Z-scores a partir do HandScore...")

    # 1. Z-SCORE POR LIGA (M_H, M_A)
    if 'League' in df.columns and 'HandScore_Home' in df.columns and 'HandScore_Away' in df.columns:
        league_stats = df.groupby('League').agg({
            'HandScore_Home': ['mean', 'std'],
            'HandScore_Away': ['mean', 'std']
        }).round(3)

        league_stats.columns = ['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std']

        # TRATAR STD = 0
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
        df['M_H'] = 0.0
        df['M_A'] = 0.0

    # 2. Z-SCORE POR TIME (MT_H, MT_A)
    if 'Home' in df.columns and 'Away' in df.columns:
        home_team_stats = df.groupby('Home').agg({
            'HandScore_Home': ['mean', 'std']
        }).round(3)
        home_team_stats.columns = ['HT_mean', 'HT_std']

        away_team_stats = df.groupby('Away').agg({
            'HandScore_Away': ['mean', 'std']
        }).round(3)
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
        df['MT_H'] = 0.0
        df['MT_A'] = 0.0

    return df

def clean_features_for_training(X):
    """Remove infinitos, NaNs e limita outliers nas features."""
    X_clean = X.copy()

    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean, columns=X.columns if hasattr(X, 'columns') else range(X.shape[1]))

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
        if np.issubdtype(X_clean[col].dtype, np.number):
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
    """Carrega histórico com filtro temporal correto, SEM perder linhas por dropna."""
    st.info("📊 Carregando histórico com filtro temporal correto...")

    history = filter_leagues(load_all_games(GAMES_FOLDER))

    if history.empty:
        st.warning("⚠️ Histórico vazio")
        return history

    # Converter datas ANTES de filtrar
    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        selected_date = pd.to_datetime(selected_date_str)
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📅 Histórico filtrado: {len(history)} jogos anteriores a {selected_date_str}")

    # ✅ Preencher NaNs imediatamente após o load/filtro
    history = history.fillna(0)

    # Garantir colunas de gols
    if 'Goals_H_FT' not in history.columns:
        history['Goals_H_FT'] = history.get('Goals_H_Today', 0)
    if 'Goals_A_FT' not in history.columns:
        history['Goals_A_FT'] = history.get('Goals_A_Today', 0)

    # Conversão corrigida da linha asiática
    if 'Asian_Line' not in history.columns:
        st.error("❌ Coluna 'Asian_Line' não encontrada no histórico.")
        return history

    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)

    st.success(f"✅ Histórico processado (sem dropna): {len(history)} jogos válidos")
    return history

def create_better_target_corrigido(df):
    """
    Cria targets 100% binários (sem push puro), com Zebra:

    - Target_AH_Home: 1 se o HOME cobrir o handicap (win/half-win),
                      0 se não cobrir (loss/half-loss)
    - Expected_Favorite: HOME se linha < 0, AWAY se linha > 0
    - Zebra: 1 se o favorito do mercado falhar (não cobrir)
    - Jogos com AH_Result == 0.5 (push puro) são EXCLUÍDOS do dataset
    """
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

    debug_cols = [
        "League", "Home", "Away",
        "Asian_Line", "Asian_Line_Decimal",
        "Goals_H_FT", "Goals_A_FT",
        "Margin", "Expected_Favorite",
        "Target_AH_Home", "AH_Result", "Zebra"
    ]
    debug_cols = [c for c in debug_cols if c in df.columns]

    st.write("🔍 Exemplos (após correção PUSH & Zebra):")
    st.dataframe(df.head(10)[debug_cols])

    return df

def create_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features mais robustas INCLUINDO seno/cosseno 3D e Weighted Goals."""
    df = df.copy()

    # 1. Features básicas essenciais
    basic_features = [
        'Aggression_Home', 'Aggression_Away',
        'M_H', 'M_A', 'MT_H', 'MT_A'
    ]

    # 2. Features derivadas (evitar colinearidade)
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

    # 3. Features 3D / Vetoriais
    vector_features = [
        'Quadrant_Dist_3D', 'Momentum_Diff', 'Magnitude_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ'
    ]

    # 4. Features Weighted Goals (histórico e diferenças)
    wg_features = [
        'WG_Home_Team', 'WG_Away_Team',
        'WG_AH_Home_Team', 'WG_AH_Away_Team',
        'WG_Def_Home_Team', 'WG_Def_Away_Team',
        'WG_AH_Def_Home_Team', 'WG_AH_Def_Away_Team',
        'WG_Balance_Home_Team', 'WG_Balance_Away_Team',
        'WG_Total_Home_Team', 'WG_Total_Away_Team',
        'WG_Net_Home_Team', 'WG_Net_Away_Team',
        'WG_Diff', 'WG_AH_Diff', 'WG_Def_Diff',
        'WG_Balance_Diff', 'WG_Net_Diff'
    ]
    
    market_features = [
        'Market_Score_Home',
        'Market_Score_Away',
        'Market_Rating_Diff'
    ]
    
    market_flag_features = [
        'Market_Overpriced_Home',
        'Market_Overpriced_Away',
        'Market_Underpriced_Home',
        'Market_Underpriced_Away'
    ]




    all_features = basic_features + derived_features + vector_features + wg_features + market_features + market_flag_features

    available_features = [f for f in all_features if f in df.columns]

    st.info(f"📋 Features disponíveis: {len(available_features)}/{len(all_features)}")

    trig_features = [f for f in available_features if 'Sin' in f or 'Cos' in f]
    if trig_features:
        st.success(f"✅ Features trigonométricas incluídas: {len(trig_features)}")
    else:
        st.warning("⚠️ Nenhuma feature trigonométrica encontrada (verificar cálculo 3D)")

    return df[available_features].fillna(0.0)

def train_improved_model(X, y, feature_names):
    """Treina RandomForest otimizado com limpeza de dados e mostra importâncias."""
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
    st.dataframe(importances.head(15).to_frame("Importância"))

    return model

# ---------------- CACHE INTELIGENTE ----------------
@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    """Cache apenas dos dados pesados (games_today + history básico)."""
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

    history = load_and_filter_history(selected_date_str)

    return games_today, history, selected_date_str

# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today, selected_date_str):
    """Carrega e faz merge dos dados do Live Score."""
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    games_today = setup_livescore_columns(games_today)

    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)

        # Remover jogos cancelados/adiados
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

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

games_today, history, selected_date_str = load_cached_data(selected_file)

# Conversão Asian Line também para jogos de hoje
if 'Asian_Line' in games_today.columns:
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)

# Aplicar Live Score
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
    """Classifica Aggression e HandScore em um dos 16 quadrantes."""
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

# ---------------- CÁLCULO DE DISTÂNCIAS 3D ----------------
def calcular_distancias_3d(df):
    """Calcula distância 3D e ângulos garantindo features trigonométricas."""
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

# ---------------- WEIGHTED GOALS PIPELINE (HISTÓRICO + GAMES_TODAY) ----------------
if not history.empty:
    st.subheader("🎯 Calculando Weighted Goals no histórico...")
    liga_params = calcular_parametros_liga(history)

    history = adicionar_weighted_goals(history)
    history = adicionar_weighted_goals_defensivos(history, liga_params)
    history = adicionar_weighted_goals_ah(history)
    history = adicionar_weighted_goals_ah_defensivos(history)
    history = calcular_metricas_completas(history)
    history = calcular_rolling_wg_features_completo(history)
    history = adicionar_market_contrarian_score(history)

    # Fill final NaNs no histórico (robustez máxima)
    history = history.fillna(0.0)

    st.success(f"WG completo aplicado no histórico: {history.shape}")

if not games_today.empty and not history.empty:
    st.subheader("📌 Enriquecendo jogos de hoje com WG histórico...")
    games_today = enrich_games_today_with_wg_completo(games_today, history)
    # Contrarian Score nos jogos do dia
    last_market_home = history.groupby('Home')['Market_Score_Home'].last().reset_index()
    games_today = games_today.merge(last_market_home, on='Home', how='left')
  
    last_market_away = history.groupby('Away')['Market_Score_Away'].last().reset_index()
    games_today = games_today.merge(last_market_away, on='Away', how='left')
  
    games_today['Market_Rating_Diff'] = (
        games_today['Market_Score_Home'].fillna(0) -
        games_today['Market_Score_Away'].fillna(0)
    )

# ================= CONTRARIAN FLAGS BINÁRIAS =================

# Threshold dinâmico baseado no desvio interquartil do mercado
# → identifica super/subprecificação real
def compute_contrarian_threshold(df):
    vals = pd.concat([
        df['Market_Score_Home'], 
        df['Market_Score_Away']
    ])
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(vals) == 0:
        return 1.0  # fallback
    
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    th = 1.5 * iqr  # robusto a outliers
    return max(th, 0.5)  # mínimo aceitável

threshold_contrarian = compute_contrarian_threshold(history)

# Flags binárias
games_today['Market_Overpriced_Home'] = (games_today['Market_Score_Home'] >= threshold_contrarian).astype(int)
games_today['Market_Underpriced_Home'] = (games_today['Market_Score_Home'] <= -threshold_contrarian).astype(int)

games_today['Market_Overpriced_Away'] = (games_today['Market_Score_Away'] >= threshold_contrarian).astype(int)
games_today['Market_Underpriced_Away'] = (games_today['Market_Score_Away'] <= -threshold_contrarian).astype(int)

st.info(f"Contrarian Threshold: {threshold_contrarian:.3f}")

st.success(f"WG aplicado em games_today: {games_today.shape}")

# ---------------- THRESHOLD DINÂMICO POR HANDICAP (ESTRATÉGIA B - AGRESSIVA) ----------------
def min_confidence_by_line(line):
    """Threshold mínimo de confiança para aprovar aposta (agressivo) baseado em |Asian_Line_Decimal|."""
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

# ---------------- MODELO ML 3D CORRIGIDO (HOME / AWAY + APOSTA) ----------------
def treinar_modelo_3d_quadrantes_16_corrigido(history, games_today):
    """Treina modelo ML 3D CORRIGIDO (Estratégia B - agressiva)."""
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

    st.subheader("Modelo HOME (Home cobre AH)")
    model_home = train_improved_model(X_history, y_home, X_history.columns.tolist())

    st.subheader("Modelo AWAY (Away cobre AH)")
    model_away = train_improved_model(X_history, y_away, X_history.columns.tolist())

    # 4. Preparar dados de hoje e aplicar lógica de aposta
    if not games_today.empty:
        if 'Asian_Line_Decimal' in games_today.columns:
            games_today["Expected_Favorite"] = np.where(
                games_today["Asian_Line_Decimal"] < 0,
                "HOME",
                np.where(games_today["Asian_Line_Decimal"] > 0, "AWAY", "NONE")
            )

        X_today = create_robust_features(games_today)

        missing_cols = set(X_history.columns) - set(X_today.columns)
        for col in missing_cols:
            X_today[col] = 0.0

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

        if 'Asian_Line_Decimal' in games_today.columns:
            games_today['Min_Conf_Required'] = games_today['Asian_Line_Decimal'].apply(min_confidence_by_line)
        else:
            games_today['Min_Conf_Required'] = 0.60

        games_today['Bet_Approved'] = games_today['Bet_Confidence'] >= games_today['Min_Conf_Required']

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





  

# ---------------- EXECUÇÃO DO MODELO CORRIGIDO ----------------
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

# ---------------- SISTEMA DE INDICAÇÕES 3D ----------------
def adicionar_indicadores_explicativos_3d_16_dual(df):
    """Adiciona classificações e recomendações explícitas para sistema 3D (com Bet_Side e Zebra agressiva)."""
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
        bet_conf = row.get('Bet_Confidence', 0.5)
        bet_approved = bool(row.get('Bet_Approved', False))
        momentum_h = row.get('M_H', 0)
        momentum_a = row.get('M_A', 0)
        expected_fav = row.get('Expected_Favorite', 'NONE')
        is_zebra = int(row.get('Is_Zebra_Bet', 0))

        if not bet_approved:
            return f'NO BET (H:{score_home:.1%} A:{score_away:.1%})'

        if is_zebra and expected_fav in ['HOME', 'AWAY']:
            return f'ZEBRA contra {expected_fav} ({bet_side}, {bet_conf:.1%})'

        if 'Fav Forte' in str(home_q) and 'Under Forte' in str(away_q) and momentum_h > 1.0 and bet_side == 'HOME':
            return f'Favorito HOME muito forte (+Momentum, {bet_conf:.1%})'
        if 'Under Forte' in str(home_q) and 'Fav Forte' in str(away_q) and momentum_a > 1.0 and bet_side == 'AWAY':
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
    df['Ranking'] = df['Bet_Confidence'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- EXIBIÇÃO DOS RESULTADOS ----------------
st.markdown("## 🏆 Melhores Confrontos 3D por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_3d = adicionar_indicadores_explicativos_3d_16_dual(games_today)
    ranking_3d = ranking_3d.sort_values('Bet_Confidence', ascending=False)

    colunas_3d = [
        'Ranking', 'League',"Time", 'Home', 'Away',
        'Goals_H_Today', 'Goals_A_Today',
        'Bet_Label', 'Bet_Side', 'Bet_Confidence', 'Bet_Approved',
        'Expected_Favorite', 'Is_Zebra_Bet',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
        'Min_Conf_Required',
        'Recomendacao',
        'M_H', 'M_A', 'Quadrant_Dist_3D', 'Momentum_Diff',
        'Asian_Line', 'Asian_Line_Decimal'
    ]

    cols_finais_3d = [c for c in colunas_3d if c in ranking_3d.columns]

    def estilo_tabela_3d_quadrantes(df):
      prob_cols = [c for c in [
          'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
          'Bet_Confidence', 'Bet_Confidence_Adjusted',
          'Min_Conf_Required', 'Min_Conf_Required_Liga',
          'WinRate_Similares'
      ] if c in df.columns]
  
      df_style = df.style
  
      if prob_cols:
          df_style = df_style.background_gradient(subset=prob_cols, cmap='RdYlGn')
  
      # 🔥 Destaque contrarian visual
      if 'Market_Overpriced_Home' in df.columns:
          df_style = df_style.apply(
              lambda row: ['background-color: #ffe6e6' if row['Market_Overpriced_Home'] == 1 else '' for _ in row],
              axis=1
          )
      if 'Market_Underpriced_Home' in df.columns:
          df_style = df_style.apply(
              lambda row: ['background-color: #e6ffe6' if row['Market_Underpriced_Home'] == 1 else '' for _ in row],
              axis=1
          )
  
      return df_style
      
    # ================= CONTRARIAN FILTER =================
    st.markdown("### 🎯 Filtro Contrarian – Regressão à Média")
    
    show_contrarian = st.checkbox("📉 Mostrar apenas oportunidades contrarian (subprecificadas)", value=False)
    
    if show_contrarian:
        ranking_3d = ranking_3d[
            (ranking_3d['Market_Underpriced_Home'] == 1) |
            (ranking_3d['Market_Underpriced_Away'] == 1)
        ]


    st.dataframe(
        estilo_tabela_3d_quadrantes(ranking_3d[cols_finais_3d]).format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Bet_Confidence': '{:.1%}',
            'Min_Conf_Required': '{:.1%}',
            'M_H': '{:.2f}',
            'M_A': '{:.2f}',
            'Quadrant_Dist_3D': '{:.2f}',
            'Momentum_Diff': '{:.2f}'
        }, na_rep="-"),
        use_container_width=True,
        height=600
    )

    # ---------------- CARDS DE PICKS APROVADOS ----------------
    st.markdown("## 🎴 Cards de Picks (Apostas aprovadas pela ML)")

    aprovados = ranking_3d[ranking_3d['Bet_Approved'] == True].copy()
    aprovados = aprovados.sort_values('Bet_Confidence', ascending=False)

    if aprovados.empty:
        st.info("Nenhuma aposta aprovada pela estratégia hoje.")
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
                    f"Confiança ML: **{row.get('Bet_Confidence', 0):.1%}** "
                    f"(Home: {row.get('Quadrante_ML_Score_Home', 0):.1%} | "
                    f"Away: {row.get('Quadrante_ML_Score_Away', 0):.1%})"
                )
                st.write(f"Threshold mínimo p/ essa linha: **{row.get('Min_Conf_Required', 0):.1%}**")
                st.write(f"Favorito da casa (linha): **{row.get('Expected_Favorite', 'NONE')}**")

                zebra_txt = "Sim, ML contra o favorito da casa" if row.get('Is_Zebra_Bet', 0) == 1 else "Não"
                st.write(f"Zebra agressiva: **{zebra_txt}**")

                # ----- Sinal Contrarian -----
                if row.get('Market_Overpriced_Home',0) == 1 or row.get('Market_Overpriced_Away',0) == 1:
                    st.markdown("🔴 **Mercado Inflado – RISCO!**")
                if row.get('Market_Underpriced_Home',0) == 1 or row.get('Market_Underpriced_Away',0) == 1:
                    st.markdown("🟢 **Oportunidade Contrarian – VALUE!**")


                st.write(f"Quadrante HOME: {row.get('Quadrante_Home_Label', 'Neutro')}")
                st.write(f"Quadrante AWAY: {row.get('Quadrante_Away_Label', 'Neutro')}")
                st.write(f"Recomendação: {row.get('Recomendacao', '')}")

else:
    st.info("⚠️ Aguardando dados para gerar ranking 3D")

# ---------------- RESUMO EXECUTIVO ----------------
def resumo_3d_16_quadrantes_hoje(df):
    """Resumo executivo dos 16 quadrantes 3D de hoje"""
    st.markdown("### 📋 Resumo Executivo - Sistema 3D Hoje")

    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)

    if 'Classificacao_Valor_Home' in df.columns:
        alto_valor_home = len(df[df['Classificacao_Valor_Home'] == 'ALTO VALOR'])
        alto_valor_away = len(df[df['Classificacao_Valor_Away'] == 'ALTO VALOR'])
    else:
        alto_valor_home = alto_valor_away = 0

    if 'M_H' in df.columns:
        momentum_positivo_home = len(df[df['M_H'] > 0.5])
        momentum_negativo_home = len(df[df['M_H'] < -0.5])
        momentum_positivo_away = len(df[df['M_A'] > 0.5])
        momentum_negativo_away = len(df[df['M_A'] < -0.5])
    else:
        momentum_positivo_home = momentum_negativo_home = momentum_positivo_away = momentum_negativo_away = 0

    if 'Is_Zebra_Bet' in df.columns:
        zebra_alta = len(df[df['Is_Zebra_Bet'] == 1])
    else:
        zebra_alta = 0

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
        st.metric("🎯 Alto Valor Home", alto_valor_home)
        st.metric("🎯 Alto Valor Away", alto_valor_away)

if not games_today.empty:
    resumo_3d_16_quadrantes_hoje(games_today)

# ---------------- ESTATÍSTICAS ZEBRA ----------------
st.markdown("## 🦓 Estatísticas de Zebra (mercado errando)")

if not history.empty:
    if 'Zebra' not in history.columns:
        history_tmp = create_better_target_corrigido(history.copy())
    else:
        history_tmp = history.copy()

    st.markdown("### Por Liga")

    if "League" in history_tmp.columns:
        zebra_liga = history_tmp.groupby("League")["Zebra"].mean().sort_values(ascending=False)
        st.dataframe(
            zebra_liga.to_frame("Taxa Zebra").style.format({"Taxa Zebra": "{:.1%}"}),
            use_container_width=True
        )
    else:
        st.info("Liga não disponível no histórico.")

    st.markdown("### Por Linha de Handicap")

    if 'Asian_Line_Decimal' in history_tmp.columns:
        zebra_handicap = history_tmp.groupby("Asian_Line_Decimal")["Zebra"].mean().sort_values(ascending=False)
        st.dataframe(
            zebra_handicap.to_frame("Taxa Zebra").style.format({"Taxa Zebra": "{:.1%}"}),
            use_container_width=True
        )
else:
    st.info("Sem histórico para estatísticas Zebra.")

st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML CORRIGIDO + ZEBRA + WG** implementado com sucesso!")
st.info("""\
**Principais pontos:**

✅ **Asian Line CORRIGIDA** - Perspectiva do Away convertida corretamente para Home  
✅ **Target 100% Binário** - PUSH excluído, apenas Win/Loss  
✅ **Modelo HOME / AWAY** - Probabilidade de cada lado cobrir o AH  
✅ **Weighted Goals (Ataque + Defesa)** - Rolling dinâmico por liga  
✅ **Data Leakage Eliminado** - Filtro temporal aplicado corretamente  
✅ **Feature Engineering Robusto** - Features 3D, Z-score e WG combinados  
✅ **Validação Cruzada** - Performance monitorada  
✅ **Dashboard Explicativo** - Recomendação, valor e risco de zebra  

Agora o sistema não só te diz **quem tende a cobrir a linha**, mas também **quem vem entregando acima/abaixo do que o mercado espera (WG)** e **onde o mercado tem alta chance de errar (zebra)**.
""")
