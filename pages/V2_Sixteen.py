# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
from datetime import datetime
import math
import plotly.graph_objects as go

# ==========================================================
# BLOCO 1 – CONFIGURAÇÕES BÁSICAS
# ==========================================================
st.set_page_config(page_title="Análise de Quadrantes - Bet Indicator", layout="wide")
st.title("🎯 Análise de 16 Quadrantes - ML Avançado (Home & Away)")

PAGE_PREFIX = "QuadrantesML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "coppa", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------

def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas do Live Score existam no DataFrame"""
    df = df.copy()
    for col in ['Goals_H_Today', 'Goals_A_Today', 'Home_Red', 'Away_Red']:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ==========================================================
# BLOCO 2 – FUNÇÕES AUXILIARES BÁSICAS
# ==========================================================

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalizar colunas de gols FT em merges antigos
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

def convert_asian_line_to_home(line_str):
    """
    Converte a string da linha asiática (do ponto de vista do AWAY da base)
    para a linha decimal do HOME, corrigindo casos tipo '-1/1.5' -> '-1/-1.5'.
    """
    if pd.isna(line_str) or line_str == "":
        return np.nan

    s = str(line_str).strip().replace(" ", "").replace(",", ".")

    # Zero
    if s in ("0", "0.0", "-0", "+0"):
        return 0.0

    try:
        # Split (ex: -0.5/1, -1/1.5 etc)
        if "/" in s:
            parts = [float(p) for p in s.split("/")]

            if len(parts) == 2:
                a, b = parts

                # Correção automática de base errada (-1/1.5 -> -1/-1.5)
                if a < 0 < b:
                    b = -abs(b)
                if b < 0 < a:
                    a = -abs(a)

                parts = [a, b]

            away_avg = sum(parts) / len(parts)
        else:
            away_avg = float(s)

        home_line = -away_avg
        # Arredonda para múltiplos de 0.25
        return round(home_line * 4) / 4.0

    except Exception:
        st.warning(f"⚠️ Linha de handicap inválida: {line_str}")
        return np.nan

# ==========================================================
# BLOCO 3 – HANDICAP ASIÁTICO V9 (OFICIAL) + LIVE RESULT
# ==========================================================

def handicap_favorito_v9(margin, line):
    """
    Handicap para FAVORITOS (linhas negativas) – V9 tabelado
    margin: gols_favorito - gols_oponente
    line: linha negativa (ex: -0.25, -1.25, etc)
    """
    line_abs = abs(line)

    # Linhas inteiras (-1, -2, ...)
    if float(line_abs).is_integer():
        if margin > line_abs:
            return 1
        elif margin == line_abs:
            return 0
        else:
            return -1

    # -0.25
    if line == -0.25:
        if margin > 0:
            return 1
        elif margin == 0:
            return -0.5
        else:
            return -1

    # -0.50
    if line == -0.50:
        if margin > 0:
            return 1
        else:
            return -1

    # -0.75
    if line == -0.75:
        if margin >= 2:
            return 1
        elif margin == 1:
            return 0.5
        else:
            return -1

    # -1.25
    if line == -1.25:
        if margin >= 2:
            return 1
        elif margin == 1:
            return -0.5
        else:
            return -1

    # -1.50
    if line == -1.50:
        if margin >= 2:
            return 1
        else:
            return -1

    # -1.75
    if line == -1.75:
        if margin >= 3:
            return 1
        elif margin == 2:
            return 0.5
        else:
            return -1

    # -2.00
    if line == -2.00:
        if margin > 2:
            return 1
        elif margin == 2:
            return 0
        else:
            return -1

    return np.nan

def handicap_underdog_v9(margin, line):
    """
    Handicap para UNDERDOGS (linhas positivas) – V9 tabelado
    margin: gols_favorito - gols_underdog  (sempre home - away do ponto de vista do favorito)
    line: linha positiva (ex: +0.25, +1.25, etc)
    """
    # Linhas inteiras (0, +1, +2, ...)
    if float(line).is_integer():
        if margin >= -line:
            return 1
        elif margin == -(line + 1):
            return 0
        else:
            return -1

    # +0.25
    if line == 0.25:
        if margin > 0:
            return 1
        elif margin == 0:
            return 0.5
        else:
            return -1

    # +0.50
    if line == 0.50:
        if margin >= 0:
            return 1
        else:
            return -1

    # +0.75
    if line == 0.75:
        if margin >= 0:
            return 1
        elif margin == -1:
            return -0.5
        else:
            return -1

    # +1.00
    if line == 1.00:
        if margin >= -1:
            return 1
        else:
            return -1

    # +1.25
    if line == 1.25:
        if margin >= -1:
            return 1
        elif margin == -2:
            return 0.5
        else:
            return -1

    # +1.50
    if line == 1.50:
        if margin >= -1:
            return 1
        else:
            return -1

    # +1.75
    if line == 1.75:
        if margin >= -1:
            return 1
        elif margin == -2:
            return -0.5
        else:
            return -1

    # +2.00
    if line == 2.00:
        if margin >= -2:
            return 1
        elif margin == -3:
            return 0
        else:
            return -1

    return np.nan

def handicap_home_v9(row):
    """Resultado AH para apostas no HOME (usando Goals_H_Today / Goals_A_Today)"""
    margin = row['Goals_H_Today'] - row['Goals_A_Today']
    line = row['Asian_Line_Decimal']

    if pd.isna(margin) or pd.isna(line):
        return np.nan

    if line < 0:  # Home favorito
        return handicap_favorito_v9(margin, line)
    else:         # Home underdog
        return handicap_underdog_v9(margin, line)

def handicap_away_v9(row):
    """Resultado AH para apostas no AWAY (espelha Home)"""
    margin = row['Goals_A_Today'] - row['Goals_H_Today']  # perspectiva do away
    line = -row['Asian_Line_Decimal']                     # inverte linha

    if pd.isna(margin) or pd.isna(line):
        return np.nan

    if line < 0:  # Away favorito
        return handicap_favorito_v9(margin, line)
    else:         # Away underdog
        return handicap_underdog_v9(margin, line)

def apply_handicap_results_v9(df: pd.DataFrame) -> pd.DataFrame:
    """Avalia handicap asiático (HOME/AWAY) com base na Recomendacao + V9 e calcula Profit_Final."""
    df = df.copy()

    def process_row(row):
        rec = str(row.get('Recomendacao', '')).upper()
        odd_home = row.get('Odd_H_Asi', row.get('Odd_H', np.nan))
        odd_away = row.get('Odd_A_Asi', row.get('Odd_A', np.nan))

        if pd.isna(row.get('Goals_H_Today')) or pd.isna(row.get('Goals_A_Today')) or pd.isna(row.get('Asian_Line_Decimal')):
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

        if 'HOME' in rec:
            val = handicap_home_v9(row)
            odd = odd_home
            side_bet = 'HOME'
        elif 'AWAY' in rec:
            val = handicap_away_v9(row)
            odd = odd_away
            side_bet = 'AWAY'
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

        # Mapear para outcome [1, 0.5, 0, -0.5, -1] + profit usando odd do handicap
        if val == 1:
            profit = (odd - 1) if not pd.isna(odd) else 1
            return pd.Series([1, "FULL WIN", profit, side_bet])
        elif val == 0.5:
            profit = (odd - 1) / 2 if not pd.isna(odd) else 0.5
            return pd.Series([0.5, "HALF WIN", profit, side_bet])
        elif val == 0:
            return pd.Series([0, "PUSH", 0, side_bet])
        elif val == -0.5:
            return pd.Series([-0.5, "HALF LOSS", -0.5, side_bet])
        elif val == -1:
            return pd.Series([-1, "LOSS", -1, side_bet])
        else:
            return pd.Series([np.nan, np.nan, np.nan, side_bet])

    df[['Outcome_Final', 'Handicap_Result_Final', 'Profit_Final', 'Side_Bet']] = df.apply(process_row, axis=1)
    df['Quadrante_Correct'] = df['Outcome_Final'] > 0

    return df

def generate_live_summary_v9(df: pd.DataFrame) -> dict:
    """Resumo Live usando sistema V9."""
    finished_games = df.dropna(subset=['Outcome_Final'])

    if finished_games.empty:
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": 0,
            "Apostas Quadrante": 0,
            "Acertos Quadrante": 0,
            "Winrate Quadrante": "0%",
            "Profit Quadrante": 0,
            "ROI Quadrante": "0%",
            "Full Wins": 0,
            "Half Wins": 0,
            "Pushes": 0,
            "Half Losses": 0,
            "Losses": 0
        }

    quadrante_bets = finished_games[finished_games['Outcome_Final'].notna()]
    total_bets = len(quadrante_bets)
    correct_bets = (quadrante_bets['Outcome_Final'] > 0).sum()
    winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
    total_profit = quadrante_bets['Profit_Final'].sum()
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

    full_wins = (quadrante_bets['Outcome_Final'] == 1).sum()
    half_wins = (quadrante_bets['Outcome_Final'] == 0.5).sum()
    pushes = (quadrante_bets['Outcome_Final'] == 0).sum()
    half_losses = (quadrante_bets['Outcome_Final'] == -0.5).sum()
    losses = (quadrante_bets['Outcome_Final'] == -1).sum()

    return {
        "Total Jogos": len(df),
        "Jogos Finalizados": len(finished_games),
        "Apostas Quadrante": total_bets,
        "Acertos Quadrante": int(correct_bets),
        "Winrate Quadrante": f"{winrate:.1f}%",
        "Profit Quadrante": f"{total_profit:.2f}u",
        "ROI Quadrante": f"{roi:.1f}%",
        "Full Wins": int(full_wins),
        "Half Wins": int(half_wins),
        "Pushes": int(pushes),
        "Half Losses": int(half_losses),
        "Losses": int(losses)
    }





def decidir_aposta_final(df: pd.DataFrame, prob_min=0.57):
    df = df.copy()

    def escolher(row):
        prob_home = row['Quadrante_ML_Score_Home']
        prob_away = row['Quadrante_ML_Score_Away']
        side_ml = row['ML_Side']
        label = row['Edge_Label']

        # Regras de aposta baseadas no preço
        if label == "🟢 FAVORITO BARATO (valor no HOME)":
            side = "HOME"
        elif label == "🟢 UNDERDOG BARATO (valor no AWAY)":
            side = "AWAY"
        elif label == "🔴 FAVORITO CARO (valor no AWAY)":
            side = "AWAY"
        elif label == "🔴 UNDERDOG CARO (valor no AWAY)":
            side = "HOME"
        else:
            return "SKIP"

        # Validação mínima de confiança
        prob_side = prob_home if side == "HOME" else prob_away
        if prob_side < prob_min:
            return "SKIP"

        # Se preço e probabilidade discordam → SKIP
        if side != side_ml:
            return "SKIP"

        return side

    df['BET_SIDE_FINAL'] = df.apply(escolher, axis=1)
    return df




# ==========================================================
# BLOCO 4 – FUNÇÃO genérica calc_handicap_result (TREINO)
# ==========================================================

def calc_handicap_result(margin, asian_line_decimal, invert=False):
    """
    Resultado médio (0, 0.5, 1.0) de um handicap asiático já em decimal,
    usado para criar TARGETS (treino).

    invert=True → espelha a margem (para Away)
    """
    if pd.isna(asian_line_decimal):
        return np.nan

    if invert:
        margin = -margin

    line = asian_line_decimal
    line_abs = abs(line)
    frac = line_abs - int(line_abs)

    # 0.25 → split em (0, 0.5)
    if frac == 0.25:
        base = int(line_abs) if line >= 0 else -int(line_abs)
        l1 = base
        l2 = base + 0.5 if line >= 0 else base - 0.5

        r1 = 1.0 if margin > l1 else (0.5 if margin == l1 else 0.0)
        r2 = 1.0 if margin > l2 else (0.5 if margin == l2 else 0.0)
        return (r1 + r2) / 2.0

    # 0.75 → split em (0.5, 1.0)
    if frac == 0.75:
        base = int(line_abs) if line >= 0 else -int(line_abs)
        l1 = base + 0.5 if line >= 0 else base - 0.5
        l2 = base + 1.0 if line >= 0 else base - 1.0

        r1 = 1.0 if margin > l1 else (0.5 if margin == l1 else 0.0)
        r2 = 1.0 if margin > l2 else (0.5 if margin == l2 else 0.0)
        return (r1 + r2) / 2.0

    # Linha inteira / meia
    return 1.0 if margin > line else (0.5 if margin == line else 0.0)

# ==========================================================
# BLOCO 5 – CARREGAMENTO DOS DADOS + LIVE SCORE
# ==========================================================

st.info("📂 Carregando dados para análise de 16 quadrantes...")

# Arquivos de GamesDay
files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
if not files:
    st.warning("Nenhum CSV encontrado na pasta GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

# Jogos do dia
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

def load_and_merge_livescore(games_today: pd.DataFrame, selected_date_str: str) -> pd.DataFrame:
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

    if not os.path.exists(livescore_file):
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

    st.info(f"📡 LiveScore file found: {livescore_file}")
    results_df = pd.read_csv(livescore_file)

    # Remove cancelados/adiados
    results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

    required_cols = [
        'Id', 'status', 'home_goal', 'away_goal',
        'home_ht_goal', 'away_ht_goal',
        'home_corners', 'away_corners',
        'home_yellow', 'away_yellow',
        'home_red', 'away_red'
    ]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        st.error(f"❌ LiveScore missing columns: {missing}")
        return games_today

    games_today = games_today.merge(
        results_df,
        on='Id',
        how='left',
        suffixes=('', '_RAW')
    )

    # Atualiza gols FT apenas em jogos finalizados
    games_today['Goals_H_Today'] = games_today['home_goal']
    games_today['Goals_A_Today'] = games_today['away_goal']
    games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

    games_today['Home_Red'] = games_today['home_red']
    games_today['Away_Red'] = games_today['away_red']

    st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
    return games_today

games_today = load_and_merge_livescore(games_today, selected_date_str)

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# Conversão Asian Line
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_home)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_home)

history = history.dropna(subset=['Asian_Line_Decimal'])
st.info(f"📊 Histórico com Asian Line válida: {len(history)} jogos")

# Filtro temporal anti-leak
if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

# Targets Home/Away (sempre binário, push=0)
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]

history["Target_AH_Home"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"]) > 0.5 else 0,
    axis=1
)

history["Target_AH_Away"] = history.apply(
    lambda r: 1 if calc_handicap_result(-r["Margin"], -r["Asian_Line_Decimal"]) > 0.5 else 0,
    axis=1
)


# ==========================================================
# BLOCO 6 – SISTEMA DE 16 QUADRANTES
# ==========================================================

st.markdown("## 🎯 Sistema de 16 Quadrantes")

QUADRANTES_16 = {
    # Fav Forte
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte",       "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado",    "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro",      "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},

    # Fav Moderado
    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte",       "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado",    "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro",      "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},

    # Under Moderado
    9: {"nome": "Under Moderado Neutro",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte","agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},

    # Under Forte
    13: {"nome": "Under Forte Neutro",      "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",    "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",       "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for q_id, cfg in QUADRANTES_16.items():
        if cfg['agg_min'] <= agg <= cfg['agg_max'] and cfg['hs_min'] <= hs <= cfg['hs_max']:
            return q_id
    return 0

for df_ in [games_today, history]:
    df_['Quadrante_Home'] = df_.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    df_['Quadrante_Away'] = df_.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )

# ==========================================================
# BLOCO 7 – DISTÂNCIAS / ANGULOS (M_H, M_A, MT_H, MT_A)
# ==========================================================

def calcular_distancias_quadrantes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = ['M_H', 'M_A', 'MT_H', 'MT_A', 'HandScore_Home', 'HandScore_Away']
    if not all(c in df.columns for c in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        st.warning(f"⚠️ Colunas ausentes para distâncias V2: {missing}")
        df[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos', 'Quadrant_Angle']] = np.nan
        return df

    dx = df['M_A'] - df['M_H']
    dy = df['MT_A'] - df['MT_H']

    df['Quadrant_Dist'] = np.sqrt(dx**2 + dy**2)
    df['Quadrant_Separation'] = 0.5 * (dx + dy)

    angle = np.arctan2(dy, dx)
    df['Quadrant_Sin'] = np.sin(angle)
    df['Quadrant_Cos'] = np.cos(angle)

    ang_deg = np.degrees(np.abs(angle))
    ang_deg = ang_deg.apply(lambda x: x if x <= 90 else 180 - x)
    df['Quadrant_Angle'] = ang_deg

    mean_hs = (df['HandScore_Home'].fillna(0) + df['HandScore_Away'].fillna(0)) / 2
    weight = 1 + (mean_hs / 60).clip(-0.5, 0.5)
    df['Quadrant_Dist'] *= weight

    return df

games_today = calcular_distancias_quadrantes(games_today)
history = calcular_distancias_quadrantes(history)

# ==========================================================
# BLOCO 8 – GRÁFICOS MATPLOTLIB DOS 16 QUADRANTES
# ==========================================================

def plot_quadrantes_16(df: pd.DataFrame, side="Home"):
    fig, ax = plt.subplots(figsize=(14, 10))

    cores_quadrantes_16 = {
        1: 'lightblue', 2: 'deepskyblue', 3: 'blue', 4: 'darkblue',
        5: 'lightgreen', 6: 'mediumseagreen', 7: 'green', 8: 'darkgreen',
        9: 'moccasin', 10: 'gold', 11: 'orange', 12: 'chocolate',
        13: 'lightcoral', 14: 'indianred', 15: 'red', 16: 'darkred'
    }

    col_q = f'Quadrante_{side}'
    col_agg = f'Aggression_{side}'
    col_hs = f'HandScore_{side}'

    for q in range(1, 17):
        mask = df[col_q] == q
        if mask.any():
            x = df.loc[mask, col_agg]
            y = df.loc[mask, col_hs]
            ax.scatter(
                x, y,
                c=cores_quadrantes_16.get(q, 'gray'),
                s=55, alpha=0.8, edgecolors='k', linewidths=0.4,
                label=f"Q{q} – {QUADRANTES_16[q]['nome']}"
            )

    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    for y in [-45, -30, -15, 15, 30, 45]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    ax.set_xlabel(f"Performance na Liga (M_{side})", fontsize=11)
    ax.set_ylabel(f"Forma vs Próprio Padrão (MT_{side})", fontsize=11)
    ax.set_title(f"🎯 16 Quadrantes – {side}", fontsize=14, weight='bold')

    handles = []
    base_leg = [(1, "Fav Forte"), (5, "Fav Moderado"), (9, "Under Moderado"), (13, "Under Forte")]
    for base, nome in base_leg:
        handles.append(plt.Line2D(
            [0], [0], marker='o', color='w', label=nome,
            markerfacecolor=cores_quadrantes_16[base], markersize=10
        ))
    ax.legend(handles=handles, loc='upper left', fontsize=10, title="Categorias")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

st.markdown("### 📈 Visualização dos 16 Quadrantes")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

# ==========================================================
# BLOCO 9 – VISUALIZAÇÃO INTERATIVA PLOTLY (M vs MT)
# ==========================================================

st.markdown("## 🎯 Visualização Interativa – Distância entre Times (Home × Away)")

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
    st.warning("⚠️ Nenhuma coluna 'League' encontrada — exibindo todos os jogos.")
    df_filtered = games_today.copy()

max_n = len(df_filtered)
if max_n == 0:
    st.warning("Nenhum jogo disponível para visualização.")
else:
    n_to_show = st.slider("Quantos confrontos exibir (Top por distância):", 10, min(max_n, 200), 40, step=5)
    df_plot = df_filtered.nlargest(n_to_show, "Quadrant_Dist").reset_index(drop=True)

    fig = go.Figure()

    for _, row in df_plot.iterrows():
        xh, xa = row.get("M_H", np.nan), row.get("M_A", np.nan)
        yh, ya = row.get("MT_H", np.nan), row.get("MT_A", np.nan)

        fig.add_trace(go.Scatter(
            x=[xh, xa],
            y=[yh, ya],
            mode="lines+markers",
            line=dict(width=1),
            marker=dict(size=5),
            hoverinfo="text",
            hovertext=(
                f"<b>{row.get('Home','N/A')} vs {row.get('Away','N/A')}</b><br>"
                f"🏆 {row.get('League','N/A')}<br>"
                f"📊 Home M: {row.get('M_H',np.nan):.2f} | MT: {row.get('MT_H',np.nan):.2f}<br>"
                f"📊 Away M: {row.get('M_A',np.nan):.2f} | MT: {row.get('MT_A',np.nan):.2f}<br>"
                f"📏 Distância: {row.get('Quadrant_Dist',np.nan):.2f}"
            ),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=df_plot["M_H"],
        y=df_plot["MT_H"],
        mode="markers+text",
        name="Home",
        marker=dict(size=8, opacity=0.8),
        text=df_plot["Home"],
        textposition="top center",
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=df_plot["M_A"],
        y=df_plot["MT_A"],
        mode="markers+text",
        name="Away",
        marker=dict(size=8, opacity=0.8),
        text=df_plot["Away"],
        textposition="top center",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"Top {n_to_show} Distâncias – 16 Quadrantes"
              + (f" | {selected_league}" if selected_league != "⚽ Todas as ligas" else ""),
        xaxis_title="Performance na Liga (M)",
        yaxis_title="Forma vs Próprio Padrão (MT)",
        template="plotly_white",
        height=700,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# BLOCO 10 – MODELO ML (HOME/AWAY) + REGRESSOR HANDICAP IDEAL
# ==========================================================

usar_catboost = st.checkbox("🚀 Usar CatBoost ao invés de RandomForest", value=True)

def treinar_modelo_quadrantes_16_dual(history: pd.DataFrame, games_today: pd.DataFrame):
    history_local = history.copy()
    games_today_local = games_today.copy()

    # =============================
    # 🔹 One-Hot Encoding Categorias
    # =============================
    qh = pd.get_dummies(history_local['Quadrante_Home'], prefix='QH')
    qa = pd.get_dummies(history_local['Quadrante_Away'], prefix='QA')
    ligas = pd.get_dummies(history_local['League'], prefix='League')

    # =============================
    # 🔹 Extras iniciais (sem Pred_Handicap)
    # =============================
    extras = history_local[['Quadrant_Dist', 'Quadrant_Separation',
                            'Quadrant_Sin', 'Quadrant_Cos', 'Quadrant_Angle']].fillna(0)

    # =============================
    # 🔹 Features iniciais sem leakage
    # =============================
    X_base = pd.concat([qh, qa, ligas, extras], axis=1)
    y_home = history_local['Target_AH_Home']
    y_away = history_local['Target_AH_Away']

    # =============================
    # 🔮 REGRESSÃO PRIMEIRO
    # =============================
    modelo_handicap = CatBoostRegressor(
        depth=7, learning_rate=0.06,
        iterations=800, loss_function='RMSE',
        random_seed=42, verbose=False
    )
    modelo_handicap.fit(X_base, history_local['Asian_Line_Decimal'])

    # Predições históricas → Feature nova
    history_local['Pred_Handicap'] = modelo_handicap.predict(X_base)

    # =============================
    # 🔁 Recriar features com Pred_Handicap incluída
    # =============================
    extras = history_local[['Quadrant_Dist', 'Quadrant_Separation',
                            'Quadrant_Sin', 'Quadrant_Cos', 'Quadrant_Angle',
                            'Pred_Handicap']].fillna(0)

    X = pd.concat([qh, qa, ligas, extras], axis=1)

    # =============================
    # 🎯 CLASSIFICAÇÃO HOME & AWAY
    # =============================
    if usar_catboost:
        modelo_home = CatBoostClassifier(
            depth=7, learning_rate=0.08,
            iterations=600, loss_function='Logloss',
            random_seed=42, verbose=False
        )
        modelo_away = CatBoostClassifier(
            depth=7, learning_rate=0.08,
            iterations=600, loss_function='Logloss',
            random_seed=42, verbose=False
        )
    else:
        modelo_home = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)
        modelo_away = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)

    modelo_home.fit(X, y_home)
    modelo_away.fit(X, y_away)

    # =============================
    # 🔮 Predições para games_today
    # =============================
    qh_today = pd.get_dummies(games_today_local['Quadrante_Home'], prefix='QH') \
        .reindex(columns=qh.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today_local['Quadrante_Away'], prefix='QA') \
        .reindex(columns=qa.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today_local['League'], prefix='League') \
        .reindex(columns=ligas.columns, fill_value=0)

    # Pred_Handicap para hoje
    extras_today_base = games_today_local[['Quadrant_Dist', 'Quadrant_Separation',
                                           'Quadrant_Sin', 'Quadrant_Cos', 'Quadrant_Angle']].fillna(0)
    X_today_base = pd.concat([qh_today, qa_today, ligas_today, extras_today_base], axis=1)
    games_today_local['Pred_Handicap'] = modelo_handicap.predict(X_today_base)

    # Recriar final com Pred_Handicap
    extras_today = games_today_local[['Quadrant_Dist', 'Quadrant_Separation',
                                      'Quadrant_Sin', 'Quadrant_Cos', 'Quadrant_Angle',
                                      'Pred_Handicap']].fillna(0)

    X_today = pd.concat([qh_today, qa_today, ligas_today, extras_today], axis=1)

    # Classificação com feature ajustada
    games_today_local['Quadrante_ML_Score_Home'] = modelo_home.predict_proba(X_today)[:, 1]
    games_today_local['Quadrante_ML_Score_Away'] = modelo_away.predict_proba(X_today)[:, 1]

    # =============================
    # 🧠 Lado mais provável
    # =============================
    games_today_local['ML_Side'] = np.where(
        games_today_local['Quadrante_ML_Score_Home'] >= games_today_local['Quadrante_ML_Score_Away'],
        'HOME', 'AWAY'
    )
    games_today_local['Quadrante_ML_Score_Main'] = np.where(
        games_today_local['ML_Side'] == 'HOME',
        games_today_local['Quadrante_ML_Score_Home'],
        games_today_local['Quadrante_ML_Score_Away']
    )

    # =============================
    # 📏 Cálculo Handicap Edge
    # =============================
    games_today_local['Handicap_Edge'] = games_today_local['Pred_Handicap'] - games_today_local['Asian_Line_Decimal']

    def classificar_edge(row):
        line = row['Asian_Line_Decimal']
        edge = row['Handicap_Edge']  # P - L
    
        if line < 0:  # HOME favorito
            if edge > 0.50:
                return "🔴 FAVORITO CARO (valor no AWAY)"
            elif edge < -0.50:
                return "🟢 FAVORITO BARATO (valor no HOME)"
            else:
                return "🔵 EQUILIBRADO"
        else:  # HOME underdog
            if edge < -0.50:
                return "🟢 UNDERDOG BARATO (valor no HOME)"
            elif edge > 0.50:
                return "🔴 UNDERDOG CARO (valor no AWAY)"
            else:
                return "🔵 EQUILIBRADO"
    
    games_today_local['Edge_Label'] = games_today_local.apply(classificar_edge, axis=1)


    st.success("✔️ Classificação + Regressão com Pred_Handicap como feature: OK!")

    return modelo_home, modelo_away, modelo_handicap, games_today_local


modelo_home, modelo_away, modelo_handicap, games_today = treinar_modelo_quadrantes_16_dual(history, games_today)

# ==========================================================
# BLOCO 11 – INDICADORES EXPLICATIVOS & PADRÕES
# ==========================================================

def adicionar_indicadores_explicativos_16_dual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(
        lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro')
    )
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(
        lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro')
    )

    # Classificação Home
    cond_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choice_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(cond_home, choice_home, default='⚖️ NEUTRO')

    # Classificação Away
    cond_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choice_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(cond_away, choice_away, default='⚖️ NEUTRO')

    def gerar_recomendacao_16_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']

        if 'Fav Forte' in home_q and 'Under Forte' in away_q:
            return f'💪 FAVORITO HOME FORTE ({score_home:.1%})'
        if 'Under Forte' in home_q and 'Fav Forte' in away_q:
            return f'💪 FAVORITO AWAY FORTE ({score_away:.1%})'
        if 'Fav Moderado' in home_q and 'Under Moderado' in away_q and 'Forte' in away_q:
            return f'🎯 VALUE NO HOME ({score_home:.1%})'
        if 'Under Moderado' in home_q and 'Fav Moderado' in away_q and 'Forte' in home_q:
            return f'🎯 VALUE NO AWAY ({score_away:.1%})'
        if ml_side == 'HOME' and score_home >= 0.60:
            return f'📈 MODELO CONFIA HOME ({score_home:.1%})'
        if ml_side == 'AWAY' and score_away >= 0.60:
            return f'📈 MODELO CONFIA AWAY ({score_away:.1%})'
        if 'Neutro' in home_q and score_away >= 0.58:
            return f'🔄 AWAY EM NEUTRO ({score_away:.1%})'
        if 'Neutro' in away_q and score_home >= 0.58:
            return f'🔄 HOME EM NEUTRO ({score_home:.1%})'
        return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_16_dual, axis=1)

    # Ranking pela prob principal
    if 'Quadrante_ML_Score_Main' not in df.columns:
        df['Quadrante_ML_Score_Main'] = np.where(
            df['ML_Side'] == 'HOME',
            df['Quadrante_ML_Score_Home'],
            df['Quadrante_ML_Score_Away']
        )
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

def calcular_pontuacao_quadrante_16(q_id: int) -> int:
    scores_base = {
        1: 85, 2: 80, 3: 75, 4: 70,
        5: 70, 6: 65, 7: 60, 8: 55,
        9: 50, 10: 45, 11: 40, 12: 35,
        13: 35, 14: 30, 15: 25, 16: 20
    }
    return scores_base.get(q_id, 50)

def gerar_score_combinado_16(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Score_Base_Home'] = df['Quadrante_Home'].apply(calcular_pontuacao_quadrante_16)
    df['Score_Base_Away'] = df['Quadrante_Away'].apply(calcular_pontuacao_quadrante_16)
    df['Score_Combinado'] = df['Score_Base_Home'] * 0.6 + df['Score_Base_Away'] * 0.4

    if 'Quadrante_ML_Score_Main' not in df.columns:
        df['Quadrante_ML_Score_Main'] = np.where(
            df['ML_Side'] == 'HOME',
            df['Quadrante_ML_Score_Home'],
            df['Quadrante_ML_Score_Away']
        )

    df['Score_Final'] = df['Score_Combinado'] * df['Quadrante_ML_Score_Main']

    cond = [
        df['Score_Final'] >= 60,
        df['Score_Final'] >= 45,
        df['Score_Final'] >= 30,
        df['Score_Final'] < 30
    ]
    choi = ['🌟 ALTO POTENCIAL', '💼 VALOR SOLIDO', '⚖️ NEUTRO', '🔴 BAIXO POTENCIAL']
    df['Classificacao_Potencial'] = np.select(cond, choi, default='⚖️ NEUTRO')

    return df

# ==========================================================
# BLOCO 12 – TABELA PRINCIPAL + LIVE MONITOR
# ==========================================================

st.markdown("## 🏆 Melhores Confrontos por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    ranking_quadrantes = games_today.copy()

    ranking_quadrantes = adicionar_indicadores_explicativos_16_dual(ranking_quadrantes)
    ranking_quadrantes = gerar_score_combinado_16(ranking_quadrantes)
    ranking_quadrantes = apply_handicap_results_v9(ranking_quadrantes)

    st.markdown("## 📡 Live Score Monitor - 16 Quadrantes (v9 Validado)")
    live_summary = generate_live_summary_v9(ranking_quadrantes)
    st.json(live_summary)

    # Ordenar privilegiando EDGE + Score
    ranking_quadrantes = ranking_quadrantes.sort_values(
        ['Handicap_Edge', 'Quadrante_ML_Score_Main', 'Score_Final'],
        ascending=[False, False, False]
    )

    colunas_possiveis = [
        'League', 'Home', 'Away',
        # Odds e Handicap
        'Asian_Line_Decimal', 'Pred_Handicap', 'Handicap_Edge', 'Edge_Label',
        # Probabilidades ML
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 'Quadrante_ML_Score_Main',
        # Recomendação final
        'ML_Side', 'Recomendacao',
        # Indicadores táticos
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrant_Dist', 'Quadrant_Angle',
        # Scoring
        'Score_Final', 'Classificacao_Potencial',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        # Live score / validation
        'Goals_H_Today', 'Goals_A_Today',
        'Outcome_Final', 'Handicap_Result_Final', 'Profit_Final',
        'Quadrante_Correct', 'Home_Red', 'Away_Red'
    ]

    cols_finais = [c for c in colunas_possiveis if c in ranking_quadrantes.columns]

    def estilo_tabela_16_quadrantes(df: pd.DataFrame):
        def cor_classificacao(val):
            s = str(val)
            if '🌟 ALTO POTENCIAL' in s or '💼 VALOR SOLIDO' in s:
                return 'font-weight: bold'
            if '🔴 BAIXO POTENCIAL' in s:
                return 'color: #b30000; font-weight: bold'
            if '🏆 ALTO VALOR' in s:
                return 'font-weight: bold; font-weight: bold'
            if '🔴 ALTO RISCO' in s:
                return 'font-weight: bold; font-weight: bold'
            if 'VALUE' in s:
                return 'font-weight: bold'
            if 'EVITAR' in s:
                return 'font-weight: bold'
            return ''

        def cor_edge(val):
            s = str(val)
            if '🟢' in s:
                return 'font-weight: bold'
            if '🟡' in s:
                return 'font-weight: bold'
            if '🔵' in s:
                return 'font-weight: bold'
            if '🟠' in s:
                return 'font-weight: bold'
            if '🔴' in s:
                return 'font-weight: bold'
            return ''

        col_style = [c for c in ['Classificacao_Potencial',
                                 'Classificacao_Valor_Home',
                                 'Classificacao_Valor_Away',
                                 'Recomendacao'] if c in df.columns]

        styler = df.style
        if col_style:
            styler = styler.applymap(cor_classificacao, subset=col_style)
        if 'Edge_Label' in df.columns:
            styler = styler.applymap(cor_edge, subset=['Edge_Label'])

        if 'Quadrante_ML_Score_Home' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')
        if 'Quadrante_ML_Score_Away' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')
        if 'Score_Final' in df.columns:
            styler = styler.background_gradient(subset=['Score_Final'], cmap='RdYlGn')
        if 'Handicap_Edge' in df.columns:
            styler = styler.background_gradient(subset=['Handicap_Edge'], cmap='PiYG')

        return styler

    # remover colunas duplicadas e resetar índice para evitar erro do Styler
    ranking_quadrantes = ranking_quadrantes.loc[:, ~ranking_quadrantes.columns.duplicated()]
    ranking_quadrantes = ranking_quadrantes.reset_index(drop=True)
    ranking_quadrantes.index.name = None

    st.dataframe(
        estilo_tabela_16_quadrantes(ranking_quadrantes[cols_finais]).format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Pred_Handicap': '{:.2f}',
            'Handicap_Edge': '{:.2f}',
            'Home_Red': '{:.0f}',
            'Away_Red': '{:.0f}',
            'Profit_Final': '{:.2f}',
            'Outcome_Final': '{:.1f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Quadrante_ML_Score_Main': '{:.1%}',
            'Score_Final': '{:.1f}'
        }, na_rep="-"),
        use_container_width=True
    )

else:
    st.info("⚠️ Aguardando dados para gerar ranking de 16 quadrantes")




games_today = decidir_aposta_final(games_today, prob_min=0.57)


# ==========================================================
# BLOCO 13 – RESUMO EXECUTIVO (OPCIONAL)
# ==========================================================

def resumo_16_quadrantes_hoje(df: pd.DataFrame):
    st.markdown("### 📋 Resumo Executivo - 16 Quadrantes Hoje")
    if df.empty:
        st.info("Nenhum dado disponível para resumo")
        return

    total_jogos = len(df)
    alto_potencial = len(df[df['Classificacao_Potencial'] == '🌟 ALTO POTENCIAL'])
    valor_solido = len(df[df['Classificacao_Potencial'] == '💼 VALOR SOLIDO'])

    alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
    alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])

    home_recomendado = len(df[df['ML_Side'] == 'HOME'])
    away_recomendado = len(df[df['ML_Side'] == 'AWAY'])

    fav_forte = len(df[df['Quadrante_Home'].isin([1,2,3,4]) | df['Quadrante_Away'].isin([1,2,3,4])])
    under_forte = len(df[df['Quadrante_Home'].isin([13,14,15,16]) | df['Quadrante_Away'].isin([13,14,15,16])])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("🌟 Alto Potencial", alto_potencial)
    with col2:
        st.metric("🎯 Alto Valor Home", alto_valor_home)
        st.metric("🎯 Alto Valor Away", alto_valor_away)
    with col3:
        st.metric("📊 Home vs Away", f"{home_recomendado} : {away_recomendado}")
        st.metric("💼 Valor Sólido", valor_solido)
    with col4:
        st.metric("⚔️ Fav Forte (qualquer lado)", fav_forte)
        st.metric("⚔️ Under Forte (qualquer lado)", under_forte)

    st.markdown("#### 📊 Distribuição de Recomendações")
    if 'Recomendacao' in df.columns:
        st.dataframe(df['Recomendacao'].value_counts(), use_container_width=True)

if not games_today.empty and 'Classificacao_Potencial' in games_today.columns:
    resumo_16_quadrantes_hoje(games_today)

st.markdown("---")

# ==========================================================
# BLOCO FINAL – MENSAGEM DE SUCESSO
# ==========================================================

st.success("🎯 **Sistema de 16 Quadrantes ML + Regressão de Handicap Ideal** implementado com sucesso!")
st.info("""
**Resumo das funcionalidades principais:**
- 🔢 16 quadrantes (Fav Forte / Moderado / Under Moderado / Forte)
- 🤖 Dual Model (Home & Away) com CatBoost/RandomForest
- 📏 Regressão de **Handicap Ideal** (`Pred_Handicap`) por jogo
- 📐 `Handicap_Edge` com alertas 🟢🟡🔴 (linha boa / ok / esmagada)
- 📡 Live Score V9 com `Outcome_Final`, `Profit_Final`, `ROI`
- 📊 Tabela principal já ordenada por Edge + Probabilidade + Score
""")
