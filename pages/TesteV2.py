##### BLOCO 1: IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS #####

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

st.set_page_config(page_title="Análise de Quadrantes - Bet Indicator", layout="wide")
st.title("🎯 Análise de 16 Quadrantes - ML Avançado (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML"
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

##### BLOCO 2: FUNÇÕES AUXILIARES BÁSICAS #####

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

def convert_asian_line_to_home(value):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Caso simples — número único
    if "/" not in value:
        try:
            num = float(value)
            return -num  # Inverte sinal (Away → Home)
        except ValueError:
            return np.nan

    # Caso duplo — média dos dois lados
    try:
        parts = [float(p) for p in value.split("/")]
        avg = np.mean(parts)
        # Mantém o sinal do primeiro número
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        # Inverte o sinal no final (Away → Home)
        return -result
    except ValueError:
        return np.nan


##### BLOCO 3: FUNÇÕES HANDICAP ASIÁTICO V9 CORRIGIDAS (TABELAS OFICIAIS) #####

def handicap_favorito_v9(margin, line):
    """
    Calcula handicap para FAVORITOS (linhas negativas)
    margin: gols_home - gols_away
    line: linha negativa (ex: -0.25, -1.25, etc)
    """
    line_abs = abs(line)
    
    # Linhas inteiras (-1, -2, etc)
    if line_abs.is_integer():
        if margin > line_abs:
            return 1      # Win
        elif margin == line_abs:
            return 0      # Push
        else:
            return -1     # Lose
    
    # Linha -0.25
    elif line == -0.25:
        if margin > 0:
            return 1      # Win
        elif margin == 0:
            return -0.5   # Half lose
        else:
            return -1     # Lose
    
    # Linha -0.50
    elif line == -0.50:
        if margin > 0:
            return 1      # Win
        else:
            return -1     # Lose
    
    # Linha -0.75
    elif line == -0.75:
        if margin >= 2:
            return 1      # Win by 2+
        elif margin == 1:
            return 0.5    # Half win
        else:
            return -1     # Lose
    
    # Linha -1.25
    elif line == -1.25:
        if margin >= 2:
            return 1      # Win by 2+
        elif margin == 1:
            return -0.5   # Half lose
        else:
            return -1     # Lose
    
    # Linha -1.50
    elif line == -1.50:
        if margin >= 2:
            return 1      # Win by 2+
        else:
            return -1     # Lose
    
    # Linha -1.75
    elif line == -1.75:
        if margin >= 3:
            return 1      # Win by 3+
        elif margin == 2:
            return 0.5    # Half win
        else:
            return -1     # Lose
    
    # Linha -2.00
    elif line == -2.00:
        if margin > 2:
            return 1      # Win by 3+
        elif margin == 2:
            return 0      # Push
        else:
            return -1     # Lose
    
    return np.nan

def handicap_underdog_v9(margin, line):
    """
    Calcula handicap para UNDERDOGS (linhas positivas)
    margin: gols_home - gols_away  
    line: linha positiva (ex: +0.25, +1.25, etc)
    """
    # Linhas inteiras (0, +1, +2, etc)
    if line.is_integer():
        if margin >= -line:
            return 1      # Win ou empate
        elif margin == -(line + 1):
            return 0      # Push (perde por exatamente line+1)
        else:
            return -1     # Lose
    
    # Linha +0.25
    elif line == 0.25:
        if margin > 0:
            return 1      # Win
        elif margin == 0:
            return 0.5    # Half win
        else:
            return -1     # Lose
    
    # Linha +0.50
    elif line == 0.50:
        if margin >= 0:
            return 1      # Win ou Draw
        else:
            return -1     # Lose
    
    # Linha +0.75
    elif line == 0.75:
        if margin >= 0:
            return 1      # Win ou Draw
        elif margin == -1:
            return -0.5   # Half lose (lose by 1)
        else:
            return -1     # Lose
    
    # Linha +1.00
    elif line == 1.00:
        if margin >= -1:
            return 1      # Win, Draw ou Lose by 1
        else:
            return -1     # Lose by 2+
    
    # Linha +1.25
    elif line == 1.25:
        if margin >= -1:
            return 1      # Win, Draw ou Lose by 1
        elif margin == -2:
            return 0.5    # Half win (lose by 2)
        else:
            return -1     # Lose by 3+
    
    # Linha +1.50
    elif line == 1.50:
        if margin >= -1:
            return 1      # Win, Draw ou Lose by 1
        else:
            return -1     # Lose by 2+
    
    # Linha +1.75
    elif line == 1.75:
        if margin >= -1:
            return 1      # Win, Draw ou Lose by 1
        elif margin == -2:
            return -0.5   # Half lose (lose by 2)
        else:
            return -1     # Lose by 3+
    
    # Linha +2.00
    elif line == 2.00:
        if margin >= -2:
            return 1      # Win, Draw ou Lose by 1-2
        elif margin == -3:
            return 0      # Push (lose by 3)
        else:
            return -1     # Lose by 4+
    
    return np.nan

def handicap_home_v9(row):
    """Calcula handicap para apostas no HOME"""
    margin = row['Goals_H_Today'] - row['Goals_A_Today']
    line = row['Asian_Line_Decimal']
    
    if line < 0:  # Home é favorito
        return handicap_favorito_v9(margin, line)
    else:  # Home é underdog
        return handicap_underdog_v9(margin, line)

def handicap_away_v9(row):
    """Calcula handicap para apostas no AWAY"""
    margin = row['Goals_A_Today'] - row['Goals_H_Today']  
    line = -row['Asian_Line_Decimal']  # Inverte a linha
    
    if line < 0:  # Away é favorito
        return handicap_favorito_v9(margin, line)
    else:  # Away é underdog
        return handicap_underdog_v9(margin, line)

def apply_handicap_results_v9(df):
    """Aplica a avaliação de Handicap Asiático e lucro (v9 CORRIGIDO)"""
    df = df.copy()
    
    def process_row(row):
        """Processa cada linha para determinar outcome e profit"""
        rec = str(row.get('Recomendacao', '')).upper()
        odd_home = row.get('Odd_H_Asi', np.nan)
        odd_away = row.get('Odd_A_Asi', np.nan)
        
        # Pular se não há recomendação clara ou dados incompletos
        if pd.isna(row.get('Goals_H_Today')) or pd.isna(row.get('Goals_A_Today')) or pd.isna(row.get('Asian_Line_Decimal')):
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
        
        # Determinar qual lado apostar baseado na recomendação
        if 'HOME' in rec:
            val = handicap_home_v9(row)  # ← NOVA FUNÇÃO
            odd = odd_home
            side_bet = 'HOME'
        elif 'AWAY' in rec:
            val = handicap_away_v9(row)  # ← NOVA FUNÇÃO
            odd = odd_away  
            side_bet = 'AWAY'
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

        # Mapear outcome para resultado e profit
        if val == 1: 
            profit = odd  if not pd.isna(odd) else 1
            return pd.Series([1, "FULL WIN", profit, side_bet])
        elif val == 0.5: 
            profit = odd / 2 if not pd.isna(odd) else 0.5
            return pd.Series([0.5, "HALF WIN", profit, side_bet])
        elif val == 0: 
            return pd.Series([0, "PUSH", 0, side_bet])
        elif val == -0.5: 
            return pd.Series([-0.5, "HALF LOSS", -0.5, side_bet])
        elif val == -1: 
            return pd.Series([-1, "LOSS", -1, side_bet])
        else: 
            return pd.Series([np.nan, np.nan, np.nan, side_bet])

    # Aplicar a todas as linhas
    df[['Outcome_Final', 'Handicap_Result_Final', 'Profit_Final', 'Side_Bet']] = df.apply(process_row, axis=1)
    
    # Calcular se a recomendação estava correta
    df['Quadrante_Correct'] = df['Outcome_Final'] > 0
    
    return df

def generate_live_summary_v9(df):
    """Gera resumo em tempo real usando o sistema v9 CORRIGIDO"""
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
    
    # Estatísticas detalhadas dos outcomes
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


##### BLOCO 15: FUNÇÃO CALC_HANDICAP_RESULT (COMPATIBILIDADE) #####

def calc_handicap_result(margin, asian_line_decimal, invert=False):
    """
    Calcula resultado do handicap asiático usando linha já convertida para decimal.
    Mantida para compatibilidade com código existente.
    """
    if pd.isna(asian_line_decimal):
        return np.nan
    
    if invert:
        margin = -margin
    
    # Para linhas fracionadas (0.25, 0.75, etc.), simulamos o split
    line_abs = abs(asian_line_decimal)
    fractional_part = line_abs - int(line_abs)
    
    if fractional_part == 0.25:
        # Linha do tipo 0.25 (equivale a 0/0.5) - split em duas apostas
        base_line = int(line_abs) if asian_line_decimal >= 0 else -int(line_abs)
        line1 = base_line
        line2 = base_line + 0.5 if asian_line_decimal >= 0 else base_line - 0.5
        
        result1 = 1.0 if margin > line1 else (0.5 if margin == line1 else 0.0)
        result2 = 1.0 if margin > line2 else (0.5 if margin == line2 else 0.0)
        
        return (result1 + result2) / 2.0
    
    elif fractional_part == 0.75:
        # Linha do tipo 0.75 (equivale a 0.5/1) - split em duas apostas
        base_line = int(line_abs) if asian_line_decimal >= 0 else -int(line_abs)
        line1 = base_line + 0.5 if asian_line_decimal >= 0 else base_line - 0.5
        line2 = base_line + 1.0 if asian_line_decimal >= 0 else base_line - 1.0
        
        result1 = 1.0 if margin > line1 else (0.5 if margin == line1 else 0.0)
        result2 = 1.0 if margin > line2 else (0.5 if margin == line2 else 0.0)
        
        return (result1 + result2) / 2.0
    
    else:
        # Linha inteira ou meia (0, 0.5, 1.0, etc.) - aposta única
        return 1.0 if margin > asian_line_decimal else (0.5 if margin == asian_line_decimal else 0.0)




##### BLOCO 4: CARREGAMENTO E PREPARAÇÃO DOS DADOS #####

st.info("📂 Carregando dados para análise de 16 quadrantes...")

# Seleção de arquivo do dia
files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

# Jogos do dia
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today, selected_date_str):
    """Carrega e faz merge dos dados do Live Score"""
    
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    
    # Setup das colunas
    games_today = setup_livescore_columns(games_today)
    
    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)
        
        # Filtrar jogos cancelados/adiados
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
            # Fazer merge com os jogos do dia
            games_today = games_today.merge(
                results_df,
                left_on='Id',
                right_on='Id',
                how='left',
                suffixes=('', '_RAW')
            )
            
            # Atualizar gols apenas para jogos finalizados
            games_today['Goals_H_Today'] = games_today['home_goal']
            games_today['Goals_A_Today'] = games_today['away_goal']
            games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
            
            # Atualizar cartões vermelhos
            games_today['Home_Red'] = games_today['home_red']
            games_today['Away_Red'] = games_today['away_red']
            
            st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
            return games_today
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

# Aplicar Live Score
games_today = load_and_merge_livescore(games_today, selected_date_str)

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# ---------------- CONVERSÃO ASIAN LINE ----------------
# Aplicar conversão no histórico e jogos de hoje
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_home)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_home)

# Filtrar apenas jogos com linha válida no histórico
history = history.dropna(subset=['Asian_Line_Decimal'])
st.info(f"📊 Histórico com Asian Line válida: {len(history)} jogos")

# Filtro anti-leakage temporal
if "Date" in history.columns:
    try:
        selected_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
    except Exception as e:
        st.error(f"Erro ao aplicar filtro temporal: {e}")

# Targets AH históricos
history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Target_AH_Home"] = history.apply(
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=False) > 0.5 else 0, axis=1
)

##### BLOCO 5: SISTEMA DE 16 QUADRANTES - DEFINIÇÕES #####

st.markdown("## 🎯 Sistema de 16 Quadrantes")

QUADRANTES_16 = {
    # 🔵 QUADRANTE 1-4: FORTE FAVORITO (+0.75 a +1.0)
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte",       "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado",    "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro",      "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},
    
    # 🟢 QUADRANTE 5-8: FAVORITO MODERADO (+0.25 a +0.75)
    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte",       "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado",    "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro",      "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},
    
    # 🟡 QUADRANTE 9-12: UNDERDOG MODERADO (-0.75 a -0.25)
    9: {"nome": "Under Moderado Neutro",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},
    
    # 🔴 QUADRANTE 13-16: FORTE UNDERDOG (-1.0 a -0.75)
    13: {"nome": "Under Forte Neutro",    "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",  "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",     "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    """Classifica Aggression e HandScore em um dos 16 quadrantes"""
    if pd.isna(agg) or pd.isna(hs):
        return 0  # Neutro/Indefinido
    
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
            
        if agg_ok and hs_ok:
            return quadrante_id
    
    return 0  # Caso não se enquadre em nenhum quadrante

# Aplicar classificação aos dados
games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)

history['Quadrante_Home'] = history.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
history['Quadrante_Away'] = history.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)

##### BLOCO 6: CÁLCULO DE DISTÂNCIAS E VETORES #####

def calcular_distancias_quadrantes(df):
    """
    V2 - Calcula distâncias e ângulos entre Home e Away considerando:
      - Eixo X: z-score na liga (M_H, M_A)
      - Eixo Y: z-score relativo ao próprio time (MT_H, MT_A)
      - Ponderação de magnitude pelo HandScore médio
    """
    df = df.copy()
    required_cols = ['M_H', 'M_A', 'MT_H', 'MT_A', 'HandScore_Home', 'HandScore_Away']
    if not all(col in df.columns for col in required_cols):
        st.warning(f"⚠️ Colunas ausentes para V2: {[c for c in required_cols if c not in df.columns]}")
        df[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos']] = np.nan
        return df

    # 🧭 Vetores Home → Away
    dx = df['M_A'] - df['M_H']       # Força relativa na liga
    dy = df['MT_A'] - df['MT_H']     # Forma relativa ao próprio time

    # 📏 Distância euclidiana base
    df['Quadrant_Dist'] = np.sqrt(dx**2 + dy**2)

    # 🎯 Separação linear combinada
    df['Quadrant_Separation'] = 0.5 * (dy + dx)

    # 🧮 Ângulo direcional e projeções trigonométricas
    angle = np.arctan2(dy, dx)
    df['Quadrant_Sin'] = np.sin(angle)
    df['Quadrant_Cos'] = np.cos(angle)

    # 🎚️ Ângulo absoluto em graus (0°–90°)
    df['Quadrant_Angle'] = np.degrees(np.abs(angle))
    df['Quadrant_Angle'] = df['Quadrant_Angle'].apply(lambda x: x if x <= 90 else 180 - x)

    # ⚖️ Ponderação de confiança usando HandScore médio
    mean_hs = (df['HandScore_Home'].fillna(0) + df['HandScore_Away'].fillna(0)) / 2
    weight = 1 + (mean_hs / 60).clip(-0.5, 0.5)  # Limita impacto extremo
    df['Quadrant_Dist'] = df['Quadrant_Dist'] * weight

    return df


# Aplicar ao games_today
games_today = calcular_distancias_quadrantes(games_today)

##### BLOCO 7: VISUALIZAÇÕES DOS 16 QUADRANTES #####

def plot_quadrantes_16(df, side="Home"):
    """Plot dos 16 quadrantes com cores distintas e legenda por categoria."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 🎨 Cores nomeadas (tons claros = neutro / escuros = extremos)
    cores_quadrantes_16 = {
        1: 'lightblue', 2: 'deepskyblue', 3: 'blue', 4: 'darkblue',          # Fav Forte
        5: 'lightgreen', 6: 'mediumseagreen', 7: 'green', 8: 'darkgreen',    # Fav Moderado
        9: 'moccasin', 10: 'gold', 11: 'orange', 12: 'chocolate',            # Under Moderado
        13: 'lightcoral', 14: 'indianred', 15: 'red', 16: 'darkred'          # Under Forte
    }

    # 🔹 Plotar cada ponto de acordo com o quadrante
    for quadrante_id in range(1, 17):
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            cor = cores_quadrantes_16.get(quadrante_id, 'gray')
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(
                x, y, c=cor, s=55, alpha=0.8, edgecolors='k', linewidths=0.4,
                label=f"Q{quadrante_id} – {QUADRANTES_16[quadrante_id]['nome']}"
            )

    # 🔲 Linhas divisórias
    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    for y in [-45, -30, -15, 15, 30, 45]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # 🏷️ Anotações dos quadrantes (posições médias)
    annot_config = [
        (0.875, 52.5, "Fav Forte\nMuito Forte", 8),
        (0.875, 37.5, "Fav Forte\nForte", 8),
        (0.875, 22.5, "Fav Forte\nModerado", 8),
        (0.875, 0, "Fav Forte\nNeutro", 8),
        (0.5, 52.5, "Fav Moderado\nMuito Forte", 8),
        (0.5, 37.5, "Fav Moderado\nForte", 8),
        (0.5, 22.5, "Fav Moderado\nModerado", 8),
        (0.5, 0, "Fav Moderado\nNeutro", 8),
        (-0.5, 0, "Under Moderado\nNeutro", 8),
        (-0.5, -22.5, "Under Moderado\nModerado", 8),
        (-0.5, -37.5, "Under Moderado\nForte", 8),
        (-0.5, -52.5, "Under Moderado\nMuito Forte", 8),
        (-0.875, 0, "Under Forte\nNeutro", 8),
        (-0.875, -22.5, "Under Forte\nModerado", 8),
        (-0.875, -37.5, "Under Forte\nForte", 8),
        (-0.875, -52.5, "Under Forte\nMuito Forte", 8)
    ]
    for x, y, text, fontsize in annot_config:
        ax.text(x, y, text, ha='center', fontsize=fontsize, weight='bold')

    # 🔧 Configurações gerais
    ax.set_xlabel(f"Performance na Liga (M_{side})", fontsize=11)
    ax.set_ylabel(f"Forma vs Próprio Padrão (MT_{side})", fontsize=11)
    ax.set_title(f"🎯 16 Quadrantes – {side}", fontsize=14, weight='bold')

    # 🔖 Legenda agrupada por família
    handles, labels = ax.get_legend_handles_labels()
    ordem = [
        (1, "Fav Forte"), (5, "Fav Moderado"),
        (9, "Under Moderado"), (13, "Under Forte")
    ]
    legenda_labels = []
    for base, nome in ordem:
        cor_exemplo = cores_quadrantes_16[base]
        legenda_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=nome,
                                         markerfacecolor=cor_exemplo, markersize=10))
    ax.legend(handles=legenda_labels, loc='upper left', fontsize=10, title="Categorias Principais")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Exibir gráficos
st.markdown("### 📈 Visualização dos 16 Quadrantes")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_16(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_16(games_today, "Away"))

##### BLOCO 8: VISUALIZAÇÃO INTERATIVA COM PLOTLY #####

import plotly.graph_objects as go

st.markdown("## 🎯 Visualização Interativa – Distância entre Times (Home × Away)")

# Filtros interativos
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
    st.warning("⚠️ Nenhuma coluna de 'League' encontrada — exibindo todos os jogos.")
    df_filtered = games_today.copy()

# Controle de número de confrontos
max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância):", 10, min(max_n, 200), 40, step=5)

# Preparar dados
df_plot = df_filtered.nlargest(n_to_show, "Quadrant_Dist").reset_index(drop=True)

# Criar gráfico Plotly
fig = go.Figure()

# Vetores Home → Away
for _, row in df_plot.iterrows():
    xh, xa = row["M_H"], row["M_A"]
    yh, ya = row["MT_H"], row["MT_A"]

    fig.add_trace(go.Scatter(
        x=[xh, xa],
        y=[yh, ya],
        mode="lines+markers",
        line=dict(color="gray", width=1),
        marker=dict(size=5),
        hoverinfo="text",
        hovertext=(
            f"<b>{row['Home']} vs {row['Away']}</b><br>"
            f"🏆 {row.get('League','N/A')}<br>"
            f"📊 Home M: {row.get('M_H','N/A'):.2f} | MT: {row.get('MT_H','N/A'):.2f}<br>"  # NOVO
            f"📊 Away M: {row.get('M_A','N/A'):.2f} | MT: {row.get('MT_A','N/A'):.2f}<br>"  # NOVO
            f"🎯 Home: {QUADRANTES_16.get(row['Quadrante_Home'], {}).get('nome', 'N/A')}<br>"
            f"🎯 Away: {QUADRANTES_16.get(row['Quadrante_Away'], {}).get('nome', 'N/A')}<br>"
            f"📏 Distância: {row['Quadrant_Dist']:.2f}"
        ),
        showlegend=False
    ))

# Pontos Home e Away
fig.add_trace(go.Scatter(
    x=df_plot["M_H"],
    y=df_plot["MT_H"],
    mode="markers+text",
    name="Home",
    marker=dict(color="royalblue", size=8, opacity=0.8),
    text=df_plot["Home"],
    textposition="top center",
    hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=df_plot["M_A"],
    y=df_plot["MT_A"],
    mode="markers+text",
    name="Away",
    marker=dict(color="orangered", size=8, opacity=0.8),
    text=df_plot["Away"],
    textposition="top center",
    hoverinfo="skip"
))

# Linha de referência
fig.add_trace(go.Scatter(
    x=[-3, 3],
    y=[ 0, 0],
    mode="lines",
    line=dict(color="limegreen", width=2, dash="dash"),
    name="Eixo X"
))

# Linha de referência
fig.add_trace(go.Scatter(
    x=[ 0, 0],
    y=[-2.2, 2.2],
    mode="lines",
    line=dict(color="limegreen", width=2, dash="dash"),
    name="Eixo Y"
))

# Layout
titulo = f"Top {n_to_show} Distâncias – 16 Quadrantes"
if selected_league != "⚽ Todas as ligas":
    titulo += f" | {selected_league}"

fig.update_layout(
    title=titulo,
    xaxis_title="Performance na Liga (M)",
    yaxis_title="Forma vs Próprio Padrão (MT)",
    template="plotly_white",
    height=700,
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, width='stretch')




##### BLOCO 14: SISTEMA DE REGRESSÃO À MÉDIA + ML EM 2 ESTÁGIOS #####

st.markdown("## 🧠 ML em 2 Estágios - Regressão à Média + Principal")


# 🔥 CORREÇÃO CRÍTICA: GARANTIR QUE AS DISTÂNCIAS EXISTAM
st.info("🔧 Garantindo cálculo de distâncias...")

# Aplicar cálculo de distâncias se não existirem
if 'Quadrant_Dist' not in history.columns:
    history = calcular_distancias_quadrantes(history)
    st.success("✅ Distâncias calculadas para histórico")

if 'Quadrant_Dist' not in games_today.columns:
    games_today = calcular_distancias_quadrantes(games_today) 
    st.success("✅ Distâncias calculadas para jogos de hoje")

# VERIFICAR SE AS COLUNAS NECESSÁRIAS EXISTEM
required_distance_cols = ['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos', 'Quadrant_Angle']
missing_in_history = [col for col in required_distance_cols if col not in history.columns]
missing_in_today = [col for col in required_distance_cols if col not in games_today.columns]

if missing_in_history:
    st.error(f"❌ Colunas de distância faltando no histórico: {missing_in_history}")
    # Criar colunas vazias para evitar erro
    for col in missing_in_history:
        history[col] = 0.0

if missing_in_today:
    st.error(f"❌ Colunas de distância faltando em games_today: {missing_in_today}") 
    for col in missing_in_today:
        games_today[col] = 0.0
        

def calcular_regressao_media_avancada(df):
    """Calcula features avançadas de regressão à média"""
    df = df.copy()
    
    # Verificar se colunas necessárias existem
    required_cols = ['M_H', 'MT_H', 'M_A', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas ausentes para regressão: {missing_cols}")
        # Inicializar colunas com NaN
        for col in ['Z_Excesso_M_Home', 'Z_Excesso_MT_Home', 'Z_Excesso_M_Away', 'Z_Excesso_MT_Away',
                   'Fator_Regressao_Home', 'Fator_Regressao_Away', 'Historico_Extremos_Home', 
                   'Historico_Extremos_Away', 'Tendencia_Regressao_Home', 'Tendencia_Regressao_Away']:
            df[col] = np.nan
        return df
    
    # 1. ESTATÍSTICAS BÁSICAS DA LIGA
    media_liga_M = df['M_H'].mean()
    media_liga_MT = df['MT_H'].mean()
    std_M = df['M_H'].std()
    std_MT = df['MT_H'].std()
    
    # 2. Z-SCORES DE EXCESSO
    df['Z_Excesso_M_Home'] = (df['M_H'] - media_liga_M) / std_M
    df['Z_Excesso_MT_Home'] = (df['MT_H'] - media_liga_MT) / std_MT
    df['Z_Excesso_M_Away'] = (df['M_A'] - media_liga_M) / std_M  
    df['Z_Excesso_MT_Away'] = (df['MT_A'] - media_liga_MT) / std_MT
    
    # 3. FATORES DE REGRESSÃO (não-lineares)
    df['Fator_Regressao_Home'] = 1 - np.exp(-abs(df['Z_Excesso_M_Home']) * 0.4)
    df['Fator_Regressao_Away'] = 1 - np.exp(-abs(df['Z_Excesso_M_Away']) * 0.4)
    
    # 4. HISTÓRICO DE EXTREMOS (quantas vezes foi extremo recentemente)
    df['Historico_Extremos_Home'] = (abs(df['Z_Excesso_M_Home']) > 1.5).astype(int)
    df['Historico_Extremos_Away'] = (abs(df['Z_Excesso_M_Away']) > 1.5).astype(int)
    
    # 5. TENDÊNCIA DE REGRESSÃO (direção do movimento)
    df['Tendencia_Regressao_Home'] = np.where(
        df['Z_Excesso_M_Home'] > 0, 
        -df['Fator_Regressao_Home'],  # Positivo → tende a cair
        df['Fator_Regressao_Home']    # Negativo → tende a subir
    )
    df['Tendencia_Regressao_Away'] = np.where(
        df['Z_Excesso_M_Away'] > 0,
        -df['Fator_Regressao_Away'],
        df['Fator_Regressao_Away']
    )
    
    return df

def treinar_ml_regressao_estagio1(history):
    """Estágio 1: ML especializado em detectar regressão à média"""
    
    # Features específicas para regressão
    features_regressao = [
        'Z_Excesso_M_Home', 'Z_Excesso_MT_Home', 
        'Z_Excesso_M_Away', 'Z_Excesso_MT_Away',
        'Fator_Regressao_Home', 'Fator_Regressao_Away',
        'Historico_Extremos_Home', 'Historico_Extremos_Away',
        'Tendencia_Regressao_Home', 'Tendencia_Regressao_Away',
        'M_H', 'MT_H', 'M_A', 'MT_A',
        'Quadrant_Dist', 'Quadrant_Angle'
    ]
    
    # Filtrar features que existem nos dados
    features_existentes = [f for f in features_regressao if f in history.columns]
    
    if not features_existentes:
        st.error("❌ Nenhuma feature de regressão encontrada!")
        return None, []
    
    X_regressao = history[features_existentes].fillna(0)
    y_regressao = history['Target_AH_Home']
    
    # Verificar se temos dados suficientes
    if len(X_regressao) < 100:
        st.warning("⚠️ Dados insuficientes para treinar ML de regressão")
        return None, features_existentes
    
    model_regressao = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        random_state=42,
        min_samples_split=20,
        class_weight='balanced'
    )
    
    model_regressao.fit(X_regressao, y_regressao)
    
    # Mostrar importância das features
    try:
        importances = pd.Series(
            model_regressao.feature_importances_, 
            index=X_regressao.columns
        ).sort_values(ascending=False)
        
        st.markdown("### 🔍 Features Mais Importantes - ML Regressão")
        st.dataframe(importances.head(10).to_frame("Importância"), width='stretch')
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")
    
    st.success(f"✅ ML Regressão (Estágio 1) treinado com {len(features_existentes)} features")
    return model_regressao, features_existentes

def treinar_ml_principal_estagio2(history, model_regressao, features_regressao):
    """Estágio 2: ML principal com features enriquecidas pela regressão"""
    
    # 1. OBTER PREVISÕES DO ESTÁGIO 1
    if model_regressao is not None:
        X_regressao_treino = history[features_regressao].fillna(0)
        probas_regressao = model_regressao.predict_proba(X_regressao_treino)[:, 1]
        
        # Adicionar previsões como novas features
        history['ML_Regressao_Score'] = probas_regressao
        history['ML_Regressao_Confianca'] = np.abs(probas_regressao - 0.5) * 2
        history['ML_Regressao_Direcao'] = np.where(probas_regressao > 0.5, 1, -1)
    else:
        # Fallback: usar features básicas de regressão
        st.warning("⚠️ Usando fallback - ML regressão não disponível")
        history['ML_Regressao_Score'] = 0.5
        history['ML_Regressao_Confianca'] = 0
        history['ML_Regressao_Direcao'] = 0
    
    # 2. PREPARAR FEATURES DO ML PRINCIPAL
    # Features básicas do sistema atual
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    
    # Features contínuas
    extras_cols = ['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 
                   'Quadrant_Cos', 'Quadrant_Angle']
    if 'Cover_Tendency' in history.columns:
        extras_cols.append('Cover_Tendency')
    extras = history[extras_cols].fillna(0)
    
    # Features estratégicas
    features_estrategicas = []
    prefixos_estrategicos = ['DESEQ_', 'VALOR_', 'CONF_']
    for prefix in prefixos_estrategicos:
        cols_estrategicas = [col for col in history.columns if col.startswith(prefix)]
        features_estrategicas.extend(cols_estrategicas)
    
    if features_estrategicas:
        estrategicas_df = history[features_estrategicas].fillna(0)
    else:
        estrategicas_df = pd.DataFrame()
    
    # 3. NOVAS FEATURES DE REGRESSÃO PARA ML PRINCIPAL
    features_regressao_ml = [
        'ML_Regressao_Score', 'ML_Regressao_Confianca', 'ML_Regressao_Direcao',
        'Z_Excesso_M_Home', 'Z_Excesso_MT_Home', 'Z_Excesso_M_Away', 'Z_Excesso_MT_Away',
        'Fator_Regressao_Home', 'Fator_Regressao_Away'
    ]
    features_regressao_existentes = [f for f in features_regressao_ml if f in history.columns]
    regressao_ml_df = history[features_regressao_existentes].fillna(0)
    
    # 4. COMBINAR TODAS AS FEATURES
    X_components = [quadrantes_home, quadrantes_away, ligas_dummies, extras, regressao_ml_df]
    if not estrategicas_df.empty:
        X_components.append(estrategicas_df)
    
    X_principal = pd.concat(X_components, axis=1)
    
    # 5. TREINAR MODELOS PRINCIPAIS
    y_home = history['Target_AH_Home']
    y_away = 1 - y_home
    
    model_home = RandomForestClassifier(
        n_estimators=500, 
        max_depth=12, 
        random_state=42, 
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model_away = RandomForestClassifier(
        n_estimators=500, 
        max_depth=12, 
        random_state=42, 
        class_weight='balanced_subsample', 
        n_jobs=-1
    )
    
    model_home.fit(X_principal, y_home)
    model_away.fit(X_principal, y_away)
    
    # Mostrar importância das features do estágio 2
    try:
        importances_principal = pd.Series(
            model_home.feature_importances_, 
            index=X_principal.columns
        ).sort_values(ascending=False)
        
        st.markdown("### 🔍 Top Features - ML Principal (Estágio 2)")
        st.dataframe(importances_principal.head(15).to_frame("Importância"), width='stretch')
        
        # Análise específica das features de regressão
        features_regressao_importances = importances_principal[
            importances_principal.index.str.startswith(('ML_Regressao', 'Z_', 'Fator_'))
        ]
        if not features_regressao_importances.empty:
            st.markdown("#### 📊 Importância das Features de Regressão")
            st.dataframe(features_regressao_importances, width='stretch')
            
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias do estágio 2: {e}")
    
    st.success(f"✅ ML Principal (Estágio 2) treinado com {X_principal.shape[1]} features")
    return model_home, model_away, X_principal.columns.tolist()

def prever_com_sistema_duplo(games_today, model_regressao, model_home, model_away, features_principal, features_regressao):
    """Previsão usando ambos os modelos em cascata"""
    
    # 1. CALCULAR FEATURES DE REGRESSÃO
    games_today = calcular_regressao_media_avancada(games_today)
    
    # 2. ESTÁGIO 1: ML DE REGRESSÃO
    if model_regressao is not None:
        features_regressao_existentes = [f for f in features_regressao if f in games_today.columns]
        if features_regressao_existentes:
            X_regressao_hoje = games_today[features_regressao_existentes].fillna(0)
            probas_regressao = model_regressao.predict_proba(X_regressao_hoje)[:, 1]
            
            games_today['ML_Regressao_Score'] = probas_regressao
            games_today['ML_Regressao_Confianca'] = np.abs(probas_regressao - 0.5) * 2
            games_today['ML_Regressao_Direcao'] = np.where(probas_regressao > 0.5, 1, -1)
        else:
            st.warning("⚠️ Features de regressão não encontradas para previsão")
            games_today['ML_Regressao_Score'] = 0.5
            games_today['ML_Regressao_Confianca'] = 0
            games_today['ML_Regressao_Direcao'] = 0
    else:
        games_today['ML_Regressao_Score'] = 0.5
        games_today['ML_Regressao_Confianca'] = 0  
        games_today['ML_Regressao_Direcao'] = 0
    
    # 3. ESTÁGIO 2: PREPARAR FEATURES PARA ML PRINCIPAL
    # Features básicas
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH')
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA')
    ligas_today = pd.get_dummies(games_today['League'], prefix='League')
    
    # Features contínuas
    extras_cols = ['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 
                   'Quadrant_Cos', 'Quadrant_Angle']
    if 'Cover_Tendency' in games_today.columns:
        extras_cols.append('Cover_Tendency')
    extras_today = games_today[extras_cols].fillna(0)
    
    # Features estratégicas
    features_estrategicas_hoje = []
    prefixos_estrategicos = ['DESEQ_', 'VALOR_', 'CONF_']
    for prefix in prefixos_estrategicos:
        cols_estrategicas = [col for col in games_today.columns if col.startswith(prefix)]
        features_estrategicas_hoje.extend(cols_estrategicas)
    
    if features_estrategicas_hoje:
        estrategicas_today = games_today[features_estrategicas_hoje].fillna(0)
    else:
        estrategicas_today = pd.DataFrame()
    
    # Features de regressão
    features_regressao_ml = [
        'ML_Regressao_Score', 'ML_Regressao_Confianca', 'ML_Regressao_Direcao',
        'Z_Excesso_M_Home', 'Z_Excesso_MT_Home', 'Z_Excesso_M_Away', 'Z_Excesso_MT_Away',
        'Fator_Regressao_Home', 'Fator_Regressao_Away'
    ]
    features_regressao_ml_existentes = [f for f in features_regressao_ml if f in games_today.columns]
    regressao_ml_today = games_today[features_regressao_ml_existentes].fillna(0)
    
    # 4. COMBINAR TODAS AS FEATURES
    X_components_today = [qh_today, qa_today, ligas_today, extras_today, regressao_ml_today]
    if not estrategicas_today.empty:
        X_components_today.append(estrategicas_today)
    
    X_hoje = pd.concat(X_components_today, axis=1)
    
    # 5. GARANTIR MESMA ORDEM DAS FEATURES DO TREINO
    X_hoje = X_hoje.reindex(columns=features_principal, fill_value=0)
    
    # 6. PREVISÕES FINAIS
    probas_home = model_home.predict_proba(X_hoje)[:, 1]
    probas_away = model_away.predict_proba(X_hoje)[:, 1]
    
    games_today['Quadrante_ML_Score_Home'] = probas_home
    games_today['Quadrante_ML_Score_Away'] = probas_away  
    games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')
    
    return games_today

# EXECUTAR SISTEMA COMPLETO
if not history.empty:
    st.markdown("### 🔄 Treinando Sistema em 2 Estágios...")
    
    # 1. Calcular regressão para histórico
    history = calcular_regressao_media_avancada(history)
    
    # 2. Estágio 1: ML Regressão
    with st.spinner("Treinando ML de Regressão (Estágio 1)..."):
        model_regressao, features_regressao = treinar_ml_regressao_estagio1(history)
    
    # 3. Estágio 2: ML Principal  
    with st.spinner("Treinando ML Principal (Estágio 2)..."):
        model_home, model_away, features_principal = treinar_ml_principal_estagio2(
            history, model_regressao, features_regressao
        )
    
    # 4. Previsões para hoje
    with st.spinner("Fazendo previsões com sistema duplo..."):
        games_today = prever_com_sistema_duplo(
            games_today, model_regressao, model_home, model_away, 
            features_principal, features_regressao
        )
    
    st.success("🎉 Sistema em 2 Estágios implementado com sucesso!")
    
    # 5. ANÁLISE DE CONTRIBUIÇÃO DOS ESTÁGIOS (ATUALIZADA)
    st.markdown("### 📊 Análise de Contribuição dos Estágios")
    
    if 'ML_Regressao_Score' in games_today.columns and 'Quadrante_ML_Score_Main' in games_today.columns:
        correlacao = games_today[['ML_Regressao_Score', 'Quadrante_ML_Score_Main']].corr().iloc[0,1]
        st.metric("📈 Correlação entre Estágios", f"{correlacao:.3f}")
        
        # 🆕 CRITÉRIOS MAIS FLEXÍVEIS PARA OPORTUNIDADES
        oportunidade_1 = games_today[
            (games_today['ML_Regressao_Score'] > 0.55) & 
            (games_today['Quadrante_ML_Score_Main'] < 0.45)
        ]
        
        oportunidade_2 = games_today[
            (games_today['ML_Regressao_Score'] < 0.45) & 
            (games_today['Quadrante_ML_Score_Main'] > 0.55)
        ]
        
        oportunidade_3 = games_today[
            (abs(games_today['ML_Regressao_Score'] - games_today['Quadrante_ML_Score_Main']) > 0.25)
        ]
        
        total_oportunidades = len(oportunidade_1) + len(oportunidade_2) + len(oportunidade_3)
        
        # 🆕 MÉTRICAS DETALHADAS
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Oportunidades Totais", total_oportunidades)
        with col2:
            st.metric("📊 Diferença Média", 
                     f"{(games_today['ML_Regressao_Score'] - games_today['Quadrante_ML_Score_Main']).mean():.3f}")
        with col3:
            st.metric("🔄 Alinhamento", 
                     "✅ BOM" if correlacao > 0.3 else "⚠️ BAIXO")
        
        # 🆕 DETALHES DAS OPORTUNIDADES
        if total_oportunidades > 0:
            st.info("💡 **Oportunidades identificadas** - Discordância significativa entre os estágios")
            
            todas_oportunidades = pd.concat([oportunidade_1, oportunidade_2, oportunidade_3]).drop_duplicates()
            
            st.dataframe(
                todas_oportunidades[['Home', 'Away', 'ML_Regressao_Score', 'Quadrante_ML_Score_Main', 
                                   'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away']]
                .assign(
                    Diferenca=lambda df: df['ML_Regressao_Score'] - df['Quadrante_ML_Score_Main'],
                    Tipo_Oportunidade=lambda df: np.where(
                        df['ML_Regressao_Score'] > df['Quadrante_ML_Score_Main'], 
                        '📈 REGRESSÃO ALTA', '📉 REGRESSÃO BAIXA'
                    )
                )
                .sort_values('Diferenca', key=abs, ascending=False)
                .style.format({
                    'ML_Regressao_Score': '{:.1%}',
                    'Quadrante_ML_Score_Main': '{:.1%}',
                    'Quadrante_ML_Score_Home': '{:.1%}',
                    'Quadrante_ML_Score_Away': '{:.1%}',
                    'Diferenca': '{:.3f}'
                })
                .background_gradient(subset=['Diferenca'], cmap='RdYlBu'),
                width='stretch'
            )
        else:
            st.success("✅ **Estágios alinhados** - Nenhuma discordância significativa detectada")
            
            # 🆕 MOSTRAR ALGUNS EXEMPLOS MESMO SEM OPORTUNIDADES
            st.info("🔍 **Amostra de alinhamento entre estágios:**")
            amostra = games_today[['Home', 'Away', 'ML_Regressao_Score', 'Quadrante_ML_Score_Main']].head(5)
            st.dataframe(
                amostra.style.format({
                    'ML_Regressao_Score': '{:.1%}',
                    'Quadrante_ML_Score_Main': '{:.1%}'
                }),
                width='stretch'
            )
    




##### BLOCO 11: ANÁLISE ESTRATÉGICA AUTOMÁTICA (Z-Scores + Handicap) #####

st.markdown("## 📊 Análise Estratégica – M & MT + Handicap")

def calcular_analise_estrategica(df):
    """Aplica a análise estratégica baseada em Deltas M/MT e Handicap"""
    df = df.copy()
    
    # Verificar se as colunas necessárias existem
    required_cols = ['M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas ausentes para análise estratégica: {missing_cols}")
        # Inicializar colunas com NaN para evitar erros
        for col in ['Delta_M', 'Delta_MT', 'Strategic_Angle', 'Strategic_Dist', 
                   'Tipo_Desequilibrio', 'Valor_Estrategico', 'Confiança_Estrategica', 'Cover_Tendency']:
            df[col] = np.nan
        return df
    
    # --- 1️⃣ Deltas principais
    df["Delta_M"] = df["M_H"] - df["M_A"]      # força estrutural (liga)
    df["Delta_MT"] = df["MT_H"] - df["MT_A"]   # forma recente (vs padrão próprio)
    
    # Garantir que Asian_Line_Decimal é numérico (se existir)
    if 'Asian_Line_Decimal' in df.columns:
        df["Asian_Line_Decimal"] = pd.to_numeric(df["Asian_Line_Decimal"], errors='coerce')
    else:
        df["Asian_Line_Decimal"] = np.nan
    
    # --- 2️⃣ Vetor: ângulo e distância (ATUALIZADO - compatível com existente)
    # Nota: Mantemos Quadrant_Dist existente, mas adicionamos perspectiva angular
    df["Strategic_Angle"] = np.degrees(np.arctan2(df["Delta_MT"], df["Delta_M"]))
    df["Strategic_Dist"] = np.sqrt(df["Delta_M"]**2 + df["Delta_MT"]**2)
    
    # --- 3️⃣ Tipo de desequilíbrio
    def classify_desequilibrio(angle, dist):
        if pd.isna(angle) or pd.isna(dist):
            return "Indefinido"
        angle_abs = abs(angle)
        if angle_abs > 60:
            return "Forma-Recente"
        elif angle_abs < 30:
            return "Força-Liga"
        elif 30 <= angle_abs <= 60 and dist > 0.5:
            return "Consistente"
        else:
            return "Equilibrado"
    
    df["Tipo_Desequilibrio"] = df.apply(
        lambda x: classify_desequilibrio(x["Strategic_Angle"], x["Strategic_Dist"]), axis=1
    )
    
    # --- 4️⃣ Lado provável (quem tende a cobrir o handicap)
    def predict_side(row):
        line = row["Asian_Line_Decimal"]
        dM = row["Delta_M"]
        dMT = row["Delta_MT"]
        
        if pd.isna(line) or pd.isna(dM) or pd.isna(dMT):
            return "Analisar"
            
        # Favorito é o Home (linha negativa)
        if line < 0:
            # favorito em má forma → value no away
            if dMT < -0.2:
                return "AWAY"
            # favorito forte e em boa forma → tende a cobrir
            elif dM > 0.3 and dMT > 0.2:
                return "HOME"
            else:
                return "EQUILIBRADO"

        # Favorito é o Away (linha positiva)
        elif line > 0:
            if dMT > 0.2:
                return "HOME"
            elif dMT < -0.2:
                return "AWAY"
            else:
                return "EQUILIBRADO"

        # Linha zero → neutro
        return "EQUILIBRADO"
    
    df["Valor_Estrategico"] = df.apply(predict_side, axis=1)
    
    # --- 5️⃣ Grau de confiança
    def classify_confidence(row):
        dist = row["Strategic_Dist"]
        if pd.isna(dist):
            return "Baixa"
        if dist >= 1.2:
            return "Alta"
        elif dist >= 0.6:
            return "Moderada"
        else:
            return "Baixa"
    
    df["Confiança_Estrategica"] = df.apply(classify_confidence, axis=1)
    
    # --- 6️⃣ Tendência contínua de cobertura (para ML supervisionada)
    def calculate_cover_tendency(row):
        line = row["Asian_Line_Decimal"]
        dM = row["Delta_M"]
        dMT = row["Delta_MT"]
        
        if pd.isna(line) or pd.isna(dM) or pd.isna(dMT):
            return np.nan
            
        # Sinal baseado na linha: positivo favorece Home, negativo favorece Away
        line_sign = -1 if line < 0 else 1
        
        return (dM * line_sign) + (dMT * line_sign)
    
    df["Cover_Tendency"] = df.apply(calculate_cover_tendency, axis=1)
    
    return df

# Aplicar análise estratégica aos dados
if not games_today.empty:
    games_today = calcular_analise_estrategica(games_today)
    st.success("✅ Análise estratégica aplicada com sucesso!")

# Exibir resultados da análise estratégica
if "Valor_Estrategico" in games_today.columns:
    st.markdown("### 🎯 Resultados da Análise Estratégica")
    
    cols_show = [
        "League", "Time", "Home", "Away", "Goals_H_Today", "Goals_A_Today", "Asian_Line_Decimal", 
        "Delta_M", "Delta_MT", "Tipo_Desequilibrio", 
        "Valor_Estrategico", "Confiança_Estrategica", "Cover_Tendency"
    ]
    
    # Filtrar colunas existentes
    cols_existentes = [c for c in cols_show if c in games_today.columns]
    
    # VERSÃO MAIS SIMPLES E DIRETA:
    st.dataframe(
        games_today[cols_existentes]
        .assign(
            Valor_Estrategico=lambda df: df['Valor_Estrategico'].map({
                'HOME': '🔵 HOME', 
                'AWAY': '🔴 AWAY', 
                'EQUILIBRADO': '⚪ EQUILIBRADO',
                'Analisar': '❓ ANALISAR'
            }),
            Cover_Tendency=lambda df: df['Cover_Tendency'].apply(
                lambda x: f"🔼 {x:.2f}" if x > 0.5 else 
                         f"↗️ {x:.2f}" if x > 0 else 
                         f"➖ {x:.2f}" if x == 0 else 
                         f"↘️ {x:.2f}" if x > -0.5 else 
                         f"🔽 {x:.2f}"
            ),
            Confiança_Estrategica=lambda df: df['Confiança_Estrategica'].map({
                'Alta': '🎯 Alta', 
                'Moderada': '✅ Moderada', 
                'Baixa': '⚠️ Baixa'
            })
        ),
        width='stretch'
    )

# --- INTEGRAÇÃO COM O MODELO EXISTENTE ---
def enriquecer_features_ml(df):
    """Adiciona as novas features estratégicas para o modelo ML"""
    if "Tipo_Desequilibrio" in df.columns:
        # One-hot encoding para tipo de desequilíbrio
        desequilibrio_dummies = pd.get_dummies(df['Tipo_Desequilibrio'], prefix='DESEQ')
        
        # One-hot encoding para valor estratégico
        valor_dummies = pd.get_dummies(df['Valor_Estrategico'], prefix='VALOR')
        
        # One-hot encoding para confiança
        confianca_dummies = pd.get_dummies(df['Confiança_Estrategica'], prefix='CONF')
        
        # Combinar com dataframe existente
        df = pd.concat([df, desequilibrio_dummies, valor_dummies, confianca_dummies], axis=1)
    
    return df

# Aplicar enriquecimento aos dados de treino e hoje
if not history.empty:
    history = enriquecer_features_ml(history)
    
if not games_today.empty:
    games_today = enriquecer_features_ml(games_today)

st.info("""
✅ **Lógica da análise estratégica integrada:**
- `Delta_M` → força relativa na liga  
- `Delta_MT` → forma atual comparada ao padrão próprio  
- `Tipo_Desequilibrio` → Forma-Recente, Força-Liga, Consistente ou Equilibrado  
- `Valor_Estrategico` → lado com maior probabilidade de cobrir o handicap  
- `Confiança_Estrategica` → baseada na distância do desequilíbrio
- `Cover_Tendency` → métrica contínua para ML (positivo = Home, negativo = Away)
""")





##### BLOCO 10: SISTEMA DE INDICAÇÕES E RECOMENDAÇÕES #####

def adicionar_indicadores_explicativos_16_dual(df):
    """Adiciona classificações e recomendações explícitas para 16 quadrantes"""
    df = df.copy()
    
    # Mapear quadrantes para labels
    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro'))
    
    # 1. CLASSIFICAÇÃO DE VALOR PARA HOME
    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')
    
    # 2. CLASSIFICAÇÃO DE VALOR PARA AWAY
    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')
    
    # 3. RECOMENDAÇÃO DE APOSTA DUAL PARA 16 QUADRANTES
    def gerar_recomendacao_16_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']
        
        # Padrões específicos para 16 quadrantes
        if 'Fav Forte' in home_q and 'Under Forte' in away_q:
            return f'💪 FAVORITO HOME FORTE ({score_home:.1%})'
        elif 'Under Forte' in home_q and 'Fav Forte' in away_q:
            return f'💪 FAVORITO AWAY FORTE ({score_away:.1%})'
        elif 'Fav Moderado' in home_q and 'Under Moderado' in away_q and 'Forte' in away_q:
            return f'🎯 VALUE NO HOME ({score_home:.1%})'
        elif 'Under Moderado' in home_q and 'Fav Moderado' in away_q and 'Forte' in home_q:
            return f'🎯 VALUE NO AWAY ({score_away:.1%})'
        elif ml_side == 'HOME' and score_home >= 0.60:
            return f'📈 MODELO CONFIA HOME ({score_home:.1%})'
        elif ml_side == 'AWAY' and score_away >= 0.60:
            return f'📈 MODELO CONFIA AWAY ({score_away:.1%})'
        elif 'Neutro' in home_q and score_away >= 0.58:
            return f'🔄 AWAY EM NEUTRO ({score_away:.1%})'
        elif 'Neutro' in away_q and score_home >= 0.58:
            return f'🔄 HOME EM NEUTRO ({score_home:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'
    
    df['Recomendacao'] = df.apply(gerar_recomendacao_16_dual, axis=1)
    
    # 4. RANKING POR MELHOR PROBABILIDADE
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)
    
    return df

def gerar_estrategias_16_quadrantes(df):
    """Gera estratégias específicas baseadas nos 16 quadrantes"""
    st.markdown("### 🎯 Estratégias por Categoria - 16 Quadrantes")
    
    estrategias = {
        'Fav Forte': {
            'descricao': '**Favoritos Fortes** - Times com alta aggression e handscore',
            'quadrantes': [1, 2, 3, 4],
            'estrategia': 'Apostar como favoritos, especialmente contra underdogs fracos',
            'confianca': 'Alta'
        },
        'Fav Moderado': {
            'descricao': '**Favoritos Moderados** - Times com aggression positiva moderada', 
            'quadrantes': [5, 6, 7, 8],
            'estrategia': 'Buscar value, especialmente quando têm handscore forte',
            'confianca': 'Média-Alta'
        },
        'Under Moderado': {
            'descricao': '**Underdogs Moderados** - Times com aggression negativa moderada',
            'quadrantes': [9, 10, 11, 12],
            'estrategia': 'Apostar contra quando enfrentam favoritos supervalorizados',
            'confianca': 'Média'
        },
        'Under Forte': {
            'descricao': '**Underdogs Fortes** - Times com aggression muito negativa',
            'quadrantes': [13, 14, 15, 16], 
            'estrategia': 'Evitar ou apostar contra, exceto em situações muito específicas',
            'confianca': 'Baixa'
        }
    }
    
    for categoria, info in estrategias.items():
        st.subheader(f"**{categoria}**")
        st.write(f"📋 {info['descricao']}")
        st.write(f"🎯 Estratégia: {info['estrategia']}")
        st.write(f"📊 Confiança: {info['confianca']}")
        
        # Mostrar quadrantes específicos
        quadrantes_str = ", ".join([f"Q{q}" for q in info['quadrantes']])
        st.write(f"🔢 Quadrantes: {quadrantes_str}")
        
        # Estatísticas da categoria
        jogos_categoria = df[
            df['Quadrante_Home'].isin(info['quadrantes']) | 
            df['Quadrante_Away'].isin(info['quadrantes'])
        ]
        
        if not jogos_categoria.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jogos Encontrados", len(jogos_categoria))
            with col2:
                avg_score = jogos_categoria['Quadrante_ML_Score_Main'].mean()
                st.metric("Score Médio", f"{avg_score:.1%}")
            with col3:
                high_value = len(jogos_categoria[jogos_categoria['Quadrante_ML_Score_Main'] >= 0.60])
                st.metric("Alto Valor", high_value)

            # 🔘 Botão para expandir / ocultar tabela
            with st.expander(f"🔍 Ver confrontos da categoria {categoria}"):
                cols_padrao = [
                    'League', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
                    'Quadrante_Home_Label', 'Quadrante_Away_Label',
                    'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away',
                    'Quadrante_ML_Score_Main', 'Recomendacao', 
                    'Quadrant_Dist', 'Quadrant_Angle'
                ]
                cols_padrao = [c for c in cols_padrao if c in jogos_categoria.columns]
                
                st.dataframe(
                    jogos_categoria[cols_padrao]
                    .sort_values('Quadrante_ML_Score_Main', ascending=False)
                    .style.format({
                        'Goals_H_Today': '{:.0f}',
                        'Goals_A_Today': '{:.0f}',
                        'Quadrante_ML_Score_Home': '{:.1%}',
                        'Quadrante_ML_Score_Away': '{:.1%}',
                        'Quadrante_ML_Score_Main': '{:.1%}',
                        'Quadrant_Dist': '{:.2f}',
                        'Quadrant_Angle': '{:.1f}°'
                    })
                    .background_gradient(subset=['Quadrante_ML_Score_Main'], cmap='RdYlGn'),
                    width='stretch'
                )

        else:
            st.info("Nenhum jogo encontrado nesta categoria.")
        
        st.write("---")

def analisar_padroes_quadrantes_16_dual(df):
    """Analisa padrões recorrentes nas combinações de 16 quadrantes"""
    st.markdown("### 🔍 Análise de Padrões por Combinação (16 Quadrantes)")
    
    # Padrões prioritários (mais gerais, sem necessidade de correspondência exata)
    padroes_16 = {
        'Fav Forte vs Under Forte': {
            'descricao': '🎯 **PADRÃO HOME FORTE** - Favorito forte contra underdog forte (fraco)',
            'lado_recomendado': 'HOME',
            'prioridade': 1,
            'score_min': 0.60
        },
        'Under Forte vs Fav Forte': {
            'descricao': '🎯 **PADRÃO AWAY FORTE** - Underdog enfrentando favorito forte',
            'lado_recomendado': 'AWAY', 
            'prioridade': 1,
            'score_min': 0.60
        },
        'Fav Moderado vs Under Moderado': {
            'descricao': '💪 **VALUE HOME** - Favorito moderado contra underdog moderado',
            'lado_recomendado': 'HOME',
            'prioridade': 2,
            'score_min': 0.55
        },
        'Under Moderado vs Fav Moderado': {
            'descricao': '💪 **VALUE AWAY** - Underdog moderado contra favorito moderado',
            'lado_recomendado': 'AWAY',
            'prioridade': 2, 
            'score_min': 0.55
        },
        'Fav Forte vs Under Moderado': {
            'descricao': '📊 **DOMÍNIO HOME** - Favorito forte contra underdog moderado',
            'lado_recomendado': 'HOME',
            'prioridade': 3,
            'score_min': 0.55
        },
        'Under Forte vs Fav Moderado': {
            'descricao': '📊 **REAÇÃO AWAY** - Underdog forte contra favorito moderado',
            'lado_recomendado': 'AWAY',
            'prioridade': 3,
            'score_min': 0.55
        }
    }
    
    # Ordenar padrões por prioridade
    padroes_ordenados = sorted(padroes_16.items(), key=lambda x: x[1]['prioridade'])
    
    for padrao, info in padroes_ordenados:
        home_q, away_q = padrao.split(' vs ')
        
        # 🔍 Busca "contém" — mais flexível que igualdade
        jogos = df[
            df['Quadrante_Home_Label'].str.contains(home_q, case=False, na=False) &
            df['Quadrante_Away_Label'].str.contains(away_q, case=False, na=False)
        ]
        
        # Filtrar por score mínimo
        score_col = 'Quadrante_ML_Score_Home' if info['lado_recomendado'] == 'HOME' else 'Quadrante_ML_Score_Away'
        jogos = jogos[jogos[score_col] >= info.get('score_min', 0.5)]
        
        # Mostrar resultados se houver
        if not jogos.empty:
            st.subheader(f"**{padrao}**")
            st.write(info['descricao'])
            st.write(f"📈 **Score mínimo:** {info['score_min']:.0%}")
            st.write(f"🎯 **Jogos encontrados:** {len(jogos)}")
            
            cols_padrao = [
                'League', 'Home', 'Away','Goals_H_Today', 'Goals_A_Today',
                'Quadrante_Home_Label', 'Quadrante_Away_Label',
                score_col, 'Recomendacao', 'Quadrant_Dist', 'Quadrant_Angle'
            ]
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]
            
            st.dataframe(
                jogos.sort_values(score_col, ascending=False)[cols_padrao]
                .style.format({
                    'Goals_H_Today': '{:.0f}',
                    'Goals_A_Today': '{:.0f}',
                    score_col: '{:.1%}',
                    'Quadrant_Dist': '{:.2f}',
                    'Quadrant_Angle': '{:.1f}°'
                })
                .background_gradient(subset=[score_col], cmap='RdYlGn'),
                width='stretch'
            )
            st.write("---")







##### BLOCO 12: SISTEMA DE SCORING COMBINADO #####

def calcular_pontuacao_quadrante_16(quadrante_id):
    """Calcula pontuação base para cada quadrante (0-100)"""
    scores_base = {
        # Fav Forte: alta pontuação
        1: 85, 2: 80, 3: 75, 4: 70,
        # Fav Moderado: média-alta
        5: 70, 6: 65, 7: 60, 8: 55,
        # Under Moderado: média-baixa  
        9: 50, 10: 45, 11: 40, 12: 35,
        # Under Forte: baixa pontuação
        13: 35, 14: 30, 15: 25, 16: 20
    }
    return scores_base.get(quadrante_id, 50)

def gerar_score_combinado_16(df):
    """Gera score combinado considerando ambos os quadrantes"""
    df = df.copy()
    
    # Score base dos quadrantes
    df['Score_Base_Home'] = df['Quadrante_Home'].apply(calcular_pontuacao_quadrante_16)
    df['Score_Base_Away'] = df['Quadrante_Away'].apply(calcular_pontuacao_quadrante_16)
    
    # Score combinado (média ponderada)
    df['Score_Combinado'] = (df['Score_Base_Home'] * 0.6 + df['Score_Base_Away'] * 0.4)
    
    # Ajustar pelo ML Score
    df['Score_Final'] = df['Score_Combinado'] * df['Quadrante_ML_Score_Main']
    
    # Classificar por potencial
    conditions = [
        df['Score_Final'] >= 60,
        df['Score_Final'] >= 45, 
        df['Score_Final'] >= 30,
        df['Score_Final'] < 30
    ]
    choices = ['🌟 ALTO POTENCIAL', '💼 VALOR SOLIDO', '⚖️ NEUTRO', '🔴 BAIXO POTENCIAL']
    df['Classificacao_Potencial'] = np.select(conditions, choices, default='⚖️ NEUTRO')
    
    return df





##### BLOCO 15: VERIFICAÇÃO DE DADOS COMPLETOS E SINALIZAÇÃO #####

def verificar_dados_completos(df):
    """
    Verifica se temos dados completos para análise e classifica cada confronto
    """
    df = df.copy()
    
    # 🔍 COLUNAS CRÍTICAS PARA ANÁLISE COMPLETA
    colunas_criticas_ml = [
        'M_H', 'M_A', 'MT_H', 'MT_A',
        'Aggression_Home', 'Aggression_Away', 
        'HandScore_Home', 'HandScore_Away',
        'Asian_Line_Decimal'
    ]
    
    colunas_criticas_quadrantes = [
        'Quadrante_Home', 'Quadrante_Away',
        'Quadrant_Dist', 'Quadrant_Angle'
    ]
    
    colunas_criticas_regressao = [
        'Z_Excesso_M_Home', 'Z_Excesso_MT_Home',
        'Fator_Regressao_Home', 'ML_Regressao_Score'
    ]
    
    # 🎯 VERIFICAÇÃO POR NÍVEL DE COMPLETUDE
    def classificar_completude(row):
        missing_criticos = sum(1 for col in colunas_criticas_ml if pd.isna(row.get(col, np.nan)))
        missing_quadrantes = sum(1 for col in colunas_criticas_quadrantes if pd.isna(row.get(col, np.nan)))
        missing_regressao = sum(1 for col in colunas_criticas_regressao if pd.isna(row.get(col, np.nan)))
        
        total_criticos = len(colunas_criticas_ml)
        total_quadrantes = len(colunas_criticas_quadrantes)
        
        # 📊 CALCULAR PERCENTUAL DE DADOS PRESENTES
        percentual_ml = ((total_criticos - missing_criticos) / total_criticos) * 100
        percentual_quadrantes = ((total_quadrantes - missing_quadrantes) / total_quadrantes) * 100
        tem_regressao = missing_regressao <= 2  # Permite até 2 faltantes
        
        # 🎚️ CLASSIFICAÇÃO
        if percentual_ml >= 90 and percentual_quadrantes >= 80 and tem_regressao:
            return "✅ DADOS COMPLETOS", "high"
        elif percentual_ml >= 70 and percentual_quadrantes >= 60:
            return "⚠️ DADOS PARCIAIS", "medium" 
        elif percentual_ml >= 50:
            return "🚨 DADOS INSUFICIENTES", "low"
        else:
            return "❌ DADOS CRÍTICOS", "critical"
    
    # 📝 APLICAR CLASSIFICAÇÃO
    resultados = df.apply(classificar_completude, axis=1)
    df[['Status_Dados', 'Nivel_Confianca']] = pd.DataFrame(resultados.tolist(), index=df.index)
    
    # 🔢 CONTAGEM DE DADOS FALTANTES POR CATEGORIA
    df['Missing_ML'] = df[colunas_criticas_ml].isna().sum(axis=1)
    df['Missing_Quadrantes'] = df[colunas_criticas_quadrantes].isna().sum(axis=1)
    df['Missing_Regressao'] = df[colunas_criticas_regressao].isna().sum(axis=1)
    df['Total_Missing'] = df['Missing_ML'] + df['Missing_Quadrantes'] + df['Missing_Regressao']
    
    return df

def aplicar_filtros_qualidade(df, nivel_minimo="medium"):
    """
    Filtra confrontos baseado na qualidade dos dados
    """
    niveis = {"critical": 0, "low": 1, "medium": 2, "high": 3}
    nivel_min = niveis.get(nivel_minimo, 2)
    
    df_filtrado = df.copy()
    df_filtrado['Nivel_Num'] = df_filtrado['Nivel_Confianca'].map(niveis)
    
    # Filtrar por nível mínimo
    mascarafiltro = df_filtrado['Nivel_Num'] >= nivel_min
    
    st.info(f"📊 Filtro aplicado: {nivel_minimo} | {mascarafiltro.sum()}/{len(df)} confrontos válidos")
    
    return df_filtrado[mascarafiltro]

def criar_visualizacao_status_dados(df):
    """
    Cria visualização do status dos dados
    """
    st.markdown("### 📋 Status de Dados por Confronto")
    
    # Estatísticas gerais
    status_counts = df['Status_Dados'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Completos", status_counts.get("✅ DADOS COMPLETOS", 0))
    with col2:
        st.metric("⚠️ Parciais", status_counts.get("⚠️ DADOS PARCIAIS", 0))
    with col3:
        st.metric("🚨 Insuficientes", status_counts.get("🚨 DADOS INSUFICIENTES", 0))
    with col4:
        st.metric("❌ Críticos", status_counts.get("❌ DADOS CRÍTICOS", 0))
    
    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 4))
    cores = {'✅ DADOS COMPLETOS': 'green', '⚠️ DADOS PARCIAIS': 'orange', 
             '🚨 DADOS INSUFICIENTES': 'red', '❌ DADOS CRÍTICOS': 'darkred'}
    
    bars = ax.bar(status_counts.index, status_counts.values, 
                  color=[cores.get(x, 'gray') for x in status_counts.index])
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    ax.set_ylabel('Número de Confrontos')
    ax.set_title('Distribuição da Qualidade dos Dados')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Tabela detalhada
    with st.expander("🔍 Ver Detalhes por Confronto"):
        cols_detalhes = [
            'League', 'Home', 'Away', 'Status_Dados', 'Nivel_Confianca',
            'Missing_ML', 'Missing_Quadrantes', 'Missing_Regressao', 'Total_Missing'
        ]
        cols_existentes = [c for c in cols_detalhes if c in df.columns]
        
        st.dataframe(
            df[cols_existentes].sort_values(['Nivel_Confianca', 'Total_Missing']),
            width='stretch'
        )

# 🔄 ATUALIZAR A FUNÇÃO DE ESTILO PARA INCLUIR STATUS
def estilo_tabela_com_status(df):
    """
    Estilo atualizado para incluir status de dados
    """
    def cor_status(val):
        if '✅' in str(val): return 'font-weight: bold'
        elif '⚠️' in str(val): return 'font-weight: bold'
        elif '🚨' in str(val): return 'font-weight: bold'
        elif '❌' in str(val): return 'font-weight: bold'
        return ''
    
    styler = df.style
    
    # Aplicar cores ao status
    if 'Status_Dados' in df.columns:
        styler = styler.applymap(cor_status, subset=['Status_Dados'])
    
    # Manter gradientes existentes
    score_cols = [col for col in ['Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 'Score_Final'] 
                  if col in df.columns]
    if score_cols:
        styler = styler.background_gradient(subset=score_cols, cmap='RdYlGn')
    
    return styler




##### BLOCO 13: EXIBIÇÃO DOS RESULTADOS E LIVE MONITOR (ATUALIZADO) #####

st.markdown("## 🏆 Melhores Confrontos por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()

    # 🆕 APLICAR VERIFICAÇÃO DE DADOS COMPLETOS
    ranking_quadrantes = verificar_dados_completos(ranking_quadrantes)
    
    # 🆕 CRIAR VISUALIZAÇÃO DO STATUS
    criar_visualizacao_status_dados(ranking_quadrantes)
    
    # 🆕 FILTRO INTERATIVO DE QUALIDADE
    st.markdown("### 🎛️ Filtro de Qualidade de Dados")
    nivel_filtro = st.selectbox(
        "Nível mínimo de confiança nos dados:",
        ["✅ DADOS COMPLETOS", "⚠️ DADOS PARCIAIS", "🚨 DADOS INSUFICIENTES", "❌ TODOS"],
        index=1
    )
    
    # Converter para chave do filtro
    nivel_map = {
        "✅ DADOS COMPLETOS": "high",
        "⚠️ DADOS PARCIAIS": "medium", 
        "🚨 DADOS INSUFICIENTES": "low",
        "❌ TODOS": "critical"
    }
    
    ranking_filtrado = aplicar_filtros_qualidade(ranking_quadrantes, nivel_map[nivel_filtro])

    # Aplicar indicadores explicativos para 16 quadrantes
    ranking_filtrado = adicionar_indicadores_explicativos_16_dual(ranking_filtrado)

    # Aplicar scoring combinado
    ranking_filtrado = gerar_score_combinado_16(ranking_filtrado)

    # Aplicar atualização em tempo real COM V9
    ranking_filtrado = apply_handicap_results_v9(ranking_filtrado)

    # Exibir resumo live ATUALIZADO
    st.markdown("## 📡 Live Score Monitor - 16 Quadrantes (v9 Validado)")
    live_summary = generate_live_summary_v9(ranking_filtrado)
    st.json(live_summary)

    # Ordenar por score final
    ranking_filtrado = ranking_filtrado.sort_values('Score_Final', ascending=False)

    # 🆕 ATUALIZAR COLUNAS PARA INCLUIR STATUS
    colunas_possiveis = [
        'Status_Dados',  # 🆕 NOVA COLUNA
        'League', 'Time', 'Home', 'Away', 
        'Goals_H_Today', 'Goals_A_Today', 'Recomendacao',
        'ML_Side', 'Side_Bet',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Score_Final', 'Classificacao_Potencial',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away',
        # Colunas Live Score V9
        'Asian_Line_Decimal', 'Handicap_Result_Final', 'Outcome_Final',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Final'
    ]

    # Filtrar colunas existentes
    cols_finais = [c for c in colunas_possiveis if c in ranking_filtrado.columns]

    # 🆕 USAR ESTILO ATUALIZADO COM STATUS
    st.dataframe(
        estilo_tabela_com_status(ranking_filtrado[cols_finais])
        .format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Home_Red': '{:.0f}',
            'Away_Red': '{:.0f}',
            'Profit_Final': '{:.2f}',
            'Outcome_Final': '{:.1f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Score_Final': '{:.1f}'
        }, na_rep="-"),
        width='stretch'
    )

    # 🆕 AVISO PARA CONFRONTOS FILTRADOS
    if len(ranking_filtrado) < len(ranking_quadrantes):
        st.warning(f"⚠️ {len(ranking_quadrantes) - len(ranking_filtrado)} confrontos foram ocultados devido à baixa qualidade dos dados")

    # ---------------- ANÁLISES ESPECÍFICAS ----------------
    analisar_padroes_quadrantes_16_dual(ranking_filtrado)
    gerar_estrategias_16_quadrantes(ranking_filtrado)

else:
    st.info("⚠️ Aguardando dados para gerar ranking de 16 quadrantes")

###############################


##########################

st.success("🎯 **Sistema de 16 Quadrantes ML** implementado com sucesso!")
st.info("""
**Resumo das melhorias:**
- 🔢 16 quadrantes para granularidade máxima
- 🎯 Estratégias específicas por categoria  
- 📊 Scoring combinado inteligente
- 🔍 Análise de padrões avançada
- 📈 Visualizações otimizadas
- ✅ Sistema V9 de handicap asiático validado
""")
