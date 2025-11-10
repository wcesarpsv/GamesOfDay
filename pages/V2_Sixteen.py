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
    y=[-4, 4],
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

st.plotly_chart(fig, use_container_width=True)

##### BLOCO 9: MODELO ML PARA 16 QUADRANTES #####

def treinar_modelo_quadrantes_16_dual(history, games_today):
    """
    Treina modelo ML para Home e Away com base nos 16 quadrantes
    """
    # Garantir cálculo das distâncias
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    # Preparar features básicas
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    # Features contínuas
    extras = history[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos','Quadrant_Angle']].fillna(0)

    # Combinar todas as features
    X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies, extras], axis=1)

    # Targets
    y_home = history['Target_AH_Home']
    y_away = 1 - y_home  # inverso lógico

    # Treinar modelos
    model_home = RandomForestClassifier(
        n_estimators=500, max_depth=12, random_state=42, class_weight='balanced_subsample', n_jobs=-1
    )
    model_away = RandomForestClassifier(
        n_estimators=500, max_depth=12, random_state=42, class_weight='balanced_subsample', n_jobs=-1
    )

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # Preparar dados para hoje
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    extras_today = games_today[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos','Quadrant_Angle']].fillna(0)

    X_today = pd.concat([qh_today, qa_today, ligas_today, extras_today], axis=1)

    # Fazer previsões
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]

    games_today['Quadrante_ML_Score_Home'] = probas_home
    games_today['Quadrante_ML_Score_Away'] = probas_away
    games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

    # Mostrar importância das features
    try:
        importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_feats = importances.head(15)
        st.markdown("### 🔍 Top Features mais importantes (Modelo HOME - 16 Quadrantes)")
        st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")

    st.success("✅ Modelo dual (Home/Away) com 16 quadrantes treinado com sucesso!")
    return model_home, model_away, games_today

# Executar treinamento
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_quadrantes_16_dual(history, games_today)
    st.success("✅ Modelo dual com 16 quadrantes treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

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
                    use_container_width=True
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
                use_container_width=True
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

##### BLOCO 13: EXIBIÇÃO DOS RESULTADOS E LIVE MONITOR #####

st.markdown("## 🏆 Melhores Confrontos por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()
    
    # Aplicar indicadores explicativos para 16 quadrantes
    ranking_quadrantes = adicionar_indicadores_explicativos_16_dual(ranking_quadrantes)
    
    # Aplicar scoring combinado
    ranking_quadrantes = gerar_score_combinado_16(ranking_quadrantes)
    
    # Aplicar atualização em tempo real COM V9
    ranking_quadrantes = apply_handicap_results_v9(ranking_quadrantes)
    
    # Exibir resumo live ATUALIZADO
    st.markdown("## 📡 Live Score Monitor - 16 Quadrantes (v9 Validado)")
    live_summary = generate_live_summary_v9(ranking_quadrantes)
    st.json(live_summary)
    
    # Ordenar por score final
    ranking_quadrantes = ranking_quadrantes.sort_values('Score_Final', ascending=False)
    
    # Colunas para exibir
    colunas_possiveis = [
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
    cols_finais = [c for c in colunas_possiveis if c in ranking_quadrantes.columns]
    
    # Função de estilo atualizada
    def estilo_tabela_16_quadrantes(df):
        def cor_classificacao(valor):
            if '🌟 ALTO POTENCIAL' in str(valor): return 'font-weight: bold'
            elif '💼 VALOR SOLIDO' in str(valor): return 'font-weight: bold'
            elif '🔴 BAIXO POTENCIAL' in str(valor): return 'font-weight: bold'
            elif '🏆 ALTO VALOR' in str(valor): return 'font-weight: bold'
            elif '🔴 ALTO RISCO' in str(valor): return 'font-weight: bold'
            elif 'VALUE' in str(valor): return 'background-color: #98FB98'
            elif 'EVITAR' in str(valor): return 'background-color: #FFCCCB'
            else: return ''
        
        colunas_para_estilo = []
        for col in ['Classificacao_Potencial', 'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao']:
            if col in df.columns:
                colunas_para_estilo.append(col)
        
        styler = df.style
        if colunas_para_estilo:
            styler = styler.applymap(cor_classificacao, subset=colunas_para_estilo)
        
        # Aplicar gradientes
        if 'Quadrante_ML_Score_Home' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')
        if 'Quadrante_ML_Score_Away' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')
        if 'Score_Final' in df.columns:
            styler = styler.background_gradient(subset=['Score_Final'], cmap='RdYlGn')
        
        return styler

    st.dataframe(
        estilo_tabela_16_quadrantes(ranking_quadrantes[cols_finais])
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
        use_container_width=True
    )
    
    # ---------------- ANÁLISES ESPECÍFICAS ----------------
    analisar_padroes_quadrantes_16_dual(ranking_quadrantes)
    gerar_estrategias_16_quadrantes(ranking_quadrantes)
    
else:
    st.info("⚠️ Aguardando dados para gerar ranking de 16 quadrantes")

##### BLOCO 14: RESUMO EXECUTIVO #####

def resumo_16_quadrantes_hoje(df):
    """Resumo executivo dos 16 quadrantes de hoje"""
    
    st.markdown("### 📋 Resumo Executivo - 16 Quadrantes Hoje")
    
    if df.empty:
        st.info("Nenhum dado disponível para resumo")
        return
    
    total_jogos = len(df)
    
    # Estatísticas de classificação
    alto_potencial = len(df[df['Classificacao_Potencial'] == '🌟 ALTO POTENCIAL'])
    valor_solido = len(df[df['Classificacao_Potencial'] == '💼 VALOR SOLIDO'])
    
    alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
    alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])
    
    home_recomendado = len(df[df['ML_Side'] == 'HOME'])
    away_recomendado = len(df[df['ML_Side'] == 'AWAY'])
    
    # Distribuição por categoria de quadrante
    fav_forte = len(df[df['Quadrante_Home'].isin([1,2,3,4]) | df['Quadrante_Away'].isin([1,2,3,4])])
    fav_moderado = len(df[df['Quadrante_Home'].isin([5,6,7,8]) | df['Quadrante_Away'].isin([5,6,7,8])])
    under_moderado = len(df[df['Quadrante_Home'].isin([9,10,11,12]) | df['Quadrante_Away'].isin([9,10,11,12])])
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
        st.metric("⚔️ Fav Forte", fav_forte)
        st.metric("⚔️ Under Forte", under_forte)
    
    # Distribuição de recomendações
    st.markdown("#### 📊 Distribuição de Recomendações")
    if 'Recomendacao' in df.columns:
        dist_recomendacoes = df['Recomendacao'].value_counts()
        st.dataframe(dist_recomendacoes, use_container_width=True)

if not games_today.empty and 'Classificacao_Potencial' in games_today.columns:
    resumo_16_quadrantes_hoje(games_today)

st.markdown("---")

##### BLOCO EXTRA: ANÁLISE DETALHADA DAS DISTÂNCIAS #####

st.markdown("## 📐 Análise Detalhada das Distâncias Euclidianas")

if 'Quadrant_Dist' in games_today.columns:
    distancias = games_today['Quadrant_Dist'].dropna()
    
    if not distancias.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Média", f"{distancias.mean():.2f}")
        with col2:
            st.metric("Mediana", f"{distancias.median():.2f}")
        with col3:
            st.metric("Máxima", f"{distancias.max():.2f}")
        with col4:
            st.metric("Mínima", f"{distancias.min():.2f}")
        
        # Análise de distribuição
        st.markdown("### 📈 Distribuição das Distâncias")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histograma
        ax1.hist(distancias, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(distancias.mean(), color='red', linestyle='--', label=f'Média: {distancias.mean():.2f}')
        ax1.axvline(distancias.median(), color='green', linestyle='--', label=f'Mediana: {distancias.median():.2f}')
        ax1.set_xlabel('Distância Euclidiana')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Distribuição das Distâncias entre Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(distancias, vert=False)
        ax2.set_xlabel('Distância Euclidiana')
        ax2.set_title('Box Plot - Dispersão das Distâncias')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Classificação por percentis
        st.markdown("### 🎯 Classificação por Níveis de Distância")
        
        p25 = distancias.quantile(0.25)
        p50 = distancias.quantile(0.50)
        p75 = distancias.quantile(0.75)
        
        st.write(f"**Percentis:**")
        st.write(f"- 25% (Baixa): ≤ {p25:.2f}")
        st.write(f"- 50% (Média): ≤ {p50:.2f}") 
        st.write(f"- 75% (Alta): ≤ {p75:.2f}")
        st.write(f"- Máxima: {distancias.max():.2f}")
        
        # Exemplos práticos
        st.markdown("### 🔍 Exemplos Práticos do Dataset")
        
        # Encontrar exemplos reais
        exemplos = {
            "Muito Baixa": distancias[distancias <= p25].head(3),
            "Média": distancias[(distancias > p25) & (distancias <= p75)].head(3),
            "Alta": distancias[distancias > p75].head(3)
        }
        
        for categoria, valores in exemplos.items():
            if not valores.empty:
                st.write(f"**{categoria} Distância** ({valores.min():.2f} - {valores.max():.2f}):")
                
                for dist_val in valores.index:
                    if dist_val in games_today.index:
                        jogo = games_today.loc[dist_val]
                        st.write(
                            f"- **{jogo.get('Home', 'N/A')} vs {jogo.get('Away', 'N/A')}**: "
                            f"Dist = {jogo['Quadrant_Dist']:.2f} | "
                            f"M_H = {jogo.get('M_H', 'N/A'):.2f} | " 
                            f"M_A = {jogo.get('M_A', 'N/A'):.2f} | "
                            f"MT_H = {jogo.get('MT_H', 'N/A'):.2f} | "
                            f"MT_A = {jogo.get('MT_A', 'N/A'):.2f}"
                        )


# Adicione isto também:
st.markdown("### 📖 Guia de Interpretação das Distâncias")

interpretacao_data = {
    "Faixa": ["0.0 - 0.5", "0.5 - 1.0", "1.0 - 1.5", "1.5 - 2.0", "2.0+"],
    "Nível": ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO", "MUITO ALTO"],
    "Significado": [
        "Times praticamente iguais em força e forma",
        "Pequeno desequilíbrio - jogo equilibrado", 
        "Desequilíbrio claro - favorito definido",
        "Desequilíbrio significativo - oportunidade potencial",
        "Desequilíbrio extremo - alta confiança no favorito"
    ],
    "Recomendação": [
        "⚖️ EVITAR - Muito incerto",
        "🤔 ANALISAR - Buscar outros fatores",
        "📊 CONSIDERAR - Bom para análise",
        "🎯 PRIORIZAR - Potencial valor", 
        "⭐ FOCAR - Alto potencial"
    ]
}

interpretacao_df = pd.DataFrame(interpretacao_data)
st.table(interpretacao_df)

# Mostrar os TOP confrontos por distância
st.markdown("### 🏆 Top 10 Confrontos Mais Desequilibrados")
top_distancias = games_today.nlargest(10, 'Quadrant_Dist')[
    ['Home', 'Away', 'League', 'Quadrant_Dist', 'M_H', 'M_A', 'MT_H', 'MT_A', 'Quadrant_Angle']
].copy()

st.dataframe(
    top_distancias.style.format({
        'Quadrant_Dist': '{:.2f}',
        'M_H': '{:.2f}', 'M_A': '{:.2f}',
        'MT_H': '{:.2f}', 'MT_A': '{:.2f}',
        'Quadrant_Angle': '{:.1f}°'
    }).background_gradient(subset=['Quadrant_Dist'], cmap='Reds'),
    use_container_width=True
)



#########################
##### BLOCO EXTRA: ANÁLISE ESTRATÉGICA DISTÂNCIA vs ÂNGULO #####

st.markdown("## 🎯 Análise Estratégica: Distância vs Ângulo")

def analise_distancia_angulo(df):
    """Análise completa da relação entre distância e ângulo"""
    
    # Criar gráfico de dispersão distância vs ângulo
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Classificar por tipo de desequilíbrio
    def classificar_tipo_desequilibrio(angle, dist):
        if dist < 1.0:
            return 'EQUILIBRADO'
        elif 15 <= angle <= 75:  # Ângulo entre 15° e 75°
            return 'CONSISTENTE'
        elif angle < 15:  # Predominância do eixo X (M)
            return 'FORÇA-LIGA'
        else:  # angle > 75 - Predominância do eixo Y (MT)
            return 'FORMA-RECENTE'
    
    df_analise = df.copy()
    df_analise['Tipo_Desequilibrio'] = df_analise.apply(
        lambda x: classificar_tipo_desequilibrio(x.get('Quadrant_Angle', 0), x.get('Quadrant_Dist', 0)), axis=1
    )
    
    # Cores por tipo
    cores = {
        'CONSISTENTE': '#2E8B57',  # Verde
        'FORÇA-LIGA': '#1E90FF',   # Azul
        'FORMA-RECENTE': '#FF8C00', # Laranja
        'EQUILIBRADO': '#A9A9A9'   # Cinza
    }
    
    # Plotar cada tipo
    for tipo, cor in cores.items():
        mask = df_analise['Tipo_Desequilibrio'] == tipo
        if mask.any():
            ax.scatter(
                df_analise.loc[mask, 'Quadrant_Dist'],
                df_analise.loc[mask, 'Quadrant_Angle'], 
                c=cor, s=80, label=tipo, alpha=0.8, 
                edgecolors='black', linewidth=0.5
            )
    
    # Linhas de referência
    ax.axhline(y=45, color='red', linestyle='--', alpha=0.7, label='Ângulo Ideal (45°)', linewidth=2)
    ax.axvline(x=1.5, color='purple', linestyle='--', alpha=0.7, label='Distância Mínima (1.5)', linewidth=2)
    
    # Áreas estratégicas
    ax.axhspan(15, 75, alpha=0.1, color='green', label='Zona Consistente')
    ax.axvspan(1.5, 4.5, alpha=0.1, color='yellow', label='Zona de Oportunidade')
    
    # Configurações do gráfico
    ax.set_xlabel('Distância Euclidiana', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ângulo (graus)', fontsize=14, fontweight='bold')
    ax.set_title('🎯 Mapa Estratégico: Tipo de Desequilíbrio entre Times', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adicionar anotações para os pontos mais importantes
    top_jogos = df_analise.nlargest(8, 'Quadrant_Dist')
    for idx, row in top_jogos.iterrows():
        ax.annotate(
            f"{row['Home'][:8]} vs {row['Away'][:8]}",
            (row['Quadrant_Dist'], row['Quadrant_Angle']),
            xytext=(8, 8), textcoords='offset points',
            fontsize=9, fontweight='bold', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    return fig, df_analise

# Aplicar análise
if not games_today.empty and 'Quadrant_Dist' in games_today.columns and 'Quadrant_Angle' in games_today.columns:
    fig_analise, games_today_analisado = analise_distancia_angulo(games_today)
    st.pyplot(fig_analise)
    
    # Estatísticas por tipo
    st.markdown("### 📊 Estatísticas por Tipo de Desequilíbrio")
    
    stats_tipo = games_today_analisado['Tipo_Desequilibrio'].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CONSISTENTE", stats_tipo.get('CONSISTENTE', 0))
    with col2:
        st.metric("FORÇA-LIGA", stats_tipo.get('FORÇA-LIGA', 0))
    with col3:
        st.metric("FORMA-RECENTE", stats_tipo.get('FORMA-RECENTE', 0))
    with col4:
        st.metric("EQUILIBRADO", stats_tipo.get('EQUILIBRADO', 0))
    
    # Tabela de análise detalhada
    st.markdown("### 📋 Classificação Estratégica dos Confrontos")
    
    cols_tabela = ['Home', 'Away', 'League', 'Quadrant_Dist', 'Quadrant_Angle', 'Tipo_Desequilibrio']
    if 'Recomendacao' in games_today_analisado.columns:
        cols_tabela.append('Recomendacao')
    if 'Quadrante_ML_Score_Main' in games_today_analisado.columns:
        cols_tabela.append('Quadrante_ML_Score_Main')
    
    tabela_analise = games_today_analisado[cols_tabela].copy()
    tabela_analise = tabela_analise.sort_values(['Tipo_Desequilibrio', 'Quadrant_Dist'], ascending=[True, False])
    
    # Função de estilo para a tabela
    def estilo_tabela_estrategica(df):
        def cor_tipo(valor):
            if valor == 'CONSISTENTE': return 'background-color: #90EE90; font-weight: bold'
            elif valor == 'FORÇA-LIGA': return 'background-color: #87CEFA; font-weight: bold'
            elif valor == 'FORMA-RECENTE': return 'background-color: #FFD700; font-weight: bold'
            elif valor == 'EQUILIBRADO': return 'background-color: #D3D3D3'
            else: return ''
        
        styler = df.style.applymap(cor_tipo, subset=['Tipo_Desequilibrio'])
        
        if 'Quadrant_Dist' in df.columns:
            styler = styler.background_gradient(subset=['Quadrant_Dist'], cmap='YlOrRd')
        if 'Quadrante_ML_Score_Main' in df.columns:
            styler = styler.background_gradient(subset=['Quadrante_ML_Score_Main'], cmap='RdYlGn')
            
        return styler
    
    st.dataframe(
        estilo_tabela_estrategica(tabela_analise).format({
            'Quadrant_Dist': '{:.2f}',
            'Quadrant_Angle': '{:.1f}°',
            'Quadrante_ML_Score_Main': '{:.1%}' if 'Quadrante_ML_Score_Main' in tabela_analise.columns else None
        }),
        use_container_width=True
    )
    
    # Guia de interpretação
    st.markdown("### 📖 Guia Estratégico de Interpretação")
    
    with st.expander("🎯 **CLIQUE AQUI para ver o guia completo de estratégias**", expanded=True):
        col_estr1, col_estr2 = st.columns(2)
        
        with col_estr1:
            st.markdown("""
            #### ⭐ **CONSISTENTE** (Ângulo 15°-75°)
            **Característica**: Desequilíbrio balanceado entre força na liga e forma recente
            
            **🎯 ESTRATÉGIA**: 
            - **PRIORITÁRIO** para apostas
            - Modelo ML tem alta confiança
            - Melhor relação sinal/ruído
            
            **📊 Ação**: Apostar conforme recomendação do modelo
            """)
            
            st.markdown("""
            #### 📊 **FORÇA-LIGA** (Ângulo 0°-15°)
            **Característica**: Desequilíbrio vem principalmente da força no campeonato
            
            **🎯 ESTRATÉGIA**: 
            - **ANALISAR** cuidadosamente
            - Verificar se a forma recente confirma
            - Cuidado com times em crise mas historicamente fortes
            
            **📊 Ação**: Validar com outros indicadores antes de apostar
            """)
        
        with col_estr2:
            st.markdown("""
            #### 🎯 **FORMA-RECENTE** (Ângulo 75°-90°)
            **Característica**: Desequilíbrio vem principalmente da forma recente
            
            **🎯 ESTRATÉGIA**: 
            - **OPORTUNIDADE** potencial
            - Mercado pode subestimar a forma
            - Bom para contra-apostas
            
            **📊 Ação**: Buscar value se odds estiverem favoráveis
            """)
            
            st.markdown("""
            #### 🤔 **EQUILIBRADO** (Distância < 1.0)
            **Característica**: Times muito similares em força e forma
            
            **🎯 ESTRATÉGIA**: 
            - **EVITAR** apostas
            - Alta incerteza
            - Menor edge do modelo
            
            **📊 Ação**: Focar em outros jogos mais definidos
            """)
    
    # Análise dos melhores oportunidades

# NO FINAL DO BLOCO EXTRA, substitua a seção "Melhores Oportunidades" por:

st.markdown("### 🏆 Melhores Oportunidades do Dia - COM LADO DA APOSTA")

# Filtrar melhores oportunidades e mostrar o lado
melhores_oportunidades = games_today_analisado[
    (games_today_analisado['Tipo_Desequilibrio'] == 'CONSISTENTE') &
    (games_today_analisado['Quadrant_Dist'] >= 1.5)
].copy()

if not melhores_oportunidades.empty:
    st.success(f"🎉 **ENCONTRADOS {len(melhores_oportunidades)} JOGOS DE ALTA QUALIDADE!**")
    
    # Determinar o lado da aposta baseado no ML Score
    def determinar_lado_aposta(row):
        if 'Quadrante_ML_Score_Home' in row.index and 'Quadrante_ML_Score_Away' in row.index:
            if row['Quadrante_ML_Score_Home'] > row['Quadrante_ML_Score_Away']:
                return 'HOME', row['Quadrante_ML_Score_Home']
            else:
                return 'AWAY', row['Quadrante_ML_Score_Away']
        elif 'ML_Side' in row.index:
            return row['ML_Side'], row.get('Quadrante_ML_Score_Main', 0)
        else:
            return 'ANALISAR', row.get('Quadrante_ML_Score_Main', 0)
    
    # Aplicar determinação do lado
    melhores_oportunidades[['Lado_Aposta', 'Score_Aposta']] = melhores_oportunidades.apply(
        lambda x: pd.Series(determinar_lado_aposta(x)), axis=1
    )
    
    # Colunas para mostrar
    cols_melhores = ['Home', 'Away', 'League', 'Lado_Aposta', 'Score_Aposta', 
                    'Quadrant_Dist', 'Quadrant_Angle', 'Tipo_Desequilibrio']
    
    if 'Recomendacao' in melhores_oportunidades.columns:
        cols_melhores.append('Recomendacao')
    
    # Ordenar por score da aposta
    melhores_ordenados = melhores_oportunidades.sort_values('Score_Aposta', ascending=False)
    
    # Criar tabela com indicação clara do lado
    st.dataframe(
        melhores_ordenados[cols_melhores]
        .style.format({
            'Quadrant_Dist': '{:.2f}',
            'Quadrant_Angle': '{:.1f}°',
            'Score_Aposta': '{:.1%}'
        })
        .applymap(lambda x: 'background-color: #90EE90; font-weight: bold' if x == 'HOME' else 
                 ('background-color: #87CEFA; font-weight: bold' if x == 'AWAY' else ''), 
                 subset=['Lado_Aposta'])
        .background_gradient(subset=['Score_Aposta'], cmap='RdYlGn')
        .background_gradient(subset=['Quadrant_Dist'], cmap='YlOrRd'),
        use_container_width=True
    )
    
    # Resumo executivo das apostas
    st.markdown("#### 📋 RESUMO DAS APOSTAS RECOMENDADAS:")
    
    apostas_home = melhores_ordenados[melhores_ordenados['Lado_Aposta'] == 'HOME']
    apostas_away = melhores_ordenados[melhores_ordenados['Lado_Aposta'] == 'AWAY']
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.subheader("🏠 APOSTAS HOME")
        if not apostas_home.empty:
            for idx, row in apostas_home.iterrows():
                st.write(f"**{row['Home']}** vs {row['Away']}")
                st.write(f"📊 Score: {row['Score_Aposta']:.1%} | Dist: {row['Quadrant_Dist']:.2f}")
                st.write("---")
        else:
            st.info("Nenhuma aposta HOME recomendada")
    
    with col_res2:
        st.subheader("✈️ APOSTAS AWAY") 
        if not apostas_away.empty:
            for idx, row in apostas_away.iterrows():
                st.write(f"{row['Home']} vs **{row['Away']}**")
                st.write(f"📊 Score: {row['Score_Aposta']:.1%} | Dist: {row['Quadrant_Dist']:.2f}")
                st.write("---")
        else:
            st.info("Nenhuma aposta AWAY recomendada")

else:
    st.warning("⚠️ Nenhum jogo na categoria 'CONSISTENTE' com distância > 1.5 encontrado.")

# Adicionar também análise específica dos conflitos com lado
st.markdown("### ⚠️ Análise de Conflitos - COM RECOMENDAÇÃO")

conflitos = games_today_analisado[
    (games_today_analisado['Quadrant_Dist'] >= 2.0) &
    (games_today_analisado['Tipo_Desequilibrio'].isin(['FORÇA-LIGA', 'FORMA-RECENTE']))
].copy()

if not conflitos.empty:
    # Determinar lado para conflitos também
    conflitos[['Lado_Aposta', 'Score_Aposta']] = conflitos.apply(
        lambda x: pd.Series(determinar_lado_aposta(x)), axis=1
    )
    
    st.warning(f"🔍 **ENCONTRADOS {len(conflitos)} JOGOS COM CONFLITO DE SINAIS**")
    
    for idx, row in conflitos.iterrows():
        with st.expander(f"🔎 {row['Home']} vs {row['Away']} - {row['League']} | APOSTA: {row['Lado_Aposta']}"):
            col_conf1, col_conf2 = st.columns(2)
            
            with col_conf1:
                st.write("**📊 Dados do Confronto:**")
                st.write(f"- **Lado Aposta**: {row['Lado_Aposta']}")
                st.write(f"- **Score**: {row['Score_Aposta']:.1%}")
                st.write(f"- Distância: {row['Quadrant_Dist']:.2f}")
                st.write(f"- Ângulo: {row['Quadrant_Angle']:.1f}°")
                st.write(f"- Tipo: {row['Tipo_Desequilibrio']}")
                st.write(f"- M_H: {row.get('M_H', 'N/A'):.2f}")
                st.write(f"- M_A: {row.get('M_A', 'N/A'):.2f}")
            
            with col_conf2:
                st.write("**🎯 Análise do Conflito:**")
                if row['Tipo_Desequilibrio'] == 'FORÇA-LIGA':
                    st.write("**CONFLITO**: Desequilíbrio vem principalmente da FORÇA NA LIGA")
                    if row['Lado_Aposta'] == 'AWAY':
                        st.success("✅ **AWAY é mais forte na liga** - aposta alinhada com o sinal principal")
                    else:
                        st.warning("⚠️ **HOME apostado mas AWAY é mais forte** - analisar cuidadosamente")
                else:  # FORMA-RECENTE
                    st.write("**CONFLITO**: Desequilíbrio vem principalmente da FORMA RECENTE")
                    if row['Lado_Aposta'] == 'HOME':
                        st.success("✅ **HOME em melhor forma** - aposta alinhada com o sinal principal")
                    else:
                        st.warning("⚠️ **AWAY apostado mas HOME em melhor forma** - analisar cuidadosamente")
            
            if 'Recomendacao' in row:
                st.write(f"**🤖 RECOMENDAÇÃO ML**: {row['Recomendacao']}")




else:
    st.warning("⚠️ Dados insuficientes para análise estratégica")

st.markdown("---")




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
