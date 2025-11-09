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
st.title("🎯 Análise de Quadrantes - ML Avançado (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas","coppa", "uefa", "afc", "sudamericana", "copa", "trophy"]

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

def convert_asian_line(line_str):
    """Converte string de linha asiática (ex: '-0.5/1') para valor decimal médio"""
    try:
        if pd.isna(line_str) or str(line_str).strip() == "":
            return None

        line_str = str(line_str).strip()

        # Se não tem "/", apenas converte direto
        if "/" not in line_str:
            val = float(line_str)
            return 0.0 if abs(val) < 1e-10 else val

        # Se tiver "/", precisamos preservar o sinal
        sign = -1 if line_str.strip().startswith("-") else 1

        # Remove o sinal para fazer o split
        parts = [abs(float(x)) for x in line_str.replace("-", "").split("/")]
        avg = sum(parts) / len(parts)

        val = sign * avg
        return 0.0 if abs(val) < 1e-10 else val
    except:
        return None


def calc_handicap_result(margin, asian_line_str, invert=False):
    """Retorna média de pontos por linha (1 win, 0.5 push, 0 loss)"""
    if pd.isna(asian_line_str):
        return np.nan
    
    # A linha já está convertida para perspectiva HOME, então NÃO inverta
    # margin = gh - ga (sempre do ponto de vista do Home)
    try:
        parts = [float(x) for x in str(asian_line_str).split('/')]
    except:
        return np.nan
    
    results = []
    for line in parts:
        # Usando a linha do HOME (já convertida)
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
    
    Segue EXATAMENTE o mesmo padrão do modelo:
    - '0/0.5'   -> +0.25 (away) → -0.25 (home)
    - '-0.5/0'  -> -0.25 (away) → +0.25 (home) 
    - '-1/1.5'  -> -1.25 (away) → +1.25 (home)
    - '1/1.5'   -> +1.25 (away) → -1.25 (home)
    - '1.5'     -> +1.50 (away) → -1.50 (home)
    - '0'       ->  0.00 (away) →  0.00 (home)
    
    Retorna: float ou None
    """
    if pd.isna(line_str) or line_str == "":
        return None
    
    try:
        line_str = str(line_str).strip()
        
        # Caso simples — número único
        if "/" not in line_str:
            num = float(line_str)
            return -num  # Inverte sinal (Away → Home)
        
        # Caso duplo — média dos dois lados com preservação de sinal
        parts = [float(p) for p in line_str.split("/")]
        avg = np.mean(parts)
        
        # Mantém o sinal do primeiro número (como no modelo)
        if str(line_str).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
            
        # Inverte o sinal no final (Away → Home)
        return -result
        
    except (ValueError, TypeError):
        return None

# ---------------- Carregar Dados ----------------
st.info("📂 Carregando dados para análise de quadrantes...")

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
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

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
        return 0  # Neutro/Indefinido
    
    for quadrante_id, config in QUADRANTES_8.items():
        agg_ok = True
        hs_ok = True
        
        # Verificar limites de Aggression
        if 'agg_min' in config and agg < config['agg_min']:
            agg_ok = False
        if 'agg_max' in config and agg > config['agg_max']:
            agg_ok = False
            
        # Verificar limites de HandScore
        if 'hs_min' in config and hs < config['hs_min']:
            hs_ok = False
        if 'hs_max' in config and hs > config['hs_max']:
            hs_ok = False
            
        if agg_ok and hs_ok:
            return quadrante_id
    
    return 0  # Caso não se enquadre em nenhum quadrante

# Aplicar classificação aos dados
games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)

history['Quadrante_Home'] = history.apply(
    lambda x: classificar_quadrante(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
history['Quadrante_Away'] = history.apply(
    lambda x: classificar_quadrante(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)




# ---------------- VISUALIZAÇÃO DOS QUADRANTES ----------------
def plot_quadrantes_avancado(df, side="Home"):
    """Plot dos 8 quadrantes com cores e anotações"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Definir cores para cada quadrante
    cores_quadrantes = {
        1: 'lightgreen',    # Underdog Value Forte
        2: 'green',         # Underdog Value
        3: 'lightcoral',    # Favorite Reliable Forte
        4: 'red',           # Favorite Reliable
        5: 'lightyellow',   # Market Overrates Forte
        6: 'yellow',        # Market Overrates
        7: 'lightgray',     # Weak Underdog Forte
        8: 'gray',          # Weak Underdog
        0: 'black'          # Neutro
    }
    
    # Plotar cada ponto com cor do quadrante
    for quadrante_id in range(9):  # 0-8
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'HandScore_{side}']
            ax.scatter(x, y, c=cores_quadrantes[quadrante_id], 
                      label=QUADRANTES_8.get(quadrante_id, {}).get('nome', 'Neutro'),
                      alpha=0.7, s=50)
    
    # Linhas divisórias dos quadrantes
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(x=-0.5, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=15, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=30, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=-15, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=-30, color='black', linestyle='--', alpha=0.3)
    
    # Anotações dos quadrantes
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

# Exibir gráficos
st.markdown("### 📈 Visualização dos Quadrantes")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_avancado(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_avancado(games_today, "Away"))


def determine_handicap_result(row):
    """
    Determina o resultado do handicap asiático com base no lado recomendado.
    Agora cobre linhas fracionadas (.25 / .75) com half-win / half-loss.
    **ATUALIZADA** para trabalhar com Asian_Line_Decimal já convertido para HOME
    """
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
        asian_line_home = row['Asian_Line_Decimal']  # ← JÁ CONVERTIDO PARA HOME
        recomendacao = str(row.get('Recomendacao', '')).upper()
    except (ValueError, TypeError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_home):
        return None

    # Detectar lado da aposta
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

    # ✅ CORREÇÃO: Para AWAY bets, invertemos a linha HOME para ter perspectiva AWAY
    if is_home_bet:
        asian_line = asian_line_home  # Já está na perspectiva HOME
    else:
        asian_line = -asian_line_home  # Inverte para perspectiva AWAY

    side = "HOME" if is_home_bet else "AWAY"

    # -----------------------
    # Half-win / Half-loss (MANTIDO ORIGINAL - FUNCIONA BEM)
    # -----------------------
    frac = abs(asian_line % 1)
    is_quarter = frac in [0.25, 0.75]

    def single_result(gh, ga, line, side):
        if side == "HOME":
            adjusted = (gh + line) - ga
        else:
            adjusted = (ga + line) - gh  # ✅ CORRIGIDO: (ga + line) - gh

        if adjusted > 0:
            return 1.0
        elif adjusted == 0:
            return 0.5
        else:
            return 0.0

    if is_quarter:
        # Gera as duas linhas equivalentes (ex: +0.25 → +0, +0.5)
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

    # -----------------------
    # Linhas padrão (0, .5, 1, 1.5, etc.)
    # -----------------------
    if side == "HOME":
        adjusted = (gh + asian_line) - ga
    else:
        adjusted = (ga + asian_line) - gh  # ✅ CORRIGIDO: (ga + line) - gh

    if adjusted > 0:
        return f"{side}_COVERED"
    elif adjusted < 0:
        return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"
    else:
        return "PUSH"

def check_handicap_recommendation_correct(rec, handicap_result):
    """Verifica se a recomendação de handicap estava correta"""
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
        return None
    
    rec = str(rec)
    
    # Para recomendações HOME
    if any(keyword in rec for keyword in ['HOME', 'Home', 'VALUE NO HOME', 'FAVORITO HOME', 'MODELO CONFIA HOME']):
        return handicap_result == "COVERED"  # ← MUDOU de "HOME_COVERED" para "COVERED"
    
    # Para recomendações AWAY  
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
        return handicap_result in ["NOT_COVERED", "PUSH"]  # ← MUDOU de "HOME_NOT_COVERED" para "NOT_COVERED"
    
    return None


def calculate_handicap_profit(rec, handicap_result, odds_row, asian_line_decimal):
    """
    Calcula o profit líquido considerando todas as linhas asiáticas (±0.25, ±0.75, ±1.25, etc.)
    com suporte a meia vitória/perda e PUSH.
    
    A linha asiática sempre representa o HANDICAP DO AWAY.
    As odds já são líquidas (não subtrair 1).
    """
    # ===============================
    # 1️⃣ Validações iniciais
    # ===============================
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid' or pd.isna(asian_line_decimal):
        return 0

    rec = str(rec).upper()

    # ===============================
    # 2️⃣ Determinar lado da aposta
    # ===============================
    is_home_bet = any(k in rec for k in ['HOME', 'FAVORITO HOME', 'VALUE NO HOME'])
    is_away_bet = any(k in rec for k in ['AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])

    if not (is_home_bet or is_away_bet):
        return 0

    # ===============================
    # 3️⃣ Selecionar odd correta
    # ===============================
    odd = odds_row.get('Odd_H_Asi', np.nan) if is_home_bet else odds_row.get('Odd_A_Asi', np.nan)
    if pd.isna(odd):
        return 0

    # ===============================
    # 4️⃣ Determinar linhas fracionadas
    # ===============================
    def split_line(line):
        """Divide quarter-lines (±0.25, ±0.75, etc.) em duas sublinhas."""
        frac = abs(line) % 1
        if frac == 0.25:
            base = math.floor(abs(line))
            base = base if line > 0 else -base
            return [base, base + (0.5 if line > 0 else -0.5)]
        elif frac == 0.75:
            base = math.floor(abs(line))
            base = base if line > 0 else -base
            return [base + (0.5 if line > 0 else -0.5), base + (1.0 if line > 0 else -1.0)]
        else:
            return [line]

    # ✅ Como a linha é do AWAY, invertemos o sinal se for aposta HOME
    asian_line_for_eval = -asian_line_decimal if is_home_bet else asian_line_decimal
    lines = split_line(asian_line_for_eval)

    # ===============================
    # 5️⃣ Função auxiliar: resultado individual
    # ===============================
    def single_profit(result):
        """Calcula o lucro individual considerando o resultado e o lado apostado."""
        if result == "PUSH":
            return 0
        elif (is_home_bet and result == "COVERED") or (is_away_bet and result == "NOT_COVERED"):
            return odd  # vitória
        elif (is_home_bet and result == "NOT_COVERED") or (is_away_bet and result == "COVERED"):
            return -1  # derrota
        return 0

    # ===============================
    # 6️⃣ Calcular média dos resultados (para quarter-lines)
    # ===============================
    if len(lines) == 2:
        p1 = single_profit(handicap_result)
        p2 = single_profit(handicap_result)
        return (p1 + p2) / 2
    else:
        return single_profit(handicap_result)



########################################
#### 🧮 BLOCO – Cálculo das Distâncias Home ↔ Away
########################################
def calcular_distancias_quadrantes(df):
    """Calcula distância, separação média e ângulo entre os pontos Home e Away."""
    df = df.copy()
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away']):
        dx = df['Aggression_Home'] - df['Aggression_Away']
        dy = df['HandScore_Home'] - df['HandScore_Away']
        df['Quadrant_Dist'] = np.sqrt(dx**2 + (dy / 60)**2 * 2.5) * 10  # escala visual ajustada
        df['Quadrant_Separation'] = 0.5 * (dy + 60 * dx)
        df['Quadrant_Angle_Geometric'] = np.degrees(np.arctan2(dy, dx))
        df['Quadrant_Angle_Normalized'] = np.degrees(np.arctan2((dy / 60), dx))
    else:
        st.warning("⚠️ Colunas Aggression/HandScore não encontradas para calcular as distâncias.")
        df['Quadrant_Dist'] = np.nan
        df['Quadrant_Separation'] = np.nan
        df['Quadrant_Angle_Geometric'] = np.nan
        df['Quadrant_Angle_Normalized'] = np.nan
    return df


########################################
#### 🎯 BLOCO – Visualização Interativa com Filtro por Liga e Ângulo (versão robusta)
########################################
import plotly.graph_objects as go

st.markdown("## 🎯 Visualização Interativa – Distância entre Times (Home × Away)")

# ==========================
# 🎛️ Filtros interativos
# ==========================
if "League" in games_today.columns and not games_today["League"].isna().all():
    leagues = sorted(games_today["League"].dropna().unique())

    # ✅ Botão para selecionar todas as ligas
    col_select_all, col_multiselect = st.columns([0.25, 0.75])
    with col_select_all:
        select_all = st.checkbox("Selecionar todas as ligas", key="checkbox_select_all_leagues")

    with col_multiselect:
        selected_leagues = st.multiselect(
            "Selecione uma ou mais ligas para análise:",
            options=leagues,
            default=leagues if select_all else [],
            help="Marque 'Selecionar todas as ligas' para incluir todas automaticamente.",
            key="multiselect_leagues"
        )

    # ✅ Filtro principal
    if selected_leagues:
        df_filtered = games_today[games_today["League"].isin(selected_leagues)].copy()
    else:
        df_filtered = games_today.copy()
else:
    st.warning("⚠️ Nenhuma coluna de 'League' encontrada — exibindo todos os jogos.")
    df_filtered = games_today.copy()

# ==========================
# 🎚️ Filtros adicionais (robustos)
# ==========================
max_n = len(df_filtered)

if max_n == 0:
    st.warning("⚠️ Nenhum confronto disponível nesta seleção de ligas.")
    st.stop()

elif max_n == 1:
    st.info("⚠️ Apenas um confronto disponível nesta seleção de ligas.")
    n_to_show = 1  # ✅ evita erro no slider
else:
    n_min, n_max, n_default = 1, min(max_n, 200), min(40, max_n)
    n_to_show = st.slider(
        "Quantos confrontos exibir (Top por distância):",
        min_value=n_min,
        max_value=n_max,
        value=n_default,
        step=1,
        key="slider_n_to_show"
    )

# ==========================
# 🎯 Filtro de ângulo
# ==========================
angle_min, angle_max = st.slider(
    "Filtrar por Ângulo (posição Home vs Away):",
    min_value=-180,
    max_value=180,
    value=(-180, 180),
    step=5,
    help="Ângulos positivos → Home acima | Ângulos negativos → Away acima",
    key="slider_angle_range"
)

# ==========================
# ⚙️ Filtro combinado
# ==========================
use_combined_filter = st.checkbox(
    "Usar filtro combinado (Distância + Ângulo)",
    value=True,
    help="Se desmarcado, exibirá apenas confrontos dentro do intervalo de ângulo.",
    key="checkbox_combined_filter"
)




# ==========================
# 📊 Aplicar filtros
# ==========================
if "Quadrant_Dist" not in df_filtered.columns:
    df_filtered = calcular_distancias_quadrantes(df_filtered)

df_angle = df_filtered[
    (df_filtered["Quadrant_Angle_Normalized"] >= angle_min)
    & (df_filtered["Quadrant_Angle_Normalized"] <= angle_max)
]

if use_combined_filter:
    df_plot = df_angle.nlargest(n_to_show, "Quadrant_Dist").reset_index(drop=True)
else:
    df_plot = df_angle.reset_index(drop=True)

# ==========================
# 🎨 Criar gráfico Plotly
# ==========================
fig = go.Figure()

for _, row in df_plot.iterrows():
    xh, xa = row["Aggression_Home"], row["Aggression_Away"]
    yh, ya = row["HandScore_Home"], row["HandScore_Away"]

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
            f"📏 Distância: {row['Quadrant_Dist']:.2f}<br>"
            f"📐 Ângulo: {row['Quadrant_Angle_Normalized']:.1f}°<br>"
            f"↕️ {'Home acima' if row['Quadrant_Angle_Normalized'] > 0 else 'Away acima'}"
        ),
        showlegend=False
    ))

# Pontos Home e Away
fig.add_trace(go.Scatter(
    x=df_plot["Aggression_Home"],
    y=df_plot["HandScore_Home"],
    mode="markers+text",
    name="Home",
    marker=dict(color="royalblue", size=8, opacity=0.8),
    text=df_plot["Home"],
    textposition="top center",
    hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=df_plot["Aggression_Away"],
    y=df_plot["HandScore_Away"],
    mode="markers+text",
    name="Away",
    marker=dict(color="orangered", size=8, opacity=0.8),
    text=df_plot["Away"],
    textposition="top center",
    hoverinfo="skip"
))

# Eixos de referência
fig.add_trace(go.Scatter(
    x=[-1, 1], y=[0, 0],
    mode="lines", line=dict(color="limegreen", width=2, dash="dash"), name="Eixo X"
))
fig.add_trace(go.Scatter(
    x=[0, 0], y=[-60, 60],
    mode="lines", line=dict(color="limegreen", width=2, dash="dash"), name="Eixo Y"
))

# Layout final
titulo = "Confrontos – Aggression × HandScore"
if use_combined_filter:
    titulo += f" | Top {n_to_show} Distâncias"
if selected_leagues:
    titulo += " | " + ", ".join(selected_leagues)
elif select_all:
    titulo += " | Todas as ligas"

fig.update_layout(
    title=titulo,
    xaxis_title="Aggression (-1 zebra ↔ +1 favorito)",
    yaxis_title="HandScore (-60 ↔ +60)",
    template="plotly_white",
    height=700,
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)


    


########################################
#### 🤖 BLOCO – Treinamento ML Dual (com Quadrant Distance Features)
########################################
from sklearn.ensemble import RandomForestClassifier

def treinar_modelo_quadrantes_dual(history, games_today):
    """
    🔁 Versão atualizada – modelo dual (Home/Away) com target baseado em cobertura real de handicap (AH).
    Compatível com todos os blocos do app (ML2, LiveScore e exibição).
    """

    # -------------------------------
    # 🧮 Calcular distâncias e ângulos
    # -------------------------------
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    # -------------------------------
    # 🎯 Criar target binário (Home cobre AH?)
    # -------------------------------
    def calc_target_handicap_cover(row):
        gh = row.get("Goals_H_FT")
        ga = row.get("Goals_A_FT")
        line_home = row.get("Asian_Line_Decimal")
        if pd.isna(gh) or pd.isna(ga) or pd.isna(line_home):
            return np.nan
        adjusted = (gh + line_home) - ga
        if adjusted > 0:
            return 1   # Home cobre o handicap
        elif adjusted < 0:
            return 0   # Home não cobre (Away vence o AH)
        else:
            return np.nan  # Push

    history["Target_AH_Home"] = history.apply(calc_target_handicap_cover, axis=1)
    history = history.dropna(subset=["Target_AH_Home"]).copy()
    history["Target_AH_Home"] = history["Target_AH_Home"].astype(int)
    history["Target_AH_Away"] = 1 - history["Target_AH_Home"]

    if history["Target_AH_Home"].nunique() < 2:
        st.warning("⚠️ Target insuficiente (todas as classes iguais) — verifique colunas de gols/linha.")
        return None, None, games_today

    # -------------------------------
    # 🧱 Features para treinamento
    # -------------------------------
    qh = pd.get_dummies(history["Quadrante_Home"], prefix="QH")
    qa = pd.get_dummies(history["Quadrante_Away"], prefix="QA")
    leagues = pd.get_dummies(history["League"], prefix="League")
    extras = history[
        ["Quadrant_Dist", "Quadrant_Separation", "Quadrant_Angle_Geometric", "Quadrant_Angle_Normalized"]
    ].fillna(0)

    X = pd.concat([leagues, extras, qh, qa], axis=1)
    y_home = history["Target_AH_Home"]
    y_away = history["Target_AH_Away"]

    # -------------------------------
    # 🤖 Treinar modelos (Home/Away)
    # -------------------------------
    model_home = RandomForestClassifier(
        n_estimators=600, max_depth=12, random_state=42,
        class_weight="balanced_subsample", n_jobs=-1
    )
    model_away = RandomForestClassifier(
        n_estimators=600, max_depth=12, random_state=42,
        class_weight="balanced_subsample", n_jobs=-1
    )

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # -------------------------------
    # 📊 Preparar features para os jogos de hoje
    # -------------------------------
    qh_today = pd.get_dummies(games_today["Quadrante_Home"], prefix="QH").reindex(columns=qh.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today["Quadrante_Away"], prefix="QA").reindex(columns=qa.columns, fill_value=0)
    leagues_today = pd.get_dummies(games_today["League"], prefix="League").reindex(columns=leagues.columns, fill_value=0)
    extras_today = games_today[
        ["Quadrant_Dist", "Quadrant_Separation", "Quadrant_Angle_Geometric", "Quadrant_Angle_Normalized"]
    ].fillna(0)

    X_today = pd.concat([leagues_today, extras_today, qh_today, qa_today], axis=1)

    # -------------------------------
    # 🔮 Fazer previsões (probabilidade de cobrir AH)
    # -------------------------------
    prob_home_cover = model_home.predict_proba(X_today)[:, 1]
    prob_away_cover = model_away.predict_proba(X_today)[:, 1]

    games_today["Quadrante_ML_Score_Home"] = prob_home_cover
    games_today["Quadrante_ML_Score_Away"] = prob_away_cover
    games_today["Quadrante_ML_Score_Main"] = np.maximum(prob_home_cover, prob_away_cover)
    games_today["ML_Side"] = np.where(prob_home_cover > prob_away_cover, "HOME", "AWAY")

    # -------------------------------
    # 📈 Exibir importância das variáveis
    # -------------------------------
    try:
        importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.markdown("### 🔍 Top Features (Modelo HOME – Cobertura AH)")
        st.dataframe(importances.head(15).to_frame("Importância"), use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")

    st.success("✅ Modelo dual atualizado com target real de cobertura de Handicap (AH)!")
    return model_home, model_away, games_today



# ---------------- SISTEMA DE INDICAÇÕES EXPLÍCITAS DUAL ----------------
def adicionar_indicadores_explicativos_dual(df):
    """Adiciona classificações e recomendações explícitas para Home e Away"""
    df = df.copy()
    
    # 1. CLASSIFICAÇÃO DE VALOR PARA HOME
    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.60,
        df['Quadrante_ML_Score_Home'] >= 0.55,
        df['Quadrante_ML_Score_Home'] >= 0.50,
        df['Quadrante_ML_Score_Home'] >= 0.45,
        df['Quadrante_ML_Score_Home'] < 0.45
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')
    
    # 2. CLASSIFICAÇÃO DE VALOR PARA AWAY
    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.60,
        df['Quadrante_ML_Score_Away'] >= 0.55,
        df['Quadrante_ML_Score_Away'] >= 0.50,
        df['Quadrante_ML_Score_Away'] >= 0.45,
        df['Quadrante_ML_Score_Away'] < 0.45
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')
    
    # 3. RECOMENDAÇÃO DE APOSTA DUAL
    def gerar_recomendacao_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']
        
        # Combinações específicas com perspectiva dual
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
    
    # 4. RANKING POR MELHOR PROBABILIDADE
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)
    
    return df

def estilo_tabela_quadrantes_dual(df):
    """Aplica estilo colorido à tabela dual"""
    def cor_classificacao(valor):
        if '🏆 ALTO VALOR' in str(valor): return 'font-weight: bold'
        elif '✅ BOM VALOR' in str(valor): return 'font-weight: bold' 
        elif '🔴 ALTO RISCO' in str(valor): return 'font-weight: bold'
        elif 'VALUE' in str(valor): return 'font-weight: bold'
        elif 'EVITAR' in str(valor): return 'font-weight: bold'
        elif 'SUPERAVALIADO' in str(valor): return 'font-weight: bold'
        else: return ''
    
    # Aplicar apenas às colunas que existem
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
    
    # Aplicar gradientes apenas às colunas que existem
    if 'Quadrante_ML_Score_Home' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')
    if 'Quadrante_ML_Score_Away' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')
    if 'Quadrante_ML_Score_Main' in df.columns:
        styler = styler.background_gradient(subset=['Quadrante_ML_Score_Main'], cmap='RdYlGn')
    
    return styler




# ---------------- ANÁLISE DE PADRÕES DUAL ----------------
def analisar_padroes_quadrantes_dual(df):
    """Analisa padrões recorrentes nas combinações de quadrantes com perspectiva dual"""
    st.markdown("### 🔍 Análise de Padrões por Combinação (Dual)")
    
    padroes = {
        'Underdog Value vs Market Overrates': {
            'descricao': '🎯 **MELHOR PADRÃO HOME** - Zebra com valor vs Favorito supervalorizado',
            'lado_recomendado': 'HOME',
            'prioridade': 1
        },
        'Market Overrates vs Underdog Value': {
            'descricao': '🎯 **MELHOR PADRÃO AWAY** - Favorito supervalorizado vs Zebra com valor', 
            'lado_recomendado': 'AWAY',
            'prioridade': 1
        },
        'Favorite Reliable vs Weak Underdog': {
            'descricao': '💪 **PADRÃO FORTE HOME** - Favorito confiável contra time fraco',
            'lado_recomendado': 'HOME',
            'prioridade': 2
        },
        'Weak Underdog vs Favorite Reliable': {
            'descricao': '💪 **PADRÃO FORTE AWAY** - Time fraco contra favorito confiável',
            'lado_recomendado': 'AWAY', 
            'prioridade': 2
        }
    }
    
    # Ordenar padrões por prioridade
    padroes_ordenados = sorted(padroes.items(), key=lambda x: x[1]['prioridade'])
    
    for padrao, info in padroes_ordenados:
        home_q, away_q = padrao.split(' vs ')
        jogos = df[
            (df['Quadrante_Home_Label'] == home_q) & 
            (df['Quadrante_Away_Label'] == away_q)
        ]
        
        if not jogos.empty:
            st.write(f"**{padrao}**")
            st.write(f"{info['descricao']}")
            
            # Selecionar colunas baseadas no lado recomendado
            if info['lado_recomendado'] == 'HOME':
                score_col = 'Quadrante_ML_Score_Home'
            else:
                score_col = 'Quadrante_ML_Score_Away'
                
            cols_padrao = ['Ranking', 'Home', 'Away', 'League', score_col, 'Recomendacao']
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]
            
            st.dataframe(
                jogos[cols_padrao]
                .sort_values(score_col, ascending=False)
                .style.format({score_col: '{:.1%}'})
                .background_gradient(subset=[score_col], cmap='RdYlGn'),
                use_container_width=True
            )
            st.write("---")

# ---------------- EXECUÇÃO PRINCIPAL ----------------
# Executar treinamento
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_quadrantes_dual(history, games_today)
    st.success("✅ Modelo dual (Home/Away) treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")


########################################
#### 🤖 BLOCO – ML2 PRO (Integrada + Target Contínuo + Meta Confidence)
########################################
from sklearn.ensemble import RandomForestRegressor

def handicap_result_continuous(margin, line):
    """Retorna escore contínuo do resultado do handicap (-1 a +1)."""
    try:
        if pd.isna(margin) or pd.isna(line):
            return np.nan
        
        diff = margin + line
        # Full win
        if diff > 0.5:
            return 1.0
        # Half win
        elif 0 < diff <= 0.5:
            return 0.5
        # Push
        elif diff == 0:
            return 0.0
        # Half loss
        elif -0.5 < diff < 0:
            return -0.5
        # Full loss
        else:
            return -1.0
    except:
        return np.nan


def treinar_ml2_handicap_integrada_pro(history, games_today, model_home, model_away):
    """
    Nova versão da ML2:
    - Usa target contínuo (-1 a +1) para representar força da cobertura
    - Integra saídas da ML1 (meta learning)
    - Retorna probabilidade e meta-confiança combinada
    """

    st.markdown("## ⚙️ Treinando ML2 Pro – Handicap Cover com Contexto da ML1")

    # =====================================================
    # 1️⃣ Criar target contínuo baseado na cobertura real
    # =====================================================
    history = history.copy()
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line_Decimal"]).copy()
    history["Margin_FT"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_Continuous"] = history.apply(
        lambda r: handicap_result_continuous(r["Margin_FT"], -r["Asian_Line_Decimal"]),
        axis=1
    )

    # Normaliza para [0,1] para usar como target de regressão
    history["Target_Continuous"] = (history["Target_Continuous"] + 1) / 2

    # =====================================================
    # 2️⃣ Preparar features (iguais à ML1 + integração ML1)
    # =====================================================
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    extras = history[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle_Geometric',
                      'Quadrant_Angle_Normalized', 'Asian_Line_Decimal']].fillna(0)
    
    X_base = pd.concat([ligas_dummies, extras, quadrantes_home, quadrantes_away], axis=1)

    # =====================================================
    # 3️⃣ Gerar previsões da ML1 para usar como features (meta learning)
    # =====================================================
    try:
        X_base_aligned = X_base.reindex(columns=model_home.feature_names_in_, fill_value=0)
        probas_home = model_home.predict_proba(X_base_aligned)[:, 1]
        probas_away = model_away.predict_proba(X_base_aligned)[:, 1]
        history["ML1_Prob_Home"] = probas_home
        history["ML1_Prob_Away"] = probas_away
        history["ML1_Diff"] = probas_home - probas_away
        X_full = pd.concat([X_base, history[["ML1_Prob_Home", "ML1_Prob_Away", "ML1_Diff"]]], axis=1)
    except Exception as e:
        st.warning(f"⚠️ Não foi possível usar saídas da ML1: {e}")
        X_full = X_base.copy()

    y = history["Target_Continuous"]

    # =====================================================
    # 4️⃣ Treinamento do modelo final (Regressor)
    # =====================================================
    model_handicap = RandomForestRegressor(
        n_estimators=700,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model_handicap.fit(X_full, y)

    # =====================================================
    # 5️⃣ Preparar dados para previsão (games_today)
    # =====================================================
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    extras_today = games_today[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle_Geometric',
                                'Quadrant_Angle_Normalized', 'Asian_Line_Decimal']].fillna(0)

    X_today_base = pd.concat([ligas_today, extras_today, qh_today, qa_today], axis=1)

    # Adicionar as features da ML1 já calculadas no games_today
    if "Quadrante_ML_Score_Home" in games_today.columns:
        X_today_full = pd.concat([
            X_today_base,
            games_today[["Quadrante_ML_Score_Home", "Quadrante_ML_Score_Away"]].rename(
                columns={
                    "Quadrante_ML_Score_Home": "ML1_Prob_Home",
                    "Quadrante_ML_Score_Away": "ML1_Prob_Away"
                }
            )
        ], axis=1)
        X_today_full["ML1_Diff"] = X_today_full["ML1_Prob_Home"] - X_today_full["ML1_Prob_Away"]
    else:
        X_today_full = X_today_base.copy()

    # Alinha colunas
    X_today_full = X_today_full.reindex(columns=X_full.columns, fill_value=0)

    # =====================================================
    # 6️⃣ Previsões finais e meta-confidence
    # =====================================================
    pred_continuous = model_handicap.predict(X_today_full)
    games_today["ML2_Prob_Home_Cover"] = np.clip(pred_continuous, 0, 1)
    games_today["ML2_Pred_Cover"] = np.where(games_today["ML2_Prob_Home_Cover"] >= 0.5, 1, 0)

    # =====================================================
    # 7️⃣ Meta Confidence combinando ML1 + ML2
    # =====================================================
    games_today["Meta_Confidence"] = (
        0.6 * games_today["ML2_Prob_Home_Cover"] +
        0.4 * games_today["Quadrante_ML_Score_Home"]
    )

    # =====================================================
    # 8️⃣ Exibição
    # =====================================================
    st.success("✅ ML2 Pro treinada com sucesso (target contínuo + integração ML1)")
    st.dataframe(
        games_today[["Time", "Home", "Away", 'Goals_H_Today', 'Goals_A_Today', "Asian_Line_Decimal", "ML2_Prob_Home_Cover", "Meta_Confidence"]]
        .sort_values("Meta_Confidence", ascending=False)
        .style.format({
            "Goals_H_Today": "{:.0f}","Goals_A_Today": "{:.0f}",
            "Asian_Line_Decimal": "{:.2f}",
            "ML2_Prob_Home_Cover": "{:.1%}",
            "Meta_Confidence": "{:.1%}"
        })
        .background_gradient(subset=["Meta_Confidence"], cmap="YlGn"),
        use_container_width=True
    )

    # Feature importance
    try:
        importances = pd.Series(model_handicap.feature_importances_, index=X_full.columns).sort_values(ascending=False)
        top_feats = importances.head(15)
        st.markdown("### 🔍 Top Features (ML2 Pro)")
        st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")

    return model_handicap, games_today



########################################
#### 🤖 BLOCO – ML2 PRO (Away Side)
########################################
from sklearn.ensemble import RandomForestRegressor

def handicap_result_continuous_away(margin, line):
    """
    Calcula o escore contínuo (-1 a +1) do resultado do handicap
    para o time visitante (AWAY). A linha recebida é do ponto de vista do HOME.
    """
    try:
        if pd.isna(margin) or pd.isna(line):
            return np.nan
        
        # Invertemos a perspectiva: se a linha é -0.5 para HOME → +0.5 para AWAY
        diff = -margin - line
        if diff > 0.5:
            return 1.0       # Full Win (Away cobre)
        elif 0 < diff <= 0.5:
            return 0.5       # Half Win
        elif diff == 0:
            return 0.0       # Push
        elif -0.5 < diff < 0:
            return -0.5      # Half Loss
        else:
            return -1.0      # Full Loss
    except:
        return np.nan


def treinar_ml2_handicap_away_pro(history, games_today, model_home, model_away):
    """
    ML2 Pro para o lado Away:
    - Target contínuo baseado em cobertura do time visitante
    - Integra saídas da ML1
    - Gera meta-confiança específica para o Away
    """

    st.markdown("## ⚙️ Treinando ML2 Pro – Handicap Cover (Away Side)")

    # =====================================================
    # 1️⃣ Criar target contínuo do lado AWAY
    # =====================================================
    history = history.copy()
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line_Decimal"]).copy()
    history["Margin_FT"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_Continuous_Away"] = history.apply(
        lambda r: handicap_result_continuous_away(r["Margin_FT"], r["Asian_Line_Decimal"]),
        axis=1
    )

    # Normaliza para [0,1]
    history["Target_Continuous_Away"] = (history["Target_Continuous_Away"] + 1) / 2

    # =====================================================
    # 2️⃣ Features e integração ML1
    # =====================================================
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    extras = history[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle_Geometric',
                      'Quadrant_Angle_Normalized', 'Asian_Line_Decimal']].fillna(0)
    
    X_base = pd.concat([ligas_dummies, extras, quadrantes_home, quadrantes_away], axis=1)

    # =====================================================
    # 3️⃣ Adicionar previsões da ML1 como features
    # =====================================================
    try:
        X_base_aligned = X_base.reindex(columns=model_away.feature_names_in_, fill_value=0)
        probas_home = model_home.predict_proba(X_base_aligned)[:, 1]
        probas_away = model_away.predict_proba(X_base_aligned)[:, 1]
        history["ML1_Prob_Home"] = probas_home
        history["ML1_Prob_Away"] = probas_away
        history["ML1_Diff"] = probas_home - probas_away
        X_full = pd.concat([X_base, history[["ML1_Prob_Home", "ML1_Prob_Away", "ML1_Diff"]]], axis=1)
    except Exception as e:
        st.warning(f"⚠️ Não foi possível usar saídas da ML1: {e}")
        X_full = X_base.copy()

    y = history["Target_Continuous_Away"]

    # =====================================================
    # 4️⃣ Treinar modelo final (regressor)
    # =====================================================
    model_handicap_away = RandomForestRegressor(
        n_estimators=700,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model_handicap_away.fit(X_full, y)

    # =====================================================
    # 5️⃣ Preparar dados de hoje
    # =====================================================
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    extras_today = games_today[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle_Geometric',
                                'Quadrant_Angle_Normalized', 'Asian_Line_Decimal']].fillna(0)

    X_today_base = pd.concat([ligas_today, extras_today, qh_today, qa_today], axis=1)

    # Adicionar features da ML1
    if "Quadrante_ML_Score_Home" in games_today.columns:
        X_today_full = pd.concat([
            X_today_base,
            games_today[["Quadrante_ML_Score_Home", "Quadrante_ML_Score_Away"]].rename(
                columns={
                    "Quadrante_ML_Score_Home": "ML1_Prob_Home",
                    "Quadrante_ML_Score_Away": "ML1_Prob_Away"
                }
            )
        ], axis=1)
        X_today_full["ML1_Diff"] = X_today_full["ML1_Prob_Home"] - X_today_full["ML1_Prob_Away"]
    else:
        X_today_full = X_today_base.copy()

    X_today_full = X_today_full.reindex(columns=X_full.columns, fill_value=0)

    # =====================================================
    # 6️⃣ Previsões e Meta Confidence (Away)
    # =====================================================
    pred_continuous = model_handicap_away.predict(X_today_full)
    games_today["ML2_Prob_Away_Cover"] = np.clip(pred_continuous, 0, 1)
    games_today["ML2_Pred_Away_Cover"] = np.where(games_today["ML2_Prob_Away_Cover"] >= 0.5, 1, 0)

    # Meta Confidence (Away)
    games_today["Meta_Confidence_Away"] = (
        0.6 * games_today["ML2_Prob_Away_Cover"] +
        0.4 * games_today["Quadrante_ML_Score_Away"]
    )

    # =====================================================
    # 7️⃣ Exibição
    # =====================================================
    st.success("✅ ML2 Pro (Away) treinada com sucesso (target contínuo + integração ML1)")
    st.dataframe(
        games_today[["Time", "Home", "Away", 'Goals_H_Today', 'Goals_A_Today', "Asian_Line_Decimal", "ML2_Prob_Away_Cover", "Meta_Confidence_Away"]]
        .sort_values("Meta_Confidence_Away", ascending=False)
        .style.format({
            "Goals_H_Today": "{:.0f}",
            "Goals_A_Today": "{:.0f}",
            "Asian_Line_Decimal": "{:.2f}",
            "ML2_Prob_Away_Cover": "{:.1%}",
            "Meta_Confidence_Away": "{:.1%}"
        })
        .background_gradient(subset=["Meta_Confidence_Away"], cmap="RdYlGn"),
        use_container_width=True
    )

    # Importância das features
    try:
        importances = pd.Series(model_handicap_away.feature_importances_, index=X_full.columns).sort_values(ascending=False)
        top_feats = importances.head(15)
        st.markdown("### 🔍 Top Features (ML2 Pro – Away)")
        st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")

    return model_handicap_away, games_today




# ============================================================
# ---------------- EXIBIÇÃO DOS RESULTADOS DUAL ----------------
# ============================================================

st.markdown("## 🏆 Melhores Confrontos por Quadrantes ML (Home & Away)")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()
    ranking_quadrantes['Quadrante_Home_Label'] = ranking_quadrantes['Quadrante_Home'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    ranking_quadrantes['Quadrante_Away_Label'] = ranking_quadrantes['Quadrante_Away'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )

    # Aplicar indicadores explicativos dual (gera a coluna 'Recomendacao')
    ranking_quadrantes = adicionar_indicadores_explicativos_dual(ranking_quadrantes)

    # 🔄 Garante que games_today tenha as mesmas colunas e recomendações
    games_today = ranking_quadrantes.copy()

    # Exibir tabela principal
    st.dataframe(
        ranking_quadrantes[
            [
               "League", "Time", "Home", "Away",  'Goals_H_Today', 'Goals_A_Today',
                "Quadrante_Home_Label", "Quadrante_Away_Label",
                "Quadrante_ML_Score_Home", "Quadrante_ML_Score_Away",
                "Recomendacao","Asian_Line_Decimal"
            ]
        ].style.format({
            "Goals_H_Today": "{:.0f}","Goals_A_Today": "{:.0f}",
            "Quadrante_ML_Score_Home": "{:.2f}",
            "Quadrante_ML_Score_Away": "{:.2f}",
        }),
        use_container_width=True
    )
else:
    st.warning("⚠️ Dados insuficientes para exibir os resultados dual.")


# ============================================================
# 📡 LIVE SCORE MONITOR – SISTEMA 3D (HANDICAP + 1X2)
# ============================================================

# 🔧 Compatibilidade: garante que a coluna 'Recomendacao' exista
if 'Recomendacao' not in games_today.columns:
    if 'Recomendacao_Handicap' in games_today.columns:
        games_today['Recomendacao'] = games_today['Recomendacao_Handicap']
    elif 'Pred_Side' in games_today.columns:
        games_today['Recomendacao'] = games_today['Pred_Side']
    elif 'Indicacao_Final' in games_today.columns:
        games_today['Recomendacao'] = games_today['Indicacao_Final']
    else:
        games_today['Recomendacao'] = ""


# ============================================================
# ⚽ RESULTADO 1X2
# ============================================================
def determine_match_result_1x2(row):
    gh, ga = row.get('Goals_H_Today'), row.get('Goals_A_Today')
    if pd.isna(gh) or pd.isna(ga):
        return None
    if gh > ga:
        return "HOME_WIN"
    elif gh < ga:
        return "AWAY_WIN"
    else:
        return "DRAW"


def check_recommendation_correct_1x2(recomendacao, match_result):
    if pd.isna(recomendacao) or match_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
        return None
    rec = str(recomendacao).upper()
    is_home = any(k in rec for k in ['HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away = any(k in rec for k in ['AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])
    if is_home and match_result == "HOME_WIN":
        return True
    elif is_away and match_result == "AWAY_WIN":
        return True
    else:
        return False


def calculate_profit_1x2(recomendacao, match_result, odds_row):
    if pd.isna(recomendacao) or match_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
        return 0
    rec = str(recomendacao).upper()
    is_home = any(k in rec for k in ['HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away = any(k in rec for k in ['AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])
    if is_home:
        odd = odds_row.get('Odd_H', np.nan)
        won = match_result == "HOME_WIN"
    elif is_away:
        odd = odds_row.get('Odd_A', np.nan)
        won = match_result == "AWAY_WIN"
    else:
        return 0
    if pd.isna(odd):
        return 0
    return (odd - 1) if won else -1


def update_real_time_data_1x2(df):
    df = df.copy()
    df['Result_1x2'] = df.apply(determine_match_result_1x2, axis=1)
    df['Quadrante_Correct_1x2'] = df.apply(
        lambda r: check_recommendation_correct_1x2(r['Recomendacao'], r['Result_1x2']), axis=1)
    df['Profit_1x2'] = df.apply(
        lambda r: calculate_profit_1x2(r['Recomendacao'], r['Result_1x2'], r), axis=1)
    return df


# ============================================================
# ⚖️ RESULTADO HANDICAP ASIÁTICO
# ============================================================
def determine_handicap_result_3d(row):
    try:
        gh = float(row['Goals_H_Today'])
        ga = float(row['Goals_A_Today'])
        asian_line = float(row['Asian_Line_Decimal'])
        rec = str(row.get('Recomendacao', '')).upper()
    except:
        return None
    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line):
        return None
    is_home = any(k in rec for k in ['HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away = any(k in rec for k in ['AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])
    if not (is_home or is_away):
        return None
    side = "HOME" if is_home else "AWAY"
    frac = abs(asian_line % 1)
    is_quarter = frac in [0.25, 0.75]

    def single_result(gh, ga, line, side):
        if side == "HOME":
            adj = (gh + line) - ga
        else:
            adj = (ga - line) - gh
        if adj > 0:
            return 1.0
        elif adj == 0:
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
        r1, r2 = single_result(gh, ga, line1, side), single_result(gh, ga, line2, side)
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
        adj = (gh + asian_line) - ga
    else:
        adj = (ga - asian_line) - gh

    if adj > 0:
        return f"{side}_COVERED"
    elif adj < 0:
        return f"{'AWAY' if side == 'HOME' else 'HOME'}_COVERED"
    else:
        return "PUSH"


def check_handicap_recommendation_correct_3d(rec, result):
    if pd.isna(rec) or result is None or '⚖️ ANALISAR' in str(rec).upper():
        return None
    rec = str(rec).upper()
    is_home = any(k in rec for k in ['HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away = any(k in rec for k in ['AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])
    if is_home and result in ["HOME_COVERED", "HALF_WIN"]:
        return True
    elif is_away and result in ["AWAY_COVERED", "HALF_WIN"]:
        return True
    elif result == "PUSH":
        return None
    else:
        return False


def calculate_handicap_profit_3d(rec, result, odds_row):
    if pd.isna(rec) or result is None or '⚖️ ANALISAR' in str(rec).upper():
        return 0
    rec = str(rec).upper()
    is_home = any(k in rec for k in ['HOME', 'VALUE NO HOME', 'MODELO CONFIA HOME'])
    is_away = any(k in rec for k in ['AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])
    odd = odds_row.get('Odd_H_Asi', np.nan) if is_home else odds_row.get('Odd_A_Asi', np.nan)
    if pd.isna(odd):
        return 0
    if (is_home and result == "HOME_COVERED") or (is_away and result == "AWAY_COVERED"):
        return odd
    elif result == "HALF_WIN":
        return odd / 2
    elif result == "HALF_LOSS":
        return -0.5
    elif result == "PUSH":
        return 0
    else:
        return -1


def update_real_time_data_3d(df):
    df = df.copy()
    df['Handicap_Result'] = df.apply(determine_handicap_result_3d, axis=1)
    df['Quadrante_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct_3d(r['Recomendacao'], r['Handicap_Result']), axis=1)
    df['Profit_Quadrante'] = df.apply(
        lambda r: calculate_handicap_profit_3d(r['Recomendacao'], r['Handicap_Result'], r), axis=1)
    return df


# ============================================================
# 🔄 EXECUÇÃO DO LIVE SCORE COMPARATIVO
# ============================================================
games_today = update_real_time_data_3d(games_today)
games_today = update_real_time_data_1x2(games_today)

st.markdown("## 📡 Live Score Monitor – Sistema 3D (AH + 1x2)")

# --- Handicap (AH)
finished_ah = games_today[games_today['Handicap_Result'].notna()]
if not finished_ah.empty:
    bets = finished_ah['Quadrante_Correct'].notna().sum()
    correct = finished_ah['Quadrante_Correct'].sum()
    profit = finished_ah['Profit_Quadrante'].sum()
    roi = profit / bets if bets > 0 else 0
    st.metric("Apostas (AH)", bets)
    st.metric("Winrate (AH)", f"{correct/bets:.1%}")
    st.metric("Lucro Total (AH)", f"{profit:.2f}u")
    st.metric("ROI (AH)", f"{roi:.1%}")
else:
    st.info("⚠️ Nenhum jogo finalizado ainda para o sistema Handicap.")

# --- 1x2
finished_1x2 = games_today[games_today['Result_1x2'].notna()]
if not finished_1x2.empty:
    bets = finished_1x2['Quadrante_Correct_1x2'].notna().sum()
    correct = finished_1x2['Quadrante_Correct_1x2'].sum()
    profit = finished_1x2['Profit_1x2'].sum()
    roi = profit / bets if bets > 0 else 0
    st.metric("Apostas (1x2)", bets)
    st.metric("Winrate (1x2)", f"{correct/bets:.1%}")
    st.metric("Lucro Total (1x2)", f"{profit:.2f}u")
    st.metric("ROI (1x2)", f"{roi:.1%}")
else:
    st.info("⚠️ Nenhum jogo finalizado ainda para o sistema 1x2.")


# ============================================================
# ⚖️ COMPARATIVO – AH x 1x2
# ============================================================
def compare_systems_summary(df):
    def calc(correct_col, profit_col):
        valid = df[correct_col].notna().sum()
        correct = df[correct_col].sum(skipna=True)
        profit = df[profit_col].sum()
        roi = profit / valid if valid > 0 else 0
        winrate = correct / valid if valid > 0 else 0
        return valid, winrate, profit, roi

    ah_bets, ah_win, ah_profit, ah_roi = calc("Quadrante_Correct", "Profit_Quadrante")
    x2_bets, x2_win, x2_profit, x2_roi = calc("Quadrante_Correct_1x2", "Profit_1x2")

    resumo = pd.DataFrame({
        "Métrica": ["Apostas", "Winrate", "Lucro Total", "ROI"],
        "Sistema Handicap (AH)": [ah_bets, f"{ah_win:.1%}", f"{ah_profit:.2f}", f"{ah_roi:.1%}"],
        "Sistema 1x2": [x2_bets, f"{x2_win:.1%}", f"{x2_profit:.2f}", f"{x2_roi:.1%}"]
    })

    st.markdown("### ⚖️ Comparativo de Performance – AH vs 1x2")
    st.dataframe(resumo, use_container_width=True)



# Executar comparativo final
compare_systems_summary(games_today)



if not history.empty:
    model_handicap, games_today = treinar_ml2_handicap_integrada_pro(history, games_today, modelo_home, modelo_away)
    model_handicap_away, games_today = treinar_ml2_handicap_away_pro(history, games_today, modelo_home, modelo_away)

else:
    st.warning("⚠️ Histórico vazio – não foi possível treinar a ML2 Pro.")



    


st.markdown("---")
st.info("🎯 **Análise de Quadrantes ML Dual** - Sistema avançado para identificação de value bets em Home e Away baseado em Aggression × HandScore")
