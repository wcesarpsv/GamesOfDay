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

st.set_page_config(page_title="Análise de Quadrantes - Bet Indicator", layout="wide")
st.title("🎯 Análise de Quadrantes - ML Avançado (Home & Away)")

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
    if invert:
        margin = -margin
    try:
        parts = [float(x) for x in str(asian_line_str).split('/')]
    except:
        return np.nan
    results = []
    for line in parts:
        if margin > line:
            results.append(1.0)
        elif margin == line:
            results.append(0.5)
        else:
            results.append(0.0)
    return np.mean(results)


def convert_asian_line_to_decimal(line_str):
    """Converte qualquer formato de Asian Line para valor decimal único"""
    if pd.isna(line_str) or line_str == "":
        return None
    
    try:
        line_str = str(line_str).strip()
        
        # Se não tem "/" é valor único
        if "/" not in line_str:
            return float(line_str)
        
        # Se tem "/" é linha fracionada - calcular média
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
        
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
            'game_id', 'status', 'home_goal', 'away_goal',
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
                right_on='game_id',
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
    lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False) > 0.5 else 0, axis=1
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
    """Determina se o HOME cobriu o handicap (linha do Away invertida)"""
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
        asian_line_decimal = row.get('Asian_Line_Decimal')  # Linha do Away
    except (ValueError, TypeError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_decimal):
        return None
    
    # Calcular margin 
    margin = gh - ga
    
    # LINHA DO AWAY → inverter sinal para cálculo do Home
    handicap_result = calc_handicap_result(margin, asian_line_decimal, invert=True)
    
    # Determinar resultado
    if handicap_result > 0.5:
        return "HOME_COVERED"
    elif handicap_result == 0.5:
        return "PUSH"
    else:
        return "HOME_NOT_COVERED"

def check_handicap_recommendation_correct(rec, handicap_result):
    """Verifica se a recomendação de handicap estava correta"""
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
        return None
    
    rec = str(rec)
    
    # Para recomendações HOME (Home deve cobrir)
    if any(keyword in rec for keyword in ['HOME', 'Home', 'VALUE NO HOME', 'FAVORITO HOME']):
        return handicap_result == "HOME_COVERED"
    
    # Para recomendações AWAY (Home NÃO deve cobrir)  
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY']):
        return handicap_result in ["HOME_NOT_COVERED", "PUSH"]
    
    return None

def calculate_handicap_profit(rec, handicap_result, odds_row):
    """Calcula profit baseado no resultado do handicap"""
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
        return 0
    
    rec = str(rec)
    
    # Para recomendações HOME - usar Odd_H
    if any(keyword in rec for keyword in ['HOME', 'Home', 'VALUE NO HOME', 'FAVORITO HOME']):
        odd = odds_row.get('Odd_H', np.nan)
        if handicap_result == "HOME_COVERED":
            return odd - 1  # WIN
        else:
            return -1  # LOSS
    
    # Para recomendações AWAY - usar Odd_A  
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY']):
        odd = odds_row.get('Odd_A', np.nan)
        if handicap_result in ["HOME_NOT_COVERED", "PUSH"]:
            return odd - 1  # WIN
        else:
            return -1  # LOSS
    
    return 0


# ---------------- FUNÇÕES LIVE SCORE ----------------



# ---------------- ML PARA RANKEAR CONFRONTOS (HOME E AWAY) ----------------
def treinar_modelo_quadrantes_dual(history, games_today):
    """Treina modelo ML para Home e Away baseado em quadrantes + ligas"""
    
    # Preparar features: one-hot encoding de quadrantes e ligas
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    
    # Combinar features
    X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies], axis=1)
    
    # Target para Home
    y_home = history['Target_AH_Home']
    
    # Target para Away (inverso do Home)
    y_away = 1 - y_home  # Quando Home perde, Away ganha
    
    # Garantir que todas as colunas existem
    quadrante_cols = list(quadrantes_home.columns) + list(quadrantes_away.columns)
    liga_cols = list(ligas_dummies.columns)
    
    # Treinar modelo para Home
    model_home = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model_home.fit(X, y_home)
    
    # Treinar modelo para Away
    model_away = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model_away.fit(X, y_away)
    
    # Preparar dados de hoje
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=liga_cols, fill_value=0)
    
    X_today = pd.concat([qh_today, qa_today, ligas_today], axis=1)
    
    # Fazer previsões para ambos os lados
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]
    
    games_today['Quadrante_ML_Score_Home'] = probas_home
    games_today['Quadrante_ML_Score_Away'] = probas_away
    
    # Score principal (maior probabilidade entre Home e Away)
    games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')
    
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
        if '🏆 ALTO VALOR' in str(valor): return 'background-color: #90EE90; font-weight: bold'
        elif '✅ BOM VALOR' in str(valor): return 'background-color: #FFFFE0; font-weight: bold' 
        elif '🔴 ALTO RISCO' in str(valor): return 'background-color: #FFB6C1; font-weight: bold'
        elif 'VALUE' in str(valor): return 'background-color: #98FB98'
        elif 'EVITAR' in str(valor): return 'background-color: #FFCCCB'
        elif 'SUPERAVALIADO' in str(valor): return 'background-color: #FFA07A'
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



# ---------------- EXIBIÇÃO DOS RESULTADOS DUAL ----------------
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
    
    # Aplicar indicadores explicativos dual
    ranking_quadrantes = adicionar_indicadores_explicativos_dual(ranking_quadrantes)

    # ---------------- ATUALIZAR COM DADOS LIVE ----------------
    def update_real_time_data(df):
        """Atualiza todos os dados em tempo real para HANDICAP"""
        # Resultados do handicap
        df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
        
        # Performance das recomendações (baseado no handicap)
        df['Quadrante_Correct'] = df.apply(
            lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
        )
        df['Profit_Quadrante'] = df.apply(
            lambda r: calculate_handicap_profit(r['Recomendacao'], r['Handicap_Result'], r), axis=1
        )
        return df

    # Aplicar atualização em tempo real
    ranking_quadrantes = update_real_time_data(ranking_quadrantes)
    
    # ---------------- RESUMO LIVE ----------------
    def generate_live_summary(df):
        """Gera resumo em tempo real dos resultados"""
        finished_games = df.dropna(subset=['Result_Today'])
        
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
        correct_bets = quadrante_bets['Quadrante_Correct'].sum()
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

    # Exibir resumo live APÓS criar ranking_quadrantes
    st.markdown("## 📡 Live Score Monitor")
    live_summary = generate_live_summary(ranking_quadrantes)
    st.json(live_summary)
    
    # Ordenar por score principal (se existir) ou pelo score do home
    if 'Quadrante_ML_Score_Main' in ranking_quadrantes.columns:
        ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score_Main', ascending=False)
    else:
        ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score_Home', ascending=False)
    
    # Colunas para exibir - incluindo Live Score
    colunas_possiveis = [
        'Ranking', 'Home', 'Away', 'League', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Quadrante_ML_Score_Main', 'Classificacao_Valor_Home', 
        'Classificacao_Valor_Away', 'Recomendacao',
        # Colunas Live Score
        'Goals_H_Today', 'Goals_A_Today', 'Asian_Line_Decimal', 'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
    ]
    
    # Filtrar colunas existentes
    cols_finais = [c for c in colunas_possiveis if c in ranking_quadrantes.columns]
    
    st.dataframe(
        estilo_tabela_quadrantes_dual(ranking_quadrantes[cols_finais].head(20))
        .format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Home_Red': '{:.0f}',
            'Away_Red': '{:.0f}',
            'Profit_Quadrante': '{:.2f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Quadrante_ML_Score_Main': '{:.1%}'
        }, na_rep="-"),
        use_container_width=True
    )
    
else:
    st.info("⚠️ Aguardando dados para gerar ranking dual")
    


# ---------------- RESUMO EXECUTIVO DUAL ----------------
def resumo_quadrantes_hoje_dual(df):
    """Resumo executivo dos quadrantes de hoje com perspectiva dual"""
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jogos", total_jogos)
    with col2:
        st.metric("🎯 Alto Valor Home", alto_valor_home)
    with col3:
        st.metric("🎯 Alto Valor Away", alto_valor_away)
    with col4:
        st.metric("📊 Home vs Away", f"{home_recomendado} : {away_recomendado}")
    
    # Distribuição de recomendações
    st.markdown("#### 📊 Distribuição de Recomendações")
    dist_recomendacoes = df['Recomendacao'].value_counts()
    st.dataframe(dist_recomendacoes, use_container_width=True)

if not games_today.empty and 'Classificacao_Valor_Home' in games_today.columns:
    resumo_quadrantes_hoje_dual(games_today)

st.markdown("---")
st.info("🎯 **Análise de Quadrantes ML Dual** - Sistema avançado para identificação de value bets em Home e Away baseado em Aggression × HandScore")
