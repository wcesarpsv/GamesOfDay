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

# ---------------- NOVO SISTEMA DE 16 QUADRANTES ----------------
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

########################################
#### 🧮 BLOCO – Cálculo das Distâncias + Vetor Angular (sin/cos)
########################################
def calcular_distancias_quadrantes(df):
    """
    Calcula:
      - Distância entre Home e Away (Quadrant_Dist)
      - Separação média (Quadrant_Separation)
      - Direção vetorial com sin/cos (em vez de ângulo bruto)
    """
    df = df.copy()
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away']):
        # ✅ Vetor Home → Away
        dx = df['Aggression_Away'] - df['Aggression_Home']
        dy = df['HandScore_Away'] - df['HandScore_Home']

        # 📏 Distância euclidiana ajustada
        df['Quadrant_Dist'] = np.sqrt(dx**2 + (dy/60)**2 * 2.5) * 10

        # 📐 Separação linear (mantida)
        df['Quadrant_Separation'] = 0.5 * (dy + 60 * dx)

        # 🧭 Ângulo e projeções trigonométricas
        angle = np.arctan2(dy, dx)
        df['Quadrant_Sin'] = np.sin(angle)
        df['Quadrant_Cos'] = np.cos(angle)

    else:
        st.warning("⚠️ Colunas Aggression/HandScore não encontradas para calcular distâncias.")
        df['Quadrant_Dist'] = np.nan
        df['Quadrant_Separation'] = np.nan
        df['Quadrant_Sin'] = np.nan
        df['Quadrant_Cos'] = np.nan

    return df


# Aplicar ao games_today
games_today = calcular_distancias_quadrantes(games_today)

# # ---------------- VISUALIZAÇÃO DOS 16 QUADRANTES ----------------
# def plot_quadrantes_16(df, side="Home"):
#     """Plot dos 16 quadrantes com cores e anotações"""
#     fig, ax = plt.subplots(figsize=(14, 10))
    
#     # Definir cores por categoria
#     # cores_categorias = {
#     #     'Fav Forte': 'blue',
#     #     'Fav Moderado': 'black', 
#     #     'Under Moderado': 'black',
#     #     'Under Forte': 'red'
#     # }
#     # 🎨 Cores nomeadas por quadrante (tons claros = neutro / escuros = extremos)
#     cores_quadrantes_16 = {
#         1: 'blue', 2: 'deepskyblue', 3: 'blue', 4: 'red',          # Fav Forte
#         5: 'lightgreen', 6: 'mediumseagreen', 7: 'green', 8: 'darkgreen',    # Fav Moderado
#         9: 'moccasin', 10: 'gold', 11: 'orange', 12: 'chocolate',            # Under Moderado
#         13: 'lightcoral', 14: 'indianred', 15: 'red', 16: 'darkred'          # Under Forte
#     }

    
#     # Plotar cada ponto com cor da categoria
#     for quadrante_id in range(1, 17):
#         mask = df[f'Quadrante_{side}'] == quadrante_id
#         if mask.any():
#             categoria = QUADRANTES_16[quadrante_id]['nome'].split()[0] + ' ' + QUADRANTES_16[quadrante_id]['nome'].split()[1]
#             # cor = cores_categorias.get(categoria, 'gray')
#              cor = cores_quadrantes_16.get(quadrante_id, 'gray')
            
#             x = df.loc[mask, f'Aggression_{side}']
#             y = df.loc[mask, f'HandScore_{side}']
#             ax.scatter(x, y, c=cor, 
#                       label=QUADRANTES_16[quadrante_id]['nome'],
#                       alpha=0.7, s=50)
    
#     # Linhas divisórias dos quadrantes (Aggression)
#     for x in [-0.75, -0.25, 0.25, 0.75]:
#         ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
#     ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
#     # Linhas divisórias dos quadrantes (HandScore)  
#     for y in [-45, -30, -15, 15, 30, 45]:
#         ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
#     ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
#     # Anotações dos quadrantes
#     annot_config = [
#         (0.875, 52.5, "Fav Forte\nMuito Forte", 8), (0.875, 37.5, "Fav Forte\nForte", 8),
#         (0.875, 22.5, "Fav Forte\nModerado", 8), (0.875, 0, "Fav Forte\nNeutro", 8),
#         (0.5, 52.5, "Fav Moderado\nMuito Forte", 8), (0.5, 37.5, "Fav Moderado\nForte", 8),
#         (0.5, 22.5, "Fav Moderado\nModerado", 8), (0.5, 0, "Fav Moderado\nNeutro", 8),
#         (-0.5, 0, "Under Moderado\nNeutro", 8), (-0.5, -22.5, "Under Moderado\nModerado", 8),
#         (-0.5, -37.5, "Under Moderado\nForte", 8), (-0.5, -52.5, "Under Moderado\nMuito Forte", 8),
#         (-0.875, 0, "Under Forte\nNeutro", 8), (-0.875, -22.5, "Under Forte\nModerado", 8),
#         (-0.875, -37.5, "Under Forte\nForte", 8), (-0.875, -52.5, "Under Forte\nMuito Forte", 8)
#     ]
    
#     for x, y, text, fontsize in annot_config:
#         ax.text(x, y, text, ha='center', fontsize=fontsize, weight='bold')
    
#     ax.set_xlabel(f'Aggression_{side} (-1 zebra ↔ +1 favorito)')
#     ax.set_ylabel(f'HandScore_{side} (-60 a +60)')
#     ax.set_title(f'16 Quadrantes - {side}')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     return fig

########################################
#### 🎨 Função de Plotagem – 16 Quadrantes com Cores Nomeadas
########################################
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
    ax.set_xlabel(f"Aggression_{side} (-1 zebra ↔ +1 favorito)", fontsize=11)
    ax.set_ylabel(f"HandScore_{side} (-60 ↔ +60)", fontsize=11)
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

# ---------------- VISUALIZAÇÃO INTERATIVA ----------------
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
            f"🎯 Home: {QUADRANTES_16.get(row['Quadrante_Home'], {}).get('nome', 'N/A')}<br>"
            f"🎯 Away: {QUADRANTES_16.get(row['Quadrante_Away'], {}).get('nome', 'N/A')}<br>"
            f"📏 Distância: {row['Quadrant_Dist']:.2f}"
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

# Layout
titulo = f"Top {n_to_show} Distâncias – 16 Quadrantes"
if selected_league != "⚽ Todas as ligas":
    titulo += f" | {selected_league}"

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

# ---------------- MODELO ML ATUALIZADO PARA 16 QUADRANTES ----------------
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
    extras = history[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos']].fillna(0)

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
    extras_today = games_today[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Sin', 'Quadrant_Cos']].fillna(0)

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

# ---------------- SISTEMA DE INDICAÇÕES PARA 16 QUADRANTES ----------------
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

# ---------------- EXECUÇÃO PRINCIPAL ----------------
# Executar treinamento
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_quadrantes_16_dual(history, games_today)
    st.success("✅ Modelo dual com 16 quadrantes treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")


# ---------------- ANÁLISE DE PADRÕES PARA 16 QUADRANTES ----------------
def analisar_padroes_quadrantes_16_dual(df):
    """Analisa padrões recorrentes nas combinações de 16 quadrantes"""
    st.markdown("### 🔍 Análise de Padrões por Combinação (16 Quadrantes)")
    
    # Padrões prioritários para 16 quadrantes
    padroes_16 = {
        'Fav Forte Forte vs Under Forte Muito Forte': {
            'descricao': '🎯 **MELHOR PADRÃO HOME** - Favorito forte contra underdog muito fraco',
            'lado_recomendado': 'HOME',
            'prioridade': 1,
            'score_min': 0.65
        },
        'Under Forte Muito Forte vs Fav Forte Forte': {
            'descricao': '🎯 **MELHOR PADRÃO AWAY** - Underdog muito fraco contra favorito forte',
            'lado_recomendado': 'AWAY', 
            'prioridade': 1,
            'score_min': 0.65
        },
        'Fav Moderado Forte vs Under Moderado Forte': {
            'descricao': '💪 **PADRÃO VALUE HOME** - Favorito moderado contra underdog moderado fraco',
            'lado_recomendado': 'HOME',
            'prioridade': 2,
            'score_min': 0.58
        },
        'Under Moderado Forte vs Fav Moderado Forte': {
            'descricao': '💪 **PADRÃO VALUE AWAY** - Underdog moderado fraco contra favorito moderado',
            'lado_recomendado': 'AWAY',
            'prioridade': 2, 
            'score_min': 0.58
        },
        'Fav Forte Neutro vs Under Forte Neutro': {
            'descricao': '📊 **PADRÃO NEUTRO HOME** - Favorito forte neutro contra underdog neutro',
            'lado_recomendado': 'HOME',
            'prioridade': 3,
            'score_min': 0.55
        },
        'Under Forte Neutro vs Fav Forte Neutro': {
            'descricao': '📊 **PADRÃO NEUTRO AWAY** - Underdog neutro contra favorito forte neutro',
            'lado_recomendado': 'AWAY',
            'prioridade': 3,
            'score_min': 0.55
        }
    }
    
    # Ordenar padrões por prioridade
    padroes_ordenados = sorted(padroes_16.items(), key=lambda x: x[1]['prioridade'])
    
    for padrao, info in padroes_ordenados:
        home_q, away_q = padrao.split(' vs ')
        
        # Buscar jogos que correspondem ao padrão
        jogos = df[
            (df['Quadrante_Home_Label'] == home_q) & 
            (df['Quadrante_Away_Label'] == away_q)
        ]
        
        # Filtrar por score mínimo se especificado
        if info['lado_recomendado'] == 'HOME':
            score_col = 'Quadrante_ML_Score_Home'
        else:
            score_col = 'Quadrante_ML_Score_Away'
            
        if 'score_min' in info:
            jogos = jogos[jogos[score_col] >= info['score_min']]
        
        if not jogos.empty:
            st.write(f"**{padrao}**")
            st.write(f"{info['descricao']}")
            st.write(f"📈 **Score mínimo**: {info.get('score_min', 0.50):.1%}")
            st.write(f"🎯 **Jogos encontrados**: {len(jogos)}")
            
            # Colunas para exibir
            cols_padrao = ['Ranking', 'Home', 'Away', 'League', score_col, 'Recomendacao', 'Quadrant_Dist']
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]
            
            # Ordenar por score
            jogos_ordenados = jogos.sort_values(score_col, ascending=False)
            
            st.dataframe(
                jogos_ordenados[cols_padrao]
                .head(10)
                .style.format({
                    score_col: '{:.1%}',
                    'Quadrant_Dist': '{:.2f}'
                })
                .background_gradient(subset=[score_col], cmap='RdYlGn'),
                use_container_width=True
            )
            st.write("---")

# ---------------- ESTRATÉGIAS AVANÇADAS PARA 16 QUADRANTES ----------------
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
        st.write(f"**{categoria}**")
        st.write(f"📋 {info['descricao']}")
        st.write(f"🎯 Estratégia: {info['estrategia']}")
        st.write(f"📊 Confiança: {info['confianca']}")
        
        # Mostrar quadrantes específicos
        quadrantes_str = ", ".join([f"Q{q}" for q in info['quadrantes']])
        st.write(f"🔢 Quadrantes: {quadrantes_str}")
        
        # Estatísticas da categoria
        jogos_categoria = df[df['Quadrante_Home'].isin(info['quadrantes']) | 
                            df['Quadrante_Away'].isin(info['quadrantes'])]
        
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
        
        st.write("---")

# ---------------- SISTEMA DE SCORING PARA 16 QUADRANTES ----------------
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

# ---------------- EXIBIÇÃO DOS RESULTADOS PARA 16 QUADRANTES ----------------
st.markdown("## 🏆 Melhores Confrontos por 16 Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()
    
    # Aplicar indicadores explicativos para 16 quadrantes
    ranking_quadrantes = adicionar_indicadores_explicativos_16_dual(ranking_quadrantes)
    
    # Aplicar scoring combinado
    ranking_quadrantes = gerar_score_combinado_16(ranking_quadrantes)
    
    # ---------------- ATUALIZAR COM DADOS LIVE ----------------
    def determine_handicap_result(row):
        """Determina se o HOME cobriu o handicap"""
        try:
            gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
            ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
            asian_line_decimal = row.get('Asian_Line_Decimal')
        except (ValueError, TypeError):
            return None

        if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_decimal):
            return None
        
        margin = gh - ga
        handicap_result = calc_handicap_result(margin, asian_line_decimal, invert=False)
        
        if handicap_result > 0.5:
            return "HOME_COVERED"
        elif handicap_result == 0.5:
            return "PUSH"
        else:
            return "HOME_NOT_COVERED"

    def check_handicap_recommendation_correct(rec, handicap_result):
        """Verifica se a recomendação estava correta"""
        if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
            return None
        
        rec = str(rec)
        
        if any(keyword in rec for keyword in ['HOME', 'Home', 'VALUE NO HOME', 'FAVORITO HOME']):
            return handicap_result == "HOME_COVERED"
        elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
            return handicap_result in ["HOME_NOT_COVERED", "PUSH"]
        
        return None

    def calculate_handicap_profit(rec, handicap_result, odds_row, asian_line_decimal):
        """Calcula profit para handicap asiático"""
        if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid' or pd.isna(asian_line_decimal):
            return 0

        rec = str(rec).upper()
        is_home_bet = any(k in rec for k in ['HOME', 'FAVORITO HOME', 'VALUE NO HOME'])
        is_away_bet = any(k in rec for k in ['AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 'MODELO CONFIA AWAY'])

        if not (is_home_bet or is_away_bet):
            return 0

        odd = odds_row.get('Odd_H_Asi', np.nan) if is_home_bet else odds_row.get('Odd_A_Asi', np.nan)
        if pd.isna(odd):
            return 0

        def split_line(line):
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

        asian_line_for_eval = -asian_line_decimal if is_home_bet else asian_line_decimal
        lines = split_line(asian_line_for_eval)

        def single_profit(result):
            if result == "PUSH":
                return 0
            elif (is_home_bet and result == "HOME_COVERED") or (is_away_bet and result == "HOME_NOT_COVERED"):
                return odd
            elif (is_home_bet and result == "HOME_NOT_COVERED") or (is_away_bet and result == "HOME_COVERED"):
                return -1
            return 0

        if len(lines) == 2:
            p1 = single_profit(handicap_result)
            p2 = single_profit(handicap_result)
            return (p1 + p2) / 2
        else:
            return single_profit(handicap_result)

    def update_real_time_data(df):
        """Atualiza todos os dados em tempo real"""
        df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
        df['Quadrante_Correct'] = df.apply(
            lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
        )
        df['Profit_Quadrante'] = df.apply(
            lambda r: calculate_handicap_profit(r['Recomendacao'], r['Handicap_Result'], r, r['Asian_Line_Decimal']), axis=1
        )
        return df

    # Aplicar atualização em tempo real
    ranking_quadrantes = update_real_time_data(ranking_quadrantes)
    
    # ---------------- RESUMO LIVE ----------------
    def generate_live_summary(df):
        """Gera resumo em tempo real"""
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

    # Exibir resumo live
    st.markdown("## 📡 Live Score Monitor - 16 Quadrantes")
    live_summary = generate_live_summary(ranking_quadrantes)
    st.json(live_summary)
    
    # Ordenar por score final
    ranking_quadrantes = ranking_quadrantes.sort_values('Score_Final', ascending=False)
    
    # Colunas para exibir
    colunas_possiveis = [
        'Ranking', 'League', 'Home', 'Away', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Score_Final', 'Classificacao_Potencial',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao',
        # Colunas Live Score
        'Goals_H_Today', 'Goals_A_Today', 'Asian_Line_Decimal', 'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
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
        estilo_tabela_16_quadrantes(ranking_quadrantes[cols_finais].head(25))
        .format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Home_Red': '{:.0f}',
            'Away_Red': '{:.0f}',
            'Profit_Quadrante': '{:.2f}',
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

# ---------------- RESUMO EXECUTIVO PARA 16 QUADRANTES ----------------
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

# Garantir que o histórico tenha os vetores sin/cos
history = calcular_distancias_quadrantes(history)

########################################
### 📊 BLOCO – Mapa Angular de Valor (EV Map)
########################################
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("## 🧭 Mapa Angular de Valor – Espaço Vetorial (sin/cos)")

try:
    # ✅ Garantir que o histórico tenha sin/cos e target
    df_ev = history.copy()
    df_ev = df_ev.dropna(subset=['Quadrant_Sin', 'Quadrant_Cos', 'Target_AH_Home'])
    
    # 🔹 Discretizar o plano (binning 2D)
    bins = 30
    df_ev['bin_sin'] = pd.cut(df_ev['Quadrant_Sin'], bins=bins)
    df_ev['bin_cos'] = pd.cut(df_ev['Quadrant_Cos'], bins=bins)

    # 🔹 Calcular média do target por célula
    heatmap_data = df_ev.groupby(['bin_sin', 'bin_cos'])['Target_AH_Home'].mean().unstack()

    # 🔹 Criar figura
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn',
        cbar_kws={'label': 'Média de Acerto (Target_AH_Home)'},
        center=0.5,
        ax=ax
    )

    ax.set_title("🧭 Mapa Angular de Valor (sin/cos) – Histórico", fontsize=14, weight='bold')
    ax.set_xlabel("Quadrant_Cos → Dominância (Aggression)")
    ax.set_ylabel("Quadrant_Sin → Eficiência (HandScore)")

    st.pyplot(fig)

    st.info("""
    **Leitura rápida:**
    - 🟢 Regiões verdes → confrontos em que o Home cobre com frequência (valor no favorito).
    - 🔴 Regiões vermelhas → confrontos em que o favorito falha (valor no underdog).
    - Eixo X: diferença de agressividade (cos)
    - Eixo Y: diferença de eficiência (sin)
    """)

except Exception as e:
    st.warning(f"⚠️ Falha ao gerar o mapa angular: {e}")


import plotly.express as px

# Garantir df_plot com colunas necessárias
df_plot = history.copy().dropna(subset=['Quadrant_Sin','Quadrant_Cos','Target_AH_Home'])

# Cor + tamanho (opcional)
df_plot['Color'] = df_plot['Target_AH_Home'].apply(lambda x: 'green' if x >= 0.5 else 'red')
df_plot['Size']  = df_plot['Quadrant_Dist'].clip(0, 40)

# ⚠️ Passe os dados que serão usados no hovertemplate via custom_data (ordem importa!)
custom_cols = ['Home','Away','League','Asian_Line','Target_AH_Home','Quadrant_Cos','Quadrant_Sin','Size']

fig = px.scatter(
    df_plot,
    x='Quadrant_Cos',
    y='Quadrant_Sin',
    color='Color',
    color_discrete_map={'green':'green','red':'red'},
    size='Size',
    custom_data=custom_cols,   # 👈 ESSENCIAL para %{customdata[i]}
    opacity=0.8,
    height=700,
    title='Mapa Angular Interativo – Home (verde) vs Falhas (vermelho)',
    template='plotly_white'    # troque para 'plotly_dark' se preferir
)

# Hover template usando APENAS tags seguras (<br>, <b>) e sem <hr>
fig.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]} vs %{customdata[1]}</b><br>" +
        "🏆 <b>Liga:</b> %{customdata[2]}<br>" +
        "⚙️ <b>Linha Asiática:</b> %{customdata[3]}<br>" +
        "🎯 <b>Target_AH_Home:</b> %{customdata[4]:.2f}<br>" +
        "📊 <b>Quadrant_Cos:</b> %{customdata[5]:.3f}<br>" +
        "📈 <b>Quadrant_Sin:</b> %{customdata[6]:.3f}<br>" +
        "📏 <b>Distância Vetorial:</b> %{customdata[7]:.1f}<extra></extra>"
    ),
    marker=dict(line=dict(width=0.5, color='rgba(0,0,0,0.3)'))  # borda leve
)

fig.update_layout(
    xaxis_title="Quadrant_Cos → Dominância (Aggression)",
    yaxis_title="Quadrant_Sin → Eficiência (HandScore)",
    showlegend=False,
    hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font_size=13, font_color="black")
)

# Eixos de referência
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)

st.plotly_chart(fig, use_container_width=True)



########################################
### 💰 ROI Map Vetorial (sin/cos)
########################################
st.markdown("## 💰 ROI Map Vetorial (sin/cos) – Onde o mercado mais erra")

# Garantir que temos dados válidos
df_roi = history.copy().dropna(subset=['Quadrant_Sin','Quadrant_Cos','Odd_H','Target_AH_Home'])

# 🔹 Calcular ROI por jogo (simples: se acertou, ganha (odd-1); se errou, perde 1)
df_roi['ROI_Game'] = np.where(df_roi['Target_AH_Home'] == 1, df_roi['Odd_H'] - 1, -1)

# 🔹 Discretizar o espaço angular
bins = np.linspace(-1, 1, 21)
df_roi['bin_sin'] = pd.cut(df_roi['Quadrant_Sin'], bins=bins, include_lowest=True)
df_roi['bin_cos'] = pd.cut(df_roi['Quadrant_Cos'], bins=bins, include_lowest=True)

# 🔹 Agrupar por célula vetorial e calcular média de ROI
roi_map = (
    df_roi.groupby(['bin_sin','bin_cos'], observed=False)['ROI_Game']
    .mean()
    .reset_index()
)

pivot = roi_map.pivot(index='bin_sin', columns='bin_cos', values='ROI_Game')

# 🔹 Plotar Heatmap (ROI médio por célula)
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(
    pivot,
    cmap='RdYlGn',
    center=0,
    cbar_kws={'label': 'ROI Médio'},
    annot=False,
    linewidths=0.3
)
ax.set_title("💰 ROI Map Vetorial (sin/cos) – Histórico", fontsize=14, weight='bold')
ax.set_xlabel("Quadrant_Cos → Dominância (Aggression)")
ax.set_ylabel("Quadrant_Sin → Eficiência (HandScore)")

st.pyplot(fig)








st.success("🎯 **Sistema de 16 Quadrantes ML** implementado com sucesso!")
st.info("""
**Resumo das melhorias:**
- 🔢 16 quadrantes para granularidade máxima
- 🎯 Estratégias específicas por categoria  
- 📊 Scoring combinado inteligente
- 🔍 Análise de padrões avançada
- 📈 Visualizações otimizadas
""")

# [CONTINUA... O RESTANTE DO CÓDIGO É SIMILAR COM AS FUNÇÕES ATUALIZADAS]
