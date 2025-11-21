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


# ============================================================
# 🆕 MarketGap Rolling Features (vs Expectativas do Mercado)
# ============================================================

st.markdown("### 🆕 Calculando MarketGap, Rolling e MEI...")

# 1️⃣ Market Gap: desempenho relativo à expectativa do mercado

def calcular_marketgap_avancado(margin, asian_line):
    # Transforma ambas as variáveis para escala logarítmica
    margin_transformada = np.sign(margin) * np.log1p(abs(margin))
    asian_line_transformada = np.sign(asian_line) * np.log1p(abs(asian_line))
    
    return margin_transformada - asian_line_transformada

# Aplicar ao histórico
history['MarketGap_Home'] = history.apply(
    lambda x: calcular_marketgap_avancado(x['Margin'], x['Asian_Line_Decimal']), 
    axis=1
)
history['MarketGap_Away'] = -history['MarketGap_Home']


# 2️⃣ Ponderação pelas odds (valor real da surpresa)
history['WeightedGap_Home'] = history['MarketGap_Home'] * history['Odd_H']
history['WeightedGap_Away'] = history['MarketGap_Away'] * history['Odd_A']

# 3️⃣ Função genérica de rolling com shift(1) para evitar leakage
history = history.sort_values('Date')

def generate_rolling(df, col_input, col_team, newcol, window):
    df[newcol] = (
        df.groupby(col_team)[col_input]
        .rolling(window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    return df

# 4️⃣ Rolling de 6 jogos (já existente) e novo rolling de 3 jogos
history = generate_rolling(history, 'WeightedGap_Home', 'Home', 'WG_Rolling_Home', window=6)
history = generate_rolling(history, 'WeightedGap_Away', 'Away', 'WG_Rolling_Away', window=6)

history = generate_rolling(history, 'WeightedGap_Home', 'Home', 'WG_Rolling_Home_3', window=3)
history = generate_rolling(history, 'WeightedGap_Away', 'Away', 'WG_Rolling_Away_3', window=3)

# Tendência de desempenho vs mercado (últimos 3 x últimos 6)
history['MarketGapTrend_Home'] = history['WG_Rolling_Home_3'] - history['WG_Rolling_Home']
history['MarketGapTrend_Away'] = history['WG_Rolling_Away_3'] - history['WG_Rolling_Away']

# Diferença bruta (continua útil)
history['WG_Rolling_Diff'] = history['WG_Rolling_Home'] - history['WG_Rolling_Away']

# 5️⃣ Tendência do handicap (AsianTrend) – só do lado HOME (Opção A)
history = generate_rolling(history, 'Asian_Line_Decimal', 'Home', 'Asian_Rolling_Home_6', window=6)
history = generate_rolling(history, 'Asian_Line_Decimal', 'Home', 'Asian_Rolling_Home_3', window=3)

history['AsianTrend_Home'] = history['Asian_Rolling_Home_3'] - history['Asian_Rolling_Home_6']

# 6️⃣ MEI – Market Efficiency Index (mercado atrasado x ajustado)
# MEI > 0 → mercado atrasado / valor;  MEI < 0 → mercado já ajustando / superajustado
history['MEI_Home'] = history['MarketGapTrend_Home'] - history['AsianTrend_Home']



# ============================================================
# 🔍 Enriquecendo games_today com rolling histórico + MEI
# ============================================================

games_today = games_today.copy()

# Mapear últimos valores por time no histórico
wg_home_map  = history.groupby('Home')['WG_Rolling_Home'].last()
wg_away_map  = history.groupby('Away')['WG_Rolling_Away'].last()
wg_diff_map  = history.groupby('Home')['WG_Rolling_Diff'].last()
mei_map      = history.groupby('Home')['MEI_Home'].last()
mg_trend_map = history.groupby('Home')['MarketGapTrend_Home'].last()
asian_trend_map = history.groupby('Home')['AsianTrend_Home'].last()

games_today['WG_Rolling_Home'] = games_today['Home'].map(wg_home_map)
games_today['WG_Rolling_Away'] = games_today['Away'].map(wg_away_map)
games_today['WG_Rolling_Diff'] = games_today['Home'].map(wg_diff_map)

# MEI e trends do ponto de vista do time mandante
games_today['MEI_Home'] = games_today['Home'].map(mei_map)
games_today['MarketGapTrend_Home'] = games_today['Home'].map(mg_trend_map)
games_today['AsianTrend_Home'] = games_today['Home'].map(asian_trend_map)

# ➜ Jogos sem histórico: guardar separadamente (sem WG ou sem MEI)
games_missing = games_today[
    games_today[['WG_Rolling_Home', 'WG_Rolling_Away']].isna().any(axis=1)
].copy()

games_today = games_today.dropna(subset=['WG_Rolling_Home', 'WG_Rolling_Away']).copy()

st.success(f"🎯 Jogos com histórico suficiente: {len(games_today)}")
st.warning(f"⚠️ Jogos sem histórico suficiente: {len(games_missing)}")


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


########################################
#### 🧮 BLOCO – Cálculo das Distâncias Home ↔ Away
########################################
def calcular_distancias_quadrantes(df):
    """Calcula distância, separação média e ângulo entre os pontos Home e Away."""
    df = df.copy()
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away']):
        dx = df['Aggression_Home'] - df['Aggression_Away']
        dy = df['HandScore_Home'] - df['HandScore_Away']
        df['Quadrant_Dist'] = np.sqrt(dx**2 + (dy/60)**2 * 2.5) * 10  # escala visual ajustada
        df['Quadrant_Separation'] = 0.5 * (dy + 60 * dx)
        df['Quadrant_Angle_Geometric'] = np.degrees(np.arctan2(dy, dx))
        df['Quadrant_Angle_Normalized'] = np.degrees(np.arctan2((dy / 60), dx))
    else:
        st.warning("⚠️ Colunas Aggression/HandScore não encontradas para calcular as distâncias.")
        df['Quadrant_Dist'] = np.nan
        df['Quadrant_Separation'] = np.nan
        df['Quadrant_Angle_Geometric'] = np.nan
    return df

# Aplicar ao games_today
games_today = calcular_distancias_quadrantes(games_today)


st.dataframe(games_today[['Home','Away','Quadrant_Dist','Quadrant_Separation','Quadrant_Angle_Geometric', 'Quadrant_Angle_Normalized']].head(10))


########################################
#### 🎯 BLOCO – Visualização Interativa com Filtro por Liga e Ângulo
########################################
import plotly.graph_objects as go

st.markdown("## 🎯 Visualização Interativa – Distância entre Times (Home × Away)")

# ==========================
# 🎛️ Filtros interativos
# ==========================
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

# ==========================
# 🎚️ Filtros adicionais
# ==========================
max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância):", 10, min(max_n, 200), 40, step=5)

# 🔹 Novo filtro de ângulo
angle_min, angle_max = st.slider(
    "Filtrar por Ângulo (posição Home vs Away):",
    min_value=-180, max_value=180, value=(-180, 180), step=5,
    help="Ângulos positivos → Home acima | Ângulos negativos → Away acima"
)

# 🔘 Checkbox de modo combinado
use_combined_filter = st.checkbox(
    "Usar filtro combinado (Distância + Ângulo)",
    value=True,
    help="Se desmarcado, exibirá apenas confrontos dentro do intervalo de ângulo, ignorando o filtro de distância."
)

# ==========================
# 📊 Preparar dados
# ==========================
if "Quadrant_Dist" not in df_filtered.columns:
    df_filtered = calcular_distancias_quadrantes(df_filtered)

# Aplicar filtro de ângulo
df_angle = df_filtered[
    (df_filtered['Quadrant_Angle_Normalized'] >= angle_min) &
    (df_filtered['Quadrant_Angle_Normalized'] <= angle_max)
]

# Aplicar lógica conforme modo selecionado
if use_combined_filter:
    # Filtro combinado: aplicar ângulo + top por distância
    df_plot = df_angle.nlargest(n_to_show, "Quadrant_Dist").reset_index(drop=True)
else:
    # Filtro somente por ângulo
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
titulo = f"Confrontos – Aggression × HandScore"
if use_combined_filter:
    titulo += f" | Top {n_to_show} Distâncias"
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
#### 🤖 BLOCO – Treinamento ML Dual (com Quadrant Distance Features)
########################################
from sklearn.ensemble import RandomForestClassifier

def treinar_modelo_quadrantes_dual(history, games_today):
    """
    Treina modelo ML para Home e Away com base nos quadrantes,
    ligas e métricas de distância entre times.
    """

    # -------------------------------
    # 🔹 Garantir cálculo das distâncias
    # -------------------------------
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    # -------------------------------
    # 🔹 Preparar features básicas
    # -------------------------------
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    # 🔹 Novas features contínuas (Distância, Separação e Ângulo)
    extras = history[[
        'Quadrant_Dist',
        'Quadrant_Separation',
        'Quadrant_Angle_Geometric',
        'Quadrant_Angle_Normalized',
        'WG_Rolling_Home',
        'WG_Rolling_Away',
        'WG_Rolling_Diff',
        'MEI_Home'
    ]].fillna(0)

    # Combinar todas as features
    X = pd.concat([ligas_dummies, extras,quadrantes_home, quadrantes_away], axis=1)
    # quadrantes_home, quadrantes_away, 

    # Targets
    y_home = history['Target_AH_Home']
    y_away = 1 - y_home  # inverso lógico

    # -------------------------------
    # 🔹 Treinar modelos
    # -------------------------------
    model_home = RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, class_weight='balanced_subsample', n_jobs=-1
    )
    model_away = RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, class_weight='balanced_subsample', n_jobs=-1
    )

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # # ============================================================
    # # 🔍 Feature Importance – Modelo HOME com novas features
    # # ============================================================
    # try:
    #     importances = pd.Series(model_home.feature_importances_, index=X.columns)
    #     top_feats = importances.sort_values(ascending=False).head(20)
    
    #     st.markdown("### 🔍 Top 20 Features mais importantes (Modelo HOME)")
    #     st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
    
    #     st.bar_chart(top_feats)
    
    # except Exception as e:
    #     st.warning(f"Não foi possível calcular importâncias: {e}")


    # -------------------------------
    # 🔹 Preparar dados para hoje
    # -------------------------------
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    extras_today = games_today[[
        'Quadrant_Dist',
        'Quadrant_Separation',
        'Quadrant_Angle_Geometric',
        'Quadrant_Angle_Normalized',
        'WG_Rolling_Home',
        'WG_Rolling_Away',
        'WG_Rolling_Diff',
        'MEI_Home'
    ]].fillna(0)

    X_today = pd.concat([ligas_today, extras_today,qh_today, qa_today], axis=1)
    # qh_today, qa_today,

    # -------------------------------
    # 🔹 Fazer previsões
    # -------------------------------
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]

    games_today['Quadrante_ML_Score_Home'] = probas_home
    games_today['Quadrante_ML_Score_Away'] = probas_away
    games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

    # -------------------------------
    # 🔹 Mostrar insights de importância
    # -------------------------------
    try:
        importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_feats = importances.head(15)
        st.markdown("### 🔍 Top Features mais importantes (Modelo HOME)")
        st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")

    st.success("✅ Modelo dual (Home/Away) treinado com sucesso com novas features!")
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



    # ---------------- STATUS DE AJUSTE DO MERCADO (MEI)
    def classificar_mei(mei):
        if pd.isna(mei):
            return "⚪ Sem histórico"
        if mei >= 0.30:
            return "🟢 Mercado atrasado (valor forte Home)"
        if mei >= 0.10:
            return "🟡 Mercado atrasado leve"
        if mei <= -0.30:
            return "🔴 Superajustado (cuidado com narrativas recentes)"
        if mei <= -0.10:
            return "🟠 Mercado em ajuste"
        return "⚫ Mercado relativamente eficiente"
    
    ranking_quadrantes['MEI_Status'] = ranking_quadrantes['MEI_Home'].apply(classificar_mei)


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
            lambda r: calculate_handicap_profit(r['Recomendacao'], r['Handicap_Result'], r, r['Asian_Line_Decimal']), axis=1
        )
        return df

    # Aplicar atualização em tempo real
    ranking_quadrantes = update_real_time_data(ranking_quadrantes)
    
    # ---------------- RESUMO LIVE ----------------
    def generate_live_summary(df):
        """Gera resumo em tempo real dos resultados de HANDICAP"""
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
        'League','Time', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'ML_Side', 'Recomendacao', 'Asian_Line_Decimal',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Quadrante_ML_Score_Main', 'Classificacao_Valor_Home', 
        'Classificacao_Valor_Away', 'WG_Rolling_Home', 'WG_Rolling_Away', 'MEI_Home', 'MEI_Status',
        # Colunas Live Score
         'Handicap_Result',
        'Home_Red', 'Away_Red', 'Quadrante_Correct', 'Profit_Quadrante'
    ]
    
    # Filtrar colunas existentes
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
            'WG_Rolling_Home': '{:.3f}',
            'WG_Rolling_Away': '{:.3f}'
        }, na_rep="-"),
        use_container_width=True
    )
    
else:
    st.info("⚠️ Aguardando dados para gerar ranking dual")
    

# ============================================================
# 🧭 BLOCO – Índice de Convergência Total (Confidence_Score)
# ============================================================

def calc_convergencia(row):
    """
    Mede o grau de convergência entre:
    - confiança do modelo
    - separação tática (distância entre times)
    - padrão dos quadrantes
    - eficiência do mercado (MEI_Home)
    """
    try:
        score_home = float(row.get('Quadrante_ML_Score_Home', 0))
        score_away = float(row.get('Quadrante_ML_Score_Away', 0))
        dist = float(row.get('Quadrant_Dist', 0))
        ml_side = "HOME" if score_home > score_away else "AWAY"
        diff = abs(score_home - score_away)
        mei = float(row.get('MEI_Home', 0))
    except Exception:
        return 0.0

    # 1️⃣ Peso da confiança do modelo (diferença H-A)
    w_ml = min(diff * 2, 1.0)  # diferença de 0.5 já é força máxima

    # 2️⃣ Peso da separação tática (distância entre quadrantes)
    w_dist = min(dist / 0.8, 1.0)

    # 3️⃣ Peso da coerência entre padrão e lado do modelo
    home_q = str(row.get('Quadrante_Home_Label', ''))
    away_q = str(row.get('Quadrante_Away_Label', ''))

    padrao_favoravel = (
        ('Underdog Value' in home_q and ml_side == 'HOME') or
        ('Market Overrates' in away_q and ml_side == 'HOME') or
        ('Favorite Reliable' in home_q and ml_side == 'HOME') or
        ('Weak Underdog' in away_q and ml_side == 'AWAY')
    )
    w_pattern = 1.0 if padrao_favoravel else 0.0

    # 4️⃣ Peso do MEI – reescala de -0.5..+0.5 para 0..1 (clipping)
    w_mei = np.clip((mei + 0.5), 0, 1)

    # 5️⃣ Convergência total (ponderada)
    confidence_score = round(
        (0.4 * w_ml + 0.25 * w_dist + 0.20 * w_pattern + 0.15 * w_mei),
        3
    )
    return confidence_score


# Aplicar cálculo
ranking_quadrantes['Confidence_Score'] = ranking_quadrantes.apply(calc_convergencia, axis=1)

# Exibir os 'Gold Matches' – cenários com tudo coerente
st.markdown("### 🥇 Gold Matches – Convergência Máxima")
gold_matches = ranking_quadrantes[ranking_quadrantes['Confidence_Score'] >= 0.75]

if not gold_matches.empty:
    st.dataframe(
        gold_matches[['League', 'Home', 'Away', 'Goals_H_Today','Goals_A_Today', 'Recomendacao', 
                      'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 'Confidence_Score']]
        .sort_values('Confidence_Score', ascending=False)
        .style.format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Quadrante_ML_Score_Home': '{:.1%}',
            'Quadrante_ML_Score_Away': '{:.1%}',
            'Confidence_Score': '{:.2f}'
        })
        .background_gradient(subset=['Confidence_Score'], cmap='YlGn'),
        use_container_width=True
    )
else:
    st.info("Nenhum confronto atingiu nível de convergência 🥇 Gold hoje.")



# ============================================================
# 🧭 Mapa de Valor – MEI × WG_Rolling_Diff
# ============================================================
st.markdown("### 🧭 Mapa de Valor – MEI × Forma vs Mercado")

valor_df = ranking_quadrantes.dropna(subset=['WG_Rolling_Diff', 'MEI_Home']).copy()
if not valor_df.empty:
    import plotly.graph_objects as go

    fig_valor = go.Figure()

    # Cores por lado sugerido
    color_map = valor_df['ML_Side'].map({'HOME': 'royalblue', 'AWAY': 'orangered'}).fillna('gray')

    fig_valor.add_trace(go.Scatter(
        x=valor_df['WG_Rolling_Diff'],
        y=valor_df['MEI_Home'],
        mode='markers',
        marker=dict(
            size=8 + 20*valor_df['Quadrante_ML_Score_Main'].fillna(0),
            opacity=0.8,
            color=color_map
        ),
        text=valor_df.apply(lambda r: f"{r['Home']} vs {r['Away']}<br>"
                                      f"WG_Diff: {r['WG_Rolling_Diff']:.2f}<br>"
                                      f"MEI_Home: {r['MEI_Home']:.2f}<br>"
                                      f"{r['MEI_Status']}<br>"
                                      f"Recomendação: {r['Recomendacao']}", axis=1),
        hoverinfo='text'
    ))

    # Linhas de referência
    fig_valor.add_vline(x=0, line=dict(color="black", width=1, dash="dash"))
    fig_valor.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

    fig_valor.update_layout(
        xaxis_title="WG_Rolling_Diff (Home - Away) – forma recente vs mercado",
        yaxis_title="MEI_Home (Market Efficiency Index)",
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig_valor, use_container_width=True)

else:
    st.info("Sem dados suficientes para exibir o mapa de valor hoje.")





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



# ============================================================
# 📄 Jogos removidos da análise (Sem histórico suficiente)
# ============================================================
if not games_missing.empty:
    st.markdown("### 📄 Jogos sem histórico suficiente (excluídos da ML)")
    st.dataframe(
        games_missing[['League','Home','Away','Asian_Line_Decimal']],
        use_container_width=True
    )



st.markdown("---")
st.info("🎯 **Análise de Quadrantes ML Dual** - Sistema avançado para identificação de value bets em Home e Away baseado em Aggression × HandScore")
