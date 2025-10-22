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


########################################
#### 🧮 BLOCO – Cálculo de Distâncias (Versão Calibrada)
########################################
def calcular_distancias_quadrantes(df):
    """
    Calcula a distância, separação e ângulo entre Home e Away de forma calibrada.
    Inclui padronização por liga e tolerância mínima de desvio padrão para estabilidade.
    """
    df = df.copy()

    required_cols = ["League", "Aggression_Home", "Aggression_Away", "HandScore_Home", "HandScore_Away"]
    if not all(col in df.columns for col in required_cols):
        st.warning("⚠️ Colunas necessárias ausentes para cálculo de distâncias calibradas.")
        df["Quadrant_Dist"] = np.nan
        df["Quadrant_Separation"] = np.nan
        df["Quadrant_Angle"] = np.nan
        return df

    # ==============================
    # 1️⃣ Cálculo da razão global (Método B base)
    # ==============================
    std_agg = df["Aggression_Home"].std()
    std_hs = df["HandScore_Home"].std()
    ratio = std_hs / std_agg if std_agg != 0 else 1.0

    # ==============================
    # 2️⃣ Padronização local por liga com tolerância mínima
    # ==============================
    df["Agg_H_std_Liga"] = df.groupby("League")["Aggression_Home"].transform("std")
    df["HS_H_std_Liga"] = df.groupby("League")["HandScore_Home"].transform("std")

    # Aplicar tolerância mínima (evita divisão por desvios muito baixos)
    df["Agg_H_std_Liga"] = df["Agg_H_std_Liga"].clip(lower=0.05)
    df["HS_H_std_Liga"] = df["HS_H_std_Liga"].clip(lower=1.0)

    # ==============================
    # 3️⃣ Cálculo das diferenças normalizadas
    # ==============================
    dx = (df["Aggression_Home"] - df["Aggression_Away"]) / df["Agg_H_std_Liga"]
    dy = ((df["HandScore_Home"] - df["HandScore_Away"]) / df["HS_H_std_Liga"]) / ratio

    # ==============================
    # 4️⃣ Distância e métricas complementares
    # ==============================
    df["Quadrant_Dist"] = np.sqrt(dx**2 + dy**2) * 10  # escala ajustada p/ visual
    df["Quadrant_Separation"] = 0.5 * (dy + 60 * dx)
    df["Quadrant_Angle"] = np.degrees(np.arctan2(dy, dx))

    return df


# Aplicar ao games_today
games_today = calcular_distancias_quadrantes(games_today)

# ########################################
# #### 🧪 BLOCO – Teste de Escalas (Distância Quadrantes)
# ########################################
# st.markdown("## 🧪 Teste de Escalas – Comparação entre Fórmulas de Distância")

# def testar_variacoes_quadrant_dist(df):
#     """Compara três métodos alternativos de cálculo de distância (A, B, C)."""
#     df = df.copy()

#     # Verificar se há colunas necessárias
#     required_cols = ["League", "Aggression_Home", "Aggression_Away", "HandScore_Home", "HandScore_Away"]
#     if not all(col in df.columns for col in required_cols):
#         st.warning("⚠️ Colunas necessárias ausentes para teste de distância.")
#         return df

#     # ==========================
#     # MÉTODO A – Normalização por liga (Z-score por liga)
#     # ==========================
#     df["Agg_H_std_Liga"] = df.groupby("League")["Aggression_Home"].transform("std").replace(0, 0.001)
#     df["HS_H_std_Liga"] = df.groupby("League")["HandScore_Home"].transform("std").replace(0, 0.001)

#     dx_A = (df["Aggression_Home"] - df["Aggression_Away"]) / df["Agg_H_std_Liga"]
#     dy_A = (df["HandScore_Home"] - df["HandScore_Away"]) / df["HS_H_std_Liga"]
#     df["Quadrant_Dist_A"] = np.sqrt(dx_A**2 + dy_A**2)

#     # ==========================
#     # MÉTODO B – Calibração empírica (razão std HandScore/Aggression)
#     # ==========================
#     std_ratio = df["HandScore_Home"].std() / df["Aggression_Home"].std() if df["Aggression_Home"].std() != 0 else 1
#     dx_B = df["Aggression_Home"] - df["Aggression_Away"]
#     dy_B = (df["HandScore_Home"] - df["HandScore_Away"]) / std_ratio
#     df["Quadrant_Dist_B"] = np.sqrt(dx_B**2 + dy_B**2)

#     # ==========================
#     # MÉTODO C – Padronização global (Z-score simples)
#     # ==========================
#     dx_C = (df["Aggression_Home"] - df["Aggression_Away"]) / df["Aggression_Home"].std()
#     dy_C = (df["HandScore_Home"] - df["HandScore_Away"]) / df["HandScore_Home"].std()
#     df["Quadrant_Dist_C"] = np.sqrt(dx_C**2 + dy_C**2)

#     # ==========================
#     # Consolidação
#     # ==========================
#     cols_show = [
#         "Home", "Away", "League",
#         "Quadrant_Dist", "Quadrant_Dist_A", "Quadrant_Dist_B", "Quadrant_Dist_C"
#     ]
#     df_result = df[cols_show].copy()

#     # ==========================
#     # Estatísticas comparativas
#     # ==========================
#     resumo = pd.DataFrame({
#         "Média": [
#             df["Quadrant_Dist"].mean(),
#             df["Quadrant_Dist_A"].mean(),
#             df["Quadrant_Dist_B"].mean(),
#             df["Quadrant_Dist_C"].mean()
#         ],
#         "Desvio Padrão": [
#             df["Quadrant_Dist"].std(),
#             df["Quadrant_Dist_A"].std(),
#             df["Quadrant_Dist_B"].std(),
#             df["Quadrant_Dist_C"].std()
#         ]
#     }, index=["Original", "Método A (Liga Z)", "Método B (Razão Std)", "Método C (Z Global)"]).round(3)

#     # Correlação entre métodos
#     corr = df[["Quadrant_Dist", "Quadrant_Dist_A", "Quadrant_Dist_B", "Quadrant_Dist_C"]].corr().round(3)

#     st.markdown("### 📊 Estatísticas comparativas")
#     st.dataframe(resumo, use_container_width=True)

#     st.markdown("### 🔗 Correlação entre métodos")
#     st.dataframe(corr, use_container_width=True)

#     st.markdown("### 📋 Amostra dos cálculos de distância (Top 15)")
#     st.dataframe(df_result.head(15), use_container_width=True)

#     return df

# # Aplicar o teste
# games_today_test = testar_variacoes_quadrant_dist(games_today)


# st.dataframe(games_today[['Home','Away','Quadrant_Dist','Quadrant_Separation','Quadrant_Angle']].head(10))


########################################
#### 🎯 BLOCO – Visualização Interativa com Filtro por Liga
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

# Controle de número de confrontos
max_n = len(df_filtered)
n_to_show = st.slider("Quantos confrontos exibir (Top por distância):", 10, min(max_n, 200), 40, step=5)

# ==========================
# 📊 Preparar dados
# ==========================
if "Quadrant_Dist" not in df_filtered.columns:
    df_filtered = calcular_distancias_quadrantes(df_filtered)

df_plot = df_filtered.nlargest(n_to_show, "Quadrant_Dist").reset_index(drop=True)

# ==========================
# 🎨 Criar gráfico Plotly
# ==========================
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
            f"📏 Distância: {row['Quadrant_Dist']:.2f}<br>"
            f"↔️ Separação: {row['Quadrant_Separation']:.1f}<br>"
            f"📐 Ângulo: {row['Quadrant_Angle']:.1f}°"
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

# Linha diagonal de referência
# Linha diagonal de referência - CORRIGIDA
fig.add_trace(go.Scatter(
    x=[0, 0],           # Mantém a mesma largura no eixo X
    y=[-60, 60],            # 🔥 MUDANÇA: Agora corta no Y=0
    mode="lines",
    line=dict(color="limegreen", width=2, dash="dash"),
    name="Linha de equilíbrio"
))

fig.add_trace(go.Scatter(
    x=[-1, 1],           # Mantém a mesma largura no eixo X
    y=[0, 0],            # 🔥 MUDANÇA: Agora corta no Y=0
    mode="lines",
    line=dict(color="limegreen", width=2, dash="dash"),
    name="Linha de equilíbrio"
))

# Layout
titulo = f"Top {n_to_show} Distâncias – Aggression × HandScore"
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
    handicap_result = calc_handicap_result(margin, asian_line_decimal, invert=False)

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
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
        return handicap_result in ["HOME_NOT_COVERED", "PUSH"]

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
        elif (is_home_bet and result == "HOME_COVERED") or (is_away_bet and result == "HOME_NOT_COVERED"):
            return odd  # vitória
        elif (is_home_bet and result == "HOME_NOT_COVERED") or (is_away_bet and result == "HOME_COVERED"):
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
#### 🧮 BLOCO – Contexto de Liga e Treinamento ML Dual (versão aprimorada)
########################################
from sklearn.ensemble import RandomForestClassifier

########################################
#### 🧮 BLOCO – Contexto de Liga com Normalização Z-Score
########################################
def adicionar_contexto_liga(df):
    """
    Adiciona médias e desvios relativos por liga (Aggression / HandScore),
    com normalização Z-score para evitar viés estrutural de mandante.
    """
    df = df.copy()

    required_cols = ["League", "Aggression_Home", "Aggression_Away", "HandScore_Home", "HandScore_Away"]
    if not all(col in df.columns for col in required_cols):
        st.warning("⚠️ Colunas necessárias para contexto de liga ausentes.")
        return df

    # ==============================
    # 📊 1️⃣ Médias por liga
    # ==============================
    league_means = (
        df.groupby("League")[["Aggression_Home", "Aggression_Away", "HandScore_Home", "HandScore_Away"]]
        .mean()
        .rename(columns={
            "Aggression_Home": "League_Agg_HomeMean",
            "Aggression_Away": "League_Agg_AwayMean", 
            "HandScore_Home": "League_HS_HomeMean",
            "HandScore_Away": "League_HS_AwayMean",
        })
    )

    df = df.merge(league_means, on="League", how="left")

    # ==============================
    # 📏 2️⃣ Desvios absolutos (time - média da liga)
    # ==============================
    df["Agg_Home_vs_Liga"] = df["Aggression_Home"] - df["League_Agg_HomeMean"]
    df["Agg_Away_vs_Liga"] = df["Aggression_Away"] - df["League_Agg_AwayMean"]
    df["HS_Home_vs_Liga"] = df["HandScore_Home"] - df["League_HS_HomeMean"]
    df["HS_Away_vs_Liga"] = df["HandScore_Away"] - df["League_HS_AwayMean"]

    # ==============================
    # ⚖️ 3️⃣ Normalização Z-Score (por liga)
    # ==============================
    for col in ["Agg_Home_vs_Liga", "Agg_Away_vs_Liga", "HS_Home_vs_Liga", "HS_Away_vs_Liga"]:
        std_col = df.groupby("League")[col].transform("std")
        # 🔧 CORREÇÃO: Evitar divisão por zero
        std_col = std_col.replace(0, 0.001)  
        df[col] = df[col] / std_col
        df[col] = df[col].fillna(0)

    # # 🔍 DEBUG DETALHADO - Verificar cada passo
    # st.markdown("#### 🔍 DEBUG DETALHADO - Cálculo Z-Score")
    
    # # 1. Verificar médias por liga
    # st.write("**1. Médias por Liga (Aggression_Home):**")
    # medias_agg = df.groupby("League")["Aggression_Home"].mean()
    # st.write(medias_agg.head(10))
    
    # # 2. Verificar desvios padrão
    # st.write("**2. Desvios Padrão por Liga (Aggression_Home):**")
    # desvios_agg = df.groupby("League")["Aggression_Home"].std()
    # st.write(desvios_agg.head(10))
    
    # # 3. Verificar se tem desvio zero
    # st.write("**3. Ligas com desvio padrão ZERO:**")
    # ligas_desvio_zero = desvios_agg[desvios_agg == 0]
    # st.write(f"Ligas com desvio zero: {len(ligas_desvio_zero)}")
    
    # # 4. Verificar valores originais
    # st.write("**4. Valores originais (exemplo):**")
    # st.write(df[["League", "Aggression_Home", "Aggression_Away"]].head(10))
    
    # # 5. Verificar resultados Z-Score
    # st.write("**5. Resultados Z-Score (exemplo):**")
    # st.write(df[["League", "Agg_Home_vs_Liga", "HS_Home_vs_Liga"]].head(10))

    return df


########################################
#### 🤖 BLOCO – Treinamento ML Dual com Contexto de Liga
########################################
def treinar_modelo_quadrantes_dual(history, games_today):
    """
    Treina modelo ML para Home e Away com base nos quadrantes,
    ligas, métricas de distância e contexto médio da liga.
    """

    # ----------------------------------
    # 🔹 Calcular distâncias e contexto de liga
    # ----------------------------------
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)

    history = adicionar_contexto_liga(history)
    games_today = adicionar_contexto_liga(games_today)

    # ----------------------------------
    # 🔹 Preparar features básicas
    # ----------------------------------
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    # 🔹 Features contínuas (distâncias, ângulos e contexto de liga)
    extras = history[[
        "Quadrant_Dist", "Quadrant_Separation", "Quadrant_Angle",
        "Agg_Home_vs_Liga", "HS_Home_vs_Liga",
        "Agg_Away_vs_Liga", "HS_Away_vs_Liga"
    ]].fillna(0)

    # Combinar tudo
    X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies, extras], axis=1)

    # Targets
    y_home = history['Target_AH_Home']
    y_away = 1 - y_home

    # ----------------------------------
    # 🔹 Treinar modelos balanceados
    # ----------------------------------
    model_home = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        n_jobs=-1, class_weight="balanced_subsample"
    )
    model_away = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        n_jobs=-1, class_weight="balanced_subsample"
    )

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # ----------------------------------
    # 🔹 Preparar dados para o dia atual
    # ----------------------------------
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)

    extras_today = games_today[[
        "Quadrant_Dist", "Quadrant_Separation", "Quadrant_Angle",
        "Agg_Home_vs_Liga", "HS_Home_vs_Liga",
        "Agg_Away_vs_Liga", "HS_Away_vs_Liga"
    ]].fillna(0)

    X_today = pd.concat([qh_today, qa_today, ligas_today, extras_today], axis=1)

    # ----------------------------------
    # 🔹 Fazer previsões
    # ----------------------------------
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]

    games_today['Quadrante_ML_Score_Home'] = probas_home
    games_today['Quadrante_ML_Score_Away'] = probas_away
    games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

    # ----------------------------------
    # 🔍 Diagnóstico: distribuição das recomendações
    # ----------------------------------
    # st.markdown("### ⚖️ Distribuição das Recomendações ML (HOME vs AWAY)")
    # dist = games_today['ML_Side'].value_counts(normalize=True).mul(100).round(1)
    # st.write(dist.to_frame("Percentual (%)"))


    # ----------------------------------
    # 🔹 Mostrar importância de features
    # ----------------------------------
    # try:
    #     avg_df = (
    #         games_today.groupby("League")[["Agg_Home_vs_Liga", "HS_Home_vs_Liga"]]
    #         .mean()
    #         .sort_values(by="Agg_Home_vs_Liga", ascending=False)
    #     )
    #     st.markdown("#### 📊 Médias Z-Score (Home vs Liga) por Competição - HOJE")
    #     st.dataframe(avg_df.style.format("{:.2f}"), use_container_width=True)
    # except Exception as e:
    #     st.warning(f"Debug Z-Score não pôde ser exibido: {e}")
    
    # st.success("✅ Modelo dual (Home/Away) treinado com sucesso com contexto de liga!")
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
            lambda r: calculate_handicap_profit(r['Recomendacao'], r['Handicap_Result'], r, r['Asian_Line_Decimal']), axis=1
        )
        return df

    # Aplicar atualização em tempo real
    ranking_quadrantes = update_real_time_data(ranking_quadrantes)

    # ####################

    # # ========================
    # # 🧪 BLOCO DE VALIDAÇÃO DO ML
    # # ========================
    
    # st.markdown("## 🧪 VALIDAÇÃO DO MODELO DE MACHINE LEARNING")
    
    # if not ranking_quadrantes.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
        
    #     # 1. 📊 DISTRIBUIÇÃO DE RECOMENDAÇÕES
    #     st.markdown("### 📊 Distribuição de Recomendações")
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         st.write("**Lado Recomendado (ML_Side):**")
    #         dist_side = ranking_quadrantes['ML_Side'].value_counts()
    #         st.dataframe(dist_side)
            
    #         # Gráfico de pizza CORRIGIDO
    #         if not dist_side.empty:
    #             fig_side = go.Figure(data=[go.Pie(
    #                 labels=dist_side.index, 
    #                 values=dist_side.values,
    #                 hole=.3
    #             )])
    #             fig_side.update_layout(title="Distribuição HOME vs AWAY")
    #             st.plotly_chart(fig_side, use_container_width=True)
        
    #     with col2:
    #         st.write("**Tipos de Recomendação:**")
    #         dist_rec = ranking_quadrantes['Recomendacao'].value_counts().head(10)
    #         st.dataframe(dist_rec)
        
    #     # 2. 📈 ANÁLISE DAS PROBABILIDADES
    #     st.markdown("### 📈 Análise das Probabilidades ML")
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.metric("Probabilidade Média HOME", f"{games_today['Quadrante_ML_Score_Home'].mean():.1%}")
    #         st.metric("Probabilidade Média AWAY", f"{games_today['Quadrante_ML_Score_Away'].mean():.1%}")
        
    #     with col2:
    #         st.metric("Máxima Probabilidade", f"{games_today['Quadrante_ML_Score_Main'].max():.1%}")
    #         st.metric("Mínima Probabilidade", f"{games_today['Quadrante_ML_Score_Main'].min():.1%}")
        
    #     with col3:
    #         # Contar recomendações fortes (>55%)
    #         strong_home = len(games_today[games_today['Quadrante_ML_Score_Home'] > 0.55])
    #         strong_away = len(games_today[games_today['Quadrante_ML_Score_Away'] > 0.55])
    #         st.metric("Recomendações Fortes HOME", strong_home)
    #         st.metric("Recomendações Fortes AWAY", strong_away)
        
    #     # Histograma das probabilidades
    #     fig_probs = go.Figure()
    #     fig_probs.add_trace(go.Histogram(
    #         x=games_today['Quadrante_ML_Score_Home'], 
    #         name='HOME', 
    #         opacity=0.7,
    #         nbinsx=20
    #     ))
    #     fig_probs.add_trace(go.Histogram(
    #         x=games_today['Quadrante_ML_Score_Away'], 
    #         name='AWAY', 
    #         opacity=0.7,
    #         nbinsx=20
    #     ))
    #     fig_probs.update_layout(
    #         title="Distribuição das Probabilidades ML",
    #         xaxis_title="Probabilidade",
    #         yaxis_title="Frequência",
    #         barmode='overlay'
    #     )
    #     st.plotly_chart(fig_probs, use_container_width=True)
        
    #     # 3. 🎯 PERFORMANCE COM DADOS LIVE (SE DISPONÍVEL)
    #     st.markdown("### 🎯 Performance com Dados em Tempo Real")
        
    #     # ✅ VERIFICAÇÃO SEGURA - usar ranking_quadrantes e checar colunas
    #     if 'Handicap_Result' in ranking_quadrantes.columns:
    #         finished_games = ranking_quadrantes.dropna(subset=['Handicap_Result'])
            
    #         if not finished_games.empty:
    #             # Jogos com recomendações do quadrante (verificar se coluna existe)
    #             if 'Quadrante_Correct' in ranking_quadrantes.columns:
    #                 quadrante_bets = finished_games[finished_games['Quadrante_Correct'].notna()]
    #             else:
    #                 quadrante_bets = pd.DataFrame()
    #                 st.info("⚠️ Coluna Quadrante_Correct não disponível")
                
    #             if not quadrante_bets.empty:
    #                 total_bets = len(quadrante_bets)
    #                 correct_bets = quadrante_bets['Quadrante_Correct'].sum()
    #                 winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
                    
    #                 # Calcular profit apenas se coluna existe
    #                 if 'Profit_Quadrante' in quadrante_bets.columns:
    #                     total_profit = quadrante_bets['Profit_Quadrante'].sum()
    #                     roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    #                 else:
    #                     total_profit = 0
    #                     roi = 0
                    
    #                 col1, col2, col3, col4 = st.columns(4)
    #                 col1.metric("Apostas do Quadrante", total_bets)
    #                 col2.metric("Acertos", f"{correct_bets} ({winrate:.1f}%)")
    #                 col3.metric("Profit Total", f"{total_profit:.2f}u")
    #                 col4.metric("ROI", f"{roi:.1f}%")
                    
    #                 # Performance por tipo de recomendação (se coluna existe)
    #                 if 'Recomendacao' in quadrante_bets.columns:
    #                     st.write("**Performance por Tipo de Recomendação:**")
    #                     performance_by_rec = quadrante_bets.groupby('Recomendacao').agg({
    #                         'Quadrante_Correct': ['count', 'sum', 'mean'],
    #                         'Profit_Quadrante': 'sum'
    #                     }).round(3)
                        
    #                     performance_by_rec.columns = ['Total_Apostas', 'Acertos', 'Winrate', 'Profit']
    #                     performance_by_rec['Winrate'] = performance_by_rec['Winrate'] * 100
    #                     st.dataframe(performance_by_rec.sort_values('Profit', ascending=False))
                    
    #             else:
    #                 st.info("⚠️ Nenhuma aposta do quadrante foi feita nos jogos finalizados")
    #         else:
    #             st.info("⏳ Aguardando jogos finalizados para análise de performance")
    #     else:
    #         st.info("⏳ Dados live não disponíveis - aguardando resultados dos jogos")
        
    #     # 4. 🔍 FEATURE IMPORTANCE (SE DISPONÍVEL)
    #     st.markdown("### 🔍 Top Features Mais Importantes")
        
    #     try:
    #         if 'modelo_home' in locals() and hasattr(modelo_home, 'feature_importances_'):
    #             # Recriar os nomes das features como foram usadas no treinamento
    #             feature_names = []
                
    #             # Quadrantes Home (QH_1 a QH_8)
    #             feature_names += [f'QH_{i}' for i in range(1, 9)]
    #             # Quadrantes Away (QA_1 a QA_8)  
    #             feature_names += [f'QA_{i}' for i in range(1, 9)]
    #             # Ligas (exemplo: League_PremierLeague, etc.)
    #             if 'League' in ranking_quadrantes.columns:
    #                 top_leagues = ranking_quadrantes['League'].value_counts().head(10).index
    #                 feature_names += [f'League_{league}' for league in top_leagues]
    #             # Features contínuas
    #             feature_names += [
    #                 'Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle',
    #                 'Agg_Home_vs_Liga', 'HS_Home_vs_Liga', 'Agg_Away_vs_Liga', 'HS_Away_vs_Liga'
    #             ]
                
    #             # Ajustar para o número real de features
    #             importances = modelo_home.feature_importances_
    #             if len(feature_names) > len(importances):
    #                 feature_names = feature_names[:len(importances)]
    #             elif len(feature_names) < len(importances):
    #                 feature_names += [f'Extra_Feature_{i}' for i in range(len(feature_names), len(importances))]
                
    #             feature_importance = pd.DataFrame({
    #                 'feature': feature_names,
    #                 'importance': importances
    #             }).sort_values('importance', ascending=False).head(15)
                
    #             fig_features = px.bar(feature_importance, x='importance', y='feature', 
    #                                  title='Top 15 Features Mais Importantes (Modelo HOME)')
    #             st.plotly_chart(fig_features, use_container_width=True)
                
    #             st.write("**Top 10 Features:**")
    #             st.dataframe(feature_importance.head(10))
    #         else:
    #             st.warning("Modelo HOME não disponível para feature importance")
            
    #     except Exception as e:
    #         st.warning(f"⚠️ Não foi possível obter feature importance: {e}")
        
    #     # 5. 🎪 ANÁLISE DE PADRÕES
    #     st.markdown("### 🎪 Análise de Padrões por Quadrante")
        
    #     # Success rate por quadrante
    #     if 'Quadrante_Home_Label' in games_today.columns and 'Quadrante_Correct' in games_today.columns:
    #         quadrante_performance = games_today.groupby('Quadrante_Home_Label').agg({
    #             'Quadrante_Correct': ['count', 'mean'],
    #             'Profit_Quadrante': 'sum'
    #         }).round(3)
            
    #         if not quadrante_performance.empty:
    #             quadrante_performance.columns = ['Total_Apostas', 'Winrate', 'Profit']
    #             quadrante_performance['Winrate'] = quadrante_performance['Winrate'] * 100
    #             st.dataframe(quadrante_performance.sort_values('Profit', ascending=False))
    
    # else:
    #     st.warning("⚠️ Dados insuficientes para validação do ML")
    
    # st.markdown("---")


    # #####################


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
        'League','Time', 'Home', 'Away', 'ML_Side',
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
            'Asian_Line_Decimal': '{:.2f}',
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
