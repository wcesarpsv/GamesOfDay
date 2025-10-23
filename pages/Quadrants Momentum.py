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

st.set_page_config(page_title="Análise de Momentum - Bet Indicator", layout="wide")
st.title("🎯 Análise de 16 Quadrantes - MOMENTUM (z-score por Liga)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "MomentumML"
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
st.info("📂 Carregando dados para análise de MOMENTUM (z-score)...")

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
try:
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
except Exception as e:
    st.error(f"Erro na conversão Asian Line: {e}")

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

# ---------------- SISTEMA DE 16 QUADRANTES COM MOMENTUM ----------------
st.markdown("## 🎯 Sistema de 16 Quadrantes - MOMENTUM (z-score)")

# DEFINIR QUADRANTES PARA MOMENTUM (range -3.5 a 3.5)
QUADRANTES_MOMENTUM = {
    # 🔵 QUADRANTE 1-4: FORTE FAVORITO (+0.75 a +1.0) + MOMENTUM ALTO
    1: {"nome": "Fav Forte Momentum Muito Alto", "agg_min": 0.75, "agg_max": 1.0, "mom_min": 2.0, "mom_max": 3.5},
    2: {"nome": "Fav Forte Momentum Alto", "agg_min": 0.75, "agg_max": 1.0, "mom_min": 1.0, "mom_max": 2.0},
    3: {"nome": "Fav Forte Momentum Moderado", "agg_min": 0.75, "agg_max": 1.0, "mom_min": 0.0, "mom_max": 1.0},
    4: {"nome": "Fav Forte Momentum Neutro", "agg_min": 0.75, "agg_max": 1.0, "mom_min": -1.0, "mom_max": 0.0},
    
    # 🟢 QUADRANTE 5-8: FAVORITO MODERADO (+0.25 a +0.75)
    5: {"nome": "Fav Moderado Momentum Muito Alto", "agg_min": 0.25, "agg_max": 0.75, "mom_min": 2.0, "mom_max": 3.5},
    6: {"nome": "Fav Moderado Momentum Alto", "agg_min": 0.25, "agg_max": 0.75, "mom_min": 1.0, "mom_max": 2.0},
    7: {"nome": "Fav Moderado Momentum Moderado", "agg_min": 0.25, "agg_max": 0.75, "mom_min": 0.0, "mom_max": 1.0},
    8: {"nome": "Fav Moderado Momentum Neutro", "agg_min": 0.25, "agg_max": 0.75, "mom_min": -1.0, "mom_max": 0.0},
    
    # 🟡 QUADRANTE 9-12: UNDERDOG MODERADO (-0.75 a -0.25)
    9: {"nome": "Under Moderado Momentum Neutro", "agg_min": -0.75, "agg_max": -0.25, "mom_min": -1.0, "mom_max": 0.0},
    10: {"nome": "Under Moderado Momentum Moderado", "agg_min": -0.75, "agg_max": -0.25, "mom_min": -2.0, "mom_max": -1.0},
    11: {"nome": "Under Moderado Momentum Alto", "agg_min": -0.75, "agg_max": -0.25, "mom_min": -3.5, "mom_max": -2.0},
    12: {"nome": "Under Moderado Momentum Muito Alto", "agg_min": -0.75, "agg_max": -0.25, "mom_min": -3.5, "mom_max": -2.0},
    
    # 🔴 QUADRANTE 13-16: FORTE UNDERDOG (-1.0 a -0.75)
    13: {"nome": "Under Forte Momentum Neutro", "agg_min": -1.0, "agg_max": -0.75, "mom_min": -1.0, "mom_max": 0.0},
    14: {"nome": "Under Forte Momentum Moderado", "agg_min": -1.0, "agg_max": -0.75, "mom_min": -2.0, "mom_max": -1.0},
    15: {"nome": "Under Forte Momentum Alto", "agg_min": -1.0, "agg_max": -0.75, "mom_min": -3.5, "mom_max": -2.0},
    16: {"nome": "Under Forte Momentum Muito Alto", "agg_min": -1.0, "agg_max": -0.75, "mom_min": -3.5, "mom_max": -2.0}
}

def classificar_quadrante_momentum(agg, momentum):
    """Classifica Aggression e Momentum em um dos 16 quadrantes"""
    if pd.isna(agg) or pd.isna(momentum):
        return 0  # Neutro/Indefinido
    
    for quadrante_id, config in QUADRANTES_MOMENTUM.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        mom_ok = (config['mom_min'] <= momentum <= config['mom_max'])
            
        if agg_ok and mom_ok:
            return quadrante_id
    
    return 0  # Caso não se enquadre em nenhum quadrante

# Aplicar classificação MOMENTUM aos dados
games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante_momentum(x.get('Aggression_Home'), x.get('M_H')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante_momentum(x.get('Aggression_Away'), x.get('M_A')), axis=1
)

history['Quadrante_Home'] = history.apply(
    lambda x: classificar_quadrante_momentum(x.get('Aggression_Home'), x.get('M_H')), axis=1
)
history['Quadrante_Away'] = history.apply(
    lambda x: classificar_quadrante_momentum(x.get('Aggression_Away'), x.get('M_A')), axis=1
)

# ---------------- CÁLCULO DE DISTÂNCIAS COM MOMENTUM ----------------
def calcular_distancias_momentum(df):
    """Calcula distância, separação média e ângulo entre os pontos Home e Away usando Momentum."""
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A']
    if all(col in df.columns for col in required_cols):
        dx = df['Aggression_Home'] - df['Aggression_Away']
        dy = df['M_H'] - df['M_A']  # Diferença de Momentum
        df['Quadrant_Dist'] = np.sqrt(dx**2 + (dy/3.5)**2 * 2.5) * 10  # escala ajustada para momentum
        df['Quadrant_Separation'] = 0.5 * (dy + 3.5 * dx)  # ajuste de escala
        df['Quadrant_Angle'] = np.degrees(np.arctan2(dy, dx))
    else:
        st.warning("⚠️ Colunas Aggression/Momentum não encontradas para calcular as distâncias.")
        df['Quadrant_Dist'] = np.nan
        df['Quadrant_Separation'] = np.nan
        df['Quadrant_Angle'] = np.nan
    return df

# Aplicar ao games_today
games_today = calcular_distancias_momentum(games_today)

# ---------------- VALIDAÇÃO DE DADOS MOMENTUM ----------------
def validar_dados_momentum(df):
    """Valida se os dados de Momentum estão presentes"""
    colunas_necessarias = ['Aggression_Home', 'M_H', 'Aggression_Away', 'M_A']
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
    
    if colunas_faltantes:
        st.warning(f"⚠️ Colunas Momentum faltantes: {colunas_faltantes}")
        return False
    
    # Verificar range do Momentum
    if 'M_H' in df.columns:
        mom_min = df['M_H'].min()
        mom_max = df['M_H'].max()
        st.info(f"📊 Range do Momentum: {mom_min:.2f} a {mom_max:.2f}")
    
    return True

# Validar dados antes de prosseguir
if not validar_dados_momentum(games_today):
    st.error("❌ Dados de Momentum insuficientes para análise.")
    st.stop()

# ---------------- VISUALIZAÇÃO DOS 16 QUADRANTES MOMENTUM ----------------
def plot_quadrantes_momentum(df, side="Home"):
    """Plot dos 16 quadrantes com Momentum (z-score)"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Definir cores por categoria
    cores_categorias = {
        'Fav Forte': 'red',
        'Fav Moderado': 'lightcoral', 
        'Under Moderado': 'lightyellow',
        'Under Forte': 'yellow'
    }
    
    # Plotar cada ponto com cor da categoria
    for quadrante_id in range(1, 17):
        mask = df[f'Quadrante_{side}'] == quadrante_id
        if mask.any():
            categoria = QUADRANTES_MOMENTUM[quadrante_id]['nome'].split()[0] + ' ' + QUADRANTES_MOMENTUM[quadrante_id]['nome'].split()[1]
            cor = cores_categorias.get(categoria, 'gray')
            
            x = df.loc[mask, f'Aggression_{side}']
            y = df.loc[mask, f'M_{side[0]}']  # M_H ou M_A
            ax.scatter(x, y, c=cor, 
                      label=QUADRANTES_MOMENTUM[quadrante_id]['nome'],
                      alpha=0.7, s=50)
    
    # Linhas divisórias dos quadrantes (Aggression)
    for x in [-0.75, -0.25, 0.25, 0.75]:
        ax.axvline(x=x, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Linhas divisórias dos quadrantes (Momentum)  
    for y in [-2.0, -1.0, 0, 1.0, 2.0]:
        ax.axhline(y=y, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Anotações dos quadrantes
    annot_config = [
        (0.875, 2.75, "Fav Forte\nMomentum Muito Alto", 7), (0.875, 1.5, "Fav Forte\nMomentum Alto", 7),
        (0.875, 0.5, "Fav Forte\nMomentum Moderado", 7), (0.875, -0.5, "Fav Forte\nMomentum Neutro", 7),
        (0.5, 2.75, "Fav Moderado\nMomentum Muito Alto", 7), (0.5, 1.5, "Fav Moderado\nMomentum Alto", 7),
        (0.5, 0.5, "Fav Moderado\nMomentum Moderado", 7), (0.5, -0.5, "Fav Moderado\nMomentum Neutro", 7),
        (-0.5, -0.5, "Under Moderado\nMomentum Neutro", 7), (-0.5, -1.5, "Under Moderado\nMomentum Moderado", 7),
        (-0.5, -2.75, "Under Moderado\nMomentum Alto", 7), (-0.5, -2.75, "Under Moderado\nMomentum Muito Alto", 7),
        (-0.875, -0.5, "Under Forte\nMomentum Neutro", 7), (-0.875, -1.5, "Under Forte\nMomentum Moderado", 7),
        (-0.875, -2.75, "Under Forte\nMomentum Alto", 7), (-0.875, -2.75, "Under Forte\nMomentum Muito Alto", 7)
    ]
    
    for x, y, text, fontsize in annot_config:
        ax.text(x, y, text, ha='center', fontsize=fontsize, weight='bold')
    
    ax.set_xlabel(f'Aggression_{side} (-1 zebra ↔ +1 favorito)')
    ax.set_ylabel(f'Momentum_{side} (z-score: -3.5 a +3.5)')
    ax.set_title(f'16 Quadrantes MOMENTUM - {side}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Exibir gráficos MOMENTUM
st.markdown("### 📈 Visualização dos 16 Quadrantes - MOMENTUM")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_quadrantes_momentum(games_today, "Home"))
with col2:
    st.pyplot(plot_quadrantes_momentum(games_today, "Away"))

# ---------------- VISUALIZAÇÃO INTERATIVA MOMENTUM ----------------
st.markdown("## 🎯 Visualização Interativa – Distância entre Times (Momentum)")

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
    yh, ya = row["M_H"], row["M_A"]

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
            f"🎯 Home: {QUADRANTES_MOMENTUM.get(row['Quadrante_Home'], {}).get('nome', 'N/A')}<br>"
            f"🎯 Away: {QUADRANTES_MOMENTUM.get(row['Quadrante_Away'], {}).get('nome', 'N/A')}<br>"
            f"📏 Distância: {row['Quadrant_Dist']:.2f}<br>"
            f"📊 M_H: {row['M_H']:.2f} | M_A: {row['M_A']:.2f}"
        ),
        showlegend=False
    ))

# Pontos Home e Away
fig.add_trace(go.Scatter(
    x=df_plot["Aggression_Home"],
    y=df_plot["M_H"],
    mode="markers+text",
    name="Home",
    marker=dict(color="royalblue", size=8, opacity=0.8),
    text=df_plot["Home"],
    textposition="top center",
    hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=df_plot["Aggression_Away"],
    y=df_plot["M_A"],
    mode="markers+text",
    name="Away",
    marker=dict(color="orangered", size=8, opacity=0.8),
    text=df_plot["Away"],
    textposition="top center",
    hoverinfo="skip"
))

# Layout
titulo = f"Top {n_to_show} Distâncias – 16 Quadrantes MOMENTUM"
if selected_league != "⚽ Todas as ligas":
    titulo += f" | {selected_league}"

fig.update_layout(
    title=titulo,
    xaxis_title="Aggression (-1 zebra ↔ +1 favorito)",
    yaxis_title="Momentum (z-score: -3.5 ↔ +3.5)",
    template="plotly_white",
    height=700,
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- MODELO ML COM MOMENTUM ----------------
def treinar_modelo_momentum_dual(history, games_today):
    """
    Treina modelo ML para Home e Away com base nos 16 quadrantes MOMENTUM
    """
    # Garantir cálculo das distâncias
    history = calcular_distancias_momentum(history)
    games_today = calcular_distancias_momentum(games_today)

    # Preparar features básicas
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')

    # Features contínuas
    extras = history[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle']].fillna(0)

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
    extras_today = games_today[['Quadrant_Dist', 'Quadrant_Separation', 'Quadrant_Angle']].fillna(0)

    X_today = pd.concat([qh_today, qa_today, ligas_today, extras_today], axis=1)

    # Fazer previsões
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]

    games_today['Momentum_ML_Score_Home'] = probas_home
    games_today['Momentum_ML_Score_Away'] = probas_away
    games_today['Momentum_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side_Momentum'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

    # Mostrar importância das features
    try:
        importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_feats = importances.head(15)
        st.markdown("### 🔍 Top Features mais importantes (Modelo MOMENTUM - HOME)")
        st.dataframe(top_feats.to_frame("Importância"), use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível calcular importâncias: {e}")

    st.success("✅ Modelo dual MOMENTUM com 16 quadrantes treinado com sucesso!")
    return model_home, model_away, games_today

# ---------------- SISTEMA DE INDICAÇÕES PARA MOMENTUM ----------------
def adicionar_indicadores_momentum_dual(df):
    """Adiciona classificações e recomendações explícitas para momentum"""
    df = df.copy()
    
    # Mapear quadrantes para labels
    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(lambda x: QUADRANTES_MOMENTUM.get(x, {}).get('nome', 'Neutro'))
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(lambda x: QUADRANTES_MOMENTUM.get(x, {}).get('nome', 'Neutro'))
    
    # 1. CLASSIFICAÇÃO DE VALOR PARA HOME
    conditions_home = [
        df['Momentum_ML_Score_Home'] >= 0.65,
        df['Momentum_ML_Score_Home'] >= 0.58,
        df['Momentum_ML_Score_Home'] >= 0.52,
        df['Momentum_ML_Score_Home'] >= 0.48,
        df['Momentum_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home_Momentum'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')
    
    # 2. CLASSIFICAÇÃO DE VALOR PARA AWAY
    conditions_away = [
        df['Momentum_ML_Score_Away'] >= 0.65,
        df['Momentum_ML_Score_Away'] >= 0.58,
        df['Momentum_ML_Score_Away'] >= 0.52,
        df['Momentum_ML_Score_Away'] >= 0.48,
        df['Momentum_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away_Momentum'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')
    
    # 3. RECOMENDAÇÃO DE APOSTA DUAL PARA MOMENTUM
    def gerar_recomendacao_momentum_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Momentum_ML_Score_Home']
        score_away = row['Momentum_ML_Score_Away']
        ml_side = row['ML_Side_Momentum']
        
        # Padrões específicos para momentum
        if 'Momentum Muito Alto' in home_q and 'Momentum Muito Alto' not in away_q:
            return f'🚀 HOME COM MOMENTUM EXPLOSIVO ({score_home:.1%})'
        elif 'Momentum Muito Alto' in away_q and 'Momentum Muito Alto' not in home_q:
            return f'🚀 AWAY COM MOMENTUM EXPLOSIVO ({score_away:.1%})'
        elif 'Fav Forte' in home_q and 'Momentum Alto' in home_q and 'Under Forte' in away_q:
            return f'💪 FAVORITO HOME COM MOMENTUM ({score_home:.1%})'
        elif 'Under Forte' in home_q and 'Fav Forte' in away_q and 'Momentum Alto' in away_q:
            return f'💪 FAVORITO AWAY COM MOMENTUM ({score_away:.1%})'
        elif ml_side == 'HOME' and score_home >= 0.60:
            return f'📈 MOMENTUM CONFIA HOME ({score_home:.1%})'
        elif ml_side == 'AWAY' and score_away >= 0.60:
            return f'📈 MOMENTUM CONFIA AWAY ({score_away:.1%})'
        elif 'Momentum Neutro' in home_q and score_away >= 0.58:
            return f'🔄 AWAY EM MOMENTUM NEUTRO ({score_away:.1%})'
        elif 'Momentum Neutro' in away_q and score_home >= 0.58:
            return f'🔄 HOME EM MOMENTUM NEUTRO ({score_home:.1%})'
        else:
            return f'⚖️ ANALISAR MOMENTUM (H:{score_home:.1%} A:{score_away:.1%})'
    
    df['Recomendacao_Momentum'] = df.apply(gerar_recomendacao_momentum_dual, axis=1)
    
    # 4. RANKING POR MELHOR PROBABILIDADE
    df['Ranking_Momentum'] = df['Momentum_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)
    
    return df

# ---------------- EXECUÇÃO PRINCIPAL MOMENTUM ----------------
# Executar treinamento MOMENTUM
if not history.empty:
    modelo_home_momentum, modelo_away_momentum, games_today = treinar_modelo_momentum_dual(history, games_today)
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo MOMENTUM")

# ---------------- ANÁLISE DE PADRÕES PARA MOMENTUM ----------------
def analisar_padroes_momentum_dual(df):
    """Analisa padrões recorrentes nas combinações de momentum"""
    st.markdown("### 🔍 Análise de Padrões MOMENTUM")
    
    # Padrões prioritários para momentum
    padroes_momentum = {
        'Fav Forte Momentum Muito Alto vs Under Forte Momentum Muito Alto': {
            'descricao': '🎯 **MELHOR PADRÃO MOMENTUM** - Favorito explosivo vs underdog em crise',
            'lado_recomendado': 'HOME',
            'prioridade': 1,
            'score_min': 0.68
        },
        'Under Forte Momentum Muito Alto vs Fav Forte Momentum Muito Alto': {
            'descricao': '🎯 **MOMENTUM CONTRÁRIO** - Underdog em crise vs favorito explosivo', 
            'lado_recomendado': 'AWAY',
            'prioridade': 1,
            'score_min': 0.68
        },
        'Fav Moderado Momentum Alto vs Under Moderado Momentum Alto': {
            'descricao': '💪 **PADRÃO VALUE MOMENTUM** - Favorito com bom momento vs underdog fraco',
            'lado_recomendado': 'HOME',
            'prioridade': 2,
            'score_min': 0.60
        }
    }
    
    for padrao, info in padroes_momentum.items():
        home_q, away_q = padrao.split(' vs ')
        
        # Buscar jogos que correspondem ao padrão
        jogos = df[
            (df['Quadrante_Home_Label'] == home_q) & 
            (df['Quadrante_Away_Label'] == away_q)
        ]
        
        # Filtrar por score mínimo se especificado
        if info['lado_recomendado'] == 'HOME':
            score_col = 'Momentum_ML_Score_Home'
        else:
            score_col = 'Momentum_ML_Score_Away'
            
        if 'score_min' in info:
            jogos = jogos[jogos[score_col] >= info['score_min']]
        
        if not jogos.empty:
            st.write(f"**{padrao}**")
            st.write(f"{info['descricao']}")
            st.write(f"📈 **Score mínimo**: {info.get('score_min', 0.50):.1%}")
            st.write(f"🎯 **Jogos encontrados**: {len(jogos)}")
            
            # Colunas para exibir
            cols_padrao = ['Ranking_Momentum', 'Home', 'Away', 'League', score_col, 'Recomendacao_Momentum', 'Quadrant_Dist']
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

# ---------------- ESTRATÉGIAS AVANÇADAS PARA MOMENTUM ----------------
def gerar_estrategias_momentum(df):
    """Gera estratégias específicas baseadas no momentum"""
    st.markdown("### 🎯 Estratégias MOMENTUM - z-score por Liga")
    
    estrategias = {
        'Momentum Explosivo': {
            'descricao': '**Times com Momentum Muito Alto (z-score > 2.0)**',
            'quadrantes': [1, 5],
            'estrategia': 'Apostar fortemente, especialmente como favoritos',
            'confianca': 'Muito Alta',
            'observacao': 'Raros e muito valiosos - indicam desempenho excepcional'
        },
        'Momentum Positivo': {
            'descricao': '**Times com Momentum Alto (z-score 1.0-2.0)**', 
            'quadrantes': [2, 6],
            'estrategia': 'Buscar value, confiar no momento positivo',
            'confianca': 'Alta',
            'observacao': 'Bom equilíbrio entre frequência e confiabilidade'
        },
        'Momentum de Crise': {
            'descricao': '**Times com Momentum Muito Baixo (z-score < -2.0)**',
            'quadrantes': [12, 16],
            'estrategia': 'Apostar contra, especialmente como underdogs',
            'confianca': 'Alta',
            'observacao': 'Times em crise tendem a piorar o desempenho'
        }
    }
    
    for categoria, info in estrategias.items():
        st.write(f"**{categoria}**")
        st.write(f"📋 {info['descricao']}")
        st.write(f"🎯 Estratégia: {info['estrategia']}")
        st.write(f"📊 Confiança: {info['confianca']}")
        st.write(f"💡 Observação: {info['observacao']}")
        
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
                avg_score = jogos_categoria['Momentum_ML_Score_Main'].mean()
                st.metric("Score Médio", f"{avg_score:.1%}")
            with col3:
                high_value = len(jogos_categoria[jogos_categoria['Momentum_ML_Score_Main'] >= 0.65])
                st.metric("Alto Valor", high_value)
        
        st.write("---")

# ---------------- SISTEMA DE SCORING PARA MOMENTUM ----------------
def calcular_pontuacao_momentum_quadrante(quadrante_id):
    """Calcula pontuação base para cada quadrante momentum (0-100)"""
    scores_base = {
        # Fav Forte com Momentum Alto: máxima pontuação
        1: 95, 2: 85, 3: 75, 4: 65,
        # Fav Moderado com Momentum Alto
        5: 90, 6: 80, 7: 70, 8: 60,
        # Under Moderado com Momentum Baixo  
        9: 55, 10: 45, 11: 35, 12: 25,
        # Under Forte com Momentum Baixo
        13: 50, 14: 40, 15: 30, 16: 20
    }
    return scores_base.get(quadrante_id, 50)

def gerar_score_combinado_momentum(df):
    """Gera score combinado considerando momentum"""
    df = df.copy()
    
    # Score base dos quadrantes momentum
    df['Score_Base_Home_Momentum'] = df['Quadrante_Home'].apply(calcular_pontuacao_momentum_quadrante)
    df['Score_Base_Away_Momentum'] = df['Quadrante_Away'].apply(calcular_pontuacao_momentum_quadrante)
    
    # Score combinado (média ponderada)
    df['Score_Combinado_Momentum'] = (df['Score_Base_Home_Momentum'] * 0.6 + df['Score_Base_Away_Momentum'] * 0.4)
    
    # Ajustar pelo ML Score Momentum
    df['Score_Final_Momentum'] = df['Score_Combinado_Momentum'] * df['Momentum_ML_Score_Main']
    
    # Classificar por potencial momentum
    conditions = [
        df['Score_Final_Momentum'] >= 70,
        df['Score_Final_Momentum'] >= 50, 
        df['Score_Final_Momentum'] >= 30,
        df['Score_Final_Momentum'] < 30
    ]
    choices = ['🚀 MOMENTUM EXPLOSIVO', '📈 MOMENTUM POSITIVO', '⚖️ MOMENTUM NEUTRO', '🔻 MOMENTUM NEGATIVO']
    df['Classificacao_Momentum'] = np.select(conditions, choices, default='⚖️ MOMENTUM NEUTRO')
    
    return df

# ---------------- EXIBIÇÃO DOS RESULTADOS MOMENTUM ----------------
st.markdown("## 🏆 Melhores Confrontos por MOMENTUM ML")

if not games_today.empty and 'Momentum_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_momentum = games_today.copy()
    
    # Aplicar indicadores explicativos para momentum
    ranking_momentum = adicionar_indicadores_momentum_dual(ranking_momentum)
    
    # Aplicar scoring combinado momentum
    ranking_momentum = gerar_score_combinado_momentum(ranking_momentum)
    
    # Ordenar por score final momentum
    ranking_momentum = ranking_momentum.sort_values('Score_Final_Momentum', ascending=False)
    
    # Colunas para exibição momentum
    colunas_momentum = [
        'League', 'Time', 'Home', 'Away', 'Goals_H_Today','Goals_A_Today','ML_Side_Momentum',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Momentum_ML_Score_Home', 'Momentum_ML_Score_Away', 
        'Score_Final_Momentum', 'Classificacao_Momentum',
        'Classificacao_Valor_Home_Momentum', 'Classificacao_Valor_Away_Momentum', 'Recomendacao_Momentum',
        'M_H', 'M_A', 'Quadrant_Dist'
    ]
    
    # Filtrar colunas existentes
    cols_finais_momentum = [c for c in colunas_momentum if c in ranking_momentum.columns]
    
    # Função de estilo para momentum
    def estilo_tabela_momentum(df):
        def cor_classificacao_momentum(valor):
            if '🚀 MOMENTUM EXPLOSIVO' in str(valor): return 'font-weight: bold'
            elif '📈 MOMENTUM POSITIVO' in str(valor): return 'font-weight: bold'
            elif '🔻 MOMENTUM NEGATIVO' in str(valor): return 'font-weight: bold'
            elif '🏆 ALTO VALOR' in str(valor): return 'font-weight: bold'
            elif '🔴 ALTO RISCO' in str(valor): return 'font-weight: bold'
            elif 'EXPLOSIVO' in str(valor): return 'background-color: #FFD700'
            else: return ''
        
        colunas_para_estilo = []
        for col in ['Classificacao_Momentum', 'Classificacao_Valor_Home_Momentum', 
                   'Classificacao_Valor_Away_Momentum', 'Recomendacao_Momentum']:
            if col in df.columns:
                colunas_para_estilo.append(col)
        
        styler = df.style
        if colunas_para_estilo:
            styler = styler.applymap(cor_classificacao_momentum, subset=colunas_para_estilo)
        
        # Aplicar gradientes
        if 'Momentum_ML_Score_Home' in df.columns:
            styler = styler.background_gradient(subset=['Momentum_ML_Score_Home'], cmap='RdYlGn')
        if 'Momentum_ML_Score_Away' in df.columns:
            styler = styler.background_gradient(subset=['Momentum_ML_Score_Away'], cmap='RdYlGn')
        if 'Score_Final_Momentum' in df.columns:
            styler = styler.background_gradient(subset=['Score_Final_Momentum'], cmap='RdYlGn')
        
        return styler

    st.dataframe(
        estilo_tabela_momentum(ranking_momentum[cols_finais_momentum].head(25))
        .format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'M_H': '{:.2f}',
            'M_A': '{:.2f}',
            'Quadrant_Dist': '{:.2f}',
            'Momentum_ML_Score_Home': '{:.1%}',
            'Momentum_ML_Score_Away': '{:.1%}',
            'Score_Final_Momentum': '{:.1f}'
        }, na_rep="-"),
        use_container_width=True
    )
    
    # ---------------- ANÁLISES ESPECÍFICAS MOMENTUM ----------------
    analisar_padroes_momentum_dual(ranking_momentum)
    gerar_estrategias_momentum(ranking_momentum)
    
else:
    st.info("⚠️ Aguardando dados para gerar ranking de MOMENTUM")

# ---------------- COMPARAÇÃO ENTRE ABORDAGENS ----------------
st.markdown("## 🔄 Comparação: HandScore vs Momentum")

if not games_today.empty and all(col in games_today.columns for col in ['Quadrante_ML_Score_Home', 'Momentum_ML_Score_Home']):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_handscore = games_today['Quadrante_ML_Score_Home'].mean()
        st.metric("📊 Média HandScore", f"{avg_handscore:.1%}")
    
    with col2:
        avg_momentum = games_today['Momentum_ML_Score_Home'].mean()
        st.metric("🚀 Média Momentum", f"{avg_momentum:.1%}")
    
    with col3:
        diff = avg_momentum - avg_handscore
        st.metric("📈 Diferença", f"{diff:+.1%}")

# ---------------- RESUMO EXECUTIVO MOMENTUM ----------------
def resumo_momentum_hoje(df):
    """Resumo executivo do momentum de hoje"""
    
    st.markdown("### 📋 Resumo Executivo - MOMENTUM Hoje")
    
    if df.empty:
        st.info("Nenhum dado disponível para resumo")
        return
    
    total_jogos = len(df)
    
    # Estatísticas de classificação momentum
    momentum_explosivo = len(df[df['Classificacao_Momentum'] == '🚀 MOMENTUM EXPLOSIVO'])
    momentum_positivo = len(df[df['Classificacao_Momentum'] == '📈 MOMENTUM POSITIVO'])
    
    alto_valor_home_momentum = len(df[df['Classificacao_Valor_Home_Momentum'] == '🏆 ALTO VALOR'])
    alto_valor_away_momentum = len(df[df['Classificacao_Valor_Away_Momentum'] == '🏆 ALTO VALOR'])
    
    # Distribuição por categoria de momentum
    momentum_muito_alto = len(df[df['Quadrante_Home'].isin([1,5]) | df['Quadrante_Away'].isin([1,5])])
    momentum_alto = len(df[df['Quadrante_Home'].isin([2,6]) | df['Quadrante_Away'].isin([2,6])])
    momentum_baixo = len(df[df['Quadrante_Home'].isin([11,12,15,16]) | df['Quadrante_Away'].isin([11,12,15,16])])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("🚀 Momentum Explosivo", momentum_explosivo)
    with col2:
        st.metric("🎯 Alto Valor Home", alto_valor_home_momentum)
        st.metric("🎯 Alto Valor Away", alto_valor_away_momentum)
    with col3:
        st.metric("📈 Momentum Positivo", momentum_positivo)
        st.metric("💼 Valor Sólido", momentum_positivo + momentum_explosivo)
    with col4:
        st.metric("⚡ Momentum Muito Alto", momentum_muito_alto)
        st.metric("🔻 Momentum Baixo", momentum_baixo)

if not games_today.empty and 'Classificacao_Momentum' in games_today.columns:
    resumo_momentum_hoje(games_today)

st.markdown("---")
st.success("🎯 **Sistema de 16 Quadrantes MOMENTUM** implementado com sucesso!")
st.info("""
**Vantagens do Momentum (z-score):**
- ✅ **Comparação justa** entre ligas diferentes
- ✅ **Detecção de outliers** - momentos verdadeiramente excepcionais  
- ✅ **Normalização estatística** - melhor para modelos ML
- ✅ **Range consistente** (-3.5 a +3.5) vs HandScore variável
- ✅ **Remove viés de liga** - Premier League vs Série B

**Próximos passos:**
1. Compare os resultados com a versão HandScore
2. Verifique se há ganho na precisão do ML
3. Analise os padrões específicos de momentum
4. Ajuste os thresholds com base no backtest
""")
