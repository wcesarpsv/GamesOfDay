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
st.title("🎯 Análise de Quadrantes - ML Avançado")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

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

# Histórico consolidado
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

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
        0: 'white'          # Neutro
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

# ---------------- ML PARA RANKEAR CONFRONTOS ----------------
def treinar_modelo_quadrantes(history, games_today):
    """Treina modelo ML baseado em quadrantes + ligas"""
    
    # Preparar features: one-hot encoding de quadrantes e ligas
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    
    # Combinar features
    X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies], axis=1)
    
    # Target: sucesso do Home no Asian Handicap
    y = history['Target_AH_Home']
    
    # Garantir que todas as colunas existem
    quadrante_cols = list(quadrantes_home.columns) + list(quadrantes_away.columns)
    liga_cols = list(ligas_dummies.columns)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model.fit(X, y)
    
    # Preparar dados de hoje
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=liga_cols, fill_value=0)
    
    X_today = pd.concat([qh_today, qa_today, ligas_today], axis=1)
    
    # Fazer previsões
    probas = model.predict_proba(X_today)[:, 1]  # Probabilidade de sucesso do Home
    games_today['Quadrante_ML_Score'] = probas
    
    return model, games_today

# Executar treinamento
if not history.empty:
    modelo_quadrantes, games_today = treinar_modelo_quadrantes(history, games_today)
    st.success("✅ Modelo de quadrantes treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

# ---------------- PERFORMANCE HISTÓRICA ----------------
def analisar_performance_quadrantes(history):
    """Analisa performance histórica por combinação de quadrantes"""
    
    # Criar combinação Home-Away
    history['Combinacao_Quadrantes'] = history['Quadrante_Home'].astype(str) + '_' + history['Quadrante_Away'].astype(str)
    
    # Calcular métricas por combinação
    performance = history.groupby('Combinacao_Quadrantes').agg({
        'Target_AH_Home': ['count', 'mean', 'std'],
        'Goals_H_FT': 'mean',
        'Goals_A_FT': 'mean'
    }).round(3)
    
    performance.columns = ['Jogos', 'Taxa_Acerto', 'Desvio_Padrao', 'Gols_Home', 'Gols_Away']
    performance['ROI_Estimado'] = (performance['Taxa_Acerto'] * 1.91 - 1) * 100
    
    # Adicionar labels
    performance['Quadrante_Home_Label'] = performance.index.map(
        lambda x: QUADRANTES_8.get(int(x.split('_')[0]), {}).get('nome', 'Neutro') if x != '0_0' else 'Neutro'
    )
    performance['Quadrante_Away_Label'] = performance.index.map(
        lambda x: QUADRANTES_8.get(int(x.split('_')[1]), {}).get('nome', 'Neutro') if x != '0_0' else 'Neutro'
    )
    
    return performance.sort_values('Taxa_Acerto', ascending=False)

# Exibir performance histórica
st.markdown("### 📊 Performance Histórica por Combinação de Quadrantes")

if not history.empty:
    performance_df = analisar_performance_quadrantes(history)
    
    st.dataframe(
        performance_df
        .style.format({
            'Taxa_Acerto': '{:.1%}',
            'ROI_Estimado': '{:.1f}%',
            'Gols_Home': '{:.1f}',
            'Gols_Away': '{:.1f}'
        })
        .background_gradient(subset=['Taxa_Acerto'], cmap='RdYlGn')
        .background_gradient(subset=['ROI_Estimado'], cmap='RdYlGn'),
        use_container_width=True
    )
else:
    st.info("⚠️ Sem dados históricos para análise de performance")

# ---------------- RANKING DOS MELHORES CONFRONTOS ----------------
st.markdown("### 🏆 Melhores Confrontos por Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()
    ranking_quadrantes['Quadrante_Home_Label'] = ranking_quadrantes['Quadrante_Home'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    ranking_quadrantes['Quadrante_Away_Label'] = ranking_quadrantes['Quadrante_Away'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    
    # Colunas para exibir
    cols_ranking = [
        'Home', 'Away', 'League', 
        'Quadrante_Home_Label', 'Quadrante_Away_Label', 
        'Quadrante_ML_Score', 'Asian_Line_Home_Display'
    ]
    
    # Filtrar colunas existentes
    cols_ranking = [c for c in cols_ranking if c in ranking_quadrantes.columns]
    
    # Ordenar por score do ML
    ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score', ascending=False)
    
    st.dataframe(
        ranking_quadrantes[cols_ranking].head(15)
        .style.format({
            'Quadrante_ML_Score': '{:.1%}',
            'Asian_Line_Home_Display': '{:+.2f}'
        })
        .background_gradient(subset=['Quadrante_ML_Score'], cmap='RdYlGn'),
        use_container_width=True
    )
else:
    st.info("⚠️ Aguardando dados para gerar ranking")

# ---------------- RESUMO EXECUTIVO ----------------
def resumo_quadrantes_hoje(games_today):
    """Resumo executivo dos quadrantes de hoje"""
    
    st.markdown("### 📋 Resumo Executivo - Quadrantes Hoje")
    
    total_jogos = len(games_today)
    jogos_quadrantes_fortes = len(games_today[
        (games_today['Quadrante_Home'].isin([1, 3, 5, 7])) | 
        (games_today['Quadrante_Away'].isin([1, 3, 5, 7]))
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jogos", total_jogos)
    with col2:
        st.metric("Jogos Quadrantes Fortes", jogos_quadrantes_fortes)
    with col3:
        taxa_fortes = (jogos_quadrantes_fortes/total_jogos*100) if total_jogos > 0 else 0
        st.metric("Taxa Quadrantes Fortes", f"{taxa_fortes:.1f}%")
    with col4:
        if 'Quadrante_ML_Score' in games_today.columns:
            melhor_score = games_today['Quadrante_ML_Score'].max()
            st.metric("Melhor Score ML", f"{melhor_score:.1%}")
        else:
            st.metric("Melhor Score ML", "N/A")

if not games_today.empty:
    resumo_quadrantes_hoje(games_today)

# ---------------- DISTRIBUIÇÃO DOS QUADRANTES ----------------
st.markdown("### 📊 Distribuição dos Quadrantes - Hoje")

if not games_today.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        dist_home = games_today['Quadrante_Home'].value_counts().sort_index()
        dist_home.index = [QUADRANTES_8.get(i, {}).get('nome', 'Neutro') for i in dist_home.index]
        st.write("**Home:**")
        st.dataframe(dist_home)
    
    with col2:
        dist_away = games_today['Quadrante_Away'].value_counts().sort_index()
        dist_away.index = [QUADRANTES_8.get(i, {}).get('nome', 'Neutro') for i in dist_away.index]
        st.write("**Away:**")
        st.dataframe(dist_away)

st.markdown("---")
st.info("🎯 **Análise de Quadrantes ML** - Sistema avançado para identificação de value bets baseado em Aggression × HandScore")