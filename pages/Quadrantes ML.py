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

# ---------------- SISTEMA DE INDICAÇÕES EXPLÍCITAS ----------------
def adicionar_indicadores_explicativos(df):
    """Adiciona classificações e recomendações explícitas"""
    df = df.copy()
    
    # 1. CLASSIFICAÇÃO DE VALOR
    conditions = [
        df['Quadrante_ML_Score'] >= 0.60,
        df['Quadrante_ML_Score'] >= 0.55,
        df['Quadrante_ML_Score'] >= 0.50,
        df['Quadrante_ML_Score'] >= 0.45,
        df['Quadrante_ML_Score'] < 0.45
    ]
    choices = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor'] = np.select(conditions, choices, default='⚖️ NEUTRO')
    
    # 2. RECOMENDAÇÃO DE APOSTA
    def gerar_recomendacao(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score = row['Quadrante_ML_Score']
        
        # Combinações específicas
        if home_q == 'Underdog Value' and away_q == 'Market Overrates':
            return '🎯 VALUE NO HOME'
        elif home_q == 'Market Overrates' and away_q == 'Underdog Value':
            return '🎯 VALUE NO AWAY'
        elif home_q == 'Favorite Reliable' and away_q == 'Weak Underdog':
            return '💪 FAVORITO CONFIÁVEL'
        elif home_q == 'Weak Underdog' and away_q == 'Favorite Reliable':
            return '🚫 EVITAR HOME'
        elif 'Market Overrates' in home_q:
            return '🔴 HOME SUPERAVALIADO'
        elif 'Market Overrates' in away_q:
            return '🔴 AWAY SUPERAVALIADO'
        elif score >= 0.55:
            return '📈 MODELO CONFIA'
        elif score <= 0.45:
            return '📉 MODELO NÃO CONFIA'
        else:
            return '⚖️ ANALISAR OUTROS FATORES'
    
    df['Recomendacao'] = df.apply(gerar_recomendacao, axis=1)
    
    # 3. RANKING POR POTENCIAL
    df['Ranking'] = df['Quadrante_ML_Score'].rank(ascending=False, method='dense').astype(int)
    
    return df

def estilo_tabela_quadrantes(df):
    """Aplica estilo colorido à tabela"""
    def cor_classificacao(valor):
        if valor == '🏆 ALTO VALOR': return 'background-color: #90EE90; font-weight: bold'
        elif valor == '✅ BOM VALOR': return 'background-color: #FFFFE0; font-weight: bold' 
        elif valor == '🔴 ALTO RISCO': return 'background-color: #FFB6C1; font-weight: bold'
        elif 'VALUE' in valor: return 'background-color: #98FB98'
        elif 'EVITAR' in valor: return 'background-color: #FFCCCB'
        elif 'SUPERAVALIADO' in valor: return 'background-color: #FFA07A'
        else: return ''
    
    return df.style.applymap(cor_classificacao, subset=['Classificacao_Valor'])\
                  .applymap(cor_classificacao, subset=['Recomendacao'])\
                  .background_gradient(subset=['Quadrante_ML_Score'], cmap='RdYlGn')

# ---------------- ANÁLISE DE PADRÕES ----------------
def analisar_padroes_quadrantes(df):
    """Analisa padrões recorrentes nas combinações de quadrantes"""
    st.markdown("### 🔍 Análise de Padrões por Combinação")
    
    padroes = {
        'Underdog Value vs Market Overrates': {
            'descricao': '🎯 **MELHOR PADRÃO** - Zebra com valor vs Favorito supervalorizado',
            'prioridade': 1
        },
        'Favorite Reliable vs Weak Underdog': {
            'descricao': '💪 **PADRÃO FORTE** - Favorito confiável contra time fraco',
            'prioridade': 2
        }, 
        'Market Overrates vs Underdog Value': {
            'descricao': '🔴 **PADRÃO PERIGOSO** - Favorito supervalorizado vs Zebra com valor',
            'prioridade': 4
        },
        'Weak Underdog vs Favorite Reliable': {
            'descricao': '🚫 **EVITAR** - Time fraco contra favorito confiável',
            'prioridade': 5
        },
        'Underdog Value vs Weak Underdog': {
            'descricao': '⚖️ **PADRÃO EQUILIBRADO** - Zebra com valor contra time fraco',
            'prioridade': 3
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
            
            cols_padrao = ['Ranking', 'Home', 'Away', 'League', 'Quadrante_ML_Score', 'Recomendacao']
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]
            
            st.dataframe(
                jogos[cols_padrao]
                .sort_values('Quadrante_ML_Score', ascending=False)
                .style.format({'Quadrante_ML_Score': '{:.1%}'})
                .background_gradient(subset=['Quadrante_ML_Score'], cmap='RdYlGn'),
                use_container_width=True
            )
            st.write("---")

# ---------------- EXIBIÇÃO DOS RESULTADOS ----------------
st.markdown("## 🏆 Melhores Confrontos por Quadrantes ML")

if not games_today.empty and 'Quadrante_ML_Score' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()
    ranking_quadrantes['Quadrante_Home_Label'] = ranking_quadrantes['Quadrante_Home'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    ranking_quadrantes['Quadrante_Away_Label'] = ranking_quadrantes['Quadrante_Away'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    
    # Aplicar indicadores explicativos
    ranking_quadrantes = adicionar_indicadores_explicativos(ranking_quadrantes)
    
    # Ordenar por score do ML
    ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score', ascending=False)
    
    # Colunas para exibir
    cols_finais = [
        'Ranking', 'Home', 'Away', 'League', 
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score', 'Classificacao_Valor', 'Recomendacao'
    ]
    
    # Filtrar colunas existentes
    cols_finais = [c for c in cols_finais if c in ranking_quadrantes.columns]
    
    st.dataframe(
        estilo_tabela_quadrantes(ranking_quadrantes[cols_finais].head(20)),
        use_container_width=True
    )
    
    # Análise de padrões
    analisar_padroes_quadrantes(ranking_quadrantes)
    
else:
    st.info("⚠️ Aguardando dados para gerar ranking")

# ---------------- RESUMO EXECUTIVO ----------------
def resumo_quadrantes_hoje(df):
    """Resumo executivo dos quadrantes de hoje"""
    
    st.markdown("### 📋 Resumo Executivo - Quadrantes Hoje")
    
    if df.empty:
        st.info("Nenhum dado disponível para resumo")
        return
    
    total_jogos = len(df)
    alto_valor = len(df[df['Classificacao_Valor'] == '🏆 ALTO VALOR'])
    bom_valor = len(df[df['Classificacao_Valor'] == '✅ BOM VALOR'])
    alto_risco = len(df[df['Classificacao_Valor'] == '🔴 ALTO RISCO'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jogos", total_jogos)
    with col2:
        st.metric("🎯 Alto Valor", alto_valor)
    with col3:
        st.metric("✅ Bom Valor", bom_valor)
    with col4:
        st.metric("🔴 Alto Risco", alto_risco)
    
    # Distribuição de recomendações
    st.markdown("#### 📊 Distribuição de Recomendações")
    dist_recomendacoes = df['Recomendacao'].value_counts()
    st.dataframe(dist_recomendacoes, use_container_width=True)

if not games_today.empty and 'Classificacao_Valor' in games_today.columns:
    resumo_quadrantes_hoje(games_today)

st.markdown("---")
st.info("🎯 **Análise de Quadrantes ML** - Sistema avançado para identificação de value bets baseado em Aggression × HandScore")
