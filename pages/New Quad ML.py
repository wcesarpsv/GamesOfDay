# BLOCO 1: IMPORTS & CONFIGURAÇÕES
########################################

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurações da página
st.set_page_config(
    page_title="Análise de Quadrantes - Bet Indicator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Análise de Quadrantes - ML Avançado")

# Configurações globais
PAGE_PREFIX = "QuadrantesML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Sistema de 8 Quadrantes - CONSTANTES
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

# Estado da sessão para dados compartilhados entre blocos
if 'games_today' not in st.session_state:
    st.session_state.games_today = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'modelo_quadrantes' not in st.session_state:
    st.session_state.modelo_quadrantes = None

st.success("✅ Bloco 1 carregado: Imports & Configurações")





# BLOCO 2: DATA LOADER
########################################

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessa o DataFrame para padronizar nomes de colunas"""
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    """Carrega todos os arquivos CSV da pasta especificada"""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra ligas indesejadas baseado em keywords"""
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

def carregar_dados():
    """Função principal para carregar e processar todos os dados"""
    
    st.info("📂 Carregando dados para análise de quadrantes...")

    # Seleção de arquivo do dia - AGORA NA SIDEBAR
    st.sidebar.markdown("### 📅 Seleção de Data")
    
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    if not files:
        st.warning("No CSV files found in GamesDay folder.")
        return None, None

    # Extrair datas dos arquivos para mostrar de forma mais amigável
    file_options = []
    for f in files:
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", f)
        if date_match:
            date_str = date_match.group(0)
            formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
            file_options.append((f, formatted_date, date_str))
    
    # Ordenar por data (mais recente primeiro)
    file_options.sort(key=lambda x: x[2], reverse=True)
    
    # Criar opções para o selectbox
    options_display = [f"{date} - {filename}" for filename, date, _ in file_options[:10]]  # Últimos 10 dias
    options_files = [filename for filename, _, _ in file_options[:10]]
    
    selected_display = st.sidebar.selectbox(
        "Select Matchday File:", 
        options_display,
        index=0  # Mais recente por padrão
    )
    
    # Encontrar o arquivo correspondente
    selected_index = options_display.index(selected_display)
    selected_file = options_files[selected_index]
    
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    
    # Mostrar data selecionada
    formatted_selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
    st.sidebar.info(f"**Data selecionada:** {formatted_selected_date}")

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
            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {formatted_selected_date}")
        except Exception as e:
            st.error(f"Erro ao aplicar filtro temporal: {e}")

    # Targets AH históricos
    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(
        lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False) > 0.5 else 0, axis=1
    )

    return games_today, history

# Executar carregamento de dados
if st.session_state.games_today is None or st.session_state.history is None:
    with st.spinner("Carregando dados..."):
        games_today, history = carregar_dados()
        if games_today is not None and history is not None:
            st.session_state.games_today = games_today
            st.session_state.history = history
            st.success("✅ Dados carregados com sucesso!")
        else:
            st.error("❌ Falha ao carregar dados")
            st.stop()

st.success("✅ Bloco 2 carregado: Data Loader")


# BLOCO 3: SISTEMA DE QUADRANTES
########################################

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

def aplicar_classificacao_quadrantes(df):
    """Aplica a classificação de quadrantes a um DataFrame"""
    df = df.copy()
    df['Quadrante_Home'] = df.apply(
        lambda x: classificar_quadrante(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    df['Quadrante_Away'] = df.apply(
        lambda x: classificar_quadrante(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )
    
    # Adicionar labels
    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    
    return df

# Aplicar classificação aos dados
if st.session_state.games_today is not None and st.session_state.history is not None:
    st.session_state.games_today = aplicar_classificacao_quadrantes(st.session_state.games_today)
    st.session_state.history = aplicar_classificacao_quadrantes(st.session_state.history)
    st.success("✅ Classificação de quadrantes aplicada!")

st.success("✅ Bloco 3 carregado: Sistema de Quadrantes")



# BLOCO 4: VISUALIZAÇÃO INTERATIVA
########################################

def criar_grafico_quadrantes_interativo(df, side="Home", liga_filtro=None):
    """Cria gráfico interativo dos quadrantes com tooltips e clique"""
    
    # Aplicar filtro de liga se especificado
    if liga_filtro and liga_filtro != "Todas":
        df_filtrado = df[df['League'] == liga_filtro].copy()
    else:
        df_filtrado = df.copy()
    
    if df_filtrado.empty:
        st.warning(f"Nenhum jogo encontrado para a liga: {liga_filtro}")
        return None
    
    # Preparar dados para o gráfico
    plot_data = df_filtrado.copy()
    plot_data['Time'] = plot_data['Home' if side == 'Home' else 'Away']
    plot_data['Oponente'] = plot_data['Away' if side == 'Home' else 'Home']
    plot_data['League'] = plot_data['League']
    plot_data['Aggression'] = plot_data[f'Aggression_{side}']
    plot_data['HandScore'] = plot_data[f'HandScore_{side}']
    plot_data['Quadrante'] = plot_data[f'Quadrante_{side}_Label']
    
    # Criar gráfico interativo com Plotly
    fig = px.scatter(
        plot_data,
        x='Aggression',
        y='HandScore',
        color='Quadrante',
        hover_data={
            'Time': True,
            'Oponente': True,
            'League': True,
            'Aggression': ':.2f',
            'HandScore': ':.0f',
            'Quadrante': True
        },
        title=f'Quadrantes {side} - Liga: {liga_filtro if liga_filtro else "Todas"}',
        labels={
            'Aggression': f'Aggression {side} (-1 zebra ↔ +1 favorito)',
            'HandScore': f'HandScore {side} (-60 a +60)'
        }
    )
    
    # Adicionar linhas de divisão dos quadrantes
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
    fig.add_hline(y=15, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=-15, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=-30, line_dash="dash", line_color="gray", opacity=0.3)
    
    fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.5)
    fig.add_vline(x=-0.5, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.3)
    
    # Melhorar layout
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    # Adicionar anotações dos quadrantes
    annotations = [
        dict(x=-0.75, y=45, text="Underdog<br>Value Forte", showarrow=False, font=dict(size=10, color="black")),
        dict(x=-0.25, y=22, text="Underdog<br>Value", showarrow=False, font=dict(size=9, color="black")),
        dict(x=0.75, y=45, text="Favorite<br>Reliable Forte", showarrow=False, font=dict(size=10, color="black")),
        dict(x=0.25, y=22, text="Favorite<br>Reliable", showarrow=False, font=dict(size=9, color="black")),
        dict(x=0.75, y=-45, text="Market<br>Overrates Forte", showarrow=False, font=dict(size=10, color="black")),
        dict(x=0.25, y=-22, text="Market<br>Overrates", showarrow=False, font=dict(size=9, color="black")),
        dict(x=-0.75, y=-45, text="Weak<br>Underdog Forte", showarrow=False, font=dict(size=10, color="black")),
        dict(x=-0.25, y=-22, text="Weak<br>Underdog", showarrow=False, font=dict(size=9, color="black"))
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig

def exibir_detalhes_time(time_data, side):
    """Exibe detalhes do time selecionado no gráfico"""
    st.markdown(f"### 🔍 Detalhes do {side}: {time_data['Time']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Oponente", time_data['Oponente'])
        st.metric("Liga", time_data['League'])
    
    with col2:
        st.metric("Aggression", f"{time_data['Aggression']:.2f}")
        st.metric("HandScore", f"{time_data['HandScore']:.0f}")
    
    with col3:
        st.metric("Quadrante", time_data['Quadrante'])
        st.metric("Status", "✅ Analisar" if "Value" in time_data['Quadrante'] else "⚠️ Cautela")

def mostrar_visualizacao_quadrantes():
    """Interface principal da visualização de quadrantes"""
    
    st.markdown("## 📈 Visualização Interativa dos Quadrantes")
    
    # Filtros na sidebar
    st.sidebar.markdown("### 🔍 Filtros de Visualização")
    
    # Filtro por liga
    ligas_disponiveis = ["Todas"] + sorted(st.session_state.games_today['League'].unique().tolist())
    liga_selecionada = st.sidebar.selectbox("Selecionar Liga:", ligas_disponiveis)
    
    # Layout dos gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_home = criar_grafico_quadrantes_interativo(
            st.session_state.games_today, 
            side="Home", 
            liga_filtro=liga_selecionada
        )
        if fig_home:
            st.plotly_chart(fig_home, use_container_width=True)
    
    with col2:
        fig_away = criar_grafico_quadrantes_interativo(
            st.session_state.games_today, 
            side="Away", 
            liga_filtro=liga_selecionada
        )
        if fig_away:
            st.plotly_chart(fig_away, use_container_width=True)

st.success("✅ Bloco 4 carregado: Visualização Interativa")


# BLOCO 5: ML & FEATURE ENGINEERING
########################################

def treinar_modelo_quadrantes(history, games_today):
    """Treina modelo ML baseado em quadrantes + ligas com one-hot encoding otimizado"""
    
    # Filtrar ligas com poucos jogos para evitar overfitting
    liga_counts = history['League'].value_counts()
    ligas_validas = liga_counts[liga_counts >= 10].index  # Mínimo 10 jogos por liga
    history_filtrado = history[history['League'].isin(ligas_validas)].copy()
    
    if history_filtrado.empty:
        st.warning("⚠️ Histórico insuficiente após filtrar ligas com poucos jogos")
        return None, games_today
    
    # Preparar features: one-hot encoding de quadrantes e ligas
    quadrantes_home = pd.get_dummies(history_filtrado['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history_filtrado['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history_filtrado['League'], prefix='League')
    
    # Combinar features
    X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies], axis=1)
    
    # Target: sucesso do Home no Asian Handicap
    y = history_filtrado['Target_AH_Home']
    
    # Garantir que todas as colunas existem
    quadrante_cols = list(quadrantes_home.columns) + list(quadrantes_away.columns)
    liga_cols = list(ligas_dummies.columns)
    todas_cols = quadrante_cols + liga_cols
    
    # Treinar modelo com regularização
    model = RandomForestClassifier(
        n_estimators=150, 
        random_state=42, 
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt'
    )
    model.fit(X, y)
    
    # Preparar dados de hoje - garantir mesma estrutura
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH')
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA')
    ligas_today = pd.get_dummies(games_today['League'], prefix='League')
    
    # Reindexar para garantir mesmas colunas do treino
    X_today = pd.concat([qh_today, qa_today, ligas_today], axis=1)
    X_today = X_today.reindex(columns=todas_cols, fill_value=0)
    
    # Fazer previsões
    probas = model.predict_proba(X_today)[:, 1]  # Probabilidade de sucesso do Home
    games_today['Quadrante_ML_Score'] = probas
    
    # Calcular importância das features
    importancia = pd.DataFrame({
        'feature': todas_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, games_today, importancia

def executar_treinamento_ml():
    """Executa o treinamento do modelo ML"""
    
    if st.session_state.history is None or st.session_state.games_today is None:
        st.error("❌ Dados não carregados para treinamento")
        return
    
    with st.spinner("🔄 Treinando modelo ML..."):
        modelo, games_atualizado, importancia = treinar_modelo_quadrantes(
            st.session_state.history, 
            st.session_state.games_today
        )
        
        if modelo is not None:
            st.session_state.modelo_quadrantes = modelo
            st.session_state.games_today = games_atualizado
            st.session_state.importancia_features = importancia
            
            st.success(f"✅ Modelo treinado com {len(st.session_state.history)} jogos históricos")
            
            # Mostrar top features
            st.markdown("#### 🎯 Features Mais Importantes do Modelo")
            st.dataframe(
                importancia.head(10)
                .style.background_gradient(subset=['importance'], cmap='Blues'),
                use_container_width=True
            )
        else:
            st.error("❌ Falha no treinamento do modelo")

# Executar treinamento se ainda não foi feito
if st.session_state.modelo_quadrantes is None:
    executar_treinamento_ml()

st.success("✅ Bloco 5 carregado: ML & Feature Engineering")



# BLOCO 6: INDICAÇÕES & RECOMENDAÇÕES
########################################

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

# Aplicar indicadores aos dados
if st.session_state.games_today is not None and 'Quadrante_ML_Score' in st.session_state.games_today.columns:
    st.session_state.games_today = adicionar_indicadores_explicativos(st.session_state.games_today)

st.success("✅ Bloco 6 carregado: Indicações & Recomendações")


# BLOCO 7: UI & LAYOUT
########################################

def criar_resumo_executivo():
    """Cria resumo executivo dos quadrantes de hoje"""
    
    st.markdown("### 📋 Resumo Executivo - Quadrantes Hoje")
    
    if st.session_state.games_today is None or st.session_state.games_today.empty:
        st.info("Nenhum dado disponível para resumo")
        return
    
    df = st.session_state.games_today
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
    if 'Recomendacao' in df.columns:
        dist_recomendacoes = df['Recomendacao'].value_counts()
        st.dataframe(dist_recomendacoes, use_container_width=True)

def mostrar_tabela_principal():
    """Exibe a tabela principal com rankings e recomendações"""
    
    st.markdown("## 🏆 Melhores Confrontos por Quadrantes ML")
    
    if (st.session_state.games_today is not None and 
        'Quadrante_ML_Score' in st.session_state.games_today.columns):
        
        df = st.session_state.games_today
        
        # Colunas para exibir
        cols_finais = [
            'Ranking', 'Home', 'Away', 'League', 
            'Quadrante_Home_Label', 'Quadrante_Away_Label',
            'Quadrante_ML_Score', 'Classificacao_Valor', 'Recomendacao'
        ]
        
        # Filtrar colunas existentes
        cols_finais = [c for c in cols_finais if c in df.columns]
        
        # Ordenar por ranking
        df_ordenado = df.sort_values('Ranking')
        
        st.dataframe(
            estilo_tabela_quadrantes(df_ordenado[cols_finais].head(20)),
            use_container_width=True
        )
        
        # Análise de padrões
        analisar_padroes_quadrantes(df_ordenado)
        
    else:
        st.info("⚠️ Aguardando dados para gerar ranking")

def main():
    """Função principal que orquestra toda a aplicação"""
    
    # Sidebar com controles
    st.sidebar.markdown("## ⚙️ Controles")
    
    # Botão para recarregar dados
    if st.sidebar.button("🔄 Recarregar Dados e Retreinar Modelo"):
        st.session_state.games_today = None
        st.session_state.history = None
        st.session_state.modelo_quadrantes = None
        st.rerun()
    
    # Seção principal
    if (st.session_state.games_today is not None and 
        st.session_state.history is not None):
        
        # Mostrar visualização interativa
        mostrar_visualizacao_quadrantes()
        
        # Mostrar resumo executivo
        criar_resumo_executivo()
        
        # Mostrar tabela principal
        mostrar_tabela_principal()
        
    else:
        st.error("❌ Erro ao carregar dados. Verifique se os arquivos CSV estão na pasta GamesDay.")

# Executar aplicação
if __name__ == "__main__":
    main()

st.success("✅ Bloco 7 carregado: UI & Layout")
st.success("🎉 TODOS OS BLOCOS CARREGADOS COM SUCESSO!")
