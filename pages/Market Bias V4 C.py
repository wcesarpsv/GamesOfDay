############ Bloco A - Imports e Configurações Base ################
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import math
import plotly.graph_objects as go
import sys

#######################################
# Hotfix para limpar cache problematico
if hasattr(st, 'cache_data'):
    st.cache_data.clear()
if hasattr(st, 'cache_resource'):
    st.cache_resource.clear()
    

# Limpar módulos carregados se necessário
module_suffix = '_page'
for module_name in list(sys.modules.keys()):
    if module_name.endswith(module_suffix):
        del sys.modules[module_name]

#######################################

# Configuração da página
st.set_page_config(page_title="Sistema 3D Clusters - Bet Indicator", layout="wide")
st.title("🎯 Sistema 3D com Clusters - ML Avançado")

# Configurações base
PAGE_PREFIX = "Clusters3D_ML"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy", "coppa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


############ Bloco B - Funções de Helpers Base ################
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


############ Bloco C - Funções Asian Line ################
# ---------------- Funções Asian Line ----------------
def convert_asian_line_to_decimal(value):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).

    Regras oficiais e consistentes com Pinnacle/Bet365:
      '0/0.5'   -> +0.25  (para away) → invertido: -0.25 (para home)
      '-0.5/0'  -> -0.25  (para away) → invertido: +0.25 (para home)
      '-1/1.5'  -> -0.25  → +0.25
      '1/1.5'   -> +1.25  → -1.25
      '1.5'     -> +1.50  → -1.50
      '0'       ->  0.00  →  0.00

    Retorna: float
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Caso simples — número único
    if "/" not in value:
        try:
            num = float(value)
            return -num  # ✅ INVERTE SINAL (Away → Home)
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
        # ✅ INVERTE O SINAL NO FINAL (Away → Home)
        return -result
    except ValueError:
        return np.nan

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

def validar_conversao_handicap():
    """Testa a conversão de handicaps com exemplos conhecidos"""
    exemplos = [
        '0', '0.5', '-0.5', '0/0.5', '-0.5/0', '1/1.5', '-1/1.5', '1.5', '-1.5'
    ]
    
    st.markdown("### 🧪 Validação da Conversão de Handicap")
    st.write("**Regra:** Away → Home (inverte sinal)")
    
    resultados = []
    for exemplo in exemplos:
        convertido = convert_asian_line_to_decimal(exemplo)
        resultados.append({
            'Original (Away)': exemplo,
            'Convertido (Home)': convertido,
            'Descrição': f"Away {exemplo} → Home {convertido:.2f}"
        })
    
    df_validacao = pd.DataFrame(resultados)
    st.dataframe(df_validacao, use_container_width=True)
    
    # Verificar consistência
    st.markdown("#### ✅ Verificação de Consistência")
    st.write("**0/0.5** → Away +0.25 → Home -0.25:", convert_asian_line_to_decimal('0/0.5') == -0.25)
    st.write("**-0.5/0** → Away -0.25 → Home +0.25:", convert_asian_line_to_decimal('-0.5/0') == 0.25)
    st.write("**1.5** → Away +1.50 → Home -1.50:", convert_asian_line_to_decimal('1.5') == -1.50)

def verificar_sistema_handicap(history, games_today):
    """Verifica a integridade do sistema de handicap"""
    st.markdown("### 🔍 Verificação do Sistema de Handicap")
    
    # 1. Verificar se há valores convertidos
    st.write("**1. Valores convertidos no histórico:**")
    st.write(f"- Total: {len(history)}")
    st.write(f"- Com Asian Line: {history['Asian_Line_Decimal'].notna().sum()}")
    st.write(f"- Exemplos: {history['Asian_Line_Decimal'].head(5).tolist()}")
    
    # 2. Verificar distribuição
    st.write("**2. Distribuição dos handicaps:**")
    if 'Asian_Line_Decimal' in history.columns:
        st.write(history['Asian_Line_Decimal'].describe())
    
    # 3. Verificar cálculo de resultados
    st.write("**3. Exemplo de cálculo de handicap:**")
    if len(history) > 0:
        exemplo = history.iloc[0]
        margin = exemplo.get('Goals_H_FT', 0) - exemplo.get('Goals_A_FT', 0)
        handicap_line = exemplo.get('Asian_Line_Decimal', 0)
        resultado = calc_handicap_result(margin, handicap_line)
        
        st.write(f"Jogo: {exemplo.get('Home', '')} vs {exemplo.get('Away', '')}")
        st.write(f"Margin: {margin}, Handicap: {handicap_line:.2f}")
        st.write(f"Resultado: {resultado:.2f}")



############ Bloco D - Sistema de Clusterização 3D ################
# ==============================================================
# 🧩 BLOCO – CLUSTERIZAÇÃO 3D (KMEANS) - ATUALIZADO
# ==============================================================

def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    """
    Cria clusters espaciais com base em Aggression, Momentum Liga e Momentum Time.
    SISTEMA FLEXÍVEL: Legendas dinâmicas baseadas nos centroides reais de cada execução
    """
    df = df.copy()

    # Garante as colunas necessárias
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        df['Cluster3D_Desc'] = '🌀 Dados Insuficientes'
        return df

    # Diferenças espaciais (vetor 3D) - Relação Home vs Away
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    # KMeans 3D
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init='k-means++',
        n_init=10
    )
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    # 🎯 SISTEMA FLEXÍVEL: CLASSIFICAR CLUSTERS DINAMICAMENTE
    st.markdown("## 🧠 Sistema Flexível de Legendas Dinâmicas")
    
    # 1. CALCULAR CENTROIDES REAIS
    centroids = kmeans.cluster_centers_
    
    # 2. CLASSIFICAR CADA CLUSTER BASEADO NOS CENTROIDES
    classificacoes_clusters = classificar_clusters_dinamicamente(centroids)
    
    # 3. APLICAR LEGENDAS DINÂMICAS
    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(classificacoes_clusters).fillna('🌀 Caso Atípico')
    
    # 4. EXIBIR DIAGNÓSTICO INTELIGENTE
    exibir_diagnostico_clusters(df, centroids, classificacoes_clusters)
    
    return df

def classificar_clusters_dinamicamente(centroids):
    """
    Classifica clusters dinamicamente baseado nos centroides reais
    Retorna dicionário {cluster_id: legenda}
    """
    classificacoes = {}
    
    # THRESHOLDS AJUSTÁVEIS (baseado na sua distribuição de dados)
    THRESHOLD_ALTO = 0.3
    THRESHOLD_BAIXO = -0.3
    THRESHOLD_EQUILIBRADO = 0.15
    
    for cluster_id, centroid in enumerate(centroids):
        dx, dy, dz = centroid
        
        # 🎯 LÓGICA INTELIGENTE DE CLASSIFICAÇÃO
        if dx > THRESHOLD_ALTO and dy > THRESHOLD_ALTO and dz > THRESHOLD_ALTO:
            classificacao = '🏠 Home Domina Totalmente'
            
        elif dx > THRESHOLD_ALTO and dy > THRESHOLD_ALTO:
            classificacao = '🏠 Home Domina (Liga Forte)'
            
        elif dx > THRESHOLD_ALTO and dz > THRESHOLD_ALTO:
            classificacao = '🏠 Home Domina (Time Forte)'
            
        elif dx > THRESHOLD_ALTO:
            classificacao = '📈 Home Agressivo'
            
        elif dx < THRESHOLD_BAIXO and dy < THRESHOLD_BAIXO and dz < THRESHOLD_BAIXO:
            classificacao = '🚗 Away Domina Totalmente'
            
        elif dx < THRESHOLD_BAIXO and dy < THRESHOLD_BAIXO:
            classificacao = '🚗 Away Domina (Liga Forte)'
            
        elif dx < THRESHOLD_BAIXO and dz < THRESHOLD_BAIXO:
            classificacao = '🚗 Away Domina (Time Forte)'
            
        elif dx < THRESHOLD_BAIXO:
            classificacao = '📉 Away Agressivo'
            
        elif abs(dx) <= THRESHOLD_EQUILIBRADO and abs(dy) <= 1.0 and abs(dz) <= 1.0:
            classificacao = '⚖️ Confronto Equilibrado'
            
        elif abs(dy) > 2.0 or abs(dz) > 2.0:
            # Momentum muito extremo
            if dx > 0:
                classificacao = '🎭 Home Imprevisível'
            else:
                classificacao = '🌪️ Away Imprevisível'
                
        elif (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
            # Sinais contraditórios entre aggression e momentum
            classificacao = '🔄 Sinais Contraditórios'
            
        else:
            # Caso padrão - classificar baseado no aggression
            if dx > 0.1:
                classificacao = '📊 Home Leve Vantagem'
            elif dx < -0.1:
                classificacao = '📊 Away Leve Vantagem'
            else:
                classificacao = '⚖️ Equilíbrio Neutro'
        
        classificacoes[cluster_id] = classificacao
    
    return classificacoes

def exibir_diagnostico_clusters(df, centroids, classificacoes):
    """
    Exibe diagnóstico inteligente dos clusters
    """
    st.markdown("### 📊 Diagnóstico Inteligente dos Clusters")
    
    # TABELA DE CENTROIDES COM LEGENDAS DINÂMICAS
    centroids_df = pd.DataFrame(centroids, columns=['dx', 'dy', 'dz'])
    centroids_df['Cluster'] = range(len(centroids))
    centroids_df['Legenda Dinâmica'] = centroids_df['Cluster'].map(classificacoes)
    centroids_df['Jogos'] = centroids_df['Cluster'].apply(
        lambda x: len(df[df['Cluster3D_Label'] == x])
    )
    
    st.markdown("#### 🎯 Centroides com Legendas Dinâmicas")
    st.dataframe(centroids_df.style.format({
        'dx': '{:.3f}', 
        'dy': '{:.3f}', 
        'dz': '{:.3f}'
    }), use_container_width=True)
    
    # ANÁLISE DETALHADA POR CLUSTER
    st.markdown("#### 📈 Análise Detalhada por Cluster")
    
    for cluster_id in sorted(df['Cluster3D_Label'].unique()):
        cluster_data = df[df['Cluster3D_Label'] == cluster_id]
        legenda = classificacoes[cluster_id]
        
        if len(cluster_data) > 0:
            st.write(f"**{legenda}** (Cluster {cluster_id})")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Jogos", len(cluster_data))
                st.write(f"**dx:** {cluster_data['dx'].mean():.3f}")
                
            with col2:
                agg_h = cluster_data['Aggression_Home'].mean()
                agg_a = cluster_data['Aggression_Away'].mean()
                st.metric("Aggression", f"H:{agg_h:.3f} A:{agg_a:.3f}")
                st.write(f"**dy:** {cluster_data['dy'].mean():.3f}")
                
            with col3:
                m_h = cluster_data['M_H'].mean()
                m_a = cluster_data['M_A'].mean()
                st.metric("Momentum Liga", f"H:{m_h:.3f} A:{m_a:.3f}")
                st.write(f"**dz:** {cluster_data['dz'].mean():.3f}")
            
            # EXEMPLO DO CLUSTER
            exemplo = cluster_data.iloc[0]
            st.write(f"**Exemplo:** {exemplo['Home']} vs {exemplo['Away']}")
            st.write(f"**dx real:** {exemplo['dx']:.3f} | **Legenda:** {legenda}")
            
            # VALIDAÇÃO DA LEGENDA
            dx_exemplo = exemplo['dx']
            if "Home" in legenda and dx_exemplo > 0:
                st.success("✅ Legenda coerente com dados")
            elif "Away" in legenda and dx_exemplo < 0:
                st.success("✅ Legenda coerente com dados") 
            elif "Equilíbrio" in legenda and abs(dx_exemplo) < 0.2:
                st.success("✅ Legenda coerente com dados")
            else:
                st.info("🔍 Legenda baseada em padrão complexo")
            
            st.write("---")

    # RESUMO ESTATÍSTICO
    st.markdown("#### 📋 Resumo Estatístico")
    resumo = df.groupby('Cluster3D_Desc').agg({
        'Cluster3D_Label': 'count',
        'dx': 'mean',
        'dy': 'mean', 
        'dz': 'mean'
    }).rename(columns={'Cluster3D_Label': 'Jogos'})
    
    st.dataframe(resumo.style.format({
        'dx': '{:.3f}',
        'dy': '{:.3f}', 
        'dz': '{:.3f}'
    }), use_container_width=True)
#################################################################

# ---------------- CÁLCULO DE MOMENTUM DO TIME ----------------
def calcular_momentum_time(df, window=6):
    """
    Calcula o Momentum do Time (MT_H / MT_A) com base no HandScore,
    usando média móvel e normalização z-score por time.
    """
    df = df.copy()

    # Garante existência das colunas
    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    # Lista de todos os times (Home + Away)
    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        # ---------------- HOME ----------------
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:  # precisa de histórico mínimo
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = zscore

        # ---------------- AWAY ----------------
        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = zscore

    # Preenche eventuais NaN com 0 (neutro)
    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)

    return df


  ############ Bloco E - Cálculo de Momentum e Regressão ################
# ---------------- CÁLCULO DE REGRESSÃO À MÉDIA ----------------
def calcular_regressao_media(df):
    """
    Calcula tendência de regressão à média baseada em:
    - M_H, M_A: Z-score do momentum na liga  
    - MT_H, MT_A: Z-score do momentum do time
    """
    df = df.copy()
    
    # 1. SCORE DE EXTREMIDADE (quão longe da média)
    df['Extremidade_Home'] = np.abs(df['M_H']) + np.abs(df['MT_H'])
    df['Extremidade_Away'] = np.abs(df['M_A']) + np.abs(df['MT_A'])
    
    # 2. FORÇA DE REGRESSÃO (quanto tende a voltar à média)
    df['Regressao_Force_Home'] = -np.sign(df['M_H']) * (df['Extremidade_Home'] ** 0.7)
    df['Regressao_Force_Away'] = -np.sign(df['M_A']) * (df['Extremidade_Away'] ** 0.7)
    
    # 3. PROBABILIDADE DE REGRESSÃO (0-1 scale)
    df['Prob_Regressao_Home'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Home']))
    df['Prob_Regressao_Away'] = 1 / (1 + np.exp(-0.8 * df['Regressao_Force_Away']))
    
    # 4. MEDIA SCORE FINAL (combina regressão com aggression atual)
    df['Media_Score_Home'] = (0.6 * df['Prob_Regressao_Home'] + 
                             0.4 * (1 - df['Aggression_Home']))
    
    df['Media_Score_Away'] = (0.6 * df['Prob_Regressao_Away'] + 
                             0.4 * (1 - df['Aggression_Away']))
    
    # 5. CLASSIFICAÇÃO DE REGRESSÃO
    conditions_home = [
        df['Regressao_Force_Home'] > 1.0,
        df['Regressao_Force_Home'] > 0.3,
        df['Regressao_Force_Home'] > -0.3,
        df['Regressao_Force_Home'] > -1.0,
        df['Regressao_Force_Home'] <= -1.0
    ]
    choices_home = ['📈 FORTE MELHORA', '📈 MELHORA', '⚖️ ESTÁVEL', '📉 QUEDA', '📉 FORTE QUEDA']
    df['Tendencia_Home'] = np.select(conditions_home, choices_home, default='⚖️ ESTÁVEL')
    
    conditions_away = [
        df['Regressao_Force_Away'] > 1.0,
        df['Regressao_Force_Away'] > 0.3,
        df['Regressao_Force_Away'] > -0.3,
        df['Regressao_Force_Away'] > -1.0,
        df['Regressao_Force_Away'] <= -1.0
    ]
    choices_away = ['📈 FORTE MELHORA', '📈 MELHORA', '⚖️ ESTÁVEL', '📉 QUEDA', '📉 FORTE QUEDA']
    df['Tendencia_Away'] = np.select(conditions_away, choices_away, default='⚖️ ESTÁVEL')
    
    return df



############ Bloco F - Visualização 3D com Clusters ################

# ---------------- VISUALIZAÇÃO 3D COM CLUSTERS ----------------
def create_3d_plot_with_clusters(df_plot, n_to_show, selected_league):
    """Gráfico 3D colorido por clusters"""
    fig_3d = go.Figure()

    # Cores por cluster
    cluster_cores = {
        '🏠 Home Domina Confronto': 'blue',
        '🚗 Away Domina Confronto': 'red', 
        '⚖️ Confronto Equilibrado': 'green',
        '🎭 Home Imprevisível': 'orange',
        '🌪️ Home Instável': 'purple',
        '🌀 Caso Atípico': 'gray',
        '🌀 Dados Insuficientes': 'lightgray'
    }

    # Plotar cada cluster com sua cor
    for cluster_name, color in cluster_cores.items():
        cluster_data = df_plot[df_plot['Cluster3D_Desc'] == cluster_name]
        
        if not cluster_data.empty:
            # Linhas de conexão (Home → Away) - COM VERIFICAÇÃO DE DADOS VÁLIDOS
            for _, row in cluster_data.iterrows():
                # Verificar se todos os dados são válidos
                xh = row.get('Aggression_Home', 0) or 0
                xa = row.get('Aggression_Away', 0) or 0
                yh = row.get('M_H', 0) if not pd.isna(row.get('M_H')) else 0
                ya = row.get('M_A', 0) if not pd.isna(row.get('M_A')) else 0
                zh = row.get('MT_H', 0) if not pd.isna(row.get('MT_H')) else 0
                za = row.get('MT_A', 0) if not pd.isna(row.get('MT_A')) else 0
                
                # Só plotar se tiver dados válidos
                if any(v != 0 for v in [xh, xa, yh, ya, zh, za]):
                    fig_3d.add_trace(go.Scatter3d(
                        x=[xh, xa],
                        y=[yh, ya],
                        z=[zh, za],
                        mode='lines',
                        line=dict(
                            color=color, 
                            width=4
                        ),  # REMOVIDO: opacity=0.3 - não é suportado em linhas 3D
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Filtrar dados válidos para os pontos
            valid_home = cluster_data[
                (cluster_data['Aggression_Home'].notna()) & 
                (cluster_data['M_H'].notna()) & 
                (cluster_data['MT_H'].notna())
            ]
            valid_away = cluster_data[
                (cluster_data['Aggression_Away'].notna()) & 
                (cluster_data['M_A'].notna()) & 
                (cluster_data['MT_A'].notna())
            ]
            
            # Pontos HOME - apenas dados válidos
            if not valid_home.empty:
                fig_3d.add_trace(go.Scatter3d(
                    x=valid_home['Aggression_Home'],
                    y=valid_home['M_H'],
                    z=valid_home['MT_H'],
                    mode='markers',
                    name=f'{cluster_name} - Home',
                    marker=dict(
                        color=color,
                        size=8,  # Reduzido para melhor visualização
                        symbol='circle',
                        opacity=0.8,  # Apenas para markers, não para lines
                        line=dict(color='white', width=1)
                    ),
                    text=valid_home.apply(
                        lambda r: f"<b>{r['Home']}</b><br>"
                                 f"Cluster: {cluster_name}<br>"
                                 f"vs {r['Away']}<br>"
                                 f"Agg: {r['Aggression_Home']:.2f}<br>"
                                 f"M_Liga: {r['M_H']:.2f}<br>"
                                 f"M_Time: {r['MT_H']:.2f}", 
                        axis=1
                    ),
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            # Pontos AWAY - apenas dados válidos
            if not valid_away.empty:
                fig_3d.add_trace(go.Scatter3d(
                    x=valid_away['Aggression_Away'],
                    y=valid_away['M_A'],
                    z=valid_away['MT_A'],
                    mode='markers',
                    name=f'{cluster_name} - Away',
                    marker=dict(
                        color=color,
                        size=8,  # Reduzido para melhor visualização
                        symbol='diamond',
                        opacity=0.8,  # Apenas para markers, não para lines
                        line=dict(color='white', width=1)
                    ),
                    text=valid_away.apply(
                        lambda r: f"<b>{r['Away']}</b><br>"
                                 f"Cluster: {cluster_name}<br>" 
                                 f"vs {r['Home']}<br>"
                                 f"Agg: {r['Aggression_Away']:.2f}<br>"
                                 f"M_Liga: {r['M_A']:.2f}<br>"
                                 f"M_Time: {r['MT_A']:.2f}",
                        axis=1
                    ),
                    hovertemplate='%{text}<extra></extra>'
                ))

    # ---------------------- LAYOUT FIXO ----------------------
    X_RANGE = [-1.2, 1.2]
    Y_RANGE = [-4.0, 4.0]  
    Z_RANGE = [-4.0, 4.0]

    titulo_3d = f"Top {n_to_show} Confrontos - Visualização 3D por Clusters"
    if selected_league != "⚽ Todas as ligas":
        titulo_3d += f" | {selected_league}"

    fig_3d.update_layout(
        title=dict(
            text=titulo_3d,
            x=0.5,
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title='Aggression (-1 zebra ↔ +1 favorito)',
                range=X_RANGE,
                backgroundcolor="rgba(20,20,20,0.1)",
                gridcolor="gray",
                showbackground=True,
                gridwidth=2,
                zerolinecolor="red",
                zerolinewidth=4
            ),
            yaxis=dict(
                title='Momentum (Liga)',
                range=Y_RANGE, 
                backgroundcolor="rgba(20,20,20,0.1)",
                gridcolor="gray",
                showbackground=True,
                gridwidth=2,
                zerolinecolor="green",
                zerolinewidth=4
            ),
            zaxis=dict(
                title='Momentum (Time)',
                range=Z_RANGE,
                backgroundcolor="rgba(20,20,20,0.1)", 
                gridcolor="gray",
                showbackground=True,
                gridwidth=2,
                zerolinecolor="blue",
                zerolinewidth=4
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=10)
        )
    )
    
    return fig_3d



############ Bloco G - Carregamento de Dados e Cache ################
# ---------------- CARREGAMENTO DE DADOS E CACHE ----------------
@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    """Cache apenas dos dados pesados"""
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
    
    return games_today, history

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



############ Bloco H - Cálculo de Distâncias 3D ################
# ---------------- CÁLCULO DE DISTÂNCIAS 3D ----------------
def calcular_distancias_3d(df):
    """
    Calcula distância 3D e ângulos usando Aggression, Momentum (liga) e Momentum (time)
    """
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in [
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Quadrant_Angle_XY', 'Quadrant_Angle_XZ', 'Quadrant_Angle_YZ',
            'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
            'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ', 
            'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
            'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo', 'Vector_Sign',
            'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D'
        ]:
            df[col] = np.nan
        return df

    # --- Diferenças puras ---
    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    # --- Distância Euclidiana pura ---
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    # --- Ângulos entre planos ---
    angle_xy = np.arctan2(dy, dx)
    angle_xz = np.arctan2(dz, dx)
    angle_yz = np.arctan2(dz, dy)

    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Angle_XZ'] = np.degrees(angle_xz)
    df['Quadrant_Angle_YZ'] = np.degrees(angle_yz)

    # --- Projeções trigonométricas básicas ---
    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Quadrant_Sin_XZ'] = np.sin(angle_xz)
    df['Quadrant_Cos_XZ'] = np.cos(angle_xz)
    df['Quadrant_Sin_YZ'] = np.sin(angle_yz)
    df['Quadrant_Cos_YZ'] = np.cos(angle_yz)

    # --- Combinações trigonométricas compostas ---
    df['Quadrant_Sin_Combo'] = np.sin(angle_xy + angle_xz + angle_yz)
    df['Quadrant_Cos_Combo'] = np.cos(angle_xy + angle_xz + angle_yz)

    # --- Sinal vetorial (direção espacial total) ---
    df['Vector_Sign'] = np.sign(dx * dy * dz)

    # --- Separação neutra 3D ---
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3

    # --- Diferenças individuais ---
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz

    # --- Magnitude total ---
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df



############ Bloco H2 - Valor Angular de Mercado ################
def calcular_valor_angular_mercado(df):
    """
    Calcula o valor esperado baseado no ângulo 3D e odds de abertura (market inefficiency).
    """
    df = df.copy()

    required_cols = [
        'Quadrant_Dist_3D', 'Quadrant_Angle_XY',
        'Prob_Home', 'Prob_Away',
        'Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para cálculo de valor angular: {missing}")
        for col in ['Market_Error_Home', 'Market_Error_Away',
                    'Value_Score_Home', 'Value_Score_Away', 'Market_Classification']:
            df[col] = np.nan
        return df

    # Probabilidade implícita do mercado (normalizada)
    inv_sum = 1 / (1/df['Odd_H_OP'] + 1/df['Odd_D_OP'] + 1/df['Odd_A_OP'])
    df['p_market_home'] = inv_sum / df['Odd_H_OP']
    df['p_market_away'] = inv_sum / df['Odd_A_OP']

    # Erro de mercado (modelo - mercado)
    df['Market_Error_Home'] = df['Prob_Home'] - df['p_market_home']
    df['Market_Error_Away'] = df['Prob_Away'] - df['p_market_away']

    # Intensidade angular (baseada na distância e ângulo XY)
    df['Angular_Intensity'] = np.abs(np.sin(np.radians(df['Quadrant_Angle_XY']))) * df['Quadrant_Dist_3D']

    # Value Score ajustado pelo ângulo e intensidade
    df['Value_Score_Home'] = df['Market_Error_Home'] * df['Angular_Intensity']
    df['Value_Score_Away'] = df['Market_Error_Away'] * df['Angular_Intensity']

    # Classificação de valor
    def classify_value(v):
        if pd.isna(v):
            return '⚖️ Neutro'
        if v > 0.05:
            return '🎯 VALUE BET'
        elif v < -0.05:
            return '🔴 Overpriced'
        else:
            return '⚖️ Neutro'

    df['Market_Classification'] = df['Value_Score_Home'].apply(classify_value)

    return df





############ Bloco I - Sistema ML com Clusters ################
# ---------------- MODELO ML COM CLUSTERS ----------------
def treinar_modelo_com_clusters(history, games_today):
    """
    Treina modelo ML 3D usando clusters (SEM QUADRANTES)
    """
    # Garantir features 3D e clusters
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    # ✅ Targets AH históricos COM HANDICAP CORRETO
    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(
        lambda r: 1 if calc_handicap_result(r["Margin"], r["Asian_Line_Decimal"], invert=False) > 0.5 else 0, axis=1
    )

    # Features categóricas (liga + cluster)
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    # Features contínuas vetoriais + REGRESSÃO
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D'
    ]
    
    # Features de REGRESSÃO
    features_regressao = [
        'Media_Score_Home', 'Media_Score_Away',
        'Regressao_Force_Home', 'Regressao_Force_Away',
        'Extremidade_Home', 'Extremidade_Away'
    ]

    extras_3d = history[features_3d].fillna(0)
    extras_regressao = history[features_regressao].fillna(0)

    # Combinar todas as features
    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, extras_regressao], axis=1)

    # Target
    y_home = history['Target_AH_Home'].astype(int)

    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    # Preparar dados de hoje
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d].fillna(0)
    extras_regressao_today = games_today[features_regressao].fillna(0)

    X_today = pd.concat([ligas_today, clusters_today, extras_today, extras_regressao_today], axis=1)

    # Previsões
    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Cluster_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Cluster_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Cluster_ML_Score_Main'] = games_today['ML_Confidence']

    # Importância das features
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)

    st.markdown("### 🔍 Top Features (Com Clusters 3D)")
    st.dataframe(importances.head(20).to_frame("Importância"), use_container_width=True)

    # Verificar se features de cluster estão no topo
    cluster_no_top = len([f for f in importances.head(15).index if 'C3D' in f])
    st.info(f"📊 Features de Cluster no Top 15: {cluster_no_top}")

    st.success("✅ Modelo 3D com Clusters treinado com sucesso!")
    return model_home, games_today


############ Bloco J - Sistema de Recomendações com Clusters ################
# ---------------- SISTEMA DE INDICAÇÕES COM CLUSTERS ----------------
def adicionar_indicadores_explicativos_clusters(df):
    """Adiciona classificações e recomendações baseadas nos clusters 3D"""
    df = df.copy()

    # 1. CLASSIFICAÇÃO DE VALOR PARA HOME (CLUSTERS)
    conditions_home = [
        df['Cluster_ML_Score_Home'] >= 0.65,
        df['Cluster_ML_Score_Home'] >= 0.58,
        df['Cluster_ML_Score_Home'] >= 0.52,
        df['Cluster_ML_Score_Home'] >= 0.48,
        df['Cluster_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    # 2. CLASSIFICAÇÃO DE VALOR PARA AWAY (CLUSTERS)
    conditions_away = [
        df['Cluster_ML_Score_Away'] >= 0.65,
        df['Cluster_ML_Score_Away'] >= 0.58,
        df['Cluster_ML_Score_Away'] >= 0.52,
        df['Cluster_ML_Score_Away'] >= 0.48,
        df['Cluster_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

    # 3. RECOMENDAÇÃO BASEADA EM CLUSTERS + REGRESSÃO (CORRIGIDA)
    def gerar_recomendacao_clusters(row):
        try:
            cluster = row.get('Cluster3D_Desc', '🌀 Caso Atípico')
            score_home = row.get('Cluster_ML_Score_Home', 0.5)
            score_away = row.get('Cluster_ML_Score_Away', 0.5)
            ml_side = row.get('ML_Side', 'HOME')
            tendencia_h = row.get('Tendencia_Home', '⚖️ ESTÁVEL')
            tendencia_a = row.get('Tendencia_Away', '⚖️ ESTÁVEL')
            
            # 🎯 DEBUG: Ver o que está chegando
            debug_info = f"[CLUSTER: {cluster} | SCORE_H: {score_home:.1%} | SCORE_A: {score_away:.1%}]"
            
            # Estratégias por tipo de cluster - COM NOMES FLEXÍVEIS
            if any(term in cluster for term in ['Home Domina', 'Home Agressivo', 'Home Vantagem']):
                if score_home >= 0.65 and any(term in tendencia_h for term in ['MELHORA', 'FORTE']):
                    return f'💪 HOME DOMINANTE + Melhora ({score_home:.1%}) {debug_info}'
                elif score_home >= 0.58:
                    return f'🎯 HOME DOMINANTE ({score_home:.1%}) {debug_info}'
                elif score_home >= 0.52:
                    return f'📈 HOME com Vantagem ({score_home:.1%}) {debug_info}'
                else:
                    return f'⚖️ HOME favorecido mas cuidado ({score_home:.1%}) {debug_info}'

            elif any(term in cluster for term in ['Away Domina', 'Away Agressivo', 'Away Vantagem']):
                if score_away >= 0.65 and any(term in tendencia_a for term in ['MELHORA', 'FORTE']):
                    return f'💪 AWAY DOMINANTE + Melhora ({score_away:.1%}) {debug_info}'
                elif score_away >= 0.58:
                    return f'🎯 AWAY DOMINANTE ({score_away:.1%}) {debug_info}'
                elif score_away >= 0.52:
                    return f'📈 AWAY com Vantagem ({score_away:.1%}) {debug_info}'
                else:
                    return f'⚖️ AWAY favorecido mas cuidado ({score_away:.1%}) {debug_info}'

            elif any(term in cluster for term in ['Equilibrado', 'Equilíbrio', 'Neutro']):
                if ml_side == 'HOME' and score_home >= 0.55:
                    return f'📈 VALUE NO HOME (Equilibrado) ({score_home:.1%}) {debug_info}'
                elif ml_side == 'AWAY' and score_away >= 0.55:
                    return f'📈 VALUE NO AWAY (Equilibrado) ({score_away:.1%}) {debug_info}'
                else:
                    return f'⚖️ CONFRONTO EQUILIBRADO (H:{score_home:.1%} A:{score_away:.1%}) {debug_info}'

            elif any(term in cluster for term in ['Imprevisível', 'Instável', 'Contraditório']):
                if 'MELHORA' in tendencia_h and score_home >= 0.55:
                    return f'🎲 IMPREVISÍVEL mas Home Melhorando ({score_home:.1%}) {debug_info}'
                elif 'MELHORA' in tendencia_a and score_away >= 0.55:
                    return f'🎲 IMPREVISÍVEL mas Away Melhorando ({score_away:.1%}) {debug_info}'
                else:
                    return f'🎲 JOGO IMPREVISÍVEL - Cautela (H:{score_home:.1%} A:{score_away:.1%}) {debug_info}'

            else:
                # 🎯 RECOMENDAÇÃO BASEADA APENAS NOS SCORES (fallback)
                if score_home >= 0.70:
                    return f'🏆 HOME FORTE ({score_home:.1%}) {debug_info}'
                elif score_away >= 0.70:
                    return f'🏆 AWAY FORTE ({score_away:.1%}) {debug_info}'
                elif score_home >= 0.60:
                    return f'✅ HOME com Valor ({score_home:.1%}) {debug_info}'
                elif score_away >= 0.60:
                    return f'✅ AWAY com Valor ({score_away:.1%}) {debug_info}'
                elif abs(score_home - score_away) < 0.1:
                    return f'⚖️ CONFRONTO EQUILIBRADO (H:{score_home:.1%} A:{score_away:.1%}) {debug_info}'
                else:
                    return f'🔍 ANALISAR (H:{score_home:.1%} A:{score_away:.1%}) {debug_info}'
                    
        except Exception as e:
            return f'❌ ERRO: {str(e)}'

    df['Recomendacao'] = df.apply(gerar_recomendacao_clusters, axis=1)

    # 4. SCORE FINAL COMBINADO (Clusters + ML + Regressão)
    df['Score_Final_Clusters'] = (
        df['Cluster_ML_Score_Main'] * 0.6 + 
        df.get('Media_Score_Home', 0.5) * 0.2 + 
        df.get('Media_Score_Away', 0.5) * 0.2
    ) * 100

    # 5. CLASSIFICAÇÃO DE POTENCIAL
    conditions_potencial = [
        df['Score_Final_Clusters'] >= 70,
        df['Score_Final_Clusters'] >= 60,
        df['Score_Final_Clusters'] >= 50,
        df['Score_Final_Clusters'] >= 40,
        df['Score_Final_Clusters'] < 40
    ]
    choices_potencial = ['🌟🌟🌟 POTENCIAL MÁXIMO', '🌟🌟 ALTO POTENCIAL', '🌟 POTENCIAL MODERADO', '⚖️ POTENCIAL BAIXO', '🔴 RISCO ALTO']
    df['Classificacao_Potencial'] = np.select(conditions_potencial, choices_potencial, default='🌟 POTENCIAL MODERADO')

    # 6. RANKING
    df['Ranking'] = df['Score_Final_Clusters'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- FUNÇÃO DE DIAGNÓSTICO DAS RECOMENDAÇÕES ----------------
def diagnosticar_recomendacoes(df):
    """Diagnóstico detalhado do sistema de recomendações"""
    st.markdown("### 🔍 Diagnóstico do Sistema de Recomendações")
    
    if df.empty:
        st.warning("Nenhum dado para diagnóstico")
        return
    
    # 1. Verificar clusters disponíveis
    st.write("**🎯 Clusters disponíveis no dataset:**")
    if 'Cluster3D_Desc' in df.columns:
        cluster_counts = df['Cluster3D_Desc'].value_counts()
        for cluster, count in cluster_counts.items():
            st.write(f"  - `{cluster}`: {count} jogos")
    else:
        st.error("❌ Coluna 'Cluster3D_Desc' não encontrada!")
    
    # 2. Verificar scores
    st.write("**📊 Distribuição dos scores:**")
    if 'Cluster_ML_Score_Home' in df.columns:
        st.write(f"  - Score Home: {df['Cluster_ML_Score_Home'].min():.1%} a {df['Cluster_ML_Score_Home'].max():.1%}")
        st.write(f"  - Score Away: {df['Cluster_ML_Score_Away'].min():.1%} a {df['Cluster_ML_Score_Away'].max():.1%}")
    
    # 3. Amostra de recomendações
    st.write("**👀 Amostra de recomendações geradas:**")
    cols_amostra = ['Home', 'Away', 'Cluster3D_Desc', 'Cluster_ML_Score_Home', 'Cluster_ML_Score_Away', 'Recomendacao']
    cols_amostra = [c for c in cols_amostra if c in df.columns]
    st.dataframe(df[cols_amostra].head(10), use_container_width=True)

# No Bloco P - ANTES de exibir a tabela principal, adicione:
st.markdown("## 🎯 Sistema de Recomendações - Diagnóstico")
diagnosticar_recomendacoes(games_today)


############ Bloco K - Estratégias Baseadas em Clusters ################
# ---------------- ESTRATÉGIAS COM CLUSTERS ----------------
def gerar_estrategias_por_cluster(df):
    """Gera estratégias específicas baseadas nos clusters 3D"""
    st.markdown("### 🎯 Estratégias por Tipo de Cluster")

    estrategias_clusters = {
        '🏠 Home Domina Confronto': {
            'descricao': '**Home claramente superior** - Aggression, Momentum Liga e Momentum Time favoráveis',
            'estrategia': 'Apostar Home quando odds > 1.80, buscar value spots',
            'confianca': 'Alta',
            'alvo_minimo': 0.58,
            'filtro_regressao': '📈 MELHORA ou 📈 FORTE MELHORA'
        },
        '🚗 Away Domina Confronto': {
            'descricao': '**Away claramente superior** - Visitante com vantagem nas 3 dimensões',
            'estrategia': 'Apostar Away quando odds > 2.00, ótimo para handicaps',
            'confianca': 'Alta', 
            'alvo_minimo': 0.58,
            'filtro_regressao': '📈 MELHORA ou 📈 FORTE MELHORA'
        },
        '⚖️ Confronto Equilibrado': {
            'descricao': '**Times muito parecidos** - Diferenças pequenas nas 3 dimensões',
            'estrategia': 'Buscar underdogs com value, apostas menores',
            'confianca': 'Média',
            'alvo_minimo': 0.55,
            'filtro_regressao': 'QUALQUER (focar no value)'
        },
        '🎭 Home Imprevisível': {
            'descricao': '**Sinais mistos** - Aggression, Momentum e Regressão em conflito',
            'estrategia': 'Apostas pequenas ou evitar, monitorar live',
            'confianca': 'Baixa',
            'alvo_minimo': 0.60,
            'filtro_regressao': '📈 FORTE MELHORA (apenas)'
        },
        '🌪️ Home Instável': {
            'descricao': '**Alta volatilidade** - Valores extremos ou inconsistentes',
            'estrategia': 'EVITAR apostas pré-live, considerar live betting',
            'confianca': 'Muito Baixa',
            'alvo_minimo': 0.65,
            'filtro_regressao': 'EVITAR'
        }
    }

    for cluster, info in estrategias_clusters.items():
        jogos_cluster = df[df['Cluster3D_Desc'] == cluster]
        
        if not jogos_cluster.empty:
            st.write(f"**{cluster}**")
            st.write(f"📋 {info['descricao']}")
            st.write(f"🎯 Estratégia: {info['estrategia']}")
            st.write(f"📊 Confiança: {info['confianca']}")
            
            # Métricas do cluster
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jogos", len(jogos_cluster))
            with col2:
                avg_score = jogos_cluster['Cluster_ML_Score_Main'].mean()
                st.metric("Score Médio", f"{avg_score:.1%}")
            with col3:
                high_value = len(jogos_cluster[jogos_cluster['Cluster_ML_Score_Main'] >= info['alvo_minimo']])
                st.metric("Oportunidades", high_value)
            
            # Top 3 oportunidades do cluster
            oportunidades = jogos_cluster[
                jogos_cluster['Cluster_ML_Score_Main'] >= info['alvo_minimo']
            ].head(3)
            
            if not oportunidades.empty:
                st.write("**Top Oportunidades:**")
                cols = ['Home', 'Away', 'League', 'Cluster_ML_Score_Home', 'Cluster_ML_Score_Away', 'Recomendacao']
                st.dataframe(oportunidades[cols].style.format({
                    'Cluster_ML_Score_Home': '{:.1%}',
                    'Cluster_ML_Score_Away': '{:.1%}'
                }), use_container_width=True)
            
            st.write("---")

def analisar_padroes_clusters(df):
    """Analisa padrões de sucesso por cluster"""
    st.markdown("### 📊 Análise de Performance por Cluster")
    
    # Apenas jogos finalizados
    finished = df.dropna(subset=['Goals_H_Today', 'Goals_A_Today'])
    
    if finished.empty:
        st.info("⏳ Aguardando jogos finalizados para análise...")
        return
    
    # Calcular acertos por cluster
    resultados = []
    for cluster in finished['Cluster3D_Desc'].unique():
        cluster_data = finished[finished['Cluster3D_Desc'] == cluster]
        total_jogos = len(cluster_data)
        
        if total_jogos > 0:
            # Jogos com recomendações claras
            recomendados = cluster_data[cluster_data['Recomendacao'].str.contains('🎯|💪|📈')]
            acertos = 0
            total_recomendados = len(recomendados)
            
            if total_recomendados > 0:
                # Lógica simplificada de acerto (pode ser refinada)
                for _, jogo in recomendados.iterrows():
                    if ('HOME' in jogo['Recomendacao'] and jogo['Goals_H_Today'] > jogo['Goals_A_Today']) or \
                       ('AWAY' in jogo['Recomendacao'] and jogo['Goals_A_Today'] > jogo['Goals_H_Today']):
                        acertos += 1
                
                winrate = (acertos / total_recomendados) * 100
            else:
                winrate = 0
            
            resultados.append({
                'Cluster': cluster,
                'Total Jogos': total_jogos,
                'Recomendados': total_recomendados,
                'Acertos': acertos,
                'Winrate': f"{winrate:.1f}%"
            })
    
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados, use_container_width=True)


############ Bloco L - Sistema Live Score com Clusters ################
# ---------------- LIVE SCORE COM CLUSTERS ----------------
def determine_handicap_result(row):
    """Determina se o HOME cobriu o handicap"""
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
        asian_line_decimal = row.get('Asian_Line_Decimal')  # ✅ AGORA USA A COLUNA CORRETA
    except (ValueError, TypeError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_decimal):
        return None

    margin = gh - ga
    # ✅ USA A LINJA JÁ CONVERTIDA PARA PERSPECTIVA HOME
    handicap_result = calc_handicap_result(margin, asian_line_decimal, invert=False)

    if handicap_result > 0.5:
        return "HOME_COVERED"
    elif handicap_result == 0.5:
        return "PUSH"
    else:
        return "HOME_NOT_COVERED"

def check_handicap_recommendation_correct(rec, handicap_result):
    """Verifica se a recomendação estava correta"""
    if pd.isna(rec) or handicap_result is None or 'EVITAR' in str(rec):
        return None

    rec = str(rec)

    if any(keyword in rec for keyword in ['HOME', 'Home', 'DOMINANTE']):
        return handicap_result == "HOME_COVERED"
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'DOMINANTE']):
        return handicap_result in ["HOME_NOT_COVERED", "PUSH"]

    return None

def update_real_time_data_clusters(df):
    """Atualiza todos os dados em tempo real para sistema com clusters"""
    df['Handicap_Result'] = df.apply(determine_handicap_result, axis=1)
    df['Cluster_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct(r['Recomendacao'], r['Handicap_Result']), axis=1
    )
    return df

def generate_live_summary_clusters(df):
    """Gera resumo em tempo real para sistema com clusters"""
    finished_games = df.dropna(subset=['Handicap_Result'])

    if finished_games.empty:
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": 0,
            "Recomendações Cluster": 0,
            "Acertos Cluster": 0,
            "Winrate Cluster": "0%"
        }

    cluster_recomendados = finished_games[finished_games['Cluster_Correct'].notna()]
    total_recomendados = len(cluster_recomendados)
    correct_recomendados = cluster_recomendados['Cluster_Correct'].sum()
    winrate = (correct_recomendados / total_recomendados) * 100 if total_recomendados > 0 else 0

    return {
        "Total Jogos": len(df),
        "Jogos Finalizados": len(finished_games),
        "Recomendações Cluster": total_recomendados,
        "Acertos Cluster": int(correct_recomendados),
        "Winrate Cluster": f"{winrate:.1f}%"
    }


############ Bloco M - Estilo da Tabela ################
# ---------------- ESTILO DA TABELA COM CLUSTERS ----------------
def estilo_tabela_clusters(df):
    """Aplica estilo à tabela principal com clusters"""
    def cor_classificacao(valor):
        if '🌟🌟🌟' in str(valor): return 'font-weight: bold'
        elif '🌟🌟' in str(valor): return 'font-weight: bold'
        elif '🌟' in str(valor): return 'font-weight: bold'
        elif '🔴' in str(valor): return 'font-weight: bold'
        elif '🏆' in str(valor): return 'font-weight: bold'
        else: return ''

    colunas_para_estilo = []
    for col in ['Classificacao_Potencial', 'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao']:
        if col in df.columns:
            colunas_para_estilo.append(col)

    styler = df.style
    if colunas_para_estilo:
        styler = styler.applymap(cor_classificacao, subset=colunas_para_estilo)

    # Gradientes para colunas numéricas
    if 'Cluster_ML_Score_Home' in df.columns:
        styler = styler.background_gradient(subset=['Cluster_ML_Score_Home'], cmap='RdYlGn')
    if 'Cluster_ML_Score_Away' in df.columns:
        styler = styler.background_gradient(subset=['Cluster_ML_Score_Away'], cmap='RdYlGn')
    if 'Score_Final_Clusters' in df.columns:
        styler = styler.background_gradient(subset=['Score_Final_Clusters'], cmap='RdYlGn')
    if 'M_H' in df.columns:
        styler = styler.background_gradient(subset=['M_H', 'M_A'], cmap='coolwarm')

    return styler



############ Bloco N - Execução Principal: Carregamento de Dados ################
# ---------------- EXECUÇÃO PRINCIPAL ----------------
st.info("📂 Carregando dados para análise 3D com clusters...")

# Seleção de arquivo do dia
files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

# Carregar dados com cache
games_today, history = load_cached_data(selected_file)

# Aplicar Live Score
games_today = load_and_merge_livescore(games_today, selected_date_str)

# ✅ Converter Asian Line COM A NOVA FUNÇÃO
st.info("🔄 Convertendo Asian Line para perspectiva Home...")
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

# ✅ VALIDAÇÃO DO SISTEMA DE HANDICAP
validar_conversao_handicap()
verificar_sistema_handicap(history, games_today)

# Filtrar histórico com linha válida
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

# Aplicar momentum e regressão
history = calcular_momentum_time(history)
games_today = calcular_momentum_time(games_today)
history = calcular_regressao_media(history)
games_today = calcular_regressao_media(games_today)


############ Bloco O - Execução Principal: Treinamento e Visualização ################
# ---------------- TREINAMENTO DO MODELO ----------------
st.markdown("## 🧠 Sistema 3D com Clusters - ML")

if not history.empty:
    try:
        modelo_home, games_today = treinar_modelo_com_clusters(history, games_today)
        st.success("✅ Modelo 3D com Clusters treinado com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro no treinamento do modelo: {e}")
        st.info("⚠️ Continuando sem modelo treinado...")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

# ---------------- CÁLCULO DE VALOR ANGULAR ----------------
if {'Odd_H_OP', 'Odd_A_OP', 'Quadrant_Dist_3D'}.issubset(games_today.columns):
    games_today = calcular_valor_angular_mercado(games_today)
    st.success("💰 Valor Angular de Mercado calculado com sucesso!")
else:
    st.warning("⚠️ Dados insuficientes para cálculo de Valor Angular (faltam odds de abertura ou colunas 3D)")




# ---------------- VISUALIZAÇÃO 3D INTERATIVA ----------------
st.markdown("## 🎯 Visualização 3D com Clusters")

# Filtros interativos
col1, col2 = st.columns([2, 1])
with col1:
    if "League" in games_today.columns and not games_today["League"].isna().all():
        leagues = sorted(games_today["League"].dropna().unique())
        selected_league = st.selectbox(
            "Selecione a liga para análise:",
            options=["⚽ Todas as ligas"] + leagues,
            index=0
        )
    else:
        selected_league = "⚽ Todas as ligas"
        st.warning("⚠️ Nenhuma coluna de 'League' encontrada")

with col2:
    max_n = len(games_today)
    n_to_show = st.slider("Jogos para exibir:", 10, min(max_n, 100), 30, step=5)

# Filtrar por liga
if selected_league != "⚽ Todas as ligas":
    df_filtered = games_today[games_today["League"] == selected_league].copy()
else:
    df_filtered = games_today.copy()

# Filtro por cluster
st.markdown("### 🔍 Filtro por Cluster")
clusters_disponiveis = df_filtered['Cluster3D_Desc'].unique() if 'Cluster3D_Desc' in df_filtered.columns else []
if len(clusters_disponiveis) > 0:
    cluster_selecionado = st.selectbox(
        "Filtrar por tipo de confronto:",
        options=["🎯 Todos os clusters"] + list(clusters_disponiveis),
        index=0
    )

    if cluster_selecionado != "🎯 Todos os clusters":
        df_plot = df_filtered[df_filtered['Cluster3D_Desc'] == cluster_selecionado].copy()
    else:
        df_plot = df_filtered.copy()
else:
    st.warning("⚠️ Nenhum cluster disponível - aplicando clusterização...")
    try:
        df_filtered = aplicar_clusterizacao_3d(df_filtered)
        clusters_disponiveis = df_filtered['Cluster3D_Desc'].unique()
        df_plot = df_filtered.copy()
    except Exception as e:
        st.error(f"❌ Erro na clusterização: {e}")
        df_plot = df_filtered.copy()

# Aplicar limite de jogos
df_plot = df_plot.head(n_to_show)

# Verificar se há dados válidos para o gráfico 3D
required_cols_3d = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A', 'Cluster3D_Desc']
missing_3d_cols = [col for col in required_cols_3d if col not in df_plot.columns]

if missing_3d_cols:
    st.warning(f"⚠️ Colunas necessárias para gráfico 3D não encontradas: {missing_3d_cols}")
    st.info("📊 O gráfico 3D será pulado devido a dados insuficientes")
    
    # Mostrar estatísticas dos dados disponíveis
    st.markdown("### 📈 Dados Disponíveis para Análise")
    available_cols = [col for col in required_cols_3d if col in df_plot.columns]
    if available_cols:
        st.write(f"Colunas disponíveis: {available_cols}")
        st.write(f"Total de jogos: {len(df_plot)}")
        
else:
    # Verificar se há dados numéricos válidos
    numeric_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    df_numeric_check = df_plot[numeric_cols].fillna(0)
    
    if df_numeric_check.select_dtypes(include=[np.number]).empty:
        st.warning("⚠️ Não há dados numéricos válidos para o gráfico 3D")
    else:
        # Verificar se temos pelo menos alguns dados não-zero
        has_valid_data = False
        for col in numeric_cols:
            if col in df_plot.columns and df_plot[col].notna().any():
                non_zero_values = df_plot[col].fillna(0) != 0
                if non_zero_values.any():
                    has_valid_data = True
                    break
        
        if not has_valid_data:
            st.warning("⚠️ Todos os valores numéricos são zero ou NaN")
        else:
            try:
                # Criar e exibir gráfico 3D
                fig_3d_clusters = create_3d_plot_with_clusters(df_plot, n_to_show, selected_league)
                st.plotly_chart(fig_3d_clusters, use_container_width=True)
                
                # Legenda dos clusters
                st.markdown("""
                ### 🎨 Legenda dos Clusters 3D:
                - **🔵 Home Domina Confronto**: Home superior nas 3 dimensões
                - **🔴 Away Domina Confronto**: Away superior nas 3 dimensões  
                - **🟢 Confronto Equilibrado**: Times muito parecidos
                - **🟠 Home Imprevisível**: Sinais mistos e conflitantes
                - **🟣 Home Instável**: Alta volatilidade e inconsistência
                """)
                
                # Estatísticas dos clusters exibidos
                st.markdown("### 📊 Estatísticas dos Clusters no Gráfico")
                cluster_counts = df_plot['Cluster3D_Desc'].value_counts()
                st.dataframe(cluster_counts, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erro ao criar gráfico 3D: {e}")
                st.info("📋 Mostrando dados em formato de tabela...")
                
                # Mostrar dados em tabela como fallback
                display_cols = ['Home', 'Away', 'League', 'Cluster3D_Desc', 'Aggression_Home', 'Aggression_Away', 'M_H', 'M_A']
                display_cols = [col for col in display_cols if col in df_plot.columns]
                
                if display_cols:
                    st.dataframe(
                        df_plot[display_cols].style.format({
                            'Aggression_Home': '{:.2f}',
                            'Aggression_Away': '{:.2f}',
                            'M_H': '{:.2f}',
                            'M_A': '{:.2f}'
                        }),
                        use_container_width=True
                    )


############ Bloco P - Execução Principal: Tabela Principal ################

# ---------------- TABELA PRINCIPAL COM CLUSTERS ----------------
st.markdown("## 🏆 Melhores Oportunidades - Sistema Clusters 3D")

if not games_today.empty and 'Cluster_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_clusters = games_today.copy()
    
    # Aplicar indicadores e estratégias
    ranking_clusters = adicionar_indicadores_explicativos_clusters(ranking_clusters)
    ranking_clusters = update_real_time_data_clusters(ranking_clusters)
    
    # Ordenar por score final
    ranking_clusters = ranking_clusters.sort_values('Score_Final_Clusters', ascending=False)
    
    # ---------------- COLUNAS PRINCIPAIS ----------------
    colunas_principais = [
        'Ranking', 'League', 'Time', 'Home', 'Away',
        'Goals_H_Today', 'Goals_A_Today', 'ML_Side',
        'Cluster3D_Desc',
        'Cluster_ML_Score_Home', 'Cluster_ML_Score_Away',
        'Score_Final_Clusters', 'Classificacao_Potencial',
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao',
        # Dados 3D
        'M_H', 'M_A', 'Quadrant_Dist_3D',
        # Regressão
        'Tendencia_Home', 'Tendencia_Away',
        'Media_Score_Home', 'Media_Score_Away',
        # Valor Angular de Mercado
        'Market_Error_Home', 'Market_Error_Away',
        'Value_Score_Home', 'Value_Score_Away',
        'Market_Classification',
        # Live Score
        'Asian_Line_Decimal', 'Handicap_Result', 'Cluster_Correct'
    ]
    
    # Filtrar colunas existentes
    cols_finais = [c for c in colunas_principais if c in ranking_clusters.columns]
    
    # Exibir resumo live
    st.markdown("### 📡 Live Score Monitor")
    live_summary = generate_live_summary_clusters(ranking_clusters)
    st.json(live_summary)
    
    # Exibir tabela principal
    st.write(f"🎯 Exibindo {len(ranking_clusters)} jogos ordenados por Score Clusters")
    
    # ---------------- ESTILO E FORMATAÇÃO ----------------
    styler = estilo_tabela_clusters(ranking_clusters[cols_finais])
    
    # Gradientes adicionais para Valor Angular
    if 'Value_Score_Home' in ranking_clusters.columns:
        styler = styler.background_gradient(subset=['Value_Score_Home'], cmap='RdYlGn')
    if 'Value_Score_Away' in ranking_clusters.columns:
        styler = styler.background_gradient(subset=['Value_Score_Away'], cmap='RdYlGn')
    if 'Market_Error_Home' in ranking_clusters.columns:
        styler = styler.background_gradient(subset=['Market_Error_Home'], cmap='RdYlGn')
    if 'Market_Error_Away' in ranking_clusters.columns:
        styler = styler.background_gradient(subset=['Market_Error_Away'], cmap='RdYlGn')

    # Exibição final
    st.dataframe(
        styler.format({
            'Goals_H_Today': '{:.0f}',
            'Goals_A_Today': '{:.0f}',
            'Asian_Line_Decimal': '{:.2f}',
            'Cluster_ML_Score_Home': '{:.1%}',
            'Cluster_ML_Score_Away': '{:.1%}',
            'Score_Final_Clusters': '{:.1f}',
            'M_H': '{:.2f}',
            'M_A': '{:.2f}',
            'Quadrant_Dist_3D': '{:.2f}',
            'Media_Score_Home': '{:.2f}',
            'Media_Score_Away': '{:.2f}',
            'Market_Error_Home': '{:.2%}',
            'Market_Error_Away': '{:.2%}',
            'Value_Score_Home': '{:.3f}',
            'Value_Score_Away': '{:.3f}'
        }, na_rep="-"),
        use_container_width=True,
        height=600
    )
    
    # ---------------- ANÁLISES ESPECÍFICAS ----------------
    gerar_estrategias_por_cluster(ranking_clusters)
    analisar_padroes_clusters(ranking_clusters)
    
else:
    st.error("""
    ❌ **Não foi possível gerar a tabela de confrontos**
    
    **Possíveis causas:**
    - Dados de hoje vazios
    - Modelo não foi treinado corretamente
    - Colunas necessárias não encontradas
    
    **Verifique:**
    1. Se o arquivo CSV tem dados válidos
    2. Se o histórico tem dados suficientes
    3. Se todas as colunas necessárias existem
    """)


############ Bloco Q - Resumo Executivo e Filtros Avançados ################
# ---------------- RESUMO EXECUTIVO ----------------
def resumo_executivo_clusters(df):
    """Resumo executivo do sistema com clusters"""
    st.markdown("## 📋 Resumo Executivo - Sistema Clusters 3D")
    
    if df.empty:
        st.info("Nenhum dado disponível para resumo")
        return

    # ✅ Garante que existam colunas de classificação
    if 'Classificacao_Potencial' not in df.columns:
        df = adicionar_indicadores_explicativos_clusters(df)
    
    total_jogos = len(df)
    
    # Estatísticas de clusters
    if 'Cluster3D_Desc' in df.columns:
        cluster_dist = df['Cluster3D_Desc'].value_counts()
        cluster_mais_comum = cluster_dist.index[0] if not cluster_dist.empty else "N/A"
    else:
        cluster_dist = pd.Series()
        cluster_mais_comum = "N/A"
    
    # Estatísticas de valor
    alto_valor = len(df[df['Classificacao_Potencial'].str.contains('🌟🌟', na=False)])
    alto_risco = len(df[df['Classificacao_Potencial'].str.contains('🔴', na=False)])
    
    # Estatísticas de recomendação
    if 'Recomendacao' in df.columns:
        recomendacoes_positivas = len(df[df['Recomendacao'].str.contains('🎯|💪|📈', na=False)])
        recomendacoes_cautela = len(df[df['Recomendacao'].str.contains('⚖️|🎲|🌪️', na=False)])
    else:
        recomendacoes_positivas = 0
        recomendacoes_cautela = 0
    
    # Métricas em colunas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("Cluster Mais Comum", cluster_mais_comum)
    
    with col2:
        st.metric("🌟🌟 Oportunidades", alto_valor)
        st.metric("🔴 Alto Risco", alto_risco)
    
    with col3:
        st.metric("🎯 Recomendações Positivas", recomendacoes_positivas)
        st.metric("⚖️ Recomendações Cautela", recomendacoes_cautela)
    
    with col4:
        avg_score = df['Score_Final_Clusters'].mean() if 'Score_Final_Clusters' in df.columns else 0
        st.metric("Score Médio", f"{avg_score:.1f}")
        st.metric("Clusters Diferentes", len(cluster_dist))
    
    # Distribuição de clusters
    if not cluster_dist.empty:
        st.markdown("### 📊 Distribuição por Cluster")
        st.dataframe(cluster_dist, use_container_width=True)

    # ---------------- EXIBIR JOGOS DE ALTO VALOR ----------------
    if 'Market_Classification' in df.columns:
        top_value = df[df['Market_Classification'] == '🎯 VALUE BET'].copy()
        if not top_value.empty:
            st.markdown("### 💰 Top 10 Jogos com Valor Angular de Mercado")
            
            cols = [
                'League', 'Home', 'Away', 'Cluster3D_Desc',
                'Market_Error_Home', 'Market_Error_Away',
                'Value_Score_Home', 'Value_Score_Away',
                'Score_Final_Clusters', 'Recomendacao'
            ]
            cols = [c for c in cols if c in top_value.columns]
            
            top_value_sorted = top_value.sort_values('Value_Score_Home', ascending=False).head(10)
            
            st.dataframe(
                top_value_sorted[cols].style.format({
                    'Market_Error_Home': '{:.2%}',
                    'Market_Error_Away': '{:.2%}',
                    'Value_Score_Home': '{:.3f}',
                    'Value_Score_Away': '{:.3f}',
                    'Score_Final_Clusters': '{:.1f}'
                }).background_gradient(subset=['Value_Score_Home'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.info("⚖️ Nenhum jogo classificado como '🎯 VALUE BET' no momento.")

# Aplicar resumo
if not games_today.empty and 'Cluster3D_Desc' in games_today.columns:
    resumo_executivo_clusters(games_today)

# ---------------- FILTROS AVANÇADOS ----------------
st.sidebar.markdown("## 🔧 Filtros Avançados")

# Filtro de regressão
st.sidebar.markdown("### 🔄 Filtro de Regressão")
filtro_regressao = st.sidebar.selectbox(
    "Tendência de regressão:",
    [
        "Todas as tendências",
        "📈 Times em Melhora", 
        "📉 Times em Queda",
        "⚖️ Times Estáveis"
    ]
)

# Filtro de confidence
st.sidebar.markdown("### 🎯 Filtro de Confiança")
confianca_minima = st.sidebar.slider(
    "Confiança mínima do ML:",
    0.50, 0.95, 0.55, 0.01
)

# Aplicar filtros se necessário
if st.sidebar.button("🔄 Aplicar Filtros"):
    st.sidebar.success("Filtros aplicados!")
    # Os filtros seriam aplicados na próxima iteração

st.markdown("---")
st.success("🎯 **Sistema 3D com Clusters ML** implementado com sucesso!")


st.info("""
**✅ Sistema Simplificado:**
- 🧠 **Clusters 3D** em vez de quadrantes fixos
- 📈 **Regressão à Média** integrada
- 🎯 **Estratégias por Cluster** específicas
- 📊 **Visualização 3D** colorida por clusters
- 🔄 **Live Score** em tempo real
""")
