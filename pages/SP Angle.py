# =====================================================================
# 🎯 SISTEMA ESPACIAL INTELIGENTE – V3 MARKET JUDGMENT
# =====================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistema Espacial Inteligente – Market Judgment V3", layout="wide")
st.title("🎯 Sistema Espacial Inteligente com Market Judgment V3")

# ------------------- CONFIG -------------------
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup","copas","uefa","afc","sudamericana","copa","trophy"]
np.random.seed(42)

# =====================================================================
# 🔧 FUNÇÕES BÁSICAS
# =====================================================================
def filter_leagues(df):
    if df.empty or "League" not in df.columns: 
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern,na=False)].copy()

def convert_asian_line_to_decimal(v):
    if pd.isna(v): 
        return np.nan
    v = str(v).strip()
    if "/" not in v:
        try: 
            return -float(v)
        except: 
            return np.nan
    try:
        parts = [float(p) for p in v.split("/")]
        avg = np.mean(parts)
        return -avg if str(v).startswith("-") else avg * -1
    except: 
        return np.nan

def calculate_ah_home_target(row):
    gh, ga, line = row.get("Goals_H_FT"), row.get("Goals_A_FT"), row.get("Asian_Line_Decimal")
    if pd.isna(gh) or pd.isna(ga) or pd.isna(line): 
        return np.nan
    return 1 if (gh + line - ga) > 0 else 0

# =====================================================================
# 📊 CÁLCULO ESPACIAL COM JULGAMENTO DE MERCADO
# =====================================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas espaciais 3D + detecção de julgamento de mercado.
    """
    if df is None or df.empty:
        st.warning("⚠️ DataFrame vazio recebido em calcular_distancias_3d().")
        return pd.DataFrame()

    df = df.copy()

    # ------------------ Garantir colunas básicas ------------------
    cols_necessarias = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    
    for col in cols_necessarias:
        if col not in df.columns:
            df[col] = 0.0
    
    # Preencher NaN com zeros
    for col in cols_necessarias:
        df[col] = df[col].fillna(0.0)

    # ------------------ Cálculo vetorial 3D ------------------
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    # ------------------ Métricas espaciais ------------------
    df['Quadrant_Dist_3D'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    df['Magnitude_3D'] = df['Quadrant_Dist_3D']

    # Ângulos e trigonometria
    angle_xy = np.arctan2(df['dy'], df['dx'])
    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    
    # Sinais vetoriais
    df['Vector_Sign'] = np.sign(df['dx'] * df['dy'] * df['dz']).fillna(0)
    df['Quadrant_Separation_3D'] = (df['dx'] + df['dy'] + df['dz']) / 3.0

    # ------------------ Distorção de julgamento ------------------
    df['Judgment_Discrepancy_H'] = (df['Aggression_Home'] * -1) * (df['M_H'] + df['MT_H'])
    df['Judgment_Discrepancy_A'] = (df['Aggression_Away'] * -1) * (df['M_A'] + df['MT_A'])
    df['Diff_Judgment'] = df['Judgment_Discrepancy_H'] - df['Judgment_Discrepancy_A']

    # ------------------ Segurança final ------------------
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df

# =====================================================================
# ⚡ CLUSTERIZAÇÃO 3D
# =====================================================================
def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Cria clusters espaciais 3D com ajuste automático do número de clusters.
    """
    if df.empty:
        df['Cluster3D_Label'] = 0
        return df

    df = df.copy()

    # Garantir colunas diferenciais
    for col in ['dx', 'dy', 'dz']:
        if col not in df.columns:
            df[col] = 0

    X = df[['dx', 'dy', 'dz']].fillna(0)

    # Se poucos jogos, reduzir clusters automaticamente
    n_samples = len(X)
    if n_samples < 2:
        df['Cluster3D_Label'] = 0
        return df

    n_clusters = min(max(2, n_clusters), n_samples)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        df['Cluster3D_Label'] = kmeans.fit_predict(X)
    except Exception as e:
        st.warning(f"⚠️ Clusterização simplificada: {e}")
        df['Cluster3D_Label'] = 0

    return df

# =====================================================================
# 🧮 SCORE ESPACIAL AJUSTADO POR JULGAMENTO
# =====================================================================
def calcular_score_espacial_inteligente(row, angulo):
    dx = row.get('dx', 0)
    dy = row.get('dy', 0) 
    dz = row.get('dz', 0)
    ang_xy = row.get('Quadrant_Angle_XY', 0)
    diff = row.get('Diff_Judgment', 0)
    cluster = row.get('Cluster3D_Label', 0)
    
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    score = 0.5
    
    # pesos simétricos
    score += np.sign(dx) * 0.12
    score += np.sign(dz) * 0.10
    
    if abs(ang_xy) < angulo: 
        score += np.sign(dx) * 0.08
    else: 
        score -= np.sign(dx) * 0.08
        
    # ajuste por julgamento
    if diff > 0: 
        score += 0.05
    elif diff < 0: 
        score -= 0.05
        
    # cluster refinamento
    if cluster == 0: 
        score += 0.02
    elif cluster == 1: 
        score -= 0.02
        
    # distancia amortecida
    if dist < 0.4: 
        score = 0.5 + (score - 0.5) * 0.4
        
    return float(np.clip(score, 0.05, 0.95))

# =====================================================================
# 🎯 TREINAMENTO E PREVISÃO (LÓGICA CORRIGIDA)
# =====================================================================
def treinar_e_prever_espacial_inteligente(history_with_results, games_today):
    """
    LÓGICA CORRIGIDA:
    - Usa dados HISTÓRICOS com resultados para treinar
    - Aplica o modelo treinado nos jogos de HOJE (sem resultados)
    """
    st.subheader("🎯 Modelo Market Judgment V3 - Previsões para Hoje")
    
    # VERIFICAÇÃO 1: Temos dados históricos para treinar?
    if history_with_results.empty:
        st.error("❌ Nenhum dado histórico com resultados para treinar o modelo!")
        st.info("""
        💡 **Solução:** 
        - Use um arquivo CSV que contenha jogos PASSADOS com resultados completos
        - O modelo precisa de histórico para aprender padrões
        - Os jogos de HOJE não precisam ter resultados (serão previstos)
        """)
        return None, games_today
    
    st.success(f"📚 Dados de treino: {len(history_with_results)} jogos históricos com resultados")
    
    # VERIFICAÇÃO 2: Temos jogos para hoje para prever?
    if games_today.empty:
        st.warning("⚠️ Nenhum jogo encontrado para hoje")
        return None, games_today
    
    st.info(f"🎯 Jogos para prever hoje: {len(games_today)}")

    # PASSO 1: Calcular features espaciais nos dados históricos
    st.write("📊 Calculando métricas espaciais...")
    history_processed = calcular_distancias_3d(history_with_results)
    
    if history_processed.empty:
        st.error("❌ Erro no processamento dos dados históricos")
        return None, games_today

    # PASSO 2: Aplicar clusterização no histórico
    st.write("🔮 Aplicando clusterização...")
    history_processed = aplicar_clusterizacao_3d(history_processed)

    # PASSO 3: Calcular score espacial no histórico (PARA TREINO)
    ang = 40
    history_processed['Score_Espacial'] = history_processed.apply(
        lambda x: calcular_score_espacial_inteligente(x, ang), axis=1
    )
    
    # PASSO 4: Definir target baseado no score (PARA TREINO)
    history_processed['Target_Espacial'] = (history_processed['Score_Espacial'] >= 0.5).astype(int)

    # PASSO 5: Treinar modelo com dados históricos
    st.write("🤖 Treinando modelo Random Forest...")
    features = ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D', 'Score_Espacial', 'Cluster3D_Label']
    
    # Garantir que todas as features existem
    features_disponiveis = [f for f in features if f in history_processed.columns]
    
    if len(features_disponiveis) < 3:
        st.error(f"❌ Features insuficientes para treinamento: {features_disponiveis}")
        return None, games_today

    X_train = history_processed[features_disponiveis].fillna(0)
    y_train = history_processed['Target_Espacial']

    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        st.success(f"✅ Modelo treinado com {len(X_train)} amostras e {len(features_disponiveis)} features")
    except Exception as e:
        st.error(f"❌ Erro no treinamento: {e}")
        return None, games_today

    # PASSO 6: APLICAR NAS PREVISÕES DE HOJE
    st.write("🔮 Aplicando previsões para os jogos de hoje...")
    
    # Processar jogos de hoje com as MESMAS transformações
    games_processed = calcular_distancias_3d(games_today)
    games_processed = aplicar_clusterizacao_3d(games_processed)
    
    # Calcular score espacial para hoje
    games_processed['Score_Espacial'] = games_processed.apply(
        lambda x: calcular_score_espacial_inteligente(x, ang), axis=1
    )

    # Fazer previsões
    X_today = games_processed[features_disponiveis].fillna(0)
    proba = np.clip(model.predict_proba(X_today)[:, 1], 0.05, 0.95)
    
    # Adicionar resultados às previsões
    games_processed['Prob_Espacial'] = proba
    games_processed['ML_Side_Espacial'] = np.where(proba >= 0.5, 'HOME', 'AWAY')
    games_processed['Confidence_Espacial'] = np.maximum(proba, 1 - proba)

    # EXIBIR RESULTADOS
    st.success("✅ Previsões concluídas!")
    
    # Tabela de julgamento de mercado
    st.markdown("### 🧭 Top 10 - Julgamento de Mercado")
    colunas_tabela = ['League', 'Home', 'Away', 'Diff_Judgment', 'ML_Side_Espacial', 'Confidence_Espacial']
    colunas_disponiveis = [col for col in colunas_tabela if col in games_processed.columns]
    
    if colunas_disponiveis:
        top_julgamento = games_processed[colunas_disponiveis].copy()
        top_julgamento['Tipo_Julgamento'] = np.where(
            top_julgamento['Diff_Judgment'] > 0, 
            '⚡ Home Subestimado', 
            '🔻 Home Superestimado'
        )
        st.dataframe(
            top_julgamento.sort_values('Diff_Judgment', ascending=False).head(10), 
            width='stretch'
        )

    return model, games_processed

# =====================================================================
# 🚀 MAIN (LÓGICA CORRIGIDA)
# =====================================================================
def main():
    st.sidebar.markdown("## Configurações V3 - Market Judgment")
    
    # Carregar arquivos disponíveis
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith('.csv')]
    
    if not files: 
        st.error(f"❌ Nenhum CSV encontrado na pasta '{GAMES_FOLDER}'")
        return
    
    fsel = st.sidebar.selectbox("Selecionar arquivo CSV:", sorted(files), index=len(files)-1)
    
    try:
        # Carregar dados
        df = pd.read_csv(os.path.join(GAMES_FOLDER, fsel))
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {e}")
        return
    
    # Filtrar ligas
    df = filter_leagues(df)
    
    # Preparar colunas de Asian Handicap
    if 'Asian_Line' in df.columns:
        df['Asian_Line_Decimal'] = df['Asian_Line'].apply(convert_asian_line_to_decimal)
    else:
        df['Asian_Line_Decimal'] = 0.0  # Valor padrão

    # Garantir colunas de goals
    if 'Goals_H_FT' not in df.columns: 
        df['Goals_H_FT'] = np.nan
        df['Goals_A_FT'] = np.nan
    
    # CALCULAR TARGET APENAS PARA IDENTIFICAR JOGOS COM RESULTADOS
    df['Target_AH_Home'] = df.apply(calculate_ah_home_target, axis=1)
    
    # SEPARAÇÃO CORRIGIDA:
    # - HISTÓRICO: jogos com resultados (Target_AH_Home não é NaN) → PARA TREINAR
    # - HOJE: todos os jogos (incluindo os sem resultados) → PARA PREVER
    
    history_with_results = df.dropna(subset=['Target_AH_Home']).copy()
    all_games_today = df.copy()  # Todos os jogos do arquivo
    
    st.sidebar.info(f"""
    📊 **Estatísticas do Arquivo:**
    - 📚 Histórico com resultados: **{len(history_with_results)}** jogos
    - 🎯 Total de jogos para análise: **{len(all_games_today)}** jogos
    - 📅 Data do arquivo: **{fsel}**
    """)
    
    # Debug expander
    with st.expander("🔍 Ver detalhes dos dados carregados"):
        st.write("**Colunas disponíveis:**", list(df.columns))
        st.write("**Primeiras linhas:**")
        st.dataframe(df.head(3), width='stretch')
        
        if not history_with_results.empty:
            st.write("**Exemplo de jogos históricos (com resultados):**")
            st.dataframe(history_with_results[['Home', 'Away', 'Goals_H_FT', 'Goals_A_FT', 'Asian_Line_Decimal', 'Target_AH_Home']].head(3), width='stretch')
    
    # Botão de treinamento
    if st.sidebar.button("🚀 Executar Market Judgment V3", type="primary"):
        with st.spinner("Processando dados e gerando previsões..."):
            model, resultados = treinar_e_prever_espacial_inteligente(history_with_results, all_games_today)
            
            if model is not None and not resultados.empty:
                # Mostrar todas as previsões
                st.markdown("### 📊 Previsões para Todos os Jogos")
                
                colunas_resultado = [
                    'League', 'Home', 'Away', 
                    'Prob_Espacial', 'ML_Side_Espacial', 'Confidence_Espacial',
                    'Diff_Judgment'
                ]
                
                colunas_disponiveis = [col for col in colunas_resultado if col in resultados.columns]
                
                if colunas_disponiveis:
                    resultados_ordenados = resultados[colunas_disponiveis].sort_values(
                        'Confidence_Espacial', 
                        ascending=False
                    )
                    
                    st.dataframe(resultados_ordenados, width='stretch')
                    
                    # Estatísticas rápidas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏠 Previsões HOME", 
                                 len(resultados[resultados['ML_Side_Espacial'] == 'HOME']))
                    with col2:
                        st.metric("✈️ Previsões AWAY", 
                                 len(resultados[resultados['ML_Side_Espacial'] == 'AWAY']))
                    with col3:
                        avg_confidence = resultados['Confidence_Espacial'].mean()
                        st.metric("🎯 Confiança Média", f"{avg_confidence:.1%}")
                    
                else:
                    st.error("❌ Colunas de resultado não encontradas")
            else:
                st.error("❌ Não foi possível gerar previsões")

    else:
        st.info("👆 Clique em **'Executar Market Judgment V3'** para gerar previsões")

if __name__ == "__main__":
    main()
