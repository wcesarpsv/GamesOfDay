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
# 📊 CÁLCULO ESPACIAL COM JULGAMENTO DE MERCADO (VERSÃO ULTRA-ROBUSTA)
# =====================================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas espaciais 3D + detecção de julgamento de mercado.
    - Versão ultra-robusta que funciona mesmo com DataFrames vazios
    """
    if df is None or df.empty:
        st.warning("⚠️ DataFrame vazio recebido em calcular_distancias_3d(). Criando estrutura básica...")
        # Criar estrutura mínima necessária
        base_cols = ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D', 
                    'Quadrant_Angle_XY', 'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 
                    'Vector_Sign', 'Quadrant_Separation_3D',
                    'Judgment_Discrepancy_H', 'Judgment_Discrepancy_A']
        return pd.DataFrame(columns=base_cols)

    df = df.copy()

    # ------------------ DEBUG: Mostrar colunas disponíveis ------------------
    st.sidebar.info(f"📊 Colunas de entrada: {list(df.columns)}")

    # ------------------ Garantir colunas básicas ABSOLUTAMENTE NECESSÁRIAS ------------------
    cols_necessarias = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    
    for col in cols_necessarias:
        if col not in df.columns:
            st.warning(f"⚠️ Coluna {col} não encontrada - criando com zeros")
            df[col] = 0.0
    
    # Preencher NaN com zeros
    for col in cols_necessarias:
        df[col] = df[col].fillna(0.0)

    # ------------------ Cálculo DIRETO sem normalização complexa ------------------
    # Usar valores originais diretamente para evitar problemas de normalização
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    # ------------------ Garantir que todas as colunas necessárias existam ------------------
    # Distância 3D
    df['Quadrant_Dist_3D'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    df['Magnitude_3D'] = df['Quadrant_Dist_3D']  # Alias para compatibilidade

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

    # DEBUG: Verificar colunas criadas
    st.sidebar.success(f"✅ Colunas criadas: {len([col for col in df.columns if col in ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D']])}/6")

    return df

# =====================================================================
# ⚡ CLUSTERIZAÇÃO 3D (VERSÃO ROBUSTA)
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
# 🎯 TREINAMENTO E EXIBIÇÃO (VERSÃO ULTRA-ROBUSTA)
# =====================================================================
def treinar_modelo_espacial_inteligente(history, games_today):
    st.subheader("Treinando Modelo Market Judgment V3")
    
    # Verificar se temos dados suficientes
    if history.empty:
        st.error("❌ Histórico vazio! Não é possível treinar o modelo.")
        st.info("💡 Verifique se existem jogos com resultados completos (Goals_H_FT e Goals_A_FT)")
        return None, games_today
    
    st.info(f"📚 Dados de treino: {len(history)} jogos históricos")
    
    # Aplicar cálculos espaciais
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    # Verificar se as colunas necessárias foram criadas
    colunas_necessarias = ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D']
    
    if history.empty:
        st.error("❌ Histórico ficou vazio após cálculo espacial!")
        return None, games_today
        
    colunas_faltantes = [col for col in colunas_necessarias if col not in history.columns]
    
    if colunas_faltantes:
        st.error(f"❌ Colunas faltantes após cálculo espacial: {colunas_faltantes}")
        st.info("📋 Colunas disponíveis no history:")
        st.write(list(history.columns))
        
        # Tentar criar colunas manualmente como fallback
        st.warning("🔄 Tentando criar colunas manualmente...")
        for col in colunas_faltantes:
            if col in ['dx', 'dy', 'dz']:
                history[col] = 0.0
                games_today[col] = 0.0
            elif col == 'Diff_Judgment':
                history[col] = 0.0
                games_today[col] = 0.0
            elif col in ['Quadrant_Dist_3D', 'Magnitude_3D']:
                history[col] = 0.0
                games_today[col] = 0.0
    
    # Aplicar clusterização
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # Calcular score espacial
    ang = 40
    history['Score_Espacial'] = history.apply(lambda x: calcular_score_espacial_inteligente(x, ang), axis=1)
    history['Target_Espacial'] = (history['Score_Espacial'] >= 0.5).astype(int)
    
    # Features para o modelo
    features_base = ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D']
    features_extras = ['Score_Espacial', 'Cluster3D_Label']
    
    # Construir lista de features disponíveis
    features = []
    for f in features_base + features_extras:
        if f in history.columns:
            features.append(f)
    
    if len(features) < 3:  # Mínimo de features necessárias
        st.error(f"❌ Features insuficientes: apenas {len(features)} disponíveis")
        st.info(f"✅ Features disponíveis: {features}")
        return None, games_today
    
    st.success(f"✅ Features para treinamento: {features}")
    
    # Treinar modelo
    X = history[features].fillna(0)
    y = history['Target_Espacial']
    
    try:
        model = RandomForestClassifier(
            n_estimators=100,  # Reduzido para mais estabilidade
            max_depth=6, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Fazer previsões
        if games_today.empty:
            st.warning("⚠️ Nenhum jogo para hoje para fazer previsões")
            games_today['Prob_Espacial'] = 0.5
            games_today['ML_Side_Espacial'] = 'NEUTRAL'
            games_today['Confidence_Espacial'] = 0.0
        else:
            X_today = games_today[features].fillna(0)
            proba = np.clip(model.predict_proba(X_today)[:, 1], 0.05, 0.95)
            
            games_today['Prob_Espacial'] = proba
            games_today['ML_Side_Espacial'] = np.where(proba >= 0.5, 'HOME', 'AWAY')
            games_today['Confidence_Espacial'] = np.maximum(proba, 1 - proba)
        
    except Exception as e:
        st.error(f"❌ Erro no treinamento do modelo: {e}")
        return None, games_today
    
    # ---- tabela de julgamento invertido ----
    st.markdown("### 🧭 Top 10 Confrontos de Julgamento Invertido")
    
    colunas_tabela = ['League', 'Home', 'Away', 'Diff_Judgment', 'ML_Side_Espacial', 'Confidence_Espacial']
    colunas_disponiveis = [col for col in colunas_tabela if col in games_today.columns]
    
    if colunas_disponiveis and not games_today.empty:
        top = games_today[colunas_disponiveis].copy()
        top['Tipo'] = np.where(top['Diff_Judgment'] > 0, '⚡ Home Subestimado', '🔻 Home Overvalued')
        st.dataframe(top.sort_values('Diff_Judgment', ascending=False).head(10), width='stretch')
    else:
        st.warning("⚠️ Colunas insuficientes para exibir tabela de julgamento")
    
    st.success("✅ Modelo Market Judgment V3 treinado!")
    return model, games_today

# =====================================================================
# 🚀 MAIN (VERSÃO ULTRA-ROBUSTA)
# =====================================================================
def main():
    st.sidebar.markdown("## Configurações V3")
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith('.csv')]
    
    if not files: 
        st.error("❌ Nenhum CSV encontrado em GamesDay")
        st.info(f"💡 Verifique se a pasta '{GAMES_FOLDER}' existe e contém arquivos CSV")
        return
    
    fsel = st.sidebar.selectbox("Arquivo:", sorted(files), index=len(files)-1)
    
    try:
        df = pd.read_csv(os.path.join(GAMES_FOLDER, fsel))
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        
        # DEBUG: Mostrar informações do arquivo
        st.sidebar.info(f"📋 Colunas no CSV: {len(df.columns)}")
        st.sidebar.write(f"📅 Período: {df['Date'].min() if 'Date' in df.columns else 'N/A'} a {df['Date'].max() if 'Date' in df.columns else 'N/A'}")
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {e}")
        return
    
    df = filter_leagues(df)
    
    # Preparar colunas necessárias
    if 'Asian_Line' in df.columns:
        df['Asian_Line_Decimal'] = df['Asian_Line'].apply(convert_asian_line_to_decimal)
    else:
        st.warning("⚠️ Coluna 'Asian_Line' não encontrada")
        df['Asian_Line_Decimal'] = np.nan
    
    # Garantir colunas de goals
    if 'Goals_H_FT' not in df.columns: 
        st.warning("⚠️ Colunas de goals não encontradas - criando com NaN")
        df['Goals_H_FT'] = np.nan
        df['Goals_A_FT'] = np.nan
    
    df['Target_AH_Home'] = df.apply(calculate_ah_home_target, axis=1)
    
    # Separar histórico e jogos de hoje
    history = df.dropna(subset=['Target_AH_Home']).copy()
    games_today = df.copy()
    
    st.sidebar.info(f"📚 Histórico: {len(history)} jogos | 🎯 Hoje: {len(games_today)} jogos")
    
    # DEBUG: Mostrar primeiras linhas
    with st.expander("🔍 Debug - Visualizar Dados Carregados"):
        st.write("**DataFrame Completo:**", df.shape)
        st.write("**Colunas:**", list(df.columns))
        st.write("**Primeiras linhas:**")
        st.dataframe(df.head(3), width='stretch')
        
        if not history.empty:
            st.write("**Histórico (com targets):**", history.shape)
            st.dataframe(history[['Home', 'Away', 'Goals_H_FT', 'Goals_A_FT', 'Target_AH_Home']].head(3), width='stretch')
    
    if st.sidebar.button("🚀 Treinar V3"):
        with st.spinner("Treinando modelo Market Judgment V3..."):
            model, res = treinar_modelo_espacial_inteligente(history, games_today)
            
            if model is not None and not res.empty:
                colunas_resultado = ['Home', 'Away', 'Prob_Espacial', 'ML_Side_Espacial', 'Confidence_Espacial']
                colunas_disponiveis = [col for col in colunas_resultado if col in res.columns]
                
                if colunas_disponiveis:
                    st.markdown("### 📊 Resultados das Previsões")
                    st.dataframe(
                        res[colunas_disponiveis].sort_values('Confidence_Espacial', ascending=False), 
                        width='stretch'
                    )
                else:
                    st.error("❌ Colunas de resultado não encontradas")
            else:
                st.error("❌ Falha no treinamento do modelo - verifique os dados de entrada")
    else:
        st.info("👆 Clique em 'Treinar V3' para rodar o detector de julgamento de mercado")

if __name__ == "__main__":
    main()
