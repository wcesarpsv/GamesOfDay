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
    if df.empty or "League" not in df.columns: return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern,na=False)].copy()

def convert_asian_line_to_decimal(v):
    if pd.isna(v): return np.nan
    v=str(v).strip()
    if "/" not in v:
        try: return -float(v)
        except: return np.nan
    try:
        parts=[float(p) for p in v.split("/")]
        avg=np.mean(parts)
        return -avg if str(v).startswith("-") else avg*-1
    except: return np.nan

def calculate_ah_home_target(row):
    gh,ga,line=row.get("Goals_H_FT"),row.get("Goals_A_FT"),row.get("Asian_Line_Decimal")
    if pd.isna(gh) or pd.isna(ga) or pd.isna(line): return np.nan
    return 1 if (gh+line-ga)>0 else 0

# =====================================================================
# 📊 CÁLCULO ESPACIAL COM JULGAMENTO DE MERCADO (VERSÃO CORRIGIDA)
# =====================================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas espaciais 3D + detecção de julgamento de mercado.
    - Versão corrigida para garantir todas as colunas necessárias
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
            st.warning(f"⚠️ Coluna {col} não encontrada - preenchida com zeros")
        df[col] = df[col].fillna(0.0)

    # ------------------ Normalização robusta por liga ------------------
    cols_norm = cols_necessarias.copy()
    
    try:
        if 'League' in df.columns and df['League'].notna().any():
            for league, g in df.groupby('League', group_keys=False):
                if len(g) > 1:  # Só normaliza se tiver pelo menos 2 jogos
                    for col in cols_norm:
                        if col in g.columns:
                            mean_val = g[col].mean()
                            std_val = g[col].std(ddof=0)
                            if std_val != 0:
                                df.loc[g.index, f"{col}_norm"] = (g[col] - mean_val) / std_val
                            else:
                                df.loc[g.index, f"{col}_norm"] = 0.0
                else:
                    # Para ligas com apenas 1 jogo, usa valor original
                    for col in cols_norm:
                        df.loc[g.index, f"{col}_norm"] = g[col]
        else:
            # Normalização global
            for col in cols_norm:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std(ddof=0)
                    if std_val != 0:
                        df[f"{col}_norm"] = (df[col] - mean_val) / std_val
                    else:
                        df[f"{col}_norm"] = 0.0
    except Exception as e:
        st.error(f"❌ Erro na normalização: {e}")
        # Fallback: usa valores originais
        for col in cols_norm:
            df[f"{col}_norm"] = df[col]

    # ------------------ Cálculo vetorial 3D (USANDO COLUNAS NORMALIZADAS) ------------------
    df['dx'] = df.get('Aggression_Home_norm', 0) - df.get('Aggression_Away_norm', 0)
    df['dy'] = df.get('M_H_norm', 0) - df.get('M_A_norm', 0)
    df['dz'] = df.get('MT_H_norm', 0) - df.get('MT_A_norm', 0)

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

    # ------------------ Distorção de julgamento (USANDO COLUNAS ORIGINAIS) ------------------
    df['Judgment_Discrepancy_H'] = (df['Aggression_Home'] * -1) * (df['M_H'] + df['MT_H'])
    df['Judgment_Discrepancy_A'] = (df['Aggression_Away'] * -1) * (df['M_A'] + df['MT_A'])
    df['Diff_Judgment'] = df['Judgment_Discrepancy_H'] - df['Judgment_Discrepancy_A']

    # ------------------ Segurança final ------------------
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # DEBUG: Mostrar colunas criadas
    st.sidebar.info(f"📊 Colunas criadas: {len(df.columns)}")
    
    return df

# =====================================================================
# ⚡ CLUSTERIZAÇÃO 3D (VERSÃO ROBUSTA)
# =====================================================================
def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Cria clusters espaciais 3D com ajuste automático do número de clusters.
    - Evita erro quando há poucos jogos (n_samples < n_clusters)
    - Garante saída consistente mesmo com bases pequenas
    """
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
# 🎯 TREINAMENTO E EXIBIÇÃO (VERSÃO CORRIGIDA)
# =====================================================================
def treinar_modelo_espacial_inteligente(history, games_today):
    st.subheader("Treinando Modelo Market Judgment V3")
    
    # Aplicar cálculos espaciais
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    # Verificar se as colunas necessárias foram criadas
    colunas_necessarias = ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D']
    colunas_faltantes = [col for col in colunas_necessarias if col not in history.columns]
    
    if colunas_faltantes:
        st.error(f"❌ Colunas faltantes após cálculo espacial: {colunas_faltantes}")
        st.info("📋 Colunas disponíveis no history:")
        st.write(list(history.columns))
        return None, games_today
    
    # Aplicar clusterização
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # Calcular score espacial
    ang = 40
    history['Score_Espacial'] = history.apply(lambda x: calcular_score_espacial_inteligente(x, ang), axis=1)
    history['Target_Espacial'] = (history['Score_Espacial'] >= 0.5).astype(int)
    
    # Features para o modelo
    features = ['dx', 'dy', 'dz', 'Diff_Judgment', 'Quadrant_Dist_3D', 'Magnitude_3D', 'Score_Espacial', 'Cluster3D_Label']
    
    # Verificar novamente se todas as features existem
    features_disponiveis = [f for f in features if f in history.columns]
    features_faltantes = [f for f in features if f not in history.columns]
    
    if features_faltantes:
        st.warning(f"⚠️ Features faltantes: {features_faltantes}")
        st.info(f"✅ Usando features disponíveis: {features_disponiveis}")
        features = features_disponiveis
    
    if not features:
        st.error("❌ Nenhuma feature disponível para treinamento!")
        return None, games_today
    
    # Treinar modelo
    X = history[features].fillna(0)
    y = history['Target_Espacial']
    
    try:
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=8, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Fazer previsões
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
    
    if colunas_disponiveis:
        top = games_today[colunas_disponiveis].copy()
        top['Tipo'] = np.where(top['Diff_Judgment'] > 0, '⚡ Home Subestimado', '🔻 Home Overvalued')
        st.dataframe(top.sort_values('Diff_Judgment', ascending=False).head(10), width='stretch')
    else:
        st.warning("⚠️ Colunas insuficientes para exibir tabela de julgamento")
    
    st.success("✅ Modelo Market Judgment V3 treinado!")
    return model, games_today

# =====================================================================
# 🚀 MAIN (VERSÃO CORRIGIDA)
# =====================================================================
def main():
    st.sidebar.markdown("## Configurações V3")
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith('.csv')]
    
    if not files: 
        st.error("Nenhum CSV encontrado em GamesDay")
        return
    
    fsel = st.sidebar.selectbox("Arquivo:", sorted(files), index=len(files)-1)
    
    try:
        df = pd.read_csv(os.path.join(GAMES_FOLDER, fsel))
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {e}")
        return
    
    df = filter_leagues(df)
    
    # Preparar colunas necessárias
    if 'Asian_Line' in df.columns:
        df['Asian_Line_Decimal'] = df['Asian_Line'].apply(convert_asian_line_to_decimal)
    
    # Garantir colunas de goals
    if 'Goals_H_FT' not in df.columns: 
        df['Goals_H_FT'] = np.nan
        df['Goals_A_FT'] = np.nan
    
    df['Target_AH_Home'] = df.apply(calculate_ah_home_target, axis=1)
    
    history = df.dropna(subset=['Target_AH_Home']).copy()
    games_today = df.copy()
    
    st.sidebar.info(f"📚 Histórico: {len(history)} jogos | 🎯 Hoje: {len(games_today)} jogos")
    
    if st.sidebar.button("🚀 Treinar V3"):
        with st.spinner("Treinando modelo Market Judgment V3..."):
            model, res = treinar_modelo_espacial_inteligente(history, games_today)
            
            if model is not None and not res.empty:
                colunas_resultado = ['Home', 'Away', 'Prob_Espacial', 'ML_Side_Espacial', 'Confidence_Espacial']
                colunas_disponiveis = [col for col in colunas_resultado if col in res.columns]
                
                if colunas_disponiveis:
                    st.dataframe(
                        res[colunas_disponiveis].sort_values('Confidence_Espacial', ascending=False), 
                        width='stretch'
                    )
                else:
                    st.error("❌ Colunas de resultado não encontradas")
            else:
                st.error("❌ Falha no treinamento do modelo")
    else:
        st.info("👆 Clique em Treinar V3 para rodar o detector de julgamento de mercado.")

if __name__ == "__main__":
    main()
