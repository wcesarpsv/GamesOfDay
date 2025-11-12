from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Analisador de Handicap Ótimo - DUAL MODEL", layout="wide")
st.title("🎯 Analisador de Handicap Ótimo - Dual Model (Home + Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "HandicapOptimizer_DualModel"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

# ============================================================
# 🔧 FUNÇÕES AUXILIARES ORIGINAIS
# ============================================================

def setup_livescore_columns(df):
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

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

def convert_asian_line_to_decimal(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if "/" not in value:
        try:
            num = float(value)
            return -num
        except ValueError:
            return np.nan
    try:
        parts = [float(p) for p in value.split("/")]
        avg = np.mean(parts)
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result
    except ValueError:
        return np.nan

def calcular_distancias_3d(df):
    """
    Calcula distâncias e ângulos 3D
    """
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 'Magnitude_3D']:
            df[col] = np.nan
        return df

    dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
    dy = (df['M_H'] - df['M_A']) / 2
    dz = (df['MT_H'] - df['MT_A']) / 2

    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz

    return df

def aplicar_clusterizacao_3d(df, n_clusters=4, random_state=42):
    """
    Cria clusters espaciais com base em Aggression, Momentum Liga e Momentum Time.
    Versão CORRIGIDA com verificação de dados suficientes.
    """
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    # 🔧 CORREÇÃO: Verificar se temos dados suficientes para clustering
    n_samples = X_cluster.shape[0]
    if n_samples < n_clusters:
        st.warning(f"⚠️ Dados insuficientes para clustering: {n_samples} amostras < {n_clusters} clusters")
        df['Cluster3D_Label'] = 0  # Atribuir todos ao mesmo cluster
        return df

    # 🔧 CORREÇÃO: Ajustar dinamicamente o número de clusters se necessário
    n_clusters_ajustado = min(n_clusters, n_samples)
    if n_clusters_ajustado < n_clusters:
        st.info(f"🔧 Ajustando n_clusters: {n_clusters} → {n_clusters_ajustado} (devido a {n_samples} amostras)")

    try:
        kmeans = KMeans(
            n_clusters=n_clusters_ajustado,
            random_state=random_state,
            init='k-means++',
            n_init=10
        )
        df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

        # Mostrar centroides apenas se temos clusters suficientes
        if n_clusters_ajustado > 1:
            centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
            centroids['Cluster'] = range(n_clusters_ajustado)
            
            st.markdown("### 🧭 Clusters 3D Criados (KMeans)")
            st.dataframe(centroids.style.format({'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}'}))
        else:
            st.info("📊 Apenas 1 cluster criado (dados insuficientes para múltiplos clusters)")

    except Exception as e:
        st.error(f"❌ Erro no clustering: {e}")
        df['Cluster3D_Label'] = 0  # Fallback: todos no cluster 0

    return df

# ============================================================
# 🎯 SISTEMA HOME: HANDICAP OPTIMIZATION - VERSÃO CONSERVADORA
# ============================================================

def calcular_handicap_otimo_calibrado_v2(row):
    """
    Versão CALIBRADA CONSERVADORA do cálculo de handicap ótimo
    Com limites mais restritivos e suavização mais forte
    """
    gh, ga = row.get('Goals_H_FT', 0), row.get('Goals_A_FT', 0)
    margin = gh - ga
    
    # 🔧 LIMITES MAIS RESTRITIVOS: Handicaps entre -1.5 e +1.5
    handicaps_possiveis = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, +0.25, +0.5, +0.75, +1.0, +1.25, +1.5]
    
    melhor_handicap = 0
    melhor_score = -10
    
    for handicap in handicaps_possiveis:
        # Simula resultado com handicap
        resultado_ajustado = margin + handicap
        
        # 🔧 SCORE MAIS CONSERVADOR: Penaliza MUITO handicaps extremos
        if resultado_ajustado > 0:
            # Ganhou - score positivo mas penaliza extremos
            base_score = 1.5
            # 🔽 PENALIDADE MAIS FORTE PARA EXTREMOS
            if abs(handicap) > 1.0:
                base_score = base_score - 0.8  # Redução de 53%
            elif abs(handicap) > 0.75:
                base_score = base_score - 0.4  # Redução de 27%
            elif abs(handicap) > 0.5:
                base_score = base_score - 0.2  # Redução de 13%
            score = base_score - abs(handicap) * 0.1
        elif resultado_ajustado == 0:
            # Push - score neutro
            score = 0.3
        else:
            # Perdeu - score negativo
            score = -0.5 - abs(handicap) * 0.15
        
        if score > melhor_score:
            melhor_score = score
            melhor_handicap = handicap
    
    # 🔽 SUAVIZAÇÃO FINAL - REDUZIR HANDICAPS EXTREMOS
    if abs(melhor_handicap) > 1.0:
        melhor_handicap = melhor_handicap * 0.6  # Reduzir 40%
    elif abs(melhor_handicap) > 0.75:
        melhor_handicap = melhor_handicap * 0.8  # Reduzir 20%
    
    return melhor_handicap

def criar_target_handicap_discreto_calibrado_v2(row):
    """
    Versão MAIS CONSERVADORA para classificação
    """
    handicap_otimo = calcular_handicap_otimo_calibrado_v2(row)
    
    # 🔽 CATEGORIAS MAIS CONSERVADORAS E EQUILIBRADAS
    if handicap_otimo <= -0.75:    # ANTES: -1.25
        return 'MODERATE_HOME'
    elif handicap_otimo <= -0.25:  # ANTES: -0.5  
        return 'LIGHT_HOME'
    elif handicap_otimo == 0:
        return 'NEUTRAL'
    elif handicap_otimo < 0.5:     # ANTES: 0.5 (ajustado)
        return 'LIGHT_AWAY'
    else:
        return 'MODERATE_AWAY'     # ANTES: 1.25

# ============================================================
# 🎯 SISTEMA AWAY: HANDICAP OPTIMIZATION - PERSPECTIVA AWAY
# ============================================================

def calcular_handicap_otimo_away(row):
    """
    Versão CALIBRADA para AWAY - Perspectiva do time visitante
    """
    gh, ga = row.get('Goals_H_FT', 0), row.get('Goals_A_FT', 0)
    margin = ga - gh  # 🔄 INVERTIDO - perspectiva AWAY
    
    # 🔧 MESMOS LIMITES: Handicaps entre -1.5 e +1.5
    handicaps_possiveis = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, +0.25, +0.5, +0.75, +1.0, +1.25, +1.5]
    
    melhor_handicap = 0
    melhor_score = -10
    
    for handicap in handicaps_possiveis:
        # Simula resultado com handicap (perspectiva AWAY)
        resultado_ajustado = margin + handicap
        
        # 🔧 SCORE CONSERVADOR (mesma lógica do HOME)
        if resultado_ajustado > 0:  # AWAY ganhou com handicap
            base_score = 1.5
            if abs(handicap) > 1.0:
                base_score = base_score - 0.8
            elif abs(handicap) > 0.75:
                base_score = base_score - 0.4
            elif abs(handicap) > 0.5:
                base_score = base_score - 0.2
            score = base_score - abs(handicap) * 0.1
        elif resultado_ajustado == 0:  # Empate
            score = 0.3
        else:  # AWAY perdeu
            score = -0.5 - abs(handicap) * 0.15
        
        if score > melhor_score:
            melhor_score = score
            melhor_handicap = handicap
    
    # 🔽 SUAVIZAÇÃO (mesma do HOME)
    if abs(melhor_handicap) > 1.0:
        melhor_handicap = melhor_handicap * 0.6
    elif abs(melhor_handicap) > 0.75:
        melhor_handicap = melhor_handicap * 0.8
    
    return melhor_handicap

def criar_target_handicap_away_discreto_calibrado(row):
    """
    Versão AWAY para classificação
    """
    handicap_otimo = calcular_handicap_otimo_away(row)
    
    # 🔽 CATEGORIAS (mesmas do HOME)
    if handicap_otimo <= -0.75:
        return 'MODERATE_AWAY'
    elif handicap_otimo <= -0.25:
        return 'LIGHT_AWAY'
    elif handicap_otimo == 0:
        return 'NEUTRAL'
    elif handicap_otimo < 0.5:
        return 'LIGHT_HOME'
    else:
        return 'MODERATE_HOME'

# ============================================================
# 🧠 MODELOS HOME CALIBRADOS - VERSÃO CONSERVADORA
# ============================================================

def treinar_modelo_handicap_regressao_calibrado_v2(history, games_today):
    """
    Modelo de Regressão CALIBRADO CONSERVADOR - HOME
    """
    st.markdown("### 📈 Modelo HOME Regressão Calibrado")
    
    # Criar target calibrado CONSERVADOR
    history['Handicap_Otimo_Calibrado'] = history.apply(calcular_handicap_otimo_calibrado_v2, axis=1)
    
    # 🔧 FILTRO MAIS RESTRITIVO
    handicap_range = [-1.25, 1.25]  # ANTES: [-2.0, 2.0]
    history_calibrado = history[
        (history['Handicap_Otimo_Calibrado'] >= handicap_range[0]) & 
        (history['Handicap_Otimo_Calibrado'] <= handicap_range[1])
    ].copy()
    
    st.info(f"📊 Dados HOME calibrados: {len(history_calibrado)} jogos")
    
    # Features espaciais
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history_calibrado.columns]
    
    if len(available_features) < 3:
        st.error("❌ Features insuficientes para treinamento HOME")
        return None, games_today, None
    
    X = history_calibrado[available_features].fillna(0)
    y = history_calibrado['Handicap_Otimo_Calibrado']
    
    # 🔧 NORMALIZAR FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Treinar modelo COM MAIS REGULARIZAÇÃO
    model = RandomForestRegressor(
        n_estimators=100,  # MENOS árvores (antes: 150)
        max_depth=5,       # MENOS profundidade (antes: 6)
        min_samples_leaf=20,  # MAIS amostras por folha (antes: 15)
        max_features=0.6,  # MENOS features (antes: 0.7)
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # 🔧 VALIDAÇÃO
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    st.success(f"✅ MAE do modelo HOME: {mae:.3f}")
    
    # Prever para jogos de hoje
    X_today = games_today[available_features].fillna(0)
    
    missing_features = set(available_features) - set(X_today.columns)
    if missing_features:
        st.warning(f"⚠️ Features faltando nos dados de hoje: {missing_features}")
        for feature in missing_features:
            X_today[feature] = 0
    
    X_today_scaled = scaler.transform(X_today[available_features])
    
    predictions = model.predict(X_today_scaled)
    
    # 🔧 SUAVIZAR PREDIÇÕES MAIS FORTEMENTE
    games_today['Handicap_Predito_Regressao_Calibrado'] = np.clip(predictions, -1.25, 1.25)
    
    return model, games_today, scaler

def treinar_modelo_handicap_classificacao_calibrado_v2(history, games_today):
    """
    Modelo de Classificação CALIBRADO CONSERVADOR - HOME
    """
    st.markdown("### 🎯 Modelo HOME Classificação Calibrado")
    
    # Criar target categórico calibrado CONSERVADOR
    history['Handicap_Categoria_Calibrado'] = history.apply(criar_target_handicap_discreto_calibrado_v2, axis=1)
    
    # Features
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history.columns]
    
    if len(available_features) < 3:
        st.error("❌ Features insuficientes para treinamento HOME")
        return None, games_today, None
    
    X = history[available_features].fillna(0)
    y = history['Handicap_Categoria_Calibrado']
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Treinar modelo CONSERVADOR
    model = RandomForestClassifier(
        n_estimators=100,  # MENOS árvores
        max_depth=5,       # MENOS profundidade
        random_state=42,
        class_weight='balanced',
        min_samples_leaf=15  # MAIS amostras
    )
    model.fit(X, y_encoded)
    
    # Prever para jogos de hoje
    X_today = games_today[available_features].fillna(0)
    
    missing_features = set(available_features) - set(X_today.columns)
    if missing_features:
        st.warning(f"⚠️ Features faltando nos dados de hoje: {missing_features}")
        for feature in missing_features:
            X_today[feature] = 0
    
    predicoes_encoded = model.predict(X_today[available_features])
    probas = model.predict_proba(X_today[available_features])
    
    games_today['Handicap_Categoria_Predito_Calibrado'] = le.inverse_transform(predicoes_encoded)
    games_today['Confianca_Categoria_Calibrado'] = np.max(probas, axis=1)
    
    # 🔧 MAPEAMENTO MAIS CONSERVADOR para handicaps numéricos
    categoria_para_handicap_calibrado_v2 = {
        'MODERATE_HOME': -0.75,   # ANTES: -1.5
        'LIGHT_HOME': -0.25,      # ANTES: -0.75
        'NEUTRAL': 0,
        'LIGHT_AWAY': +0.25,      # ANTES: +0.25
        'MODERATE_AWAY': +0.75    # ANTES: +1.5
    }
    
    games_today['Handicap_Predito_Classificacao_Calibrado'] = games_today['Handicap_Categoria_Predito_Calibrado'].map(categoria_para_handicap_calibrado_v2)
    
    st.info(f"📊 Distribuição categorias HOME: {dict(history['Handicap_Categoria_Calibrado'].value_counts())}")
    
    return model, games_today, le

# ============================================================
# 🧠 MODELOS AWAY CALIBRADOS - REGRESSÃO E CLASSIFICAÇÃO
# ============================================================

def treinar_modelo_away_handicap_regressao_calibrado(history, games_today):
    """
    Modelo de Regressão CALIBRADO para AWAY
    """
    st.markdown("### 📈 Modelo AWAY Regressão Calibrado")
    
    # Criar target calibrado AWAY
    history['Handicap_Otimo_AWAY_Calibrado'] = history.apply(calcular_handicap_otimo_away, axis=1)
    
    # 🔧 FILTRO MAIS RESTRITIVO
    handicap_range = [-1.25, 1.25]
    history_calibrado_away = history[
        (history['Handicap_Otimo_AWAY_Calibrado'] >= handicap_range[0]) & 
        (history['Handicap_Otimo_AWAY_Calibrado'] <= handicap_range[1])
    ].copy()
    
    st.info(f"📊 Dados AWAY calibrados: {len(history_calibrado_away)} jogos")
    
    # Features para AWAY (usamos as mesmas features 3D)
    features_3d_away = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features_away = [f for f in features_3d_away if f in history_calibrado_away.columns]
    
    if len(available_features_away) < 3:
        st.error("❌ Features insuficientes para treinamento AWAY")
        return None, games_today, None
    
    X_away = history_calibrado_away[available_features_away].fillna(0)
    y_away = history_calibrado_away['Handicap_Otimo_AWAY_Calibrado']
    
    # 🔧 NORMALIZAR FEATURES
    scaler_away = StandardScaler()
    X_away_scaled = scaler_away.fit_transform(X_away)
    
    # Treinar modelo AWAY
    model_away = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        max_features=0.6,
        random_state=42
    )
    model_away.fit(X_away_scaled, y_away)
    
    # 🔧 VALIDAÇÃO AWAY
    y_away_pred = model_away.predict(X_away_scaled)
    mae_away = mean_absolute_error(y_away, y_away_pred)
    st.success(f"✅ MAE do modelo AWAY: {mae_away:.3f}")
    
    # Prever para jogos de hoje - AWAY
    X_today_away = games_today[available_features_away].fillna(0)
    
    missing_features_away = set(available_features_away) - set(X_today_away.columns)
    if missing_features_away:
        for feature in missing_features_away:
            X_today_away[feature] = 0
    
    X_today_away_scaled = scaler_away.transform(X_today_away[available_features_away])
    
    predictions_away = model_away.predict(X_today_away_scaled)
    
    # 🔧 SUAVIZAR PREDIÇÕES AWAY
    games_today['Handicap_AWAY_Predito_Regressao_Calibrado'] = np.clip(predictions_away, -1.25, 1.25)
    
    return model_away, games_today, scaler_away

def treinar_modelo_away_handicap_classificacao_calibrado(history, games_today):
    """
    Modelo de Classificação CALIBRADO para AWAY
    """
    st.markdown("### 🎯 Modelo AWAY Classificação Calibrado")
    
    # Criar target categórico calibrado AWAY
    history['Handicap_Categoria_AWAY_Calibrado'] = history.apply(criar_target_handicap_away_discreto_calibrado, axis=1)
    
    # Features
    features_3d_away = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features_away = [f for f in features_3d_away if f in history.columns]
    
    if len(available_features_away) < 3:
        st.error("❌ Features insuficientes para treinamento AWAY")
        return None, games_today, None
    
    X_away = history[available_features_away].fillna(0)
    y_away = history['Handicap_Categoria_AWAY_Calibrado']
    
    # Codificar labels AWAY
    le_away = LabelEncoder()
    y_away_encoded = le_away.fit_transform(y_away)
    
    # Treinar modelo AWAY
    model_away = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced',
        min_samples_leaf=15
    )
    model_away.fit(X_away, y_away_encoded)
    
    # Prever para jogos de hoje - AWAY
    X_today_away = games_today[available_features_away].fillna(0)
    
    missing_features_away = set(available_features_away) - set(X_today_away.columns)
    if missing_features_away:
        for feature in missing_features_away:
            X_today_away[feature] = 0
    
    predicoes_away_encoded = model_away.predict(X_today_away[available_features_away])
    probas_away = model_away.predict_proba(X_today_away[available_features_away])
    
    games_today['Handicap_Categoria_AWAY_Predito_Calibrado'] = le_away.inverse_transform(predicoes_away_encoded)
    games_today['Confianca_Categoria_AWAY_Calibrado'] = np.max(probas_away, axis=1)
    
    # 🔧 MAPEAMENTO para handicaps numéricos AWAY
    categoria_para_handicap_away_calibrado = {
        'MODERATE_AWAY': -0.75,
        'LIGHT_AWAY': -0.25,
        'NEUTRAL': 0,
        'LIGHT_HOME': +0.25,
        'MODERATE_HOME': +0.75
    }
    
    games_today['Handicap_AWAY_Predito_Classificacao_Calibrado'] = games_today['Handicap_Categoria_AWAY_Predito_Calibrado'].map(categoria_para_handicap_away_calibrado)
    
    st.info(f"📊 Distribuição categorias AWAY: {dict(history['Handicap_Categoria_AWAY_Calibrado'].value_counts())}")
    
    return model_away, games_today, le_away

# ============================================================
# 📊 ANÁLISE DUAL - HOME + AWAY MODELS
# ============================================================

def analisar_value_bets_dual_modelos(games_today):
    """
    🧠 ANÁLISE INTELIGENTE - Encontrar os CONFRONTOS CERTOS
    Foca na RELAÇÃO entre times + DISTORÇÃO do handicap
    """
    st.markdown("## 💎 Análise DUAL - Home & Away Models")

    results = []
    
    for _, row in games_today.iterrows():
        asian_line = row.get('Asian_Line_Decimal', 0)
        
        # 🎯 PREDIÇÕES DOS MODELOS (HOME + AWAY)
        pred_home_reg = row.get('Handicap_Predito_Regressao_Calibrado', 0)
        pred_home_cls = row.get('Handicap_Predito_Classificacao_Calibrado', 0)
        pred_home = 0.7 * pred_home_reg + 0.3 * pred_home_cls
        
        pred_away_reg = row.get('Handicap_AWAY_Predito_Regressao_Calibrado', 0) 
        pred_away_cls = row.get('Handicap_AWAY_Predito_Classificacao_Calibrado', 0)
        pred_away = 0.7 * pred_away_reg + 0.3 * pred_away_cls
        
        # 🔍 ANÁLISE INTELIGENTE DA RELAÇÃO ENTRE TIMES
        forca_relativa = pred_home - pred_away
        equilibrio = abs(forca_relativa) < 0.2
        
        # 💡 DECISÕES INTELIGENTES POR CENÁRIO
        recomendacao_final, confidence = "NO CLEAR EDGE", "LOW"
        
        # CENÁRIO 1: TIMES EQUILIBRADOS + HANDICAP DISTORCIDO
        if equilibrio and abs(asian_line) > 0.25:
            if asian_line < -0.25:  # Mercado superestima Home
                value_gap = abs(asian_line) - abs(forca_relativa)
                if value_gap > 0.3:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
                elif value_gap > 0.15:
                    recomendacao_final, confidence = "BET AWAY", "MEDIUM"
                    
            elif asian_line > 0.25:  # Mercado superestima Away
                value_gap = abs(asian_line) - abs(forca_relativa)
                if value_gap > 0.3:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
                elif value_gap > 0.15:
                    recomendacao_final, confidence = "BET HOME", "MEDIUM"
        
        # CENÁRIO 2: TIME CLARAMENTE MAIS FORTE + HANDICAP JUSTO
        elif not equilibrio and abs(asian_line) < 0.75:
            if forca_relativa > 0.3 and asian_line < 0:  # Home forte, handicap razoável
                cobertura = forca_relativa - abs(asian_line)
                if cobertura > 0.2:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
                elif cobertura > 0.1:
                    recomendacao_final, confidence = "BET HOME", "MEDIUM"
                    
            elif forca_relativa < -0.3 and asian_line > 0:  # Away forte, handicap razoável
                cobertura = abs(forca_relativa) - abs(asian_line)
                if cobertura > 0.2:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
                elif cobertura > 0.1:
                    recomendacao_final, confidence = "BET AWAY", "MEDIUM"
        
        # CENÁRIO 3: HANDICAP EXTREMO - SÓ APOSTAR COM CONVICÇÃO
        elif abs(asian_line) >= 1.0:
            if forca_relativa > 0.5 and asian_line < -1.0:
                if forca_relativa > abs(asian_line) + 0.3:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
            elif forca_relativa < -0.5 and asian_line > 1.0:
                if abs(forca_relativa) > abs(asian_line) + 0.3:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
        
        # 📊 CALCULAR VALUE GAPS (mantendo formato original)
        value_gap_home = pred_home - asian_line
        value_gap_away = pred_away - (-asian_line)
        
        # 🆕 LIVE SCORE
        goals_h_today = row.get('Goals_H_Today')
        goals_a_today = row.get('Goals_A_Today')
        home_red = row.get('Home_Red')
        away_red = row.get('Away_Red')
        
        live_score_info = ""
        if pd.notna(goals_h_today) and pd.notna(goals_a_today):
            live_score_info = f"⚽ {goals_h_today}-{goals_a_today}"
            if pd.notna(home_red) and home_red > 0:
                live_score_info += f" 🟥H{home_red}"
            if pd.notna(away_red) and away_red > 0:
                live_score_info += f" 🟥A{away_red}"
        
        results.append({
            'League': row.get('League'),
            'Home': row.get('Home'),
            'Away': row.get('Away'),
            'Asian_Line': asian_line,
            
            # Modelo HOME
            'Handicap_HOME_Predito': round(pred_home, 2),
            'Value_Gap_HOME': round(value_gap_home, 2),
            
            # Modelo AWAY
            'Handicap_AWAY_Predito': round(pred_away, 2), 
            'Value_Gap_AWAY': round(value_gap_away, 2),
            
            # Decisão Final
            'Recomendacao': recomendacao_final,
            'Confidence': confidence,
            'Edge_Difference': round(abs(value_gap_home - value_gap_away), 2),
            'Live_Score': live_score_info
        })
    
    df_results = pd.DataFrame(results)
    
    # 🔍 FILTRAR APENAS JOGOS COM VALUE
    bets_validos = df_results[df_results['Recomendacao'] != 'NO CLEAR EDGE']
    
    return df_results, bets_validos

# ============================================================
# 📈 VISUALIZAÇÃO DUAL
# ============================================================

def plot_analise_dual_modelos(games_today):
    """
    Visualização da análise DUAL (HOME + AWAY)
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Value Gaps HOME vs AWAY
    value_gaps_home = []
    value_gaps_away = []
    
    for _, row in games_today.iterrows():
        asian_line = row.get('Asian_Line_Decimal', 0)
        pred_home = 0.7 * row.get('Handicap_Predito_Regressao_Calibrado', 0) + 0.3 * row.get('Handicap_Predito_Classificacao_Calibrado', 0)
        pred_away = 0.7 * row.get('Handicap_AWAY_Predito_Regressao_Calibrado', 0) + 0.3 * row.get('Handicap_AWAY_Predito_Classificacao_Calibrado', 0)
        
        value_gaps_home.append(pred_home - asian_line)
        value_gaps_away.append(pred_away - (-asian_line))
    
    x_pos = range(len(value_gaps_home))
    ax1.bar([x - 0.2 for x in x_pos], value_gaps_home, 0.4, label='HOME Value Gap', alpha=0.7, color='green')
    ax1.bar([x + 0.2 for x in x_pos], value_gaps_away, 0.4, label='AWAY Value Gap', alpha=0.7, color='blue')
    ax1.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    ax1.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Value Threshold')
    ax1.axhline(y=-0.15, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Jogos')
    ax1.set_ylabel('Value Gap')
    ax1.set_title('Value Gaps: HOME vs AWAY Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparação Handicaps Preditos
    handicaps_home = []
    handicaps_away = []
    asian_lines = []
    
    for _, row in games_today.iterrows():
        pred_home = 0.7 * row.get('Handicap_Predito_Regressao_Calibrado', 0) + 0.3 * row.get('Handicap_Predito_Classificacao_Calibrado', 0)
        pred_away = 0.7 * row.get('Handicap_AWAY_Predito_Regressao_Calibrado', 0) + 0.3 * row.get('Handicap_AWAY_Predito_Classificacao_Calibrado', 0)
        
        handicaps_home.append(pred_home)
        handicaps_away.append(pred_away)
        asian_lines.append(row.get('Asian_Line_Decimal', 0))
    
    ax2.scatter(asian_lines, handicaps_home, alpha=0.7, s=60, label='HOME Predito', color='green')
    ax2.scatter(asian_lines, handicaps_away, alpha=0.7, s=60, label='AWAY Predito', color='blue')
    ax2.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', alpha=0.3, label='Linha de Mercado')
    ax2.set_xlabel('Asian Line (Mercado)')
    ax2.set_ylabel('Handicap Predito')
    ax2.set_title('Handicaps Preditos vs Mercado')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL - DUAL MODEL
# ============================================================

def main_calibrado():
    # ---------------- Carregar Dados ----------------
    st.info("📂 Carregando dados para Análise DUAL MODEL...")
    
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    if not files:
        st.warning("No CSV files found in GamesDay folder.")
        return
    
    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)
    
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    
    # Carregar dados
    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)
        
        def classificar_league_tier(league_name: str) -> int:
            if pd.isna(league_name):
                return 3
            name = league_name.lower()
            if any(x in name for x in [
                'premier', 'la liga', 'serie a', 'bundesliga', 'ligue 1',
                'eredivisie', 'primeira liga', 'brasileirão', 'super league',
                'mls', 'championship', 'liga pro', 'a-league'
            ]):
                return 1
            if any(x in name for x in [
                'serie b', 'segunda', 'league 1', 'liga ii', 'liga 2', 'division 2',
                'bundesliga 2', 'ligue 2', 'championship', 'j-league', 'k-league',
                'superettan', '1st division', 'national league', 'liga nacional'
            ]):
                return 2
            return 3
        
        def aplicar_filtro_tier(df: pd.DataFrame, max_tier=2) -> pd.DataFrame:
            if 'League' not in df.columns:
                st.warning("⚠️ Coluna 'League' ausente — filtro de tier não aplicado.")
                df['League_Tier'] = 3
                return df
            df = df.copy()
            df['League_Tier'] = df['League'].apply(classificar_league_tier)
            filtrado = df[df['League_Tier'] <= max_tier].copy()
            st.info(f"🎯 Ligas filtradas (Tier ≤ {max_tier}): {len(filtrado)}/{len(df)} jogos mantidos")
            return filtrado
        
        # Aplicar o filtro
        history = aplicar_filtro_tier(history, max_tier=2)
        games_today = aplicar_filtro_tier(games_today, max_tier=2)
        
        from sklearn.preprocessing import OneHotEncoder
        
        # Selecionar as 10 ligas mais comuns no histórico
        top_ligas = history['League'].value_counts().head(10).index
        history['League_Clean'] = history['League'].where(history['League'].isin(top_ligas), 'Other')
        games_today['League_Clean'] = games_today['League'].where(games_today['League'].isin(top_ligas), 'Other')
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(history[['League_Clean']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['League_Clean']))
        
        # Adicionar ao histórico
        history = pd.concat([history.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        
        # Aplicar o mesmo encoder aos jogos de hoje
        encoded_today = encoder.transform(games_today[['League_Clean']])
        encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(['League_Clean']))
        games_today = pd.concat([games_today.reset_index(drop=True), encoded_today_df.reset_index(drop=True)], axis=1)

        return games_today, history
    
    games_today, history = load_cached_data(selected_file)
    
    if games_today.empty:
        st.warning("⚠️ Nenhum jogo encontrado após filtrar ligas principais.")
        return
    
    if history.empty:
        st.warning("⚠️ Histórico vazio após filtrar ligas principais.")
        return
    
    # ---------------- Converter Asian Line ----------------
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    
    history = history.dropna(subset=['Asian_Line_Decimal'])
    games_today = games_today.dropna(subset=['Asian_Line_Decimal'])
    
    # ---------------- Aplicar Filtro Temporal ----------------
    if "Date" in history.columns and "Date" in games_today.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
        except Exception as e:
            st.warning(f"⚠️ Erro ao aplicar filtro temporal: {e}")
    
    
    # ---------------- Adicionar Live Score Columns ----------------
    st.markdown("## 📊 Configurando Live Score...")
    
    # Configurar colunas de live score nos dados
    games_today = setup_livescore_columns(games_today)
    history = setup_livescore_columns(history)
    
    # Mostrar se temos dados de live score disponíveis
    live_score_games = games_today[
        (games_today['Goals_H_Today'].notna()) | 
        (games_today['Goals_A_Today'].notna()) |
        (games_today['Home_Red'].notna()) | 
        (games_today['Away_Red'].notna())
    ]
    
    if not live_score_games.empty:
        st.success(f"🎯 Dados de Live Score encontrados para {len(live_score_games)} jogos!")
        st.dataframe(live_score_games[['Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 'Home_Red', 'Away_Red']])
    else:
        st.info("ℹ️ Nenhum dado de Live Score disponível - usando apenas dados pré-jogo")
    
    # ---------------- Calcular Features 3D ----------------
    st.markdown("## 🧮 Calculando Features 3D...")
    
    # Verificar se as colunas necessárias existem
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_history = [col for col in required_cols if col not in history.columns]
    missing_today = [col for col in required_cols if col not in games_today.columns]
    
    if missing_history or missing_today:
        st.error(f"❌ Colunas necessárias faltando: History={missing_history}, Today={missing_today}")
        return
    
    # ❌ NÃO calcular momentum_time - MT já vem pré-calculado!
    # ✅ Apenas calcular distâncias 3D e clusters
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # ---------------- Treinar Modelos DUAL ----------------
    st.markdown("## 🧠 Treinando Modelos DUAL (HOME + AWAY)...")
    
    if st.button("🚀 Executar Análise DUAL", type="primary"):
        with st.spinner("Treinando modelos DUAL..."):
            # 🎯 TREINAR MODELOS HOME
            modelo_home_regressao, games_today, scaler_home = treinar_modelo_handicap_regressao_calibrado_v2(history, games_today)
            modelo_home_classificacao, games_today, label_encoder_home = treinar_modelo_handicap_classificacao_calibrado_v2(history, games_today)
            
            # 🎯 TREINAR MODELOS AWAY
            modelo_away_regressao, games_today, scaler_away = treinar_modelo_away_handicap_regressao_calibrado(history, games_today)
            modelo_away_classificacao, games_today, label_encoder_away = treinar_modelo_away_handicap_classificacao_calibrado(history, games_today)
            
            # 🔄 USAR ANÁLISE DUAL
            df_value_bets_dual, bets_validos_dual = analisar_value_bets_dual_modelos(games_today)
            
            # Exibir resultados
            st.markdown("## 📊 Resultados - Análise DUAL")
            
            if bets_validos_dual.empty:
                st.warning("⚠️ Nenhuma recomendação de value bet encontrada")
            else:
                st.dataframe(bets_validos_dual, use_container_width=True)
                
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    home_bets = len(bets_validos_dual[bets_validos_dual['Recomendacao'].str.contains('HOME')])
                    st.metric("🏠 HOME Bets", home_bets)
                with col2:
                    away_bets = len(bets_validos_dual[bets_validos_dual['Recomendacao'].str.contains('AWAY')])
                    st.metric("✈️ AWAY Bets", away_bets)
                with col3:
                    strong_bets = len(bets_validos_dual[bets_validos_dual['Confidence'] == 'HIGH'])
                    st.metric("🎯 Strong Bets", strong_bets)
                with col4:
                    total_bets = len(bets_validos_dual)
                    st.metric("📊 Total Recomendações", total_bets)
            
            # Visualizações
            st.pyplot(plot_analise_dual_modelos(games_today))
            
            st.success("✅ Análise DUAL concluída com sucesso!")
            st.balloons()

if __name__ == "__main__":
    main_calibrado()
