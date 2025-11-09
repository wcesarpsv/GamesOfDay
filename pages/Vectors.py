from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.cluster import KMeans
import xgboost as xgb

st.set_page_config(page_title="Ensemble Angular - Bet Indicator", layout="wide")
st.title("🧠 Ensemble com Features Angulares - Approach 2")

# ---------------- Configurações ----------------
PAGE_PREFIX = "EnsembleAngular"
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

def convert_asian_line_to_decimal(value):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Caso simples — número único
    if "/" not in value:
        try:
            num = float(value)
            return -num  # Inverte sinal (Away → Home)
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
        # Inverte o sinal no final (Away → Home)
        return -result
    except ValueError:
        return np.nan

# ============================================================
# 🎯 CÁLCULO DO TARGET – COBERTURA REAL DE HANDICAP (AH)
# ============================================================

def calculate_ah_home_target(row):
    """
    Calcula o target binário indicando se o time da casa cobriu o handicap asiático.
    """
    gh = row.get("Goals_H_FT")
    ga = row.get("Goals_A_FT")
    line_home = row.get("Asian_Line_Decimal")

    # Verificação de dados ausentes
    if pd.isna(gh) or pd.isna(ga) or pd.isna(line_home):
        return np.nan

    # Cálculo ajustado do placar considerando o handicap da casa
    adjusted = (gh + line_home) - ga

    # Resultado final
    if adjusted > 0:
        return 1   # Home cobre o handicap
    elif adjusted < 0:
        return 0   # Home não cobre o handicap
    else:
        return 0   # Push (tratado como 0 para manter consistência)

# ==============================================================
# 🧩 BLOCO – CLUSTERIZAÇÃO 3D (KMEANS)
# ==============================================================

def aplicar_clusterizacao_3d(df, n_clusters=4, random_state=42):
    """
    Cria clusters espaciais com base em Aggression, Momentum Liga e Momentum Time.
    """
    df = df.copy()

    # Garante as colunas necessárias
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = -1
        return df

    # Diferenças espaciais (vetor 3D)
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

    # Calcular centroide de cada cluster para diagnóstico
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
    centroids['Cluster'] = range(n_clusters)

    st.markdown("### 🧭 Clusters 3D Criados (KMeans)")
    st.dataframe(centroids.style.format({'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}'}))

    # Adicionar também uma descrição textual leve
    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map({
        0: '⚡ Agressivos + Momentum Positivo',
        1: '💤 Reativos + Momentum Negativo',
        2: '⚖️ Equilibrados',
        3: '🔥 Alta Variância'
    }).fillna('🌀 Outro')

    return df

# ============================================================
# 🧮 CÁLCULO DE MOMENTUM DO TIME (MT_H e MT_A)
# ============================================================

def calcular_momentum_time(df, window=6):
    """
    Calcula o Momentum do Time (MT_H / MT_A) com base no HandScore.
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
        if mask_home.sum() > 2:
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

# ============================================================
# 📐 CÁLCULO DE DISTÂNCIAS 3D
# ============================================================

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

# ============================================================
# 🧭 ENGENHARIA DE FEATURES ANGULARES AVANÇADAS
# ============================================================

def engenhar_features_angulares(df):
    """
    Gera features geométricas adicionais (ângulos, direções e estabilidade vetorial)
    para que o modelo aprenda padrões espaciais de cobertura AH.
    """
    df = df.copy()

    # 🔹 GARANTIR COLUNAS BASE (com fallback robusto)
    required_base_cols = ['dx', 'dy', 'dz']
    for col in required_base_cols:
        if col not in df.columns:
            # Calcular diferenças se não existirem
            if 'Aggression_Home' in df.columns and 'Aggression_Away' in df.columns:
                df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
            else:
                df['dx'] = 0.0
                
            if 'M_H' in df.columns and 'M_A' in df.columns:
                df['dy'] = df['M_H'] - df['M_A']
            else:
                df['dy'] = 0.0
                
            if 'MT_H' in df.columns and 'MT_A' in df.columns:
                df['dz'] = df['MT_H'] - df['MT_A']
            else:
                df['dz'] = 0.0
            break

    # 🔹 CALCULAR ÂNGULOS SE NÃO EXISTIREM
    if 'Quadrant_Angle_XY' not in df.columns:
        df['Quadrant_Angle_XY'] = np.degrees(np.arctan2(df['dy'], df['dx']))
        df['Quadrant_Angle_XZ'] = np.degrees(np.arctan2(df['dz'], df['dx']))
        df['Quadrant_Angle_YZ'] = np.degrees(np.arctan2(df['dz'], df['dy']))

    # =======================
    # 🔹 1. ESTABILIDADE ANGULAR (MELHORADA)
    # =======================
    # Normalizar ângulos para evitar descontinuidades
    angle_xy_norm = np.radians(df['Quadrant_Angle_XY'] % 360)
    angle_xz_norm = np.radians(df['Quadrant_Angle_XZ'] % 360) 
    angle_yz_norm = np.radians(df['Quadrant_Angle_YZ'] % 360)
    
    df['Angle_Stability'] = (
        np.cos(angle_xy_norm) * np.cos(angle_xz_norm) * np.cos(angle_yz_norm)
    )
    
    # Turbulência com suavização
    df['Angle_Turbulence'] = 1 - (df['Angle_Stability'] + 1) / 2

    # =======================
    # 🔹 2. DIREÇÃO VETORIAL (COM ROBUSTEZ)
    # =======================
    df['Vector_Magnitude'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    
    # Evitar divisão por zero com small epsilon
    epsilon = 1e-8
    magnitude_safe = df['Vector_Magnitude'] + epsilon
    
    df['Vector_Direction_X'] = df['dx'] / magnitude_safe
    df['Vector_Direction_Y'] = df['dy'] / magnitude_safe  
    df['Vector_Direction_Z'] = df['dz'] / magnitude_safe

    # =======================
    # 🔹 3. INTERAÇÕES ANGULARES AVANÇADAS
    # =======================
    df['Angle_Interaction_XY_XZ'] = np.cos(angle_xy_norm - angle_xz_norm)
    df['Angle_Interaction_XY_YZ'] = np.cos(angle_xy_norm - angle_yz_norm)
    df['Angle_Interaction_XZ_YZ'] = np.cos(angle_xz_norm - angle_yz_norm)
    
    # 🔹 NOVO: Sincronia Angular (quanto os vetores "convergem")
    df['Angular_Sync'] = (
        df['Angle_Interaction_XY_XZ'] + 
        df['Angle_Interaction_XY_YZ'] + 
        df['Angle_Interaction_XZ_YZ']
    ) / 3

    # =======================
    # 🔹 4. FEATURES DE DOMINÂNCIA VETORIAL
    # =======================
    df['Vector_Dominance_X'] = abs(df['Vector_Direction_X'])
    df['Vector_Dominance_Y'] = abs(df['Vector_Direction_Y']) 
    df['Vector_Dominance_Z'] = abs(df['Vector_Direction_Z'])
    
    # Qual dimensão domina mais?
    df['Primary_Dimension'] = np.argmax([
        df['Vector_Dominance_X'], 
        df['Vector_Dominance_Y'],
        df['Vector_Dominance_Z']
    ], axis=0)

    # =======================
    # 🔹 5. PROBABILIDADE HISTÓRICA POR ÂNGULO (ADAPTATIVA)
    # =======================
    try:
        # Usar bins dinâmicos baseados na distribuição
        angle_xy_clean = df['Quadrant_Angle_XY'].dropna()
        if len(angle_xy_clean) > 10:  # Mínimo para estatística
            bins = np.linspace(angle_xy_clean.min(), angle_xy_clean.max(), 12)
            angle_bins = pd.cut(df['Quadrant_Angle_XY'], bins=bins, duplicates='drop')
            
            # Calcular probabilidade histórica de cobertura AH
            if 'Target_AH_Home' in df.columns:
                df_angle_stats = df.groupby(angle_bins)['Target_AH_Home'].agg(['mean', 'count'])
                df_angle_stats = df_angle_stats[df_angle_stats['count'] >= 3]  # Mínimo de amostras
                
                df['Angle_Historical_Prob'] = df['Quadrant_Angle_XY'].map(df_angle_stats['mean'])
                df['Angle_Sample_Size'] = df['Quadrant_Angle_XY'].map(df_angle_stats['count'])
            else:
                df['Angle_Historical_Prob'] = 0.5
                df['Angle_Sample_Size'] = 0
        else:
            df['Angle_Historical_Prob'] = 0.5
            df['Angle_Sample_Size'] = 0
    except Exception as e:
        df['Angle_Historical_Prob'] = 0.5
        df['Angle_Sample_Size'] = 0

    # =======================
    # 🔹 6. FEATURES DE QUADRANTE ANGULAR
    # =======================
    # Classificar em quadrantes angulares (45 graus cada)
    df['Angular_Quadrant'] = (df['Quadrant_Angle_XY'] // 45) % 8
    
    # Sinal angular (positivo/negativo)
    df['Angular_Sign_Consistency'] = (
        np.sign(df['Quadrant_Angle_XY']) * 
        np.sign(df['Quadrant_Angle_XZ']) * 
        np.sign(df['Quadrant_Angle_YZ'])
    )

    # =======================
    # 🔹 7. LIMPEZA E NORMALIZAÇÃO FINAL
    # =======================
    # Preencher NaN com valores neutros
    angular_features = [
        'Angle_Stability', 'Angle_Turbulence', 
        'Vector_Direction_X', 'Vector_Direction_Y', 'Vector_Direction_Z',
        'Angle_Interaction_XY_XZ', 'Angle_Interaction_XY_YZ', 'Angle_Interaction_XZ_YZ',
        'Angular_Sync', 'Vector_Dominance_X', 'Vector_Dominance_Y', 'Vector_Dominance_Z',
        'Angle_Historical_Prob', 'Angular_Sign_Consistency'
    ]
    
    for feature in angular_features:
        if feature in df.columns:
            if 'Direction' in feature or 'Stability' in feature or 'Interaction' in feature:
                df[feature] = df[feature].fillna(0)
            elif 'Prob' in feature:
                df[feature] = df[feature].fillna(0.5)
            else:
                df[feature] = df[feature].fillna(df[feature].median() if df[feature].notna().any() else 0)

    return df

# ============================================================
# 🎯 ENSEMBLE FOCADO EM FEATURES ANGULARES
# ============================================================

def criar_ensemble_angular(history, games_today):
    """
    Ensemble que prioriza features angulares + seleção inteligente
    """
    st.markdown("## 🧠 Iniciando Ensemble com Features Angulares...")
    
    # 1. Feature Engineering Completa
    with st.spinner("🧮 Calculando features 3D..."):
        history = calcular_distancias_3d(history)
        games_today = calcular_distancias_3d(games_today)
    
    with st.spinner("🧭 Aplicando clusterização 3D..."):
        history = aplicar_clusterizacao_3d(history)
        games_today = aplicar_clusterizacao_3d(games_today)
    
    with st.spinner("📐 Gerando features angulares..."):
        history = engenhar_features_angulares(history)
        games_today = engenhar_features_angulares(games_today)

    # 2. Feature Engineering para Cluster
    history['Cluster3D_Label'] = history['Cluster3D_Label'].astype(float)
    games_today['Cluster3D_Label'] = games_today['Cluster3D_Label'].astype(float)

    mean_c = history['Cluster3D_Label'].mean()
    std_c = history['Cluster3D_Label'].std(ddof=0) or 1
    history['C3D_ZScore'] = (history['Cluster3D_Label'] - mean_c) / std_c
    games_today['C3D_ZScore'] = (games_today['Cluster3D_Label'] - mean_c) / std_c

    history['C3D_Sin'] = np.sin(history['Cluster3D_Label'])
    history['C3D_Cos'] = np.cos(history['Cluster3D_Label'])
    games_today['C3D_Sin'] = np.sin(games_today['Cluster3D_Label'])
    games_today['C3D_Cos'] = np.cos(games_today['Cluster3D_Label'])

    # 3. Seleção de Features por Categoria
    feature_categories = {
        'Angular': [
            'Angle_Stability', 'Angle_Turbulence', 'Angular_Sync',
            'Vector_Direction_X', 'Vector_Direction_Y', 'Vector_Direction_Z',
            'Angle_Interaction_XY_XZ', 'Angle_Interaction_XY_YZ', 'Angle_Interaction_XZ_YZ',
            'Vector_Dominance_X', 'Vector_Dominance_Y', 'Vector_Dominance_Z',
            'Angle_Historical_Prob', 'Angular_Sign_Consistency'
        ],
        'Espacial': [
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Magnitude_3D',
            'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
            'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ', 'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
            'Vector_Sign'
        ],
        'Cluster': [
            'Cluster3D_Label', 'C3D_ZScore', 'C3D_Sin', 'C3D_Cos'
        ],
        'Baseline': [
            'Asian_Line_Decimal', 'Aggression_Home', 'Aggression_Away',
            'M_H', 'M_A', 'MT_H', 'MT_A'
        ]
    }
    
    # Coletar todas as features disponíveis
    all_features = []
    for category, features in feature_categories.items():
        available = [f for f in features if f in history.columns]
        all_features.extend(available)
        st.info(f"📊 {category}: {len(available)}/{len(features)} features")

    st.success(f"🎯 Total de features disponíveis: {len(all_features)}")

    # 4. Preparar dados de treino
    X = history[all_features].fillna(0)
    y = history['Target_AH_Home']

    # 5. Feature Selection Automática
    with st.spinner("🔍 Selecionando melhores features..."):
        selector_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        selector = SelectFromModel(selector_model, threshold='median')
        X_selected = selector.fit_transform(X, y)
        selected_features = np.array(all_features)[selector.get_support()]
        
        st.success(f"✅ Features selecionadas: {len(selected_features)}/{len(all_features)}")

    # 6. Modelo Principal (XGBoost)
    with st.spinner("🤖 Treinando modelo XGBoost..."):
        modelo_principal = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='logloss'
        )
        
        modelo_principal.fit(X_selected, y)

    # 7. Previsões para jogos de hoje
    with st.spinner("🔮 Fazendo previsões..."):
        X_today = games_today[all_features].fillna(0)
        X_today_selected = selector.transform(X_today)
        
        probas = modelo_principal.predict_proba(X_today_selected)[:, 1]
        
        games_today['Prob_Ensemble_Angular'] = probas
        games_today['ML_Side_Ensemble'] = np.where(probas > 0.5, 'HOME', 'AWAY')
        games_today['Confidence_Ensemble'] = np.maximum(probas, 1 - probas)

    # 8. Feature Importance
    with st.spinner("📊 Analisando importância das features..."):
        importancia = modelo_principal.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importancia
        }).sort_values('Importance', ascending=False)

    return games_today, modelo_principal, feat_imp_df, selected_features

# ============================================================
# 📊 VALIDAÇÃO E VISUALIZAÇÃO
# ============================================================

def validar_ensemble(games_today, feat_imp_df, selected_features):
    """Validação completa do ensemble angular"""
    
    st.markdown("## 📊 Resultados do Ensemble Angular")
    
    # 1. Métricas Principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_confidence = games_today['Confidence_Ensemble'].mean()
        st.metric("🎯 Confiança Média", f"{avg_confidence:.1%}")
    
    with col2:
        home_picks = (games_today['ML_Side_Ensemble'] == 'HOME').sum()
        st.metric("🏠 Recomendações HOME", home_picks)
    
    with col3:
        away_picks = (games_today['ML_Side_Ensemble'] == 'AWAY').sum()
        st.metric("✈️ Recomendações AWAY", away_picks)
    
    with col4:
        total_games = len(games_today)
        st.metric("📊 Total de Jogos", total_games)
    
    # 2. Feature Importance Plot
    st.markdown("### 📈 Top 15 Features Mais Importantes")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_15 = feat_imp_df.head(15)
    y_pos = np.arange(len(top_15))
    
    ax.barh(y_pos, top_15['Importance'], align='center', color='skyblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_15['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importância')
    ax.set_title('Top 15 Features por Importância')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 3. Tabela de Feature Importance
    st.markdown("### 📋 Detalhes das Features Selecionadas")
    st.dataframe(feat_imp_df.head(20), use_container_width=True)
    
    # 4. Distribuição das Previsões
    st.markdown("### 📊 Distribuição das Probabilidades")
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histograma de probabilidades
    ax1.hist(games_today['Prob_Ensemble_Angular'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_xlabel('Probabilidade')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição das Probabilidades')
    ax1.grid(alpha=0.3)
    
    # Distribuição por lado
    side_counts = games_today['ML_Side_Ensemble'].value_counts()
    ax2.pie(side_counts.values, labels=side_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribuição HOME vs AWAY')
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # 5. Tabela de Previsões Detalhadas
    st.markdown("### ⚽ Previsões Detalhadas para Hoje")
    
    display_cols = [
        'Home', 'Away', 'League', 'Asian_Line_Decimal',
        'Prob_Ensemble_Angular', 'ML_Side_Ensemble', 'Confidence_Ensemble'
    ]
    
    # Adicionar dados do Live Score se disponíveis
    if 'Goals_H_Today' in games_today.columns and 'Goals_A_Today' in games_today.columns:
        display_cols.extend(['Goals_H_Today', 'Goals_A_Today'])
    
    # Filtrar colunas que existem
    display_cols = [col for col in display_cols if col in games_today.columns]
    
    display_df = games_today[display_cols].copy()
    
    # Formatar colunas numéricas
    if 'Prob_Ensemble_Angular' in display_df.columns:
        display_df['Prob_Ensemble_Angular'] = display_df['Prob_Ensemble_Angular'].apply(lambda x: f"{x:.1%}")
    if 'Confidence_Ensemble' in display_df.columns:
        display_df['Confidence_Ensemble'] = display_df['Confidence_Ensemble'].apply(lambda x: f"{x:.1%}")
    if 'Asian_Line_Decimal' in display_df.columns:
        display_df['Asian_Line_Decimal'] = display_df['Asian_Line_Decimal'].apply(lambda x: f"{x:+.2f}")
    
    # Ordenar por confiança
    if 'Confidence_Ensemble' in games_today.columns:
        display_df = display_df.iloc[games_today['Confidence_Ensemble'].argsort()[::-1]]
    
    st.dataframe(display_df, use_container_width=True)

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    """Função principal do Approach 2"""
    
    st.sidebar.markdown("## ⚙️ Configurações do Ensemble")
    
    # ---------------- Carregar Dados ----------------
    st.info("📂 Carregando dados para o Ensemble Angular...")

    # Seleção de arquivo do dia
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    if not files:
        st.warning("No CSV files found in GamesDay folder.")
        st.stop()

    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.sidebar.selectbox("Select Matchday File:", options, index=len(options)-1)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

    # ---------------- CACHE INTELIGENTE ----------------
    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        """Cache apenas dos dados pesados"""
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        games_today = filter_leagues(games_today)
        
        history = filter_leagues(load_all_games(GAMES_FOLDER))
        history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
        
        return games_today, history

    games_today, history = load_cached_data(selected_file)

    # ---------------- LIVE SCORE INTEGRATION ----------------
    def load_and_merge_livescore(games_today, selected_date_str):
        """Carrega e faz merge dos dados do Live Score"""
        livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

        games_today = setup_livescore_columns(games_today)

        if os.path.exists(livescore_file):
            st.info(f"📡 LiveScore file found: {livescore_file}")
            results_df = pd.read_csv(livescore_file)
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
                games_today = games_today.merge(
                    results_df,
                    left_on='Id',
                    right_on='Id',
                    how='left',
                    suffixes=('', '_RAW')
                )

                games_today['Goals_H_Today'] = games_today['home_goal']
                games_today['Goals_A_Today'] = games_today['away_goal']
                games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

                games_today['Home_Red'] = games_today['home_red']
                games_today['Away_Red'] = games_today['away_red']

                st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
                return games_today
        else:
            st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
            return games_today

    games_today = load_and_merge_livescore(games_today, selected_date_str)

    # ---------------- PREPARAÇÃO DO HISTÓRICO ----------------
    # Aplicar conversão Asian Line
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

    # Filtrar histórico com linha válida
    history = history.dropna(subset=['Asian_Line_Decimal'])
    
    # Filtro anti-leakage temporal
    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
        except Exception as e:
            st.error(f"Erro ao aplicar filtro temporal: {e}")

    # Calcular targets
    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(calculate_ah_home_target, axis=1)
    history = history.dropna(subset=["Target_AH_Home"]).copy()
    history["Target_AH_Home"] = history["Target_AH_Home"].astype(int)

    # Calcular Momentum do Time
    history = calcular_momentum_time(history)
    games_today = calcular_momentum_time(games_today)

    # ---------------- BOTÃO DE EXECUÇÃO ----------------
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Executar Ensemble Angular", type="primary"):
        with st.spinner("Executando Ensemble Angular completo..."):
            try:
                games_resultado, modelo, feat_imp_df, selected_features = criar_ensemble_angular(history, games_today)
                validar_ensemble(games_resultado, feat_imp_df, selected_features)
                
                # Salvar resultados
                games_resultado.to_csv(f"ensemble_angular_{selected_date_str}.csv", index=False)
                st.success(f"💾 Resultados salvos em: ensemble_angular_{selected_date_str}.csv")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Erro durante a execução: {e}")
                import traceback
                st.error(traceback.format_exc())
    else:
        st.info("👆 Clique em 'Executar Ensemble Angular' no sidebar para iniciar")

    # ---------------- INFO ADICIONAL ----------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **🎯 Sobre o Approach 2:**
    - 🤖 XGBoost com features angulares
    - 🔍 Seleção automática de features
    - 📊 Feature importance detalhada
    - 🧮 Features de estabilidade vetorial
    - ⚡ Mais rápido e interpretável
    """)

if __name__ == "__main__":
    main()
