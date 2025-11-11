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
from sklearn.cluster import KMeans
from scipy import stats

st.set_page_config(page_title="Sistema Espacial Inteligente - Bet Indicator", layout="wide")
st.title("🎯 Sistema Espacial Inteligente com Otimização Automática")

# ---------------- Configurações ----------------
PAGE_PREFIX = "Espacial_Inteligente"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------
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
    """Converte handicaps asiáticos para decimal"""
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

# ============================================================
# 🎯 CÁLCULO DO TARGET BASE
# ============================================================
def calculate_ah_home_target(row):
    """Calcula o target binário para handicap asiático"""
    gh = row.get("Goals_H_FT")
    ga = row.get("Goals_A_FT")
    line_home = row.get("Asian_Line_Decimal")

    if pd.isna(gh) or pd.isna(ga) or pd.isna(line_home):
        return np.nan

    adjusted = (gh + line_home) - ga
    return 1 if adjusted > 0 else 0

# ============================================================
# 🧠 CÁLCULO DE MOMENTUM DO TIME
# ============================================================
def calcular_momentum_time(df, window=6):
    """Calcula MT_H e MT_A com base no HandScore"""
    df = df.copy()

    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        # Momentum como mandante
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = zscore

        # Momentum como visitante
        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = zscore

    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)
    return df

# ============================================================
# 🧩 CLUSTERIZAÇÃO 3D
# ============================================================
def aplicar_clusterizacao_3d(df, n_clusters=4, random_state=42):
    """Cria clusters espaciais 3D"""
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes: {missing}")
        df['Cluster3D_Label'] = -1
        return df

    # Vetor 3D de diferenças
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init=10)
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    # Descrição dos clusters
    df['Cluster3D_Desc'] = df['Cluster3D_Label'].map({
        0: '⚡ Agressivos + Momentum Positivo',
        1: '💤 Reativos + Momentum Negativo', 
        2: '⚖️ Equilibrados',
        3: '🔥 Alta Variância'
    }).fillna('🌀 Outro')

    return df

# ============================================================
# 📐 CÁLCULO DE DISTÂNCIAS 3D
# ============================================================
def calcular_distancias_3d(df):
    """Calcula métricas espaciais 3D"""
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Colunas faltando: {missing_cols}")
        for col in ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Quadrant_Angle_XY', 
                   'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Vector_Sign', 'Magnitude_3D']:
            df[col] = np.nan
        return df

    # Normalização dos eixos
    dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
    dy = (df['M_H'] - df['M_A']) / 2
    dz = (df['MT_H'] - df['MT_A']) / 2

    # Métricas espaciais
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    
    angle_xy = np.arctan2(dy, dx)
    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df

# ============================================================
# 🎯 SISTEMA DE OTIMIZAÇÃO DE ÂNGULOS
# ============================================================
def encontrar_angulo_otimo(history, target_col='Target_AH_Home', min_samples=100):
    """Encontra automaticamente o ângulo ótimo para separação estável/instável"""
    try:
        st.markdown("### 🔍 Buscando Ângulo Ótimo")
        
        if 'Quadrant_Angle_XY' not in history.columns:
            st.warning("❌ Coluna Quadrant_Angle_XY não encontrada")
            return 40
        
        resultados = []
        angulos_testar = range(10, 80, 5)  # 10° a 75° em passos de 5°
        
        progress_bar = st.progress(0)
        total_angulos = len(angulos_testar)
        
        for i, angulo_limite in enumerate(angulos_testar):
            progress_bar.progress((i + 1) / total_angulos)
            
            # Separar estáveis vs instáveis
            mask_estavel = history['Quadrant_Angle_XY'].abs() < angulo_limite
            mask_instavel = history['Quadrant_Angle_XY'].abs() >= angulo_limite
            
            # Métricas para estáveis
            if mask_estavel.sum() >= min_samples:
                estavel_data = history[mask_estavel]
                acuracia_estavel = estavel_data[target_col].mean()
                volume_estavel = len(estavel_data)
                roi_estavel = (acuracia_estavel * 0.90 - (1 - acuracia_estavel)) * 100
            else:
                acuracia_estavel = volume_estavel = roi_estavel = 0
            
            # Métricas para instáveis
            if mask_instavel.sum() >= min_samples:
                instavel_data = history[mask_instavel]
                acuracia_instavel = instavel_data[target_col].mean()
                volume_instavel = len(instavel_data)
                roi_instavel = (acuracia_instavel * 0.90 - (1 - acuracia_instavel)) * 100
            else:
                acuracia_instavel = volume_instavel = roi_instavel = 0
            
            # Score de qualidade
            if volume_estavel >= min_samples and volume_instavel >= min_samples:
                diferenca_acuracia = acuracia_estavel - acuracia_instavel
                score_qualidade = diferenca_acuracia * (volume_estavel + volume_instavel) / 2000
            else:
                score_qualidade = -1
            
            resultados.append({
                'angulo_limite': angulo_limite,
                'acuracia_estavel': acuracia_estavel,
                'volume_estavel': volume_estavel,
                'roi_estavel': roi_estavel,
                'acuracia_instavel': acuracia_instavel,
                'volume_instavel': volume_instavel,
                'diferenca_acuracia': diferenca_acuracia,
                'score_qualidade': score_qualidade
            })
        
        # Encontrar ângulo ótimo
        df_resultados = pd.DataFrame(resultados)
        df_validos = df_resultados[df_resultados['score_qualidade'] > 0]
        
        if len(df_validos) > 0:
            angulo_otimo_idx = df_validos['score_qualidade'].idxmax()
            angulo_otimo = df_validos.loc[angulo_otimo_idx, 'angulo_limite']
            melhor_score = df_validos.loc[angulo_otimo_idx, 'score_qualidade']
            
            st.success(f"🎯 Ângulo ótimo encontrado: {angulo_otimo}° (Score: {melhor_score:.4f})")
            
            # Top 5 ângulos
            st.markdown("#### 📊 Top 5 Ângulos por Performance")
            top_5 = df_validos.nlargest(5, 'score_qualidade')[
                ['angulo_limite', 'acuracia_estavel', 'volume_estavel', 
                 'acuracia_instavel', 'volume_instavel', 'diferenca_acuracia']
            ].round(4)
            
            st.dataframe(top_5.style.format({
                'acuracia_estavel': '{:.1%}',
                'acuracia_instavel': '{:.1%}',
                'diferenca_acuracia': '{:.1%}'
            }))
            
            # Gráfico de performance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_resultados['angulo_limite'], df_resultados['acuracia_estavel'], 
                   label='Zona Estável', marker='o', linewidth=2, color='green')
            ax.plot(df_resultados['angulo_limite'], df_resultados['acuracia_instavel'],
                   label='Zona Instável', marker='s', linewidth=2, color='red')
            ax.axvline(x=angulo_otimo, color='blue', linestyle='--', alpha=0.7, 
                      label=f'Ângulo Ótimo: {angulo_otimo}°')
            ax.set_xlabel('Ângulo Limite (°)')
            ax.set_ylabel('Acurácia')
            ax.set_title('Performance por Faixa Angular')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            return angulo_otimo
        else:
            st.warning("⚠️ Nenhum ângulo válido encontrado. Usando padrão: 40°")
            return 40
            
    except Exception as e:
        st.error(f"❌ Erro na otimização: {e}")
        return 40

# ============================================================
# 🧠 TARGET ESPACIAL INTELIGENTE (VERSÃO OTIMIZADA)
# ============================================================
def criar_target_espacial_inteligente(row, angulo_limite=40):
    """
    Target espacial inteligente com ângulo otimizado
    Versão balanceada e simétrica
    """
    try:
        dx = row.get('Aggression_Home', 0) - row.get('Aggression_Away', 0)
        dy = row.get('M_H', 0) - row.get('M_A', 0)
        dz = row.get('MT_H', 0) - row.get('MT_A', 0)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if np.isnan(dx) or np.isnan(dy) or np.isnan(dz):
            return 0

        # Ângulo otimizado dinamicamente
        angulo_xy = np.degrees(np.arctan2(dy, dx))
        angulo_estavel = abs(angulo_xy) < angulo_limite
        
        distancia_ok = dist > 0.6
        cluster_val = row.get('Cluster3D_Label', 0)
        cluster_confiavel = cluster_val in [0, 2]  # Clusters estáveis

        # Lógica de valor simétrica
        if distancia_ok and cluster_confiavel:
            # HOME value: Aggression positivo + Momentum positivo + Ângulo estável
            if dx > 0 and dz >= 0 and angulo_estavel:
                return 1
            # AWAY value: Aggression negativo + Momentum negativo + Ângulo instável
            elif dx < 0 and dz < 0 and not angulo_estavel:
                return 0
            else:
                # Neutro - distribuição balanceada
                return np.random.choice([0, 1])
        else:
            # Fallback baseado na direção principal
            if dx > 0.2:
                return 1
            elif dx < -0.2:
                return 0
            else:
                return np.random.choice([0, 1])
                
    except Exception as e:
        return 0

# ============================================================
# 🚀 TREINAMENTO DO MODELO ESPACIAL
# ============================================================
def treinar_modelo_espacial_inteligente(history, games_today):
    """
    Treina o modelo espacial inteligente com otimização automática
    """
    st.markdown("## 🧠 Treinando Modelo Espacial Inteligente")
    
    # 1. OTIMIZAR ÂNGULO
    angulo_otimo = encontrar_angulo_otimo(history)
    
    # 2. APLICAR TARGET ESPACIAL COM ÂNGULO OTIMIZADO
    st.info(f"🔄 Criando Target Espacial com ângulo ótimo: {angulo_otimo}°")
    history['Target_Espacial'] = history.apply(
        lambda x: criar_target_espacial_inteligente(x, angulo_otimo), 
        axis=1
    )
    
    # Estatísticas do target
    dist_espacial = history['Target_Espacial'].value_counts().to_dict()
    st.info(f"📊 Distribuição Target Espacial: {dist_espacial}")
    
    # 3. PREPARAR FEATURES
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # Feature engineering para clusters
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
    
    # 4. FEATURES FINAIS
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    
    features_espaciais = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Angle_XY',
        'Vector_Sign', 'Magnitude_3D',
        'C3D_ZScore', 'C3D_Sin', 'C3D_Cos'
    ]
    
    available_features = [f for f in features_espaciais if f in history.columns]
    st.info(f"🔧 Features espaciais utilizadas: {available_features}")
    
    # 5. TREINAR MODELO
    X = pd.concat([ligas_dummies, history[available_features]], axis=1).fillna(0)
    y = history['Target_Espacial'].astype(int)
    
    # Balanceamento de classes
    from sklearn.utils import resample
    major = history[history['Target_Espacial'] == 0]
    minor = history[history['Target_Espacial'] == 1]
    
    if len(minor) > 10 and len(major) > 10:
        minor_upsampled = resample(minor, replace=True, n_samples=len(major), random_state=42)
        history_balanced = pd.concat([major, minor_upsampled])
        X_balanced = pd.concat([ligas_dummies, history_balanced[available_features]], axis=1).fillna(0)
        y_balanced = history_balanced['Target_Espacial'].astype(int)
        st.info(f"⚖️ Dataset balanceado: {len(major)} x {len(minor_upsampled)}")
    else:
        X_balanced, y_balanced = X, y
        st.warning("⚠️ Dataset muito pequeno para balanceamento")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model.fit(X_balanced, y_balanced)
    
    # 6. PREVISÕES
    ligas_today = pd.get_dummies(games_today['League'], prefix='League')
    ligas_today = ligas_today.reindex(columns=ligas_dummies.columns, fill_value=0)
    
    X_today = pd.concat([ligas_today, games_today[available_features]], axis=1).fillna(0)
    
    # Garantir mesma ordem de features
    missing_cols = set(X.columns) - set(X_today.columns)
    for col in missing_cols:
        X_today[col] = 0
    X_today = X_today[X.columns]
    
    proba = model.predict_proba(X_today)[:, 1]
    proba = np.clip(proba, 0.05, 0.95)  # Evitar extremos
    
    # 7. RESULTADOS
    games_today['Prob_Espacial'] = proba
    games_today['ML_Side_Espacial'] = np.where(proba >= 0.5, 'HOME', 'AWAY')
    games_today['Confidence_Espacial'] = np.round(np.maximum(proba, 1 - proba), 3)
    games_today['Angulo_Otimizado'] = angulo_otimo
    
    # Estatísticas finais
    mean_conf = games_today['Confidence_Espacial'].mean()
    home_preds = (games_today['ML_Side_Espacial'] == 'HOME').sum()
    away_preds = (games_today['ML_Side_Espacial'] == 'AWAY').sum()
    
    st.success(f"✅ Modelo treinado: {len(games_today)} jogos | Confiança média: {mean_conf:.1%}")
    st.success(f"🏠 HOME: {home_preds} | ✈️ AWAY: {away_preds}")
    
    return model, games_today

# ============================================================
# 📊 VISUALIZAÇÃO DE RESULTADOS
# ============================================================
def exibir_resultados_espaciais(games_today):
    """Exibe resultados do modelo espacial"""
    st.markdown("## 📊 Resultados do Modelo Espacial Inteligente")
    
    # Filtrar colunas relevantes
    cols_display = [
        'League', 'Home', 'Away', 'Asian_Line_Decimal',
        'Goals_H_Today', 'Goals_A_Today', 'Prob_Espacial', 
        'ML_Side_Espacial', 'Confidence_Espacial', 'Angulo_Otimizado'
    ]
    
    cols_display = [c for c in cols_display if c in games_today.columns]
    
    resultados_df = games_today[cols_display].copy()
    
    # Formatar probabilidades
    if 'Prob_Espacial' in resultados_df.columns:
        resultados_df['Prob_Espacial'] = resultados_df['Prob_Espacial'].apply(lambda x: f"{x:.1%}")
    if 'Confidence_Espacial' in resultados_df.columns:
        resultados_df['Confidence_Espacial'] = resultados_df['Confidence_Espacial'].apply(lambda x: f"{x:.1%}")
    
    # Ordenar por confiança
    if 'Confidence_Espacial' in games_today.columns:
        resultados_df = resultados_df.sort_values('Confidence_Espacial', ascending=False)
    
    st.dataframe(resultados_df, use_container_width=True)
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Jogos", len(games_today))
    with col2:
        home_count = (games_today['ML_Side_Espacial'] == 'HOME').sum()
        st.metric("🏠 Recomendações HOME", home_count)
    with col3:
        away_count = (games_today['ML_Side_Espacial'] == 'AWAY').sum()
        st.metric("✈️ Recomendações AWAY", away_count)
    with col4:
        if 'Confidence_Espacial' in games_today.columns:
            conf_media = games_today['Confidence_Espacial'].mean()
            st.metric("🎯 Confiança Média", f"{conf_media:.1%}")

# ============================================================
# 🎯 FUNÇÃO PRINCIPAL
# ============================================================
def main():
    """Função principal do sistema espacial inteligente"""
    
    # ---------------- CARREGAR DADOS ----------------
    st.sidebar.markdown("## ⚙️ Configurações")
    
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    if not files:
        st.error("❌ Nenhum arquivo CSV encontrado na pasta GamesDay")
        return
    
    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.sidebar.selectbox("Selecionar Arquivo:", options, index=len(options)-1)
    
    # Carregar dados com cache
    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        games_today = filter_leagues(games_today)
        
        history = filter_leagues(load_all_games(GAMES_FOLDER))
        history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
        
        return games_today, history
    
    games_today, history = load_cached_data(selected_file)
    
    # ---------------- LIVE SCORE ----------------
    def load_and_merge_livescore(games_today, selected_date_str):
        livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
        games_today = setup_livescore_columns(games_today)
        
        if os.path.exists(livescore_file):
            results_df = pd.read_csv(livescore_file)
            results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
            
            required_cols = ['Id', 'status', 'home_goal', 'away_goal', 'home_red', 'away_red']
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            
            if not missing_cols:
                games_today = games_today.merge(
                    results_df, left_on='Id', right_on='Id', how='left', suffixes=('', '_RAW')
                )
                games_today['Goals_H_Today'] = games_today['home_goal']
                games_today['Goals_A_Today'] = games_today['away_goal']
                games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
                games_today['Home_Red'] = games_today['home_red']
                games_today['Away_Red'] = games_today['away_red']
                st.success("✅ LiveScore integrado")
        
        return games_today
    
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    games_today = load_and_merge_livescore(games_today, selected_date_str)
    
    # ---------------- PREPARAR DADOS ----------------
    st.info(f"📊 Carregados: {len(games_today)} jogos de hoje | {len(history)} histórico")
    
    # Converter Asian Line
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    
    history = history.dropna(subset=['Asian_Line_Decimal'])
    
    # Target base
    history["Target_AH_Home"] = history.apply(calculate_ah_home_target, axis=1)
    history = history.dropna(subset=["Target_AH_Home"]).copy()
    history["Target_AH_Home"] = history["Target_AH_Home"].astype(int)
    
    # Calcular momentum
    history = calcular_momentum_time(history)
    games_today = calcular_momentum_time(games_today)
    
    # ---------------- INTERFACE PRINCIPAL ----------------
    st.markdown("## 🎯 Sistema Espacial Inteligente")
    
    st.info("""
    **🎯 Este sistema utiliza:**
    - 🧠 **Otimização automática** de ângulos espaciais
    - 📐 **Análise 3D** com Aggression, Momentum e MT
    - 🎯 **Target inteligente** simétrico (HOME/AWAY)
    - ⚡ **Clusterização** para identificar padrões
    """)
    
    # Botão de treinamento
    if st.sidebar.button("🚀 Treinar Modelo Espacial", type="primary"):
        with st.spinner("Treinando modelo espacial inteligente..."):
            modelo, resultados = treinar_modelo_espacial_inteligente(history, games_today)
            
            if modelo is not None:
                exibir_resultados_espaciais(resultados)
                
                # Salvar resultados
                resultados.to_csv(f"resultados_espacial_{selected_date_str}.csv", index=False)
                st.success(f"💾 Resultados salvos em: resultados_espacial_{selected_date_str}.csv")
                st.balloons()
            else:
                st.error("❌ Falha no treinamento do modelo")
    
    else:
        st.info("👆 Clique em **'Treinar Modelo Espacial'** para iniciar a análise")

# ============================================================
# 🚀 EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    main()
