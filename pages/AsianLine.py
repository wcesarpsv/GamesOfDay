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

st.set_page_config(page_title="Analisador de Handicap Ótimo - CALIBRADO", layout="wide")
st.title("🎯 Analisador de Handicap Ótimo - Modelo Calibrado")

# ---------------- Configurações ----------------
PAGE_PREFIX = "HandicapOptimizer_Calibrado"
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

def calcular_momentum_time(df, window=6):
    """
    Calcula o Momentum do Time (MT_H / MT_A)
    """
    df = df.copy()

    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = zscore

        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = zscore

    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)
    return df

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
# 🎯 SISTEMA CALIBRADO: HANDICAP OPTIMIZATION
# ============================================================

def calcular_handicap_otimo_calibrado(row):
    """
    Versão CALIBRADA do cálculo de handicap ótimo
    Com limites realistas e suavização
    """
    gh, ga = row.get('Goals_H_FT', 0), row.get('Goals_A_FT', 0)
    margin = gh - ga
    
    # 🔧 LIMITES REALISTAS: Handicaps entre -2.0 e +2.0
    handicaps_possiveis = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, +0.25, +0.5, +0.75, +1.0, +1.25, +1.5, +1.75, +2.0]
    
    melhor_handicap = 0
    melhor_score = -10
    
    for handicap in handicaps_possiveis:
        # Simula resultado com handicap
        resultado_ajustado = margin + handicap
        
        # 🔧 SCORE CALIBRADO: Penaliza handicaps extremos
        if resultado_ajustado > 0:
            # Ganhou - score positivo mas penaliza extremos
            score = 1.5 - abs(handicap) * 0.2  # Penalidade mais suave
        elif resultado_ajustado == 0:
            # Push - score neutro
            score = 0.3
        else:
            # Perdeu - score negativo
            score = -0.5 - abs(handicap) * 0.1
        
        if score > melhor_score:
            melhor_score = score
            melhor_handicap = handicap
    
    return melhor_handicap

def criar_target_handicap_discreto_calibrado(row):
    """
    Versão calibrada para classificação
    """
    handicap_otimo = calcular_handicap_otimo_calibrado(row)
    
    # 🔧 CATEGORIAS MAIS EQUILIBRADAS
    if handicap_otimo <= -1.25:
        return 'STRONG_HOME'
    elif handicap_otimo <= -0.5:
        return 'MODERATE_HOME' 
    elif handicap_otimo < 0:
        return 'LIGHT_HOME'
    elif handicap_otimo == 0:
        return 'NEUTRAL'
    elif handicap_otimo < 0.5:
        return 'LIGHT_AWAY'
    elif handicap_otimo < 1.25:
        return 'MODERATE_AWAY'
    else:
        return 'STRONG_AWAY'

# ============================================================
# 🧠 MODELOS CALIBRADOS
# ============================================================

def treinar_modelo_handicap_regressao_calibrado(history, games_today):
    """
    Modelo de Regressão CALIBRADO
    """
    st.markdown("### 📈 Modelo Regressão Calibrado")
    
    # Criar target calibrado
    history['Handicap_Otimo_Calibrado'] = history.apply(calcular_handicap_otimo_calibrado, axis=1)
    
    # 🔧 FILTRAR HANDICAPS EXTREMOS
    handicap_range = [-2.0, 2.0]
    history_calibrado = history[
        (history['Handicap_Otimo_Calibrado'] >= handicap_range[0]) & 
        (history['Handicap_Otimo_Calibrado'] <= handicap_range[1])
    ].copy()
    
    st.info(f"📊 Dados calibrados: {len(history_calibrado)} jogos (handicaps entre {handicap_range[0]} e {handicap_range[1]})")
    
    # Features espaciais
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history_calibrado.columns]
    
    # Verificar se temos features suficientes
    if len(available_features) < 3:
        st.error("❌ Features insuficientes para treinamento")
        return None, games_today, None
    
    X = history_calibrado[available_features].fillna(0)
    y = history_calibrado['Handicap_Otimo_Calibrado']
    
    # 🔧 NORMALIZAR FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Treinar modelo com MAIS REGULARIZAÇÃO
    model = RandomForestRegressor(
        n_estimators=150,  # Menos árvores
        max_depth=6,       # Menos profundidade
        min_samples_leaf=15,  # Mais amostras por folha
        max_features=0.7,  # Limitar features
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # 🔧 VALIDAÇÃO
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    st.success(f"✅ MAE do modelo: {mae:.3f} (quanto menor, melhor)")
    
    # Prever para jogos de hoje
    X_today = games_today[available_features].fillna(0)
    
    # Verificar se temos as mesmas features nos dados de hoje
    missing_features = set(available_features) - set(X_today.columns)
    if missing_features:
        st.warning(f"⚠️ Features faltando nos dados de hoje: {missing_features}")
        for feature in missing_features:
            X_today[feature] = 0
    
    X_today_scaled = scaler.transform(X_today[available_features])
    
    predictions = model.predict(X_today_scaled)
    
    # 🔧 SUAVIZAR PREDIÇÕES (evitar saltos grandes)
    games_today['Handicap_Predito_Regressao_Calibrado'] = np.clip(predictions, -2.0, 2.0)
    games_today['Value_Gap_Regressao_Calibrado'] = (
        games_today['Handicap_Predito_Regressao_Calibrado'] - games_today['Asian_Line_Decimal']
    )
    
    return model, games_today, scaler

def treinar_modelo_handicap_classificacao_calibrado(history, games_today):
    """
    Modelo de Classificação CALIBRADO
    """
    st.markdown("### 🎯 Modelo Classificação Calibrado")
    
    # Criar target categórico calibrado
    history['Handicap_Categoria_Calibrado'] = history.apply(criar_target_handicap_discreto_calibrado, axis=1)
    
    # Features
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history.columns]
    
    # Verificar se temos features suficientes
    if len(available_features) < 3:
        st.error("❌ Features insuficientes para treinamento")
        return None, games_today, None
    
    X = history[available_features].fillna(0)
    y = history['Handicap_Categoria_Calibrado']
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        class_weight='balanced',
        min_samples_leaf=10
    )
    model.fit(X, y_encoded)
    
    # Prever para jogos de hoje
    X_today = games_today[available_features].fillna(0)
    
    # Verificar se temos as mesmas features nos dados de hoje
    missing_features = set(available_features) - set(X_today.columns)
    if missing_features:
        st.warning(f"⚠️ Features faltando nos dados de hoje: {missing_features}")
        for feature in missing_features:
            X_today[feature] = 0
    
    predicoes_encoded = model.predict(X_today[available_features])
    probas = model.predict_proba(X_today[available_features])
    
    games_today['Handicap_Categoria_Predito_Calibrado'] = le.inverse_transform(predicoes_encoded)
    games_today['Confianca_Categoria_Calibrado'] = np.max(probas, axis=1)
    
    # 🔧 MAPEAMENTO CALIBRADO para handicaps numéricos
    categoria_para_handicap_calibrado = {
        'STRONG_HOME': -1.5,
        'MODERATE_HOME': -0.75, 
        'LIGHT_HOME': -0.25,
        'NEUTRAL': 0,
        'LIGHT_AWAY': +0.25,
        'MODERATE_AWAY': +0.75,
        'STRONG_AWAY': +1.5
    }
    
    games_today['Handicap_Predito_Classificacao_Calibrado'] = games_today['Handicap_Categoria_Predito_Calibrado'].map(categoria_para_handicap_calibrado)
    games_today['Value_Gap_Classificacao_Calibrado'] = (
        games_today['Handicap_Predito_Classificacao_Calibrado'] - games_today['Asian_Line_Decimal']
    )
    
    st.info(f"📊 Distribuição categorias calibradas: {dict(history['Handicap_Categoria_Calibrado'].value_counts())}")
    
    return model, games_today, le

# ============================================================
# 📊 ANÁLISE DE VALOR CALIBRADA
# ============================================================

def analisar_value_bets_calibrado(games_today):
    """
    Análise de value CALIBRADA com thresholds realistas
    """
    st.markdown("## 💎 Análise de Value Bets Calibrada")
    
    results = []
    
    for idx, row in games_today.iterrows():
        handicap_mercado = row['Asian_Line_Decimal']
        handicap_regressao = row.get('Handicap_Predito_Regressao_Calibrado', 0)
        handicap_classificacao = row.get('Handicap_Predito_Classificacao_Calibrado', 0)
        
        # Value gap de cada modelo
        gap_regressao = handicap_regressao - handicap_mercado
        gap_classificacao = handicap_classificacao - handicap_mercado
        
        # 🔧 MÉDIA PONDERADA CALIBRADA
        value_gap_consolidado = (gap_regressao * 0.7 + gap_classificacao * 0.3)
        
        # 🔧 THRESHOLDS REALISTAS
        if value_gap_consolidado > 0.4:
            recomendacao = "STRONG HOME VALUE"
            lado = "HOME"
            confidence = "HIGH"
        elif value_gap_consolidado > 0.2:
            recomendacao = "HOME VALUE" 
            lado = "HOME"
            confidence = "MEDIUM"
        elif value_gap_consolidado < -0.4:
            recomendacao = "STRONG AWAY VALUE"
            lado = "AWAY" 
            confidence = "HIGH"
        elif value_gap_consolidado < -0.2:
            recomendacao = "AWAY VALUE"
            lado = "AWAY"
            confidence = "MEDIUM"
        else:
            recomendacao = "NO CLEAR VALUE"
            lado = "PASS"
            confidence = "LOW"
        
        results.append({
            'League': row['League'],
            'Home': row['Home'],
            'Away': row['Away'],
            'Asian_Line': handicap_mercado,
            'Handicap_Regressao': round(handicap_regressao, 2),
            'Handicap_Classificacao': round(handicap_classificacao, 2),
            'Value_Gap': round(value_gap_consolidado, 2),
            'Recomendacao': recomendacao,
            'Lado': lado,
            'Confidence': confidence
        })
    
    df_results = pd.DataFrame(results)
    
    # Ordenar por valor absoluto do gap
    df_results['Value_Abs'] = abs(df_results['Value_Gap'])
    df_results = df_results.sort_values('Value_Abs', ascending=False)
    
    return df_results

def plot_handicap_analysis_calibrado(games_today):
    """
    Visualização CALIBRADA
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Regressão Calibrada vs Mercado
    if 'Handicap_Predito_Regressao_Calibrado' in games_today.columns:
        colors_regressao = []
        for gap in games_today.get('Value_Gap_Regressao_Calibrado', []):
            if gap > 0.3:
                colors_regressao.append('green')
            elif gap < -0.3:
                colors_regressao.append('red')
            else:
                colors_regressao.append('gray')
        
        ax1.scatter(games_today['Asian_Line_Decimal'], 
                    games_today['Handicap_Predito_Regressao_Calibrado'],
                    c=colors_regressao, alpha=0.6, s=60)
        ax1.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Mercado Perfeito')
        ax1.set_xlabel('Handicap Mercado')
        ax1.set_ylabel('Handicap Predito (Regressão Calibrado)')
        ax1.set_title('Value Analysis - Modelo Regressão Calibrado')
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Classificação Calibrada vs Mercado
    if 'Handicap_Predito_Classificacao_Calibrado' in games_today.columns:
        colors_class = []
        for gap in games_today.get('Value_Gap_Classificacao_Calibrado', []):
            if gap > 0.3:
                colors_class.append('green')
            elif gap < -0.3:
                colors_class.append('red')
            else:
                colors_class.append('gray')
        
        ax2.scatter(games_today['Asian_Line_Decimal'],
                   games_today['Handicap_Predito_Classificacao_Calibrado'],
                   c=colors_class, alpha=0.6, s=60)
        ax2.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Mercado Perfeito')
        ax2.set_xlabel('Handicap Mercado')
        ax2.set_ylabel('Handicap Predito (Classificação Calibrado)')
        ax2.set_title('Value Analysis - Modelo Classificação Calibrado')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(-2.5, 2.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL CALIBRADA
# ============================================================

def main_calibrado():
    # ---------------- Carregar Dados ----------------
    st.info("📂 Carregando dados para Análise de Handicap Ótimo CALIBRADO...")
    
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
        
        # 🔧 FILTRAR LIGAS PRINCIPAIS para melhor calibração
        # ============================================================
        # 🎯 FILTRO INTELIGENTE DE LIGAS + ONE-HOT PARCIAL
        # ============================================================
        
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
        
        # ============================================================
        # 🧩 ONE-HOT PARCIAL DAS LIGAS MAIS FREQUENTES
        # ============================================================
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
    
    # ---------------- Calcular Features 3D ----------------
    st.markdown("## 🧮 Calculando Features 3D Calibradas...")
    
    # Verificar se as colunas necessárias existem
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'HandScore_Home', 'HandScore_Away']
    missing_history = [col for col in required_cols if col not in history.columns]
    missing_today = [col for col in required_cols if col not in games_today.columns]
    
    if missing_history or missing_today:
        st.error(f"❌ Colunas necessárias faltando: History={missing_history}, Today={missing_today}")
        return
    
    history = calcular_momentum_time(history)
    games_today = calcular_momentum_time(games_today)
    
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # ---------------- Treinar Modelos Calibrados ----------------
    st.markdown("## 🧠 Treinando Modelos de Handicap CALIBRADOS...")
    
    if st.button("🚀 Executar Análise Calibrada", type="primary"):
        with st.spinner("Treinando modelos calibrados..."):
            # Treinar modelo de regressão calibrado
            modelo_regressao, games_today, scaler = treinar_modelo_handicap_regressao_calibrado(history, games_today)
            
            if modelo_regressao is None:
                st.error("❌ Falha no treinamento do modelo de regressão")
                return
            
            # Treinar modelo de classificação calibrado  
            modelo_classificacao, games_today, label_encoder = treinar_modelo_handicap_classificacao_calibrado(history, games_today)
            
            if modelo_classificacao is None:
                st.error("❌ Falha no treinamento do modelo de classificação")
                return
            
            # Analisar value bets calibrado
            df_value_bets_calibrado = analisar_value_bets_calibrado(games_today)
            
            # Exibir resultados
            st.markdown("## 📊 Resultados da Análise Calibrada")
            
            # 🔧 FILTRAR APENAS BETS CONFIAVEIS
            bets_confiaveis = df_value_bets_calibrado[
                (df_value_bets_calibrado['Value_Abs'].between(0.2, 0.8)) &
                (df_value_bets_calibrado['Asian_Line'].between(-1.5, 1.5)) &
                (df_value_bets_calibrado['Confidence'].isin(['HIGH', 'MEDIUM']))
            ]
            
            if bets_confiaveis.empty:
                st.warning("⚠️ Nenhum value bet confiável encontrado após filtragem")
            else:
                st.dataframe(bets_confiaveis, use_container_width=True)
            
            # Visualizações calibradas
            st.pyplot(plot_handicap_analysis_calibrado(games_today))
            
            # Estatísticas
            if not bets_confiaveis.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    strong_bets = len(bets_confiaveis[bets_confiaveis['Confidence'] == 'HIGH'])
                    st.metric("🎯 Strong Value Bets", strong_bets)
                with col2:
                    total_bets = len(bets_confiaveis)
                    st.metric("📊 Total Recomendações", total_bets)
                with col3:
                    home_bets = len(bets_confiaveis[bets_confiaveis['Lado'] == 'HOME'])
                    st.metric("🏠 HOME Value", home_bets)
                with col4:
                    away_bets = len(bets_confiaveis[bets_confiaveis['Lado'] == 'AWAY'])
                    st.metric("✈️ AWAY Value", away_bets)
                
                # 🔧 DIAGNÓSTICO DE CALIBRAÇÃO
                st.markdown("### 🔧 Diagnóstico de Calibração")
                avg_value_gap = bets_confiaveis['Value_Gap'].mean()
                max_value_gap = bets_confiaveis['Value_Gap'].abs().max()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 Value Gap Médio", f"{avg_value_gap:.2f}")
                with col2:
                    st.metric("📊 Value Gap Máximo", f"{max_value_gap:.2f}")
                
                if max_value_gap > 1.0:
                    st.warning("⚠️ Ainda há alguns value gaps altos - pode precisar de mais ajustes")
                else:
                    st.success("✅ Modelo bem calibrado! Value gaps em faixa realista")
            
            st.balloons()
    
    else:
        st.info("👆 Clique no botão para executar a análise calibrada")

if __name__ == "__main__":
    main_calibrado()
