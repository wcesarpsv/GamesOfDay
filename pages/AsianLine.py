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
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Analisador de Handicap Ótimo - Bet Indicator", layout="wide")
st.title("🎯 Analisador de Handicap Ótimo - Modelo 3D Avançado")

# ---------------- Configurações ----------------
PAGE_PREFIX = "HandicapOptimizer_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------
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
# 🎯 NOVO SISTEMA: HANDICAP OPTIMIZATION
# ============================================================

def calcular_handicap_otimo_real(row):
    """
    Calcula qual handicap asiático teria sido ideal baseado no resultado REAL
    Retorna o handicap que maximizaria o valor
    """
    gh, ga = row.get('Goals_H_FT', 0), row.get('Goals_A_FT', 0)
    margin = gh - ga
    
    # Handicaps mais comuns no mercado
    handicaps_possiveis = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, +0.25, +0.5, +0.75, +1.0, +1.25, +1.5]
    
    melhor_handicap = 0
    melhor_performance = -10
    
    for handicap in handicaps_possiveis:
        # Simula resultado com handicap
        resultado_ajustado = margin + handicap
        
        # Score: quanto melhor o handicap, maior a "performance"
        if resultado_ajustado > 0:
            # Ganhou com handicap - penaliza handicaps muito agressivos
            score = 2.0 - abs(handicap) * 0.3
        elif resultado_ajustado == 0:
            # Push - score neutro
            score = 0.5
        else:
            # Perdeu - score negativo
            score = -1.0
        
        if score > melhor_performance:
            melhor_performance = score
            melhor_handicap = handicap
    
    return melhor_handicap

def criar_target_handicap_discreto(row):
    """
    Versão discreta para classificação multi-classe
    """
    handicap_otimo = calcular_handicap_otimo_real(row)
    
    # Agrupa em categorias discretas
    if handicap_otimo <= -1.0:
        return 'STRONG_HOME'
    elif handicap_otimo <= -0.5:
        return 'MODERATE_HOME' 
    elif handicap_otimo < 0:
        return 'LIGHT_HOME'
    elif handicap_otimo == 0:
        return 'NEUTRAL'
    elif handicap_otimo < 0.5:
        return 'LIGHT_AWAY'
    elif handicap_otimo < 1.0:
        return 'MODERATE_AWAY'
    else:
        return 'STRONG_AWAY'

# ============================================================
# 🧮 SISTEMA 3D ORIGINAL (mantido para features)
# ============================================================

def aplicar_clusterizacao_3d(df, n_clusters=4, random_state=42):
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init=10)
    df['Cluster3D_Label'] = kmeans.fit_predict(X_cluster)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
    centroids['Cluster'] = range(n_clusters)
    
    st.markdown("### 🧭 Clusters 3D Criados (KMeans)")
    st.dataframe(centroids.style.format({'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}'}))

    return df

def calcular_momentum_time(df, window=6):
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

# ============================================================
# 🧠 MODELOS DE HANDICAP OPTIMIZATION
# ============================================================

def treinar_modelo_handicap_regressao(history, games_today):
    """
    Modelo de Regressão: Prediz o handicap ótimo contínuo
    """
    st.markdown("### 📈 Modelo Regressão: Handicap Ótimo Contínuo")
    
    # Criar target
    history['Handicap_Otimo'] = history.apply(calcular_handicap_otimo_real, axis=1)
    
    # Features espaciais
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history.columns]
    
    X = history[available_features].fillna(0)
    y = history['Handicap_Otimo']
    
    # Treinar modelo
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        min_samples_leaf=10
    )
    model.fit(X, y)
    
    # Prever para jogos de hoje
    X_today = games_today[available_features].fillna(0)
    games_today['Handicap_Predito_Regressao'] = model.predict(X_today)
    
    # Calcular value gap
    games_today['Value_Gap_Regressao'] = games_today['Handicap_Predito_Regressao'] - games_today['Asian_Line_Decimal']
    
    st.success(f"✅ Regressão treinada: {len(history)} amostras")
    st.info(f"📊 Handicap Ótimo médio histórico: {history['Handicap_Otimo'].mean():.2f}")
    
    return model, games_today

def treinar_modelo_handicap_classificacao(history, games_today):
    """
    Modelo de Classificação: Prediz categoria de handicap
    """
    st.markdown("### 🎯 Modelo Classificação: Categoria de Handicap")
    
    # Criar target categórico
    history['Handicap_Categoria'] = history.apply(criar_target_handicap_discreto, axis=1)
    
    # Features
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history.columns]
    
    X = history[available_features].fillna(0)
    y = history['Handicap_Categoria']
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y_encoded)
    
    # Prever para jogos de hoje
    X_today = games_today[available_features].fillna(0)
    predicoes_encoded = model.predict(X_today)
    probas = model.predict_proba(X_today)
    
    games_today['Handicap_Categoria_Predito'] = le.inverse_transform(predicoes_encoded)
    games_today['Confianca_Categoria'] = np.max(probas, axis=1)
    
    # Mapear categoria para handicap numérico aproximado
    categoria_para_handicap = {
        'STRONG_HOME': -1.25,
        'MODERATE_HOME': -0.75, 
        'LIGHT_HOME': -0.25,
        'NEUTRAL': 0,
        'LIGHT_AWAY': +0.25,
        'MODERATE_AWAY': +0.75,
        'STRONG_AWAY': +1.25
    }
    
    games_today['Handicap_Predito_Classificacao'] = games_today['Handicap_Categoria_Predito'].map(categoria_para_handicap)
    games_today['Value_Gap_Classificacao'] = games_today['Handicap_Predito_Classificacao'] - games_today['Asian_Line_Decimal']
    
    st.success(f"✅ Classificação treinada: {len(history)} amostras")
    st.info(f"📊 Distribuição categorias: {dict(history['Handicap_Categoria'].value_counts())}")
    
    return model, games_today, le

# ============================================================
# 📊 ANÁLISE DE VALOR E RECOMENDAÇÕES
# ============================================================

def analisar_value_bets(games_today):
    """
    Analisa oportunidades de value baseado na diferença entre handicap predito e mercado
    """
    st.markdown("## 💎 Análise de Value Bets")
    
    results = []
    
    for idx, row in games_today.iterrows():
        handicap_mercado = row['Asian_Line_Decimal']
        handicap_regressao = row.get('Handicap_Predito_Regressao', 0)
        handicap_classificacao = row.get('Handicap_Predito_Classificacao', 0)
        
        # Value gap de cada modelo
        gap_regressao = handicap_regressao - handicap_mercado
        gap_classificacao = handicap_classificacao - handicap_mercado
        
        # Value gap consolidado (média ponderada)
        value_gap_consolidado = (gap_regressao * 0.6 + gap_classificacao * 0.4)
        
        # Determinar recomendação
        if value_gap_consolidado > 0.3:
            recomendacao = "STRONG HOME VALUE"
            lado = "HOME"
            confidence = "HIGH"
        elif value_gap_consolidado > 0.15:
            recomendacao = "HOME VALUE" 
            lado = "HOME"
            confidence = "MEDIUM"
        elif value_gap_consolidado < -0.3:
            recomendacao = "STRONG AWAY VALUE"
            lado = "AWAY" 
            confidence = "HIGH"
        elif value_gap_consolidado < -0.15:
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
    
    # Ordenar por valor absoluto do gap (maior value primeiro)
    df_results['Value_Abs'] = abs(df_results['Value_Gap'])
    df_results = df_results.sort_values('Value_Abs', ascending=False)
    
    return df_results

def plot_handicap_analysis(games_today):
    """
    Visualização do espaço de handicaps predito vs mercado
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Regressão vs Mercado
    colors_regressao = []
    for gap in games_today.get('Value_Gap_Regressao', []):
        if gap > 0.2:
            colors_regressao.append('green')
        elif gap < -0.2:
            colors_regressao.append('red')
        else:
            colors_regressao.append('gray')
    
    ax1.scatter(games_today['Asian_Line_Decimal'], 
                games_today.get('Handicap_Predito_Regressao', 0),
                c=colors_regressao, alpha=0.6, s=60)
    ax1.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Linha Ideal')
    ax1.set_xlabel('Handicap Mercado')
    ax1.set_ylabel('Handicap Predito (Regressão)')
    ax1.set_title('Value Analysis - Modelo Regressão')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Classificação vs Mercado
    if 'Handicap_Predito_Classificacao' in games_today.columns:
        colors_class = []
        for gap in games_today.get('Value_Gap_Classificacao', []):
            if gap > 0.2:
                colors_class.append('green')
            elif gap < -0.2:
                colors_class.append('red')
            else:
                colors_class.append('gray')
        
        ax2.scatter(games_today['Asian_Line_Decimal'],
                   games_today['Handicap_Predito_Classificacao'],
                   c=colors_class, alpha=0.6, s=60)
        ax2.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Linha Ideal')
        ax2.set_xlabel('Handicap Mercado')
        ax2.set_ylabel('Handicap Predito (Classificação)')
        ax2.set_title('Value Analysis - Modelo Classificação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    # ---------------- Carregar Dados ----------------
    st.info("📂 Carregando dados para Análise de Handicap Ótimo...")
    
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
        games_today = filter_leagues(games_today)
        
        history = filter_leagues(load_all_games(GAMES_FOLDER))
        history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
        
        return games_today, history
    
    games_today, history = load_cached_data(selected_file)
    
    # ---------------- Live Score Integration ----------------
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
    
    games_today = load_and_merge_livescore(games_today, selected_date_str)
    
    # ---------------- Converter Asian Line ----------------
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    
    history = history.dropna(subset=['Asian_Line_Decimal'])
    
    # ---------------- Aplicar Filtro Temporal ----------------
    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
        except Exception as e:
            st.error(f"Erro ao aplicar filtro temporal: {e}")
    
    # ---------------- Calcular Features 3D ----------------
    st.markdown("## 🧮 Calculando Features 3D...")
    
    history = calcular_momentum_time(history)
    games_today = calcular_momentum_time(games_today)
    
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # ---------------- Treinar Modelos ----------------
    st.markdown("## 🧠 Treinando Modelos de Handicap Ótimo...")
    
    if st.button("🚀 Executar Análise de Handicap Ótimo", type="primary"):
        with st.spinner("Treinando modelos e analisando value..."):
            # Treinar modelo de regressão
            modelo_regressao, games_today = treinar_modelo_handicap_regressao(history, games_today)
            
            # Treinar modelo de classificação  
            modelo_classificacao, games_today, label_encoder = treinar_modelo_handicap_classificacao(history, games_today)
            
            # Analisar value bets
            df_value_bets = analisar_value_bets(games_today)
            
            # Exibir resultados
            st.markdown("## 📊 Resultados da Análise")
            
            # Value bets recomendados
            st.dataframe(df_value_bets, use_container_width=True)
            
            # Visualizações
            st.pyplot(plot_handicap_analysis(games_today))
            
            # Estatísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                strong_bets = len(df_value_bets[df_value_bets['Confidence'] == 'HIGH'])
                st.metric("🎯 Strong Value Bets", strong_bets)
            with col2:
                home_bets = len(df_value_bets[df_value_bets['Lado'] == 'HOME'])
                st.metric("🏠 HOME Value", home_bets)
            with col3:
                away_bets = len(df_value_bets[df_value_bets['Lado'] == 'AWAY'])
                st.metric("✈️ AWAY Value", away_bets)
            
            st.balloons()
    
    else:
        st.info("👆 Clique no botão para executar a análise de handicap ótimo")

if __name__ == "__main__":
    main()
