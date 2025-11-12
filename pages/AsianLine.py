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

# 🔧 CORREÇÃO: Aumentar limite de células para Pandas Styler
pd.set_option("styler.render.max_elements", 500000)

st.set_page_config(page_title="Analisador de Handicap Ótimo - CALIBRADO", layout="wide")
st.title("🎯 Analisador de Handicap Ótimo - Modelo Calibrado")

# ---------------- Configurações ----------------
PAGE_PREFIX = "HandicapOptimizer_Calibrado"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

# ============================================================
# 🔧 FUNÇÕES AUXILIARES CORRIGIDAS
# ============================================================

def setup_livescore_columns(df):
    if df.empty:
        return df
        
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
    if df.empty:
        return df
        
    df = df.copy()
    
    # Verificar e renomear colunas de gols
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    
    # Garantir que colunas essenciais existam
    essential_cols = ['Home', 'Away', 'League', 'Goals_H_FT', 'Goals_A_FT']
    for col in essential_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        st.warning(f"⚠️ Pasta '{folder}' não encontrada!")
        return pd.DataFrame()
        
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        st.warning(f"⚠️ Nenhum arquivo CSV encontrado na pasta '{folder}'!")
        return pd.DataFrame()
        
    dfs = []
    for f in files:
        try:
            df_temp = pd.read_csv(os.path.join(folder, f))
            df_processed = preprocess_df(df_temp)
            dfs.append(df_processed)
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar {f}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
        
    # Garantir que League seja string
    df["League"] = df["League"].astype(str)
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
    Calcula o Momentum do Time (MT_H / MT_A) - VERSÃO CORRIGIDA
    """
    if df.empty:
        return df
        
    df = df.copy()

    # Verificar se colunas essenciais existem
    required_cols = ['Home', 'Away', 'HandScore_Home', 'HandScore_Away']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para momentum: {missing_cols}")
        df['MT_H'] = 0
        df['MT_A'] = 0
        return df

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
    Calcula distâncias e ângulos 3D - VERSÃO CORRIGIDA
    """
    if df.empty:
        return df
        
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT']:
            df[col] = 0
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
    Cria clusters espaciais - VERSÃO CORRIGIDA
    """
    if df.empty:
        return df
        
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

    n_samples = X_cluster.shape[0]
    if n_samples < n_clusters:
        st.warning(f"⚠️ Dados insuficientes para clustering: {n_samples} amostras < {n_clusters} clusters")
        df['Cluster3D_Label'] = 0
        return df

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

        if n_clusters_ajustado > 1:
            centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['dx', 'dy', 'dz'])
            centroids['Cluster'] = range(n_clusters_ajustado)
            
            st.markdown("### 🧭 Clusters 3D Criados (KMeans)")
            st.dataframe(centroids.style.format({'dx': '{:.2f}', 'dy': '{:.2f}', 'dz': '{:.2f}'}))
        else:
            st.info("📊 Apenas 1 cluster criado (dados insuficientes para múltiplos clusters)")

    except Exception as e:
        st.error(f"❌ Erro no clustering: {e}")
        df['Cluster3D_Label'] = 0

    return df

# ============================================================
# 🎯 SISTEMA CALIBRADO: HANDICAP OPTIMIZATION
# ============================================================

def calcular_handicap_otimo_calibrado(row):
    """
    Versão CALIBRADA do cálculo de handicap ótimo
    """
    gh = row.get('Goals_H_FT', 0)
    ga = row.get('Goals_A_FT', 0)
    margin = gh - ga
    
    handicaps_possiveis = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, +0.25, +0.5, +0.75, +1.0, +1.25, +1.5, +1.75, +2.0]
    
    melhor_handicap = 0
    melhor_score = -10
    
    for handicap in handicaps_possiveis:
        resultado_ajustado = margin + handicap
        
        if resultado_ajustado > 0:
            score = 1.5 - abs(handicap) * 0.2
        elif resultado_ajustado == 0:
            score = 0.3
        else:
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
    
    if history.empty:
        st.error("❌ Dados históricos vazios!")
        return None, games_today, None
    
    # Criar target calibrado
    history['Handicap_Otimo_Calibrado'] = history.apply(calcular_handicap_otimo_calibrado, axis=1)
    
    # Filtrar handicaps extremos
    handicap_range = [-2.0, 2.0]
    history_calibrado = history[
        (history['Handicap_Otimo_Calibrado'] >= handicap_range[0]) & 
        (history['Handicap_Otimo_Calibrado'] <= handicap_range[1])
    ].copy()
    
    if history_calibrado.empty:
        st.error("❌ Nenhum dado após filtragem de handicaps!")
        return None, games_today, None
        
    st.info(f"📊 Dados calibrados: {len(history_calibrado)} jogos (handicaps entre {handicap_range[0]} e {handicap_range[1]})")
    
    # Features espaciais
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history_calibrado.columns]
    
    if len(available_features) < 3:
        st.error(f"❌ Features insuficientes para treinamento. Disponíveis: {available_features}")
        return None, games_today, None
    
    X = history_calibrado[available_features].fillna(0)
    y = history_calibrado['Handicap_Otimo_Calibrado']
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Treinar modelo
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=15,
        max_features=0.7,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Validação
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    st.success(f"✅ MAE do modelo: {mae:.3f} (quanto menor, melhor)")
    
    # Prever para jogos de hoje
    if games_today.empty:
        st.warning("⚠️ Nenhum jogo de hoje para prever!")
        return model, games_today, scaler
        
    X_today = games_today[available_features].fillna(0)
    
    # Verificar se temos as mesmas features nos dados de hoje
    missing_features = set(available_features) - set(X_today.columns)
    if missing_features:
        st.warning(f"⚠️ Features faltando nos dados de hoje: {missing_features}")
        for feature in missing_features:
            X_today[feature] = 0
    
    X_today_scaled = scaler.transform(X_today[available_features])
    
    predictions = model.predict(X_today_scaled)
    
    # Suavizar predições
    games_today['Handicap_Predito_Regressao_Calibrado'] = np.clip(predictions, -2.0, 2.0)
    
    # Calcular Asian Line se disponível
    if 'Asian_Line_Decimal' in games_today.columns:
        games_today['Value_Gap_Regressao_Calibrado'] = (
            games_today['Handicap_Predito_Regressao_Calibrado'] - games_today['Asian_Line_Decimal']
        )
    
    return model, games_today, scaler

def treinar_modelo_handicap_classificacao_calibrado(history, games_today):
    """
    Modelo de Classificação CALIBRADO
    """
    st.markdown("### 🎯 Modelo Classificação Calibrado")
    
    if history.empty:
        st.error("❌ Dados históricos vazios!")
        return None, games_today, None
    
    # Criar target categórico calibrado
    history['Handicap_Categoria_Calibrado'] = history.apply(criar_target_handicap_discreto_calibrado, axis=1)
    
    # Features
    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    
    available_features = [f for f in features_3d if f in history.columns]
    
    if len(available_features) < 3:
        st.error(f"❌ Features insuficientes para treinamento. Disponíveis: {available_features}")
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
    if games_today.empty:
        st.warning("⚠️ Nenhum jogo de hoje para prever!")
        return model, games_today, le
        
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
    
    # Mapeamento para handicaps numéricos
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
    
    if 'Asian_Line_Decimal' in games_today.columns:
        games_today['Value_Gap_Classificacao_Calibrado'] = (
            games_today['Handicap_Predito_Classificacao_Calibrado'] - games_today['Asian_Line_Decimal']
        )
    
    st.info(f"📊 Distribuição categorias calibradas: {dict(history['Handicap_Categoria_Calibrado'].value_counts())}")
    
    return model, games_today, le

# ============================================================
# 📊 ANÁLISE DE VALOR CALIBRADA
# ============================================================

def analisar_value_bets_corrigido(games_today):
    """
    Versão CORRIGIDA da análise de value
    """
    st.markdown("## 💎 Análise de Value Bets CORRIGIDA")
    
    if games_today.empty:
        st.warning("⚠️ Nenhum jogo de hoje para analisar!")
        return pd.DataFrame()
    
    results = []
    
    for idx, row in games_today.iterrows():
        handicap_mercado = row.get('Asian_Line_Decimal', 0)
        handicap_regressao = row.get('Handicap_Predito_Regressao_Calibrado', 0)
        handicap_classificacao = row.get('Handicap_Predito_Classificacao_Calibrado', 0)
        
        # Value gap CORRIGIDO
        gap_regressao = handicap_regressao - handicap_mercado
        gap_classificacao = handicap_classificacao - handicap_mercado
        
        # Consolidação dos gaps
        if (handicap_mercado < 0 and handicap_regressao > 0) or (handicap_mercado > 0 and handicap_regressao < 0):
            value_gap_consolidado = gap_regressao if abs(gap_regressao) > abs(gap_classificacao) else gap_classificacao
        else:
            value_gap_consolidado = (gap_regressao * 0.7 + gap_classificacao * 0.3)
        
        # Thresholds realistas
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
        
        # Diagnóstico de sinal
        sinal_alinhado = "✅" if (handicap_mercado * handicap_regressao) >= 0 else "⚠️"
        
        results.append({
            'League': row.get('League', 'N/A'),
            'Home': row.get('Home', 'N/A'),
            'Away': row.get('Away', 'N/A'),
            'Asian_Line': handicap_mercado,
            'Handicap_Regressao': round(handicap_regressao, 2),
            'Handicap_Classificacao': round(handicap_classificacao, 2),
            'Value_Gap': round(value_gap_consolidado, 2),
            'Sinal_Alinhado': sinal_alinhado,
            'Recomendacao': recomendacao,
            'Lado': lado,
            'Confidence': confidence
        })
    
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        df_results['Value_Abs'] = abs(df_results['Value_Gap'])
        df_results = df_results.sort_values('Value_Abs', ascending=False)
    
    return df_results

def plot_handicap_analysis_corrigido(games_today):
    """
    Visualização CORRIGIDA
    """
    if games_today.empty:
        st.warning("⚠️ Nenhum dado para visualizar!")
        return None
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Regressão vs Mercado
    if 'Handicap_Predito_Regressao_Calibrado' in games_today.columns and 'Asian_Line_Decimal' in games_today.columns:
        colors_regressao = []
        sizes_regressao = []
        
        for idx, row in games_today.iterrows():
            handicap_mercado = row['Asian_Line_Decimal']
            handicap_predito = row['Handicap_Predito_Regressao_Calibrado']
            gap = handicap_predito - handicap_mercado
            
            if gap > 0.3:
                colors_regressao.append('green')
                sizes_regressao.append(80)
            elif gap > 0.15:
                colors_regressao.append('lightgreen')
                sizes_regressao.append(60)
            elif gap < -0.3:
                colors_regressao.append('red')
                sizes_regressao.append(80)
            elif gap < -0.15:
                colors_regressao.append('lightcoral')
                sizes_regressao.append(60)
            else:
                colors_regressao.append('gray')
                sizes_regressao.append(40)
        
        ax1.scatter(games_today['Asian_Line_Decimal'], 
                    games_today['Handicap_Predito_Regressao_Calibrado'],
                    c=colors_regressao, alpha=0.7, s=sizes_regressao)
        ax1.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Mercado Perfeito')
        ax1.set_xlabel('Handicap Mercado')
        ax1.set_ylabel('Handicap Predito (Regressão)')
        ax1.set_title('Value Analysis - Modelo Regressão (Corrigido)')
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Classificação vs Mercado
    if 'Handicap_Predito_Classificacao_Calibrado' in games_today.columns and 'Asian_Line_Decimal' in games_today.columns:
        colors_class = []
        for idx, row in games_today.iterrows():
            gap = row.get('Value_Gap_Classificacao_Calibrado', 0)
            if gap > 0.3:
                colors_class.append('green')
            elif gap > 0.15:
                colors_class.append('lightgreen')
            elif gap < -0.3:
                colors_class.append('red')
            elif gap < -0.15:
                colors_class.append('lightcoral')
            else:
                colors_class.append('gray')
        
        ax2.scatter(games_today['Asian_Line_Decimal'],
                   games_today['Handicap_Predito_Classificacao_Calibrado'],
                   c=colors_class, alpha=0.7, s=60)
        ax2.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='Mercado Perfeito')
        ax2.set_xlabel('Handicap Mercado')
        ax2.set_ylabel('Handicap Predito (Classificação)')
        ax2.set_title('Value Analysis - Modelo Classificação')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(-2.5, 2.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diagnóstico de Sinais
    sinais_alinhados = 0
    sinais_opostos = 0
    
    for idx, row in games_today.iterrows():
        handicap_mercado = row.get('Asian_Line_Decimal', 0)
        handicap_regressao = row.get('Handicap_Predito_Regressao_Calibrado', 0)
        
        if (handicap_mercado * handicap_regressao) >= 0:
            sinais_alinhados += 1
        else:
            sinais_opostos += 1
    
    labels = ['Sinais Alinhados', 'Sinais Opostos']
    sizes = [sinais_alinhados, sinais_opostos]
    colors = ['lightblue', 'lightcoral']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Diagnóstico: Alinhamento de Sinais\n(Mercado vs Modelo)')
    
    # Plot 4: Distribuição de Value Gaps
    if 'Value_Gap_Regressao_Calibrado' in games_today.columns:
        gaps = games_today['Value_Gap_Regressao_Calibrado']
        ax4.hist(gaps, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Linha Zero')
        ax4.set_xlabel('Value Gap')
        ax4.set_ylabel('Frequência')
        ax4.set_title('Distribuição dos Value Gaps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL - VERSÃO CORRIGIDA
# ============================================================

def main():
    st.sidebar.header("⚙️ Configurações")
    
    # Modo de demonstração
    demo_mode = st.sidebar.checkbox("🧪 Modo Demonstração (Dados de Exemplo)", value=True)
    
    if demo_mode:
        st.info("🔧 Executando em modo de demonstração com dados de exemplo...")
        
        # Criar dados históricos de exemplo
        np.random.seed(42)
        n_historical = 200
        
        history_df = pd.DataFrame({
            'League': ['Premier League'] * n_historical,
            'Home': [f'Team_{i}' for i in range(n_historical)],
            'Away': [f'Team_{i+100}' for i in range(n_historical)],
            'Goals_H_FT': np.random.randint(0, 5, n_historical),
            'Goals_A_FT': np.random.randint(0, 5, n_historical),
            'Asian_Line': np.random.choice([-0.5, -0.25, 0, 0.25, 0.5, -1.0, 1.0], n_historical),
            'Aggression_Home': np.random.normal(0, 1, n_historical),
            'Aggression_Away': np.random.normal(0, 1, n_historical),
            'M_H': np.random.normal(0, 1, n_historical),
            'M_A': np.random.normal(0, 1, n_historical),
            'HandScore_Home': np.random.normal(0, 1, n_historical),
            'HandScore_Away': np.random.normal(0, 1, n_historical),
        })
        
        # Criar jogos de hoje de exemplo
        games_today = pd.DataFrame({
            'League': ['Premier League', 'La Liga', 'Serie A', 'Bundesliga'],
            'Home': ['Barcelona', 'Real Madrid', 'Juventus', 'Bayern Munich'],
            'Away': ['Atletico Madrid', 'Sevilla', 'AC Milan', 'Borussia Dortmund'],
            'Asian_Line': [-0.5, 0.25, -0.25, 0.75],
            'Aggression_Home': [0.8, -0.3, 0.5, 1.2],
            'Aggression_Away': [-0.2, 0.6, -0.1, 0.3],
            'M_H': [0.7, -0.4, 0.3, 0.9],
            'M_A': [-0.3, 0.5, -0.2, 0.4],
            'HandScore_Home': [0.6, -0.5, 0.4, 1.0],
            'HandScore_Away': [-0.4, 0.3, -0.3, 0.2],
        })
        
        st.success(f"✅ Dados de exemplo criados: {len(history_df)} históricos, {len(games_today)} jogos de hoje")
        
    else:
        # Carregar dados reais
        with st.spinner("📂 Carregando dados históricos..."):
            history_df = load_all_games(GAMES_FOLDER)
            history_df = filter_leagues(history_df)
        
        if history_df.empty:
            st.error("❌ Nenhum dado histórico encontrado! Ative o modo demonstração.")
            return
        
        st.success(f"✅ Dados históricos carregados: {len(history_df)} jogos")
        
        # Carregar jogos de hoje
        with st.spinner("📂 Carregando jogos de hoje..."):
            games_today = load_all_games(LIVESCORE_FOLDER)
            games_today = filter_leagues(games_today)
            games_today = setup_livescore_columns(games_today)
        
        if games_today.empty:
            st.warning("⚠️ Nenhum jogo de hoje encontrado! Usando apenas dados históricos para demonstração.")
            # Criar games_today vazio mas com estrutura
            games_today = pd.DataFrame(columns=history_df.columns)
    
    # Processar Asian Lines
    if 'Asian_Line' in history_df.columns:
        history_df['Asian_Line_Decimal'] = history_df['Asian_Line'].apply(convert_asian_line_to_decimal)
    if 'Asian_Line' in games_today.columns:
        games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    
    # Calcular features
    with st.spinner("🧮 Calculando métricas..."):
        history_df = calcular_momentum_time(history_df)
        history_df = calcular_distancias_3d(history_df)
        history_df = aplicar_clusterizacao_3d(history_df)
        
        if not games_today.empty:
            games_today = calcular_momentum_time(games_today)
            games_today = calcular_distancias_3d(games_today)
            games_today = aplicar_clusterizacao_3d(games_today)
    
    # Treinar modelos
    with st.spinner("🤖 Treinando modelo de regressão..."):
        model_reg, games_today, scaler_reg = treinar_modelo_handicap_regressao_calibrado(history_df, games_today)
    
    with st.spinner("🤖 Treinando modelo de classificação..."):
        model_clf, games_today, le_clf = treinar_modelo_handicap_classificacao_calibrado(history_df, games_today)
    
    # Análise de value bets
    if model_reg is not None and model_clf is not None and not games_today.empty:
        with st.spinner("💎 Analisando value bets..."):
            df_value_bets = analisar_value_bets_corrigido(games_today)
        
        # Mostrar resultados
        st.markdown("## 📊 Resultados dos Value Bets")
        
        # Value bets fortes
        strong_values = df_value_bets[df_value_bets['Confidence'].isin(['HIGH', 'MEDIUM'])]
        if not strong_values.empty:
            st.markdown("### 🎯 Value Bets Identificados")
            # 🔧 CORREÇÃO: Usar st.dataframe em vez de styler para evitar o erro
            st.dataframe(
                strong_values,
                column_config={
                    'Asian_Line': st.column_config.NumberColumn(format="%.2f"),
                    'Handicap_Regressao': st.column_config.NumberColumn(format="%.2f"),
                    'Handicap_Classificacao': st.column_config.NumberColumn(format="%.2f"),
                    'Value_Gap': st.column_config.NumberColumn(format="%.2f")
                },
                width='stretch'  # 🔧 CORREÇÃO: use_container_width substituído
            )
        else:
            st.info("ℹ️ Nenhum value bet forte identificado hoje")
        
        # Todos os jogos
        st.markdown("### 📋 Todos os Jogos Analisados")
        # 🔧 CORREÇÃO: Usar st.dataframe simples para evitar o erro
        st.dataframe(
            df_value_bets,
            column_config={
                'Asian_Line': st.column_config.NumberColumn(format="%.2f"),
                'Handicap_Regressao': st.column_config.NumberColumn(format="%.2f"),
                'Handicap_Classificacao': st.column_config.NumberColumn(format="%.2f"),
                'Value_Gap': st.column_config.NumberColumn(format="%.2f")
            },
            width='stretch'  # 🔧 CORREÇÃO: use_container_width substituído
        )
        
        # Gráficos
        st.markdown("### 📈 Análise Visual")
        fig = plot_handicap_analysis_corrigido(games_today)
        if fig is not None:
            st.pyplot(fig)
        
        # Estatísticas
        if not df_value_bets.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                total_games = len(df_value_bets)
                st.metric("Total Jogos", total_games)
            
            with col2:
                value_bets_count = len(strong_values)
                st.metric("Value Bets", value_bets_count)
            
            with col3:
                if total_games > 0:
                    value_rate = (value_bets_count / total_games) * 100
                    st.metric("Taxa de Value", f"{value_rate:.1f}%")
    
    else:
        st.error("❌ Falha no treinamento dos modelos ou nenhum jogo para analisar!")

# ============================================================
# 🎬 INICIAR APLICAÇÃO
# ============================================================

if __name__ == "__main__":
    main()
