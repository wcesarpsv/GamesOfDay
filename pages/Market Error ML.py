# market_error_ml_3d_dual.py
########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(
    page_title="Value Bet Detector - 3D Dual ML", 
    layout="wide",
    page_icon="🎯"
)

st.title("🎯 3D Dual ML – Value Bet Intelligence")
st.markdown("### Meta-Modelo 3D com Detecção Dual (Home & Away)")

# Configurações principais
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "coppa", "afc","trophy"]

########################################
####### Bloco 3 – Helper Functions #####
########################################
def load_all_games(folder):
    """Carrega todos os arquivos CSV do histórico"""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def filter_leagues(df):
    """Filtra ligas indesejadas"""
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def prepare_history(df):
    """Prepara dados históricos para treinamento"""
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

def compute_double_chance_odds(df):
    """Calcula odds de dupla chance"""
    if all(col in df.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
        probs = pd.DataFrame()
        probs['p_H'] = 1 / df['Odd_H']
        probs['p_D'] = 1 / df['Odd_D']
        probs['p_A'] = 1 / df['Odd_A']
        probs = probs.div(probs.sum(axis=1), axis=0)
        df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
        df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

########################################
### Bloco 3.1 – Features 3D Avançadas ##
########################################
def calcular_distancias_3d(df):
    """Features 3D avançadas - adaptação do conceito original"""
    df = df.copy()
    
    # Usar Aggression, M e MT como dimensões (fallback se não existirem)
    aggression_h = df.get('Aggression_Home', df.get('M_H', 0))
    aggression_a = df.get('Aggression_Away', df.get('M_A', 0))
    m_h = df.get('M_H', 0)
    m_a = df.get('M_A', 0) 
    mt_h = df.get('MT_H', df.get('M_H', 0))
    mt_a = df.get('MT_A', df.get('M_A', 0))
    
    # Vetor de diferenças 3D
    dx = aggression_h - aggression_a
    dy = m_h - m_a
    dz = mt_h - mt_a
    
    # Features de magnitude e direção
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    
    # Features trigonométricas
    a_xy = np.arctan2(dy, dx.replace(0, 1e-9))
    a_xz = np.arctan2(dz, dx.replace(0, 1e-9)) 
    a_yz = np.arctan2(dz, dy.replace(0, 1e-9))
    
    df['Quadrant_Sin_XY'] = np.sin(a_xy)
    df['Quadrant_Cos_XY'] = np.cos(a_xy)
    df['Quadrant_Sin_XZ'] = np.sin(a_xz)
    df['Quadrant_Cos_XZ'] = np.cos(a_xz)
    df['Quadrant_Sin_YZ'] = np.sin(a_yz)
    df['Quadrant_Cos_YZ'] = np.cos(a_yz)
    
    # Combinações
    combo = a_xy + a_xz + a_yz
    df['Quadrant_Sin_Combo'] = np.sin(combo)
    df['Quadrant_Cos_Combo'] = np.cos(combo)
    
    # Sinal do vetor (concordância de direção)
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return df

def aplicar_clusterizacao_3d(df, n_clusters=5):
    """Agrupa jogos por padrões 3D similares"""
    df = df.copy()
    
    # Criar diferenças para clustering
    aggression_h = df.get('Aggression_Home', df.get('M_H', 0))
    aggression_a = df.get('Aggression_Away', df.get('M_A', 0))
    
    df['dx'] = aggression_h - aggression_a
    df['dy'] = df.get('M_H', 0) - df.get('M_A', 0)
    df['dz'] = df.get('MT_H', df.get('M_H', 0)) - df.get('MT_A', df.get('M_A', 0))
    
    Xc = df[['dx', 'dy', 'dz']].fillna(0)
    
    # Aplicar KMeans apenas se temos dados suficientes
    if len(Xc) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster3D_Label'] = kmeans.fit_predict(Xc)
    else:
        df['Cluster3D_Label'] = 0
    
    return df

########################################
### Bloco 3.2 – Classificações Liga ####
########################################
def classify_leagues_variation(history_df):
    """Classifica ligas por variação de momentum"""
    agg = (
        history_df.groupby('League')
        .agg(
            M_H_Min=('M_H','min'), M_H_Max=('M_H','max'),
            M_A_Min=('M_A','min'), M_A_Max=('M_A','max'),
            Hist_Games=('M_H','count')
        ).reset_index()
    )
    agg['Variation_Total'] = (agg['M_H_Max'] - agg['M_H_Min']) + (agg['M_A_Max'] - agg['M_A_Min'])
    def label(v):
        if v > 6.0: return "High Variation"
        if v >= 3.0: return "Medium Variation"
        return "Low Variation"
    agg['League_Classification'] = agg['Variation_Total'].apply(label)
    return agg[['League','League_Classification','Variation_Total','Hist_Games']]

def compute_league_bands(history_df):
    """Calcula bandas por liga para momentum"""
    hist = history_df.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']
    diff_q = (
        hist.groupby('League')['M_Diff']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'P20_Diff', 0.8:'P80_Diff'})
            .reset_index()
    )
    home_q = (
        hist.groupby('League')['M_H']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Home_P20', 0.8:'Home_P80'})
            .reset_index()
    )
    away_q = (
        hist.groupby('League')['M_A']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Away_P20', 0.8:'Away_P80'})
            .reset_index()
    )
    out = diff_q.merge(home_q, on='League', how='inner').merge(away_q, on='League', how='inner')
    return out

def dominant_side(row, threshold=0.90):
    """Classifica lado dominante"""
    m_h, m_a = row['M_H'], row['M_A']
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold:
        return "Home strong"
    if m_h <= -threshold:
        return "Home weak"
    if m_a >= threshold:
        return "Away strong"
    if m_a <= -threshold:
        return "Away weak"
    return "Mixed / Neutral"

########################################
### Bloco 3.3 – Data Loading Helpers ###
########################################
def get_available_dates():
    """Obtém todas as datas disponíveis nos arquivos"""
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
    dates = []
    for file in files:
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", file)
        if date_match:
            dates.append(date_match.group(0))
    
    dates = sorted(dates)
    if len(dates) > 7:
        dates = dates[-7:]
    
    return dates

def load_specific_date(target_date):
    """Carrega dados de uma data específica"""
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv") and target_date in f]
    
    if not files:
        st.error(f"No data found for date: {target_date}")
        return None
    
    file_path = os.path.join(GAMES_FOLDER, files[0])
    games = pd.read_csv(file_path)
    games = filter_leagues(games)
    
    return games

def prepare_games_data(games, history):
    """Prepara os dados dos jogos com classificações e bandas"""
    games = compute_double_chance_odds(games)
    games['M_Diff'] = games['M_H'] - games['M_A']
    
    # Adicionar classificações de liga
    league_class = classify_leagues_variation(history)
    league_bands = compute_league_bands(history)
    
    games = games.merge(league_class, on='League', how='left')
    games = games.merge(league_bands, on='League', how='left')
    
    # Calcular bandas
    if all(col in games.columns for col in ['M_H', 'Home_P20', 'Home_P80']):
        games['Home_Band'] = np.where(
            games['M_H'] <= games['Home_P20'], 'Bottom 20%',
            np.where(games['M_H'] >= games['Home_P80'], 'Top 20%', 'Balanced')
        )
    
    if all(col in games.columns for col in ['M_A', 'Away_P20', 'Away_P80']):
        games['Away_Band'] = np.where(
            games['M_A'] <= games['Away_P20'], 'Bottom 20%',
            np.where(games['M_A'] >= games['Away_P80'], 'Top 20%', 'Balanced')
        )
    
    games['Dominant'] = games.apply(dominant_side, axis=1)
    
    return games

def merge_livescore_data(games_df, target_date):
    """Merge com dados do LiveScore para obter placares em tempo real"""
    livescore_folder = "LiveScore"
    livescore_file = os.path.join(livescore_folder, f"Resultados_RAW_{target_date}.csv")
    
    if 'Goals_H_Today' not in games_df.columns:
        games_df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in games_df.columns:
        games_df['Goals_A_Today'] = np.nan
    
    if os.path.exists(livescore_file):
        st.info(f"LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)
        
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
        
        required_cols = ['Id', 'status', 'home_goal', 'away_goal']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            st.warning(f"LiveScore file missing columns: {missing_cols}")
        else:
            games_df = games_df.merge(
                results_df,
                left_on='Id',
                right_on='Id',
                how='left',
                suffixes=('', '_RAW')
            )
            
            games_df['Goals_H_Today'] = games_df['home_goal']
            games_df['Goals_A_Today'] = games_df['away_goal']
            games_df.loc[games_df['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
            
            st.success(f"✅ LiveScore data merged successfully! Found {len(results_df)} games.")
    else:
        st.warning(f"No LiveScore results file found for date: {target_date}")
    
    return games_df

def determine_result(row):
    """Determina o resultado final baseado nos gols"""
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
    except (ValueError, TypeError):
        return None

    if pd.isna(gh) or pd.isna(ga):
        return None
    if gh > ga:
        return "Home"
    elif gh < ga:
        return "Away"
    else:
        return "Draw"

########################################
##### Bloco 4 – Sidebar Configs ########
########################################
def setup_sidebar():
    """Configura a barra lateral"""
    st.sidebar.header("🔧 3D Dual ML Configuration")
    
    # Seletor de Data
    st.sidebar.subheader("📅 Date Selection")
    available_dates = get_available_dates()
    
    if available_dates:
        selected_date = st.sidebar.selectbox(
            "Select Date to Analyze:",
            options=available_dates,
            index=len(available_dates)-1
        )
        st.sidebar.info(f"Analyzing: {selected_date}")
        return selected_date
    else:
        st.sidebar.error("No date files found!")
        st.stop()

def setup_parameters():
    """Configura os parâmetros do modelo"""
    st.sidebar.subheader("🎯 Value Detection Parameters")
    min_value_gap = st.sidebar.slider(
        "Minimum Value Gap", 
        min_value=0.01, max_value=0.20, value=0.05, step=0.01
    )
    
    value_confidence_threshold = st.sidebar.slider(
        "Value Confidence Threshold", 
        min_value=0.50, max_value=0.90, value=0.55, step=0.01
    )
    
    min_odds = st.sidebar.number_input(
        "Minimum Odds", 
        min_value=1.5, max_value=5.0, value=1.8, step=0.1
    )
    
    return min_value_gap, value_confidence_threshold, min_odds

def setup_league_filters():
    """Configura os filtros de liga"""
    st.sidebar.subheader("🏆 League Filters")
    show_high_var = st.sidebar.checkbox("High Variation Leagues", value=True)
    show_medium_var = st.sidebar.checkbox("Medium Variation Leagues", value=True)
    show_low_var = st.sidebar.checkbox("Low Variation Leagues", value=False)
    
    league_filters = []
    if show_high_var:
        league_filters.append("High Variation")
    if show_medium_var:
        league_filters.append("Medium Variation") 
    if show_low_var:
        league_filters.append("Low Variation")
    
    return league_filters

########################################
####### Bloco 5 – Data Loading #########
########################################
@st.cache_data
def load_training_data():
    """Carrega dados de treinamento"""
    all_games = load_all_games(GAMES_FOLDER)
    all_games = filter_leagues(all_games)
    history = prepare_history(all_games)
    
    if history.empty:
        st.error("No valid historical data found.")
        return None
    
    return history

@st.cache_data
def load_analysis_data(target_date, _history):
    """Carrega dados específicos para análise incluindo LiveScore"""
    games_target = load_specific_date(target_date)
    if games_target is None:
        return None
    
    games_processed = prepare_games_data(games_target, _history)
    games_processed = merge_livescore_data(games_processed, target_date)
    
    if all(col in games_processed.columns for col in ['Goals_H_Today', 'Goals_A_Today']):
        games_processed['Result_Today'] = games_processed.apply(determine_result, axis=1)
        
        finished_games = games_processed.dropna(subset=['Result_Today'])
        if len(finished_games) > 0:
            st.success(f"📊 {len(finished_games)} games have final results available!")
    
    return games_processed

def ensure_features_exist(games_today):
    """Garante que todas as features necessárias existam - VERSÃO 3D"""
    
    st.header("🔧 3D Feature Engineering - Debug")
    
    # 1. Primeiro: Verificar e limpar dados críticos
    st.subheader("1. Limpeza de Dados Críticos")
    
    # Remover linhas onde M_H ou M_A são NaN (CRÍTICO!)
    initial_count = len(games_today)
    games_today = games_today.dropna(subset=['M_H', 'M_A']).copy()
    final_count = len(games_today)
    removed_count = initial_count - final_count
    
    st.write(f"✅ Removidas {removed_count} linhas com M_H ou M_A faltantes")
    st.write(f"✅ Restaram {final_count} jogos válidos")
    
    if final_count == 0:
        st.error("❌ Nenhum jogo válido após limpeza! Verifique os dados de M_H e M_A.")
        return None
    
    # 2. Aplicar features 3D
    st.subheader("2. Aplicando Features 3D")
    games_today = calcular_distancias_3d(games_today)
    games_today = aplicar_clusterizacao_3d(games_today)
    st.success("✅ Features 3D aplicadas")
    
    # 3. Criar features derivadas ESSENCIAIS
    st.subheader("3. Criação de Features Derivadas")
    
    # M_Diff (CRÍTICA)
    games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
    st.success("✅ M_Diff criada")
    
    # Odds de Dupla Chance (CRÍTICA)
    if all(col in games_today.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
        probs_dc = pd.DataFrame()
        probs_dc['p_H'] = 1 / games_today['Odd_H']
        probs_dc['p_D'] = 1 / games_today['Odd_D']
        probs_dc['p_A'] = 1 / games_today['Odd_A']
        probs_dc = probs_dc.div(probs_dc.sum(axis=1), axis=0)
        games_today['Odd_1X'] = 1 / (probs_dc['p_H'] + probs_dc['p_D'])
        games_today['Odd_X2'] = 1 / (probs_dc['p_A'] + probs_dc['p_D'])
        st.success("✅ Odd_1X e Odd_X2 criadas")
    
    # 4. Criar features com fallback simples
    st.subheader("4. Features com Fallback")
    
    # Dominant side
    try:
        games_today['Dominant'] = games_today.apply(dominant_side, axis=1)
        st.success("✅ Dominant criada")
    except Exception as e:
        st.error(f"❌ Erro em dominant_side: {e}")
        games_today['Dominant'] = "Mixed / Neutral"
        st.success("✅ Dominant criada com fallback")
    
    # League classification (fallback)
    games_today['League_Classification'] = 'Medium Variation'
    st.success("✅ League_Classification criada")
    
    # 5. Verificação final
    st.subheader("5. Verificação Final")
    expected_features = ['M_Diff', 'Dominant', 'League_Classification', 
                        'Odd_1X', 'Odd_X2', 'Cluster3D_Label', 'Quadrant_Dist_3D', 'Vector_Sign']
    missing_final = [f for f in expected_features if f not in games_today.columns]
    
    if missing_final:
        st.error(f"❌ Features ainda faltando: {missing_final}")
        return None
    else:
        st.success(f"🎉 Todas as {len(expected_features)} features criadas com sucesso!")
        st.write(f"📊 Shape final: {games_today.shape}")
        
        # Mostrar estatísticas das novas features
        st.write("**Estatísticas das Features 3D:**")
        st.write(f"- Cluster3D_Label: {games_today['Cluster3D_Label'].value_counts().to_dict()}")
        st.write(f"- Quadrant_Dist_3D: min={games_today['Quadrant_Dist_3D'].min():.2f}, max={games_today['Quadrant_Dist_3D'].max():.2f}")
        st.write(f"- Vector_Sign: {games_today['Vector_Sign'].value_counts().to_dict()}")
        
        return games_today

########################################
### Bloco 6 – ML Model Training #######
########################################
@st.cache_resource
def train_main_model(_history, target_date):
    """Treina o modelo principal de classificação SEM data leak"""
    
    if 'Date' in _history.columns:
        try:
            _history['Date'] = pd.to_datetime(_history['Date'])
            target_date_dt = pd.to_datetime(target_date)
            training_data = _history[_history['Date'] < target_date_dt].copy()
        except:
            training_data = _history.copy()
            st.warning("⚠️ Date conversion failed - using all historical data")
    else:
        training_data = _history.copy()
        st.warning("⚠️ No 'Date' column found - using all historical data")
    
    if training_data.empty:
        st.error("No training data available after date filtering!")
        return None, None, None
    
    st.info(f"📊 Training model with {len(training_data)} games before {target_date}")

    # Preparar target
    def map_result(row):
        if row['Goals_H_FT'] > row['Goals_A_FT']:
            return "Home"
        elif row['Goals_H_FT'] < row['Goals_A_FT']:
            return "Away"
        else:
            return "Draw"

    training_data['Result'] = training_data.apply(map_result, axis=1)

    # Aplicar features 3D ao histórico
    training_data = calcular_distancias_3d(training_data)
    training_data = aplicar_clusterizacao_3d(training_data)

    # Features incluindo as 3D (REMOVENDO AS ANTIGAS)
    features_raw = [
        'M_H', 'M_A', 'Diff_Power', 'M_Diff',
        'Dominant', 'League_Classification',
        'Odd_H', 'Odd_D', 'Odd_A', 'Odd_1X', 'Odd_X2',
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign', 'Magnitude_3D',
        'Cluster3D_Label'
    ]
    # Manter apenas colunas que existem
    features_raw = [f for f in features_raw if f in training_data.columns]
    
    st.info(f"🔧 Usando {len(features_raw)} features (incluindo 3D): {features_raw}")

    X = training_data[features_raw].copy()
    y = training_data['Result']

    # Codificar variáveis categóricas
    cat_cols = [c for c in ['Dominant','League_Classification','Cluster3D_Label'] if c in X]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    if cat_cols:
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                       encoded_df.reset_index(drop=True)], axis=1)

    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    return model, encoder, features_raw



########################################
### Bloco 6.1 – Dual Value Models #####
########################################
@st.cache_resource
def train_dual_value_models(history, target_date):
    """Treina modelos separados para value em Home e Away"""
    
    # Filtrar dados históricos (SEM data leak)
    if 'Date' in history.columns:
        history['Date'] = pd.to_datetime(history['Date'])
        target_date_dt = pd.to_datetime(target_date)
        training_data = history[history['Date'] < target_date_dt].copy()
    else:
        training_data = history.copy()
        st.warning("⚠️ No 'Date' column - using all historical data")
    
    if training_data.empty:
        st.error("No training data available for dual models!")
        return None, None, None, None
    
    st.info(f"🎯 Training dual models with {len(training_data)} games")

    # Criar targets binários para Home e Away
    training_data['Target_Value_Home'] = (training_data['Goals_H_FT'] > training_data['Goals_A_FT']).astype(int)
    training_data['Target_Value_Away'] = (training_data['Goals_H_FT'] < training_data['Goals_A_FT']).astype(int)
    
    # Aplicar features 3D
    training_data = calcular_distancias_3d(training_data)
    training_data = aplicar_clusterizacao_3d(training_data)
    
    # Features base + 3D (GARANTINDO QUE M_Diff EXISTE)
    if 'M_Diff' not in training_data.columns:
        training_data['M_Diff'] = training_data['M_H'] - training_data['M_A']
    
    base_features = ['M_H', 'M_A', 'Diff_Power', 'M_Diff', 'Odd_H', 'Odd_A']
    
    # Adicionar features 3D
    feat3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ', 'Vector_Sign', 'Magnitude_3D'
    ]
    
    # One-hot encoding para clusters e ligas
    cluster_dummies = pd.get_dummies(training_data['Cluster3D_Label'], prefix='C3D')
    
    # PARA LIGAS: usar apenas ligas que existem no histórico de treino
    league_dummies = pd.get_dummies(training_data['League'], prefix='League') if 'League' in training_data.columns else pd.DataFrame()
    league_columns = list(league_dummies.columns)
    
    # Combinar todas as features
    available_features = base_features + [f for f in feat3d if f in training_data.columns]
    X = pd.concat([
        training_data[available_features].fillna(0),
        cluster_dummies,
        league_dummies
    ], axis=1)
    
    # GARANTIR que as features têm nomes consistentes
    X = X.astype(float)  # Evitar problemas de tipo

    st.info(f"🔍 Estrutura final do treino: {X.shape[1]} features")
    st.info(f"🔍 Clusters: {cluster_dummies.shape[1]}, Ligas: {league_dummies.shape[1]}")
    
    # Treinar modelo para Home
    model_home = RandomForestClassifier(
        n_estimators=200, max_depth=12, 
        min_samples_split=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_home.fit(X, training_data['Target_Value_Home'])
    
    # Treinar modelo para Away  
    model_away = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_split=15, min_samples_leaf=5, 
        class_weight='balanced', random_state=24, n_jobs=-1
    )
    model_away.fit(X, training_data['Target_Value_Away'])
    
    st.success(f"✅ Dual models trained with {X.shape[1]} features")
    return model_home, model_away, available_features, league_columns

def calculate_dual_ev(games_today, model_home, model_away, feature_columns, league_columns):
    """Calcula EV separado para Home e Away usando modelos duais"""
    
    if model_home is None or model_away is None:
        st.warning("Dual models not available - skipping dual EV calculation")
        return games_today
    
    # Aplicar features 3D nos dados atuais
    games_today = calcular_distancias_3d(games_today)
    games_today = aplicar_clusterizacao_3d(games_today)
    
    # Garantir que M_Diff existe
    if 'M_Diff' not in games_today.columns:
        games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
    
    # DEBUG: Mostrar features disponíveis
    st.info(f"🔍 Features disponíveis: {len(games_today.columns)}")
    
    # Preparar features para predição - MÉTODO MAIS SEGURO
    # Usar APENAS as features que sabemos que existem
    available_features = []
    for f in feature_columns:
        if f in games_today.columns:
            available_features.append(f)
    
    # One-hot encoding - MÉTODO MAIS CONTROLADO
    cluster_dummies = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D')
    
    # PARA LIGAS: criar DataFrame vazio e preencher apenas as colunas esperadas
    league_dummies = pd.DataFrame(0, index=games_today.index, columns=league_columns)
    if 'League' in games_today.columns:
        # Mapear ligas atuais para as colunas esperadas
        league_mapping = {}
        for col in league_columns:
            # Extrair o nome da liga do prefixo 'League_'
            league_name = col.replace('League_', '')
            league_mapping[league_name] = col
        
        # Preencher as dummies
        for idx, league in games_today['League'].items():
            if league in league_mapping:
                col_name = league_mapping[league]
                league_dummies.loc[idx, col_name] = 1
    
    # Construir X_pred de forma mais controlada
    X_pred = pd.DataFrame(index=games_today.index)
    
    # Adicionar features numéricas
    for feature in available_features:
        if feature in games_today.columns:
            X_pred[feature] = games_today[feature].fillna(0)
    
    # Adicionar clusters
    for col in cluster_dummies.columns:
        X_pred[col] = cluster_dummies[col]
    
    # Adicionar ligas (apenas as esperadas)
    for col in league_columns:
        X_pred[col] = league_dummies[col]
    
    # GARANTIR ESTRUTURA IDÊNTICA AO TREINO
    try:
        # Obter features esperadas pelo modelo
        expected_features = model_home.feature_names_in_
        
        st.info(f"🔍 Modelo espera: {len(expected_features)} features")
        st.info(f"🔍 Nós temos: {len(X_pred.columns)} features")
        
        # VERIFICAR DIFERENÇAS
        missing_cols = set(expected_features) - set(X_pred.columns)
        extra_cols = set(X_pred.columns) - set(expected_features)
        
        if missing_cols:
            st.warning(f"❌ Faltando {len(missing_cols)} colunas: {list(missing_cols)[:5]}...")
            for col in missing_cols:
                X_pred[col] = 0
        
        if extra_cols:
            st.warning(f"❌ {len(extra_cols)} colunas extras serão removidas: {list(extra_cols)[:5]}...")
            X_pred = X_pred.drop(columns=list(extra_cols))
        
        # ORDENAR EXATAMENTE como no treino
        X_pred = X_pred[expected_features]
        
        # VERIFICAÇÃO FINAL
        if list(X_pred.columns) != list(expected_features):
            st.error("❌ CRÍTICO: Ordem ainda não coincide!")
            st.error(f"Esperado: {expected_features[:3]}...")
            st.error(f"Obtido: {list(X_pred.columns)[:3]}...")
            return games_today
        
        st.success("✅ Estrutura de features VERIFICADA - fazendo predições...")
        
        # Predições de probabilidade
        proba_home = model_home.predict_proba(X_pred)[:, 1]
        proba_away = model_away.predict_proba(X_pred)[:, 1]
        
        # Calcular EV para ambos os lados
        games_today['EV_Home_Dual'] = (proba_home * games_today['Odd_H']) - 1
        games_today['EV_Away_Dual'] = (proba_away * games_today['Odd_A']) - 1
        
        # Probabilidades dos modelos duais
        games_today['Dual_Proba_Home'] = proba_home
        games_today['Dual_Proba_Away'] = proba_away
        
        st.success(f"✅ Dual EV calculado para {len(games_today)} jogos!")
        
        # Estatísticas
        positive_ev_home = (games_today['EV_Home_Dual'] > 0).sum()
        positive_ev_away = (games_today['EV_Away_Dual'] > 0).sum()
        st.info(f"📊 EV positivo: Home={positive_ev_home}, Away={positive_ev_away}")
        
    except Exception as e:
        st.error(f"❌ Erro crítico no dual EV: {e}")
        # Fallback seguro
        games_today['EV_Home_Dual'] = 0
        games_today['EV_Away_Dual'] = 0
        games_today['Dual_Proba_Home'] = 0.5
        games_today['Dual_Proba_Away'] = 0.5
    
    return games_today


########################################
### Bloco 7 – Market Error Analysis ####
########################################
def calculate_market_probabilities(games_today):
    """Calcula probabilidades implícitas do mercado"""
    probs = pd.DataFrame()
    probs['p_H'] = 1 / games_today['Odd_H']
    probs['p_D'] = 1 / games_today['Odd_D']
    probs['p_A'] = 1 / games_today['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)

    games_today['Imp_Prob_H'] = probs['p_H']
    games_today['Imp_Prob_D'] = probs['p_D']
    games_today['Imp_Prob_A'] = probs['p_A']
    
    return games_today

def prepare_prediction_data(games_today, features_raw, encoder):
    """Prepara dados para predição do modelo"""
    # Selecionar apenas features disponíveis
    available_features = [f for f in features_raw if f in games_today.columns]
    
    if len(available_features) == 0:
        st.error("No features available for prediction!")
        return None
    
    X_pred = games_today[available_features].copy()
    
    # One-hot encoding
    cat_cols = [c for c in ['Dominant', 'League_Classification', 'Cluster3D_Label'] if c in X_pred]
    
    if cat_cols and encoder is not None:
        try:
            for col in cat_cols:
                X_pred[col] = X_pred[col].fillna('Unknown')
            
            encoded_today = encoder.transform(X_pred[cat_cols])
            encoded_today_df = pd.DataFrame(
                encoded_today, 
                columns=encoder.get_feature_names_out(cat_cols),
                index=X_pred.index
            )
            X_pred = pd.concat([
                X_pred.drop(columns=cat_cols).reset_index(drop=True),
                encoded_today_df.reset_index(drop=True)
            ], axis=1)
        except Exception as e:
            st.warning(f"Encoding failed: {e}. Using original features.")
            X_pred = X_pred.drop(columns=cat_cols, errors='ignore')

    X_pred = X_pred.fillna(0)
    X_pred = X_pred.loc[:, X_pred.notna().any(axis=0)]
    
    return X_pred

def make_predictions(main_model, X_pred, games_today):
    """Faz predições com o modelo principal"""
    try:
        ml_proba = main_model.predict_proba(X_pred)
        games_today["ML_Proba_Home"] = ml_proba[:, list(main_model.classes_).index("Home")]
        games_today["ML_Proba_Draw"] = ml_proba[:, list(main_model.classes_).index("Draw")] 
        games_today["ML_Proba_Away"] = ml_proba[:, list(main_model.classes_).index("Away")]

        # Calcular Market Error e EV
        games_today['Market_Error_Home'] = games_today['ML_Proba_Home'] - games_today['Imp_Prob_H']
        games_today['Market_Error_Away'] = games_today['ML_Proba_Away'] - games_today['Imp_Prob_A']
        games_today['Market_Error_Draw'] = games_today['ML_Proba_Draw'] - games_today['Imp_Prob_D']
        
        games_today['EV_Home'] = (games_today['ML_Proba_Home'] * games_today['Odd_H']) - 1
        games_today['EV_Away'] = (games_today['ML_Proba_Away'] * games_today['Odd_A']) - 1
        games_today['EV_Draw'] = (games_today['ML_Proba_Draw'] * games_today['Odd_D']) - 1
        
        st.success(f"✅ Successfully generated ML probabilities for {len(games_today)} games!")
        return games_today
        
    except Exception as pred_error:
        st.error(f"Prediction failed: {pred_error}")
        # Fallback
        games_today["ML_Proba_Home"] = games_today['Imp_Prob_H']
        games_today["ML_Proba_Draw"] = games_today['Imp_Prob_D']
        games_today["ML_Proba_Away"] = games_today['Imp_Prob_A']
        st.warning("Using market probabilities as fallback")
        return games_today

########################################
### Bloco 8 – Visualizations ###########
########################################
def create_value_scatter_plot(games_today, selected_date):
    """Cria gráfico de dispersão para value bets"""
    plot_data = games_today[['Imp_Prob_H', 'ML_Proba_Home', 'EV_Home', 'Market_Error_Home', 
                            'Home', 'Away', 'League', 'Odd_H']].copy()
    plot_data = plot_data.dropna()
    
    if not plot_data.empty:
        fig = px.scatter(
            plot_data,
            x='Imp_Prob_H',
            y='ML_Proba_Home',
            color='EV_Home',
            size=abs(plot_data['Market_Error_Home']),
            hover_data=['Home', 'Away', 'League', 'Odd_H'],
            title=f'ML Probability vs Market Probability (Home) - {selected_date}',
            color_continuous_scale='RdYlGn',
            range_color=[-0.5, 0.5]
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Market = ML'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for scatter plot")

def create_error_distribution(games_today, selected_date, min_value_gap):
    """Cria distribuição de erros de mercado"""
    error_data = games_today[['Market_Error_Home', 'Market_Error_Away']].copy().dropna()
    
    if not error_data.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=error_data['Market_Error_Home'], name='Home Error', opacity=0.7))
        fig2.add_trace(go.Histogram(x=error_data['Market_Error_Away'], name='Away Error', opacity=0.7))
        
        fig2.update_layout(
            title=f'Distribution of Market Errors - {selected_date}',
            barmode='overlay',
            xaxis_title='Market Error',
            yaxis_title='Count'
        )
        
        fig2.add_vline(x=min_value_gap, line_dash="dash", line_color="red", 
                      annotation_text="Value Threshold")
        fig2.add_vline(x=-min_value_gap, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available for error distribution")

def create_3d_cluster_visualization(games_today, selected_date):
    """Cria visualização dos clusters 3D"""
    if 'Cluster3D_Label' in games_today.columns and 'Quadrant_Dist_3D' in games_today.columns:
        # Garantir que as colunas para hover existem
        hover_columns = ['Home', 'Away', 'League']
        available_hover = [col for col in hover_columns if col in games_today.columns]
        
        # Adicionar EV_Home_Dual apenas se existir
        if 'EV_Home_Dual' in games_today.columns:
            available_hover.append('EV_Home_Dual')
        
        fig = px.scatter_3d(
            games_today,
            x='M_H',
            y='M_A', 
            z='Quadrant_Dist_3D',
            color='Cluster3D_Label',
            hover_data=available_hover,
            title=f'3D Cluster Visualization - {selected_date}',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

########################################
### Bloco 9 – Dual Value Detection ####
########################################
def detect_dual_value_bets(games_today, min_value_gap, value_confidence_threshold, min_odds):
    """Detecta value bets usando sistema dual"""
    
    def pick_dual_value_side(row):
        ev_home = row.get('EV_Home_Dual', -1)
        ev_away = row.get('EV_Away_Dual', -1)
        proba_home = row.get('Dual_Proba_Home', 0)
        proba_away = row.get('Dual_Proba_Away', 0)
        odd_h = row.get('Odd_H', 1)
        odd_a = row.get('Odd_A', 1)
        
        # Critérios para value bet
        home_value = (ev_home >= min_value_gap and 
                     proba_home >= value_confidence_threshold and 
                     odd_h >= min_odds)
        
        away_value = (ev_away >= min_value_gap and 
                     proba_away >= value_confidence_threshold and 
                     odd_a >= min_odds)
        
        if home_value and away_value:
            # Escolher o melhor EV
            if ev_home >= ev_away:
                return f"🏠 Value Home (EV: {ev_home:.3f})"
            else:
                return f"✈️ Value Away (EV: {ev_away:.3f})"
        elif home_value:
            return f"🏠 Value Home (EV: {ev_home:.3f})"
        elif away_value:
            return f"✈️ Value Away (EV: {ev_away:.3f})"
        else:
            return "❌ No Value"
    
    games_today['Dual_Value_Pick'] = games_today.apply(pick_dual_value_side, axis=1)
    
    return games_today

def display_dual_value_bets(value_bets, selected_date):
    """Exibe oportunidades de value bets do sistema dual"""
    if not value_bets.empty:
        value_bets = value_bets.sort_values(['Time'], ascending=False)
        
        cols_to_show = [
            'League','Time', 'Home', 'Away', 'Dual_Value_Pick',
            'Dual_Proba_Home', 'Dual_Proba_Away', 
            'EV_Home_Dual', 'EV_Away_Dual',
            'ML_Proba_Home', 'ML_Proba_Away',
            'Market_Error_Home', 'Market_Error_Away',
            'Odd_H', 'Odd_A',
            'Cluster3D_Label', 'Quadrant_Dist_3D'
        ]
        
        available_cols = [c for c in cols_to_show if c in value_bets.columns]
        
        st.dataframe(
            value_bets[available_cols]
            .style.format({
                'Dual_Proba_Home': '{:.3f}', 'Dual_Proba_Away': '{:.3f}',
                'EV_Home_Dual': '{:+.3f}', 'EV_Away_Dual': '{:+.3f}',
                'ML_Proba_Home': '{:.3f}', 'ML_Proba_Away': '{:.3f}',
                'Market_Error_Home': '{:+.3f}', 'Market_Error_Away': '{:+.3f}',
                'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}',
                'Quadrant_Dist_3D': '{:.2f}'
            })
            .background_gradient(subset=['EV_Home_Dual', 'EV_Away_Dual'], cmap='RdYlGn'),
            use_container_width=True,
            height=600
        )
        
        st.success(f"🎉 Found {len(value_bets)} dual value bet opportunities for {selected_date}!")
        
        # Estatísticas por cluster
        if 'Cluster3D_Label' in value_bets.columns:
            st.subheader("📊 Value Bets por Cluster 3D")
            cluster_stats = value_bets.groupby('Cluster3D_Label').agg({
                'Dual_Value_Pick': 'count',
                'EV_Home_Dual': 'mean',
                'EV_Away_Dual': 'mean'
            }).round(3)
            st.dataframe(cluster_stats)
    else:
        st.warning(f"No dual value bet opportunities found for {selected_date} with current filters.")

########################################
### Bloco 10 – Main Execution #########
########################################
def main():
    """Função principal"""
    try:
        # Setup
        selected_date = setup_sidebar()
        min_value_gap, value_confidence_threshold, min_odds = setup_parameters()
        league_filters = setup_league_filters()
        
        # Load data
        with st.spinner("Loading training data..."):
            history = load_training_data()
        if history is None:
            st.stop()

        with st.spinner(f"Loading data for {selected_date}..."):
            games_today = load_analysis_data(selected_date, history)
        if games_today is None:
            st.stop()

        # Ensure features exist (agora com 3D)
        games_today = ensure_features_exist(games_today)
        if games_today is None:
            st.stop()
        
        # Apply league filters
        if league_filters and 'League_Classification' in games_today.columns:
            games_today = games_today[games_today['League_Classification'].isin(league_filters)]
            st.info(f"Filtered leagues: {', '.join(league_filters)}")

        # Train main model
        try:
            main_model, encoder, features_raw = train_main_model(history, selected_date)
            if main_model is None:
                st.stop()
            st.success("✅ Main ML model trained successfully!")
            
        except Exception as e:
            st.error(f"Error training main model: {e}")
            st.stop()

        # Train dual models
        try:
            dual_model_home, dual_model_away, dual_features, league_columns = train_dual_value_models(history, selected_date)
            if dual_model_home is not None:
                st.success("✅ Dual value models trained successfully!")
            else:
                st.warning("⚠️ Dual models could not be trained")
        except Exception as e:
            st.error(f"Error training dual models: {e}")
            dual_model_home, dual_model_away, dual_features, league_columns = None, None, None, None

        # Market Error Analysis
        st.header(f"📊 3D Market Error Analysis - {selected_date}")
        
        if games_today.empty:
            st.error("No games data available for analysis!")
            return
            
        # Calculate market probabilities
        games_today = calculate_market_probabilities(games_today)
        
        # Prepare and make predictions
        X_pred = prepare_prediction_data(games_today, features_raw, encoder)
        if X_pred is None:
            st.stop()
            
        games_today = make_predictions(main_model, X_pred, games_today)
        
        # Calculate dual EV
        if dual_model_home is not None:
            games_today = calculate_dual_ev(games_today, dual_model_home, dual_model_away, dual_features, league_columns)
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            create_value_scatter_plot(games_today, selected_date)
        with col2:
            create_error_distribution(games_today, selected_date, min_value_gap)
        
        # 3D Visualization
        create_3d_cluster_visualization(games_today, selected_date)

        # Dual Value Bets Detection
        st.header("🧠 3D Dual Value Detection")
        games_today = detect_dual_value_bets(games_today, min_value_gap, value_confidence_threshold, min_odds)
        
        dual_value_bets = games_today[games_today['Dual_Value_Pick'] != "❌ No Value"].copy()
        display_dual_value_bets(dual_value_bets, selected_date)
        
        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            **💡 3D Dual ML Value Detection Methodology for {selected_date}:**
            - **3D Features**: Vector distances, trigonometric relationships, cluster patterns
            - **Dual Models**: Separate ML models for Home and Away value detection  
            - **Market Error**: Difference between ML probability and market implied probability
            - **Expected Value (EV)**: (ML Probability × Odds) - 1 (calculated separately for Home/Away)
            - **Cluster Analysis**: Games grouped by 3D momentum patterns
            - **⚠️ No Data Leak**: All models trained only on historical data before {selected_date}
            """
        )
        
    except Exception as e:
        st.error(f"Error in main execution: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
