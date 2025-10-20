# market_error_ml.py
########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(
    page_title="Value Bet Detector - Market Error ML", 
    layout="wide",
    page_icon="🎯"
)

st.title("🎯 Market Error ML – Value Bet Intelligence")
st.markdown("### Meta-Modelo Avançado para Detecção de Apostas com Valor")

# Configurações principais (consistentes com o app principal)
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "afc","trophy"]

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

def get_available_dates():
    """Obtém todas as datas disponíveis nos arquivos (últimos 7 dias)"""
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
    dates = []
    for file in files:
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", file)
        if date_match:
            dates.append(date_match.group(0))
    
    # Ordenar e pegar apenas os últimos 7 dias
    dates = sorted(dates)
    if len(dates) > 7:
        dates = dates[-7:]
    
    return dates

def load_specific_date(target_date):
    """Carrega dados de uma data específica"""
    file_pattern = f"*{target_date}*.csv"
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
    
    # Ensure goal columns exist
    if 'Goals_H_Today' not in games_df.columns:
        games_df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in games_df.columns:
        games_df['Goals_A_Today'] = np.nan
    
    # Merge with the correct LiveScore file
    if os.path.exists(livescore_file):
        st.info(f"LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)
        
        # FILTER OUT CANCELED AND POSTPONED GAMES
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
        
        required_cols = [
            'game_id', 'status', 'home_goal', 'away_goal',
            'home_ht_goal', 'away_ht_goal',
            'home_corners', 'away_corners', 
            'home_yellow', 'away_yellow',
            'home_red', 'away_red'
        ]
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            st.warning(f"LiveScore file missing columns: {missing_cols}")
        else:
            games_df = games_df.merge(
                results_df,
                left_on='Id',
                right_on='game_id',
                how='left',
                suffixes=('', '_RAW')
            )
            
            # Update goals only for finished games
            games_df['Goals_H_Today'] = games_df['home_goal']
            games_df['Goals_A_Today'] = games_df['away_goal']
            games_df.loc[games_df['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
            
            # ADD RED CARD COLUMNS
            games_df['Home_Red'] = games_df['home_red']
            games_df['Away_Red'] = games_df['away_red']
            
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
st.sidebar.header("🔧 Meta-Model Configuration")

# Seletor de Data
st.sidebar.subheader("📅 Date Selection")
available_dates = get_available_dates()

if available_dates:
    selected_date = st.sidebar.selectbox(
        "Select Date to Analyze:",
        options=available_dates,
        index=len(available_dates)-1  # Última data por padrão
    )
    st.sidebar.info(f"Analyzing: {selected_date}")
else:
    st.sidebar.error("No date files found!")
    st.stop()

# Parâmetros do Value Detection
st.sidebar.subheader("🎯 Value Detection Parameters")
min_value_gap = st.sidebar.slider(
    "Minimum Value Gap", 
    min_value=0.01, max_value=0.20, value=0.05, step=0.01,
    help="Diferença mínima entre ML Prob e Market Prob para considerar value bet"
)

value_confidence_threshold = st.sidebar.slider(
    "Value Confidence Threshold", 
    min_value=0.50, max_value=0.90, value=0.55, step=0.01,
    help="Confiança mínima do meta-modelo para recomendar value bet"
)

min_odds = st.sidebar.number_input(
    "Minimum Odds", 
    min_value=1.5, max_value=5.0, value=1.8, step=0.1,
    help="Odds mínimas para considerar value bet"
)

# Filtros de Liga
st.sidebar.subheader("🏆 League Filters")
show_high_var = st.sidebar.checkbox("High Variation Leagues", value=True)
show_medium_var = st.sidebar.checkbox("Medium Variation Leagues", value=True)
show_low_var = st.sidebar.checkbox("Low Variation Leagues", value=False)

########################################
####### Bloco 5 – Load & Prep Data #####
########################################
@st.cache_data
def load_training_data():
    """Carrega dados de treinamento (SEM data leak)"""
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
    
    # 🔥 ADICIONAR MERGE COM LIVESCORE
    games_processed = merge_livescore_data(games_processed, target_date)
    
    # Adicionar coluna de resultado se temos dados de gols
    if all(col in games_processed.columns for col in ['Goals_H_Today', 'Goals_A_Today']):
        games_processed['Result_Today'] = games_processed.apply(determine_result, axis=1)
        
        # Estatísticas dos jogos finalizados
        finished_games = games_processed.dropna(subset=['Result_Today'])
        if len(finished_games) > 0:
            st.success(f"📊 {len(finished_games)} games have final results available!")
    
    return games_processed

# 🔥 CORREÇÃO: CARREGAR DADOS ANTES DE TREINAR MODELO
with st.spinner("Loading training data..."):
    history = load_training_data()

if history is None:
    st.stop()

with st.spinner(f"Loading data for {selected_date}..."):
    games_today = load_analysis_data(selected_date, history)

if games_today is None:
    st.stop()

# Aplicar filtros de liga
league_filters = []
if show_high_var:
    league_filters.append("High Variation")
if show_medium_var:
    league_filters.append("Medium Variation") 
if show_low_var:
    league_filters.append("Low Variation")

if league_filters and 'League_Classification' in games_today.columns:
    games_today = games_today[games_today['League_Classification'].isin(league_filters)]
    st.info(f"Filtered leagues: {', '.join(league_filters)}")

########################################
### Bloco 6 – Train Main ML Model ######
########################################
@st.cache_resource
def train_main_model(_history, target_date):
    """Treina o modelo principal de classificação SEM data leak"""
    
    # FILTRO CRÍTICO: usar apenas dados ANTERIORES à data de análise
    if 'Date' in _history.columns:
        # Converter para datetime se necessário
        try:
            _history['Date'] = pd.to_datetime(_history['Date'])
            target_date_dt = pd.to_datetime(target_date)
            training_data = _history[_history['Date'] < target_date_dt].copy()
        except:
            # Se não conseguir converter datas, usar todos os dados (fallback)
            training_data = _history.copy()
            st.warning("⚠️ Date conversion failed - using all historical data")
    else:
        # Se não há coluna de data, usar todos os dados com warning
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

    # Features do modelo principal
    features_raw = [
        'M_H','M_A','Diff_Power','M_Diff',
        'Home_Band','Away_Band','Dominant','League_Classification',
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2'
    ]
    # Manter apenas colunas que existem
    features_raw = [f for f in features_raw if f in training_data.columns]

    X = training_data[features_raw].copy()
    y = training_data['Result']

    # Codificar variáveis categóricas
    BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
    if 'Home_Band' in X: 
        X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
    if 'Away_Band' in X: 
        X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

    cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    if cat_cols:
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                       encoded_df.reset_index(drop=True)], axis=1)

    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=200,
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

# 🔥 AGORA SIM: Treinar modelo principal (history já está definido)
try:
    main_model, encoder, features_raw = train_main_model(history, selected_date)
    if main_model is None:
        st.stop()
    st.success("✅ Main ML model trained successfully!")
except Exception as e:
    st.error(f"Error training main model: {e}")
    st.stop()

########################################
### Bloco 6 – Train Main ML Model ######
########################################
@st.cache_resource
def train_main_model(_history, target_date):
    """Treina o modelo principal de classificação SEM data leak"""
    
    # FILTRO CRÍTICO: usar apenas dados ANTERIORES à data de análise
    if 'Date' in _history.columns:
        # Converter para datetime se necessário
        try:
            _history['Date'] = pd.to_datetime(_history['Date'])
            target_date_dt = pd.to_datetime(target_date)
            training_data = _history[_history['Date'] < target_date_dt].copy()
        except:
            # Se não conseguir converter datas, usar todos os dados (fallback)
            training_data = _history.copy()
            st.warning("⚠️ Date conversion failed - using all historical data")
    else:
        # Se não há coluna de data, usar todos os dados com warning
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

    # Features do modelo principal
    features_raw = [
        'M_H','M_A','Diff_Power','M_Diff',
        'Home_Band','Away_Band','Dominant','League_Classification',
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2'
    ]
    # Manter apenas colunas que existem
    features_raw = [f for f in features_raw if f in training_data.columns]

    X = training_data[features_raw].copy()
    y = training_data['Result']

    # Codificar variáveis categóricas
    BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
    if 'Home_Band' in X: 
        X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
    if 'Away_Band' in X: 
        X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

    cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    if cat_cols:
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                       encoded_df.reset_index(drop=True)], axis=1)

    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=200,
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

# Treinar modelo principal
try:
    main_model, encoder, features_raw = train_main_model(history, selected_date)
    if main_model is None:
        st.stop()
    st.success("✅ Main ML model trained successfully!")
except Exception as e:
    st.error(f"Error training main model: {e}")
    st.stop()

########################################
### Bloco 7 – Market Error Analysis ####
########################################
st.header(f"📊 Market Error Analysis - {selected_date}")

# Verificar se temos as colunas necessárias de odds
required_odds_cols = ['Odd_H', 'Odd_D', 'Odd_A']
if not all(col in games_today.columns for col in required_odds_cols):
    st.error(f"Missing required odds columns: {required_odds_cols}")
    st.stop()

# Calcular probabilidades implícitas do mercado
try:
    probs = pd.DataFrame()
    probs['p_H'] = 1 / games_today['Odd_H']
    probs['p_D'] = 1 / games_today['Odd_D']
    probs['p_A'] = 1 / games_today['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)

    games_today['Imp_Prob_H'] = probs['p_H']
    games_today['Imp_Prob_D'] = probs['p_D']
    games_today['Imp_Prob_A'] = probs['p_A']

    # Aplicar modelo principal para obter probabilidades ML
    X_today = games_today[[f for f in features_raw if f in games_today.columns]].copy()
    
    # Mapear bandas numéricas
    BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
    if 'Home_Band' in X_today: 
        X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
    if 'Away_Band' in X_today: 
        X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

    cat_cols = [c for c in ['Dominant','League_Classification'] if c in X_today]
    
    if cat_cols and encoder is not None:
        encoded_today = encoder.transform(X_today[cat_cols])
        encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
        X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                             encoded_today_df.reset_index(drop=True)], axis=1)

    # Preencher NaN com 0 para evitar erros
    X_today = X_today.fillna(0)
    
    ml_proba = main_model.predict_proba(X_today)
    games_today["ML_Proba_Home"] = ml_proba[:, list(main_model.classes_).index("Home")]
    games_today["ML_Proba_Draw"] = ml_proba[:, list(main_model.classes_).index("Draw")]
    games_today["ML_Proba_Away"] = ml_proba[:, list(main_model.classes_).index("Away")]

    # Calcular Market Error
    games_today['Market_Error_Home'] = games_today['ML_Proba_Home'] - games_today['Imp_Prob_H']
    games_today['Market_Error_Away'] = games_today['ML_Proba_Away'] - games_today['Imp_Prob_A']
    games_today['Market_Error_Draw'] = games_today['ML_Proba_Draw'] - games_today['Imp_Prob_D']

    # Calcular Expected Value
    games_today['EV_Home'] = (games_today['ML_Proba_Home'] * games_today['Odd_H']) - 1
    games_today['EV_Away'] = (games_today['ML_Proba_Away'] * games_today['Odd_A']) - 1
    games_today['EV_Draw'] = (games_today['ML_Proba_Draw'] * games_today['Odd_D']) - 1

    ########################################
    ##### Bloco 8 – Value Scatter Plot ####
    ########################################
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Value Bet Scatter Plot")
        
        # Criar DataFrame limpo para o plot
        plot_data = games_today[['Imp_Prob_H', 'ML_Proba_Home', 'EV_Home', 'Market_Error_Home', 
                                'Home', 'Away', 'League', 'Odd_H']].copy()
        plot_data = plot_data.dropna()
        
        if not plot_data.empty:
            # Criar scatter plot interativo
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
            
            # Adicionar linha de igualdade
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

    with col2:
        st.subheader("📈 Market Error Distribution")
        
        # Preparar dados para histograma
        error_data = games_today[['Market_Error_Home', 'Market_Error_Away']].copy().dropna()
        
        if not error_data.empty:
            # Criar histograma
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

    ########################################
    ### Bloco 9 – Train Meta-Models #######
    ########################################
    st.header("🧠 Meta-Model Training - Value Detection")

    # Preparar dados históricos para meta-modelo (SEM data leak)
    value_history = history.copy()
    
    # Aplicar mesmo filtro de data para meta-modelo
    if 'Date' in value_history.columns:
        try:
            value_history['Date'] = pd.to_datetime(value_history['Date'])
            target_date_dt = pd.to_datetime(selected_date)
            value_history = value_history[value_history['Date'] < target_date_dt].copy()
        except:
            pass  # Manter todos se conversão falhar

    # Aplicar modelo principal ao histórico para obter ML_Proba
    try:
        X_hist = value_history[[c for c in features_raw if c in value_history.columns]].copy()
        
        # Mapear bandas numéricas
        BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
        if 'Home_Band' in X_hist: 
            X_hist['Home_Band_Num'] = X_hist['Home_Band'].map(BAND_MAP)
        if 'Away_Band' in X_hist: 
            X_hist['Away_Band_Num'] = X_hist['Away_Band'].map(BAND_MAP)

        # One-hot encoding
        cat_cols = [c for c in ['Dominant','League_Classification'] if c in X_hist]
        if cat_cols and encoder is not None:
            encoded_hist = encoder.transform(X_hist[cat_cols])
            encoded_hist_df = pd.DataFrame(encoded_hist, columns=encoder.get_feature_names_out(cat_cols))
            X_hist = pd.concat([X_hist.drop(columns=cat_cols).reset_index(drop=True),
                                encoded_hist_df.reset_index(drop=True)], axis=1)

        # Preencher NaN
        X_hist = X_hist.fillna(0)
        
        # Prever probabilidades com modelo principal
        ml_proba_hist = main_model.predict_proba(X_hist)
        value_history["ML_Proba_Home"] = ml_proba_hist[:, list(main_model.classes_).index("Home")]
        value_history["ML_Proba_Away"] = ml_proba_hist[:, list(main_model.classes_).index("Away")]
        
    except Exception as e:
        st.warning(f"Could not generate ML probabilities for history: {e}")
        # Usar valores dummy para continuar
        value_history["ML_Proba_Home"] = 0.5
        value_history["ML_Proba_Away"] = 0.5

    # Calcular probabilidades implícitas históricas
    if all(col in value_history.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
        probs_hist = pd.DataFrame()
        probs_hist['p_H'] = 1 / value_history['Odd_H']
        probs_hist['p_D'] = 1 / value_history['Odd_D'] 
        probs_hist['p_A'] = 1 / value_history['Odd_A']
        probs_hist = probs_hist.div(probs_hist.sum(axis=1), axis=0)
        
        value_history['Imp_Prob_H'] = probs_hist['p_H']
        value_history['Imp_Prob_A'] = probs_hist['p_A']

    # Mapear resultado
    def map_result_hist(row):
        if row['Goals_H_FT'] > row['Goals_A_FT']:
            return "Home"
        elif row['Goals_H_FT'] < row['Goals_A_FT']:
            return "Away"
        return "Draw"

    value_history['Result'] = value_history.apply(map_result_hist, axis=1)

    # Target Original - simplificado
    value_history['Target_Value_Home'] = (
        (value_history['Result'] == "Home")
    ).astype(int)

    value_history['Target_Value_Away'] = (
        (value_history['Result'] == "Away") 
    ).astype(int)

    # Target EV Teórico
    if all(col in value_history.columns for col in ['ML_Proba_Home', 'Odd_H']):
        value_history['EV_Home'] = (value_history['ML_Proba_Home'] * value_history['Odd_H']) - 1
        value_history['EV_Away'] = (value_history['ML_Proba_Away'] * value_history['Odd_A']) - 1
        value_history['Target_EV_Home'] = (value_history['EV_Home'] > 0).astype(int)
        value_history['Target_EV_Away'] = (value_history['EV_Away'] > 0).astype(int)

    # Treinar meta-modelos simplificados
    features_value = ['M_H', 'M_A', 'Diff_Power', 'M_Diff', 'Odd_H', 'Odd_A']
    features_value = [f for f in features_value if f in value_history.columns]
    
    if features_value and len(value_history) > 0:
        X_val = value_history[features_value].fillna(0)
        
        # Modelo para Home
        value_model_home = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        value_model_home.fit(X_val, value_history['Target_Value_Home'])

        # Modelo para Away
        value_model_away = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=24,
            n_jobs=-1
        )
        value_model_away.fit(X_val, value_history['Target_Value_Away'])

        # Aplicar meta-modelos aos dados atuais
        X_today_val = games_today[features_value].fillna(0)
        val_pred_home = value_model_home.predict_proba(X_today_val)[:, 1]
        val_pred_away = value_model_away.predict_proba(X_today_val)[:, 1]

        games_today['Value_Prob_Home'] = val_pred_home
        games_today['Value_Prob_Away'] = val_pred_away

        # Classificar value bets
        def pick_value_side(row):
            if 'Value_Prob_Home' not in row or 'Value_Prob_Away' not in row:
                return "❌ No Value"
                
            v_home, v_away = row['Value_Prob_Home'], row['Value_Prob_Away']
            odd_h = row.get('Odd_H', 1)
            odd_a = row.get('Odd_A', 1)
            me_home = row.get('Market_Error_Home', 0)
            me_away = row.get('Market_Error_Away', 0)
            
            # Aplicar filtros
            if (v_home >= value_confidence_threshold and 
                v_home > v_away and 
                odd_h >= min_odds and
                me_home >= min_value_gap):
                return f"🟢 Value Home ({v_home:.2f})"
            elif (v_away >= value_confidence_threshold and 
                  v_away > v_home and 
                  odd_a >= min_odds and
                  me_away >= min_value_gap):
                return f"🟠 Value Away ({v_away:.2f})"
            else:
                return "❌ No Value"

        games_today['Value_ML_Pick'] = games_today.apply(pick_value_side, axis=1)

        ########################################
        ##### Bloco 10 – Top Value Bets #######
        ########################################
        st.header("🔥 Top Value Bet Opportunities")
        
        # Filtrar value bets
        value_bets = games_today[games_today['Value_ML_Pick'] != "❌ No Value"].copy()
        
        if not value_bets.empty:
            # Ordenar por confiança do meta-modelo
            value_bets = value_bets.sort_values(['Value_Prob_Home', 'Value_Prob_Away'], ascending=False)
            
            # Exibir tabela de oportunidades
            cols_to_show = [
                'League', 'Home', 'Away', 'Value_ML_Pick',
                'Value_Prob_Home', 'Value_Prob_Away', 
                'ML_Proba_Home', 'ML_Proba_Away',
                'Imp_Prob_H', 'Imp_Prob_A',
                'Market_Error_Home', 'Market_Error_Away',
                'EV_Home', 'EV_Away',
                'Odd_H', 'Odd_A'
            ]
            
            available_cols = [c for c in cols_to_show if c in value_bets.columns]
            
            st.dataframe(
                value_bets[available_cols]
                .style.format({
                    'Value_Prob_Home': '{:.3f}',
                    'Value_Prob_Away': '{:.3f}',
                    'ML_Proba_Home': '{:.3f}',
                    'ML_Proba_Away': '{:.3f}',
                    'Imp_Prob_H': '{:.3f}',
                    'Imp_Prob_A': '{:.3f}',
                    'Market_Error_Home': '{:+.3f}',
                    'Market_Error_Away': '{:+.3f}',
                    'EV_Home': '{:+.3f}',
                    'EV_Away': '{:+.3f}',
                    'Odd_H': '{:.2f}',
                    'Odd_A': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            st.success(f"🎉 Found {len(value_bets)} value bet opportunities for {selected_date}!")
            
        else:
            st.warning(f"No value bet opportunities found for {selected_date} with current filters.")

    else:
        st.warning("Not enough features available for meta-model training")

    # 🔥 NOVO BLOCO - Performance Analysis with Live Results
    if 'Result_Today' in games_today.columns:
        st.header("🏆 Value Bet Performance with Live Results")
        
        finished_games = games_today.dropna(subset=['Result_Today'])
        
        if not finished_games.empty:
            # Função para verificar se recommendation foi correta
            def check_recommendation(rec, result):
                if pd.isna(rec) or result is None or rec == '❌ No Value':
                    return None
                rec = str(rec)
                if 'Value Home' in rec:
                    return result == "Home"
                elif 'Value Away' in rec:
                    return result == "Away"
                return None
            
            # Aplicar verificação
            finished_games['Value_Correct'] = finished_games.apply(
                lambda r: check_recommendation(r['Value_ML_Pick'], r['Result_Today']), axis=1
            )
            
            # Calcular estatísticas
            value_bets_made = finished_games[finished_games['Value_ML_Pick'].str.contains('Value', na=False)]
            correct_bets = value_bets_made['Value_Correct'].sum()
            total_value_bets = len(value_bets_made)
            
            if total_value_bets > 0:
                win_rate = (correct_bets / total_value_bets) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Value Bets Made", total_value_bets)
                with col2:
                    st.metric("Correct Predictions", int(correct_bets))
                with col3:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                # Tabela de resultados
                results_cols = [
                    'League', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today', 
                    'Result_Today', 'Value_ML_Pick', 'Value_Correct',
                    'ML_Proba_Home', 'ML_Proba_Away', 'Market_Error_Home', 'Market_Error_Away'
                ]
                available_results = [c for c in results_cols if c in finished_games.columns]
                
                st.dataframe(
                    finished_games[available_results]
                    .style.format({
                        'ML_Proba_Home': '{:.3f}', 'ML_Proba_Away': '{:.3f}',
                        'Market_Error_Home': '{:+.3f}', 'Market_Error_Away': '{:+.3f}'
                    })
                    .apply(lambda x: ['background: lightgreen' if x['Value_Correct'] == True 
                                    else 'background: lightcoral' if x['Value_Correct'] == False 
                                    else '' for _ in x], axis=1),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No value bets were made on finished games.")
        else:
            st.info("No games with final results available yet.")

    ########################################
    ##### Bloco 11 – Detailed Analysis #####
    ########################################
    st.header("🔍 Detailed Market Error Analysis")
    
    # Tabela completa com todos os jogos
    detailed_cols = [
        'League', 'Home', 'Away', 
        'Goals_H_Today', 'Goals_A_Today', 'Result_Today',  # 🔥 NOVAS COLUNAS
        'ML_Proba_Home', 'Imp_Prob_H', 'Market_Error_Home', 'EV_Home',
        'ML_Proba_Away', 'Imp_Prob_A', 'Market_Error_Away', 'EV_Away',
        'Odd_H', 'Odd_A', 'Odd_D'
    ]
    
    # Adicionar Value_Prob se disponível
    if 'Value_Prob_Home' in games_today.columns:
        detailed_cols.extend(['Value_Prob_Home', 'Value_Prob_Away', 'Value_ML_Pick'])
    
    available_detailed = [c for c in detailed_cols if c in games_today.columns]
    
    # Filtrar dados para a tabela
    display_data = games_today[available_detailed].copy().dropna()
    
    if not display_data.empty:
        st.dataframe(
            display_data
            .style.format({
                'ML_Proba_Home': '{:.3f}', 'ML_Proba_Away': '{:.3f}',
                'Goals_H_Today': '{:.0f}', 'Goals_A_Today': '{:.0f}',
                'Imp_Prob_H': '{:.3f}', 'Imp_Prob_A': '{:.3f}',
                'Market_Error_Home': '{:+.3f}', 'Market_Error_Away': '{:+.3f}',
                'EV_Home': '{:+.3f}', 'EV_Away': '{:+.3f}',
                'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}', 'Odd_D': '{:.2f}'
            }),
            use_container_width=True,
            height=600
        )
    else:
        st.warning("No data available for detailed analysis")

except Exception as e:
    st.error(f"Error in market error analysis: {e}")
    import traceback
    st.code(traceback.format_exc())

########################################
######## Bloco 12 – Footer #############
########################################
st.markdown("---")
st.markdown(
    f"""
    **💡 Value Bet Detection Methodology for {selected_date}:**
    - **Market Error**: Difference between ML probability and market implied probability  
    - **Expected Value (EV)**: (ML Probability × Odds) - 1
    - **Meta-Model**: Machine learning model trained on data BEFORE {selected_date}
    - **Value Confidence**: Probability that a bet represents genuine value
    - **⚠️ No Data Leak**: Models trained only on historical data before selected date
    """
)
