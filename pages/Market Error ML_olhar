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

# Configurações principais
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
        
        required_cols = ['game_id', 'status', 'home_goal', 'away_goal']
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
    st.sidebar.header("🔧 Meta-Model Configuration")
    
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
    """Garante que todas as features necessárias existam"""
    expected_features = ['M_Diff', 'Home_Band', 'Away_Band', 'Dominant', 'League_Classification', 'Odd_1X', 'Odd_X2']
    missing_features = [f for f in expected_features if f not in games_today.columns]
    
    if missing_features:
        st.warning(f"⚠️ Features faltantes: {missing_features}")
        st.info("Aplicando correção manual...")
        
        # 1. Calcular M_Diff
        if 'M_Diff' not in games_today.columns:
            games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
            st.success("✅ M_Diff criada")

        # 2. Calcular odds dupla chance
        if all(col in games_today.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
            if 'Odd_1X' not in games_today.columns or 'Odd_X2' not in games_today.columns:
                probs_dc = pd.DataFrame()
                probs_dc['p_H'] = 1 / games_today['Odd_H']
                probs_dc['p_D'] = 1 / games_today['Odd_D']
                probs_dc['p_A'] = 1 / games_today['Odd_A']
                probs_dc = probs_dc.div(probs_dc.sum(axis=1), axis=0)
                games_today['Odd_1X'] = 1 / (probs_dc['p_H'] + probs_dc['p_D'])
                games_today['Odd_X2'] = 1 / (probs_dc['p_A'] + probs_dc['p_D'])
                st.success("✅ Odd_1X e Odd_X2 criadas")

        # 3. Criar bandas simples
        if 'Home_Band' not in games_today.columns:
            games_today['Home_Band'] = np.where(
                games_today['M_H'] > 0.5, 'Top 20%',
                np.where(games_today['M_H'] < -0.5, 'Bottom 20%', 'Balanced')
            )
            st.success("✅ Home_Band criada")

        if 'Away_Band' not in games_today.columns:
            games_today['Away_Band'] = np.where(
                games_today['M_A'] > 0.5, 'Top 20%', 
                np.where(games_today['M_A'] < -0.5, 'Bottom 20%', 'Balanced')
            )
            st.success("✅ Away_Band criada")

        # 4. Criar dominant side
        if 'Dominant' not in games_today.columns:
            try:
                games_today['Dominant'] = games_today.apply(dominant_side, axis=1)
                st.success("✅ Dominant criada")
            except Exception as e:
                st.error(f"Erro ao criar Dominant: {e}")
                games_today['Dominant'] = "Mixed / Neutral"

        # 5. League classification fallback
        if 'League_Classification' not in games_today.columns:
            games_today['League_Classification'] = 'Medium Variation'
            st.success("✅ League_Classification criada")

        st.success("🎉 Todas as features recriadas manualmente!")
    else:
        st.success("✅ Todas as features já existem!")
    
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

    # Features do modelo principal
    features_raw = [
        'M_H','M_A','Diff_Power','M_Diff',
        'Home_Band','Away_Band','Dominant','League_Classification',
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2'
    ]
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
    
    # Mapear bandas numéricas
    BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}
    if 'Home_Band' in X_pred:
        X_pred['Home_Band_Num'] = X_pred['Home_Band'].map(BAND_MAP).fillna(2)
    if 'Away_Band' in X_pred:
        X_pred['Away_Band_Num'] = X_pred['Away_Band'].map(BAND_MAP).fillna(2)

    # One-hot encoding
    cat_cols = [c for c in ['Dominant', 'League_Classification'] if c in X_pred]
    
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

########################################
### Bloco 9 – Meta-Models #############
########################################
def train_meta_models(history, games_today, features_raw, main_model, encoder, selected_date):
    """Treina meta-modelos para detecção de value bets"""
    value_history = history.copy()
    
    # Aplicar filtro de data
    if 'Date' in value_history.columns:
        try:
            value_history['Date'] = pd.to_datetime(value_history['Date'])
            target_date_dt = pd.to_datetime(selected_date)
            value_history = value_history[value_history['Date'] < target_date_dt].copy()
        except:
            pass

    # Aplicar modelo principal ao histórico
    try:
        X_hist = value_history[[c for c in features_raw if c in value_history.columns]].copy()
        
        BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
        if 'Home_Band' in X_hist: 
            X_hist['Home_Band_Num'] = X_hist['Home_Band'].map(BAND_MAP)
        if 'Away_Band' in X_hist: 
            X_hist['Away_Band_Num'] = X_hist['Away_Band'].map(BAND_MAP)

        cat_cols = [c for c in ['Dominant','League_Classification'] if c in X_hist]
        if cat_cols and encoder is not None:
            encoded_hist = encoder.transform(X_hist[cat_cols])
            encoded_hist_df = pd.DataFrame(encoded_hist, columns=encoder.get_feature_names_out(cat_cols))
            X_hist = pd.concat([X_hist.drop(columns=cat_cols).reset_index(drop=True),
                                encoded_hist_df.reset_index(drop=True)], axis=1)

        X_hist = X_hist.fillna(0)
        
        ml_proba_hist = main_model.predict_proba(X_hist)
        value_history["ML_Proba_Home"] = ml_proba_hist[:, list(main_model.classes_).index("Home")]
        value_history["ML_Proba_Away"] = ml_proba_hist[:, list(main_model.classes_).index("Away")]
        
    except Exception as e:
        st.warning(f"Could not generate ML probabilities for history: {e}")
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
    value_history['Target_Value_Home'] = (value_history['Result'] == "Home").astype(int)
    value_history['Target_Value_Away'] = (value_history['Result'] == "Away").astype(int)

    # Treinar meta-modelos
    features_value = ['M_H', 'M_A', 'Diff_Power', 'M_Diff', 'Odd_H', 'Odd_A']
    features_value = [f for f in features_value if f in value_history.columns]
    
    if features_value and len(value_history) > 0:
        X_val = value_history[features_value].fillna(0)
        
        value_model_home = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1
        )
        value_model_home.fit(X_val, value_history['Target_Value_Home'])

        value_model_away = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced', random_state=24, n_jobs=-1
        )
        value_model_away.fit(X_val, value_history['Target_Value_Away'])

        # Aplicar meta-modelos
        X_today_val = games_today[features_value].fillna(0)
        val_pred_home = value_model_home.predict_proba(X_today_val)[:, 1]
        val_pred_away = value_model_away.predict_proba(X_today_val)[:, 1]

        games_today['Value_Prob_Home'] = val_pred_home
        games_today['Value_Prob_Away'] = val_pred_away
        
        return games_today
    else:
        st.warning("Not enough features available for meta-model training")
        return games_today

########################################
### Bloco 10 – Value Bet Detection ####
########################################
def detect_value_bets(games_today, min_value_gap, value_confidence_threshold, min_odds):
    """Detecta value bets baseado nos meta-modelos"""
    def pick_value_side(row):
        if 'Value_Prob_Home' not in row or 'Value_Prob_Away' not in row:
            return "❌ No Value"
            
        v_home, v_away = row['Value_Prob_Home'], row['Value_Prob_Away']
        odd_h = row.get('Odd_H', 1)
        odd_a = row.get('Odd_A', 1)
        me_home = row.get('Market_Error_Home', 0)
        me_away = row.get('Market_Error_Away', 0)
        
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
    return games_today

def display_value_bets(value_bets, selected_date):
    """Exibe oportunidades de value bets"""
    if not value_bets.empty:
        value_bets = value_bets.sort_values(['Value_Prob_Home', 'Value_Prob_Away'], ascending=False)
        
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
                'Value_Prob_Home': '{:.3f}', 'Value_Prob_Away': '{:.3f}',
                'ML_Proba_Home': '{:.3f}', 'ML_Proba_Away': '{:.3f}',
                'Imp_Prob_H': '{:.3f}', 'Imp_Prob_A': '{:.3f}',
                'Market_Error_Home': '{:+.3f}', 'Market_Error_Away': '{:+.3f}',
                'EV_Home': '{:+.3f}', 'EV_Away': '{:+.3f}',
                'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        st.success(f"🎉 Found {len(value_bets)} value bet opportunities for {selected_date}!")
    else:
        st.warning(f"No value bet opportunities found for {selected_date} with current filters.")

########################################
### Bloco 11 – Main Execution #########
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

        # Ensure features exist
        games_today = ensure_features_exist(games_today)
        
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
            
            # Verify features
            st.subheader("🔍 Verificação das Features do Modelo")
            st.write(f"Features_raw: {features_raw}")
            st.write(f"Quantidade: {len(features_raw)}")
            
        except Exception as e:
            st.error(f"Error training main model: {e}")
            st.stop()

        # Market Error Analysis
        st.header(f"📊 Market Error Analysis - {selected_date}")
        
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
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            create_value_scatter_plot(games_today, selected_date)
        with col2:
            create_error_distribution(games_today, selected_date, min_value_gap)

        # Meta-models and Value Bets
        st.header("🧠 Meta-Model Training - Value Detection")
        games_today = train_meta_models(history, games_today, features_raw, main_model, encoder, selected_date)
        
        games_today = detect_value_bets(games_today, min_value_gap, value_confidence_threshold, min_odds)
        
        value_bets = games_today[games_today['Value_ML_Pick'] != "❌ No Value"].copy()
        display_value_bets(value_bets, selected_date)
        
        # Footer
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
        
    except Exception as e:
        st.error(f"Error in main execution: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
