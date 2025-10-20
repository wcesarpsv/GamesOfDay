# market_error_ml.py
########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
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

########################################
##### Bloco 4 – Sidebar Configs ########
########################################
st.sidebar.header("🔧 Meta-Model Configuration")

# Parâmetros do Value Detection
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
st.sidebar.subheader("🎯 League Filters")
show_high_var = st.sidebar.checkbox("High Variation Leagues", value=True)
show_medium_var = st.sidebar.checkbox("Medium Variation Leagues", value=True)
show_low_var = st.sidebar.checkbox("Low Variation Leagues", value=False)

########################################
####### Bloco 5 – Load & Prep Data #####
########################################
@st.cache_data
def load_data():
    """Carrega e prepara todos os dados necessários"""
    
    # Carregar histórico completo
    all_games = load_all_games(GAMES_FOLDER)
    all_games = filter_leagues(all_games)
    history = prepare_history(all_games)
    
    if history.empty:
        st.error("No valid historical data found.")
        return None, None, None
    
    # Carregar jogos mais recentes para análise
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
    if not files:
        st.error("No CSV files found.")
        return None, None, None
        
    latest_file = sorted(files)[-1]
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, latest_file))
    games_today = filter_leagues(games_today)
    
    # Aplicar processamento consistente
    games_today = compute_double_chance_odds(games_today)
    games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
    
    # Adicionar classificações de liga
    league_class = classify_leagues_variation(history)
    league_bands = compute_league_bands(history)
    
    games_today = games_today.merge(league_class, on='League', how='left')
    games_today = games_today.merge(league_bands, on='League', how='left')
    
    # Calcular bandas
    games_today['Home_Band'] = np.where(
        games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
        np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
    )
    games_today['Away_Band'] = np.where(
        games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
        np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
    )
    
    games_today['Dominant'] = games_today.apply(dominant_side, axis=1)
    
    return history, games_today, latest_file

# Carregar dados
with st.spinner("Loading data and training models..."):
    history, games_today, latest_file = load_data()

if history is None:
    st.stop()

########################################
### Bloco 6 – Train Main ML Model ######
########################################
@st.cache_resource
def train_main_model(_history):
    """Treina o modelo principal de classificação"""
    
    # Preparar target
    def map_result(row):
        if row['Goals_H_FT'] > row['Goals_A_FT']:
            return "Home"
        elif row['Goals_H_FT'] < row['Goals_A_FT']:
            return "Away"
        else:
            return "Draw"

    _history = _history.copy()
    _history['Result'] = _history.apply(map_result, axis=1)

    # Features do modelo principal
    features_raw = [
        'M_H','M_A','Diff_Power','M_Diff',
        'Home_Band','Away_Band','Dominant','League_Classification',
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
        'EV','Games_Analyzed'
    ]
    features_raw = [f for f in features_raw if f in _history.columns]

    X = _history[features_raw].copy()
    y = _history['Result']

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
        n_estimators=800,
        max_depth=12,
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
main_model, encoder, features_raw = train_main_model(history)

########################################
### Bloco 7 – Market Error Analysis ####
########################################
st.header("📊 Market Error Analysis")

# Calcular probabilidades implícitas do mercado
if all(col in games_today.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / games_today['Odd_H']
    probs['p_D'] = 1 / games_today['Odd_D']
    probs['p_A'] = 1 / games_today['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)

    games_today['Imp_Prob_H'] = probs['p_H']
    games_today['Imp_Prob_D'] = probs['p_D']
    games_today['Imp_Prob_A'] = probs['p_A']

    # Aplicar modelo principal para obter probabilidades ML
    X_today = games_today[features_raw].copy()
    
    if 'Home_Band' in X_today: 
        X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
    if 'Away_Band' in X_today: 
        X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

    cat_cols = [c for c in ['Dominant','League_Classification'] if c in X_today]
    
    if cat_cols:
        encoded_today = encoder.transform(X_today[cat_cols])
        encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
        X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                             encoded_today_df.reset_index(drop=True)], axis=1)

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
        
        # Criar scatter plot interativo
        fig = px.scatter(
            games_today,
            x='Imp_Prob_H',
            y='ML_Proba_Home',
            color='EV_Home',
            size='abs(Market_Error_Home)',
            hover_data=['Home', 'Away', 'League', 'Odd_H'],
            title='ML Probability vs Market Probability (Home)',
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
        
        # Adicionar áreas de value bet
        fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="LightGreen", width=2),
                     fillcolor="Green", opacity=0.1)
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Market Error Distribution")
        
        # Histograma dos errors
        fig2 = px.histogram(
            games_today,
            x=['Market_Error_Home', 'Market_Error_Away'],
            nbins=30,
            title='Distribution of Market Errors',
            barmode='overlay',
            opacity=0.7
        )
        
        fig2.add_vline(x=min_value_gap, line_dash="dash", line_color="red", annotation_text="Value Threshold")
        fig2.add_vline(x=-min_value_gap, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig2, use_container_width=True)

    ########################################
    ### Bloco 9 – Train Meta-Models #######
    ########################################
    st.header("🧠 Meta-Model Training - Value Detection")

    # Preparar dados históricos para meta-modelo
    value_history = history.copy()

    # Aplicar modelo principal ao histórico
    X_hist = value_history[[c for c in features_raw if c in value_history.columns]].copy()
    
    # Mapear bandas numéricas
    BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
    if 'Home_Band' in X_hist: 
        X_hist['Home_Band_Num'] = X_hist['Home_Band'].map(BAND_MAP)
    if 'Away_Band' in X_hist: 
        X_hist['Away_Band_Num'] = X_hist['Away_Band'].map(BAND_MAP)

    # One-hot encoding
    cat_cols = [c for c in ['Dominant','League_Classification'] if c in X_hist]
    if cat_cols:
        encoded_hist = encoder.transform(X_hist[cat_cols])
        encoded_hist_df = pd.DataFrame(encoded_hist, columns=encoder.get_feature_names_out(cat_cols))
        X_hist = pd.concat([X_hist.drop(columns=cat_cols).reset_index(drop=True),
                            encoded_hist_df.reset_index(drop=True)], axis=1)

    # Prever probabilidades com modelo principal
    ml_proba_hist = main_model.predict_proba(X_hist)
    value_history["ML_Proba_Home"] = ml_proba_hist[:, list(main_model.classes_).index("Home")]
    value_history["ML_Proba_Away"] = ml_proba_hist[:, list(main_model.classes_).index("Away")]

    # Calcular probabilidades implícitas históricas
    for col in ['Odd_H', 'Odd_D', 'Odd_A']:
        value_history[f'Imp_{col}'] = 1 / value_history[col]
    imp_sum = value_history[['Imp_Odd_H', 'Imp_Odd_D', 'Imp_Odd_A']].sum(axis=1)
    for col in ['Imp_Odd_H', 'Imp_Odd_D', 'Imp_Odd_A']:
        value_history[col] = value_history[col] / imp_sum

    # Mapear resultado
    def map_result_hist(row):
        if row['Goals_H_FT'] > row['Goals_A_FT']:
            return "Home"
        elif row['Goals_H_FT'] < row['Goals_A_FT']:
            return "Away"
        return "Draw"

    value_history['Result'] = value_history.apply(map_result_hist, axis=1)

    # Target Original
    value_history['Target_Value_Home'] = (
        (value_history['Result'] == "Home") &
        (1 / value_history['Odd_H'] > value_history['Imp_Odd_H'])
    ).astype(int)

    value_history['Target_Value_Away'] = (
        (value_history['Result'] == "Away") &
        (1 / value_history['Odd_A'] > value_history['Imp_Odd_A'])
    ).astype(int)

    # Target EV Teórico
    value_history['EV_Home'] = (value_history['ML_Proba_Home'] * value_history['Odd_H']) - 1
    value_history['EV_Away'] = (value_history['ML_Proba_Away'] * value_history['Odd_A']) - 1
    value_history['Target_EV_Home'] = (value_history['EV_Home'] > 0).astype(int)
    value_history['Target_EV_Away'] = (value_history['EV_Away'] > 0).astype(int)

    # Treinar meta-modelos
    features_value = ['M_H', 'M_A', 'Diff_Power', 'M_Diff', 'Odd_H', 'Odd_D', 'Odd_A']
    X_val = value_history[features_value].fillna(0)

    # Modelo para Home
    value_model_home = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    value_model_home.fit(X_val, value_history['Target_Value_Home'])

    # Modelo para Away
    value_model_away = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
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
        v_home, v_away = row['Value_Prob_Home'], row['Value_Prob_Away']
        odd_h, odd_a = row['Odd_H'], row['Odd_A']
        
        # Aplicar filtros
        if (v_home >= value_confidence_threshold and 
            v_home > v_away and 
            odd_h >= min_odds and
            row['Market_Error_Home'] >= min_value_gap):
            return f"🟢 Value Home ({v_home:.2f})"
        elif (v_away >= value_confidence_threshold and 
              v_away > v_home and 
              odd_a >= min_odds and
              row['Market_Error_Away'] >= min_value_gap):
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
            })
            .background_gradient(subset=['Value_Prob_Home', 'Value_Prob_Away'], cmap='RdYlGn')
            .background_gradient(subset=['EV_Home', 'EV_Away'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        st.success(f"🎉 Found {len(value_bets)} value bet opportunities!")
        
    else:
        st.warning("No value bet opportunities found with current filters.")
    
    ########################################
    ### Bloco 11 – Performance Analysis ###
    ########################################
    st.header("📈 Value Bet Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análise de correlação entre targets
        st.subheader("Target Correlation Analysis")
        
        if all(col in value_history.columns for col in ['Target_Value_Home', 'Target_EV_Home']):
            from scipy.stats import pearsonr
            
            corr_home, _ = pearsonr(value_history['Target_Value_Home'], value_history['Target_EV_Home'])
            corr_away, _ = pearsonr(value_history['Target_Value_Away'], value_history['Target_EV_Away'])
            
            concord_home = (value_history['Target_Value_Home'] == value_history['Target_EV_Home']).mean() * 100
            concord_away = (value_history['Target_Value_Away'] == value_history['Target_EV_Away']).mean() * 100
            
            metrics_df = pd.DataFrame({
                'Metric': ['Correlation Home', 'Correlation Away', 'Concordance Home', 'Concordance Away'],
                'Value': [corr_home, corr_away, concord_home, concord_away]
            })
            
            st.dataframe(metrics_df.style.format({'Value': '{:.3f}'}))
    
    with col2:
        # Feature Importance dos Meta-Modelos
        st.subheader("Meta-Model Feature Importance")
        
        feature_importance_home = pd.DataFrame({
            'feature': features_value,
            'importance': value_model_home.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance_home,
            x='importance',
            y='feature',
            orientation='h',
            title='Value Model Feature Importance (Home)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    ########################################
    ##### Bloco 12 – Detailed Analysis #####
    ########################################
    st.header("🔍 Detailed Market Error Analysis")
    
    # Tabela completa com todos os jogos
    detailed_cols = [
        'League', 'Home', 'Away', 'Value_ML_Pick',
        'ML_Proba_Home', 'Imp_Prob_H', 'Market_Error_Home', 'EV_Home',
        'ML_Proba_Away', 'Imp_Prob_A', 'Market_Error_Away', 'EV_Away',
        'Value_Prob_Home', 'Value_Prob_Away',
        'Odd_H', 'Odd_A', 'Odd_D'
    ]
    
    available_detailed = [c for c in detailed_cols if c in games_today.columns]
    
    st.dataframe(
        games_today[available_detailed]
        .sort_values(['Value_Prob_Home', 'Value_Prob_Away'], ascending=False)
        .style.format({
            'ML_Proba_Home': '{:.3f}', 'ML_Proba_Away': '{:.3f}',
            'Imp_Prob_H': '{:.3f}', 'Imp_Prob_A': '{:.3f}',
            'Market_Error_Home': '{:+.3f}', 'Market_Error_Away': '{:+.3f}',
            'EV_Home': '{:+.3f}', 'EV_Away': '{:+.3f}',
            'Value_Prob_Home': '{:.3f}', 'Value_Prob_Away': '{:.3f}',
            'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}', 'Odd_D': '{:.2f}'
        })
        .background_gradient(subset=['Market_Error_Home', 'Market_Error_Away'], cmap='RdYlGn')
        .background_gradient(subset=['EV_Home', 'EV_Away'], cmap='RdYlGn'),
        use_container_width=True,
        height=600
    )

else:
    st.error("Required odds columns (Odd_H, Odd_D, Odd_A) not found in the data.")

########################################
######## Bloco 13 – Footer #############
########################################
st.markdown("---")
st.markdown(
    """
    **💡 Value Bet Detection Methodology:**
    - **Market Error**: Difference between ML probability and market implied probability
    - **Expected Value (EV)**: (ML Probability × Odds) - 1
    - **Meta-Model**: Machine learning model trained to detect historical value patterns
    - **Value Confidence**: Probability that a bet represents genuine value
    """
)
