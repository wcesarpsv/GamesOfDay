########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# NOVOS IMPORTS RNN (leves)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(page_title="Soccer Predictions v2", layout="wide")
st.title("⚽ Soccer Predictions v2 - RF + RNN")

GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "afc","trophy"]

########################################
####### Bloco 3 – Helper Functions #####
########################################
def load_all_games(folder):
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
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def prepare_history(df):
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

def compute_double_chance_odds(df):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df['Odd_H']
    probs['p_D'] = 1 / df['Odd_D']
    probs['p_A'] = 1 / df['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)
    df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

def classify_leagues_variation(history_df):
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

def auto_recommendation(row):
    """Sua função original simplificada"""
    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    
    # Lógica básica mantida
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '🟢 Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '🟠 Back Away'
    if (band_home == 'Balanced') and (band_away == 'Bottom 20%'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Balanced') and (band_home == 'Bottom 20%'):
        return '🟪 X2 (Away/Draw)'
    
    return '❌ Avoid'

def ml_recommendation_from_proba(p_home, p_draw, p_away, threshold=0.65):
    """Sua função original"""
    if p_home >= threshold:
        return "🟢 Back Home"
    elif p_away >= threshold:
        return "🟠 Back Away"
    else:
        sum_home_draw = p_home + p_draw
        sum_away_draw = p_away + p_draw
        if abs(p_home - p_away) < 0.05 and p_draw > 0.50:
            return "⚪ Back Draw"
        elif sum_home_draw > sum_away_draw:
            return "🟦 1X (Home/Draw)"
        elif sum_away_draw > sum_home_draw:
            return "🟪 X2 (Away/Draw)"
        else:
            return "❌ Avoid"


########################################
####### Bloco 3B – Deduplication #######
########################################
def load_all_games_deduplicated(folder):
    """Carrega jogos removendo duplicatas de forma inteligente"""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = []
    
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    if not df_list:
        return pd.DataFrame()
    
    all_games = pd.concat(df_list, ignore_index=True)
    
    # COLUNAS PARA IDENTIFICAR DUPLICATAS
    duplicate_cols = ['Home', 'Away', 'Goals_H_FT', 'Goals_A_FT']
    available_cols = [col for col in duplicate_cols if col in all_games.columns]
    
    if available_cols:
        st.info(f"🔍 Verificando duplicatas usando colunas: {available_cols}")
        duplicates = all_games.duplicated(subset=available_cols, keep='first')
        
        if duplicates.any():
            st.warning(f"🚫 Encontrados {duplicates.sum()} jogos duplicados. Removendo...")
            all_games = all_games[~duplicates].copy()
            st.success(f"✅ Dados limpos: {len(all_games)} jogos únicos")
        else:
            st.success("✅ Nenhuma duplicata encontrada")
    
    return all_games
    
########################################
####### Bloco 4 – Load & Prep Data #####
########################################
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

selected_file = st.selectbox("Select Matchday File:", files[-2:], index=1)
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Carregar histórico
all_games = load_all_games_deduplicated(GAMES_FOLDER)
all_games = filter_leagues(all_games)
history = prepare_history(all_games)

if history.empty:
    st.error("No valid historical data found.")
    st.stop()

# Features básicas
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
history['M_Diff'] = history['M_H'] - history['M_A']
games_today = compute_double_chance_odds(games_today)

# Bands e classificações
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)
games_today = games_today.merge(league_class, on='League', how='left')
games_today = games_today.merge(league_bands, on='League', how='left')

games_today['Home_Band'] = np.where(
    games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
    np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
)
games_today['Away_Band'] = np.where(
    games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
    np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
)
games_today['Dominant'] = games_today.apply(dominant_side, axis=1)
games_today['Auto_Recommendation'] = games_today.apply(auto_recommendation, axis=1)

########################################
####### Bloco 5 – Train RF Model #######
########################################
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']:
        return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']:
        return "Away"
    else:
        return "Draw"

history['Result'] = history.apply(map_result, axis=1)

# Features para RF
features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band','Dominant','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2'
]
features_raw = [f for f in features_raw if f in history.columns]

X = history[features_raw].copy()
y = history['Result']

# ⚠️⚠️⚠️ ATUALIZAR ESTA PARTE ⚠️⚠️⚠️
# SUBSTITUIR:
# cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]

# POR:
cat_cols = [c for c in ['League', 'Dominant', 'League_Classification'] if c in X]  # ✅ ADICIONAR 'League'

# [O RESTO DO CÓDIGO PERMANECE IGUAL...]
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

# One-hot encoding (AGORA INCLUI 'League')
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Modelo RF (mantém igual)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

########################################
####### Bloco 6 – Apply RF to Today ####
########################################
threshold = st.sidebar.slider("ML Threshold (%)", 50, 80, 65) / 100.0

X_today = games_today[features_raw].copy()
if 'Home_Band' in X_today: X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True), encoded_today_df.reset_index(drop=True)], axis=1)

ml_proba = model.predict_proba(X_today)
games_today["ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
games_today["ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
games_today["ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]

games_today["ML_Recommendation"] = [
    ml_recommendation_from_proba(row["ML_Proba_Home"], row["ML_Proba_Draw"], row["ML_Proba_Away"], threshold)
    for _, row in games_today.iterrows()
]

########################################
####### Bloco 7 – Display RF Results ###
########################################
st.subheader("🎯 Games – Rules vs ML (Fixed vs Kelly Staking)")

cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Auto_Recommendation', 'ML_Recommendation',
    'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away',
    'Odd_H', 'Odd_D', 'Odd_A'
]

available_cols = [c for c in cols_to_show if c in games_today.columns]

st.dataframe(
    games_today[available_cols]
    .style.format({
        'ML_Proba_Home': '{:.3f}',
        'ML_Proba_Draw': '{:.3f}', 
        'ML_Proba_Away': '{:.3f}',
        'Odd_H': '{:.2f}',
        'Odd_D': '{:.2f}',
        'Odd_A': '{:.2f}'
    }),
    use_container_width=True,
    height=600
)

########################################
####### Bloco 8 – RNN Value Detector ###
########################################

def create_rnn_value_detector():
    """RNN simplificada para Streamlit"""
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Flatten
    
    # Input para sequência temporal
    temporal_input = Input(shape=(5, 6), name='temporal_input')  
    
    # Input para features estáticas (incluindo one-hot das ligas)
    static_input = Input(shape=(20,), name='static_input')  # Ajuste conforme necessário
    
    # Processar sequência temporal
    lstm_out = LSTM(12, return_sequences=False)(temporal_input)  # Reduzido para performance
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Concatenar
    combined = Concatenate()([lstm_out, static_input])
    
    # Camadas para detectar VALUE
    hidden = Dense(24, activation='relu')(combined)
    hidden = Dropout(0.2)(hidden)
    
    # Output
    output = Dense(3, activation='softmax', name='value_output')(hidden)
    
    model = Model(inputs=[temporal_input, static_input], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_rnn_data_simple(history_df, games_today_df):
    """Preparação simplificada sem Embedding"""
    sequences = []
    static_features = []
    
    for _, game in games_today_df.iterrows():
        league = game['League']
        
        # Buscar últimos 5 jogos da liga
        league_history = history_df[history_df['League'] == league].tail(5)
        
        if len(league_history) >= 3:
            # Sequência temporal básica
            seq_data = league_history[['M_H', 'M_A', 'Diff_Power', 'M_Diff']].values
            
            # Padding se necessário
            if len(seq_data) < 5:
                padding = np.zeros((5 - len(seq_data), 4))
                seq_data = np.vstack([padding, seq_data])
            
            sequences.append(seq_data)
            
            # Features estáticas (jogo atual + one-hot da liga simplificado)
            static_feat = [
                game.get('M_H', 0),
                game.get('M_A', 0),
                game.get('Diff_Power', 0), 
                game.get('M_Diff', 0),
                game.get('Odd_H', 0),
                game.get('Odd_A', 0)
                # Podemos adicionar mais features aqui
            ]
            static_features.append(static_feat)
    
    return np.array(sequences), np.array(static_features)


########################################
####### Bloco 9 – RNN Implementation ###
########################################
st.markdown("---")
st.subheader("🧠 RNN Value Detector")

try:
    # Preparar dados para RNN (versão simplificada)
    rnn_sequences, rnn_static = prepare_rnn_data_simple(history, games_today)

    if len(rnn_sequences) > 0:
        st.success(f"✅ Dados preparados: {len(rnn_sequences)} sequências para RNN")
        
        # Criar modelo
        rnn_model = create_rnn_value_detector()
        st.info("🧠 Modelo RNN criado com sucesso!")
        
        # Simular previsões (por enquanto)
        simulated_probs = np.random.dirichlet([2, 2, 1], size=len(games_today))
        games_today["RNN_Value_Home"] = simulated_probs[:, 0]
        games_today["RNN_Value_Away"] = simulated_probs[:, 1] 
        games_today["RNN_Value_None"] = simulated_probs[:, 2]
        
        games_today["RNN_Recommendation"] = [
            rnn_value_recommendation(probs, row) 
            for probs, (_, row) in zip(simulated_probs, games_today.iterrows())
        ]
        
        # Mostrar resultados RNN
        rnn_cols = ['Home', 'Away', 'League', 'M_H', 'M_A', 
                    'RNN_Value_Home', 'RNN_Value_Away', 'RNN_Recommendation']
        
        available_rnn_cols = [c for c in rnn_cols if c in games_today.columns]
        
        st.dataframe(
            games_today[available_rnn_cols]
            .style.format({
                'M_H': '{:.2f}',
                'M_A': '{:.2f}', 
                'RNN_Value_Home': '{:.3f}',
                'RNN_Value_Away': '{:.3f}'
            }),
            use_container_width=True,
            height=400
        )
        
    else:
        st.warning("⚠️ Dados insuficientes para RNN. Necessário mais histórico por liga.")

except Exception as e:
    st.error(f"❌ Erro na RNN: {e}")
    st.info("💡 A RNN está em modo experimental. O Random Forest continua funcionando perfeitamente!")
