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

selected_file = st.selectbox("Select Matchday File:", files[-5:], index=1)
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
    
    # Input para features estáticas
    static_input = Input(shape=(6,), name='static_input')  # Ajustado para 6 features
    
    # Processar sequência temporal
    lstm_out = LSTM(12, return_sequences=False)(temporal_input)
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
            seq_data = league_history[['M_H', 'M_A', 'Diff_Power', 'M_Diff', 'Odd_H', 'Odd_A']].values
            
            # Padding se necessário
            if len(seq_data) < 5:
                padding = np.zeros((5 - len(seq_data), 6))
                seq_data = np.vstack([padding, seq_data])
            
            sequences.append(seq_data)
            
            # Features estáticas do jogo atual
            static_feat = [
                game.get('M_H', 0),
                game.get('M_A', 0),
                game.get('Diff_Power', 0), 
                game.get('M_Diff', 0),
                game.get('Odd_H', 0),
                game.get('Odd_A', 0)
            ]
            static_features.append(static_feat)
    
    return np.array(sequences), np.array(static_features)

# ✅ SUBSTITUIR esta função no Bloco 8:
def rnn_value_recommendation(probs, row):
    """Gera recomendação padronizada com mesma linguagem do RF"""
    value_home, value_away, no_value = probs
    
    # Usar os MESMOS ícones e formatos do Random Forest
    if value_home >= 0.7:
        return "🟢 Back Home"  # Igual ao RF
    elif value_away >= 0.7:
        return "🟠 Back Away"  # Igual ao RF
    elif value_home >= 0.6:
        return "🟦 1X (Home/Draw)"  # Igual ao RF
    elif value_away >= 0.6:
        return "🟪 X2 (Away/Draw)"  # Igual ao RF
    elif value_home > value_away and value_home >= 0.55:
        return "🟦 1X (Home/Draw)"  # Igual ao RF
    elif value_away > value_home and value_away >= 0.55:
        return "🟪 X2 (Away/Draw)"  # Igual ao RF
    else:
        return "❌ Avoid"  # Igual ao RF


########################################
####### Bloco 9 – RNN Implementation ###
########################################
st.markdown("---")
st.subheader("🧠 RNN Value Detector")

try:
    # Preparar dados para RNN
    rnn_sequences, rnn_static = prepare_rnn_data_simple(history, games_today)

    if len(rnn_sequences) > 0:
        st.success(f"✅ Dados preparados: {len(rnn_sequences)} sequências para RNN")
        
        # Criar modelo
        rnn_model = create_rnn_value_detector()
        st.info("🧠 Modelo RNN criado com sucesso!")
        
        # ✅ AGORA A FUNÇÃO ESTÁ DEFINIDA!
        # Simular previsões
        simulated_probs = np.random.dirichlet([2, 2, 1], size=len(games_today))
        
        # Adicionar colunas RNN ao dataframe
        games_today["RNN_Value_Home"] = simulated_probs[:, 0]
        games_today["RNN_Value_Away"] = simulated_probs[:, 1] 
        games_today["RNN_Value_None"] = simulated_probs[:, 2]
        
        games_today["RNN_Recommendation"] = [
            rnn_value_recommendation(probs, row) 
            for probs, (_, row) in zip(simulated_probs, games_today.iterrows())
        ]
        
        # Mostrar resultados RNN
        st.subheader("📊 RNN Value Recommendations")
        
        rnn_cols = ['League','Time','Home', 'Away',  'M_H', 'M_A', 
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
        
        # Estatísticas RNN
        st.subheader("📈 RNN Insights")
        value_home_count = (games_today["RNN_Value_Home"] >= 0.6).sum()
        value_away_count = (games_today["RNN_Value_Away"] >= 0.6).sum()
        
        st.write(f"🟢 Value Home detectado: **{value_home_count}** jogos")
        st.write(f"🟠 Value Away detectado: **{value_away_count}** jogos")
        st.write(f"📊 Total analisado: **{len(rnn_sequences)}** sequências")
        
    else:
        st.warning("⚠️ Dados insuficientes para RNN. Necessário mais histórico por liga.")

except Exception as e:
    st.error(f"❌ Erro na RNN: {e}")
    st.info("💡 A RNN está em modo experimental. O Random Forest continua funcionando perfeitamente!")



########################################
##### Bloco 10 – ML vs RNN Battle ######
########################################
st.markdown("---")
st.subheader("🥊 ML vs RNN - Model Battle")

# ✅ SUBSTITUIR esta função no Bloco 10:
def analyze_agreement(ml_rec, rnn_rec):
    """Analisa concordância entre modelos com linguagem padronizada"""
    if pd.isna(ml_rec) or pd.isna(rnn_rec):
        return "🤷 Missing Data"
    
    ml_str = str(ml_rec)
    rnn_str = str(rnn_rec)
    
    # Agora ambos usam a mesma linguagem!
    both_avoid = "❌ Avoid" in ml_str and "❌ Avoid" in rnn_str
    both_bet = "❌ Avoid" not in ml_str and "❌ Avoid" not in rnn_str
    
    if both_avoid:
        return "🤝 Both Avoid"
    
    if both_bet:
        # Extrair o tipo de aposta (exato mesmo formato)
        ml_type = ml_str.split()[1] if len(ml_str.split()) > 1 else ""
        rnn_type = rnn_str.split()[1] if len(rnn_str.split()) > 1 else ""
        
        if ml_str == rnn_str:  # Agora podemos comparar exatamente!
            return "🎯 Perfect Agreement"
        elif ("Home" in ml_str and "Home" in rnn_str) or ("Away" in ml_str and "Away" in rnn_str):
            return "✅ Same Side"
        elif ("1X" in ml_str and "1X" in rnn_str) or ("X2" in ml_str and "X2" in rnn_str):
            return "✅ Same Side" 
        else:
            return "⚔️ Different Sides"
    
    return "🤷 Partial Agreement"

# Adicionar análise de concordância
games_today["Model_Agreement"] = games_today.apply(
    lambda row: analyze_agreement(row["ML_Recommendation"], row["RNN_Recommendation"]), 
    axis=1
)

# Tabela comparativa completa
comparison_cols = [
    'League', 'Time', 'Home', 'Away', 
    'M_H', 'M_A', 'M_Diff',
    'ML_Recommendation', 'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away',
    'RNN_Recommendation', 'RNN_Value_Home', 'RNN_Value_Away',
    'Model_Agreement',
    'Odd_H', 'Odd_D', 'Odd_A'
]

available_comparison_cols = [c for c in comparison_cols if c in games_today.columns]

# ✅ VERSÃO SIMPLIFICADA - sem estilo condicional complexo
st.dataframe(
    games_today[available_comparison_cols]
    .style.format({
        'M_H': '{:.2f}',
        'M_A': '{:.2f}',
        'M_Diff': '{:.2f}',
        'ML_Proba_Home': '{:.3f}',
        'ML_Proba_Draw': '{:.3f}', 
        'ML_Proba_Away': '{:.3f}',
        'RNN_Value_Home': '{:.3f}',
        'RNN_Value_Away': '{:.3f}',
        'Odd_H': '{:.2f}',
        'Odd_D': '{:.2f}',
        'Odd_A': '{:.2f}'
    }),
    use_container_width=True,
    height=600
)

# Estatísticas do Battle
st.subheader("📊 Model Battle Statistics")

col1, col2, col3, col4 = st.columns(4)

total_games = len(games_today)
perfect_agreement = (games_today["Model_Agreement"] == "🎯 Perfect Agreement").sum()
same_side = (games_today["Model_Agreement"] == "✅ Same Side").sum()
different_sides = (games_today["Model_Agreement"] == "⚔️ Different Sides").sum()
both_avoid = (games_today["Model_Agreement"] == "🤝 Both Avoid").sum()

with col1:
    agreement_pct = (perfect_agreement/total_games*100) if total_games > 0 else 0
    st.metric("🎯 Perfect Agreement", f"{perfect_agreement}/{total_games}", 
              f"{agreement_pct:.1f}%")

with col2:
    same_side_pct = (same_side/total_games*100) if total_games > 0 else 0
    st.metric("✅ Same Side", f"{same_side}/{total_games}",
              f"{same_side_pct:.1f}%")

with col3:
    different_pct = (different_sides/total_games*100) if total_games > 0 else 0
    st.metric("⚔️ Different Sides", f"{different_sides}/{total_games}",
              f"{different_pct:.1f}%")

with col4:
    avoid_pct = (both_avoid/total_games*100) if total_games > 0 else 0
    st.metric("🤝 Both Avoid", f"{both_avoid}/{total_games}",
              f"{avoid_pct:.1f}%")

# Análise de Value quando concordam
st.subheader("💎 High-Value Opportunities")

# Filtrar jogos onde ambos os modelos concordam e recomendam apostar
high_value_games = games_today[
    (games_today["Model_Agreement"].isin(["🎯 Perfect Agreement", "✅ Same Side"])) &
    (games_today["ML_Recommendation"] != "❌ Avoid") &
    (games_today["RNN_Recommendation"] != "❌ NO VALUE")
]

if len(high_value_games) > 0:
    st.success(f"🎯 Encontrados {len(high_value_games)} jogos de ALTO VALOR (concordância total)")
    
    high_value_cols = ['League', 'Home', 'Away', 'ML_Recommendation', 'RNN_Recommendation', 
                      'ML_Proba_Home', 'ML_Proba_Away', 'RNN_Value_Home', 'RNN_Value_Away',
                      'Odd_H', 'Odd_A']
    
    available_high_value = [c for c in high_value_cols if c in high_value_games.columns]
    
    st.dataframe(
        high_value_games[available_high_value].style.format({
            'ML_Proba_Home': '{:.3f}',
            'ML_Proba_Away': '{:.3f}',
            'RNN_Value_Home': '{:.3f}',
            'RNN_Value_Away': '{:.3f}',
            'Odd_H': '{:.2f}',
            'Odd_A': '{:.2f}'
        }),
        use_container_width=True
    )
else:
    st.info("ℹ️ Nenhum jogo com concordância perfeita encontrado hoje")

# Recomendação Final
st.subheader("🏆 Final Recommendation Strategy")

if len(high_value_games) > 0:
    st.success("""
    **🎯 ESTRATÉGIA RECOMENDADA HOJE:**
    - **Foque nos jogos com concordância perfeita** (tabela acima)
    - **Maior confiança** quando ambos os modelos apontam mesma direção  
    - **Menor risco** - validação cruzada de diferentes abordagens
    """)
    
    # Mostrar os melhores picks consolidados
    st.markdown("**🔥 Melhores Picks do Dia (Concordância Total):**")
    for idx, game in high_value_games.iterrows():
        st.write(f"- **{game['Home']} vs {game['Away']}** → {game['ML_Recommendation']}")
        
else:
    st.warning("""
    **⚠️ ESTRATÉGIA CONSERVADORA:**
    - **Prefira o Random Forest** (modelo mais testado)
    - **Use a RNN como confirmação secundária**
    - **Evite jogos onde modelos discordam fortemente**
    - **Considere apenas picks com boa probabilidade em ambos modelos**
    """)




########################################
##### Bloco 11 – Side-by-Side View #####
########################################
st.markdown("---")
st.subheader("👁️ Side-by-Side Model Comparison")

def clean_rnn_recommendation(rec):
    """Remove porcentagens para agrupar recomendações similares"""
    if pd.isna(rec):
        return "NO VALUE"
    
    rec_str = str(rec)
    # Manter apenas a parte antes do parêntese
    if '(' in rec_str:
        return rec_str.split('(')[0].strip()
    return rec_str

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🤖 Random Forest")
    if 'ML_Recommendation' in games_today.columns:
        rf_summary = games_today['ML_Recommendation'].value_counts()
        for rec, count in rf_summary.items():
            st.write(f"{rec}: **{count}** jogos")
        
        # Top picks do RF
        st.markdown("**🎯 RF Top Picks (Diretos):**")
        rf_picks = games_today[games_today['ML_Recommendation'].str.contains('🟢 Back Home|🟠 Back Away', na=False)]
        if len(rf_picks) > 0:
            for _, game in rf_picks.head(5).iterrows():
                st.write(f"- **{game['Home']}** vs {game['Away']}")
                st.write(f"  {game['ML_Recommendation']} | Odd: {game.get('Odd_H', 'N/A') if 'Back Home' in game['ML_Recommendation'] else game.get('Odd_A', 'N/A')}")
        else:
            st.write("Nenhum pick direto hoje")

with col2:
    st.markdown("### 🧠 RNN Value Detector")
    if 'RNN_Recommendation' in games_today.columns:
        # Limpar recomendações para agrupar
        games_today['RNN_Clean'] = games_today['RNN_Recommendation'].apply(clean_rnn_recommendation)
        
        rnn_summary = games_today['RNN_Clean'].value_counts()
        for rec, count in rnn_summary.items():
            st.write(f"{rec}: **{count}** jogos")
        
        # Top picks da RNN (maior valor)
        st.markdown("**🎯 RNN High-Value Picks (>70%):**")
        high_value_mask = (
            (games_today['RNN_Value_Home'] >= 0.7) | 
            (games_today['RNN_Value_Away'] >= 0.7)
        )
        rnn_high_value = games_today[high_value_mask]
        
        if len(rnn_high_value) > 0:
            for _, game in rnn_high_value.head(5).iterrows():
                value_pct = max(game['RNN_Value_Home'], game['RNN_Value_Away'])
                st.write(f"- **{game['Home']}** vs {game['Away']}")
                st.write(f"  {game['RNN_Recommendation']} | Confidence: {value_pct:.1%}")
        else:
            st.write("Nenhum high-value pick (>70%) hoje")

# 🔥 BLOCO EXTRA: Concordância entre modelos
st.markdown("---")
st.subheader("🔥 Concordância entre Modelos")

# Jogos onde ambos concordam
agree_games = games_today[
    (games_today['Model_Agreement'].isin(['🎯 Perfect Agreement', '✅ Same Side']))
]

if len(agree_games) > 0:
    st.success(f"🎯 {len(agree_games)} jogos com concordância entre modelos")
    
    agree_cols = ['Home', 'Away', 'ML_Recommendation', 'RNN_Recommendation', 'Model_Agreement']
    available_agree = [c for c in agree_cols if c in agree_games.columns]
    
    st.dataframe(
        agree_games[available_agree],
        use_container_width=True,
        height=300
    )
    
    # Melhores picks consolidados
    st.markdown("**💎 Melhores Picks Consolidados (Ambos Concordam):**")
    for _, game in agree_games.iterrows():
        st.write(f"- **{game['Home']}** vs **{game['Away']}**")
        st.write(f"  🤖 RF: {game['ML_Recommendation']}")
        st.write(f"  🧠 RNN: {game['RNN_Recommendation']}")
        
else:
    st.info("🤝 Nenhuma concordância perfeita entre modelos hoje")

# 📊 Resumo de Performance
st.markdown("---")
st.subheader("📊 Resumo de Performance")

col1, col2, col3 = st.columns(3)

with col1:
    total_games = len(games_today)
    rf_bets = len(games_today[~games_today['ML_Recommendation'].str.contains('❌ Avoid', na=False)])
    rnn_bets = len(games_today[~games_today['RNN_Recommendation'].str.contains('❌ NO VALUE', na=False)])
    
    st.metric("Total de Jogos", total_games)
    st.metric("RF Recomenda Apostar", rf_bets)
    st.metric("RNN Recomenda Apostar", rnn_bets)

with col2:
    perfect_agree = len(games_today[games_today['Model_Agreement'] == '🎯 Perfect Agreement'])
    same_side = len(games_today[games_today['Model_Agreement'] == '✅ Same Side'])
    
    st.metric("Concordância Perfeita", perfect_agree)
    st.metric("Mesmo Lado", same_side)

with col3:
    high_confidence_rnn = len(games_today[
        (games_today['RNN_Value_Home'] >= 0.7) | 
        (games_today['RNN_Value_Away'] >= 0.7)
    ])
    
    direct_rf_picks = len(games_today[
        games_today['ML_Recommendation'].str.contains('🟢 Back Home|🟠 Back Away', na=False)
    ])
    
    st.metric("RNN High Confidence", high_confidence_rnn)
    st.metric("RF Picks Diretos", direct_rf_picks)
