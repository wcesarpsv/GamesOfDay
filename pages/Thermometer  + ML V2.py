########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(page_title="Today's Picks - Momentum Thermometer + ML", layout="wide")
st.title("📊 Momentum Thermometer + ML Prototype")

# Configurações principais
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "afc","trophy"]

M_DIFF_MARGIN = 0.30
POWER_MARGIN = 10
DOMINANT_THRESHOLD = 0.90


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


########################################
####### Bloco 4 – Load Data ############
########################################

files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# Últimos dois arquivos (Hoje e Ontem)
options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Carregar os jogos do dia selecionado
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Apenas jogos sem placar final
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# Carregar todos os arquivos para formar o histórico
all_games = load_all_games(GAMES_FOLDER)
all_games = filter_leagues(all_games)

# Preparar histórico (somente jogos finalizados e com as colunas obrigatórias)
history = prepare_history(all_games)

if history.empty:
    st.error("No valid historical data found. Check if the CSV files have all required columns.")
    st.stop()

# Extrair a data do arquivo selecionado (YYYY-MM-DD)
import re
date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
if date_match:
    selected_date_str = date_match.group(0)
else:
    selected_date_str = datetime.now().strftime("%Y-%m-%d")


########################################
####### Bloco 4B – LiveScore Merge #####
########################################
livescore_folder = "LiveScore"
livescore_file = os.path.join(livescore_folder, f"Resultados_RAW_{selected_date_str}.csv")

# Ensure goal columns exist
if 'Goals_H_Today' not in games_today.columns:
    games_today['Goals_H_Today'] = np.nan
if 'Goals_A_Today' not in games_today.columns:
    games_today['Goals_A_Today'] = np.nan

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
        st.error(f"The file {livescore_file} is missing these columns: {missing_cols}")
    else:
        games_today = games_today.merge(
            results_df,
            left_on='Id',
            right_on='game_id',
            how='left',
            suffixes=('', '_RAW')
        )

        # Update goals only for finished games
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

        # ADD RED CARD COLUMNS
        games_today['Home_Red'] = games_today['home_red']
        games_today['Away_Red'] = games_today['away_red']
else:
    st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")



########################################
####### Bloco 5 – Features Extras ######
########################################
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
history['M_Diff'] = history['M_H'] - history['M_A']

def compute_double_chance_odds(df):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df['Odd_H']
    probs['p_D'] = 1 / df['Odd_D']
    probs['p_A'] = 1 / df['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)
    df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

games_today = compute_double_chance_odds(games_today)


########################################
####### Bloco 5B – Win Prob Helper #####
########################################
def event_side_for_winprob(auto_rec):
    if pd.isna(auto_rec): return None
    s = str(auto_rec)
    if 'Back Home' in s: return 'HOME'
    if 'Back Away' in s: return 'AWAY'
    if 'Back Draw' in s: return 'DRAW'
    if '1X' in s: return '1X'
    if 'X2' in s: return 'X2'
    return None


########################################
####### Bloco 5C – Bands & Dominant ####
########################################
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

def dominant_side(row, threshold=DOMINANT_THRESHOLD):
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

# Merge com classificações
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


########################################
####### Bloco 6 – Auto Recommendation ##
########################################
def auto_recommendation(row,
                        diff_mid_lo=0.20, diff_mid_hi=0.80,
                        diff_mid_hi_highvar=0.75, power_gate=1, power_gate_highvar=5):

    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    league_cls= row.get('League_Classification', 'Medium Variation')
    m_a       = row.get('M_A')
    m_h       = row.get('M_H')
    odd_d     = row.get('Odd_D')

    # 1) Strong edges -> Direct Back
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '🟢 Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '🟠 Back Away'

    if dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if diff_m is not None and diff_m >= 0.90:
            return '🟢 Back Home'
    if dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home == 'Balanced':
        if diff_m is not None and diff_m <= -0.90:
            return '🟪 X2 (Away/Draw)'

    # 2) Both Balanced (with thresholds)
    if (band_home == 'Balanced') and (band_away == 'Balanced') and (diff_m is not None) and (diff_pow is not None):
        if league_cls == 'High Variation':
            if (diff_m >= 0.45 and diff_m < diff_mid_hi_highvar and diff_pow >= power_gate_highvar):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -0.45 and diff_m > -diff_mid_hi_highvar and diff_pow <= -power_gate_highvar):
                return '🟪 X2 (Away/Draw)'
        else:
            if (diff_m >= diff_mid_lo and diff_m < diff_mid_hi and diff_pow >= power_gate):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -diff_mid_lo and diff_m > -diff_mid_hi and diff_pow <= -power_gate):
                return '🟪 X2 (Away/Draw)'

    # 3) Balanced vs Bottom20%
    if (band_home == 'Balanced') and (band_away == 'Bottom 20%'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Balanced') and (band_home == 'Bottom 20%'):
        return '🟪 X2 (Away/Draw)'

    # 4) Top20% vs Balanced
    if (band_home == 'Top 20%') and (band_away == 'Balanced'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Top 20%') and (band_home == 'Balanced'):
        return '🟪 X2 (Away/Draw)'

    # 5) Filtro Draw (novo)
    if (odd_d is not None and 2.5 <= odd_d <= 6.0) and (diff_pow is not None and -10 <= diff_pow <= 10):
        if (m_h is not None and 0 <= m_h <= 1) or (m_a is not None and 0 <= m_a <= 0.5):
            return '⚪ Back Draw'

    # 6) Fallback
    return '❌ Avoid'

# Aplicar recomendação
games_today['Auto_Recommendation'] = games_today.apply(lambda r: auto_recommendation(r), axis=1)


########################################
####### Bloco 7 – Train ML Model #######
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

# 🆕 NOVO: Codificar as recomendações do Auto Recommendation
def encode_auto_recommendation(auto_rec):
    if '🟢 Back Home' in auto_rec:
        return 4  # Forte Home
    elif '🟠 Back Away' in auto_rec:
        return 3  # Forte Away  
    elif '🟦 1X' in auto_rec:
        return 2  # Home/Draw
    elif '🟪 X2' in auto_rec:
        return 1  # Away/Draw
    elif '⚪ Back Draw' in auto_rec:
        return 5  # Draw
    else:
        return -1 # Avoid/Neutro

# 🆕 APLICAR AUTO RECOMMENDATION AO HISTÓRICO (simular o que teria sido recomendado)
print("🔄 Calculando Auto Recommendation para dados históricos...")
history['Auto_Rec_Simulated'] = history.apply(lambda r: auto_recommendation(r), axis=1)
history['Auto_Rec_Encoded'] = history['Auto_Rec_Simulated'].apply(encode_auto_recommendation)

# 🆕 FEATURES ATUALIZADAS - INCLUINDO AUTO RULES
features_raw = [
    
'HandScore_Home_HT', 'HandScore_Away_HT',
'Aggression_Home', 'Aggression_Away',
'Diff_HT_P',
'M_Diff', 'Diff_Power',
'League_Classification', 'Games_Analyzed',
'Auto_Rec_Encoded'
]

# Manter apenas features que existem no histórico
features_raw = [f for f in features_raw if f in history.columns]

print(f"✅ Features para ML: {features_raw}")

X = history[features_raw].copy()
y = history['Result']

# Mapeamento de bands para numérico
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: 
    X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: 
    X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

# Codificar variáveis categóricas
cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

# 🎯 TREINAR MODELO COM NOVAS FEATURES
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

print("🤖 Treinando Random Forest com Auto Rules features...")
model.fit(X, y)
print("✅ Modelo treinado com sucesso!")


########################################
####### Bloco 7B – ML Recommendation Function #######
########################################

def ml_recommendation_from_proba(p_home, p_draw, p_away, threshold=0.65):
    """
    Converte probabilidades do ML em recomendações de apostas
    """
    if p_home >= threshold:
        return "🟢 Back Home"
    elif p_away >= threshold:
        return "🟠 Back Away"
    else:
        sum_home_draw = p_home + p_draw + 0.2 
        sum_away_draw = p_away + p_draw
        if abs(p_home - p_away) < 0.05 and p_draw > 0.50:
            return "⚪ Back Draw"
        elif sum_home_draw > sum_away_draw:
            return "🟦 1X (Home/Draw)"
        elif sum_away_draw > sum_home_draw :
            return "🟪 X2 (Away/Draw)"
        else:
            return "❌ Avoid"
            



########################################
####### Bloco 8 – Apply ML to Today ####
########################################
threshold = st.sidebar.slider(
    "ML Threshold for Direct Win (%)", 
    min_value=50, max_value=80, value=65, step=1
) / 100.0

# 🆕 CALCULAR AUTO RECOMMENDATION PRIMEIRO (para ter a feature)
print("🔄 Calculando Auto Recommendation para jogos de hoje...")
games_today['Auto_Recommendation'] = games_today.apply(lambda r: auto_recommendation(r), axis=1)
games_today['Auto_Rec_Encoded'] = games_today['Auto_Recommendation'].apply(encode_auto_recommendation)

# 🆕 PREPARAR FEATURES PARA HOJE (INCLUINDO AUTO_REC_ENCODED)
X_today = games_today[features_raw].copy()

if 'Home_Band' in X_today: 
    X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: 
    X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

# Aplicar mesmo encoder das categorias
if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                         encoded_today_df.reset_index(drop=True)], axis=1)

# 🧩 Verificar linhas com dados completos
valid_mask = X_today.notna().all(axis=1)
valid_rows = X_today[valid_mask]
invalid_rows = X_today[~valid_mask]

# Inicializar colunas padrão
games_today["ML_Proba_Home"] = np.nan
games_today["ML_Proba_Draw"] = np.nan
games_today["ML_Proba_Away"] = np.nan
games_today["ML_Recommendation"] = "❌ Avoid"

# 🎯 Aplicar modelo SOMENTE nas linhas completas
if not valid_rows.empty:
    print(f"🤖 Aplicando ML em {len(valid_rows)} jogos com dados completos...")
    ml_preds = model.predict(valid_rows)
    ml_proba = model.predict_proba(valid_rows)

    # Mapear probabilidades de volta ao índice original
    games_today.loc[valid_mask, "ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
    games_today.loc[valid_mask, "ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
    games_today.loc[valid_mask, "ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]

    games_today.loc[valid_mask, "ML_Recommendation"] = [
        ml_recommendation_from_proba(row["ML_Proba_Home"], 
                                     row["ML_Proba_Draw"], 
                                     row["ML_Proba_Away"],
                                     threshold=threshold)
        for _, row in games_today.loc[valid_mask].iterrows()
    ]

# ⚠️ Jogos com features ausentes
if not invalid_rows.empty:
    st.warning(f"{len(invalid_rows)} jogos marcados como ❌ Avoid por falta de dados completos para ML.")

print("✅ ML Recommendations atualizadas com verificação de dados completos!")






########################################
##### Bloco 8B – Avaliar Resultados ####
########################################
def determine_result(row):
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

games_today['Result_Today'] = games_today.apply(determine_result, axis=1)

def check_recommendation(rec, result):
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return None
    rec = str(rec)
    if 'Back Home' in rec:
        return result == "Home"
    elif 'Back Away' in rec:
        return result == "Away"
    elif 'Back Draw' in rec:
        return result == "Draw"
    elif '1X' in rec:
        return result in ["Home", "Draw"]
    elif 'X2' in rec:
        return result in ["Away", "Draw"]
    return None

games_today['Auto_Correct'] = games_today.apply(lambda r: check_recommendation(r['Auto_Recommendation'], r['Result_Today']), axis=1)
games_today['ML_Correct'] = games_today.apply(lambda r: check_recommendation(r['ML_Recommendation'], r['Result_Today']), axis=1)

def calculate_profit(rec, result, odds_row):
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return 0
    rec = str(rec)
    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        return odd - 1 if result == "Home" else -1
    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        return odd - 1 if result == "Away" else -1
    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        return odd - 1 if result == "Draw" else -1
    elif '1X' in rec:
        odd = odds_row.get('Odd_1X', np.nan)
        return odd - 1 if result in ["Home", "Draw"] else -1
    elif 'X2' in rec:
        odd = odds_row.get('Odd_X2', np.nan)
        return odd - 1 if result in ["Away", "Draw"] else -1
    return 0


# Calcular profit separadamente
games_today['Profit_Auto'] = games_today.apply(
    lambda r: calculate_profit(r['Auto_Recommendation'], r['Result_Today'], r), axis=1
)
games_today['Profit_ML'] = games_today.apply(
    lambda r: calculate_profit(r['ML_Recommendation'], r['Result_Today'], r), axis=1
)


########################################
##### Bloco 8C – Resumo Agregado #######
########################################
finished_games = games_today.dropna(subset=['Result_Today'])

def kelly_stake(probability, odds, bankroll=1000, kelly_fraction=0.25, min_stake=1, max_stake=100):
    """
    Calculate Kelly Criterion stake size with practical limits
    """
    if pd.isna(probability) or pd.isna(odds) or odds <= 1 or probability <= 0:
        return 0

    # Calculate edge and recommended stake fraction
    edge = probability * odds - 1
    if edge <= 0:
        return 0

    # Full Kelly fraction
    full_kelly_fraction = edge / (odds - 1)

    # Apply fractional Kelly and convert to absolute stake
    fractional_kelly = full_kelly_fraction * kelly_fraction
    recommended_stake = fractional_kelly * bankroll

    # Apply practical limits
    if recommended_stake < min_stake:
        return 0  # Don't bet if below minimum
    elif recommended_stake > max_stake:
        return max_stake
    else:
        return round(recommended_stake, 2)

def calculate_profit_with_kelly(rec, result, odds_row, ml_probabilities, bankroll=1000, kelly_fraction=0.25, min_stake=1, max_stake=100):
    """
    Calculate profit using Kelly Criterion stake sizing with practical limits
    """
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return 0, 0

    rec = str(rec)
    stake_fixed = 1  # Your original fixed stake

    # Determine bet type and get relevant probability
    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        prob = ml_probabilities.get('Home', 0.5)
        stake_kelly = kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result == "Home" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Home" else -stake_kelly

    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        prob = ml_probabilities.get('Away', 0.5)
        stake_kelly = kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result == "Away" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Away" else -stake_kelly

    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        prob = ml_probabilities.get('Draw', 0.5)
        stake_kelly = kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result == "Draw" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Draw" else -stake_kelly

    elif '1X' in rec:
        odd = odds_row.get('Odd_1X', np.nan)
        prob = ml_probabilities.get('Home', 0) + ml_probabilities.get('Draw', 0)
        stake_kelly = kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result in ["Home", "Draw"] else -1
        profit_kelly = (odd - 1) * stake_kelly if result in ["Home", "Draw"] else -stake_kelly

    elif 'X2' in rec:
        odd = odds_row.get('Odd_X2', np.nan)
        prob = ml_probabilities.get('Away', 0) + ml_probabilities.get('Draw', 0)
        stake_kelly = kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result in ["Away", "Draw"] else -1
        profit_kelly = (odd - 1) * stake_kelly if result in ["Away", "Draw"] else -stake_kelly

    else:
        return 0, 0

    return profit_fixed, profit_kelly

# Add Kelly parameters in sidebar
st.sidebar.subheader("Kelly Criterion Parameters")

bankroll = st.sidebar.number_input(
    "Bankroll Size", 
    min_value=100, max_value=10000, value=1000, step=100,
    help="Total bankroll for Kelly stake calculation"
)

kelly_fraction = st.sidebar.slider(
    "Kelly Fraction", 
    min_value=0.1, max_value=1.0, value=0.25, step=0.05,
    help="Fraction of full Kelly stake to use (lower = more conservative)"
)

min_stake = st.sidebar.number_input(
    "Minimum Stake", 
    min_value=1, max_value=50, value=1, step=1,
    help="Minimum stake amount per bet"
)

max_stake = st.sidebar.number_input(
    "Maximum Stake", 
    min_value=10, max_value=500, value=100, step=10,
    help="Maximum stake amount per bet"
)

# Calculate profits for both methods
games_today['Profit_Auto_Fixed'] = games_today.apply(
    lambda r: calculate_profit(r['Auto_Recommendation'], r['Result_Today'], r), axis=1
)
games_today['Profit_ML_Fixed'] = games_today.apply(
    lambda r: calculate_profit(r['ML_Recommendation'], r['Result_Today'], r), axis=1
)

# Calculate Kelly profits
games_today[['Profit_Auto_Fixed', 'Profit_Auto_Kelly']] = games_today.apply(
    lambda r: calculate_profit_with_kelly(
        r['Auto_Recommendation'], 
        r['Result_Today'], 
        r,
        {'Home': r.get('ML_Proba_Home', 0.5), 
         'Draw': r.get('ML_Proba_Draw', 0.5), 
         'Away': r.get('ML_Proba_Away', 0.5)},
        bankroll, kelly_fraction, min_stake, max_stake
    ), 
    axis=1, result_type='expand'
)

games_today[['Profit_ML_Fixed', 'Profit_ML_Kelly']] = games_today.apply(
    lambda r: calculate_profit_with_kelly(
        r['ML_Recommendation'], 
        r['Result_Today'], 
        r,
        {'Home': r.get('ML_Proba_Home', 0.5), 
         'Draw': r.get('ML_Proba_Draw', 0.5), 
         'Away': r.get('ML_Proba_Away', 0.5)},
        bankroll, kelly_fraction, min_stake, max_stake
    ), 
    axis=1, result_type='expand'
)

# Add Kelly stake columns for transparency
def get_kelly_stake_only(rec, odds_row, ml_probabilities, bankroll=1000, kelly_fraction=0.25, min_stake=1, max_stake=100):
    """Get only the Kelly stake amount for display"""
    if pd.isna(rec) or rec == '❌ Avoid':
        return 0

    rec = str(rec)

    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        prob = ml_probabilities.get('Home', 0.5)
        return kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)

    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        prob = ml_probabilities.get('Away', 0.5)
        return kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)

    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        prob = ml_probabilities.get('Draw', 0.5)
        return kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)

    elif '1X' in rec:
        odd = odds_row.get('Odd_1X', np.nan)
        prob = ml_probabilities.get('Home', 0) + ml_probabilities.get('Draw', 0)
        return kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)

    elif 'X2' in rec:
        odd = odds_row.get('Odd_X2', np.nan)
        prob = ml_probabilities.get('Away', 0) + ml_probabilities.get('Draw', 0)
        return kelly_stake(prob, odd, bankroll, kelly_fraction, min_stake, max_stake)

    return 0

games_today['Kelly_Stake_Auto'] = games_today.apply(
    lambda r: get_kelly_stake_only(
        r['Auto_Recommendation'], 
        r,
        {'Home': r.get('ML_Proba_Home', 0.5), 
         'Draw': r.get('ML_Proba_Draw', 0.5), 
         'Away': r.get('ML_Proba_Away', 0.5)},
        bankroll, kelly_fraction, min_stake, max_stake
    ), 
    axis=1
)

games_today['Kelly_Stake_ML'] = games_today.apply(
    lambda r: get_kelly_stake_only(
        r['ML_Recommendation'], 
        r,
        {'Home': r.get('ML_Proba_Home', 0.5), 
         'Draw': r.get('ML_Proba_Draw', 0.5), 
         'Away': r.get('ML_Proba_Away', 0.5)},
        bankroll, kelly_fraction, min_stake, max_stake
    ), 
    axis=1
)

########################################
##### Bloco 8D – Resumo Agregado Expandido ###
########################################
finished_games = games_today.dropna(subset=['Result_Today'])

def summary_stats_comprehensive(df, prefix):
    bets = df[df[f'{prefix}_Correct'].notna()]
    total_bets = len(bets)
    correct_bets = bets[f'{prefix}_Correct'].sum()
    winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0

    # Fixed stake profits
    total_profit_fixed = bets[f'Profit_{prefix}_Fixed'].sum()
    roi_fixed = (total_profit_fixed / total_bets) * 100 if total_bets > 0 else 0

    # Kelly stake profits
    total_profit_kelly = bets[f'Profit_{prefix}_Kelly'].sum()
    total_stake_kelly = bets[f'Kelly_Stake_{prefix}'].sum()
    roi_kelly = (total_profit_kelly / total_stake_kelly) * 100 if total_stake_kelly > 0 else 0

    # Average stake sizes
    avg_stake_kelly = bets[f'Kelly_Stake_{prefix}'].mean() if total_bets > 0 else 0

    # Risk metrics for Kelly
    kelly_bets = bets[bets[f'Kelly_Stake_{prefix}'] > 0]
    avg_edge = "N/A"
    if len(kelly_bets) > 0:
        # Calculate average edge for Kelly bets
        edges = []
        for _, bet in kelly_bets.iterrows():
            rec = bet[f'{prefix}_Recommendation']
            if 'Back Home' in rec:
                prob = bet.get('ML_Proba_Home', 0.5)
                odds = bet.get('Odd_H', 1)
            elif 'Back Away' in rec:
                prob = bet.get('ML_Proba_Away', 0.5)
                odds = bet.get('Odd_A', 1)
            elif 'Back Draw' in rec:
                prob = bet.get('ML_Proba_Draw', 0.5)
                odds = bet.get('Odd_D', 1)
            elif '1X' in rec:
                prob = bet.get('ML_Proba_Home', 0) + bet.get('ML_Proba_Draw', 0)
                odds = bet.get('Odd_1X', 1)
            elif 'X2' in rec:
                prob = bet.get('ML_Proba_Away', 0) + bet.get('ML_Proba_Draw', 0)
                odds = bet.get('Odd_X2', 1)
            else:
                continue
            edges.append(prob * odds - 1)
        avg_edge = round(np.mean(edges) * 100, 2) if edges else "N/A"

    return {
        "Total Jogos": len(df),
        "Apostas Feitas": total_bets,
        "Acertos": int(correct_bets),
        "Winrate (%)": round(winrate, 2),
        "Profit Fixed (Stake=1)": round(total_profit_fixed, 2),
        "ROI Fixed (%)": round(roi_fixed, 2),
        "Profit Kelly": round(total_profit_kelly, 2),
        "Total Stake Kelly": round(total_stake_kelly, 2),
        "ROI Kelly (%)": round(roi_kelly, 2),
        "Avg Kelly Stake": round(avg_stake_kelly, 2),
        "Kelly Bets Made": len(kelly_bets),
        "Avg Edge (%)": avg_edge
    }

summary_auto_comprehensive = summary_stats_comprehensive(finished_games, "Auto")
summary_ml_comprehensive = summary_stats_comprehensive(finished_games, "ML")


########################################
##### Bloco Extra – ML Detailed Performance by Recommendation Type #####
########################################
st.markdown("### 🔍 ML Performance Breakdown by Recommendation Type")

# Filtrar apenas jogos com recomendação ML válida e resultado conhecido
ml_eval = finished_games.copy()
ml_eval = ml_eval[ml_eval['ML_Recommendation'].notna() & (ml_eval['ML_Recommendation'] != '❌ Avoid')]

# Função auxiliar para agrupar por tipo
def summarize_by_recommendation(df):
    summary = []
    for rec_type, group in df.groupby('ML_Recommendation'):
        total = len(group)
        correct = group['ML_Correct'].sum()
        winrate = (correct / total) * 100 if total > 0 else 0
        profit = group['Profit_ML_Fixed'].sum()
        roi = (profit / total) * 100 if total > 0 else 0
        avg_odd = group[['Odd_H', 'Odd_D', 'Odd_A']].mean(numeric_only=True).mean()
        summary.append({
            "Recommendation": rec_type,
            "Bets": total,
            "Correct": int(correct),
            "Winrate (%)": round(winrate, 2),
            "Total Profit": round(profit, 2),
            "ROI (%)": round(roi, 2),
            "Avg Odd": round(avg_odd, 2)
        })
    return pd.DataFrame(summary).sort_values("ROI (%)", ascending=False)

ml_summary_table = summarize_by_recommendation(ml_eval)

# Exibir no Streamlit
if not ml_summary_table.empty:
    st.dataframe(
        ml_summary_table.style.format({
            "Winrate (%)": "{:.2f}",
            "Total Profit": "{:+.2f}",
            "ROI (%)": "{:+.2f}",
            "Avg Odd": "{:.2f}"
        }),
        use_container_width=True,
        height=500
    )

    # Estatísticas gerais
    best_type = ml_summary_table.iloc[0]
    st.success(f"🏁 Melhor desempenho: **{best_type['Recommendation']}** – ROI {best_type['ROI (%)']:+.2f}% com Winrate de {best_type['Winrate (%)']:.2f}% ({best_type['Bets']} apostas).")
else:
    st.warning("Nenhum jogo com recomendações válidas da ML foi encontrado para análise.")




st.subheader("📈 Day's Summary - Fixed vs Kelly Staking")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance Auto Recommendation")
    st.json(summary_auto_comprehensive)

with col2:
    st.markdown("### Performance Machine Learning")
    st.json(summary_ml_comprehensive)

# Display Kelly parameters being used
st.info(f"""
**Kelly Parameters:** Bankroll = ${bankroll:,} | Kelly Fraction = {kelly_fraction} | Min Stake = ${min_stake} | Max Stake = ${max_stake}
""")






########################################
##### Bloco 9 – Exibição Final Expandida ###
########################################
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Goals_H_Today', 'Goals_A_Today',
    'Auto_Recommendation', 'ML_Recommendation',
    'Home_Red', 'Away_Red',
    'Auto_Correct', 'ML_Correct',
    'Profit_Auto_Fixed', 'Profit_Auto_Kelly', 'Kelly_Stake_Auto',
    'Profit_ML_Fixed', 'Profit_ML_Kelly', 'Kelly_Stake_ML',
    'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away',
    'Odd_H', 'Odd_D', 'Odd_A'  # Added odds for transparency
]

available_cols = [c for c in cols_to_show if c in games_today.columns]

st.subheader("📊 Games – Rules vs ML (Fixed vs Kelly Staking)")
st.dataframe(
    games_today[available_cols]
    .style.format({
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'Home_Red': '{:.0f}',
        'Away_Red': '{:.0f}',
        'Profit_Auto_Fixed': '{:.2f}',
        'Profit_Auto_Kelly': '{:.2f}',
        'Profit_ML_Fixed': '{:.2f}',
        'Profit_ML_Kelly': '{:.2f}',
        'Kelly_Stake_Auto': '{:.2f}',
        'Kelly_Stake_ML': '{:.2f}',
        'ML_Proba_Home': '{:.3f}',
        'ML_Proba_Draw': '{:.3f}',
        'ML_Proba_Away': '{:.3f}',
        'Odd_H': '{:.2f}',
        'Odd_D': '{:.2f}', 
        'Odd_A': '{:.2f}'
    }),
    use_container_width='stretch',
    height=1200,
)



########################################
##### BLOCO 10 – MARKET ERROR INTELLIGENCE (MEI LAYER) #####
########################################
st.markdown("### 💡 Market Error Intelligence (Value Detector)")

# Calcular probabilidades implícitas das odds de fechamento
if all(col in games_today.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / games_today['Odd_H']
    probs['p_D'] = 1 / games_today['Odd_D']
    probs['p_A'] = 1 / games_today['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)

    games_today['Imp_Prob_H'] = probs['p_H']
    games_today['Imp_Prob_D'] = probs['p_D']
    games_today['Imp_Prob_A'] = probs['p_A']

    # Calcular o erro de mercado (quanto o modelo discorda das odds)
    games_today['Market_Error_Home'] = games_today['ML_Proba_Home'] - games_today['Imp_Prob_H']
    games_today['Market_Error_Away'] = games_today['ML_Proba_Away'] - games_today['Imp_Prob_A']
    games_today['Market_Error_Draw'] = games_today['ML_Proba_Draw'] - games_today['Imp_Prob_D']

    # Classificar o lado de valor
    def classify_value_pick(row, min_gap=0.05):
        me_home = row['Market_Error_Home']
        me_away = row['Market_Error_Away']
        if (me_home > min_gap) and (me_home > me_away):
            return "🟢 Value on Home"
        elif (me_away > min_gap) and (me_away > me_home):
            return "🟠 Value on Away"
        elif abs(me_home - me_away) <= 0.03 and max(me_home, me_away) > min_gap:
            return "⚪ Balanced Value (Both sides close)"
        return "❌ No clear value"

    games_today['Value_Pick'] = games_today.apply(classify_value_pick, axis=1)

    # Exibir ranking dos maiores gaps
    value_cols = [
        'League', 'Home', 'Away', 
        'Odd_H', 'Odd_D', 'Odd_A',
        'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away',
        'Imp_Prob_H', 'Imp_Prob_D', 'Imp_Prob_A',
        'Market_Error_Home', 'Market_Error_Away', 'Value_Pick'
    ]

    st.dataframe(
        games_today[value_cols]
        .sort_values(['Market_Error_Home','Market_Error_Away'], ascending=False)
        .style.format({
            'Odd_H':'{:.2f}', 'Odd_D':'{:.2f}', 'Odd_A':'{:.2f}',
            'ML_Proba_Home':'{:.3f}', 'ML_Proba_Draw':'{:.3f}', 'ML_Proba_Away':'{:.3f}',
            'Imp_Prob_H':'{:.3f}', 'Imp_Prob_D':'{:.3f}', 'Imp_Prob_A':'{:.3f}',
            'Market_Error_Home':'{:+.3f}', 'Market_Error_Away':'{:+.3f}'
        }),
        use_container_width=True,
        height=800
    )

    # Estatísticas rápidas de valor
    avg_me_home = games_today['Market_Error_Home'].mean()
    avg_me_away = games_today['Market_Error_Away'].mean()
    pct_value_home = (games_today['Market_Error_Home'] > 0.05).mean() * 100
    pct_value_away = (games_today['Market_Error_Away'] > 0.05).mean() * 100

    st.info(f"""
    **📊 Diagnóstico de Valor de Mercado**
    - Média Market_Error_Home: {avg_me_home:+.3f}
    - Média Market_Error_Away: {avg_me_away:+.3f}
    - % de jogos com valor em Home: {pct_value_home:.1f}%
    - % de jogos com valor em Away: {pct_value_away:.1f}%
    """)

else:
    st.warning("Odds ausentes — impossível calcular Market Error Intelligence.")




########################################
##### BLOCO 11 – MARKET ERROR ML (VALUE LEARNING) #####
########################################
st.markdown("### 🧠 Market Error ML – Aprendizado de Valor (Meta-Modelo)")

# Garantir que Market_Error_Home/Away estão disponíveis
if all(col in games_today.columns for col in ['Market_Error_Home', 'Market_Error_Away']):
    # Preparar dataset de treinamento com histórico (jogos finalizados)
    value_history = history.copy()

    required_cols = ['Odd_H', 'Odd_A', 'Odd_D', 'M_H', 'M_A', 'Diff_Power', 'M_Diff']
    available_cols = [c for c in required_cols if c in value_history.columns]

    # Calcular probabilidades implícitas e simular previsões históricas
    for col in ['Odd_H', 'Odd_D', 'Odd_A']:
        value_history[f'Imp_{col}'] = 1 / value_history[col]
    imp_sum = value_history[['Imp_Odd_H', 'Imp_Odd_D', 'Imp_Odd_A']].sum(axis=1)
    for col in ['Imp_Odd_H', 'Imp_Odd_D', 'Imp_Odd_A']:
        value_history[col] = value_history[col] / imp_sum

    # Mapear resultado
    def map_result(row):
        if row['Goals_H_FT'] > row['Goals_A_FT']:
            return "Home"
        elif row['Goals_H_FT'] < row['Goals_A_FT']:
            return "Away"
        return "Draw"

    value_history['Result'] = value_history.apply(map_result, axis=1)

    # Targets binários: se o lado "ganhou contra o mercado"
    value_history['Target_Value_Home'] = (
        (value_history['Result'] == "Home") &
        (1 / value_history['Odd_H'] > value_history['Imp_Odd_H'])
    ).astype(int)

    value_history['Target_Value_Away'] = (
        (value_history['Result'] == "Away") &
        (1 / value_history['Odd_A'] > value_history['Imp_Odd_A'])
    ).astype(int)

    # Features básicas
    features_value = [
        'M_H', 'M_A', 'Diff_Power', 'M_Diff',
        'Odd_H', 'Odd_D', 'Odd_A'
    ]
    X_val = value_history[features_value].fillna(0)
    y_val = value_history['Target_Value_Home']  # modelo exemplo para o lado Home

    from sklearn.ensemble import RandomForestClassifier
    value_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    value_model.fit(X_val, y_val)

    # Aplicar modelo de valor aos jogos do dia
    X_today_val = games_today[features_value].fillna(0)
    val_pred_home = value_model.predict_proba(X_today_val)[:, 1]
    games_today['Value_Prob_Home'] = val_pred_home

    # Fazer o mesmo para o lado Away
    y_val_away = value_history['Target_Value_Away']
    value_model_away = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=24,
        n_jobs=-1
    )
    value_model_away.fit(X_val, y_val_away)
    val_pred_away = value_model_away.predict_proba(X_today_val)[:, 1]
    games_today['Value_Prob_Away'] = val_pred_away

    # Escolher lado com maior confiança de valor
    def pick_value_side(row, min_threshold=0.55):
        v_home, v_away = row['Value_Prob_Home'], row['Value_Prob_Away']
        if v_home >= min_threshold and v_home > v_away:
            return f"🟢 Value ML: Back Home ({v_home:.2f})"
        elif v_away >= min_threshold and v_away > v_home:
            return f"🟠 Value ML: Back Away ({v_away:.2f})"
        else:
            return "❌ No Value Signal"

    games_today['Value_ML_Pick'] = games_today.apply(pick_value_side, axis=1)

    # Exibir tabela
    st.dataframe(
        games_today[['League', 'Home', 'Away',
                     'Odd_H', 'Odd_D', 'Odd_A',
                     'Market_Error_Home', 'Market_Error_Away',
                     'Value_Prob_Home', 'Value_Prob_Away', 'Value_ML_Pick']]
        .sort_values(['Value_Prob_Home','Value_Prob_Away'], ascending=False)
        .style.format({
            'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
            'Market_Error_Home': '{:+.3f}', 'Market_Error_Away': '{:+.3f}',
            'Value_Prob_Home': '{:.2f}', 'Value_Prob_Away': '{:.2f}'
        }),
        use_container_width=True,
        height=900
    )

    st.success("✅ Meta-Modelo de Valor treinado e aplicado com sucesso!")

else:
    st.warning("Market Error ainda não calculado — execute o Bloco 10 primeiro.")





