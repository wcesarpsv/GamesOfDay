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
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']:
        return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']:
        return "Away"
    else:
        return "Draw"

history['Result'] = history.apply(map_result, axis=1)

features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band','Dominant','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed'
]
features_raw = [f for f in features_raw if f in history.columns]

X = history[features_raw].copy()
y = history['Result']

# Map bands
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: 
    X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: 
    X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

# Encode categorical vars
cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

# =====================================
# Treino do modelo calibrado isotônico
# =====================================
classes = ["Home", "Draw", "Away"]
y_bin = label_binarize(y, classes=classes)

base_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

# Apply calibration directly to the multi-class model
calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="isotonic",
    cv=5,
    ensemble=True  # This uses proper cross-validation to prevent data leakage
)

calibrated_model.fit(X, y)  # Use the original multi-class target 'y'

# The model is now ready for predictions
# You can use calibrated_model.predict_proba(X) directly

########################################
####### Bloco 8 – Apply ML to Today ####
########################################
threshold = st.sidebar.slider(
    "ML Threshold for Direct Win (%)", 
    min_value=50, max_value=80, value=65, step=1
) / 100.0

def ml_recommendation_from_proba(p_home, p_draw, p_away, threshold=0.65):
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

X_today = games_today[features_raw].copy()

if 'Home_Band' in X_today: 
    X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: 
    X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                         encoded_today_df.reset_index(drop=True)], axis=1)

ml_preds = calibrated_model.predict(X_today)
ml_proba = calibrated_model.predict_proba(X_today)

games_today["ML_Proba_Home"] = ml_proba[:, list(calibrated_model.classes_).index("Home")]
games_today["ML_Proba_Draw"] = ml_proba[:, list(calibrated_model.classes_).index("Draw")]
games_today["ML_Proba_Away"] = ml_proba[:, list(calibrated_model.classes_).index("Away")]

games_today["ML_Recommendation"] = [
    ml_recommendation_from_proba(row["ML_Proba_Home"], 
                                 row["ML_Proba_Draw"], 
                                 row["ML_Proba_Away"],
                                 threshold=threshold)
    for _, row in games_today.iterrows()
]

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

def kelly_stake(probability, odds, kelly_fraction=0.25):
    """
    Calculate Kelly Criterion stake size
    kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
    """
    if pd.isna(probability) or pd.isna(odds) or odds <= 1:
        return 0
    
    # Calculate edge and recommended stake
    edge = probability * odds - 1
    if edge <= 0:
        return 0
    
    full_kelly = edge / (odds - 1)
    # Use fractional Kelly to reduce risk
    fractional_kelly = full_kelly * kelly_fraction
    # Cap at reasonable stake size
    return min(fractional_kelly, 0.5)  # Maximum 50% of bankroll

def calculate_profit_with_kelly(rec, result, odds_row, ml_probabilities, kelly_fraction=0.25):
    """
    Calculate profit using Kelly Criterion stake sizing
    """
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return 0, 0
    
    rec = str(rec)
    stake_fixed = 1  # Your original fixed stake
    
    # Determine bet type and get relevant probability
    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        prob = ml_probabilities.get('Home', 0.5)  # Fallback to 0.5 if no probability
        stake_kelly = kelly_stake(prob, odd, kelly_fraction)
        profit_fixed = odd - 1 if result == "Home" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Home" else -stake_kelly
        
    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        prob = ml_probabilities.get('Away', 0.5)
        stake_kelly = kelly_stake(prob, odd, kelly_fraction)
        profit_fixed = odd - 1 if result == "Away" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Away" else -stake_kelly
        
    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        prob = ml_probabilities.get('Draw', 0.5)
        stake_kelly = kelly_stake(prob, odd, kelly_fraction)
        profit_fixed = odd - 1 if result == "Draw" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Draw" else -stake_kelly
        
    elif '1X' in rec:
        odd = odds_row.get('Odd_1X', np.nan)
        prob = ml_probabilities.get('Home', 0) + ml_probabilities.get('Draw', 0)
        stake_kelly = kelly_stake(prob, odd, kelly_fraction)
        profit_fixed = odd - 1 if result in ["Home", "Draw"] else -1
        profit_kelly = (odd - 1) * stake_kelly if result in ["Home", "Draw"] else -stake_kelly
        
    elif 'X2' in rec:
        odd = odds_row.get('Odd_X2', np.nan)
        prob = ml_probabilities.get('Away', 0) + ml_probabilities.get('Draw', 0)
        stake_kelly = kelly_stake(prob, odd, kelly_fraction)
        profit_fixed = odd - 1 if result in ["Away", "Draw"] else -1
        profit_kelly = (odd - 1) * stake_kelly if result in ["Away", "Draw"] else -stake_kelly
        
    else:
        return 0, 0
    
    return profit_fixed, profit_kelly

# Add Kelly Fraction parameter
kelly_fraction = st.sidebar.slider(
    "Kelly Fraction", 
    min_value=0.1, max_value=1.0, value=0.25, step=0.05,
    help="Fraction of full Kelly stake to use (lower = more conservative)"
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
        kelly_fraction
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
        kelly_fraction
    ), 
    axis=1, result_type='expand'
)

# Add Kelly stake columns for transparency
def get_kelly_stake_only(rec, odds_row, ml_probabilities, kelly_fraction=0.25):
    """Get only the Kelly stake amount for display"""
    if pd.isna(rec) or rec == '❌ Avoid':
        return 0
    
    rec = str(rec)
    
    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        prob = ml_probabilities.get('Home', 0.5)
        return kelly_stake(prob, odd, kelly_fraction)
        
    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        prob = ml_probabilities.get('Away', 0.5)
        return kelly_stake(prob, odd, kelly_fraction)
        
    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        prob = ml_probabilities.get('Draw', 0.5)
        return kelly_stake(prob, odd, kelly_fraction)
        
    elif '1X' in rec:
        odd = odds_row.get('Odd_1X', np.nan)
        prob = ml_probabilities.get('Home', 0) + ml_probabilities.get('Draw', 0)
        return kelly_stake(prob, odd, kelly_fraction)
        
    elif 'X2' in rec:
        odd = odds_row.get('Odd_X2', np.nan)
        prob = ml_probabilities.get('Away', 0) + ml_probabilities.get('Draw', 0)
        return kelly_stake(prob, odd, kelly_fraction)
        
    return 0

games_today['Kelly_Stake_Auto'] = games_today.apply(
    lambda r: get_kelly_stake_only(
        r['Auto_Recommendation'], 
        r,
        {'Home': r.get('ML_Proba_Home', 0.5), 
         'Draw': r.get('ML_Proba_Draw', 0.5), 
         'Away': r.get('ML_Proba_Away', 0.5)},
        kelly_fraction
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
        kelly_fraction
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
        "Avg Kelly Stake": round(avg_stake_kelly, 3)
    }

summary_auto_comprehensive = summary_stats_comprehensive(finished_games, "Auto")
summary_ml_comprehensive = summary_stats_comprehensive(finished_games, "ML")

st.subheader("📈 Day's Summary - Fixed vs Kelly Staking")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance Auto Recommendation")
    st.json(summary_auto_comprehensive)

with col2:
    st.markdown("### Performance Machine Learning")
    st.json(summary_ml_comprehensive)

# Display Kelly fraction being used
st.info(f"**Using Kelly Fraction: {kelly_fraction}** (Quarter Kelly = 0.25, Half Kelly = 0.5, Full Kelly = 1.0)")

########################################
##### Bloco 9 – Exibição Final Expandida ###
########################################
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Goals_H_Today', 'Goals_A_Today',
    'Auto_Recommendation', 'ML_Recommendation',
    'Auto_Correct', 'ML_Correct',
    'Profit_Auto_Fixed', 'Profit_Auto_Kelly', 'Kelly_Stake_Auto',
    'Profit_ML_Fixed', 'Profit_ML_Kelly', 'Kelly_Stake_ML',
    'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away'
]

available_cols = [c for c in cols_to_show if c in games_today.columns]

st.subheader("📊 Games – Rules vs ML (Fixed vs Kelly Staking)")
st.dataframe(
    games_today[available_cols]
    .style.format({
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'Profit_Auto_Fixed': '{:.2f}',
        'Profit_Auto_Kelly': '{:.2f}',
        'Profit_ML_Fixed': '{:.2f}',
        'Profit_ML_Kelly': '{:.2f}',
        'Kelly_Stake_Auto': '{:.3f}',
        'Kelly_Stake_ML': '{:.3f}',
        'ML_Proba_Home': '{:.3f}',
        'ML_Proba_Draw': '{:.3f}',
        'ML_Proba_Away': '{:.3f}'
    }),
    use_container_width=True,
    height=1200
)

# Additional analysis: Compare performance metrics
st.subheader("🔍 Comparative Analysis")

comparison_data = {
    'Strategy': ['Auto Rules - Fixed', 'Auto Rules - Kelly', 'ML - Fixed', 'ML - Kelly'],
    'Total Profit': [
        summary_auto_comprehensive['Profit Fixed (Stake=1)'],
        summary_auto_comprehensive['Profit Kelly'],
        summary_ml_comprehensive['Profit Fixed (Stake=1)'],
        summary_ml_comprehensive['Profit Kelly']
    ],
    'ROI (%)': [
        summary_auto_comprehensive['ROI Fixed (%)'],
        summary_auto_comprehensive['ROI Kelly (%)'],
        summary_ml_comprehensive['ROI Fixed (%)'],
        summary_ml_comprehensive['ROI Kelly (%)']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df.style.format({'Total Profit': '{:.2f}', 'ROI (%)': '{:.2f}'}))
