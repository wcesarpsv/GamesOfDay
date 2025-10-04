########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import itertools
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(page_title="Today's Picks - ML + Parlay System", layout="wide")
st.title("🤖 ML Betting System + Auto Parlay Recommendations")

# Configurações principais
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "afc","trophy"]
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

def compute_double_chance_odds(df):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df['Odd_H']
    probs['p_D'] = 1 / df['Odd_D']
    probs['p_A'] = 1 / df['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)
    df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

def improve_data_quality(history):
    """Clean and enhance historical data"""
    if history.empty:
        return history
        
    # Remove outliers
    Q1 = history['M_H'].quantile(0.05)
    Q3 = history['M_H'].quantile(0.95)
    history = history[(history['M_H'] >= Q1) & (history['M_H'] <= Q3)]
    
    # Ensure sufficient data per league
    if 'League' in history.columns:
        league_counts = history['League'].value_counts()
        valid_leagues = league_counts[league_counts >= 50].index
        history = history[history['League'].isin(valid_leagues)]
    
    # Remove suspicious odds
    odds_columns = ['Odd_H', 'Odd_A', 'Odd_D']
    available_odds = [col for col in odds_columns if col in history.columns]
    
    for col in available_odds:
        history = history[(history[col] >= 1.2) & (history[col] <= 20)]
    
    return history

def create_advanced_features(df):
    """Create enhanced features for better model performance"""
    df = df.copy()
    
    # Power ratios and normalized metrics
    df['Power_Ratio'] = df['M_H'] / (df['M_A'] + 0.001)
    df['Total_Power'] = df['M_H'] + df['M_A']
    df['Power_Diff_Normalized'] = (df['M_H'] - df['M_A']) / (df['Total_Power'] + 0.001)
    
    # Odds-based features
    if 'Odd_H' in df.columns:
        df['Fair_Prob_Home'] = 1 / df['Odd_H']
        if 'Odd_A' in df.columns and 'Odd_D' in df.columns:
            df['Market_Margin'] = df['Fair_Prob_Home'] + (1 / df['Odd_A']) + (1 / df['Odd_D']) - 1
    
    # Home advantage factor
    df['Home_Advantage'] = df['M_H'] * 0.1
    
    return df

def clean_dataframe(df):
    """Remove NaN and infinite values from dataframe"""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df

def dynamic_threshold_adjustment(games_today):
    """Adjust threshold based on number of games and market quality"""
    if games_today.empty:
        return 0.65
    
    n_games = len(games_today)
    base_threshold = 0.65
    
    if n_games < 10:
        base_threshold += 0.05
    elif n_games > 30:
        base_threshold += 0.03
    
    if 'Odd_H' in games_today.columns:
        avg_odds_quality = games_today[['Odd_H', 'Odd_D', 'Odd_A']].mean().mean()
        if avg_odds_quality > 2.0:
            base_threshold -= 0.02
        
    return max(0.55, min(0.75, base_threshold))

def enhanced_ml_recommendation(row, threshold=0.65, min_value=0.02):
    """Enhanced recommendation with value betting principles"""
    
    if pd.isna(row.get('ML_Proba_Home')) or pd.isna(row.get('ML_Proba_Away')) or pd.isna(row.get('ML_Proba_Draw')):
        return "❌ Avoid"
    
    p_home = row['ML_Proba_Home']
    p_draw = row['ML_Proba_Draw'] 
    p_away = row['ML_Proba_Away']
    
    # Calculate expected value for each bet
    ev_home = p_home * row.get('Odd_H', 2.0) - 1
    ev_away = p_away * row.get('Odd_A', 2.0) - 1  
    ev_draw = p_draw * row.get('Odd_D', 3.0) - 1
    ev_1x = (p_home + p_draw) * row.get('Odd_1X', 1.5) - 1
    ev_x2 = (p_away + p_draw) * row.get('Odd_X2', 1.5) - 1
    
    recommendations = []
    
    if p_home >= threshold and ev_home >= min_value:
        recommendations.append(("🟢 Back Home", ev_home))
    if p_away >= threshold and ev_away >= min_value:
        recommendations.append(("🟠 Back Away", ev_away)) 
    if p_draw >= threshold and ev_draw >= min_value:
        recommendations.append(("⚪ Back Draw", ev_draw))
    
    if ev_1x >= min_value and (p_home + p_draw) >= 0.70:
        recommendations.append(("🟦 1X (Home/Draw)", ev_1x))
    if ev_x2 >= min_value and (p_away + p_draw) >= 0.70:
        recommendations.append(("🟪 X2 (Away/Draw)", ev_x2))
    
    if recommendations:
        best_rec = max(recommendations, key=lambda x: x[1])
        return best_rec[0]
    else:
        return "❌ Avoid"
        

########################################
####### Bloco 4 – Load Data ############
########################################
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Carregar os jogos do dia selecionado
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Apenas jogos sem placar final
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# Carregar histórico para treinar o modelo
all_games = load_all_games(GAMES_FOLDER)
all_games = filter_leagues(all_games)
history = prepare_history(all_games)

# Apply data quality improvements
history = improve_data_quality(history)

if history.empty:
    st.error("No valid historical data found.")
    st.stop()

# Extrair data do arquivo selecionado
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

# Inicializar colunas de gols
games_today['Goals_H_Today'] = np.nan
games_today['Goals_A_Today'] = np.nan

if os.path.exists(livescore_file):
    st.info(f"LiveScore file found: {livescore_file}")
    results_df = pd.read_csv(livescore_file)
    
    required_cols = ['game_id', 'status', 'home_goal', 'away_goal']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if not missing_cols:
        games_today = games_today.merge(
            results_df,
            left_on='Id',
            right_on='game_id',
            how='left',
            suffixes=('', '_RAW')
        )
        # Atualizar gols apenas para jogos finalizados
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
else:
    st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")

########################################
####### Bloco 5 – Features Engineering ##
########################################
# Apply advanced feature engineering
games_today = create_advanced_features(games_today)
history = create_advanced_features(history)

games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
history['M_Diff'] = history['M_H'] - history['M_A']
games_today = compute_double_chance_odds(games_today)

# Bandas e classificações de liga
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
    diff_q = hist.groupby('League')['M_Diff'].quantile([0.20, 0.80]).unstack().rename(columns={0.2:'P20_Diff', 0.8:'P80_Diff'}).reset_index()
    home_q = hist.groupby('League')['M_H'].quantile([0.20, 0.80]).unstack().rename(columns={0.2:'Home_P20', 0.8:'Home_P80'}).reset_index()
    away_q = hist.groupby('League')['M_A'].quantile([0.20, 0.80]).unstack().rename(columns={0.2:'Away_P20', 0.8:'Away_P80'}).reset_index()
    out = diff_q.merge(home_q, on='League', how='inner').merge(away_q, on='League', how='inner')
    return out

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

########################################
####### Bloco 6 – Train ML Model #######
########################################
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']: return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']: return "Away"
    else: return "Draw"

history['Result'] = history.apply(map_result, axis=1)

# Enhanced feature set
features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed',
    # New enhanced features
    'Power_Ratio', 'Total_Power', 'Power_Diff_Normalized',
    'Fair_Prob_Home', 'Market_Margin', 'Home_Advantage'
]
# Only include features that exist in the dataframe
features_raw = [f for f in features_raw if f in history.columns]

X = history[features_raw].copy()
y = history['Result']

# Clean the data before training
X = clean_dataframe(X)

BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: 
    X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
    X = X.drop('Home_Band', axis=1)
if 'Away_Band' in X: 
    X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)
    X = X.drop('Away_Band', axis=1)

cat_cols = [c for c in ['League_Classification'] if c in X]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

# Use only Random Forest for stability (removes Gradient Boosting that causes errors)
st.info("🏗️ Training Enhanced Random Forest Model...")

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

st.success("✅ Model trained successfully!")

########################################
####### Bloco 7 – Apply ML to Today ####
########################################
# Use dynamic threshold
threshold = dynamic_threshold_adjustment(games_today)
st.sidebar.metric("🎯 Dynamic ML Threshold", f"{threshold:.1%}")

def check_missing_features(row, features_required):
    """Verifica se há dados faltantes nas features essenciais"""
    missing_features = []
    
    for feature in features_required:
        if feature in row:
            if pd.isna(row[feature]) or row[feature] == '':
                missing_features.append(feature)
        else:
            missing_features.append(feature)
    
    return missing_features

# Lista de features obrigatórias
required_features = [
    'M_H', 'M_A', 'Diff_Power', 'M_Diff',
    'Home_Band', 'Away_Band', 'League_Classification',
    'Odd_H', 'Odd_D', 'Odd_A'
]

X_today = games_today[features_raw].copy()

# Aplicar validação de dados faltantes
games_today["ML_Data_Valid"] = True
games_today["Missing_Features"] = ""

for idx, row in games_today.iterrows():
    missing = check_missing_features(row, required_features)
    if missing:
        games_today.at[idx, "ML_Data_Valid"] = False
        games_today.at[idx, "Missing_Features"] = ", ".join(missing)

# Aplicar o modelo apenas aos jogos com dados completos
valid_games_mask = games_today["ML_Data_Valid"] == True
X_today_valid = X_today[valid_games_mask].copy()

# Clean today's data
X_today_valid = clean_dataframe(X_today_valid)

# Apply the same transformations as training data
if 'Home_Band' in X_today_valid: 
    X_today_valid['Home_Band_Num'] = X_today_valid['Home_Band'].map(BAND_MAP)
    X_today_valid = X_today_valid.drop('Home_Band', axis=1)
if 'Away_Band' in X_today_valid: 
    X_today_valid['Away_Band_Num'] = X_today_valid['Away_Band'].map(BAND_MAP)
    X_today_valid = X_today_valid.drop('Away_Band', axis=1)

if cat_cols:
    encoded_today = encoder.transform(X_today_valid[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today_valid = pd.concat([X_today_valid.drop(columns=cat_cols).reset_index(drop=True),
                             encoded_today_df.reset_index(drop=True)], axis=1)

# Ensure same columns as training data
missing_cols = set(X.columns) - set(X_today_valid.columns)
for col in missing_cols:
    X_today_valid[col] = 0
X_today_valid = X_today_valid[X.columns]

# Inicializar colunas de probabilidade com NaN
games_today["ML_Proba_Home"] = np.nan
games_today["ML_Proba_Draw"] = np.nan
games_today["ML_Proba_Away"] = np.nan
games_today["ML_Recommendation"] = "❌ Avoid"

# Aplicar modelo apenas nos jogos válidos
if not X_today_valid.empty:
    ml_proba = model.predict_proba(X_today_valid)
    
    # Preencher apenas os jogos válidos
    valid_indices = games_today[valid_games_mask].index
    
    games_today.loc[valid_indices, "ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
    games_today.loc[valid_indices, "ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
    games_today.loc[valid_indices, "ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]
    
    # Gerar recomendações usando enhanced function
    for idx in valid_indices:
        p_home = games_today.at[idx, "ML_Proba_Home"]
        p_draw = games_today.at[idx, "ML_Proba_Draw"] 
        p_away = games_today.at[idx, "ML_Proba_Away"]
        
        games_today.at[idx, "ML_Recommendation"] = enhanced_ml_recommendation(
            games_today.loc[idx], threshold
        )

# Mostrar estatísticas de validação
invalid_count = len(games_today) - valid_games_mask.sum()
if invalid_count > 0:
    st.warning(f"⚠️ {invalid_count} jogos excluídos por dados insuficientes")
    
    invalid_games = games_today[~valid_games_mask]
    if not invalid_games.empty:
        with st.expander("📋 Ver jogos com dados insuficientes"):
            st.dataframe(invalid_games[['Home', 'Away', 'League', 'Missing_Features']])

########################################
##### Bloco 8 – Kelly Criterion ########
########################################

# SEÇÃO 1: PARÂMETROS ML PRINCIPAL
st.sidebar.header("🎯 ML Principal System")

bankroll = st.sidebar.number_input("ML Bankroll Size", 100, 10000, 1000, 100, help="Bankroll para apostas individuais do ML")
kelly_fraction = st.sidebar.slider("Kelly Fraction ML", 0.1, 1.0, 0.25, 0.05, help="Fração do Kelly para apostas individuais (mais conservador = menor)")
min_stake = st.sidebar.number_input("Minimum Stake ML", 1, 50, 1, 1, help="Stake mínimo por aposta individual")
max_stake = st.sidebar.number_input("Maximum Stake ML", 10, 500, 100, 10, help="Stake máximo por aposta individual")

# Resumo ML Principal
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 ML Principal**")
st.sidebar.markdown("• Apostas individuais com edge comprovado  \n• Kelly determina stake ideal  \n• Foco em valor a longo prazo")

def kelly_stake(probability, odds, bankroll=1000, kelly_fraction=0.25, min_stake=1, max_stake=100):
    if pd.isna(probability) or pd.isna(odds) or odds <= 1 or probability <= 0: return 0
    edge = probability * odds - 1
    if edge <= 0: return 0
    full_kelly_fraction = edge / (odds - 1)
    fractional_kelly = full_kelly_fraction * kelly_fraction
    recommended_stake = fractional_kelly * bankroll
    if recommended_stake < min_stake: return 0
    elif recommended_stake > max_stake: return max_stake
    else: return round(recommended_stake, 2)

def get_kelly_stake_ml(row):
    rec = row['ML_Recommendation']
    if pd.isna(rec) or rec == '❌ Avoid': return 0
    
    if 'Back Home' in rec: return kelly_stake(row['ML_Proba_Home'], row['Odd_H'], bankroll, kelly_fraction, min_stake, max_stake)
    elif 'Back Away' in rec: return kelly_stake(row['ML_Proba_Away'], row['Odd_A'], bankroll, kelly_fraction, min_stake, max_stake)
    elif 'Back Draw' in rec: return kelly_stake(row['ML_Proba_Draw'], row['Odd_D'], bankroll, kelly_fraction, min_stake, max_stake)
    elif '1X' in rec: return kelly_stake(row['ML_Proba_Home'] + row['ML_Proba_Draw'], row['Odd_1X'], bankroll, kelly_fraction, min_stake, max_stake)
    elif 'X2' in rec: return kelly_stake(row['ML_Proba_Away'] + row['ML_Proba_Draw'], row['Odd_X2'], bankroll, kelly_fraction, min_stake, max_stake)
    return 0

games_today['Kelly_Stake_ML'] = games_today.apply(get_kelly_stake_ml, axis=1)

########################################
##### Bloco 9 – Result Tracking ########
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

games_today['ML_Correct'] = games_today.apply(lambda r: check_recommendation(r['ML_Recommendation'], r['Result_Today']), axis=1)

def calculate_profit_with_kelly(rec, result, odds_row, ml_probabilities):
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return 0, 0
    
    rec = str(rec)
    stake_fixed = 1
    
    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        stake_kelly = kelly_stake(ml_probabilities.get('Home', 0.5), odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result == "Home" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Home" else -stake_kelly
        
    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        stake_kelly = kelly_stake(ml_probabilities.get('Away', 0.5), odd, bankroll, kelly_fraction, min_stake, max_stake)
        profit_fixed = odd - 1 if result == "Away" else -1
        profit_kelly = (odd - 1) * stake_kelly if result == "Away" else -stake_kelly
        
    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        stake_kelly = kelly_stake(ml_probabilities.get('Draw', 0.5), odd, bankroll, kelly_fraction, min_stake, max_stake)
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

# Calcular profits
games_today[['Profit_ML_Fixed', 'Profit_ML_Kelly']] = games_today.apply(
    lambda r: calculate_profit_with_kelly(
        r['ML_Recommendation'], 
        r['Result_Today'], 
        r,
        {'Home': r.get('ML_Proba_Home', 0.5), 
         'Draw': r.get('ML_Proba_Draw', 0.5), 
         'Away': r.get('ML_Proba_Away', 0.5)}
    ), 
    axis=1, result_type='expand'
)

########################################
#### Bloco 10 – Auto Parlay System #####
########################################

# SEÇÃO 2: PARÂMETROS PARLAY
st.sidebar.header("🎰 Parlay System")

parlay_bankroll = st.sidebar.number_input("Parlay Bankroll", 50, 5000, 200, 50, help="Bankroll separado para parlays")
min_parlay_prob = st.sidebar.slider("Min Probability Parlay", 0.50, 0.70, 0.50, 0.01, help="Probabilidade mínima para considerar jogo no parlay")
max_parlay_suggestions = st.sidebar.slider("Max Parlay Suggestions", 1, 10, 5, 1, help="Número máximo de sugestões de parlay")

# CONTROLE DE LEGS
st.sidebar.markdown("---")
min_parlay_legs = st.sidebar.slider("Min Legs", 2, 4, 2, 1, help="Número mínimo de jogos no parlay")
max_parlay_legs = st.sidebar.slider("Max Legs", 2, 4, 4, 1, help="Número máximo de jogos no parlay")

# FILTROS PARA FINS DE SEMANA
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Filtros Fim de Semana**")
weekend_filter = st.sidebar.checkbox("Ativar Filtro FDS", value=True, help="Filtros mais rigorosos para muitos jogos")
max_eligible_games = st.sidebar.slider("Máx Jogos Elegíveis", 10, 50, 20, 5, help="Limitar jogos para cálculo (evitar travamento)")

# Resumo Parlay System
st.sidebar.markdown("---")
st.sidebar.markdown("**🎰 Parlay System**")
st.sidebar.markdown("• Combina jogos sem edge individual  \n• Busca EV positivo em combinações  \n• Bankroll separado do principal")

def calculate_parlay_odds(games_list, games_df):
    total_prob = 1.0
    total_odds = 1.0
    game_details = []
    
    for game_idx, bet_type in games_list:
        game = games_df.loc[game_idx]
        if bet_type == 'Home':
            prob = game['ML_Proba_Home']
            odds = game['Odd_H']
        elif bet_type == 'Away':
            prob = game['ML_Proba_Away']
            odds = game['Odd_A']
        elif bet_type == 'Draw':
            prob = game['ML_Proba_Draw']
            odds = game['Odd_D']
        elif bet_type == '1X':
            prob = game['ML_Proba_Home'] + game['ML_Proba_Draw']
            odds = game['Odd_1X']
        elif bet_type == 'X2':
            prob = game['ML_Proba_Away'] + game['ML_Proba_Draw']
            odds = game['Odd_X2']
        
        total_prob *= prob
        total_odds *= odds
        game_details.append({
            'game': f"{game['Home']} vs {game['Away']}",
            'bet': bet_type,
            'prob': prob,
            'odds': round(odds, 2)
        })
    
    expected_value = total_prob * total_odds - 1
    return total_prob, round(total_odds, 2), expected_value, game_details

def generate_parlay_suggestions(games_df, bankroll_parlay=200, min_prob=0.50, max_suggestions=5, min_legs=2, max_legs=4, weekend_filter=True, max_eligible=20):
    games_today_filtered = games_df.copy()
    
    eligible_games = []
    
    for idx, row in games_today_filtered.iterrows():
        if row['ML_Recommendation'] != '❌ Avoid':
            rec = row['ML_Recommendation']
            
            if 'Back Home' in rec:
                prob = row['ML_Proba_Home']
                odds = row['Odd_H']
                bet_type = 'Home'
                edge = prob * odds - 1
            elif 'Back Away' in rec:
                prob = row['ML_Proba_Away'] 
                odds = row['Odd_A']
                bet_type = 'Away'
                edge = prob * odds - 1
            elif 'Back Draw' in rec:
                prob = row['ML_Proba_Draw']
                odds = row['Odd_D']
                bet_type = 'Draw'
                edge = prob * odds - 1
            elif '1X' in rec:
                prob = row['ML_Proba_Home'] + row['ML_Proba_Draw']
                odds = row['Odd_1X']
                bet_type = '1X'
                edge = prob * odds - 1
            elif 'X2' in rec:
                prob = row['ML_Proba_Away'] + row['ML_Proba_Draw']
                odds = row['Odd_X2']
                bet_type = 'X2'
                edge = prob * odds - 1
            else:
                continue
            
            # FILTROS MAIS RIGOROSOS PARA FINS DE SEMANA
            if weekend_filter and len(games_today_filtered) > 15:
                # Critérios mais rigorosos quando há muitos jogos
                if prob > (min_prob + 0.05) and edge > -0.05:  # Prob maior e edge menos negativo
                    eligible_games.append((idx, bet_type, prob, round(odds, 2), edge))
            else:
                # Critérios normais para poucos jogos
                if prob > min_prob:
                    eligible_games.append((idx, bet_type, prob, round(odds, 2), edge))
    
    # LIMITAR NÚMERO DE JOGOS ELEGÍVEIS
    if len(eligible_games) > max_eligible:
        # Ordenar por probabilidade e pegar os melhores
        eligible_games.sort(key=lambda x: x[2], reverse=True)  # Ordenar por prob
        eligible_games = eligible_games[:max_eligible]
        st.warning(f"⚡ Limite ativado: {len(eligible_games)} jogos elegíveis (de {len(games_today_filtered)} totais)")
    
    st.info(f"🎯 Jogos elegíveis para parlays: {len(eligible_games)}")
    
    # CALCULAR QUANTIDADE MÁXIMA DE COMBINAÇÕES
    total_combinations = 0
    if min_legs <= 2 and len(eligible_games) >= 2:
        total_combinations += math.comb(len(eligible_games), 2)
    if min_legs <= 3 and max_legs >= 3 and len(eligible_games) >= 3:
        total_combinations += math.comb(len(eligible_games), 3)
    if max_legs >= 4 and len(eligible_games) >= 4:
        total_combinations += math.comb(len(eligible_games), 4)
    
    st.info(f"🧮 Combinações possíveis: {total_combinations:,}")
    
    # AVISO SE MUITAS COMBINAÇÕES
    if total_combinations > 1000:
        st.warning("⚠️ Muitas combinações! Use filtros mais rigorosos.")
    
    parlay_suggestions = []
    
    # PARLAYS DE 2 LEGS - CRITÉRIOS MAIS RIGOROSOS PARA FDS
    if min_legs <= 2 and len(eligible_games) >= 2:
        ev_threshold = 0.08 if weekend_filter and len(eligible_games) > 15 else 0.05
        prob_threshold = 0.30 if weekend_filter and len(eligible_games) > 15 else 0.25
        
        for combo in itertools.combinations(eligible_games, 2):
            games_list = [(game[0], game[1]) for game in combo]
            prob, odds, ev, details = calculate_parlay_odds(games_list, games_today_filtered)
            
            if ev > ev_threshold and prob > prob_threshold:
                stake = min(parlay_bankroll * 0.08, parlay_bankroll * 0.12 * prob)
                stake = round(stake, 2)
                
                if stake >= 5:
                    parlay_suggestions.append({
                        'type': '2-Leg Parlay',
                        'games': games_list,
                        'probability': prob,
                        'odds': odds,
                        'ev': ev,
                        'stake': stake,
                        'potential_win': round(stake * odds - stake, 2),
                        'details': details
                    })
    
    # PARLAYS DE 3 LEGS
    if min_legs <= 3 and max_legs >= 3 and len(eligible_games) >= 3:
        ev_threshold = 0.05 if weekend_filter and len(eligible_games) > 15 else 0.02
        prob_threshold = 0.20 if weekend_filter and len(eligible_games) > 15 else 0.15
        
        for combo in itertools.combinations(eligible_games, 3):
            games_list = [(game[0], game[1]) for game in combo]
            prob, odds, ev, details = calculate_parlay_odds(games_list, games_today_filtered)
            
            if ev > ev_threshold and prob > prob_threshold:
                stake = min(parlay_bankroll * 0.05, parlay_bankroll * 0.08 * prob)
                stake = round(stake, 2)
                
                if stake >= 3:
                    parlay_suggestions.append({
                        'type': '3-Leg Parlay',
                        'games': games_list,
                        'probability': prob,
                        'odds': odds,
                        'ev': ev,
                        'stake': stake,
                        'potential_win': round(stake * odds - stake, 2),
                        'details': details
                    })
    
    # PARLAYS DE 4 LEGS - APENAS SE POUCOS JOGOS
    if max_legs >= 4 and len(eligible_games) >= 4 and len(eligible_games) <= 25:  # Só calcular se até 25 jogos
        for combo in itertools.combinations(eligible_games, 4):
            games_list = [(game[0], game[1]) for game in combo]
            prob, odds, ev, details = calculate_parlay_odds(games_list, games_today_filtered)
            
            if ev > 0.10 and prob > 0.10:
                stake = min(parlay_bankroll * 0.03, parlay_bankroll * 0.05 * prob)
                stake = round(stake, 2)
                
                if stake >= 2:
                    parlay_suggestions.append({
                        'type': '4-Leg Parlay',
                        'games': games_list,
                        'probability': prob,
                        'odds': odds,
                        'ev': ev,
                        'stake': stake,
                        'potential_win': round(stake * odds - stake, 2),
                        'details': details
                    })
    
    # Ordenar por Expected Value
    parlay_suggestions.sort(key=lambda x: x['ev'], reverse=True)
    
    st.info(f"🎰 Total de parlays gerados: {len(parlay_suggestions)}")
    
    return parlay_suggestions[:max_suggestions]

# Gerar sugestões de parlay COM NOVOS PARÂMETROS
parlay_suggestions = generate_parlay_suggestions(
    games_today, 
    parlay_bankroll, 
    min_parlay_prob, 
    max_parlay_suggestions,
    min_parlay_legs,
    max_parlay_legs,
    weekend_filter,      # NOVO
    max_eligible_games   # NOVO
)

########################################
##### Bloco 11 – Performance Summary ###
########################################
finished_games = games_today.dropna(subset=['Result_Today'])

def track_model_performance(games_today, finished_games):
    """Monitor model performance in real-time"""
    
    if len(finished_games) == 0:
        return
    
    recent_bets = finished_games[finished_games['ML_Recommendation'] != '❌ Avoid']
    
    if len(recent_bets) > 0:
        accuracy = recent_bets['ML_Correct'].mean()
        total_ev = recent_bets['Profit_ML_Fixed'].sum()
        
        st.sidebar.metric("📊 Recent Accuracy", f"{accuracy:.1%}")
        st.sidebar.metric("💰 Recent EV", f"{total_ev:.2f}")
        
        # Alert if performance drops
        if accuracy < 0.45:
            st.sidebar.error("⚠️ Performance alert: Consider adjusting thresholds")

def summary_stats_ml(df):
    bets = df[df['ML_Correct'].notna()]
    total_bets = len(bets)
    correct_bets = bets['ML_Correct'].sum()
    winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
    
    # Fixed stake profits
    total_profit_fixed = bets['Profit_ML_Fixed'].sum()
    roi_fixed = (total_profit_fixed / total_bets) * 100 if total_bets > 0 else 0
    
    # Kelly stake profits
    total_profit_kelly = bets['Profit_ML_Kelly'].sum()
    total_stake_kelly = bets['Kelly_Stake_ML'].sum()
    roi_kelly = (total_profit_kelly / total_stake_kelly) * 100 if total_stake_kelly > 0 else 0
    
    # Average stake sizes
    avg_stake_kelly = bets['Kelly_Stake_ML'].mean() if total_bets > 0 else 0
    
    # Kelly bets made
    kelly_bets = bets[bets['Kelly_Stake_ML'] > 0]

    return {
        "Total Games": len(df),
        "Bets Made": total_bets,
        "Correct": int(correct_bets),
        "Winrate (%)": round(winrate, 2),
        "Profit Fixed (Stake=1)": round(total_profit_fixed, 2),
        "ROI Fixed (%)": round(roi_fixed, 2),
        "Profit Kelly": round(total_profit_kelly, 2),
        "Total Stake Kelly": round(total_stake_kelly, 2),
        "ROI Kelly (%)": round(roi_kelly, 2),
        "Avg Kelly Stake": round(avg_stake_kelly, 2),
        "Kelly Bets Made": len(kelly_bets)
    }

summary_ml = summary_stats_ml(finished_games)
track_model_performance(games_today, finished_games)

########################################
##### Bloco 12 – SUPER PARLAY OF THE DAY #
########################################

# SEÇÃO 4: SUPER PARLAY
st.sidebar.header("🎉 SUPER PARLAY OF THE DAY")

super_parlay_stake = st.sidebar.number_input("Super Parlay Stake", 10, 100, 10, 10, help="Stake fixo para o Super Parlay (aposta divertida)")
target_super_odds = st.sidebar.slider("Target Odds", 20, 100, 50, 5, help="Odd alvo para o Super Parlay")

# Resumo Super Parlay
st.sidebar.markdown("---")
st.sidebar.markdown("**🎉 SUPER PARLAY**")
st.sidebar.markdown("• Combina as maiores probabilidades  \n• Odd alvo: ~50  \n• Aposta divertida ($2-5)  \n• Ideal para compartilhar")

def generate_super_parlay(games_df, target_odds=50, max_games=8):
    """Gera um SUPER PARLAY com as maiores probabilidades até atingir a odd alvo"""
    
    # Filtrar apenas jogos de hoje com recomendação
    games_today = games_df[games_df['ML_Recommendation'] != '❌ Avoid'].copy()
    
    if len(games_today) < 3:
        return None
    
    # Criar lista de todas as probabilidades disponíveis
    all_bets = []
    
    for idx, row in games_today.iterrows():
        rec = row['ML_Recommendation']
        
        if 'Back Home' in rec:
            prob = row['ML_Proba_Home']
            odds = row['Odd_H']
            bet_type = 'Home'
        elif 'Back Away' in rec:
            prob = row['ML_Proba_Away']
            odds = row['Odd_A']
            bet_type = 'Away'
        elif 'Back Draw' in rec:
            prob = row['ML_Proba_Draw']
            odds = row['Odd_D']
            bet_type = 'Draw'
        elif '1X' in rec:
            prob = row['ML_Proba_Home'] + row['ML_Proba_Draw']
            odds = row['Odd_1X']
            bet_type = '1X'
        elif 'X2' in rec:
            prob = row['ML_Proba_Away'] + row['ML_Proba_Draw']
            odds = row['Odd_X2']
            bet_type = 'X2'
        else:
            continue
        
        all_bets.append({
            'game_idx': idx,
            'bet_type': bet_type,
            'probability': prob,
            'odds': odds,
            'game': f"{row['Home']} vs {row['Away']}",
            'league': row['League']
        })
    
    # Ordenar por probabilidade (maior primeiro)
    all_bets.sort(key=lambda x: x['probability'], reverse=True)
    
    # Selecionar combinação que mais se aproxima da odd alvo
    best_combination = []
    current_odds = 1.0
    current_prob = 1.0
    
    for bet in all_bets[:max_games]:  # Limitar a 8 jogos no máximo
        if current_odds * bet['odds'] <= target_odds * 1.5:  # Não ultrapassar muito a odd alvo
            best_combination.append(bet)
            current_odds *= bet['odds']
            current_prob *= bet['probability']
            
            # Parar quando atingir ou ultrapassar a odd alvo
            if current_odds >= target_odds:
                break
    
    # Calcular estatísticas finais
    if len(best_combination) >= 3:  # Mínimo de 3 legs
        expected_value = current_prob * current_odds - 1
        potential_win = super_parlay_stake * current_odds - super_parlay_stake
        
        return {
            'type': f'SUPER PARLAY ({len(best_combination)} legs)',
            'games': [(bet['game_idx'], bet['bet_type']) for bet in best_combination],
            'probability': current_prob,
            'odds': round(current_odds, 2),
            'ev': expected_value,
            'stake': super_parlay_stake,
            'potential_win': round(potential_win, 2),
            'details': [{
                'game': bet['game'],
                'bet': bet['bet_type'],
                'prob': bet['probability'],
                'odds': round(bet['odds'], 2),
                'league': bet['league']
            } for bet in best_combination]
        }
    
    return None

# Gerar SUPER PARLAY
super_parlay = generate_super_parlay(games_today, target_super_odds)

########################################
##### Bloco 13 – Display Results #######
########################################

# SEÇÃO 3: RESUMO GERAL - ATUALIZADO
st.sidebar.header("📊 System Summary")
st.sidebar.markdown(f"""
**⚙️ Configuração Atual**  
• **ML Bankroll:** ${bankroll:,}  
• **Parlay Bankroll:** ${parlay_bankroll:,}  
• **Super Parlay Stake:** ${super_parlay_stake}  
• **Kelly Fraction:** {kelly_fraction}  
• **Min Prob Parlay:** {min_parlay_prob:.0%}  
• **Parlay Legs:** {min_parlay_legs}-{max_parlay_legs}  
• **Super Parlay Target:** {target_super_odds}  
""")

st.header("📈 Day's Summary - Machine Learning Performance")
st.json(summary_ml)

st.header("🎯 Machine Learning Recommendations")

# ATUALIZADO: Adicionar coluna de validação
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
    'ML_Recommendation', 'ML_Data_Valid', 'ML_Correct', 'Kelly_Stake_ML',  # NOVO
    'Profit_ML_Fixed', 'Profit_ML_Kelly',
    'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away', 
    'Odd_H', 'Odd_D', 'Odd_A'
]

available_cols = [c for c in cols_to_show if c in games_today.columns]

st.dataframe(
    games_today[available_cols].style.format({
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'Kelly_Stake_ML': '{:.2f}',
        'Profit_ML_Fixed': '{:.2f}',
        'Profit_ML_Kelly': '{:.2f}',
        'ML_Proba_Home': '{:.3f}',
        'ML_Proba_Draw': '{:.3f}',
        'ML_Proba_Away': '{:.3f}',
        'Odd_H': '{:.2f}',
        'Odd_D': '{:.2f}',
        'Odd_A': '{:.2f}'
    }).apply(lambda x: ['background-color: #ffcccc' if x.name == 'ML_Data_Valid' and x.iloc[0] == False else '' 
                       for i in range(len(x))], axis=1),  # Destaque visual
    use_container_width=True,
    height=800
)
    

st.header("🎰 Auto Parlay Recommendations")

if parlay_suggestions:
    # Mostrar estatísticas dos parlays
    legs_count = {}
    for parlay in parlay_suggestions:
        leg_type = parlay['type']
        legs_count[leg_type] = legs_count.get(leg_type, 0) + 1
    
    stats_text = " | ".join([f"{count}x {leg}" for leg, count in legs_count.items()])
    st.success(f"📊 Distribuição: {stats_text}")
    
    for i, parlay in enumerate(parlay_suggestions):
        with st.expander(f"#{i+1} {parlay['type']} - Prob: {parlay['probability']:.1%} | Odds: {parlay['odds']} | EV: {parlay['ev']:+.1%}"):
            st.write(f"**Stake Sugerido:** ${parlay['stake']} | **Potencial:** ${parlay['potential_win']}")
            
            for detail in parlay['details']:
                st.write(f"• {detail['game']} - {detail['bet']} (Prob: {detail['prob']:.1%}, Odd: {detail['odds']})")
else:
    st.info("No profitable parlay suggestions found for today.")
    
    

# SUPER PARLAY SECTION
st.header("🎉 SUPER PARLAY OF THE DAY")

if super_parlay:
    # Display especial para o SUPER PARLAY
    st.success("🔥 **SPECIAL OF THE DAY!** 🔥")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilidade", f"{super_parlay['probability']:.1%}")
    with col2:
        st.metric("Odds", f"{super_parlay['odds']:.2f}")
    with col3:
        st.metric("Potencial", f"${super_parlay['potential_win']:.2f}")
    
    st.write(f"**Stake Recomendado:** ${super_parlay['stake']} | **Expected Value:** {super_parlay['ev']:+.1%}")
    
    # Mostrar jogos em formato mais visual
    st.subheader("🎯 Jogos Selecionados:")
    for i, detail in enumerate(super_parlay['details'], 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{i}. {detail['game']}** ({detail['league']})")
        with col2:
            st.write(f"**{detail['bet']}** (Odd: {detail['odds']})")
    
    # Botão para compartilhar (simulado)
    st.markdown("---")
    st.markdown("**📱 Compartilhe este Super Parlay!**")
    
else:
    st.info("Não foi possível gerar um Super Parlay hoje. Tente ajustar a odd alvo ou aguarde mais jogos.")

st.success("🚀 Enhanced ML System Active: Ensemble Model + Advanced Features + Dynamic Thresholds")
