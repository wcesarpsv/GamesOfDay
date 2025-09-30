########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import itertools
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


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

features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed'
]
features_raw = [f for f in features_raw if f in history.columns]

X = history[features_raw].copy()
y = history['Result']

BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

cat_cols = [c for c in ['League_Classification'] if c in X]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

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


########################################
####### Bloco 7 – Apply ML to Today ####
########################################
threshold = st.sidebar.slider("ML Threshold for Direct Win (%)", 50, 80, 65) / 100.0

def ml_recommendation_from_proba(p_home, p_draw, p_away, threshold=0.65):
    if p_home >= threshold: return "🟢 Back Home"
    elif p_away >= threshold: return "🟠 Back Away"
    else:
        sum_home_draw = p_home + p_draw
        sum_away_draw = p_away + p_draw
        if abs(p_home - p_away) < 0.05 and p_draw > 0.50: return "⚪ Back Draw"
        elif sum_home_draw > sum_away_draw: return "🟦 1X (Home/Draw)"
        elif sum_away_draw > sum_home_draw: return "🟪 X2 (Away/Draw)"
        else: return "❌ Avoid"

X_today = games_today[features_raw].copy()
if 'Home_Band' in X_today: X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                         encoded_today_df.reset_index(drop=True)], axis=1)

ml_proba = model.predict_proba(X_today)
games_today["ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
games_today["ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
games_today["ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]

games_today["ML_Recommendation"] = [
    ml_recommendation_from_proba(row["ML_Proba_Home"], row["ML_Proba_Draw"], row["ML_Proba_Away"], threshold)
    for _, row in games_today.iterrows()
]


########################################
##### Bloco 8 – Kelly Criterion ########
########################################

# SEÇÃO 1: PARÂMETROS ML PRINCIPAL
st.sidebar.header("🎯 ML Principal System")

bankroll = st.sidebar.number_input("ML Bankroll Size", 100, 10000, 1000, 100, help="Bankroll para apostas individuais do ML")
kelly_fraction = st.sidebar.slider("Kelly Fraction ML", 0.1, 1.0, 0.25, 0.05, help="Fração do Kelly para apostas individuais (mais conservador = menor)")
min_stake = st.sidebar.number_input("Minimum Stake ML", 1, 50, 1, 1, help="Stake mínimo por aposta individual")
max_stake = st.sidebar.number_input("Maximum Stake ML", 10, 500, 100, 10, help="Stake máximo por aposta individual")

# Resumo ML Principal - CORRIGIDO
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; border-left: 4px solid #1890ff; margin: 10px 0;">
<small><strong>🎯 ML Principal</strong><br>
• Apostas individuais com edge comprovado<br>
• Kelly determina stake ideal<br>
• Foco em valor a longo prazo</small>
</div>
""", unsafe_allow_html=True)

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
games_today['Profit_ML_Fixed'] = games_today.apply(
    lambda r: calculate_profit(r['ML_Recommendation'], r['Result_Today'], r), axis=1
)

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

# Resumo Parlay System - CORRIGIDO
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

def generate_parlay_suggestions(games_df, bankroll_parlay=200, min_prob=0.50, max_suggestions=5):
    # Filtrar apenas jogos de hoje
    today = datetime.now().strftime("%Y-%m-%d")
    games_today_filtered = games_df[games_df['Date'] == today].copy()
    
    eligible_games = []
    
    for idx, row in games_today_filtered.iterrows():
        kelly_zero = row['Kelly_Stake_ML'] == 0
        
        # 🔥 NOVA LÓGICA: Usar a recomendação do ML (não a maior prob)
        if kelly_zero and row['ML_Recommendation'] != '❌ Avoid':
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
            
            # Só adicionar se atender critério mínimo de probabilidade
            if prob > min_prob:
                eligible_games.append((idx, bet_type, prob, round(odds, 2)))
    
    parlay_suggestions = []
    
    # Parlays de 2 legs - FOCAR EM EV POSITIVO
    for combo in itertools.combinations(eligible_games, 2):
        games_list = [(game[0], game[1]) for game in combo]
        prob, odds, ev, details = calculate_parlay_odds(games_list, games_today_filtered)
        
        # 🔥 CRITÉRIO PRINCIPAL: EV POSITIVO
        if ev > 0 and prob > 0.20 and odds > 1.80:
            stake = min(parlay_bankroll * 0.05, parlay_bankroll * 0.08 * prob)
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
    
    # Ordenar por Expected Value (mais importante)
    parlay_suggestions.sort(key=lambda x: x['ev'], reverse=True)
    
    return parlay_suggestions[:max_suggestions]

# Gerar sugestões de parlay
parlay_suggestions = generate_parlay_suggestions(
    games_today, parlay_bankroll, min_parlay_prob, max_parlay_suggestions
)

########################################
##### Bloco 11 – Performance Summary ###
########################################
finished_games = games_today.dropna(subset=['Result_Today'])

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


########################################
##### Bloco 12 – Display Results #######
########################################

# SEÇÃO 3: RESUMO GERAL - CORRIGIDO
st.sidebar.header("📊 System Summary")
st.sidebar.markdown(f"""
<div style="background-color: #f6ffed; padding: 10px; border-radius: 5px; border-left: 4px solid #52c41a; margin: 10px 0;">
<small><strong>⚙️ Configuração Atual</strong><br>
• <strong>ML Bankroll:</strong> ${bankroll:,}<br>
• <strong>Parlay Bankroll:</strong> ${parlay_bankroll:,}<br>
• <strong>Kelly Fraction:</strong> {kelly_fraction}<br>
• <strong>Min Prob Parlay:</strong> {min_parlay_prob:.0%}</small>
</div>
""", unsafe_allow_html=True)

st.header("📈 Day's Summary - Machine Learning Performance")
st.json(summary_ml)

st.header("🎯 Machine Learning Recommendations")

# Mostrar Kelly stakes e recomendações ML
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
    'ML_Recommendation', 'ML_Correct', 'Kelly_Stake_ML',
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
    }),
    use_container_width=True
)

st.header("🎰 Auto Parlay Recommendations")

if parlay_suggestions:
    for i, parlay in enumerate(parlay_suggestions):
        with st.expander(f"#{i+1} {parlay['type']} - Prob: {parlay['probability']:.1%} | Odds: {parlay['odds']} | EV: {parlay['ev']:+.1%}"):
            st.write(f"**Stake Sugerido:** ${parlay['stake']} | **Potencial:** ${parlay['potential_win']}")
            
            for detail in parlay['details']:
                st.write(f"• {detail['game']} - {detail['bet']} (Prob: {detail['prob']:.1%}, Odd: {detail['odds']})")
else:
    st.info("No profitable parlay suggestions found for today.")
