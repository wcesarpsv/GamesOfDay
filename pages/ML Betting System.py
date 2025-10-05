########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import itertools
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
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
    """Carrega todos os arquivos CSV do folder especificado"""
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
    """Filtra ligas baseado nas keywords excluídas"""
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def prepare_history(df):
    """Prepara dados históricos removendo valores NaN"""
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])

def compute_double_chance_odds(df):
    """Calcula odds para dupla chance (1X e X2)"""
    if df.empty or not all(col in df.columns for col in ['Odd_H', 'Odd_D', 'Odd_A']):
        return df
        
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
    if 'M_H' in history.columns:
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
    if df.empty:
        return df
        
    df = df.copy()
    
    # Power ratios and normalized metrics - BALANCEADAS
    if all(col in df.columns for col in ['M_H', 'M_A']):
        df['Power_Ratio_Home'] = df['M_H'] / (df['M_A'] + 0.001)
        df['Power_Ratio_Away'] = df['M_A'] / (df['M_H'] + 0.001)
        df['Total_Power'] = df['M_H'] + df['M_A']
        df['Power_Diff_Normalized'] = (df['M_H'] - df['M_A']) / (df['Total_Power'] + 0.001)
    
    # Odds-based features - BALANCEADAS
    if all(col in df.columns for col in ['Odd_H', 'Odd_A', 'Odd_D']):
        df['Fair_Prob_Home'] = 1 / df['Odd_H']
        df['Fair_Prob_Away'] = 1 / df['Odd_A']
        df['Fair_Prob_Draw'] = 1 / df['Odd_D']
        df['Market_Margin'] = df['Fair_Prob_Home'] + df['Fair_Prob_Away'] + df['Fair_Prob_Draw'] - 1
        
        # Probabilidade relativa Home vs Away
        df['Prob_Ratio_Home_Away'] = df['Fair_Prob_Home'] / (df['Fair_Prob_Away'] + 0.001)
        df['Prob_Diff_Home_Away'] = df['Fair_Prob_Home'] - df['Fair_Prob_Away']
    
    # Home/Away advantage - BALANCEADAS
    if all(col in df.columns for col in ['M_H', 'M_A']):
        df['Home_Advantage'] = df['M_H'] * 0.1
        df['Away_Strength'] = df['M_A'] * 0.1
        df['Advantage_Ratio'] = df['Home_Advantage'] / (df['Away_Strength'] + 0.001)
    
    # Value detection features
    if all(col in df.columns for col in ['ML_Proba_Home', 'ML_Proba_Away', 'Fair_Prob_Home', 'Fair_Prob_Away']):
        df['Value_Home'] = (df['ML_Proba_Home'] - df['Fair_Prob_Home']) / (df['Fair_Prob_Home'] + 0.001)
        df['Value_Away'] = (df['ML_Proba_Away'] - df['Fair_Prob_Away']) / (df['Fair_Prob_Away'] + 0.001)
        df['Value_Diff'] = df['Value_Home'] - df['Value_Away']
    
    return df

def clean_dataframe(df):
    """Remove NaN and infinite values from dataframe"""
    if df.empty:
        return df
        
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df

def auto_adjust_threshold(games_today, target_recommendations=8):
    """Ajusta automaticamente o threshold baseado no número de recomendações"""
    if games_today.empty:
        return 0.60
    
    valid_games = games_today[games_today['ML_Data_Valid'] == True]
    if len(valid_games) == 0:
        return 0.60
    
    # Testa diferentes thresholds
    test_threshold = 0.50
    max_threshold = 0.70
    
    best_threshold = 0.60
    best_count = 0
    
    while test_threshold <= max_threshold:
        count = 0
        for idx, row in valid_games.iterrows():
            if not row['ML_Data_Valid']:
                continue
                
            p_home = row.get('ML_Proba_Home', 0)
            p_draw = row.get('ML_Proba_Draw', 0) 
            p_away = row.get('ML_Proba_Away', 0)
            
            # Calcular EVs para ver se tem valor
            ev_home = p_home * row.get('Odd_H', 2.0) - 1
            ev_away = p_away * row.get('Odd_A', 2.0) - 1
            ev_draw = p_draw * row.get('Odd_D', 3.0) - 1
            
            max_prob = max(p_home, p_draw, p_away)
            max_ev = max(ev_home, ev_away, ev_draw)
            
            if max_prob >= test_threshold and max_ev >= 0.02:
                count += 1
        
        # Prefere thresholds que dão perto do target
        if abs(count - target_recommendations) < abs(best_count - target_recommendations):
            best_threshold = test_threshold
            best_count = count
        
        test_threshold += 0.02
    
    return best_threshold


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

def is_realistic_odd(probability, odds, bet_type='single'):
    """
    Verifica se a odd é realista para a probabilidade
    Evita recomendações baseadas em odds infladas/distorcidas
    """
    if pd.isna(probability) or pd.isna(odds) or odds <= 1:
        return False
    
    # REGRAS POR TIPO DE APOSTA
    if bet_type == 'single':
        if odds > 8.0 and probability < 0.35:
            return False  # Odd muito alta + prob muito baixa
        if odds > 6.0 and probability < 0.30:
            return False  # Odd alta + prob insuficiente
        if odds > 4.0 and probability < 0.25:
            return False  # Risco muito alto
    elif bet_type in ['1x', 'x2']:
        if odds > 3.0 and probability < 0.60:
            return False  # Double chance com odd alta e prob baixa
    
    return True

def should_avoid_suspicious_combination(probability, odds):
    """
    Identifica combinações suspeitas que são armadilhas comuns
    """
    # CASO 1: Probabilidade muito baixa com odd muito alta
    if probability < 0.30 and odds > 6.0:
        return True, "Probabilidade muito baixa com odd inflada"
    
    # CASO 2: Discrepância extrema entre prob e odd
    implied_prob = 1 / odds
    discrepancy = abs(probability - implied_prob)
    if discrepancy > 0.25:  # Diferença maior que 25%
        return True, f"Discrepância muito alta: ML {probability:.1%} vs Odd {implied_prob:.1%}"
    
    # CASO 3: Probabilidade borderline com odd extrema
    if 0.25 <= probability <= 0.35 and odds > 7.0:
        return True, "Probabilidade borderline com odd extrema"
    
    return False, ""


def enhanced_ml_recommendation_v2(row, threshold=0.65, min_value=0.05):
    """VERSÃO CORRIGIDA - Com proteção contra odds infladas"""
    
    if pd.isna(row.get('ML_Proba_Home')) or pd.isna(row.get('ML_Proba_Away')) or pd.isna(row.get('ML_Proba_Draw')):
        return "❌ Avoid"
    
    p_home = row['ML_Proba_Home']
    p_draw = row['ML_Proba_Draw'] 
    p_away = row['ML_Proba_Away']
    
    # Calculate expected value for each bet
    ev_home = p_home * row.get('Odd_H', 2.0) - 1
    ev_away = p_away * row.get('Odd_A', 2.0) - 1  
    ev_draw = p_draw * row.get('Odd_D', 3.0) - 1
    ev_1x = (p_home + p_draw) * row.get('Odd_1X', 1.3) - 1
    ev_x2 = (p_away + p_draw) * row.get('Odd_X2', 1.3) - 1
    
    # NOVO: VERIFICAÇÕES DE SEGURANÇA
    recommendations = []
    
    # 1. SINGLE BETS - COM VERIFICAÇÃO DE SEGURANÇA
    if (p_home >= threshold and ev_home >= min_value and 
        is_realistic_odd(p_home, row.get('Odd_H', 2.0))):
        avoid, reason = should_avoid_suspicious_combination(p_home, row.get('Odd_H', 2.0))
        if not avoid:
            recommendations.append(("🟢 Back Home", ev_home, p_home))
    
    if (p_away >= threshold and ev_away >= min_value and 
        is_realistic_odd(p_away, row.get('Odd_A', 2.0))):
        avoid, reason = should_avoid_suspicious_combination(p_away, row.get('Odd_A', 2.0))
        if not avoid:
            recommendations.append(("🟠 Back Away", ev_away, p_away))
    
    if (p_draw >= threshold and ev_draw >= min_value and 
        is_realistic_odd(p_draw, row.get('Odd_D', 3.0))):
        avoid, reason = should_avoid_suspicious_combination(p_draw, row.get('Odd_D', 3.0))
        if not avoid:
            recommendations.append(("⚪ Back Draw", ev_draw, p_draw))
    
    # 2. DOUBLE CHANCE - COM VERIFICAÇÃO MAIS RIGOROSA
    dc_threshold = 0.70
    if (ev_1x >= min_value and (p_home + p_draw) >= dc_threshold and
        is_realistic_odd(p_home + p_draw, row.get('Odd_1X', 1.3), '1x')):
        avoid, reason = should_avoid_suspicious_combination(p_home + p_draw, row.get('Odd_1X', 1.3))
        if not avoid:
            recommendations.append(("🟦 1X (Home/Draw)", ev_1x, p_home + p_draw))
    
    if (ev_x2 >= min_value and (p_away + p_draw) >= dc_threshold and
        is_realistic_odd(p_away + p_draw, row.get('Odd_X2', 1.3), 'x2')):
        avoid, reason = should_avoid_suspicious_combination(p_away + p_draw, row.get('Odd_X2', 1.3))
        if not avoid:
            recommendations.append(("🟪 X2 (Away/Draw)", ev_x2, p_away + p_draw))
    
    # 3. ORDENAR por EV e pegar a melhor
    if recommendations:
        best_rec = max(recommendations, key=lambda x: x[1])  # Melhor EV
        return best_rec[0]
    
    # 4. HIGH EV EXCEPTION - COM VERIFICAÇÃO EXTRA RIGOROSA
    high_ev_threshold = 0.15
    high_ev_bets = []
    
    # Para high EV, exigir probabilidade MÍNIMA e verificação de segurança
    if (ev_home >= high_ev_threshold and p_home >= 0.55 and
        is_realistic_odd(p_home, row.get('Odd_H', 2.0))):
        avoid, reason = should_avoid_suspicious_combination(p_home, row.get('Odd_H', 2.0))
        if not avoid:
            high_ev_bets.append(("🟢 Back Home", ev_home))
    
    if (ev_away >= high_ev_threshold and p_away >= 0.55 and
        is_realistic_odd(p_away, row.get('Odd_A', 2.0))):
        avoid, reason = should_avoid_suspicious_combination(p_away, row.get('Odd_A', 2.0))
        if not avoid:
            high_ev_bets.append(("🟠 Back Away", ev_away))
    
    if (ev_draw >= high_ev_threshold and p_draw >= 0.55 and
        is_realistic_odd(p_draw, row.get('Odd_D', 3.0))):
        avoid, reason = should_avoid_suspicious_combination(p_draw, row.get('Odd_D', 3.0))
        if not avoid:
            high_ev_bets.append(("⚪ Back Draw", ev_draw))
    
    if high_ev_bets:
        return max(high_ev_bets, key=lambda x: x[1])[0]
    
    return "❌ Avoid"



def analyze_league_confidence(history_df):
    """
    Analisa a confiabilidade de cada liga baseado em dados históricos
    """
    if history_df.empty or 'League' not in history_df.columns:
        return pd.DataFrame()
    
    # Análise por liga
    league_stats = history_df.groupby('League').agg({
        'M_H': 'count',                          # Total de jogos
        'Home': lambda x: x.nunique(),           # Times únicos home
        'Away': lambda x: x.nunique(),           # Times únicos away
    }).rename(columns={
        'M_H': 'total_games',
        'Home': 'unique_home_teams', 
        'Away': 'unique_away_teams'
    }).reset_index()
    
    # Calcular maturidade da liga
    def calculate_confidence_level(row):
        total_games = row['total_games']
        avg_teams_per_game = (row['unique_home_teams'] + row['unique_away_teams']) / 2
        
        # CRITÉRIOS DE CONFIABILIDADE
        if total_games >= 100 and avg_teams_per_game >= 15:
            return "🟢 Alta"      # Liga estabelecida + muitos dados
        elif total_games >= 50 and avg_teams_per_game >= 10:
            return "🟡 Média"     # Liga boa mas menos amostras
        elif total_games >= 20 and avg_teams_per_game >= 8:
            return "🔴 Baixa"     # Liga nova/poucos dados
        else:
            return "🔴 Baixa"     # Muito poucos dados
    
    league_stats['League_Confidence'] = league_stats.apply(calculate_confidence_level, axis=1)
    
    return league_stats[['League', 'total_games', 'unique_home_teams', 'unique_away_teams', 'League_Confidence']]


def get_strict_required_features():
    """Lista RIGOROSA de todas as features obrigatórias - ZERO TOLERANCE"""
    return [
        # CORE ABSOLUTO - Sem essas, não tem ML
        'M_H', 'M_A', 'Diff_Power', 'M_Diff',
        'Odd_H', 'Odd_D', 'Odd_A', 'Odd_1X', 'Odd_X2',
        
        # FEATURES AVANÇADAS - Essenciais para o modelo
        'Power_Ratio_Home', 'Power_Ratio_Away', 'Total_Power', 'Power_Diff_Normalized',
        'Fair_Prob_Home', 'Fair_Prob_Away', 'Fair_Prob_Draw', 'Market_Margin',
        'Prob_Ratio_Home_Away', 'Prob_Diff_Home_Away',
        'Home_Advantage', 'Away_Strength', 'Advantage_Ratio',
        
        # FEATURES DE BANDAS
        'Home_Band', 'Away_Band', 'League_Classification'
    ]

def strict_feature_validation(row, required_features):
    """VALIDAÇÃO RIGOROSA - 1 feature faltante = INVALIDO"""
    missing_features = []
    
    for feature in required_features:
        if feature not in row.index:
            missing_features.append(f"{feature} (COLUNA AUSENTE)")
        elif pd.isna(row[feature]):
            missing_features.append(f"{feature} (VALOR NaN)")
        elif row[feature] == '':
            missing_features.append(f"{feature} (VALOR VAZIO)")
    
    return len(missing_features) == 0, missing_features

def get_league_confidence_map(confidence_df):
    """
    Cria um mapa de confiança para fácil acesso
    """
    if confidence_df.empty:
        return {}
    
    return dict(zip(confidence_df['League'], confidence_df['League_Confidence']))


def classify_league_fallback(league_name, history_df):
    """Classificação fallback para ligas não encontradas no histórico"""
    if history_df.empty or 'League' not in history_df.columns:
        return "🔴 Dados Insuficientes"
    
    # Verificar se a liga existe mas tem poucos dados
    league_games = history_df[history_df['League'] == league_name]
    if len(league_games) > 0:
        game_count = len(league_games)
        if game_count >= 50:
            return "🟡 Liga Emergente"
        elif game_count >= 20:
            return "🔴 Liga Poucos Dados"
        else:
            return "🔴 Liga Muito Nova"
    
    # Liga completamente nova (não existe no histórico)
    return "🔴 Liga Inédita"



########################################
####### Bloco 4 – Load Data ############
########################################

# Carregar arquivos disponíveis
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# Selecionar os 2 arquivos mais recentes
options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Carregar os jogos do dia selecionado
try:
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    
    # Apenas jogos sem placar final
    if 'Goals_H_FT' in games_today.columns:
        games_today = games_today[games_today['Goals_H_FT'].isna()].copy()
        
except Exception as e:
    st.error(f"Error loading games file: {e}")
    st.stop()

# Carregar histórico para treinar o modelo
try:
    all_games = load_all_games(GAMES_FOLDER)
    all_games = filter_leagues(all_games)
    history = prepare_history(all_games)

    # Apply data quality improvements
    history = improve_data_quality(history)

    if history.empty:
        st.error("No valid historical data found.")
        st.stop()
        
except Exception as e:
    st.error(f"Error processing historical data: {e}")
    st.stop()

# Extrair data do arquivo selecionado
import re
date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
if date_match:
    selected_date_str = date_match.group(0)
else:
    selected_date_str = datetime.now().strftime("%Y-%m-%d")

st.success(f"✅ Loaded {len(games_today)} games for {selected_date_str}")




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
    try:
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
            st.success("✅ LiveScore data merged successfully!")
        else:
            st.warning(f"Missing columns in LiveScore file: {missing_cols}")
            
    except Exception as e:
        st.error(f"Error loading LiveScore file: {e}")
else:
    st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")





########################################
####### Bloco 5 – Features Engineering ##
########################################

# Apply advanced feature engineering
try:
    # 🔥 NOVO: ANÁLISE DE CONFIABILIDADE DAS LIGAS
    st.info("📊 Analisando confiabilidade das ligas...")

    try:
        # Analisar confiabilidade baseada em dados históricos
        league_confidence_df = analyze_league_confidence(history)
        
        if not league_confidence_df.empty:
            # Mostrar estatísticas das ligas
            high_conf_leagues = league_confidence_df[league_confidence_df['League_Confidence'] == "🟢 Alta"]
            medium_conf_leagues = league_confidence_df[league_confidence_df['League_Confidence'] == "🟡 Média"]
            low_conf_leagues = league_confidence_df[league_confidence_df['League_Confidence'] == "🔴 Baixa"]
            
            st.sidebar.success(f"🎯 Ligas: {len(high_conf_leagues)}🟢 {len(medium_conf_leagues)}🟡 {len(low_conf_leagues)}🔴")
            
            # Criar mapa de confiança para uso posterior
            league_confidence_map = get_league_confidence_map(league_confidence_df)
            
            # Adicionar coluna de confiança aos jogos de hoje
            games_today['League_Confidence'] = games_today['League'].map(league_confidence_map)
            games_today['League_Confidence'] = games_today['League_Confidence'].fillna("🔴 Baixa")
            
        else:
            st.warning("Não foi possível analisar confiabilidade das ligas")
            games_today['League_Confidence'] = "🔴 Baixa"  # Default
            
    except Exception as e:
        st.warning(f"Erro na análise de confiabilidade: {e}")
        games_today['League_Confidence'] = "🔴 Baixa"  # Default em caso de erro

    # CONTINUAÇÃO DO CÓDIGO ORIGINAL (agora dentro do try principal)
    games_today = create_advanced_features(games_today)
    history = create_advanced_features(history)

    # Verificar se as novas features BALANCEADAS foram criadas
    balanced_features = [
        'Power_Ratio_Home', 'Power_Ratio_Away', 'Fair_Prob_Home', 'Fair_Prob_Away',
        'Home_Advantage', 'Away_Strength', 'Prob_Ratio_Home_Away', 'Prob_Diff_Home_Away'
    ]

    created_features = [f for f in balanced_features if f in games_today.columns]
    home_created = len([f for f in created_features if 'Home' in f])
    away_created = len([f for f in created_features if 'Away' in f])

    st.sidebar.info(f"🔧 {home_created}🏠/{away_created}🚌 features balanceadas")

    # Criar features básicas essenciais
    if all(col in games_today.columns for col in ['M_H', 'M_A']):
        games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
        
    if all(col in history.columns for col in ['M_H', 'M_A']):
        history['M_Diff'] = history['M_H'] - history['M_A']
        
    # Calcular odds de dupla chance
    games_today = compute_double_chance_odds(games_today)

except Exception as e:
    st.error(f"Error in feature engineering: {e}")
    st.stop()

# Bandas e classificações de liga
def classify_leagues_variation(history_df):
    """Classifica ligas por variação de performance"""
    if history_df.empty or 'League' not in history_df.columns:
        return pd.DataFrame()
        
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
    """Calcula bandas de performance por liga"""
    if history_df.empty or 'League' not in history_df.columns:
        return pd.DataFrame()
        
    hist = history_df.copy()
    
    if all(col in hist.columns for col in ['M_H', 'M_A']):
        hist['M_Diff'] = hist['M_H'] - hist['M_A']
        
        diff_q = hist.groupby('League')['M_Diff'].quantile([0.20, 0.80]).unstack().rename(
            columns={0.2:'P20_Diff', 0.8:'P80_Diff'}).reset_index()
        home_q = hist.groupby('League')['M_H'].quantile([0.20, 0.80]).unstack().rename(
            columns={0.2:'Home_P20', 0.8:'Home_P80'}).reset_index()
        away_q = hist.groupby('League')['M_A'].quantile([0.20, 0.80]).unstack().rename(
            columns={0.2:'Away_P20', 0.8:'Away_P80'}).reset_index()
            
        out = diff_q.merge(home_q, on='League', how='inner').merge(away_q, on='League', how='inner')
        return out
    else:
        return pd.DataFrame()

# Aplicar classificações de liga
# Aplicar classificações de liga
try:
    league_class = classify_leagues_variation(history)
    league_bands = compute_league_bands(history)
    
    if not league_class.empty:
        games_today = games_today.merge(league_class, on='League', how='left')
        
        # 🔥 CORREÇÃO: Preencher NaN com classificação fallback
        nan_mask = games_today['League_Classification'].isna()
        if nan_mask.any():
            st.warning(f"⚠️ {nan_mask.sum()} ligas sem classificação histórica - aplicando fallback...")
            for idx in games_today[nan_mask].index:
                league_name = games_today.at[idx, 'League']
                fallback_class = classify_league_fallback(league_name, history)
                games_today.at[idx, 'League_Classification'] = fallback_class
    else:
        # Se league_class estiver vazio, classificar TODAS as ligas como fallback
        st.warning("⚠️ Nenhuma classificação de liga disponível - usando fallback para todas")
        games_today['League_Classification'] = games_today['League'].apply(
            lambda x: classify_league_fallback(x, history)
        )
    
    if not league_bands.empty:
        games_today = games_today.merge(league_bands, on='League', how='left')

    # Criar bandas home/away
    if all(col in games_today.columns for col in ['M_H', 'Home_P20', 'Home_P80']):
        games_today['Home_Band'] = np.where(
            games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
            np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
        )
        
    if all(col in games_today.columns for col in ['M_A', 'Away_P20', 'Away_P80']):
        games_today['Away_Band'] = np.where(
            games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
            np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
        )

except Exception as e:
    st.warning(f"Some league features could not be created: {e}")


# 🔍 DIAGNÓSTICO DETALHADO DAS LIGAS
st.header("🔍 Diagnóstico de Classificação de Ligas")

# Verificar ligas problemáticas
unique_leagues_today = games_today['League'].unique()
unique_leagues_history = history['League'].unique() if 'League' in history.columns else []

missing_leagues = set(unique_leagues_today) - set(unique_leagues_history)
new_leagues = set(unique_leagues_today) - set(unique_leagues_history)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ligas Hoje", len(unique_leagues_today))
with col2:
    st.metric("Ligas Histórico", len(unique_leagues_history))
with col3:
    st.metric("Ligas Novas", len(new_leagues))

if new_leagues:
    with st.expander("📋 Lista de Ligas Novas/Inéditas"):
        st.write(f"**{len(new_leagues)} ligas não encontradas no histórico:**")
        for league in sorted(new_leagues):
            st.write(f"• {league}")

# Estatísticas de classificação
if 'League_Classification' in games_today.columns:
    st.subheader("📊 Distribuição de Classificações")
    classification_counts = games_today['League_Classification'].value_counts()
    
    # Mostrar contagens
    for classification, count in classification_counts.items():
        st.write(f"{classification}: {count} ligas")
    
    # Verificar se ainda há NaN
    nan_count = games_today['League_Classification'].isna().sum()
    if nan_count > 0:
        st.error(f"❌ AINDA EXISTEM {nan_count} LIGAS COM NaN!")
        nan_leagues = games_today[games_today['League_Classification'].isna()]['League'].unique()
        st.write("Ligas problemáticas:", list(nan_leagues))
    else:
        st.success("✅ TODAS as ligas foram classificadas com sucesso!")



########################################
####### Bloco 6 – Train ML Model #######
########################################

# Preparar dados históricos para treino
try:
    history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

    def map_result(row):
        """Mapeia resultado para classificação"""
        try:
            if row['Goals_H_FT'] > row['Goals_A_FT']: 
                return "Home"
            elif row['Goals_H_FT'] < row['Goals_A_FT']: 
                return "Away"
            else: 
                return "Draw"
        except:
            return "Draw"

    history['Result'] = history.apply(map_result, axis=1)

    # Enhanced feature set - BALANCEADA
    features_raw = [
        # Features básicas
        'M_H', 'M_A', 'Diff_Power', 'M_Diff',
        
        # Bandas e ligas
        'Home_Band', 'Away_Band', 'League_Classification',
        
        # Odds
        'Odd_H', 'Odd_D', 'Odd_A', 'Odd_1X', 'Odd_X2',
        
        # Novas features BALANCEADAS
        'Power_Ratio_Home', 'Power_Ratio_Away', 'Total_Power', 'Power_Diff_Normalized',
        'Fair_Prob_Home', 'Fair_Prob_Away', 'Fair_Prob_Draw', 'Market_Margin',
        'Prob_Ratio_Home_Away', 'Prob_Diff_Home_Away',
        'Home_Advantage', 'Away_Strength', 'Advantage_Ratio'
    ]

    # Só incluir features que existem no dataframe
    features_raw = [f for f in features_raw if f in history.columns]

    # Verificar se temos features balanceadas
    home_features = [f for f in features_raw if 'Home' in f]
    away_features = [f for f in features_raw if 'Away' in f]
    neutral_features = [f for f in features_raw if 'Home' not in f and 'Away' not in f]

    st.sidebar.info(f"🏠 {len(home_features)} Home | 🚌 {len(away_features)} Away | ⚖️ {len(neutral_features)} Neutral")

    X = history[features_raw].copy()
    y = history['Result']

    # Clean the data before training
    X = clean_dataframe(X)

    # Mapear bandas para números
    BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
    if 'Home_Band' in X: 
        X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
        X = X.drop('Home_Band', axis=1)
    if 'Away_Band' in X: 
        X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)
        X = X.drop('Away_Band', axis=1)

    # One-hot encoding para variáveis categóricas
    cat_cols = [c for c in ['League_Classification'] if c in X]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_cols:
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                       encoded_df.reset_index(drop=True)], axis=1)

    # Treinar modelo Random Forest
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

except Exception as e:
    st.error(f"Error training ML model: {e}")
    st.stop()




########################################
####### Bloco 7 – Apply ML to Today ####
########################################

# 🔥 VALIDAÇÃO ZERO TOLERANCE - 1 FEATURE FALTANTE = AVOID
st.info("🔍 Validando integridade dos dados para ML...")

required_features = get_strict_required_features()

# Inicializar todas as recomendações como AVOID por padrão
games_today["ML_Data_Valid"] = False
games_today["Missing_Features"] = ""
games_today["ML_Recommendation"] = "❌ Avoid"
games_today["ML_Proba_Home"] = np.nan
games_today["ML_Proba_Draw"] = np.nan  
games_today["ML_Proba_Away"] = np.nan

# Validar CADA jogo individualmente
valid_indices = []

for idx, row in games_today.iterrows():
    is_valid, missing = strict_feature_validation(row, required_features)
    
    if is_valid:
        games_today.at[idx, "ML_Data_Valid"] = True
        games_today.at[idx, "Missing_Features"] = "✅ COMPLETE"
        valid_indices.append(idx)
    else:
        games_today.at[idx, "ML_Data_Valid"] = False
        games_today.at[idx, "Missing_Features"] = ", ".join(missing)
        # Já está como "❌ Avoid" por padrão

# Mostrar estatísticas de validação
valid_count = len(valid_indices)
invalid_count = len(games_today) - valid_count

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Jogos", len(games_today))
with col2:
    st.metric("✅ Válidos para ML", valid_count)
with col3:
    st.metric("❌ Inválidos", invalid_count)

# MOSTRAR DETALHES DOS JOGOS INVALIDOS
invalid_games = games_today[~games_today["ML_Data_Valid"]]
if not invalid_games.empty:
    with st.expander("🚨 JOGOS EXCLUÍDOS - Faltam Features", expanded=True):
        st.warning(f"{len(invalid_games)} jogos excluídos por dados incompletos:")
        display_invalid = invalid_games[['Home', 'Away', 'League', 'Missing_Features']].copy()
        display_invalid['Missing_Count'] = display_invalid['Missing_Features'].apply(lambda x: len(x.split(',')) if x != "✅ COMPLETE" else 0)
        display_invalid = display_invalid.sort_values('Missing_Count', ascending=False)
        st.dataframe(display_invalid, use_container_width=True)

# ⚠️ SE NENHUM JOGO VÁLIDO, PARAR AQUI
if len(valid_indices) == 0:
    st.error("🚫 CRÍTICO: NENHUM jogo possui todas as features necessárias para ML!")
    st.error("Verifique a qualidade dos dados nos arquivos CSV.")
    st.stop()

st.success(f"🎯 {valid_count} jogos validados para processamento ML")

# CONFIGURAÇÃO DE THRESHOLD (mantido do código original)
threshold_option = st.sidebar.selectbox(
    "Threshold Strategy",
    ["Auto-Adjust", "Dynamic", "Fixed"],
    index=0
)

if threshold_option == "Auto-Adjust":
    target_recs = st.sidebar.slider("Target Recommendations", 3, 15, 8)
    threshold = auto_adjust_threshold(games_today, target_recs)
elif threshold_option == "Dynamic":
    threshold = dynamic_threshold_adjustment(games_today)
else:
    threshold = st.sidebar.slider("Fixed Threshold", 0.50, 0.80, 0.65)

min_ev_value = st.sidebar.slider("Min EV Value", 0.00, 0.10, 0.02, 0.01)
st.sidebar.metric("🎯 ML Threshold", f"{threshold:.1%}")

# CONTINUAR APENAS COM JOGOS VÁLIDOS
try:
    # Preparar dados VÁLIDOS para predição
    X_today_valid = games_today.loc[valid_indices][features_raw].copy()
    
    # Clean today's data
    X_today_valid = clean_dataframe(X_today_valid)

    # Apply the same transformations as training data
    if 'Home_Band' in X_today_valid: 
        X_today_valid['Home_Band_Num'] = X_today_valid['Home_Band'].map(BAND_MAP)
        X_today_valid = X_today_valid.drop('Home_Band', axis=1)
    if 'Away_Band' in X_today_valid: 
        X_today_valid['Away_Band_Num'] = X_today_valid['Away_Band'].map(BAND_MAP)
        X_today_valid = X_today_valid.drop('Away_Band', axis=1)

    if 'cat_cols' in locals() and cat_cols:
        try:
            encoded_today = encoder.transform(X_today_valid[cat_cols])
            encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
            X_today_valid = pd.concat([X_today_valid.drop(columns=cat_cols).reset_index(drop=True),
                                     encoded_today_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            st.warning(f"Encoding error: {e}")

    # Ensure same columns as training data
    missing_cols = set(X.columns) - set(X_today_valid.columns)
    for col in missing_cols:
        X_today_valid[col] = 0
    X_today_valid = X_today_valid[X.columns]

    # Aplicar modelo APENAS nos jogos válidos
    st.info("🤖 Aplicando modelo ML nos jogos válidos...")
    ml_proba = model.predict_proba(X_today_valid)
    
    # Preencher probabilidades APENAS para jogos válidos
    games_today.loc[valid_indices, "ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
    games_today.loc[valid_indices, "ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
    games_today.loc[valid_indices, "ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]
    
    # Gerar recomendações ML APENAS para jogos válidos
    recommendation_count = 0
    for idx in valid_indices:
        recommendation = enhanced_ml_recommendation_v2(
            games_today.loc[idx], threshold, min_ev_value
        )
        games_today.at[idx, "ML_Recommendation"] = recommendation
        if recommendation != "❌ Avoid":
            recommendation_count += 1
    
    st.success(f"✅ ML aplicado com sucesso! {recommendation_count} recomendações geradas")
        
except Exception as e:
    st.error(f"Erro ao aplicar ML: {e}")
    # Manter todos como "Avoid" em caso de erro
    games_today["ML_Recommendation"] = "❌ Avoid"



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
    """Calcula stake usando Kelly Criterion"""
    if pd.isna(probability) or pd.isna(odds) or odds <= 1 or probability <= 0: 
        return 0
    edge = probability * odds - 1
    if edge <= 0: 
        return 0
    full_kelly_fraction = edge / (odds - 1)
    fractional_kelly = full_kelly_fraction * kelly_fraction
    recommended_stake = fractional_kelly * bankroll
    if recommended_stake < min_stake: 
        return 0
    elif recommended_stake > max_stake: 
        return max_stake
    else: 
        return round(recommended_stake, 2)

def get_kelly_stake_ml(row):
    """Aplica Kelly Criterion baseado na recomendação ML"""
    rec = row['ML_Recommendation']
    if pd.isna(rec) or rec == '❌ Avoid': 
        return 0
    
    if 'Back Home' in rec: 
        return kelly_stake(row['ML_Proba_Home'], row['Odd_H'], bankroll, kelly_fraction, min_stake, max_stake)
    elif 'Back Away' in rec: 
        return kelly_stake(row['ML_Proba_Away'], row['Odd_A'], bankroll, kelly_fraction, min_stake, max_stake)
    elif 'Back Draw' in rec: 
        return kelly_stake(row['ML_Proba_Draw'], row['Odd_D'], bankroll, kelly_fraction, min_stake, max_stake)
    elif '1X' in rec: 
        return kelly_stake(row['ML_Proba_Home'] + row['ML_Proba_Draw'], row['Odd_1X'], bankroll, kelly_fraction, min_stake, max_stake)
    elif 'X2' in rec: 
        return kelly_stake(row['ML_Proba_Away'] + row['ML_Proba_Draw'], row['Odd_X2'], bankroll, kelly_fraction, min_stake, max_stake)
    return 0

# Aplicar Kelly Criterion
games_today['Kelly_Stake_ML'] = games_today.apply(get_kelly_stake_ml, axis=1)




########################################
##### Bloco 9 – Result Tracking ########
########################################

def determine_result(row):
    """Determina resultado baseado nos gols"""
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
    """Verifica se recomendação estava correta"""
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
    """Calcula profit considerando Kelly stake"""
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
    """Calcula odds e probabilidade do parlay"""
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
        else:
            continue
            
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
    """Gera sugestões de parlays APENAS com jogos válidos para ML"""
    
    # 🔥 FILTRAR APENAS JOGOS COM ML VÁLIDO (não é "❌ Avoid")
    valid_ml_games = games_df[games_df['ML_Recommendation'] != '❌ Avoid'].copy()
    
    if len(valid_ml_games) == 0:
        st.warning("⚠️ Nenhum jogo válido para parlays - todos estão como '❌ Avoid'")
        return []
    
    st.info(f"🎯 {len(valid_ml_games)} jogos válidos disponíveis para parlays")
    
    eligible_games = []
    
    for idx, row in valid_ml_games.iterrows():
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
            
            # 🔥 CORREÇÃO: Usar valid_ml_games em vez de games_today_filtered
            if weekend_filter and len(valid_ml_games) > 15:
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
        # 🔥 CORREÇÃO: Usar valid_ml_games em vez de games_today_filtered
        st.warning(f"⚡ Limite ativado: {len(eligible_games)} jogos elegíveis (de {len(valid_ml_games)} totais)")
    
    st.info(f"🎯 Jogos elegíveis para parlays: {len(eligible_games)}")
    
    # CALCULAR QUANTIDADE MÁXIMA DE COMBINAÇÕES
    total_combinations = 0
    for legs in range(min_legs, max_legs + 1):
        if len(eligible_games) >= legs:
            total_combinations += math.comb(len(eligible_games), legs)
    
    st.info(f"🧮 Combinações possíveis: {total_combinations:,}")
    
    # AVISO SE MUITAS COMBINAÇÕES
    if total_combinations > 1000:
        st.warning("⚠️ Muitas combinações! Use filtros mais rigorosos.")
        return []
    
    parlay_suggestions = []
    
    # GERAR PARLAYS POR NÚMERO DE LEGS
    for num_legs in range(min_legs, max_legs + 1):
        if len(eligible_games) < num_legs:
            continue
            
        # Definir thresholds baseado no número de legs
        if num_legs == 2:
            ev_threshold = 0.03 if weekend_filter and len(eligible_games) > 15 else 0.03
            prob_threshold = 0.20 if weekend_filter and len(eligible_games) > 15 else 0.20
            stake_multiplier = 0.08
        elif num_legs == 3:
            ev_threshold = 0.01 if weekend_filter and len(eligible_games) > 15 else 0.01
            prob_threshold = 0.15 if weekend_filter and len(eligible_games) > 15 else 0.15
            stake_multiplier = 0.05
        else:  # 4 legs
            ev_threshold = 0.005
            prob_threshold = 0.10
            stake_multiplier = 0.03
        
        # Gerar combinações
        for combo in itertools.combinations(eligible_games, num_legs):
            games_list = [(game[0], game[1]) for game in combo]
            # 🔥 CORREÇÃO: Usar games_df (parâmetro original) em vez de games_today_filtered
            prob, odds, ev, details = calculate_parlay_odds(games_list, games_df)
            
            if ev > ev_threshold and prob > prob_threshold:
                stake = min(parlay_bankroll * stake_multiplier, parlay_bankroll * (stake_multiplier + 0.04) * prob)
                stake = round(stake, 2)
                
                min_stake_required = 2 if num_legs == 4 else (3 if num_legs == 3 else 5)
                if stake >= min_stake_required:
                    parlay_suggestions.append({
                        'type': f'{num_legs}-Leg Parlay',
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
    """Calcula estatísticas de performance do ML"""
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

# Calcular estatísticas
try:
    summary_ml = summary_stats_ml(finished_games)
    track_model_performance(games_today, finished_games)
except Exception as e:
    st.error(f"Error calculating performance stats: {e}")
    summary_ml = {}




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
try:
    super_parlay = generate_super_parlay(games_today, target_super_odds)
except Exception as e:
    st.error(f"Error generating super parlay: {e}")
    super_parlay = None





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

# HEADER PRINCIPAL
st.header("📈 Day's Summary - Machine Learning Performance")
if summary_ml:
    st.json(summary_ml)
else:
    st.info("No finished games to display performance stats")

st.header("🎯 Machine Learning Recommendations")

# COLUNAS PARA DISPLAY - ATUALIZADO COM CONFIABILIDADE
cols_to_show = [
    'Date', 'Time', 'League', 'League_Confidence', 'Home', 'Away',  # ← ADICIONADO League_Confidence
    'Goals_H_Today', 'Goals_A_Today', 'ML_Recommendation', 
    'ML_Data_Valid', 'ML_Correct', 'Kelly_Stake_ML',
    'Profit_ML_Fixed', 'Profit_ML_Kelly',
    'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away', 
    'Odd_H', 'Odd_D', 'Odd_A'
]

available_cols = [c for c in cols_to_show if c in games_today.columns]

# Função para formatação condicional - ATUALIZADA
# def highlight_confidence_rows(row):
#     styles = [''] * len(row)
    
#     # Destaque por confiança da liga
#     if 'League_Confidence' in row.index:
#         confidence = row['League_Confidence']
#         if confidence == "🟢 Alta":
#             styles = ['background-color: #e6f7e6'] * len(row)  # Verde claro
#         elif confidence == "🟡 Média":
#             styles = ['background-color: #fff9e6'] * len(row)  # Amarelo claro
#         elif confidence == "🔴 Baixa":
#             styles = ['background-color: #ffe6e6'] * len(row)  # Vermelho claro
    
#     # Destaque para dados inválidos (sobrescreve confiança)
#     if 'ML_Data_Valid' in row.index and row['ML_Data_Valid'] == False:
#         styles = ['background-color: #ffcccc'] * len(row)  # Vermelho forte
    
#     return styles

# Display dos dados - CORRIGIDO (fechando o parêntese do apply)
try:
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
        use_container_width=True,
        height=600
    )
except Exception as e:
    st.error(f"Error displaying recommendations: {e}")

# PARLAY RECOMMENDATIONS
st.header("🎰 Auto Parlay Recommendations")

if 'parlay_suggestions' in locals() and parlay_suggestions:
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
