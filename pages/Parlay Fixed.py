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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import re

########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(page_title="Today's Picks - ML + Parlay System", layout="wide")
st.title("🤖 ML Betting System + Auto Parlay Recommendations")

# Configurações principais
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "coppa", "afc","trophy"]
DOMINANT_THRESHOLD = 0.90

# 🔥 CORREÇÃO: Definir BAND_MAP que estava faltando
BAND_MAP = {
    'Band 1': 1, 'Band 2': 2, 'Band 3': 3, 'Band 4': 4, 'Band 5': 5,
    'Band 6': 6, 'Band 7': 7, 'Band 8': 8, 'Band 9': 9, 'Band 10': 10
}

########################################
####### Bloco 3 – Helper Functions #####
########################################

# 🟢🟢🟢 FUNÇÃO ASIAN LINE CORRIGIDA - DO SCRIPT MODELO 🟢🟢🟢
def convert_asian_line_to_decimal(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).

    Regras oficiais e consistentes com Pinnacle/Bet365:
      '0/0.5'   -> +0.25  (para away) → invertido: -0.25 (para home)
      '-0.5/0'  -> -0.25  (para away) → invertido: +0.25 (para home)
      '-1/1.5'  -> -1.25  → +1.25
      '1/1.5'   -> +1.25  → -1.25
      '1.5'     -> +1.50  → -1.50
      '0'       ->  0.00  →  0.00

    Retorna: float ou None se inválido
    """
    if pd.isna(line_str) or line_str == "":
        return None

    line_str = str(line_str).strip()

    # Caso especial: linha zero
    if line_str == "0" or line_str == "0.0":
        return 0.0

    # Caso simples — número único
    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num  # ✅ CORREÇÃO: Inverte sinal (Away → Home)
        except ValueError:
            return None

    # Caso duplo — média dos dois lados com tratamento de sinal
    try:
        parts = [float(p) for p in line_str.split("/")]
        
        # Calcula média mantendo a lógica de sinal
        avg = sum(parts) / len(parts)
        
        # Determina o sinal base baseado no primeiro elemento
        first_part = parts[0]
        if first_part < 0:
            result = -abs(avg)
        else:
            result = abs(avg)
            
        # ✅ CORREÇÃO CRÍTICA: Inverte o sinal no final (Away → Home)
        return -result
        
    except (ValueError, TypeError):
        return None

def calc_handicap_result(margin, asian_line_decimal, invert=False):
    """Retorna média de pontos por linha (1 win, 0.5 push, 0 loss)"""
    if pd.isna(asian_line_decimal) or pd.isna(margin):
        return np.nan
    
    if invert:
        margin = -margin
    
    # ✅ AGORA usa o decimal já convertido corretamente
    line = asian_line_decimal
    
    if margin > line:
        return 1.0
    elif margin == line:
        return 0.5
    else:
        return 0.0

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
import re

# Buscar arquivos de jogos
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# 🔥 Exibir até os últimos 5 dias (mantendo ordem cronológica)
options = files[-5:] if len(files) >= 5 else files
selected_file = st.selectbox("Select Matchday File (up to last 5 days):", options, index=len(options)-1)

# Carregar os jogos do dia selecionado
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# 🟢🟢🟢 ADICIONAR CONVERSÃO ASIAN LINE CORRETA 🟢🟢🟢
if 'Asian_Line' in games_today.columns:
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    st.success(f"✅ Asian Line convertida: {len(games_today[games_today['Asian_Line_Decimal'].notna()])} jogos com handicap válido")
else:
    st.warning("⚠️ Coluna 'Asian_Line' não encontrada nos dados de hoje")

########################################
### 🔒 PROTEÇÃO ANTI-LEAK – GOALS SAFE ###
########################################
# Garantir que a ML NUNCA veja gols do dia atual
goal_cols = [c for c in games_today.columns if 'Goal' in c or 'Goals_' in c]

if goal_cols:
    # Cópia de segurança apenas para exibição posterior
    goals_snapshot = games_today[goal_cols + ['Home', 'Away']].copy()
    # Remover colunas de gols antes de qualquer uso pela ML
    games_today = games_today.drop(columns=goal_cols, errors='ignore')
    # Recriar colunas vazias para compatibilidade
    for c in goal_cols:
        games_today[c] = np.nan

# Carregar histórico completo (para treino)
all_games = load_all_games(GAMES_FOLDER)
all_games = filter_leagues(all_games)
history = prepare_history(all_games)

# 🟢🟢🟢 ADICIONAR CONVERSÃO ASIAN LINE NO HISTÓRICO 🟢🟢🟢
if 'Asian_Line' in history.columns:
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    history = history.dropna(subset=['Asian_Line_Decimal'])
    st.success(f"✅ Histórico com Asian Line válida: {len(history)} jogos")
else:
    st.warning("⚠️ Coluna 'Asian_Line' não encontrada no histórico")

# ✅ Extrair data do arquivo selecionado
date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
if date_match:
    selected_date_str = date_match.group(0)
    selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")
else:
    selected_date_str = datetime.now().strftime("%Y-%m-%d")
    selected_date = datetime.now()

# 🔒 Garantir que o histórico não contenha jogos do dia selecionado
if 'Date' in history.columns:
    history = history[pd.to_datetime(history['Date'], errors='coerce') < selected_date]

if history.empty:
    st.error("No valid historical data found.")
    st.stop()

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
    
    required_cols = ['Id', 'status', 'home_goal', 'away_goal']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if not missing_cols:
        games_today = games_today.merge(
            results_df,
            left_on='Id',
            right_on='Id',
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
####### Bloco 5 – Feature Engineering 3D #
########################################

def calcular_distancias_3d(df):
    df = df.copy()
    for c in ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']:
        if c not in df.columns:
            df[c] = 0.0
    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    a_xy = np.arctan2(dy, dx)
    a_xz = np.arctan2(dz, dx)
    a_yz = np.arctan2(dz, np.where(dy==0, 1e-9, dy))
    df['Quadrant_Sin_XY'] = np.sin(a_xy); df['Quadrant_Cos_XY'] = np.cos(a_xy)
    df['Quadrant_Sin_XZ'] = np.sin(a_xz); df['Quadrant_Cos_XZ'] = np.cos(a_xz)
    df['Quadrant_Sin_YZ'] = np.sin(a_yz); df['Quadrant_Cos_YZ'] = np.cos(a_yz)
    combo = a_xy + a_xz + a_yz
    df['Quadrant_Sin_Combo'] = np.sin(combo); df['Quadrant_Cos_Combo'] = np.cos(combo)
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    return df

def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    df = df.copy()
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']
    Xc = df[['dx','dy','dz']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df['Cluster3D_Label'] = kmeans.fit_predict(Xc)
    return df

def ensure_3d_features(df):
    df = calcular_distancias_3d(df)
    df = aplicar_clusterizacao_3d(df, n_clusters=5)
    return df

# =====================================================
# 🧩 Aplicação às bases
# =====================================================
history = ensure_3d_features(history)
games_today = ensure_3d_features(games_today)

# =====================================================
# 🧮 CORREÇÃO: Garantia de odds 1X2 e Double Chance no histórico
# =====================================================

if all(c in history.columns for c in ['Odd_H','Odd_D','Odd_A']):
    # CORREÇÃO: Criar DataFrame temporário para as probabilidades
    probs = pd.DataFrame()
    probs['p_H'] = 1 / history['Odd_H']
    probs['p_D'] = 1 / history['Odd_D'] 
    probs['p_A'] = 1 / history['Odd_A']
    
    # CORREÇÃO: Normalizar apenas as colunas de probabilidade
    sum_probs = probs.sum(axis=1)
    probs_normalized = probs.div(sum_probs, axis=0)
    
    # Atribuir de volta ao history
    history['p_H'] = probs_normalized['p_H']
    history['p_D'] = probs_normalized['p_D']
    history['p_A'] = probs_normalized['p_A']
    
    # Calcular odds double chance
    history['Odd_1X'] = 1 / (history['p_H'] + history['p_D'])
    history['Odd_X2'] = 1 / (history['p_A'] + history['p_D'])
else:
    # fallback: odds fictícias (para não quebrar)
    st.warning("⚠️ Colunas de odds não encontradas no histórico. Usando valores padrão.")
    history['Odd_1X'] = 2.0
    history['Odd_X2'] = 2.0
    history['p_H'] = 0.33
    history['p_D'] = 0.33  
    history['p_A'] = 0.34

# =====================================================
# 🧩 Garantia de features 3D (fallback)
# =====================================================
expected_cols = [
    'Quadrant_Dist_3D','Quadrant_Separation_3D','Magnitude_3D',
    'Quadrant_Sin_XY','Quadrant_Cos_XY',
    'Quadrant_Sin_XZ','Quadrant_Cos_XZ',
    'Quadrant_Sin_YZ','Quadrant_Cos_YZ',
    'Quadrant_Sin_Combo','Quadrant_Cos_Combo',
    'Vector_Sign','Cluster3D_Label'
]
for c in expected_cols:
    if c not in history.columns:
        history[c] = 0.0
    if c not in games_today.columns:
        games_today[c] = 0.0

# Odds continuam sendo usadas em partes do sistema (Kelly/Parlay),
# então mantemos as colunas necessárias
games_today = compute_double_chance_odds(games_today)

########################################
####### Bloco 6 – Train ML Model 3D ####
########################################

# Checkbox para incluir/excluir odds como features
use_odds_features = st.sidebar.checkbox("📊 Incluir features de Odds no treino", value=True)

# Define resultado categórico (target)
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])
def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']: return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']: return "Away"
    else: return "Draw"
history['Result'] = history.apply(map_result, axis=1)

# =============================
# 🎯 Seleção de features - APENAS 3D + OPCIONALMENTE ODDS
# =============================
features_3d = [
    'Quadrant_Dist_3D','Quadrant_Separation_3D','Magnitude_3D',
    'Quadrant_Sin_XY','Quadrant_Cos_XY',
    'Quadrant_Sin_XZ','Quadrant_Cos_XZ',
    'Quadrant_Sin_YZ','Quadrant_Cos_YZ',
    'Quadrant_Sin_Combo','Quadrant_Cos_Combo',
    'Vector_Sign','Cluster3D_Label'
]

features_odds = ['Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2']

# 🔥 CORREÇÃO: Combinação final de features - APENAS 3D + OPCIONALMENTE ODDS
features_raw = features_3d + (features_odds if use_odds_features else [])

st.info(f"🎯 Features usadas no modelo: {len(features_raw)} features (3D {'+ Odds' if use_odds_features else 'apenas'})")

# 🔍 Diagnóstico de colunas ausentes
missing_cols = [c for c in features_raw if c not in history.columns]
if missing_cols:
    st.error(f"🚨 As seguintes colunas 3D não existem no history: {missing_cols}")
    st.stop()
else:
    st.success("✅ Todas as colunas de features 3D estão presentes no histórico.")

# Preparo do dataset
X = history[features_raw].copy()
y = history['Result']

# One-hot encoding para o Cluster3D_Label
cat_cols = ['Cluster3D_Label']
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(X[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
               encoded_df.reset_index(drop=True)], axis=1)

# 🤖 Treinamento do modelo
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
st.success("✅ Modelo ML 3D treinado com sucesso (RandomForestClassifier)")

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

# Função para verificar dados faltantes - APENAS FEATURES 3D
def check_missing_features(row, features_required):
    """Verifica se há dados faltantes nas features 3D essenciais"""
    missing_features = []
    
    for feature in features_required:
        if feature in row:
            if pd.isna(row[feature]) or row[feature] == '':
                missing_features.append(feature)
        else:
            missing_features.append(feature)
    
    return missing_features

# 🔥 CORREÇÃO: Lista de features 3D obrigatórias (APENAS 3D)
required_features_3d = [
    'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Magnitude_3D',
    'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
    'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ', 
    'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
    'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
    'Vector_Sign', 'Cluster3D_Label'
]

X_today = games_today[features_raw].copy()

# Aplicar validação de dados faltantes - APENAS FEATURES 3D
games_today["ML_Data_Valid"] = True
games_today["Missing_Features"] = ""

for idx, row in games_today.iterrows():
    missing = check_missing_features(row, required_features_3d)  # 🔥 Usando apenas 3D
    if missing:
        games_today.at[idx, "ML_Data_Valid"] = False
        games_today.at[idx, "Missing_Features"] = ", ".join(missing)

# Aplicar o modelo apenas aos jogos com dados completos NAS FEATURES 3D
valid_games_mask = games_today["ML_Data_Valid"] == True
X_today_valid = X_today[valid_games_mask].copy()

st.info(f"🎯 Jogos com features 3D completas: {valid_games_mask.sum()} de {len(games_today)}")

# 🔥 CORREÇÃO: Preencher valores NaN antes do encoding
if not X_today_valid.empty:
    # Preencher NaN nas colunas numéricas
    numeric_cols = X_today_valid.select_dtypes(include=[np.number]).columns
    X_today_valid[numeric_cols] = X_today_valid[numeric_cols].fillna(0)
    
    # 🔥 CORREÇÃO: Garantir que cat_cols exista e tenha dados válidos
    if cat_cols:
        # Preencher NaN nas colunas categóricas do encoder
        for col in cat_cols:
            if col in X_today_valid.columns:
                X_today_valid[col] = X_today_valid[col].fillna(-1)
        
        try:
            # Verificar se há dados para transformar
            if not X_today_valid[cat_cols].empty:
                encoded_today = encoder.transform(X_today_valid[cat_cols])
                encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
                X_today_valid = pd.concat([X_today_valid.drop(columns=cat_cols).reset_index(drop=True),
                                         encoded_today_df.reset_index(drop=True)], axis=1)
            else:
                st.warning("⚠️ Nenhum dado válido para encoding categórico")
                # Criar colunas vazias para manter a estrutura
                encoded_cols = encoder.get_feature_names_out(cat_cols)
                for col in encoded_cols:
                    X_today_valid[col] = 0
        except Exception as e:
            st.error(f"❌ Erro no encoding categórico: {e}")
            # Fallback: criar colunas vazias
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            for col in encoded_cols:
                X_today_valid[col] = 0
    else:
        st.warning("⚠️ Nenhuma coluna categórica definida para encoding")

# Inicializar colunas de probabilidade com NaN
games_today["ML_Proba_Home"] = np.nan
games_today["ML_Proba_Draw"] = np.nan
games_today["ML_Proba_Away"] = np.nan
games_today["ML_Recommendation"] = "❌ Avoid"

# Aplicar modelo apenas nos jogos válidos
if not X_today_valid.empty:
    try:
        # 🔥 CORREÇÃO: Garantir que as features estejam na mesma ordem do treino
        # Obter as features do modelo treinado
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X.columns
        
        # Reordenar e preencher features faltantes
        for feature in expected_features:
            if feature not in X_today_valid.columns:
                X_today_valid[feature] = 0
        
        # Manter apenas as features esperadas pelo modelo
        X_today_valid = X_today_valid[expected_features]
        
        # Preencher quaisquer valores NaN restantes
        X_today_valid = X_today_valid.fillna(0)
        
        ml_proba = model.predict_proba(X_today_valid)
        
        # Preencher apenas os jogos válidos
        valid_indices = games_today[valid_games_mask].index
        
        games_today.loc[valid_indices, "ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
        games_today.loc[valid_indices, "ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
        games_today.loc[valid_indices, "ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]
        
        # Gerar recomendações apenas para jogos válidos
        for idx in valid_indices:
            p_home = games_today.at[idx, "ML_Proba_Home"]
            p_draw = games_today.at[idx, "ML_Proba_Draw"] 
            p_away = games_today.at[idx, "ML_Proba_Away"]
            
            games_today.at[idx, "ML_Recommendation"] = ml_recommendation_from_proba(
                p_home, p_draw, p_away, threshold
            )
            
        st.success(f"✅ Previsões ML 3D aplicadas em {len(valid_indices)} jogos válidos")
        
    except Exception as e:
        st.error(f"❌ Erro ao aplicar modelo ML 3D: {e}")
        st.error("Verifique se as features 3D do dia atual correspondem às do treino")
else:
    st.warning("⚠️ Nenhum jogo com features 3D válidas para previsão")

# Mostrar estatísticas de validação
invalid_count = len(games_today) - valid_games_mask.sum()
if invalid_count > 0:
    st.warning(f"⚠️ {invalid_count} jogos excluídos por features 3D incompletas")
    
    # Mostrar detalhes dos jogos com problemas
    invalid_games = games_today[~valid_games_mask]
    if not invalid_games.empty:
        with st.expander("📋 Ver jogos com features 3D incompletas"):
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
    """Determina o resultado real (Home/Away/Draw) com base nos gols de hoje"""
    try:
        gh = float(row.get('Goals_H_Today', np.nan))
        ga = float(row.get('Goals_A_Today', np.nan))
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

# 🟢🟢🟢 NOVA FUNÇÃO: Determinar resultado do handicap asiático 🟢🟢🟢
def determine_handicap_result(row):
    """Determina se o HOME cobriu o handicap asiático"""
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
        asian_line_decimal = row.get('Asian_Line_Decimal')
    except (ValueError, TypeError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line_decimal):
        return None

    margin = gh - ga
    # ✅ AGORA usa a função CORRIGIDA com Asian_Line_Decimal já convertido
    handicap_result = calc_handicap_result(margin, asian_line_decimal, invert=False)

    if handicap_result > 0.5:
        return "HOME_COVERED"
    elif handicap_result == 0.5:
        return "PUSH"
    else:
        return "HOME_NOT_COVERED"

games_today['Result_Today'] = games_today.apply(determine_result, axis=1)

# 🟢🟢🟢 ADICIONAR RESULTADO DO HANDICAP 🟢🟢🟢
games_today['Handicap_Result'] = games_today.apply(determine_handicap_result, axis=1)

def check_recommendation(rec, result):
    """Verifica se a recomendação da ML bateu com o resultado real"""
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

# 🟢🟢🟢 NOVA FUNÇÃO: Verificar recomendação do handicap 🟢🟢🟢
def check_handicap_recommendation(rec, handicap_result):
    """Verifica se a recomendação estava correta para handicap"""
    if pd.isna(rec) or handicap_result is None or rec == '❌ Avoid':
        return None

    rec = str(rec)

    if any(keyword in rec for keyword in ['HOME', 'Home', 'VALUE NO HOME', 'FAVORITO HOME']):
        return handicap_result == "HOME_COVERED"
    elif any(keyword in rec for keyword in ['AWAY', 'Away', 'VALUE NO AWAY', 'FAVORITO AWAY', 'MODELO CONFIA AWAY']):
        return handicap_result in ["HOME_NOT_COVERED", "PUSH"]

    return None

games_today['ML_Correct'] = games_today.apply(
    lambda r: check_recommendation(r['ML_Recommendation'], r['Result_Today']),
    axis=1
)

# 🟢🟢🟢 ADICIONAR VERIFICAÇÃO DO HANDICAP 🟢🟢🟢
games_today['Handicap_Correct'] = games_today.apply(
    lambda r: check_handicap_recommendation(r['ML_Recommendation'], r['Handicap_Result']),
    axis=1
)

def calculate_profit(rec, result, odds_row):
    """Lucro fixo (stake = 1 unidade)"""
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
    """Lucro ajustado pelo critério de Kelly"""
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return 0, 0
    
    rec = str(rec)
    stake_fixed = 1

    # 🔥 CORREÇÃO: Estrutura condicional completa
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

# Calcular lucros apenas se houver jogos válidos
if not games_today.empty:
    games_today['Profit_ML_Fixed'] = games_today.apply(
        lambda r: calculate_profit(
            r['ML_Recommendation'], r['Result_Today'], r
        ),
        axis=1
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
else:
    st.warning("⚠️ Nenhum jogo válido encontrado para este dia (todos finalizados ou arquivo vazio).")
    games_today['Profit_ML_Fixed'] = np.nan
    games_today['Profit_ML_Kelly'] = np.nan

########################################
#### Bloco 10 – Auto Parlay System #####
########################################

# SEÇÃO 2: PARÂMETROS PARLAY
st.sidebar.header("🎰 Parlay System")

parlay_bankroll = st.sidebar.number_input("Parlay Bankroll", 50, 5000, 200, 50, help="Bankroll separado para parlays")
min_parlay_prob = st.sidebar.slider("Min Probability Parlay", 0.50, 0.70, 0.50, 0.01, help="Probabilidade mínima para considerar jogo no parlay")
max_parlay_suggestions = st.sidebar.slider("Max Parlay Suggestions", 1, 10, 5, 1, help="Número máximo de sugestões de parlay")

# 🔥 NOVO: CONTROLE DE LEGS
st.sidebar.markdown("---")
min_parlay_legs = st.sidebar.slider("Min Legs", 2, 4, 2, 1, help="Número mínimo de jogos no parlay")
max_parlay_legs = st.sidebar.slider("Max Legs", 2, 4, 4, 1, help="Número máximo de jogos no parlay")

# 🔥 NOVO: FILTROS PARA FINS DE SEMANA
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
    
    for Idx, bet_type in games_list:
        game = games_df.loc[Idx]
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

########################################
####### Bloco 10B – Parlay Validator ###
########################################

def check_bet_hit(result, bet_type):
    """Retorna True, False ou None dependendo se a aposta bateu"""
    if result not in ["Home", "Away", "Draw"]:
        return None
    if bet_type == "Home":
        return result == "Home"
    elif bet_type == "Away":
        return result == "Away"
    elif bet_type == "Draw":
        return result == "Draw"
    elif bet_type == "1X":
        return result in ["Home", "Draw"]
    elif bet_type == "X2":
        return result in ["Away", "Draw"]
    return None

def authenticate_parlay(parlay, games_df):
    """Valida automaticamente cada jogo do parlay com base nos resultados"""
    updated_details = []
    all_hits = []
    pending = False

    for detail in parlay["details"]:
        game = games_df[
            (games_df["Home"] == detail["game"].split(" vs ")[0]) &
            (games_df["Away"] == detail["game"].split(" vs ")[1])
        ]
        if not game.empty:
            gh = game.iloc[0].get("Goals_H_Today", np.nan)
            ga = game.iloc[0].get("Goals_A_Today", np.nan)
            result_today = game.iloc[0].get("Result_Today", None)

            if pd.isna(gh) or pd.isna(ga):
                status = "⏳"
                hit = None
                pending = True
            else:
                hit = check_bet_hit(result_today, detail["bet"])
                status = "✅" if hit else "❌"
            score = f" ({int(gh)}-{int(ga)})" if pd.notna(gh) and pd.notna(ga) else ""
        else:
            status = "⏳"
            hit = None
            score = ""
            pending = True

        updated_details.append({
            **detail,
            "status": status,
            "score": score
        })
        if hit is not None:
            all_hits.append(hit)

    # Determinar status final
    if pending:
        final_status = "⚪ PENDING"
    elif all(all_hits):
        final_status = "🟢 HIT"
    else:
        final_status = "🔴 LOST"

    parlay["details"] = updated_details
    parlay["final_status"] = final_status
    return parlay

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
            
            # 🔥 FILTROS MAIS RIGOROSOS PARA FINS DE SEMANA
            if weekend_filter and len(games_today_filtered) > 15:
                if prob > (min_prob + 0.05) and edge > -0.05:
                    eligible_games.append((idx, bet_type, prob, round(odds, 2), edge, f"{row['Home']} vs {row['Away']}"))
            else:
                if prob > min_prob:
                    eligible_games.append((idx, bet_type, prob, round(odds, 2), edge, f"{row['Home']} vs {row['Away']}"))
    
    # 🔥 LIMITAR NÚMERO DE JOGOS ELEGÍVEIS
    if len(eligible_games) > max_eligible:
        eligible_games.sort(key=lambda x: x[2], reverse=True)
        eligible_games = eligible_games[:max_eligible]
        st.warning(f"⚡ Limite ativado: {len(eligible_games)} jogos elegíveis (de {len(games_today_filtered)} totais)")
    
    st.info(f"🎯 Jogos elegíveis para parlays: {len(eligible_games)}")
    
    # 🔥 NOVA ESTRATÉGIA: EVITAR REPETIÇÃO DE JOGOS ENTRE PARLAYS
    parlay_suggestions = []
    used_games = set()  # Controlar jogos já usados
    
    # 🔥 PARLAYS DE 2 LEGS
    if min_legs <= 2 and len(eligible_games) >= 2:
        ev_threshold = 0.08 if weekend_filter and len(eligible_games) > 15 else 0.05
        prob_threshold = 0.30 if weekend_filter and len(eligible_games) > 15 else 0.25
        
        # Ordenar por EV e probabilidade
        eligible_games_sorted = sorted(eligible_games, key=lambda x: (x[4], x[2]), reverse=True)
        
        for i in range(len(eligible_games_sorted)):
            for j in range(i + 1, len(eligible_games_sorted)):
                game1 = eligible_games_sorted[i]
                game2 = eligible_games_sorted[j]
                
                # Verificar se os jogos já foram usados
                if game1[0] in used_games or game2[0] in used_games:
                    continue
                
                games_list = [(game1[0], game1[1]), (game2[0], game2[1])]
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
                        
                        # Marcar jogos como usados
                        used_games.add(game1[0])
                        used_games.add(game2[0])
                        
                        # Limitar número de sugestões
                        if len(parlay_suggestions) >= max_suggestions:
                            break
            
            if len(parlay_suggestions) >= max_suggestions:
                break
    
    # 🔥 PARLAYS DE 3 LEGS (apenas se não atingiu max_suggestions)
    if len(parlay_suggestions) < max_suggestions and min_legs <= 3 and max_legs >= 3 and len(eligible_games) >= 3:
        ev_threshold = 0.05 if weekend_filter and len(eligible_games) > 15 else 0.02
        prob_threshold = 0.20 if weekend_filter and len(eligible_games) > 15 else 0.15
        
        # Jogos ainda não usados
        unused_games = [game for game in eligible_games if game[0] not in used_games]
        
        if len(unused_games) >= 3:
            # Combinar jogos não usados
            for combo in itertools.combinations(unused_games, 3):
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
                        
                        # Marcar jogos como usados
                        for game in combo:
                            used_games.add(game[0])
                        
                        # Limitar número de sugestões
                        if len(parlay_suggestions) >= max_suggestions:
                            break
    
    # 🔥 PARLAYS DE 4 LEGS (apenas se não atingiu max_suggestions)
    if len(parlay_suggestions) < max_suggestions and max_legs >= 4 and len(eligible_games) >= 4:
        unused_games = [game for game in eligible_games if game[0] not in used_games]
        
        if len(unused_games) >= 4:
            for combo in itertools.combinations(unused_games, 4):
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
                        
                        # Marcar jogos como usados
                        for game in combo:
                            used_games.add(game[0])
                        
                        if len(parlay_suggestions) >= max_suggestions:
                            break
    
    # Ordenar por Expected Value
    parlay_suggestions.sort(key=lambda x: x['ev'], reverse=True)
    
    st.info(f"🎰 Total de parlays gerados: {len(parlay_suggestions)}")
    st.info(f"🎯 Jogos utilizados: {len(used_games)} de {len(eligible_games)} elegíveis")
    
    return parlay_suggestions[:max_suggestions]

# Gerar sugestões de parlay COM NOVOS PARÂMETROS
parlay_suggestions = generate_parlay_suggestions(
    games_today, 
    parlay_bankroll, 
    min_parlay_prob, 
    max_parlay_suggestions,
    min_parlay_legs,
    max_parlay_legs,
    weekend_filter,
    max_eligible_games
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
    
    # 🟢🟢🟢 ADICIONAR ESTATÍSTICAS DE HANDICAP 🟢🟢🟢
    handicap_bets = df[df['Handicap_Correct'].notna()]
    total_handicap_bets = len(handicap_bets)
    correct_handicap_bets = handicap_bets['Handicap_Correct'].sum()
    winrate_handicap = (correct_handicap_bets / total_handicap_bets) * 100 if total_handicap_bets > 0 else 0

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
        "Kelly Bets Made": len(kelly_bets),
        # 🟢🟢🟢 NOVAS ESTATÍSTICAS 🟢🟢🟢
        "Handicap Bets Made": total_handicap_bets,
        "Handicap Correct": int(correct_handicap_bets),
        "Handicap Winrate (%)": round(winrate_handicap, 2)
    }

summary_ml = summary_stats_ml(finished_games)

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
            'Idx': idx,
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
            'games': [(bet['Idx'], bet['bet_type']) for bet in best_combination],
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

# 🔥 ATUALIZADO: Adicionar colunas de handicap
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
    'ML_Recommendation', 'ML_Data_Valid', 'ML_Correct', 'Handicap_Correct', 'Kelly_Stake_ML',
    'Profit_ML_Fixed', 'Profit_ML_Kelly',
    'ML_Proba_Home', 'ML_Proba_Draw', 'ML_Proba_Away', 
    'Odd_H', 'Odd_D', 'Odd_A',
    'Asian_Line_Decimal', 'Handicap_Result'  # 🔥 NOVO - colunas de handicap
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
        'Odd_A': '{:.2f}',
        'Asian_Line_Decimal': '{:.2f}'  # 🔥 NOVO
    }).apply(lambda x: ['background-color: #ffcccc' if x.name == 'ML_Data_Valid' and x.iloc[0] == False else '' 
                       for i in range(len(x))], axis=1),
    use_container_width=True,
    height=800
)

########################################
##### Bloco 13A – Auto Parlay Display ###
########################################
st.header("🎰 Auto Parlay Recommendations")

if parlay_suggestions:
    # Validar resultados de cada parlay
    for i in range(len(parlay_suggestions)):
        parlay_suggestions[i] = authenticate_parlay(parlay_suggestions[i], games_today)

    # Mostrar estatísticas dos parlays
    legs_count = {}
    for parlay in parlay_suggestions:
        leg_type = parlay["type"]
        legs_count[leg_type] = legs_count.get(leg_type, 0) + 1

    stats_text = " | ".join([f"{count}x {leg}" for leg, count in legs_count.items()])
    st.success(f"📊 Distribuição: {stats_text}")

    for i, parlay in enumerate(parlay_suggestions):
        status = parlay.get("final_status", "⚪ PENDING")
        with st.expander(f"#{i+1} {parlay['type']} – {status} | Prob: {parlay['probability']:.1%} | Odds: {parlay['odds']} | EV: {parlay['ev']:+.1%}"):
            st.write(f"**Stake:** ${parlay['stake']} | **Potencial:** ${parlay['potential_win']}")
            for detail in parlay["details"]:
                st.write(f"{detail['status']} {detail['game']} – {detail['bet']} (Odd: {detail['odds']}, Prob: {detail['prob']:.1%}){detail['score']}")
else:
    st.info("No profitable parlay suggestions found for today.")

# 🔥🔥🔥 SUPER PARLAY SECTION
########################################
##### Bloco 13B – Super Parlay Display ###
########################################
st.header("🎉 SUPER PARLAY OF THE DAY")

if super_parlay:
    super_parlay = authenticate_parlay(super_parlay, games_today)
    status = super_parlay.get("final_status", "⚪ PENDING")

    st.success(f"🔥 **SPECIAL OF THE DAY!** – {status}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilidade", f"{super_parlay['probability']:.1%}")
    with col2:
        st.metric("Odds", f"{super_parlay['odds']:.2f}")
    with col3:
        st.metric("Potencial", f"${super_parlay['potential_win']:.2f}")

    st.write(f"**Stake Recomendado:** ${super_parlay['stake']} | **Expected Value:** {super_parlay['ev']:+.1%}")

    st.subheader("🎯 Jogos Selecionados:")
    for i, detail in enumerate(super_parlay["details"], 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{detail['status']} **{i}. {detail['game']}** ({detail['league']}){detail['score']}")
        with col2:
            st.write(f"**{detail['bet']}** (Odd: {detail['odds']})")
else:
    st.info("Não foi possível gerar um Super Parlay hoje. Tente ajustar a odd alvo ou aguarde mais jogos.")

# st.markdown("---")
# st.success("🎯 **SISTEMA PARLAY CORRIGIDO** - Lógica de Handicap Asiático implementada com sucesso!")

# st.info("""
# **🚀 CORREÇÃO APLICADA - Asian Line Consertada**

# ✅ **Problema Resolvido:**
# - Função `convert_asian_line_to_decimal` agora inverte corretamente o sinal
# - Segue padrões Pinnacle/Bet365: Away → Home perspective
# - Cálculos de handicap agora funcionam corretamente

# **Exemplos Corrigidos:**
# - '0/0.5'   → -0.25 (✅ CORRETO)  
# - '-0.5/0'  → +0.25 (✅ CORRETO)
# - '0.5'     → -0.5  (✅ CORRETO)

# **Impacto:**
# - ML treinado com targets corretos
# - Recomendações baseadas em handicaps precisos
# - Profit/ROI calculado corretamente
# - Sistema Parlay mantém sua funcionalidade original
# """)
