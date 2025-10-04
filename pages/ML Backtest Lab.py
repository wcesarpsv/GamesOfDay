########################################
# BLOCO 1 – IMPORTS & CONFIG (CORRIGIDO)
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# Tentativa opcional (XGBoost/LightGBM)
XGB_AVAILABLE, LGBM_AVAILABLE = True, True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBM_AVAILABLE = False

st.set_page_config(page_title="ML Backtest Lab – Regras x ML", layout="wide")
st.title("⚡ ML Backtest Lab – Regras x ML")

########################################
# BLOCO 2 – CONSTANTES & AJUSTES
########################################
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "afc","nordeste"]
DOMINANT_THRESHOLD = 0.90

# Aparência/UX
st.markdown("""
Este laboratório permite:
- Treinar e comparar **ML** com seu **modelo de regras**.
- Ajustar **hiperparâmetros** via UI.
- Visualizar **ROI**, **Winrate**, **LogLoss**, **AUC**.
- Ver **gráfico de calibração** (linha perfeita vs curva do modelo).
""")
st.divider()

########################################
# BLOCO 3 – HELPERS DE DADOS (CORRIGIDO)
########################################
def load_all_games(folder):
    if not os.path.exists(folder):
        st.error(f"Pasta não encontrada: {folder}")
        return pd.DataFrame()
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = []
    for file in files:
        path = os.path.join(folder, file)
        try:
            df = pd.read_csv(path)
            df["__srcfile"] = file
            df_list.append(df)
        except Exception as e:
            st.error(f"Erro ao ler {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].astype(str).str.lower().str.contains(pattern, na=False)].copy()

def prepare_history(df):
    # Padroniza League para evitar problemas no merge
    df['League'] = df['League'].astype(str).str.strip().str.lower()

    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Coluna obrigatória ausente no histórico: {col}")
            return pd.DataFrame()

    out = df.dropna(subset=['Goals_H_FT', 'Goals_A_FT']).copy()
    # Garantir 'Date'
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce', dayfirst=False)
    else:
        out['Date'] = pd.NaT
    return out

def compute_league_bands(history_df):
    # Filtra ligas com menos de 10 jogos
    history_df = history_df.groupby('League').filter(lambda x: len(x) >= 10)
    hist = history_df.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']

    # Calcula quantis por liga
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
    m_h, m_a = row.get('M_H'), row.get('M_A')
    if pd.isna(m_h) or pd.isna(m_a): return "Mixed / Neutral"
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold: return "Home strong"
    if m_h <= -threshold: return "Home weak"
    if m_a >= threshold: return "Away strong"
    if m_a <= -threshold: return "Away weak"
    return "Mixed / Neutral"


########################################
# BLOCO 3.5 – FUNÇÕES AUSENTES (NOVO)
########################################

# 3.5.1 Funções Auxiliares Originais
def compute_double_chance_odds(df):
    """Calcula odds para dupla chance 1X e X2"""
    df = df.copy()
    if 'Odd_H' in df.columns and 'Odd_D' in df.columns:
        # Fórmula correta para odds de dupla chance
        df['Odd_1X'] = 1 / (1/df['Odd_H'] + 1/df['Odd_D'])
    else:
        df['Odd_1X'] = np.nan
        
    if 'Odd_A' in df.columns and 'Odd_D' in df.columns:
        df['Odd_X2'] = 1 / (1/df['Odd_A'] + 1/df['Odd_D'])
    else:
        df['Odd_X2'] = np.nan
        
    return df

def classify_leagues_variation(history_df):
    """Classifica ligas por nível de variação dos momentos"""
    if history_df.empty:
        return pd.DataFrame(columns=['League', 'League_Classification'])
        
    variation_data = []
    for league in history_df['League'].unique():
        league_data = history_df[history_df['League'] == league]
        if len(league_data) < 5:  # Mínimo de jogos para análise
            continue
            
        # Calcula variação dos momentos
        var_home = league_data['M_H'].std()
        var_away = league_data['M_A'].std()
        avg_var = (var_home + var_away) / 2
        
        # Classificação baseada em quartis
        if pd.isna(avg_var):
            classification = 'Medium Variation'
        elif avg_var > 0.7:
            classification = 'High Variation'
        elif avg_var > 0.4:
            classification = 'Medium Variation'
        else:
            classification = 'Low Variation'
            
        variation_data.append({
            'League': league, 
            'League_Classification': classification
        })
        
    return pd.DataFrame(variation_data) if variation_data else pd.DataFrame(columns=['League', 'League_Classification'])


#################################################
# 3.5.2 Funções de EV e Lucro com Filtro (NOVAS)
#################################################

def calculate_ev(prob, odd):
    """Calcula o Valor Esperado: EV = (Probabilidade * Odd) - 1"""
    # Retorna NaN se a Odd for inválida (Odd <= 1.0 ou NaN)
    if pd.isna(prob) or pd.isna(odd) or odd <= 1.0:
        return np.nan
    return (prob * odd) - 1

def calculate_profit_with_ev_filter(rec, result, odds_row, prob_row, ev_threshold):
    """
    Calcula o lucro, APENAS se o EV da aposta (identificado pela recomendação) 
    for maior que o threshold.
    Assume que as colunas 'ML_EV_*' já foram calculadas e estão em prob_row (que é a row do DF).
    """
    if pd.isna(rec) or result is None or rec == '❌ Avoid': return 0.0
    r = str(rec)
    
    # 1. Definir a aposta, buscar Odd e EV correspondente
    current_ev = -1.0
    odd = np.nan
    target_result = None

    if 'Back Home' in r:
        odd = odds_row.get('Odd_H', np.nan)
        current_ev = prob_row.get('ML_EV_Home', -1.0)
        target_result = "Home"
    elif 'Back Away' in r:
        odd = odds_row.get('Odd_A', np.nan)
        current_ev = prob_row.get('ML_EV_Away', -1.0)
        target_result = "Away"
    elif 'Back Draw' in r:
        odd = odds_row.get('Odd_D', np.nan)
        current_ev = prob_row.get('ML_EV_Draw', -1.0)
        target_result = "Draw"
    elif '1X' in r:
        odd = odds_row.get('Odd_1X', np.nan)
        current_ev = prob_row.get('ML_EV_1X', -1.0)
        target_result = ["Home", "Draw"]
    elif 'X2' in r:
        odd = odds_row.get('Odd_X2', np.nan)
        current_ev = prob_row.get('ML_EV_X2', -1.0)
        target_result = ["Away", "Draw"]
    else:
        return 0.0

    # 2. Aplicar o filtro de EV
    if pd.isna(odd) or current_ev < ev_threshold:
        return 0.0 # Não faz aposta (Profit = 0)
    
    # 3. Calcular o Lucro
    if target_result is None:
        return 0.0
        
    is_win = False
    if isinstance(target_result, list):
        is_win = result in target_result
    else:
        is_win = result == target_result

    if is_win:
        return odd - 1.0 # Ganho
    else:
        return -1.0 # Perda (aposta de 1 unidade)

########################################
# BLOCO 4 – REGRAS (AUTO RECOMMENDATION)
########################################
def auto_recommendation(row,
                        diff_mid_lo=0.20, diff_mid_hi=0.80,
                        diff_mid_hi_highvar=0.75, power_gate=1, power_gate_highvar=5):
    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power', np.nan)
    league_cls= row.get('League_Classification', 'Medium Variation')
    m_a       = row.get('M_A', np.nan)
    m_h       = row.get('M_H', np.nan)
    odd_d     = row.get('Odd_D', np.nan)

    # 1) Fortes
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '🟢 Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '🟠 Back Away'

    if dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if pd.notna(diff_m) and diff_m >= 0.90:
            return '🟢 Back Home'
    if dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home == 'Balanced':
        if pd.notna(diff_m) and diff_m <= -0.90:
            return '🟪 X2 (Away/Draw)'

    # 2) Ambos Balanced com thresholds
    if (band_home == 'Balanced') and (band_away == 'Balanced') and pd.notna(diff_m) and pd.notna(diff_pow):
        if league_cls == 'High Variation':
            if (0.45 <= diff_m < diff_mid_hi_highvar and diff_pow >= power_gate_highvar):
                return '🟦 1X (Home/Draw)'
            if (-diff_mid_hi_highvar < diff_m <= -0.45 and diff_pow <= -power_gate_highvar):
                return '🟪 X2 (Away/Draw)'
        else:
            if (diff_mid_lo <= diff_m < diff_mid_hi and diff_pow >= power_gate):
                return '🟦 1X (Home/Draw)'
            if (-diff_mid_hi < diff_m <= -diff_mid_lo and diff_pow <= -power_gate):
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

    # 5) Draw filter
    if pd.notna(odd_d) and 2.5 <= odd_d <= 6.0 and pd.notna(diff_pow) and -10 <= diff_pow <= 10:
        if (pd.notna(m_h) and 0 <= m_h <= 1) or (pd.notna(m_a) and 0 <= m_a <= 0.5):
            return '⚪ Back Draw'

    return '❌ Avoid'

def map_result(row):
    gh, ga = row['Goals_H_FT'], row['Goals_A_FT']
    if gh > ga:   return "Home"
    if gh < ga:   return "Away"
    return "Draw"

def determine_result_today(row):
    gh, ga = row.get('home_goal'), row.get('away_goal')
    if pd.isna(gh) or pd.isna(ga): return None
    if gh > ga: return "Home"
    if gh < ga: return "Away"
    return "Draw"

def check_recommendation(rec, result):
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return None
    rec = str(rec)
    if 'Back Home' in rec: return result == "Home"
    if 'Back Away' in rec: return result == "Away"
    if 'Back Draw' in rec: return result == "Draw"
    if '1X' in rec:        return result in ["Home", "Draw"]
    if 'X2' in rec:        return result in ["Away", "Draw"]
    return None

def calculate_profit(rec, result, odds_row):
    if pd.isna(rec) or result is None or rec == '❌ Avoid': return 0.0
    r = str(rec)
    if 'Back Home' in r:
        odd = odds_row.get('Odd_H', np.nan); return (odd - 1) if result == "Home" else -1
    if 'Back Away' in r:
        odd = odds_row.get('Odd_A', np.nan); return (odd - 1) if result == "Away" else -1
    if 'Back Draw' in r:
        odd = odds_row.get('Odd_D', np.nan); return (odd - 1) if result == "Draw" else -1
    if '1X' in r:
        odd = odds_row.get('Odd_1X', np.nan); return (odd - 1) if result in ["Home","Draw"] else -1
    if 'X2' in r:
        odd = odds_row.get('Odd_X2', np.nan); return (odd - 1) if result in ["Away","Draw"] else -1
    return 0.0

########################################
# BLOCO 5 – UI: DADOS & BACKTEST (CORRIGIDO COM PERÍODO DE TESTE)
########################################
st.header("📂 Dados & Backtest")

data_mode = st.radio("Carregar dados:", ["Usar pasta GamesDay", "Upload manual"], horizontal=True)
if data_mode == "Upload manual":
    up = st.file_uploader("Envie um CSV único com histórico + jogos", type=["csv"])
    if up:
        all_games = pd.read_csv(up)
    else:
        all_games = pd.DataFrame()
else:
    all_games = load_all_games(GAMES_FOLDER)

if all_games.empty:
    st.warning("Sem dados para análise. Carregue/garanta CSVs em 'GamesDay'.")
    st.stop()

# VALIDAÇÃO DE COLUNAS OBRIGATÓRIAS
required_cols = ['M_H', 'M_A', 'Goals_H_FT', 'Goals_A_FT', 'League']
missing_cols = [col for col in required_cols if col not in all_games.columns]
if missing_cols:
    st.error(f"❌ Colunas obrigatórias faltando: {missing_cols}")
    st.stop()

all_games = filter_leagues(all_games)
all_games = compute_double_chance_odds(all_games)

# Parse de datas
if 'Date' in all_games.columns:
    all_games['Date'] = pd.to_datetime(all_games['Date'], errors='coerce', dayfirst=False)
else:
    all_games['Date'] = pd.NaT

history = prepare_history(all_games)
if history.empty:
    st.error("Histórico inválido (precisa ter gols finais e colunas obrigatórias).")
    st.stop()

# Bands e variação por liga (derivados do histórico)
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

# Merge em history
history = history.merge(league_class, on='League', how='left')
history = history.merge(league_bands, on='League', how='left')
history['M_Diff'] = history['M_H'] - history['M_A']

# Garantir que as colunas de bands existem antes de usá-las
if 'Home_P20' in history.columns and 'Home_P80' in history.columns:
    history['Home_Band'] = np.where(
        history['M_H'] <= history['Home_P20'], 'Bottom 20%',
        np.where(history['M_H'] >= history['Home_P80'], 'Top 20%', 'Balanced')
    )
else:
    history['Home_Band'] = 'Balanced'

if 'Away_P20' in history.columns and 'Away_P80' in history.columns:
    history['Away_Band'] = np.where(
        history['M_A'] <= history['Away_P20'], 'Bottom 20%',
        np.where(history['M_A'] >= history['Away_P80'], 'Top 20%', 'Balanced')
    )
else:
    history['Away_Band'] = 'Balanced'

history['Dominant'] = history.apply(dominant_side, axis=1)
history['Result'] = history.apply(map_result, axis=1)

# Intervalo de backtest - SISTEMA CORRIGIDO
valid_dates = history['Date'].dropna()
if valid_dates.empty:
    st.error("❌ Nenhuma data válida encontrada nos dados.")
    st.stop()

min_d, max_d = valid_dates.min(), valid_dates.max()

st.subheader("🗓️ Períodos de Treino e Teste")

colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**📅 Período de Treino**")
    train_start_date = st.date_input("Início do treino", value=(max_d - timedelta(days=180)).date())
    train_end_date = st.date_input("Fim do treino", value=(max_d - timedelta(days=30)).date())
    
with colB:
    st.markdown("**🧪 Período de Teste**")
    test_start_date = st.date_input("Início do teste", value=(max_d - timedelta(days=29)).date())
    test_end_date = st.date_input("Fim do teste", value=max_d.date())
    
with colC:
    st.markdown("**⚙️ Configuração**")
    lookback_days = st.number_input("Dias de lookback para features", 30, 400, 120, 
                                   help="Quantos dias anteriores usar para calcular features como média móvel")

# Validações de datas
train_start_dt = pd.to_datetime(train_start_date)
train_end_dt = pd.to_datetime(train_end_date)
test_start_dt = pd.to_datetime(test_start_date)
test_end_dt = pd.to_datetime(test_end_date)

if train_start_dt >= train_end_dt:
    st.error("❌ Data inicial do treino deve ser anterior à data final do treino.")
    st.stop()

if test_start_dt >= test_end_dt:
    st.error("❌ Data inicial do teste deve ser anterior à data final do teste.")
    st.stop()

if train_end_dt >= test_start_dt:
    st.error("❌ Período de treino deve terminar antes do período de teste.")
    st.stop()

# Split por período - SISTEMA CORRIGIDO
train_mask = (history['Date'] >= train_start_dt) & (history['Date'] <= train_end_dt)
test_mask = (history['Date'] >= test_start_dt) & (history['Date'] <= test_end_dt)

train_df = history[train_mask].copy()
test_df = history[test_mask].copy()

# VALIDAÇÃO CRÍTICA: garantir que temos dados
if train_df.empty:
    st.error("❌ Nenhum dado para treino no período selecionado.")
    st.info(f"Datas disponíveis: {min_d.date()} a {max_d.date()}")
    st.stop()

if test_df.empty:
    st.error("❌ Nenhum dado para teste no período selecionado.")
    st.info(f"Datas disponíveis: {min_d.date()} a {max_d.date()}")
    st.stop()

# Estatísticas dos períodos
train_days = (train_end_dt - train_start_dt).days
test_days = (test_end_dt - test_start_dt).days

st.success(f"""
✅ **Divisão de dados configurada:**
- **Treino:** {train_start_date} a {train_end_date} ({train_days} dias, {train_df.shape[0]} jogos)
- **Teste:** {test_start_date} a {test_end_date} ({test_days} dias, {test_df.shape[0]} jogos)
- **Lookback:** {lookback_days} dias para features
""")


########################################
# BLOCO 6 – UI: MODELO & HIPERPARÂMETROS
########################################
st.header("⚙️ Configuração do Modelo")

model_options = ["Random Forest", "Logistic Regression"]
if XGB_AVAILABLE:  model_options.append("XGBoost")
if LGBM_AVAILABLE: model_options.append("LightGBM")

model_choice = st.selectbox("Selecione o modelo ML", model_options, index=0)

st.subheader("🔧 Hiperparâmetros")
params = {}
if model_choice == "Random Forest":
    params["n_estimators"] = st.slider("n_estimators", 100, 1200, 800, step=50)
    params["max_depth"] = st.slider("max_depth", 2, 30, 12)
    params["min_samples_split"] = st.slider("min_samples_split", 2, 50, 10)
    params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 20, 4)
    params["max_features"] = st.selectbox("max_features", ["sqrt", "log2", None], index=0)
    params["class_weight"] = st.selectbox("class_weight", [None, "balanced", "balanced_subsample"], index=2)
elif model_choice == "Logistic Regression":
    params["C"] = st.slider("C (Regularização)", 0.01, 10.0, 1.0)
    params["solver"] = st.selectbox("solver", ["lbfgs", "liblinear", "saga"], index=0)
    params["max_iter"] = st.slider("max_iter", 100, 2000, 500, step=100)
elif model_choice == "XGBoost":
    params["n_estimators"] = st.slider("n_estimators", 100, 1500, 400, step=50)
    params["max_depth"] = st.slider("max_depth", 2, 15, 6)
    params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)
    params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8)
    params["colsample_bytree"] = st.slider("colsample_bytree", 0.5, 1.0, 0.8)
elif model_choice == "LightGBM":
    params["num_leaves"] = st.slider("num_leaves", 10, 500, 31)
    params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)
    params["feature_fraction"] = st.slider("feature_fraction", 0.4, 1.0, 0.8)
    params["bagging_fraction"] = st.slider("bagging_fraction", 0.4, 1.0, 0.8)
    params["n_estimators"] = st.slider("n_estimators", 50, 1500, 400, step=50)

st.subheader("⚡ Opções Extras")
apply_calibration = st.checkbox("Aplicar calibração isotônica", value=False)
compare_rules = st.checkbox("Comparar com modelo de regras", value=True)
save_csv = st.checkbox("Salvar previsões (CSV)", value=False)

st.divider()

########################################
# BLOCO 7 – Ajustado: Features e build_X (ATUALIZADO)
########################################

# Lista de features usadas no modelo
features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band','Dominant','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed'
]
# Garante que só ficam features existentes no DataFrame
features_raw = [f for f in features_raw if f in history.columns]

# Mapeamento de bandas
BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}

# Função para preparar X e y
def build_X(df, fit_encoder=False, encoder=None, cat_cols=None):
    """
    Prepara o dataframe de entrada (X) para treino ou teste:
    - Garante que todas as colunas de features existam
    - Faz mapeamento de bandas para valores numéricos
    - Aplica OneHotEncoder nas colunas categóricas
    - Retorna dataframe final somente com valores numéricos
    """
    # Garante que todas as features existem no df
    for col in features_raw:
        if col not in df.columns:
            df[col] = np.nan

    X = df[features_raw].copy()

    # Mapear Home_Band e Away_Band
    if 'Home_Band' in X.columns:
        X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP).fillna(2).astype(int)
    if 'Away_Band' in X.columns:
        X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP).fillna(2).astype(int)

    # Identificar colunas categóricas
    if cat_cols is None:
        cat_cols = [c for c in ['Dominant','League_Classification'] if c in X.columns]

    # OneHotEncoder
    if fit_encoder:
        if cat_cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(X[cat_cols])
        else:
            encoder = None
            encoded = np.zeros((len(X), 0))
    else:
        if encoder and cat_cols:
            encoded = encoder.transform(X[cat_cols])
        else:
            encoded = np.zeros((len(X), 0))

    # Converter para dataframe
    if encoded.size > 0 and encoder is not None:
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(cat_cols), 
            index=X.index
        )
    else:
        encoded_df = pd.DataFrame(index=X.index)

    # Combinar numéricas + one-hot
    X_num = X.drop(columns=cat_cols, errors='ignore').reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    X_out = pd.concat([X_num, encoded_df], axis=1)

    # Garantir apenas valores numéricos e sem NaN
    X_out = X_out.apply(pd.to_numeric, errors='coerce')
    X_out.fillna(0, inplace=True)

    return X_out, encoder, cat_cols

########################################
# BLOCO 7B – Criação de X_train, X_test, y_train
########################################

# Target
y_train = train_df['Result'].copy()
y_test = test_df['Result'].copy()

# Criar X_train e X_test
X_train, encoder, cat_cols = build_X(train_df, fit_encoder=True)
X_test, _, _ = build_X(test_df, fit_encoder=False, encoder=encoder, cat_cols=cat_cols)

# Debug opcional
st.write("DEBUG - X_train shape:", X_train.shape)
st.write("DEBUG - X_test shape:", X_test.shape)
st.write("DEBUG - y_train shape:", y_train.shape)


########################################
# BLOCO 8A – Validação inicial corrigido
########################################

if X_train.empty or y_train.empty:
    st.error("Erro crítico: X_train ou y_train estão vazios. Sem dados para treinar o modelo.")
    st.stop()

# Garante que y_train esteja alinhado com X_train
y_train = y_train.reset_index(drop=True)

# Criar a máscara para remover linhas com NaN
mask = ~X_train.isnull().any(axis=1)

# Aplicar a máscara
X_train = X_train.loc[mask].copy()
y_train = y_train.loc[mask].copy()

# Debug
st.write("DEBUG pós-limpeza - X_train:", X_train.shape)
st.write("DEBUG pós-limpeza - y_train:", y_train.shape)



########################################
# FUNÇÃO make_model
########################################
def make_model(choice, params):
    if choice == "Random Forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if choice == "Logistic Regression":
        return LogisticRegression(random_state=42, **params)
    if choice == "XGBoost" and XGB_AVAILABLE:
        return XGBClassifier(random_state=42, eval_metric="logloss", **params)
    if choice == "LightGBM" and LGBM_AVAILABLE:
        return LGBMClassifier(random_state=42, **params)
    raise ValueError("Modelo não suportado ou não disponível")



########################################
# SUBBLOCO 8B – Treinamento seguro (com botão)
########################################

# Botão para executar o treinamento
if st.button("🚀 Rodar Treinamento / Teste"):
    with st.spinner("Treinando modelo..."):
        base_model = make_model(model_choice, params)

        if apply_calibration:
            model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
        else:
            model = base_model

        try:
            model.fit(X_train, y_train)
            st.success("Treinamento concluído com sucesso!")

            # Guardar o modelo no session_state para usar depois
            st.session_state["trained_model"] = model
        except ValueError as e:
            st.error("Erro durante o treinamento. Confira detalhes abaixo:")
            st.code(str(e))
            st.stop()


########################################
# SUBBLOCO 8D – Predição e Recomendações (ajustado)
########################################

# 8D.1 Geração de Probabilidades e Predições
if "trained_model" in st.session_state:
    model = st.session_state["trained_model"]

    # Geração de probabilidades e predições
    proba_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    pred_test = model.predict(X_test)

    classes_ = list(model.classes_)

    # Função auxiliar para pegar probabilidades por classe
    def p(cls):
        if proba_test is None: 
            return np.zeros(len(X_test))
        idx = classes_.index(cls) if cls in classes_ else None
        return proba_test[:, idx] if idx is not None else np.zeros(len(X_test))

    # Copiar test_df e adicionar colunas
    test_df = test_df.copy()
    test_df["ML_Proba_Home"] = p("Home")
    test_df["ML_Proba_Draw"] = p("Draw")
    test_df["ML_Proba_Away"] = p("Away")
    test_df["ML_Pred"] = pred_test

    # 8D.2 UI de Limiares (Thresholds)
    # Ajuste do limiar para decisão
    st.subheader("🎯 Limiar para Decisão e Valor Esperado (EV)")
    colT1, colT2 = st.columns(2)
    with colT1:
        threshold = st.slider("Threshold (%) para Back Home/Away (Decisão ML)", 40, 85, 65, step=1) / 100.0
    with colT2:
        # NOVO: Limite de EV
        ev_threshold = st.slider("Threshold (%) Mínimo de EV (Filtro de Lucro)", 0, 30, 5, step=1) / 100.0
    
    # Armazenar o EV Threshold no state para uso posterior
    st.session_state['EV_THRESHOLD'] = ev_threshold

    # 8D.3 Recomendação ML
    # Função de recomendação baseada nas probabilidades
    def ml_rec_from_proba(row, thr=0.65):
        ph, pd_, pa = row['ML_Proba_Home'], row['ML_Proba_Draw'], row['ML_Proba_Away']
        if ph >= thr:
            return "🟢 Back Home"
        if pa >= thr:
            return "🟠 Back Away"
        sum_hd, sum_ad = ph + pd_, pa + pd_
        if abs(ph - pa) < 0.05 and pd_ > 0.35:
            return "⚪ Back Draw"
        if sum_hd > sum_ad:
            return "🟦 1X (Home/Draw)"
        if sum_ad > sum_hd:
            return "🟪 X2 (Away/Draw)"
        return "❌ Avoid"

    # Aplicar a recomendação
    test_df["ML_Recommendation"] = test_df.apply(ml_rec_from_proba, axis=1, thr=threshold)

else:
    st.warning("⚠️ Treine o modelo primeiro clicando no botão acima.")


########################################
# SUBBLOCO 8C – Debug opcional
########################################
show_debug = st.checkbox("Mostrar debug detalhado dos dados", value=False)

if show_debug:
    st.write("Primeiras linhas de X_train:")
    st.dataframe(X_train.head(20))
    
    st.write("Primeiras linhas de y_train:")
    st.dataframe(y_train.head(20))
    
    st.write("Valores únicos por coluna:")
    for col in X_train.columns:
        st.write(col, X_train[col].unique()[:10])



########################################
# BLOCO 9 – COMPARAÇÃO COM REGRAS & PROFIT (CORRIGIDO)
########################################

# Garantir que temos dados para teste
if test_df.empty:
    st.error("❌ Nenhum dado disponível para teste.")
    st.stop()

# Só prosseguir se já tiver modelo treinado e recomendações ML
if "trained_model" not in st.session_state or "ML_Recommendation" not in test_df.columns:
    st.warning("⚠️ Treine o modelo clicando no botão acima para ver métricas e lucros.")
    st.stop()

# Padronizar ligas
test_df['League'] = test_df['League'].astype(str).str.strip().str.lower()
league_class['League'] = league_class['League'].astype(str).str.strip().str.lower()
league_bands['League'] = league_bands['League'].astype(str).str.strip().str.lower()

# Merge (classificação de liga é útil; bands só se existir)
test_df = test_df.merge(league_class, on='League', how='left', suffixes=("", "_lc"))

# VERIFICAR MERGE COM LEAGUE_BANDS ANTES DE USAR
if not league_bands.empty:
    ligas_antes = len(test_df)
    test_df = test_df.merge(league_bands, on='League', how='left', suffixes=("", "_lb"))
    ligas_depois = len(test_df)
    
    if ligas_depois == 0:
        st.warning("⚠️ Merge com league_bands removeu todos os dados. Usando apenas classificações.")
        # Recriar test_df sem o merge problemático
        test_df = history[test_mask].copy()
        test_df = test_df.merge(league_class, on='League', how='left')

# ---- Bands: usar *_Num se existir; caso contrário, derivar; não travar se P20/P80 faltar
REV_MAP = {1: "Bottom 20%", 2: "Balanced", 3: "Top 20%"}

def classify_band(value, low, high):
    if pd.isna(value) or pd.isna(low) or pd.isna(high): 
        return "Balanced"
    if value <= low: 
        return "Bottom 20%"
    if value >= high: 
        return "Top 20%"
    return "Balanced"

# Home Band
if 'Home_Band_Num' not in test_df.columns:
    if 'Home_Band' in test_df.columns:
        test_df['Home_Band_Num'] = test_df['Home_Band'].map(BAND_MAP).fillna(2).astype(int)
    elif {'Home_P20','Home_P80'}.issubset(test_df.columns):
        test_df['Home_Band'] = test_df.apply(
            lambda r: classify_band(r.get('M_H'), r.get('Home_P20'), r.get('Home_P80')), 
            axis=1
        )
        test_df['Home_Band_Num'] = test_df['Home_Band'].map(BAND_MAP).fillna(2).astype(int)
    else:
        test_df['Home_Band_Num'] = 2
        test_df['Home_Band'] = "Balanced"

# Away Band
if 'Away_Band_Num' not in test_df.columns:
    if 'Away_Band' in test_df.columns:
        test_df['Away_Band_Num'] = test_df['Away_Band'].map(BAND_MAP).fillna(2).astype(int)
    elif {'Away_P20','Away_P80'}.issubset(test_df.columns):
        test_df['Away_Band'] = test_df.apply(
            lambda r: classify_band(r.get('M_A'), r.get('Away_P20'), r.get('Away_P80')), 
            axis=1
        )
        test_df['Away_Band_Num'] = test_df['Away_Band'].map(BAND_MAP).fillna(2).astype(int)
    else:
        test_df['Away_Band_Num'] = 2
        test_df['Away_Band'] = "Balanced"

# Labels textuais para exibição
if 'Home_Band' not in test_df.columns:
    test_df['Home_Band'] = test_df['Home_Band_Num'].map(REV_MAP)
if 'Away_Band' not in test_df.columns:
    test_df['Away_Band'] = test_df['Away_Band_Num'].map(REV_MAP)

# Calcular M_Diff e Dominant
if 'M_Diff' not in test_df.columns:
    test_df['M_Diff'] = test_df['M_H'] - test_df['M_A']
    
test_df['Dominant'] = test_df.apply(dominant_side, axis=1)

# Recomendações (regras)
if compare_rules:
    test_df['Auto_Recommendation'] = test_df.apply(auto_recommendation, axis=1)
else:
    test_df['Auto_Recommendation'] = np.nan


# 9.1 Geração do EV (NOVO)
# --------------------------------------------------------------------------------------
# NOVO: CÁLCULO DO EV PARA TODAS AS POSSÍVEIS APOSTAS ML (USADO PARA FILTRAGEM DE LUCRO)
# --------------------------------------------------------------------------------------
if "trained_model" in st.session_state:
    # 1. Calcular EV para as apostas simples (H, D, A)
    test_df['ML_EV_Home'] = test_df.apply(
        lambda r: calculate_ev(r['ML_Proba_Home'], r.get('Odd_H')), axis=1
    )
    test_df['ML_EV_Draw'] = test_df.apply(
        lambda r: calculate_ev(r['ML_Proba_Draw'], r.get('Odd_D')), axis=1
    )
    test_df['ML_EV_Away'] = test_df.apply(
        lambda r: calculate_ev(r['ML_Proba_Away'], r.get('Odd_A')), axis=1
    )

    # 2. Calcular Probabilidades e EV para Dupla Chance (1X, X2)
    test_df['ML_Proba_1X'] = test_df['ML_Proba_Home'] + test_df['ML_Proba_Draw']
    test_df['ML_EV_1X'] = test_df.apply(
        lambda r: calculate_ev(r['ML_Proba_1X'], r.get('Odd_1X')), axis=1
    )
    test_df['ML_Proba_X2'] = test_df['ML_Proba_Away'] + test_df['ML_Proba_Draw']
    test_df['ML_EV_X2'] = test_df.apply(
        lambda r: calculate_ev(r['ML_Proba_X2'], r.get('Odd_X2')), axis=1
    )

# 9.2 Cálculo de Lucros com Filtro EV (MODIFICADO)
# Obter o threshold de EV do Session State (default para 0.0 se não existir)
ev_threshold_final = st.session_state.get('EV_THRESHOLD', 0.00)

# Função auxiliar para o cálculo de lucro simples (Auto) - Garantindo que a função exista
def calculate_profit_simple(rec, result, odds_row):
    """Calcula o lucro sem filtro de EV, usado para a comparação de Regras (Auto)"""
    if pd.isna(rec) or result is None or rec == '❌ Avoid': return 0.0
    r = str(rec)
    
    if 'Back Home' in r:
        odd = odds_row.get('Odd_H', np.nan); return (odd - 1) if result == "Home" and not pd.isna(odd) else -1
    if 'Back Away' in r:
        odd = odds_row.get('Odd_A', np.nan); return (odd - 1) if result == "Away" and not pd.isna(odd) else -1
    if 'Back Draw' in r:
        odd = odds_row.get('Odd_D', np.nan); return (odd - 1) if result == "Draw" and not pd.isna(odd) else -1
    if '1X' in r:
        odd = odds_row.get('Odd_1X', np.nan); return (odd - 1) if result in ["Home","Draw"] and not pd.isna(odd) else -1
    if 'X2' in r:
        odd = odds_row.get('Odd_X2', np.nan); return (odd - 1) if result in ["Away","Draw"] and not pd.isna(odd) else -1
    return 0.0


# Lucros
# Profit ML: AGORA USA O FILTRO DE EV! (calculate_profit_with_ev_filter é do Bloco 3.5.2)
test_df['Profit_ML'] = test_df.apply(
    lambda r: calculate_profit_with_ev_filter(
        r['ML_Recommendation'], r['Result'], r, r, ev_threshold_final
    ), 
    axis=1
)

# Profit Auto: Usa o cálculo simples (sem filtro de EV, para comparação)
test_df['Profit_Auto'] = test_df.apply(
    lambda r: calculate_profit_simple(r['Auto_Recommendation'], r['Result'], r), 
    axis=1
)

# Acertos
test_df['ML_Correct'] = test_df.apply(lambda r: check_recommendation(r['ML_Recommendation'], r['Result']), axis=1)
test_df['Auto_Correct'] = test_df.apply(lambda r: check_recommendation(r['Auto_Recommendation'], r['Result']), axis=1)

st.success(f"✅ Dados preparados: {len(test_df)} jogos para análise")


########################################
# BLOCO 10 – MÉTRICAS & SUMÁRIOS (CORRIGIDO COM GOLS)
########################################
st.header("📈 Métricas do Teste (na data selecionada)")

def safe_auc(y_true, proba_df, labels=("Home","Draw","Away")):
    try:
        # OvR ponderado
        y = pd.Categorical(y_true, categories=list(labels))
        Y_bin = pd.get_dummies(y)
        aucs = []
        for c in labels:
            if c in Y_bin and f"ML_Proba_{c}" in proba_df:
                aucs.append(roc_auc_score(Y_bin[c], proba_df[f"ML_Proba_{c}"]))
        return float(np.mean(aucs)) if aucs else np.nan
    except Exception:
        return np.nan

metrics_cols = ["ML_Proba_Home","ML_Proba_Draw","ML_Proba_Away"]
auc_val = safe_auc(test_df['Result'], test_df[metrics_cols])
try:
    # Para logloss multiclasse, precisamos do array de probs alinhado em classes_
    if proba_test is not None:
        # Reordenar para [Home,Draw,Away] se existirem
        wanted = ["Home","Draw","Away"]
        cols = []
        for w in wanted:
            if w in classes_:
                cols.append(proba_test[:, classes_.index(w)])
            else:
                cols.append(np.zeros(len(test_df)))
        proba_for_logloss = np.vstack(cols).T
        logloss_val = log_loss(test_df['Result'], proba_for_logloss, labels=wanted)
    else:
        logloss_val = np.nan
except Exception:
    logloss_val = np.nan

acc_val = accuracy_score(test_df['Result'], test_df['ML_Pred']) if len(test_df) else np.nan
brier_val = brier_score_loss(
    (test_df['Result']=="Home").astype(int),
    test_df["ML_Proba_Home"]
) if "ML_Proba_Home" in test_df and len(test_df) else np.nan

ml_bets = test_df[test_df['ML_Recommendation']!='❌ Avoid']
auto_bets = test_df[test_df['Auto_Recommendation']!='❌ Avoid']

summary = {
    "Jogos (teste)": int(len(test_df)),
    "Apostas ML": int(len(ml_bets)),
    "Winrate ML (%)": round(100 * (ml_bets['ML_Correct'].sum()/len(ml_bets)) ,2) if len(ml_bets) else 0.0,
    "Profit ML": round(test_df['Profit_ML'].sum(), 2),
    "AUC (OvR)": None if np.isnan(auc_val) else round(auc_val, 4),
    "LogLoss": None if np.isnan(logloss_val) else round(logloss_val, 4),
    "Brier (Home)": None if np.isnan(brier_val) else round(brier_val, 4),
}
if compare_rules:
    summary.update({
        "Apostas Regras": int(len(auto_bets)),
        "Winrate Regras (%)": round(100 * (auto_bets['Auto_Correct'].sum()/len(auto_bets)) ,2) if len(auto_bets) else 0.0,
        "Profit Regras": round(test_df['Profit_Auto'].sum(), 2),
    })

st.json(summary)


########################################
# BLOCO 10B – MÉTRICAS EXTRAS (ECE e MCC)
########################################
from sklearn.metrics import matthews_corrcoef

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE)
    y_true: array binário (1 para Home, 0 para não-Home)
    y_prob: probabilidades previstas para Home
    """
    df_ece = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if df_ece.empty:
        return np.nan
    
    df_ece['bin'] = pd.cut(df_ece['p'], bins=np.linspace(0,1,n_bins+1), include_lowest=True)
    ece = 0.0
    for _, g in df_ece.groupby('bin'):
        if len(g) == 0: 
            continue
        acc = g['y'].mean()
        conf = g['p'].mean()
        ece += (len(g)/len(df_ece)) * abs(acc - conf)
    return ece

# Calcular ECE (para classe Home) e MCC (geral)
ece_val = expected_calibration_error((test_df['Result']=="Home").astype(int), test_df["ML_Proba_Home"])
mcc_val = matthews_corrcoef(test_df['Result'], test_df['ML_Pred'])

# Adicionar ao resumo
summary.update({
    "ECE (Home)": None if np.isnan(ece_val) else round(ece_val, 4),
    "MCC": round(mcc_val, 4)
})

# Mostrar novamente o resumo atualizado
st.subheader("📊 Resumo de Métricas (com extras)")
st.json(summary)





########################################
# BLOCO 11 – CHECKBOXES DE VISUALIZAÇÃO
########################################
st.header("📊 Visualizações (ligue/desligue)")
colv1, colv2, colv3 = st.columns(3)
with colv1:
    show_table = st.checkbox("Mostrar tabela final", value=True)
    show_roi = st.checkbox("ROI acumulado (dia)", value=True)
with colv2:
    show_hist = st.checkbox("Histograma de probabilidades (Home)", value=False)
    show_calib = st.checkbox("Gráfico de calibração (Home)", value=True)
with colv3:
    show_feat_imp = st.checkbox("Importância das features (se disponível)", value=False)

########################################
# BLOCO 11B – Confusion Matrix
########################################
show_cm = st.checkbox("Mostrar Confusion Matrix", value=False)



########################################
# BLOCO 12 – GRÁFICOS
########################################
def plot_roi(df, profit_col, title):
    df = df.copy()
    # Se houver data, usar agrupamento por liga ou por partida; aqui acumulamos por ordem natural
    df['ROI_acu'] = df[profit_col].cumsum()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.arange(len(df)), df['ROI_acu'])
    ax.set_title(title)
    ax.set_xlabel("Jogos (ordem no teste)")
    ax.set_ylabel("ROI acumulado (stake=1)")
    st.pyplot(fig)

def plot_hist_proba(series, title):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(series.dropna(), bins=20)
    ax.set_title(title)
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

def plot_calibration_curve(prob, outcomes, title, n_bins=10):
    # outcomes_bin: 1 se "Home", 0 caso contrário
    dfc = pd.DataFrame({"p": prob, "y": outcomes}).dropna()
    if dfc.empty:
        st.warning("Sem dados suficientes para calibrar.")
        return
    dfc['bin'] = pd.cut(dfc['p'], bins=np.linspace(0,1,n_bins+1), include_lowest=True)
    g = dfc.groupby('bin')
    p_mean = g['p'].mean()
    y_rate = g['y'].mean()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1],'--', label='Linha Perfeita')
    ax.plot(p_mean, y_rate, marker='o', label='Modelo')
    ax.set_title(title)
    ax.set_xlabel('Prob. prevista (média do bin)')
    ax.set_ylabel('Taxa real observada')
    ax.legend()
    st.pyplot(fig)

# Tabela - CORRIGIDO PARA INCLUIR GOLS
if show_table:
    cols_to_show = [
        'Date','League','Home','Away',
        'Goals_H_FT', 'Goals_A_FT',  # GOLS ADICIONADOS AQUI
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
        'Result',
        'Auto_Recommendation','ML_Recommendation',
        'Profit_Auto','Profit_ML',
        'ML_Proba_Home','ML_Proba_Draw','ML_Proba_Away'
    ]
    available_cols = [c for c in cols_to_show if c in test_df.columns]
    st.subheader("📋 Tabela – Teste (regras x ML) - COM GOLS")
    st.dataframe(
        test_df[available_cols]
        .style.format({
            'Odd_H':'{:.2f}','Odd_D':'{:.2f}','Odd_A':'{:.2f}',
            'Odd_1X':'{:.2f}','Odd_X2':'{:.2f}',
            'Profit_Auto':'{:.2f}','Profit_ML':'{:.2f}',
            'ML_Proba_Home':'{:.2f}','ML_Proba_Draw':'{:.2f}','ML_Proba_Away':'{:.2f}',
            'Goals_H_FT':'{:.0f}','Goals_A_FT':'{:.0f}',  # FORMATO PARA GOLS
        }),
        use_container_width=True, height=600
    )

# ROI
if show_roi:
    st.subheader("📈 ROI acumulado")
    if compare_rules and len(auto_bets):
        plot_roi(test_df[test_df['Auto_Recommendation']!='❌ Avoid'], 'Profit_Auto', "ROI – Regras (apenas apostas feitas)")
    if len(ml_bets):
        plot_roi(test_df[test_df['ML_Recommendation']!='❌ Avoid'], 'Profit_ML', "ROI – ML (apenas apostas feitas)")

# Histograma
if show_hist and "ML_Proba_Home" in test_df:
    st.subheader("📊 Histograma – Probabilidade (Home)")
    plot_hist_proba(test_df["ML_Proba_Home"], "Distribuição de Probabilidades (Home)")

# Calibração (Home)
if show_calib and "ML_Proba_Home" in test_df:
    st.subheader("📉 Calibração – Home (modelo vs linha perfeita)")
    y_home = (test_df['Result']=="Home").astype(int)
    plot_calibration_curve(test_df["ML_Proba_Home"], y_home, "Calibração (Home) – Modelo vs Linha Perfeita", n_bins=10)

# Importância das features (somente para modelos com atributo)
if show_feat_imp and hasattr(getattr(model, 'base_estimator_', model), "feature_importances_"):
    st.subheader("🔥 Importância das Features (modelo baseado em árvores)")
    est = getattr(model, 'base_estimator_', model)
    importances = getattr(est, "feature_importances_", None)
    if importances is not None:
        fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(fi.index[::-1], fi.values[::-1])
        ax.set_title("Top 20 Features")
        st.pyplot(fig)
    else:
        st.info("O modelo não expõe 'feature_importances_'.")
elif show_feat_imp:
    st.info("Importância de features disponível apenas para árvores (RF/XGB/LGBM).")


def plot_conf_matrix(y_true, y_pred, labels=["Home","Draw","Away"]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Títulos e eixos
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        ylabel="Resultado Real",
        xlabel="Resultado Previsto",
        title="Matriz de Confusão"
    )
    
    # Colocar os números em cada célula
    fmt = "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    st.pyplot(fig)

# Exibir apenas se ativado
if show_cm:
    st.subheader("🔲 Matriz de Confusão")
    plot_conf_matrix(test_df['Result'], test_df['ML_Pred'])



########################################
# BLOCO 13 – EXPORT
########################################
if save_csv:
    out_cols = [
        'Date','League','Home','Away','Result',
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
        'Auto_Recommendation','ML_Recommendation',
        'Profit_Auto','Profit_ML',
        'ML_Proba_Home','ML_Proba_Draw','ML_Proba_Away',
        '__srcfile'
    ]
    export_cols = [c for c in out_cols if c in test_df.columns]
    csv_bytes = test_df[export_cols].to_csv(index=False).encode('utf-8')
    st.download_button("💾 Baixar CSV de previsões (teste)", data=csv_bytes, file_name="ml_backtest_lab_test.csv", mime="text/csv")

st.success("Pronto! Você pode ajustar datas, mudar o modelo e ligar/desligar os gráficos.")


########################################
# BLOCO 14 – GESTÃO DE BANKROLL AVANÇADA
########################################

class ConservativeBankrollManager:
    """
    Gerenciador conservador de bankroll com:
    - Kelly fractional (1/4)
    - Filtro de EV mínimo (5%)
    - Limite máximo de stake (3%)
    - Tracking de bankroll em tempo real
    """
    
    def __init__(self, initial_bankroll=1000):
        self.initial_br = initial_bankroll
        self.current_br = initial_bankroll
        self.history = []
        self.stake_history = []
        
    def calculate_stake(self, prob, odd, min_ev=0.05, kelly_frac=0.25, max_stake_pct=0.03):
        """
        Calcula stake conservador com múltiplas camadas de proteção
        
        Args:
            prob: Nossa probabilidade estimada (0-1)
            odd: Odd decimal da casa
            min_ev: EV mínimo necessário (default: 5%)
            kelly_frac: Fração do Kelly (default: 1/4)
            max_stake_pct: Stake máximo em % do bankroll (default: 3%)
            
        Returns:
            tuple: (stake_amount, ev, recommendation)
        """
        
        # 1. Calcular EV básico
        ev = calculate_ev(prob, odd)
        
        # 2. Aplicar filtro de EV mínimo
        if pd.isna(ev) or ev < min_ev:
            return 0.0, ev, "❌ EV abaixo do mínimo"
        
        # 3. Calcular Kelly Full
        try:
            kelly_full = (prob * odd - 1) / (odd - 1)
            # Limitar Kelly entre 0 e 1 (evitar edge cases)
            kelly_full = max(0.0, min(kelly_full, 1.0))
        except (ZeroDivisionError, ValueError):
            return 0.0, ev, "❌ Odd inválida para Kelly"
        
        # 4. Aplicar Kelly Fractional
        kelly_frac_calc = kelly_full * kelly_frac
        
        # 5. Aplicar limite máximo de stake
        stake_pct = min(kelly_frac_calc, max_stake_pct)
        
        # 6. Garantir stake mínimo (0.5% do bankroll) para evitar apostas insignificantes
        min_stake_pct = 0.005
        if stake_pct < min_stake_pct:
            return 0.0, ev, "❌ Stake muito pequena"
        
        # 7. Calcular valor absoluto do stake
        stake_amount = stake_pct * self.current_br
        
        return stake_amount, ev, f"✅ Stake: {stake_pct:.2%} (${stake_amount:.2f})"
    
    def place_bet(self, stake_amount, odd, outcome_success):
        """
        Registra uma aposta e atualiza o bankroll
        
        Args:
            stake_amount: Valor apostado
            odd: Odd decimal
            outcome_success: True se ganhou, False se perdeu
            
        Returns:
            float: Profit da aposta
        """
        if outcome_success:
            profit = stake_amount * (odd - 1)
        else:
            profit = -stake_amount
        
        # Atualizar bankroll
        self.current_br += profit
        
        # Registrar no histórico
        bet_record = {
            'stake': stake_amount,
            'odd': odd,
            'outcome': 'win' if outcome_success else 'loss',
            'profit': profit,
            'bankroll_after': self.current_br,
            'timestamp': datetime.now()
        }
        self.history.append(bet_record)
        self.stake_history.append(stake_amount)
        
        return profit
    
    def get_metrics(self):
        """Retorna métricas atuais do bankroll"""
        if not self.history:
            return {
                'current_bankroll': self.current_br,
                'total_profit': self.current_br - self.initial_br,
                'total_return_pct': 0.0,
                'total_bets': 0,
                'win_rate': 0.0,
                'avg_stake_pct': 0.0
            }
        
        wins = sum(1 for bet in self.history if bet['outcome'] == 'win')
        total_bets = len(self.history)
        total_profit = self.current_br - self.initial_br
        
        return {
            'current_bankroll': self.current_br,
            'total_profit': total_profit,
            'total_return_pct': (total_profit / self.initial_br) * 100,
            'total_bets': total_bets,
            'win_rate': (wins / total_bets) * 100 if total_bets > 0 else 0,
            'avg_stake_pct': (np.mean(self.stake_history) / self.initial_br) * 100,
            'max_stake_pct': (np.max(self.stake_history) / self.initial_br) * 100 if self.stake_history else 0
        }
    
    def reset(self):
        """Reseta o bankroll para o estado inicial"""
        self.current_br = self.initial_br
        self.history = []
        self.stake_history = []

# Instância global do bankroll manager
bankroll_mgr = ConservativeBankrollManager(initial_bankroll=1000)



########################################
# BLOCO 15 – UI CONTROLE BANKROLL
########################################

st.header("💰 Gestão de Bankroll Avançada")

col_br1, col_br2, col_br3 = st.columns(3)

with col_br1:
    st.subheader("📊 Configuração")
    initial_br = st.number_input("Bankroll Inicial ($)", min_value=100, max_value=10000, value=1000, step=100)
    min_ev_threshold = st.slider("EV Mínimo (%)", 1, 20, 5, help="EV mínimo para considerar aposta") / 100.0
    kelly_fraction = st.selectbox("Kelly Fraction", [0.125, 0.25, 0.5], index=1, 
                                 format_func=lambda x: f"1/{int(1/x)} Kelly", 
                                 help="Fração conservadora do Kelly")
    max_stake_pct = st.slider("Stake Máximo (% bankroll)", 1, 10, 3, help="Máximo por aposta") / 100.0

with col_br2:
    st.subheader("🎯 Status Atual")
    metrics = bankroll_mgr.get_metrics()
    st.metric("Bankroll Atual", f"${metrics['current_bankroll']:.2f}")
    st.metric("Profit Total", f"${metrics['total_profit']:.2f}")
    st.metric("Return %", f"{metrics['total_return_pct']:.2f}%")
    
with col_br3:
    st.subheader("📈 Performance")
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    st.metric("Total Apostas", metrics['total_bets'])
    st.metric("Stake Médio", f"{metrics['avg_stake_pct']:.1f}%")

# Botão para resetar bankroll
if st.button("🔄 Resetar Bankroll"):
    bankroll_mgr = ConservativeBankrollManager(initial_bankroll=initial_br)
    st.success("Bankroll resetado!")

st.divider()



########################################
# BLOCO 16 – FUNÇÕES ATUALIZADAS PARA STAKING
########################################

def calculate_profit_with_staking(row, bankroll_manager, bet_type="ML"):
    """
    Calcula profit considerando staking proporcional ao bankroll
    e filtros de EV
    
    Args:
        row: Linha do DataFrame com odds e probabilidades
        bankroll_manager: Instância do gerenciador de bankroll
        bet_type: "ML" ou "Auto" para escolher a recomendação
        
    Returns:
        tuple: (profit, stake_amount, ev_value, bet_placed)
    """
    if bet_type == "ML":
        rec = row.get('ML_Recommendation', '❌ Avoid')
        prob_prefix = 'ML_Proba_'
    else:
        rec = row.get('Auto_Recommendation', '❌ Avoid')
        prob_prefix = 'ML_Proba_'  # Usar mesma base de probabilidades para comparação justa
    
    if pd.isna(rec) or rec == '❌ Avoid':
        return 0.0, 0.0, 0.0, False
    
    # Determinar tipo de aposta e obter odd correspondente
    rec_str = str(rec)
    target_result = None
    odd = np.nan
    prob = 0.0
    
    if 'Back Home' in rec_str:
        odd = row.get('Odd_H', np.nan)
        prob = row.get(f'{prob_prefix}Home', 0.0)
        target_result = "Home"
    elif 'Back Away' in rec_str:
        odd = row.get('Odd_A', np.nan)
        prob = row.get(f'{prob_prefix}Away', 0.0)
        target_result = "Away"
    elif 'Back Draw' in rec_str:
        odd = row.get('Odd_D', np.nan)
        prob = row.get(f'{prob_prefix}Draw', 0.0)
        target_result = "Draw"
    elif '1X' in rec_str:
        odd = row.get('Odd_1X', np.nan)
        prob = row.get(f'{prob_prefix}Home', 0.0) + row.get(f'{prob_prefix}Draw', 0.0)
        target_result = ["Home", "Draw"]
    elif 'X2' in rec_str:
        odd = row.get('Odd_X2', np.nan)
        prob = row.get(f'{prob_prefix}Away', 0.0) + row.get(f'{prob_prefix}Draw', 0.0)
        target_result = ["Away", "Draw"]
    else:
        return 0.0, 0.0, 0.0, False
    
    # Verificar se resultado está disponível
    result = row.get('Result')
    if result is None:
        return 0.0, 0.0, 0.0, False
    
    # Calcular stake usando bankroll manager
    stake_amount, ev, stake_reason = bankroll_manager.calculate_stake(
        prob, odd, 
        min_ev=min_ev_threshold,
        kelly_frac=kelly_fraction,
        max_stake_pct=max_stake_pct
    )
    
    # Se stake é zero, não apostar
    if stake_amount == 0:
        return 0.0, 0.0, ev, False
    
    # Determinar se ganhou a aposta
    is_win = False
    if isinstance(target_result, list):
        is_win = result in target_result
    else:
        is_win = result == target_result
    
    # Registrar aposta e obter profit
    profit = bankroll_manager.place_bet(stake_amount, odd, is_win)
    
    return profit, stake_amount, ev, True




########################################
# BLOCO 17 – INTEGRAÇÃO BANKROLL NO CÁLCULO DE PROFITS
########################################

# Inicializar o bankroll manager (colocar depois da UI)
bankroll_mgr = ConservativeBankrollManager(initial_bankroll=initial_br)

# NOVO: Função para calcular todos os profits com staking
def calculate_all_profits_with_staking(test_df, bankroll_manager):
    """
    Calcula profits para ML e Auto usando sistema de staking avançado
    """
    results_ml = []
    results_auto = []
    
    for idx, row in test_df.iterrows():
        # Profit ML com staking
        profit_ml, stake_ml, ev_ml, bet_placed_ml = calculate_profit_with_staking(
            row, bankroll_manager, bet_type="ML"
        )
        
        # Profit Auto com staking (usando bankroll separado para comparação justa)
        profit_auto, stake_auto, ev_auto, bet_placed_auto = calculate_profit_with_staking(
            row, bankroll_manager, bet_type="Auto"
        )
        
        results_ml.append({
            'profit': profit_ml,
            'stake': stake_ml,
            'ev': ev_ml,
            'bet_placed': bet_placed_ml
        })
        
        results_auto.append({
            'profit': profit_auto, 
            'stake': stake_auto,
            'ev': ev_auto,
            'bet_placed': bet_placed_auto
        })
    
    return results_ml, results_auto

# ATUALIZAR: Substituir o cálculo de profits antigo (no Bloco 9)
st.header("💰 Aplicando Sistema de Bankroll")

if st.button("🎯 Calcular Profits com Staking Avançado"):
    with st.spinner("Calculando stakes e profits..."):
        # Fazer backup do bankroll atual
        current_metrics = bankroll_mgr.get_metrics()
        
        # Calcular profits com staking
        results_ml, results_auto = calculate_all_profits_with_staking(test_df, bankroll_mgr)
        
        # Adicionar colunas ao test_df
        test_df['Profit_ML_Stake'] = [r['profit'] for r in results_ml]
        test_df['Stake_ML'] = [r['stake'] for r in results_ml]
        test_df['EV_ML'] = [r['ev'] for r in results_ml]
        test_df['Bet_Placed_ML'] = [r['bet_placed'] for r in results_ml]
        
        test_df['Profit_Auto_Stake'] = [r['profit'] for r in results_auto]
        test_df['Stake_Auto'] = [r['stake'] for r in results_auto]
        test_df['EV_Auto'] = [r['ev'] for r in results_auto]
        test_df['Bet_Placed_Auto'] = [r['bet_placed'] for r in results_auto]
        
        st.success(f"✅ Cálculo completo! {sum(test_df['Bet_Placed_ML'])} apostas ML realizadas")
        
        # Mostrar resumo
        total_stake_ml = test_df['Stake_ML'].sum()
        total_profit_ml = test_df['Profit_ML_Stake'].sum()
        roi_ml = (total_profit_ml / total_stake_ml * 100) if total_stake_ml > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Apostado ML", f"${total_stake_ml:.2f}")
        with col2:
            st.metric("Profit Total ML", f"${total_profit_ml:.2f}")
        with col3:
            st.metric("ROI ML", f"{roi_ml:.1f}%")

st.divider()


########################################
# BLOCO 18 – MÉTRICAS ATUALIZADAS COM STAKING
########################################

# NOVO: Métricas específicas para staking
def calculate_staking_metrics(test_df, profit_col, stake_col, bet_placed_col):
    """Calcula métricas específicas para sistema de staking"""
    bets_df = test_df[test_df[bet_placed_col] == True]
    
    if len(bets_df) == 0:
        return {
            'total_bets': 0,
            'total_stake': 0,
            'total_profit': 0,
            'roi_pct': 0,
            'avg_ev': 0,
            'avg_stake': 0,
            'win_rate': 0
        }
    
    total_stake = bets_df[stake_col].sum()
    total_profit = bets_df[profit_col].sum()
    wins = len(bets_df[bets_df[profit_col] > 0])
    
    return {
        'total_bets': len(bets_df),
        'total_stake': total_stake,
        'total_profit': total_profit,
        'roi_pct': (total_profit / total_stake * 100) if total_stake > 0 else 0,
        'avg_ev': bets_df['EV_ML'].mean() if 'EV_ML' in bets_df else 0,
        'avg_stake': bets_df[stake_col].mean(),
        'win_rate': (wins / len(bets_df)) * 100
    }

# ATUALIZAR: Mostrar métricas de staking no lugar das antigas
st.header("📊 Métricas com Staking Avançado")

if 'Profit_ML_Stake' in test_df.columns:
    col_met1, col_met2 = st.columns(2)
    
    with col_met1:
        st.subheader("🤖 ML com Staking")
        ml_metrics = calculate_staking_metrics(test_df, 'Profit_ML_Stake', 'Stake_ML', 'Bet_Placed_ML')
        st.metric("Total Apostas", ml_metrics['total_bets'])
        st.metric("Total Stake", f"${ml_metrics['total_stake']:.2f}")
        st.metric("Profit Total", f"${ml_metrics['total_profit']:.2f}")
        st.metric("ROI", f"{ml_metrics['roi_pct']:.1f}%")
        st.metric("Win Rate", f"{ml_metrics['win_rate']:.1f}%")
        st.metric("EV Médio", f"{ml_metrics['avg_ev']:.3f}")
    
    with col_met2:
        st.subheader("📋 Regras com Staking")
        auto_metrics = calculate_staking_metrics(test_df, 'Profit_Auto_Stake', 'Stake_Auto', 'Bet_Placed_Auto')
        st.metric("Total Apostas", auto_metrics['total_bets'])
        st.metric("Total Stake", f"${auto_metrics['total_stake']:.2f}")
        st.metric("Profit Total", f"${auto_metrics['total_profit']:.2f}")
        st.metric("ROI", f"{auto_metrics['roi_pct']:.1f}%")
        st.metric("Win Rate", f"{auto_metrics['win_rate']:.1f}%")
        st.metric("EV Médio", f"{auto_metrics['avg_ev']:.3f}")

else:
    st.info("👆 Clique em 'Calcular Profits com Staking Avançado' para ver as métricas")



########################################
# BLOCO 19 – VISUALIZAÇÕES ATUALIZADAS
########################################

# NOVO: Gráfico de evolução do bankroll
def plot_bankroll_evolution(test_df, profit_col, title):
    """Plota evolução do bankroll com staking"""
    if profit_col not in test_df.columns:
        return
    
    bets_only = test_df[test_df[profit_col] != 0].copy()
    if len(bets_only) == 0:
        return
    
    cumulative_profit = bets_only[profit_col].cumsum() + 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(cumulative_profit)), cumulative_profit, linewidth=2)
    ax.set_title(f"{title} - Evolução do Bankroll")
    ax.set_xlabel("Número de Apostas")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True, alpha=0.3)
    
    # Adicionar linha do bankroll inicial
    ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Bankroll Inicial')
    ax.legend()
    
    st.pyplot(fig)

# ATUALIZAR: Adicionar gráficos de bankroll às visualizações
st.header("📈 Visualizações com Staking")

if 'Profit_ML_Stake' in test_df.columns:
    plot_bankroll_evolution(test_df, 'Profit_ML_Stake', "ML")
    
    if compare_rules:
        plot_bankroll_evolution(test_df, 'Profit_Auto_Stake', "Regras")
