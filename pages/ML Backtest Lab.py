########################################
# BLOCO 1 – IMPORTS & CONFIG
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
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa", "afc"]
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
# BLOCO 3 – HELPERS DE DADOS
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
    # Requer gols finais para ter rótulo
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

def compute_double_chance_odds(df):
    for c in ['Odd_H','Odd_D','Odd_A']:
        if c not in df.columns: df[c] = np.nan
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df['Odd_H']
    probs['p_D'] = 1 / df['Odd_D']
    probs['p_A'] = 1 / df['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)
    df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

def classify_leagues_variation(history_df):
    hist = history_df.copy()
    agg = (
        hist.groupby('League')
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
# BLOCO 5 – UI: DADOS & BACKTEST
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
history['Home_Band'] = np.where(
    history['M_H'] <= history['Home_P20'], 'Bottom 20%',
    np.where(history['M_H'] >= history['Home_P80'], 'Top 20%', 'Balanced')
)
history['Away_Band'] = np.where(
    history['M_A'] <= history['Away_P20'], 'Bottom 20%',
    np.where(history['M_A'] >= history['Away_P80'], 'Top 20%', 'Balanced')
)
history['Dominant'] = history.apply(dominant_side, axis=1)
history['Result'] = history.apply(map_result, axis=1)

# Intervalo de backtest
valid_dates = history['Date'].dropna()
min_d, max_d = (valid_dates.min(), valid_dates.max()) if not valid_dates.empty else (None, None)

colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Data inicial (treino)", value=min_d.date() if min_d else None)
with colB:
    end_date = st.date_input("Data final (teste)", value=max_d.date() if max_d else None)

if start_date and end_date and start_date > end_date:
    st.error("A data inicial deve ser anterior ou igual à data final.")
    st.stop()

# Split por data (simples): treino = < end_date - N dias | teste = <= end_date
lookback_days = st.number_input("Tamanho do período de treino (dias) antes da data final", 14, 400, 60)
end_dt = pd.to_datetime(end_date) if end_date else max_d
train_start_dt = (end_dt - timedelta(days=int(lookback_days)))

train_mask = (history['Date'] >= train_start_dt) & (history['Date'] < end_dt)
test_mask  = (history['Date'] == end_dt)

train_df = history[train_mask].copy()
test_df  = history[test_mask].copy()

st.info(f"Treino: {train_df.shape[0]} jogos | Teste (data={end_dt.date() if end_dt else 'N/A'}): {test_df.shape[0]} jogos")

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
# BLOCO 7 – Ajustado: Função build_X
########################################
def build_X(df, fit_encoder=False, encoder=None, cat_cols=None):
    """
    Prepara o dataframe de entrada (X) para treino ou teste:
    - Garante que todas as colunas de features existam
    - Faz mapeamento de bandas para valores numéricos
    - Aplica OneHotEncoder nas colunas categóricas
    - Retorna dataframe final somente com valores numéricos
    """
    # Garantir todas as features presentes
    for col in features_raw:
        if col not in df.columns:
            df[col] = np.nan

    X = df[features_raw].copy()

    # Mapear Home_Band e Away_Band
    if 'Home_Band' in X.columns:
        X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
    if 'Away_Band' in X.columns:
        X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

    # Identificar colunas categóricas
    if cat_cols is None:
        cat_cols = [c for c in ['Dominant','League_Classification'] if c in X.columns]

    # Fit ou Transform do encoder
    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(X[cat_cols]) if cat_cols else np.zeros((len(X),0))
    else:
        encoded = encoder.transform(X[cat_cols]) if (encoder and cat_cols) else np.zeros((len(X),0))

    # Converter para dataframe
    encoded_df = (
        pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
        if encoded.size else pd.DataFrame(index=X.index)
    )

    # Combinar numéricas + one-hot
    X_num = X.drop(columns=cat_cols, errors='ignore').reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    X_out = pd.concat([X_num, encoded_df], axis=1)

    # Garantir apenas valores numéricos e sem NaN
    X_out = X_out.apply(pd.to_numeric, errors='coerce')
    X_out.fillna(0, inplace=True)

    return X_out, encoder, cat_cols

# ########################################
# # BLOCO 8 – TREINO, PREDIÇÃO, CALIBRAÇÃO
# ########################################
# def make_model(choice, params):
#     if choice == "Random Forest":
#         return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
#     if choice == "Logistic Regression":
#         return LogisticRegression(random_state=42, **params)
#     if choice == "XGBoost" and XGB_AVAILABLE:
#         return XGBClassifier(random_state=42, eval_metric="logloss", **params)
#     if choice == "LightGBM" and LGBM_AVAILABLE:
#         return LGBMClassifier(random_state=42, **params)
#     raise ValueError("Modelo não suportado/indisponível no ambiente.")

# with st.spinner("Treinando modelo..."):
#     base_model = make_model(model_choice, params)
#     if apply_calibration:
#         model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
#     else:
#         model = base_model
#     model.fit(X_train, y_train)

# proba_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
# pred_test  = model.predict(X_test)

# classes_ = list(model.classes_)
# # Mapear probabilidades por classe
# def p(cls):
#     if proba_test is None: return np.zeros(len(X_test))
#     idx = classes_.index(cls) if cls in classes_ else None
#     return proba_test[:, idx] if idx is not None else np.zeros(len(X_test))

# test_df = test_df.copy()
# test_df["ML_Proba_Home"] = p("Home")
# test_df["ML_Proba_Draw"] = p("Draw")
# test_df["ML_Proba_Away"] = p("Away")
# test_df["ML_Pred"] = pred_test

# # Recomendação a partir das probabilidades
# st.subheader("🎯 Limiar para Back direto")
# threshold = st.slider("Threshold (%) para Back Home/Away", 50, 85, 65, step=1) / 100.0

# def ml_rec_from_proba(row, thr=0.65):
#     ph, pd_, pa = row['ML_Proba_Home'], row['ML_Proba_Draw'], row['ML_Proba_Away']
#     if ph >= thr: return "🟢 Back Home"
#     if pa >= thr: return "🟠 Back Away"
#     sum_hd, sum_ad = ph + pd_, pa + pd_
#     if abs(ph - pa) < 0.05 and pd_ > 0.35:
#         return "⚪ Back Draw"
#     if sum_hd > sum_ad:  return "🟦 1X (Home/Draw)"
#     if sum_ad > sum_hd:  return "🟪 X2 (Away/Draw)"
#     return "❌ Avoid"

# test_df["ML_Recommendation"] = test_df.apply(ml_rec_from_proba, axis=1, thr=threshold)


########################################
# SUBBLOCO 8A – Limpeza antes do treino
########################################
st.subheader("🔍 Pré-validação dos dados antes do treino")

# Remover qualquer linha com NaN em X_train
mask = ~X_train.isnull().any(axis=1)
X_train = X_train.loc[mask].copy()
y_train = y_train.loc[mask].copy()

# Garantir tipos numéricos
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Validar shapes
st.write("Dimensões finais após limpeza:")
st.write("X_train:", X_train.shape)
st.write("y_train:", y_train.shape)

# Conferir se os dados estão alinhados
if len(X_train) != len(y_train):
    st.error(f"Desalinhamento detectado! X_train tem {len(X_train)} linhas e y_train tem {len(y_train)} linhas.")
    st.stop()

# Conferir tipos de cada coluna
st.write("Tipos de dados em X_train:")
st.write(X_train.dtypes)

########################################
# SUBBLOCO 8B – Treinamento seguro
########################################
with st.spinner("Treinando modelo..."):
    base_model = make_model(model_choice, params)
    
    if apply_calibration:
        model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
    else:
        model = base_model

    try:
        model.fit(X_train, y_train)
        st.success("Treinamento concluído com sucesso!")
    except ValueError as e:
        st.error("Erro durante o treinamento. Confira detalhes abaixo:")
        st.code(str(e))
        st.stop()


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
# BLOCO 9 – COMPARAÇÃO COM REGRAS & PROFIT
########################################
# Preparar insumos de regras no conjunto de teste (usa quantis calculados no histórico)
test_df = test_df.merge(league_class, on='League', how='left')
test_df = test_df.merge(league_bands, on='League', how='left')

test_df['M_Diff'] = test_df['M_H'] - test_df['M_A']
test_df['Home_Band'] = np.where(
    test_df['M_H'] <= test_df['Home_P20'], 'Bottom 20%',
    np.where(test_df['M_H'] >= test_df['Home_P80'], 'Top 20%', 'Balanced')
)
test_df['Away_Band'] = np.where(
    test_df['M_A'] <= test_df['Away_P20'], 'Bottom 20%',
    np.where(test_df['M_A'] >= test_df['Away_P80'], 'Top 20%', 'Balanced')
)
test_df['Dominant'] = test_df.apply(dominant_side, axis=1)

if compare_rules:
    test_df['Auto_Recommendation'] = test_df.apply(auto_recommendation, axis=1)
else:
    test_df['Auto_Recommendation'] = np.nan

# Resultado verdadeiro (já temos em history/test_df)
# Lucros
test_df['Profit_ML'] = test_df.apply(lambda r: calculate_profit(r['ML_Recommendation'], r['Result'], r), axis=1)
test_df['Profit_Auto'] = test_df.apply(lambda r: calculate_profit(r['Auto_Recommendation'], r['Result'], r), axis=1)

# Correção (acerto/erro)
test_df['ML_Correct'] = test_df.apply(lambda r: check_recommendation(r['ML_Recommendation'], r['Result']), axis=1)
test_df['Auto_Correct'] = test_df.apply(lambda r: check_recommendation(r['Auto_Recommendation'], r['Result']), axis=1)

########################################
# BLOCO 10 – MÉTRICAS & SUMÁRIOS
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
    # Espaço p/ multi-model no futuro
    # show_compare_models = st.checkbox("Comparar múltiplos modelos", value=False)

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

# Tabela
if show_table:
    cols_to_show = [
        'Date','League','Home','Away',
        'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
        'Result',
        'Auto_Recommendation','ML_Recommendation',
        'Profit_Auto','Profit_ML',
        'ML_Proba_Home','ML_Proba_Draw','ML_Proba_Away'
    ]
    available_cols = [c for c in cols_to_show if c in test_df.columns]
    st.subheader("📋 Tabela – Teste (regras x ML)")
    st.dataframe(
        test_df[available_cols]
        .style.format({
            'Odd_H':'{:.2f}','Odd_D':'{:.2f}','Odd_A':'{:.2f}',
            'Odd_1X':'{:.2f}','Odd_X2':'{:.2f}',
            'Profit_Auto':'{:.2f}','Profit_ML':'{:.2f}',
            'ML_Proba_Home':'{:.2f}','ML_Proba_Draw':'{:.2f}','ML_Proba_Away':'{:.2f}',
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
