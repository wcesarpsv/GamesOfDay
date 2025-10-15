# ########################################################
# Bloco 1 – Imports & Config
# ########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from datetime import date, timedelta, datetime
from collections import Counter

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SMOTE para balanceamento
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Bet Indicator – Triple View", layout="wide")
st.title("📊 Bet Indicator – Triple View (1X2 + OU + BTTS)")

# Paths
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "copas", "uefa", "nordeste", "afc","trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


# ########################################################
# Bloco 2 – Funções auxiliares (ATUALIZADAS DO BINARY)
# ########################################################
def load_all_games(folder):
    """Carrega todos os CSVs da pasta e remove duplicados por (Date, Home, Away)."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    if not df_list:
        return pd.DataFrame()
    
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all.drop_duplicates(subset=["Date", "Home", "Away","Goals_H_FT","Goals_A_FT"], keep="first")

def filter_leagues(df):
    """Remove ligas indesejadas (Copa, UEFA, etc)."""
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def load_selected_csvs(folder):
    """Carrega CSVs selecionados com sistema de data do Binary"""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    
    # Últimos dois arquivos (Hoje e Ontem) - igual ao código Binary
    options = files[-2:] if len(files) >= 2 else files
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)
    
    # Extrair a data do arquivo selecionado (YYYY-MM-DD)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    if date_match:
        selected_date_str = date_match.group(0)
    else:
        selected_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Carregar o arquivo selecionado
    games_today = pd.read_csv(os.path.join(folder, selected_file))
    games_today = filter_leagues(games_today)
    
    # ========== MERGE COM LIVESCORE (DO BINARY) ==========
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

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

    # 🔹 Mantém apenas jogos futuros (sem placares ainda) - baseado nos dados originais
    if 'Goals_H_FT' in games_today.columns:
        games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

    return games_today

def save_model(model, filename):
    path = os.path.join(MODELS_FOLDER, filename)
    with open(path, "wb") as f:
        joblib.dump(model, f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None


# ########################################################
# Bloco 3 – Carregar Dados (ATUALIZADO)
# ########################################################
st.info("📂 Loading data...")

# Carregar dados históricos com função melhorada do Binary
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

if history.empty:
    st.error("⚠️ No valid historical data found in GamesDay.")
    st.stop()

# Carregar jogos de hoje com sistema melhorado
games_today = load_selected_csvs(GAMES_FOLDER)

if games_today.empty:
    st.error("⚠️ No valid matches selected.")
    st.stop()


# ########################################################
# Bloco 4 – Targets
# ########################################################
history["Target"] = history.apply(
    lambda row: 0 if row["Goals_H_FT"] > row["Goals_A_FT"]
    else (1 if row["Goals_H_FT"] == row["Goals_A_FT"] else 2),
    axis=1,
)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)


# ########################################################
# Bloco 5 – Features & One-Hot Leagues
# ########################################################
# Adicionar features de momentum do Binary
def add_momentum_features(df):
    df['PesoMomentum_H'] = abs(df['M_H']) / (abs(df['M_H']) + abs(df['M_A']))
    df['PesoMomentum_A'] = abs(df['M_A']) / (abs(df['M_H']) + abs(df['M_A']))
    df['CustoMomentum_H'] = df.apply(
        lambda x: x['Odd_H'] / abs(x['M_H']) if abs(x['M_H']) > 0 else np.nan, axis=1
    )
    df['CustoMomentum_A'] = df.apply(
        lambda x: x['Odd_A'] / abs(x['M_A']) if abs(x['M_A']) > 0 else np.nan, axis=1
    )
    return df

history = add_momentum_features(history)
games_today = add_momentum_features(games_today)

history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]
history['Diff_Abs'] = (history['M_H'] - history['M_A']).abs()
games_today['Diff_Abs'] = (games_today['M_H'] - games_today['M_A']).abs()

# Features atualizadas com momentum
features_1x2 = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A", 
                "Diff_Abs", "PesoMomentum_H", "PesoMomentum_A", "CustoMomentum_H", "CustoMomentum_A"]
features_ou_btts = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "OU_Total",
                   "Diff_Abs", "PesoMomentum_H", "PesoMomentum_A", "CustoMomentum_H", "CustoMomentum_A","OverScore_Home","OverScore_Away"]

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X_1x2 = pd.concat([history[features_1x2], history_leagues], axis=1)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1)

X_today_1x2 = pd.concat([games_today[features_1x2], games_today_leagues], axis=1)
X_today_ou = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)
X_today_btts = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)




########################################
#### Bloco 5.1 – XG2 + α por Liga (Skellam 1X2)
########################################
from math import isfinite
from scipy.stats import skellam
import json

st.markdown("#### ⚙️ Otimização de α por Liga (Skellam 1X2)")

# ===== 1) Conversões base (reaproveita Odds + Momentum) =====
def odds_to_mu(odd_home, odd_draw, odd_away):
    if pd.isna(odd_home) or pd.isna(odd_draw) or pd.isna(odd_away):
        return np.nan, np.nan
    if min(odd_home, odd_away) <= 1.0:
        return np.nan, np.nan
    inv = (1/odd_home + 1/odd_draw + 1/odd_away)
    p_home = (1/odd_home) / inv
    p_away = (1/odd_away) / inv
    # mapeamento simples p→μ (ajuste fino depois, se quiser)
    mu_h = 0.4 + 2.4 * p_home
    mu_a = 0.4 + 2.4 * p_away
    return mu_h, mu_a

def xg_from_momentum(row):
    base = 1.3
    denom = abs(row.get("M_H", 0.0)) + abs(row.get("M_A", 0.0)) + 1e-6
    mu_h = base + 0.8 * (row.get("M_H", 0.0)/denom) + 0.4 * (row.get("Diff_Power", 0.0)/100)
    mu_a = base + 0.8 * (row.get("M_A", 0.0)/denom) - 0.4 * (row.get("Diff_Power", 0.0)/100)
    return max(mu_h, 0.05), max(mu_a, 0.05)

def blend_xg(row, alpha):
    mu_odd_h, mu_odd_a = odds_to_mu(row["Odd_H"], row["Odd_D"], row["Odd_A"])
    mu_perf_h, mu_perf_a = xg_from_momentum(row)
    if not (np.isfinite(mu_odd_h) and np.isfinite(mu_odd_a) and np.isfinite(mu_perf_h) and np.isfinite(mu_perf_a)):
        return np.nan, np.nan
    mu_h = alpha * mu_odd_h + (1 - alpha) * mu_perf_h
    mu_a = alpha * mu_odd_a + (1 - alpha) * mu_perf_a
    # clamps leves para estabilidade numérica da Skellam
    mu_h = float(np.clip(mu_h, 0.05, 5.0))
    mu_a = float(np.clip(mu_a, 0.05, 5.0))
    return mu_h, mu_a

def skellam_probs_1x2(mu_h, mu_a):
    """Retorna (p_home, p_draw, p_away) usando Skellam."""
    p_home = 1 - skellam.cdf(0, mu_h, mu_a)
    p_draw = skellam.pmf(0, mu_h, mu_a)
    p_away = skellam.cdf(-1, mu_h, mu_a)
    return float(p_home), float(p_draw), float(p_away)

def mc_logloss_1x2(p, y):
    """LogLoss multiclasses: y ∈ {0:Home,1:Draw,2:Away}; p = (pH,pD,pA)."""
    eps = 1e-12
    return -np.log(max(p[y], eps))

# ===== 2) Hiperparâmetros de otimização =====
alpha_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)  # 0.00, 0.05, ..., 1.00
alpha_global_prior = st.sidebar.slider("α global (prior)", 0.0, 1.0, 0.50, 0.05)
shrinkage_m = st.sidebar.slider("Força da suavização (m)", 50, 1000, 300, 50)
min_samples_per_league = st.sidebar.slider("Mínimo de jogos/ligas para α próprio", 50, 1000, 200, 50)

# ===== 3) Preparar base histórica limpa e target 1X2 =====
hist = history.copy()
hist = hist.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Odd_H", "Odd_D", "Odd_A", "League"])
if hist.empty:
    st.warning("Histórico insuficiente para otimizar α por liga.")
    alpha_by_league = {}
else:
    # target 1X2
    hist["Target_1X2"] = np.where(
        hist["Goals_H_FT"] > hist["Goals_A_FT"], 0,
        np.where(hist["Goals_H_FT"] < hist["Goals_A_FT"], 2, 1)
    )

    # ===== 4) Otimização por liga =====
    per_league_rows = []
    alpha_raw = {}   # melhor α puro por liga (sem shrink)
    n_by_lg = {}

    leagues = hist["League"].dropna().unique().tolist()
    for lg in leagues:
        df_lg = hist[hist["League"] == lg].copy()
        df_lg = df_lg.dropna(subset=["Odd_H", "Odd_D", "Odd_A"])
        if len(df_lg) < 5:
            continue

        # Avaliar grid de α
        best_alpha, best_ll = None, np.inf
        for a in alpha_grid:
            ll_sum, n_ok = 0.0, 0
            for _, r in df_lg.iterrows():
                mu_pair = blend_xg(r, a)
                if not np.isfinite(mu_pair[0]) or not np.isfinite(mu_pair[1]):
                    continue
                pH, pD, pA = skellam_probs_1x2(mu_pair[0], mu_pair[1])
                y = int(r["Target_1X2"])
                ll_sum += mc_logloss_1x2((pH, pD, pA), y)
                n_ok += 1
            if n_ok >= 20 and ll_sum / n_ok < best_ll:
                best_ll = ll_sum / n_ok
                best_alpha = float(a)

        if best_alpha is not None:
            alpha_raw[lg] = best_alpha
            n_by_lg[lg] = len(df_lg)
            per_league_rows.append({"League": lg, "N": len(df_lg), "Alpha_raw": best_alpha, "LogLoss": round(best_ll, 4)})

    # ===== 5) Suavização Bayesiana (shrink para α global) =====
    # alpha_shrunk = (n / (n + m)) * alpha_raw + (m / (n + m)) * alpha_global_prior
    alpha_by_league = {}
    for lg, a_raw in alpha_raw.items():
        n = n_by_lg.get(lg, 0)
        if n < min_samples_per_league:
            w = n / (n + shrinkage_m)
        else:
            w = n / (n + shrinkage_m)  # ainda suaviza, só que pesa mais os dados
        a_shr = float(w * a_raw + (1 - w) * alpha_global_prior)
        alpha_by_league[lg] = round(a_shr, 3)

    # Tabela de diagnóstico
    if per_league_rows:
        df_alpha = pd.DataFrame(per_league_rows).sort_values(["N"], ascending=False)
        df_alpha["Alpha_shrunk"] = df_alpha["League"].map(alpha_by_league)
        st.markdown("##### 📊 α por liga (raw vs shrunk)")
        st.dataframe(df_alpha, use_container_width=True)

# ===== 6) Persistir mapeamento (opcional)
try:
    path_alpha = os.path.join(MODELS_FOLDER, "alpha_by_league.json")
    with open(path_alpha, "w", encoding="utf-8") as f:
        json.dump({"alpha_global": alpha_global_prior,
                   "alpha_by_league": alpha_by_league,
                   "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
    st.caption(f"α por liga salvo em {path_alpha}")
except Exception as e:
    st.warning(f"Não foi possível salvar alpha_by_league: {e}")

# ===== 7) Aplicar α por liga para jogos de hoje e calcular XG2 =====
def get_alpha_for_league(lg):
    if isinstance(lg, str) and lg in alpha_by_league:
        return alpha_by_league[lg]
    return alpha_global_prior  # fallback

def compute_xg2_row_today(row):
    a = get_alpha_for_league(row.get("League", None))
    mu_h, mu_a = blend_xg(row, a)
    return mu_h, mu_a, a

games_today["XG2_H"], games_today["XG2_A"], games_today["Alpha_League"] = zip(
    *games_today.apply(compute_xg2_row_today, axis=1)
)

########################################
#### Bloco 5.2 – α por Liga (Skellam Over/Under 2.5)
########################################
from scipy.stats import poisson

st.markdown("#### ⚙️ Otimização de α por Liga (Over/Under 2.5)")

def prob_over25(mu_h, mu_a):
    """P(Over 2.5 gols) assumindo Poisson independentes."""
    p_under = 0.0
    for i in range(3):
        for j in range(3 - i):
            p_under += poisson.pmf(i, mu_h) * poisson.pmf(j, mu_a)
    return 1 - p_under

def bin_logloss(p, y):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))

hist_ou = history.dropna(subset=["Goals_H_FT","Goals_A_FT","League","Odd_H","Odd_D","Odd_A"]).copy()
hist_ou["Target_OU25"] = (hist_ou["Goals_H_FT"] + hist_ou["Goals_A_FT"] > 2.5).astype(int)

alpha_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
alpha_by_league_ou = {}
records = []

for lg, df_lg in hist_ou.groupby("League"):
    if len(df_lg) < 20:
        continue
    best_alpha, best_ll = None, np.inf
    for a in alpha_grid:
        ll_sum, n_ok = 0.0, 0
        for _, r in df_lg.iterrows():
            mu_h, mu_a = blend_xg(r, a)
            if not (np.isfinite(mu_h) and np.isfinite(mu_a)):
                continue
            p_over = prob_over25(mu_h, mu_a)
            ll_sum += bin_logloss(p_over, r["Target_OU25"])
            n_ok += 1
        if n_ok >= 20 and ll_sum / n_ok < best_ll:
            best_alpha, best_ll = a, ll_sum / n_ok
    if best_alpha is not None:
        records.append({"League": lg, "Alpha_OU25": best_alpha, "LogLoss": round(best_ll, 4), "N": len(df_lg)})
        alpha_by_league_ou[lg] = best_alpha

if records:
    df_ou = pd.DataFrame(records).sort_values("N", ascending=False)
    st.dataframe(df_ou, use_container_width=True)

def get_alpha_ou(lg):
    return alpha_by_league_ou.get(lg, alpha_global_prior)

def compute_xg2_ou(row):
    a = get_alpha_ou(row.get("League"))
    mu_h, mu_a = blend_xg(row, a)
    return mu_h, mu_a, a

games_today["XG2_H_OU"], games_today["XG2_A_OU"], games_today["Alpha_OU25"] = zip(
    *games_today.apply(compute_xg2_ou, axis=1)
)


########################################
#### Bloco 5.3 – α por Liga (Skellam BTTS)
########################################
st.markdown("#### ⚙️ Otimização de α por Liga (BTTS Yes/No)")

def prob_btts_yes(mu_h, mu_a):
    """P(Ambos marcam) assumindo Poisson independentes."""
    p_no = poisson.pmf(0, mu_h) + poisson.pmf(0, mu_a) - poisson.pmf(0, mu_h)*poisson.pmf(0, mu_a)
    return 1 - p_no

hist_btts = history.dropna(subset=["Goals_H_FT","Goals_A_FT","League","Odd_H","Odd_D","Odd_A"]).copy()
hist_btts["Target_BTTS"] = ((hist_btts["Goals_H_FT"] > 0) & (hist_btts["Goals_A_FT"] > 0)).astype(int)

alpha_by_league_btts = {}
records = []

for lg, df_lg in hist_btts.groupby("League"):
    if len(df_lg) < 20:
        continue
    best_alpha, best_ll = None, np.inf
    for a in alpha_grid:
        ll_sum, n_ok = 0.0, 0
        for _, r in df_lg.iterrows():
            mu_h, mu_a = blend_xg(r, a)
            if not (np.isfinite(mu_h) and np.isfinite(mu_a)):
                continue
            p_yes = prob_btts_yes(mu_h, mu_a)
            ll_sum += bin_logloss(p_yes, r["Target_BTTS"])
            n_ok += 1
        if n_ok >= 20 and ll_sum / n_ok < best_ll:
            best_alpha, best_ll = a, ll_sum / n_ok
    if best_alpha is not None:
        records.append({"League": lg, "Alpha_BTTS": best_alpha, "LogLoss": round(best_ll, 4), "N": len(df_lg)})
        alpha_by_league_btts[lg] = best_alpha

if records:
    df_btts = pd.DataFrame(records).sort_values("N", ascending=False)
    st.dataframe(df_btts, use_container_width=True)

def get_alpha_btts(lg):
    return alpha_by_league_btts.get(lg, alpha_global_prior)

def compute_xg2_btts(row):
    a = get_alpha_btts(row.get("League"))
    mu_h, mu_a = blend_xg(row, a)
    return mu_h, mu_a, a

games_today["XG2_H_BTTS"], games_today["XG2_A_BTTS"], games_today["Alpha_BTTS"] = zip(
    *games_today.apply(compute_xg2_btts, axis=1)
)






# ########################################################
# Bloco 6 – Configurações ML (Sidebar) - ATUALIZADO
# ########################################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"]
)
use_smote = st.sidebar.checkbox("Use SMOTE for balancing", value=True)
retrain = st.sidebar.checkbox("Retrain models", value=False)

st.sidebar.markdown("""
**ℹ️ Usage recommendations:**
- 🔹 *Random Forest*: simple and fast baseline.  
- 🔹 *Random Forest Tuned*: suitable for market **1X2**.  
- 🔹 *XGBoost Tuned*: suitable for markets **Over/Under 2.5** e **BTTS**.  
- 🔹 *SMOTE*: recommended for imbalanced datasets
""")


# ########################################################
# Bloco 7 – Treino & Avaliação (COM SMOTE - CORRIGIDO)
# ########################################################
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_fc.pkl"
    model = None

    if not retrain:
        model = load_model(filename)

    # 🔥 CORREÇÃO: Limpeza CONSISTENTE de dados
    # Remover NaN e infinitos ANTES do split
    X_clean = X.copy()
    y_clean = y.copy()
    
    # Combinar para limpeza consistente
    data_clean = X_clean.copy()
    data_clean['target'] = y_clean
    
    # Remover linhas com NaN ou infinitos
    data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
    data_clean = data_clean.dropna()
    
    if data_clean.empty:
        st.error(f"❌ No valid data after cleaning for {name}")
        return {}, None
        
    # Separar novamente mantendo a mesma ordem
    X_clean = data_clean.drop('target', axis=1)
    y_clean = data_clean['target']
    
    st.info(f"📊 Dataset {name}: {len(X_clean)} samples after cleaning")

    # 🔥 CORREÇÃO CRÍTICA: Garantir que as colunas são as mesmas
    # Salvar os nomes das colunas originais
    original_columns = X_clean.columns.tolist()
    
    # Split dos dados LIMPOS
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # 🔥 GARANTIR que validação tem as mesmas colunas que treino
    X_val = X_val[original_columns]
    
    # Aplicar SMOTE se selecionado
    if use_smote:
        st.info(f"🔄 Applying SMOTE for {name} (before: {dict(Counter(y_train))})")
        try:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.info(f"📊 After SMOTE: {dict(Counter(y_train))}")
        except Exception as e:
            st.error(f"❌ SMOTE failed for {name}: {e}")
            st.warning("🔄 Continuing without SMOTE...")

    if model is None:
        if ml_model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")

        elif ml_model_choice == "Random Forest Tuned":
            rf_params = {
                "1X2": {'n_estimators': 600, 'max_depth': 14, 'min_samples_split': 10,
                        'min_samples_leaf': 1, 'max_features': 'sqrt'},
                "OverUnder25": {'n_estimators': 600, 'max_depth': 5, 'min_samples_split': 9,
                                'min_samples_leaf': 3, 'max_features': 'sqrt'},
                "BTTS": {'n_estimators': 400, 'max_depth': 18, 'min_samples_split': 4,
                         'min_samples_leaf': 5, 'max_features': 'sqrt'},
            }
            model = RandomForestClassifier(random_state=42, class_weight="balanced_subsample", **rf_params[name])

        elif ml_model_choice == "XGBoost Tuned":
            xgb_params = {
                "1X2": {'n_estimators': 219, 'max_depth': 9, 'learning_rate': 0.05,
                        'subsample': 0.9, 'colsample_bytree': 0.8,
                        'eval_metric': 'mlogloss', 'use_label_encoder': False},
                "OverUnder25": {'n_estimators': 488, 'max_depth': 10, 'learning_rate': 0.03,
                                'subsample': 0.9, 'colsample_bytree': 0.7,
                                'eval_metric': 'logloss', 'use_label_encoder': False},
                "BTTS": {'n_estimators': 695, 'max_depth': 6, 'learning_rate': 0.04,
                         'subsample': 0.8, 'colsample_bytree': 0.8,
                         'eval_metric': 'logloss', 'use_label_encoder': False},
            }
            model = XGBClassifier(random_state=42, **xgb_params[name])

        model.fit(X_train, y_train)
        save_model(model, filename)

    # 🔥 CORREÇÃO: Garantir que X_val tem as mesmas colunas que o modelo espera
    missing_cols = set(model.feature_names_in_) - set(X_val.columns)
    extra_cols = set(X_val.columns) - set(model.feature_names_in_)
    
    if missing_cols:
        st.warning(f"⚠️ Adding missing columns to validation set: {missing_cols}")
        for col in missing_cols:
            X_val[col] = 0
    
    if extra_cols:
        st.warning(f"⚠️ Removing extra columns from validation set: {extra_cols}")
        X_val = X_val[model.feature_names_in_]
    
    # Agora fazer as previsões
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)

    if num_classes == 2:
        bs = brier_score_loss(y_val, probs[:, 1])
        bs = f"{bs:.3f}"
    else:
        y_onehot = pd.get_dummies(y_val).values
        bs_raw = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        bs = f"{bs_raw:.3f} (multi)"

    metrics = {
        "Model": f"{ml_model_choice} - {name}",
        "Accuracy": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
        "Brier": bs,
        "SMOTE": "Yes" if use_smote else "No",
        "Samples": len(X_clean)
    }

    return metrics, model

# ########################################################
# Bloco 8 – Treinar Modelos
# ########################################################
stats = []
res, model_multi = train_and_evaluate(X_1x2, history["Target"], "1X2", 3)
stats.append(res)
res, model_ou = train_and_evaluate(X_ou, history["Target_OU25"], "OverUnder25", 2)
stats.append(res)
res, model_btts = train_and_evaluate(X_btts, history["Target_BTTS"], "BTTS", 2)
stats.append(res)

df_stats = pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)


# ########################################################
# Bloco 9 – Previsões (COM CORREÇÃO DE FEATURES)
# ########################################################

def safe_predict_proba(model, X_data, feature_names):
    """Previsão segura com alinhamento de features"""
    # Criar DataFrame com as features que o modelo espera
    X_aligned = pd.DataFrame(0, index=X_data.index, columns=feature_names)
    
    # Copiar as colunas que existem em ambos
    common_cols = set(feature_names) & set(X_data.columns)
    for col in common_cols:
        X_aligned[col] = X_data[col].fillna(0)  # Preencher NaN
    
    try:
        return model.predict_proba(X_aligned)
    except Exception as e:
        st.error(f"❌ Prediction error for {model.__class__.__name__}: {e}")
        # Fallback: probabilidades uniformes
        n_samples = len(X_data)
        n_classes = len(model.classes_) if hasattr(model, 'classes_') else 2
        return np.full((n_samples, n_classes), 1.0/n_classes)

# Previsões com alinhamento correto de features
probs_1x2 = safe_predict_proba(model_multi, X_today_1x2, model_multi.feature_names_in_)
probs_ou = safe_predict_proba(model_ou, X_today_ou, model_ou.feature_names_in_)
probs_btts = safe_predict_proba(model_btts, X_today_btts, model_btts.feature_names_in_)

games_today["p_home"], games_today["p_draw"], games_today["p_away"] = probs_1x2.T
games_today["p_over25"], games_today["p_under25"] = probs_ou.T
games_today["p_btts_yes"], games_today["p_btts_no"] = probs_btts.T


# ########################################################
# Bloco 10 – Styling e Display (ATUALIZADO COM PLACAR)
# ########################################################
def color_prob(val, color):
    alpha = int(val * 255)
    return f"background-color: rgba({color}, {alpha/255:.2f})"

def style_probs(val, col):
    if col == "p_home": return color_prob(val, "0,200,0")
    elif col == "p_draw": return color_prob(val, "150,150,150")
    elif col == "p_away": return color_prob(val, "255,140,0")
    elif col == "p_over25": return color_prob(val, "0,100,255")
    elif col == "p_under25": return color_prob(val, "128,0,128")
    elif col == "p_btts_yes": return color_prob(val, "0,200,200")
    elif col == "p_btts_no": return color_prob(val, "200,0,0")
    return ""

# 🔥 COLUNAS ATUALIZADAS - Adicionando Goals_H_Today e Goals_A_Today
cols_final = [
    "Date", "Time", "League", "Home", "Away",
    "Goals_H_Today", "Goals_A_Today",  # 🔥 NOVAS COLUNAS DE PLACAR
    "Odd_H", "Odd_D", "Odd_A",
    "p_home", "p_draw", "p_away",
    "p_over25", "p_under25",
    "p_btts_yes", "p_btts_no"
]

styled_df = (
    games_today[cols_final]
    .style.format({
        "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
        "p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}",
        "p_over25": "{:.1%}", "p_under25": "{:.1%}",
        "p_btts_yes": "{:.1%}", "p_btts_no": "{:.1%}",
        "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}"  # 🔥 FORMATAÇÃO DO PLACAR
    }, na_rep="—")
    .applymap(lambda v: style_probs(v, "p_home"), subset=["p_home"])
    .applymap(lambda v: style_probs(v, "p_draw"), subset=["p_draw"])
    .applymap(lambda v: style_probs(v, "p_away"), subset=["p_away"])
    .applymap(lambda v: style_probs(v, "p_over25"), subset=["p_over25"])
    .applymap(lambda v: style_probs(v, "p_under25"), subset=["p_under25"])
    .applymap(lambda v: style_probs(v, "p_btts_yes"), subset=["p_btts_yes"])
    .applymap(lambda v: style_probs(v, "p_btts_no"), subset=["p_btts_no"])

)

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df, use_container_width=True, height=1000)

# 🔹 Botão para download do CSV (ATUALIZADO)
import io
csv_buffer = io.BytesIO()
games_today.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
csv_buffer.seek(0)

st.download_button(
    label="📥 Download Predictions CSV",
    data=csv_buffer,
    file_name=f"Bet_Indicator_Triple_View_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)

# ########################################################
# Block 11 – Hybrid Forecast (Historical vs ML)
# ########################################################
st.markdown("## 🔮 Hybrid Forecast – Perspective vs ML")

try:
    import numpy as np

    # 🔹 Ensure we have a reference date
    if not games_today.empty and "Date" in games_today.columns:
        selected_date = pd.to_datetime(games_today["Date"], errors="coerce").dt.date.iloc[0]
    else:
        selected_date = None

    # ===== Historical Perspective =====
    all_dfs = []
    for f in os.listdir(GAMES_FOLDER):
        if f.lower().endswith(".csv"):
            try:
                df_tmp = pd.read_csv(os.path.join(GAMES_FOLDER, f))
                df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('^Unnamed')]
                df_tmp.columns = df_tmp.columns.str.strip()
                all_dfs.append(df_tmp)
            except:
                continue

    if all_dfs and selected_date is not None:
        df_history = pd.concat(all_dfs, ignore_index=True)

        # 🧹 Remove duplicates (usando função melhorada do Binary)
        df_history = df_history.drop_duplicates(
            subset=["Date", "Home", "Away", "Goals_H_FT", "Goals_A_FT"],
            keep="first"
        )

        # Normalize dates and exclude today's matches
        if "Date" in df_history.columns:
            df_history["Date"] = pd.to_datetime(df_history["Date"], errors="coerce").dt.date
            df_history = df_history[df_history["Date"] != selected_date]

        # Create Diff_M and bins
        df_history["Diff_M"] = df_history["M_H"] - df_history["M_A"]
        df_history["DiffPower_bin"] = pd.cut(df_history["Diff_Power"], bins=range(-50, 55, 10))
        df_history["DiffM_bin"] = pd.cut(df_history["Diff_M"], bins=np.arange(-10, 10.5, 1.0))
        df_history["DiffHTP_bin"] = pd.cut(df_history["Diff_HT_P"], bins=range(-30, 35, 5))

        # Real match outcome
        def get_result(row):
            if row["Goals_H_FT"] > row["Goals_A_FT"]:
                return "Home"
            elif row["Goals_H_FT"] < row["Goals_A_FT"]:
                return "Away"
            else:
                return "Draw"

        df_history["Result"] = df_history.apply(get_result, axis=1)

        # Prepare today's matches (using games_today)
        df_day = games_today.copy()
        df_day = df_day.loc[:, ~df_day.columns.str.contains('^Unnamed')]
        df_day.columns = df_day.columns.str.strip()
        df_day["Date"] = pd.to_datetime(df_day["Date"], errors="coerce").dt.date
        df_day = df_day[df_day["Date"] == selected_date]
        df_day["Diff_M"] = df_day["M_H"] - df_day["M_A"]
        df_day = df_day.dropna(subset=["Diff_Power", "Diff_M", "Diff_HT_P"])

        # Bin intervals
        dp_bins = pd.IntervalIndex(df_history["DiffPower_bin"].cat.categories)
        dm_bins = pd.IntervalIndex(df_history["DiffM_bin"].cat.categories)
        dhtp_bins = pd.IntervalIndex(df_history["DiffHTP_bin"].cat.categories)

        # Counters
        total_matches, home_wins, away_wins, draws = 0, 0, 0, 0

        for _, game in df_day.iterrows():
            try:
                if (
                    dp_bins.contains(game["Diff_Power"]).any() and
                    dm_bins.contains(game["Diff_M"]).any() and
                    dhtp_bins.contains(game["Diff_HT_P"]).any()
                ):
                    dp_bin = dp_bins.get_loc(game["Diff_Power"])
                    dm_bin = dm_bins.get_loc(game["Diff_M"])
                    dhtp_bin = dhtp_bins.get_loc(game["Diff_HT_P"])
                else:
                    continue

                subset = df_history[
                    (df_history["DiffPower_bin"] == dp_bins[dp_bin]) &
                    (df_history["DiffM_bin"] == dm_bins[dm_bin]) &
                    (df_history["DiffHTP_bin"] == dhtp_bins[dhtp_bin])
                ]

                if not subset.empty:
                    total_matches += len(subset)
                    home_wins += (subset["Result"] == "Home").sum()
                    away_wins += (subset["Result"] == "Away").sum()
                    draws += (subset["Result"] == "Draw").sum()
            except:
                continue

        if total_matches > 0:
            pct_home = 100 * home_wins / total_matches
            pct_away = 100 * away_wins / total_matches
            pct_draw = 100 * draws / total_matches
        else:
            pct_home, pct_away, pct_draw = 0, 0, 0

    # ===== ML Forecast =====
    if not games_today.empty:
        ml_probs = model_multi.predict_proba(X_today_1x2)
        df_preds = pd.DataFrame(ml_probs, columns=["p_home", "p_draw", "p_away"])

        ml_home = df_preds["p_home"].mean() * 100
        ml_draw = df_preds["p_draw"].mean() * 100
        ml_away = df_preds["p_away"].mean() * 100
    else:
        ml_home, ml_draw, ml_away = 0, 0, 0

    # ===== Side by side display =====
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 📊 Historical Perspective")
        st.write(f"**Home Wins:** {pct_home:.1f}%")
        st.write(f"**Draws:** {pct_draw:.1f}%")
        st.write(f"**Away Wins:** {pct_away:.1f}%")
        st.caption(f"Based on {total_matches:,} similar historical matches (excluding today)")
    with cols[1]:
        st.markdown("### 🤖 ML Forecast (Trained Model)")
        st.write(f"**Home Wins:** {ml_home:.1f}%")
        st.write(f"**Draws:** {ml_draw:.1f}%")
        st.write(f"**Away Wins:** {ml_away:.1f}%")
        st.caption(f"Based on {len(games_today)} matches today")

except Exception as e:
    st.warning(f"⚠️ Hybrid Forecast could not be generated: {e}")



# ########################################################
# Block 12 – Divergence Index with Gauge
# ########################################################
try:
    import plotly.graph_objects as go

    # ===== Divergence Index =====
    divergence = abs(ml_home - pct_home) + abs(ml_draw - pct_draw) + abs(ml_away - pct_away)

    if divergence < 10:
        status_icon, status_text = "🟢", "High confidence (ML aligned with historical)"
    elif divergence < 25:
        status_icon, status_text = "🟡", "Medium confidence (some divergence)"
    else:
        status_icon, status_text = "🔴", "Low confidence (ML diverges strongly from historical)"

    # Detailed differences
    st.markdown("### 🔍 Difference: Historical vs ML")
    st.write(f"- Home: {ml_home - pct_home:+.1f} pp")
    st.write(f"- Draw: {ml_draw - pct_draw:+.1f} pp")
    st.write(f"- Away: {ml_away - pct_away:+.1f} pp")

    # Global index
    st.markdown("### 📈 Global Divergence Index")
    st.write(f"{status_icon} {status_text}")
    st.caption(f"Total divergence index: {divergence:.1f} percentage points")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=divergence,
        title={'text': "Divergence Index"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 25], 'color': "khaki"},
                {'range': [25, 50], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': divergence
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Divergence Block could not be generated: {e}")
