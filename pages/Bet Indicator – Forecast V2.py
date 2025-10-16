# ########################################################
# Bloco 1 – Imports & Config
# ########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Bet Indicator – Forecast V2", layout="wide")
st.title("📊 Bet Indicator – Forecast V2 (Enhanced Features)")

# Paths
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "copas", "uefa", "nordeste", "afc","trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


# ########################################################
# Bloco 2 – Funções auxiliares
# ########################################################
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(os.path.join(folder, f)) for f in files], ignore_index=True)

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    
    today_file = files[-1]
    yesterday_file = files[-2] if len(files) >= 2 else None

    st.markdown("### 📂 Select matches to display")
    col1, col2 = st.columns(2)
    today_checked = col1.checkbox("Today Matches", value=True)
    yesterday_checked = col2.checkbox("Yesterday Matches", value=False)

    selected_dfs = []
    if today_checked:
        selected_dfs.append(pd.read_csv(os.path.join(folder, today_file)))
    if yesterday_checked and yesterday_file:
        selected_dfs.append(pd.read_csv(os.path.join(folder, yesterday_file)))

    if not selected_dfs:
        return pd.DataFrame()
    return pd.concat(selected_dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

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
# Bloco 3 – Carregar Dados
# ########################################################
st.info("📂 Loading data...")

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

if history.empty:
    st.error("⚠️ No valid historical data found in GamesDay.")
    st.stop()

games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

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
# Bloco 5 – Features Avançadas
# ########################################################

# --- Calcular Odds derivadas (1X e X2)
def compute_double_chance_odds(df):
    df = df.copy()
    if set(["Odd_H", "Odd_D", "Odd_A"]).issubset(df.columns):
        probs = pd.DataFrame()
        probs["p_H"] = 1 / df["Odd_H"]
        probs["p_D"] = 1 / df["Odd_D"]
        probs["p_A"] = 1 / df["Odd_A"]
        probs = probs.div(probs.sum(axis=1), axis=0)
        df["Odd_1X"] = 1 / (probs["p_H"] + probs["p_D"])
        df["Odd_X2"] = 1 / (probs["p_A"] + probs["p_D"])
    return df

history = compute_double_chance_odds(history)
games_today = compute_double_chance_odds(games_today)

# --- Momentum Difference
history["M_Diff"] = history["M_H"] - history["M_A"]
games_today["M_Diff"] = games_today["M_H"] - games_today["M_A"]

# --- Classificação ligas
def classify_leagues_variation(history_df):
    agg = (
        history_df.groupby("League")
        .agg(
            M_H_Min=("M_H","min"), M_H_Max=("M_H","max"),
            M_A_Min=("M_A","min"), M_A_Max=("M_A","max"),
            Hist_Games=("M_H","count")
        ).reset_index()
    )
    agg["Variation_Total"] = (agg["M_H_Max"] - agg["M_H_Min"]) + (agg["M_A_Max"] - agg["M_A_Min"])
    def label(v):
        if v > 6.0: return "High Variation"
        if v >= 3.0: return "Medium Variation"
        return "Low Variation"
    agg["League_Classification"] = agg["Variation_Total"].apply(label)
    return agg[["League","League_Classification","Variation_Total","Hist_Games"]]

# --- Compute league bands
def compute_league_bands(history_df):
    hist = history_df.copy()
    hist["M_Diff"] = hist["M_H"] - hist["M_A"]
    diff_q = (
        hist.groupby("League")["M_Diff"]
            .quantile([0.20,0.80]).unstack()
            .rename(columns={0.2:"P20_Diff",0.8:"P80_Diff"})
            .reset_index()
    )
    home_q = (
        hist.groupby("League")["M_H"]
            .quantile([0.20,0.80]).unstack()
            .rename(columns={0.2:"Home_P20",0.8:"Home_P80"})
            .reset_index()
    )
    away_q = (
        hist.groupby("League")["M_A"]
            .quantile([0.20,0.80]).unstack()
            .rename(columns={0.2:"Away_P20",0.8:"Away_P80"})
            .reset_index()
    )
    out = diff_q.merge(home_q,on="League",how="inner").merge(away_q,on="League",how="inner")
    return out

# --- Dominância
def dominant_side(row, threshold=0.90):
    m_h, m_a = row["M_H"], row["M_A"]
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold: return "Home strong"
    if m_h <= -threshold: return "Home weak"
    if m_a >= threshold: return "Away strong"
    if m_a <= -threshold: return "Away weak"
    return "Mixed / Neutral"

# --- Merge features
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

for name, df in [("history", history), ("games_today", games_today)]:
    df = df.merge(league_class, on="League", how="left")
    df = df.merge(league_bands, on="League", how="left")
    df["Home_Band"] = np.where(
        df["M_H"] <= df["Home_P20"], "Bottom 20%",
        np.where(df["M_H"] >= df["Home_P80"], "Top 20%", "Balanced")
    )
    df["Away_Band"] = np.where(
        df["M_A"] <= df["Away_P20"], "Bottom 20%",
        np.where(df["M_A"] >= df["Away_P80"], "Top 20%", "Balanced")
    )
    df["Dominant"] = df.apply(dominant_side, axis=1)
    df["Home_Band_Num"] = df["Home_Band"].map({"Bottom 20%":1,"Balanced":2,"Top 20%":3})
    df["Away_Band_Num"] = df["Away_Band"].map({"Bottom 20%":1,"Balanced":2,"Top 20%":3})
    df["Band_Diff"] = df["Home_Band_Num"] - df["Away_Band_Num"]
    if name == "history":
        history = df
    else:
        games_today = df


# ########################################################
# Bloco 6 – Configurações ML (Sidebar)
# ########################################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"]
)
retrain = st.sidebar.checkbox("Retrain models", value=False)

st.sidebar.markdown("""
**ℹ️ Usage recommendations:**
- 🔹 *Random Forest*: simple and fast baseline.  
- 🔹 *Random Forest Tuned*: suitable for market **1X2**.  
- 🔹 *XGBoost Tuned*: suitable for markets **Over/Under 2.5** and **BTTS**.  
""")


# ########################################################
# Bloco 7 – Treino & Avaliação
# ########################################################
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_fc_v2.pkl"
    model = None

    # Carregar modelo salvo se retrain desmarcado
    if not retrain:
        model = load_model(filename)

    # Se não existir modelo salvo ou retrain = True, treina novamente
    if model is None:
        if ml_model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=300, 
                random_state=42, 
                class_weight="balanced_subsample"
            )

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

        # Split para validação
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        save_model(model, filename)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Avaliação
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
    }

    return metrics, model


# ########################################################
# Bloco 8 – Montar Features e Treinar Modelos
# ########################################################

# --- Definir blocos de features
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A", "Odd_1X", "Odd_X2"],
    "strength": [
        "Diff_Power","M_H","M_A","M_Diff",
        "Diff_HT_P","M_HT_H","M_HT_A","OU_Total"
    ],
    "categorical": [
        "Home_Band_Num","Away_Band_Num","Dominant","League_Classification"
    ]
}

# --- One-Hot Encoding para League, Dominant e League_Classification
def build_feature_matrix(df, leagues_df, fit_encoder=False, encoder=None):
    dfs = []
    for block_name, cols in feature_blocks.items():
        if block_name == "categorical": 
            continue
        available_cols = [c for c in cols if c in df.columns]
        if available_cols:
            dfs.append(df[available_cols])
    if leagues_df is not None and not leagues_df.empty:
        dfs.append(leagues_df)
    
    cat_cols = [c for c in ["Dominant","League_Classification"] if c in df.columns]
    if cat_cols:
        if fit_encoder:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        dfs.append(encoded_df)

    for col in ["Home_Band_Num","Away_Band_Num"]:
        if col in df.columns:
            dfs.append(df[[col]])
    
    X = pd.concat(dfs, axis=1)
    return X, encoder

# --- Preparar Leagues
history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

# --- Construir matrizes
X_1x2, encoder_cat = build_feature_matrix(history, history_leagues, fit_encoder=True)
X_today_1x2, _ = build_feature_matrix(games_today, games_today_leagues, fit_encoder=False, encoder=encoder_cat)
X_today_1x2 = X_today_1x2.reindex(columns=X_1x2.columns, fill_value=0)

# Mesmo conjunto para OU e BTTS
X_ou = X_1x2.copy()
X_today_ou = X_today_1x2.copy()
X_btts = X_1x2.copy()
X_today_btts = X_today_1x2.copy()

# --- Treinar modelos
stats = []
res, model_multi = train_and_evaluate(X_1x2, history["Target"], "1X2", 3)
stats.append(res)
res, model_ou = train_and_evaluate(X_ou, history["Target_OU25"], "OverUnder25", 2)
stats.append(res)
res, model_btts = train_and_evaluate(X_btts, history["Target_BTTS"], "BTTS", 2)
stats.append(res)

df_stats = pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation) TOP")
st.dataframe(df_stats, use_container_width=True)


# ########################################################
# Bloco 9 – Previsões
# ########################################################
games_today["p_home"], games_today["p_draw"], games_today["p_away"] = model_multi.predict_proba(X_today_1x2).T
games_today["p_over25"], games_today["p_under25"] = model_ou.predict_proba(X_today_ou).T
games_today["p_btts_yes"], games_today["p_btts_no"] = model_btts.predict_proba(X_today_btts).T


# ########################################################
# Bloco 10 – Styling e Display
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

cols_final = [
    "Date","Time","League","Home","Away",
    "Odd_H","Odd_D","Odd_A",
    "p_home","p_draw","p_away",
    "p_over25","p_under25",
    "p_btts_yes","p_btts_no"
]

styled_df = (
    games_today[cols_final]
    .style.format({
        "Odd_H": "{:.2f}","Odd_D": "{:.2f}","Odd_A": "{:.2f}",
        "p_home": "{:.1%}","p_draw": "{:.1%}","p_away": "{:.1%}",
        "p_over25": "{:.1%}","p_under25": "{:.1%}",
        "p_btts_yes": "{:.1%}","p_btts_no": "{:.1%}",
    }, na_rep="—")
    .applymap(lambda v: style_probs(v, "p_home"), subset=["p_home"])
    .applymap(lambda v: style_probs(v, "p_draw"), subset=["p_draw"])
    .applymap(lambda v: style_probs(v, "p_away"), subset=["p_away"])
    .applymap(lambda v: style_probs(v, "p_over25"), subset=["p_over25"])
    .applymap(lambda v: style_probs(v, "p_under25"), subset=["p_under25"])
    .applymap(lambda v: style_probs(v, "p_btts_yes"), subset=["p_btts_yes"])
    .applymap(lambda v: style_probs(v, "p_btts_no"), subset=["p_btts_no"])
)

# st.markdown("### 📌 Predictions for Selected Matches (Forecast V2)")
# st.dataframe(styled_df, use_container_width=True, height=1000)

# =========================================================
# ⚙️ Calibrar α por Liga (Odds vs Momentum)
# =========================================================
from scipy.stats import skellam, poisson
import json

st.markdown("#### ⚙️ Calibrating α by League (auto-optimized, cached)")

alpha_global_prior = st.sidebar.slider("α global (prior)", 0.0, 1.0, 0.50, 0.05)
shrinkage_m = st.sidebar.slider("Força da suavização (m)", 50, 1000, 300, 50)
min_samples_per_league = st.sidebar.slider("Mínimo de jogos/ligas", 50, 1000, 200, 50)

path_alpha = os.path.join(MODELS_FOLDER, "alpha_by_league.json")

def odds_to_mu(odd_home, odd_draw, odd_away):
    if pd.isna(odd_home) or pd.isna(odd_draw) or pd.isna(odd_away):
        return np.nan, np.nan
    inv = (1/odd_home + 1/odd_draw + 1/odd_away)
    p_home = (1/odd_home) / inv
    p_away = (1/odd_away) / inv
    mu_h = 0.4 + 2.4 * p_home
    mu_a = 0.4 + 2.4 * p_away
    return mu_h, mu_a

def xg_from_momentum(row):
    base = 1.3
    denom = abs(row.get("M_H", 0.0)) + abs(row.get("M_A", 0.0)) + 1e-6
    mu_h = base + 0.8*(row.get("M_H",0.0)/denom) + 0.4*(row.get("Diff_Power",0.0)/100)
    mu_a = base + 0.8*(row.get("M_A",0.0)/denom) - 0.4*(row.get("Diff_Power",0.0)/100)
    return max(mu_h,0.05), max(mu_a,0.05)

@st.cache_data(show_spinner=True)
def compute_alpha_by_league_all(history_df, alpha_global_prior, shrinkage_m, min_samples_per_league):
    alpha_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2)
    alpha_by_league = {}
    for lg, df_lg in history_df.groupby("League"):
        if len(df_lg) < 20: 
            continue
        best_alpha, best_ll = None, np.inf
        for a in alpha_grid:
            ll_sum, n_ok = 0.0, 0
            for _, r in df_lg.iterrows():
                mu_odd_h, mu_odd_a = odds_to_mu(r["Odd_H"], r["Odd_D"], r["Odd_A"])
                mu_perf_h, mu_perf_a = xg_from_momentum(r)
                mu_h = a*mu_odd_h + (1-a)*mu_perf_h
                mu_a = a*mu_odd_a + (1-a)*mu_perf_a
                if not np.isfinite(mu_h) or not np.isfinite(mu_a): 
                    continue
                pH = 1 - skellam.cdf(0, mu_h, mu_a)
                pD = skellam.pmf(0, mu_h, mu_a)
                pA = skellam.cdf(-1, mu_h, mu_a)
                y = 0 if r["Goals_H_FT"] > r["Goals_A_FT"] else (2 if r["Goals_H_FT"] < r["Goals_A_FT"] else 1)
                eps = 1e-12
                ll_sum += -np.log([pH, pD, pA][y] + eps)
                n_ok += 1
            if n_ok >= 20 and ll_sum/n_ok < best_ll:
                best_ll = ll_sum/n_ok
                best_alpha = a
        if best_alpha is not None:
            n = len(df_lg)
            shrink_alpha = (n/(n+shrinkage_m))*best_alpha + (shrinkage_m/(n+shrinkage_m))*alpha_global_prior
            alpha_by_league[lg] = round(shrink_alpha,3)
    return alpha_by_league

# carregar ou gerar
if os.path.exists(path_alpha):
    with open(path_alpha, "r", encoding="utf-8") as f:
        data = json.load(f)
    alpha_by_league = data.get("alpha_by_league", {})
    st.caption(f"✅ α loaded from cache ({len(alpha_by_league)} leagues)")
else:
    st.info("⏳ Computing α by league…")
    alpha_by_league = compute_alpha_by_league_all(history, alpha_global_prior, shrinkage_m, min_samples_per_league)
    with open(path_alpha, "w", encoding="utf-8") as f:
        json.dump({"alpha_by_league": alpha_by_league}, f, ensure_ascii=False, indent=2)
    st.success(f"💾 α computed & saved for {len(alpha_by_league)} leagues")



# =========================================================
# TAB 2 – Skellam Model (1X2 + AH) – versão calibrada
# =========================================================
with tab2:
    st.markdown("### 🎲 Skellam Model (1X2 + AH) – α por Liga (calibrado)")

    # ------------------------------------------------------
    # 1️⃣ Converter linha asiática (frações → média decimal)
    # ------------------------------------------------------
    def convert_asian_line(line_str):
        """Converte string tipo '-0.25/0' para média float."""
        try:
            if pd.isna(line_str) or str(line_str).strip() == "":
                return None
            line_str = str(line_str).strip().replace(",", ".").replace(" ", "")
            if "/" in line_str:
                parts = [float(x) for x in line_str.split("/")]
                return float(np.mean(parts))
            return float(line_str)
        except Exception:
            return None

    if "Asian_Line" in games_today.columns:
        games_today["Asian_Home"] = games_today["Asian_Line"].apply(convert_asian_line)
    else:
        st.warning("⚠️ Column 'Asian_Line' not found – Skellam AH disabled.")
        games_today["Asian_Home"] = np.nan

    # ------------------------------------------------------
    # 2️⃣ Funções base (Odds → xG + Momentum)
    # ------------------------------------------------------
    from scipy.stats import poisson, skellam
    import math, json

    def odds_to_mu(odd_home, odd_draw, odd_away):
        """Converte odds 1X2 em taxas de gols esperadas (mu_h, mu_a)."""
        if pd.isna(odd_home) or pd.isna(odd_draw) or pd.isna(odd_away):
            return np.nan, np.nan
        inv = (1/odd_home + 1/odd_draw + 1/odd_away)
        p_home = (1/odd_home) / inv
        p_away = (1/odd_away) / inv
        mu_h = 0.4 + 2.4 * p_home
        mu_a = 0.4 + 2.4 * p_away
        return mu_h, mu_a

    def xg_from_momentum(row):
        """Cria xG ajustado por Momentum e Diff_Power."""
        base = 1.3
        denom = abs(row.get("M_H", 0.0)) + abs(row.get("M_A", 0.0)) + 1e-6
        mu_h = base + 0.8 * (row.get("M_H", 0.0)/denom) + 0.4 * (row.get("Diff_Power", 0.0)/100)
        mu_a = base + 0.8 * (row.get("M_A", 0.0)/denom) - 0.4 * (row.get("Diff_Power", 0.0)/100)
        return max(mu_h, 0.05), max(mu_a, 0.05)

    # ------------------------------------------------------
    # 3️⃣ Carregar α por liga (cache)
    # ------------------------------------------------------
    path_alpha = os.path.join(MODELS_FOLDER, "alpha_by_league.json")
    alpha_by_league, alpha_by_league_ou, alpha_by_league_btts = {}, {}, {}
    alpha_global_prior = 0.50  # padrão global

    if os.path.exists(path_alpha):
        with open(path_alpha, "r", encoding="utf-8") as f:
            data = json.load(f)
        alpha_by_league = data.get("alpha_by_league", {})
        alpha_by_league_ou = data.get("alpha_by_league_ou", {})
        alpha_by_league_btts = data.get("alpha_by_league_btts", {})
        st.caption(f"✅ α loaded from cache ({len(alpha_by_league)} leagues)")
    else:
        st.warning("⚠️ α cache file not found. Run Dual View once to generate it.")

    # ------------------------------------------------------
    # 4️⃣ Blend Odds + Momentum via α
    # ------------------------------------------------------
    def get_alpha(lg, mapping, default):
        return mapping.get(lg, default)

    def compute_xg2_all(row):
        def blend(alpha):
            mu_odd_h, mu_odd_a = odds_to_mu(row["Odd_H"], row["Odd_D"], row["Odd_A"])
            mu_perf_h, mu_perf_a = xg_from_momentum(row)
            mu_h = alpha * mu_odd_h + (1 - alpha) * mu_perf_h
            mu_a = alpha * mu_odd_a + (1 - alpha) * mu_perf_a
            return float(np.clip(mu_h, 0.05, 5.0)), float(np.clip(mu_a, 0.05, 5.0))
        a1 = get_alpha(row.get("League"), alpha_by_league, alpha_global_prior)
        mu1_h, mu1_a = blend(a1)
        return mu1_h, mu1_a, a1

    games_today[["XG2_H", "XG2_A", "Alpha_League"]] = games_today.apply(
        compute_xg2_all, axis=1, result_type="expand"
    )

    # ------------------------------------------------------
    # 5️⃣ Funções Skellam
    # ------------------------------------------------------
    def skellam_1x2(mu_h, mu_a):
        mu_h, mu_a = float(np.clip(mu_h, 0.05, 5.0)), float(np.clip(mu_a, 0.05, 5.0))
        p_home = 1 - skellam.cdf(0, mu_h, mu_a)
        p_draw = skellam.pmf(0, mu_h, mu_a)
        p_away = skellam.cdf(-1, mu_h, mu_a)
        return p_home, p_draw, p_away

    def skellam_handicap(mu_h, mu_a, line):
        """Probabilidades do Home ganhar/push/perder dado o handicap."""
        try:
            mu_h, mu_a = float(np.clip(mu_h, 0.05, 5.0)), float(np.clip(mu_a, 0.05, 5.0))
            if pd.isna(line): return np.nan, np.nan, np.nan
            line = float(line)
        except Exception:
            return np.nan, np.nan, np.nan

        # Inteiro
        if abs(line - round(line)) < 1e-9:
            k = int(round(line))
            win = 1 - skellam.cdf(k, mu_h, mu_a)
            push = skellam.pmf(k, mu_h, mu_a)
            lose = skellam.cdf(k - 1, mu_h, mu_a)
            return win, push, lose
        # Meia
        if abs(line * 2 - round(line * 2)) < 1e-9 and abs(line * 4 - round(line * 4)) > 1e-9:
            if line > 0:
                thr = math.floor(-line)
                win = 1 - skellam.cdf(thr, mu_h, mu_a)
                lose = skellam.cdf(thr, mu_h, mu_a)
            else:
                k = abs(line)
                win = 1 - skellam.cdf(math.ceil(k), mu_h, mu_a)
                lose = skellam.cdf(math.ceil(k), mu_h, mu_a)
            return win, 0.0, lose
        # Quarta (±0.25, ±0.75 …)
        if abs(line * 4 - round(line * 4)) < 1e-9:
            low, high = line - 0.25, line + 0.25
            def single(l):
                if abs(l - round(l)) < 1e-9:
                    k = int(round(l))
                    return 1 - skellam.cdf(k, mu_h, mu_a), skellam.pmf(k, mu_h, mu_a), skellam.cdf(k - 1, mu_h, mu_a)
                if l > 0:
                    thr = math.floor(-l)
                    return 1 - skellam.cdf(thr, mu_h, mu_a), 0.0, skellam.cdf(thr, mu_h, mu_a)
                k = abs(l)
                return 1 - skellam.cdf(math.ceil(k), mu_h, mu_a), 0.0, skellam.cdf(math.ceil(k), mu_h, mu_a)
            r1, r2 = single(low), single(high)
            return 0.5 * (r1[0] + r2[0]), 0.5 * (r1[1] + r2[1]), 0.5 * (r1[2] + r2[2])
        return np.nan, np.nan, np.nan

    # ------------------------------------------------------
    # 6️⃣ Aplicar Skellam (1X2 + AH)
    # ------------------------------------------------------
    games_today["Skellam_pH"], games_today["Skellam_pD"], games_today["Skellam_pA"] = zip(
        *games_today.apply(
            lambda r: skellam_1x2(r["XG2_H"], r["XG2_A"])
            if pd.notna(r["XG2_H"]) and pd.notna(r["XG2_A"]) else (np.nan, np.nan, np.nan),
            axis=1,
        )
    )

    games_today["Skellam_AH_Win"], games_today["Skellam_AH_Push"], games_today["Skellam_AH_Lose"] = zip(
        *games_today.apply(
            lambda r: skellam_handicap(r["XG2_H"], r["XG2_A"], r["Asian_Home"])
            if pd.notna(r["XG2_H"]) and pd.notna(r["Asian_Home"]) else (np.nan, np.nan, np.nan),
            axis=1,
        )
    )

    # ------------------------------------------------------
    # 7️⃣ EV teórico (Skellam vs odds)
    # ------------------------------------------------------
    def implied_prob(odd): return 1 / odd if pd.notna(odd) and odd > 0 else np.nan
    games_today["Impl_H"] = games_today["Odd_H"].apply(implied_prob)
    games_today["Impl_A"] = games_today["Odd_A"].apply(implied_prob)
    games_today["EV_H_Skellam"] = games_today["Skellam_pH"] - games_today["Impl_H"]
    games_today["EV_A_Skellam"] = games_today["Skellam_pA"] - games_today["Impl_A"]

    # ------------------------------------------------------
    # 8️⃣ Exibir tabela principal
    # ------------------------------------------------------
    df_skellam = games_today[
        [
            "League", "Home", "Away", "Asian_Line", "Asian_Home",
            "XG2_H", "XG2_A", "Alpha_League",
            "Skellam_pH", "Skellam_pD", "Skellam_pA",
            "Skellam_AH_Win", "Skellam_AH_Push", "Skellam_AH_Lose",
            "Odd_H", "Odd_A", "Impl_H", "Impl_A",
            "EV_H_Skellam", "EV_A_Skellam",
        ]
    ].copy()

    def hl(val):
        color = "rgba(0,200,0,0.25)" if pd.notna(val) and val > 0 else "rgba(255,0,0,0.15)"
        return f"background-color:{color}"

    st.dataframe(
        df_skellam.style.format({
            "Asian_Home": "{:+.2f}",
            "XG2_H": "{:.2f}", "XG2_A": "{:.2f}",
            "Alpha_League": "{:.2f}",
            "Skellam_pH": "{:.1%}", "Skellam_pD": "{:.1%}", "Skellam_pA": "{:.1%}",
            "Skellam_AH_Win": "{:.1%}", "Skellam_AH_Push": "{:.1%}", "Skellam_AH_Lose": "{:.1%}",
            "Odd_H": "{:.2f}", "Odd_A": "{:.2f}",
            "Impl_H": "{:.1%}", "Impl_A": "{:.1%}",
            "EV_H_Skellam": "{:+.1%}", "EV_A_Skellam": "{:+.1%}",
        }).applymap(hl, subset=["EV_H_Skellam", "EV_A_Skellam"]),
        use_container_width=True, height=700,
    )

    # ------------------------------------------------------
    # 9️⃣ Value Scanner – Skellam
    # ------------------------------------------------------
    st.markdown("## 🎯 Value Scanner – Skellam (α calibrado)")
    EV_SK_THRESHOLD = st.sidebar.slider("EV mínimo (Skellam)", 0.01, 0.10, 0.03, 0.01)
    df_val_sk = df_skellam.copy()
    df_val_sk["Best_Skellam"] = np.where(
        df_val_sk["EV_H_Skellam"] >= df_val_sk["EV_A_Skellam"], "Home", "Away"
    )
    df_val_sk["EV_Best_Skellam"] = df_val_sk[["EV_H_Skellam", "EV_A_Skellam"]].max(axis=1)
    picks_sk = df_val_sk[df_val_sk["EV_Best_Skellam"] > EV_SK_THRESHOLD].sort_values(
        "EV_Best_Skellam", ascending=False
    )

    if not picks_sk.empty:
        st.success(f"🎯 {len(picks_sk)} value bets (Skellam) EV > {EV_SK_THRESHOLD:.0%}")
        st.dataframe(
            picks_sk[[
                "League", "Home", "Away", "Asian_Home",
                "Best_Skellam", "EV_Best_Skellam",
                "Odd_H", "Odd_A", "Skellam_pH", "Skellam_pA",
            ]].style.format({
                "Asian_Home": "{:+.2f}",
                "EV_Best_Skellam": "{:+.1%}",
                "Odd_H": "{:.2f}", "Odd_A": "{:.2f}",
                "Skellam_pH": "{:.1%}", "Skellam_pA": "{:.1%}",
            }),
            use_container_width=True,
        )
    else:
        st.warning("Nenhuma aposta de valor (Skellam) acima do threshold.")
