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
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "copas", "uefa", "nordeste", "afc"]

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

st.markdown("### 📌 Predictions for Selected Matches (Forecast V2)")
st.dataframe(styled_df, use_container_width=True, height=1000)

