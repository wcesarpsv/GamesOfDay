##################### BLOCO 1 – IMPORTS & CONFIG #####################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bet Indicator – Triple View", layout="wide")
st.title("📊 Bet Indicator – Triple View (1X2 + OU + BTTS + Goal Categories)")

GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


##################### BLOCO 2 – HELPERS #####################
def preprocess_df(df):
    df = df.copy()
    # Renomear colunas duplicadas
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    # Garantir Bet Result
    if "Bet Result" not in df.columns:
        df["Bet Result"] = 0
    return df


def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Remover duplicados com base nas colunas principais
    if set(["Date", "Home", "Away", "League"]).issubset(df.columns):
        df = df.drop_duplicates(subset=["Date", "Home", "Away", "League"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    return df


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

    dfs = []
    if today_checked:
        dfs.append(preprocess_df(pd.read_csv(os.path.join(folder, today_file))))
    if yesterday_checked and yesterday_file:
        dfs.append(preprocess_df(pd.read_csv(os.path.join(folder, yesterday_file))))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Remover duplicados
    if set(["Date", "Home", "Away", "League"]).issubset(df.columns):
        df = df.drop_duplicates(subset=["Date", "Home", "Away", "League"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    return df


def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()


def save_model(model, filename):
    with open(os.path.join(MODELS_FOLDER, filename), "wb") as f:
        joblib.dump(model, f)


def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None


##################### BLOCO 3 – LOAD DATA #####################
st.info("📂 Loading data...")
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()
if history.empty:
    st.stop()
games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()
if games_today.empty:
    st.stop()

# Targets
history["Target"] = history.apply(
    lambda r: 0 if r["Goals_H_FT"] > r["Goals_A_FT"] else (1 if r["Goals_H_FT"] == r["Goals_A_FT"] else 2), axis=1
)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)


##################### BLOCO 4 – EXTRA FEATURES (GOAL COST/VALUE + DYNAMIC CATEGORIES) #####################
def rolling_stats(sub_df, col, window=5, min_periods=1):
    return sub_df.sort_values("Date")[col].rolling(window=window, min_periods=min_periods).mean()

# 1. Calculate cost and value of goals
history["Custo_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Odd_H"] / history["Goals_H_FT"], 0)
history["Custo_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Odd_A"] / history["Goals_A_FT"], 0)

history["Valor_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Bet Result"] / history["Goals_H_FT"], 0)
history["Valor_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Bet Result"] / history["Goals_A_FT"], 0)

# For today’s matches (no results yet → 0)
games_today["Custo_Gol_H"] = 0
games_today["Custo_Gol_A"] = 0
games_today["Valor_Gol_H"] = 0
games_today["Valor_Gol_A"] = 0

# 2. Rolling averages (last 5 matches, min 2) with shift for training
history = history.sort_values("Date")

history["Media_CustoGol_H"] = history.groupby("Home", group_keys=False).apply(
    lambda x: rolling_stats(x, "Custo_Gol_H")
).shift(1)
history["Media_ValorGol_H"] = history.groupby("Home", group_keys=False).apply(
    lambda x: rolling_stats(x, "Valor_Gol_H")
).shift(1)

history["Media_CustoGol_A"] = history.groupby("Away", group_keys=False).apply(
    lambda x: rolling_stats(x, "Custo_Gol_A")
).shift(1)
history["Media_ValorGol_A"] = history.groupby("Away", group_keys=False).apply(
    lambda x: rolling_stats(x, "Valor_Gol_A")
).shift(1)

# 3. Dynamic thresholds (percentiles)
t_c = history[["Media_CustoGol_H", "Media_CustoGol_A"]].stack().quantile(0.6)
t_v = history[["Media_ValorGol_H", "Media_ValorGol_A"]].stack().quantile(0.4)

st.sidebar.markdown(
    f"### 🔎 Dynamic thresholds (percentiles)\n- Cost (p60): {t_c:.2f}\n- Value (p40): {t_v:.2f}"
)

# 4. Classification
def classify_row_dynamic(custo, valor, t_c, t_v):
    if pd.isna(custo) or pd.isna(valor):
        return "—"
    if custo <= t_c and valor > t_v:
        return "🟢"
    elif custo <= t_c and valor <= t_v:
        return "⚪"
    elif custo > t_c and valor > t_v:
        return "🟡"
    else:
        return "🔴"

history["Categoria_Gol_H"] = history.apply(
    lambda r: classify_row_dynamic(r["Media_CustoGol_H"], r["Media_ValorGol_H"], t_c, t_v), axis=1
)
history["Categoria_Gol_A"] = history.apply(
    lambda r: classify_row_dynamic(r["Media_CustoGol_A"], r["Media_ValorGol_A"], t_c, t_v), axis=1
)

# 5. Categories for today’s matches (use min 2, max 5 last games)
def get_last_category(team, side, min_games=2, max_games=5):
    df = history[history["Home"] == team] if side == "H" else history[history["Away"] == team]
    df = df.sort_values("Date").tail(max_games)  # last up to 5 matches
    
    if len(df) < min_games:
        return "—"  # not enough history
    
    # calculate mean ignoring zeros (no goals)
    custo_mean = df[f"Custo_Gol_{side}"].replace(0, np.nan).mean()
    valor_mean = df[f"Valor_Gol_{side}"].replace(0, np.nan).mean()
    
    if pd.isna(custo_mean) or pd.isna(valor_mean):
        return "—"
    
    return classify_row_dynamic(custo_mean, valor_mean, t_c, t_v)

games_today["Categoria_Gol_H"] = games_today["Home"].apply(lambda t: get_last_category(t, "H"))
games_today["Categoria_Gol_A"] = games_today["Away"].apply(lambda t: get_last_category(t, "A"))

# 6. One-hot encoding for categories
cat_h = pd.get_dummies(history["Categoria_Gol_H"], prefix="Cat_H")
cat_a = pd.get_dummies(history["Categoria_Gol_A"], prefix="Cat_A")

cat_h_today = pd.get_dummies(games_today["Categoria_Gol_H"], prefix="Cat_H").reindex(columns=cat_h.columns, fill_value=0)
cat_a_today = pd.get_dummies(games_today["Categoria_Gol_A"], prefix="Cat_A").reindex(columns=cat_a.columns, fill_value=0)




##################### BLOCO 5 – BASE FEATURES #####################
if "cat_h" not in locals() or "cat_a" not in locals():
    st.error("❌ Goal categories (BLOCK 4) were not calculated before BLOCK 5.")
    st.stop()

# Feature differences
history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]

# Feature sets
features_1x2 = [
    "Odd_H", "Odd_D", "Odd_A",
    "Diff_Power", "M_H", "M_A", "Diff_M",
    "Diff_HT_P", "M_HT_H", "M_HT_A"
]

features_ou_btts = [
    "Odd_H", "Odd_D", "Odd_A",
    "Diff_Power", "M_H", "M_A", "Diff_M",
    "Diff_HT_P", "OU_Total"
]

# One-hot encode leagues
history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

# Final datasets
X_1x2 = pd.concat([history[features_1x2], history_leagues, cat_h, cat_a], axis=1)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1)

# Today’s matches
X_today_1x2 = pd.concat([games_today[features_1x2], games_today_leagues, cat_h_today, cat_a_today], axis=1)
X_today_1x2 = X_today_1x2.reindex(columns=X_1x2.columns, fill_value=0)

X_today_ou = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)
X_today_ou = X_today_ou.reindex(columns=X_ou.columns, fill_value=0)

X_today_btts = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)
X_today_btts = X_today_btts.reindex(columns=X_btts.columns, fill_value=0)


##################### BLOCO 6 – SIDEBAR #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"])
retrain = st.sidebar.checkbox("Retrain models", value=False)


##################### BLOCO 7 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_CG.pkl"
    feature_cols = X.columns.tolist()

    model_bundle = None if retrain else load_model(filename)

    if model_bundle is None:
        # New model
        if ml_model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced_subsample"
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

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = X_train.reindex(columns=feature_cols, fill_value=0)
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

        model.fit(X_train, y_train)
        save_model((model, feature_cols), filename)

    else:
        if isinstance(model_bundle, tuple):
            model, feature_cols = model_bundle
        else:
            model = model_bundle
            feature_cols = X.columns.tolist()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = X_train.reindex(columns=feature_cols, fill_value=0)
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)
    if num_classes == 2:
        bs = f"{brier_score_loss(y_val, probs[:, 1]):.3f}"
    else:
        y_onehot = pd.get_dummies(y_val).values
        bs_raw = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        bs = f"{bs_raw:.3f} (multi)"

    metrics = {"Model": f"{ml_model_choice} - {name}", "Accuracy": f"{acc:.3f}", "LogLoss": f"{ll:.3f}", "Brier": bs}
    return metrics, (model, feature_cols)


##################### BLOCO 8 – TRAIN MODELS #####################
stats = []
res, model_multi = train_and_evaluate(X_1x2, history["Target"], "1X2", 3); stats.append(res)
res, model_ou = train_and_evaluate(X_ou, history["Target_OU25"], "OverUnder25", 2); stats.append(res)
res, model_btts = train_and_evaluate(X_btts, history["Target_BTTS"], "BTTS", 2); stats.append(res)

df_stats = pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)


##################### BLOCO 9 – PREDICTIONS #####################
model_multi, cols1 = model_multi
model_ou, cols2 = model_ou
model_btts, cols3 = model_btts

X_today_1x2 = X_today_1x2.reindex(columns=cols1, fill_value=0)
X_today_ou = X_today_ou.reindex(columns=cols2, fill_value=0)
X_today_btts = X_today_btts.reindex(columns=cols3, fill_value=0)

games_today["p_home"], games_today["p_draw"], games_today["p_away"] = model_multi.predict_proba(X_today_1x2).T
games_today["p_over25"], games_today["p_under25"] = model_ou.predict_proba(X_today_ou).T
games_today["p_btts_yes"], games_today["p_btts_no"] = model_btts.predict_proba(X_today_btts).T


##################### BLOCO 10 – DISPLAY #####################
def color_prob(val, color):
    alpha = int(val * 255)
    return f"background-color: rgba({color}, {alpha/255:.2f})"

def style_probs(val, col):
    if col == "p_home": return color_prob(val, "0,200,0")
    if col == "p_draw": return color_prob(val, "150,150,150")
    if col == "p_away": return color_prob(val, "255,140,0")
    if col == "p_over25": return color_prob(val, "0,100,255")
    if col == "p_under25": return color_prob(val, "128,0,128")
    if col == "p_btts_yes": return color_prob(val, "0,200,200")
    if col == "p_btts_no": return color_prob(val, "200,0,0")
    return ""

cols_final = [
    "Date","Time","League","Home","Away",
    "Odd_H","Odd_D","Odd_A",
    "Categoria_Gol_H","Categoria_Gol_A",
    "p_home","p_draw","p_away","p_over25","p_under25","p_btts_yes","p_btts_no"
]

styled_df = (
    games_today[cols_final]
    .style.format({
        "Odd_H":"{:.2f}","Odd_D":"{:.2f}","Odd_A":"{:.2f}",
        "p_home":"{:.1%}","p_draw":"{:.1%}","p_away":"{:.1%}",
        "p_over25":"{:.1%}","p_under25":"{:.1%}",
        "p_btts_yes":"{:.1%}","p_btts_no":"{:.1%}"
    }, na_rep="—")
    .applymap(lambda v: style_probs(v,"p_home"), subset=["p_home"])
    .applymap(lambda v: style_probs(v,"p_draw"), subset=["p_draw"])
    .applymap(lambda v: style_probs(v,"p_away"), subset=["p_away"])
    .applymap(lambda v: style_probs(v,"p_over25"), subset=["p_over25"])
    .applymap(lambda v: style_probs(v,"p_under25"), subset=["p_under25"])
    .applymap(lambda v: style_probs(v,"p_btts_yes"), subset=["p_btts_yes"])
    .applymap(lambda v: style_probs(v,"p_btts_no"), subset=["p_btts_no"])
)

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df, use_container_width=True, height=1000)


##################### BLOCO 11 – GOAL CATEGORY ANALYSIS #####################
st.markdown("## 📊 Goal Category Results Analysis")

def resultado_jogo(row):
    if row["Goals_H_FT"] > row["Goals_A_FT"]:
        return "Win_H"
    elif row["Goals_H_FT"] < row["Goals_A_FT"]:
        return "Win_A"
    else:
        return "Draw"

history["Result"] = history.apply(resultado_jogo, axis=1)

def stats_por_categoria(df, col_categoria):
    stats = (
        df.groupby(col_categoria)["Result"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    stats["Games"] = df.groupby(col_categoria)["Result"].count().values
    return stats

cat_home_stats = stats_por_categoria(history, "Categoria_Gol_H")
cat_away_stats = stats_por_categoria(history, "Categoria_Gol_A")

st.markdown("### 🏠 Home Teams")
st.dataframe(cat_home_stats.style.format("{:.1%}", subset=["Draw","Win_H","Win_A"]))

st.markdown("### 🚗 Away Teams")
st.dataframe(cat_away_stats.style.format("{:.1%}", subset=["Draw","Win_H","Win_A"]))

st.markdown("""
📌 Interpretation  
🟢 = efficient team and decisive goals → “winning” profile.  
⚪ = efficient team but low impact goals → tends to draw.  
🟡 = inefficient team, but goals matter when they score → unstable profile.  
🔴 = inefficient team and irrelevant goals → “loser” profile.  
""")