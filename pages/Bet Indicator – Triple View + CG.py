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

    # Garantir coluna Bet Result
    if "Bet Result" not in df.columns:
        df["Bet Result"] = 0

    # Garantir odds, se não existirem
    for col in ["Odd_H", "Odd_D", "Odd_A"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)


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

# Load full history
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

# Ensure no duplicates: Date + Home + Away
if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
else:
    history = history.drop_duplicates(keep="first")

if history.empty:
    st.stop()

# Load today's matches
games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))

# Ensure no duplicates
if set(["Date", "Home", "Away"]).issubset(games_today.columns):
    games_today = games_today.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
else:
    games_today = games_today.drop_duplicates(keep="first")

# Remove matches that already have final scores
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.stop()

# Targets
history["Target"] = history.apply(
    lambda r: 0 if r["Goals_H_FT"] > r["Goals_A_FT"]
    else (1 if r["Goals_H_FT"] == r["Goals_A_FT"] else 2), axis=1
)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)


##################### BLOCO 4 – EXTRA FEATURES #####################
def rolling_stats(sub_df, col, window=5, min_periods=1):
    return sub_df.sort_values("Date")[col].rolling(window=window, min_periods=min_periods).mean()

# 1. Cost & Value
history["Custo_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Odd_H"] / history["Goals_H_FT"], 0)
history["Custo_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Odd_A"] / history["Goals_A_FT"], 0)

history["Valor_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Bet Result"] / history["Goals_H_FT"], 0)
history["Valor_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Bet Result"] / history["Goals_A_FT"], 0)

# Today’s matches → inicializa com NaN
for col in ["Custo_Gol_H", "Custo_Gol_A", "Valor_Gol_H", "Valor_Gol_A",
            "Media_CustoGol_H", "Media_ValorGol_H", "Media_CustoGol_A", "Media_ValorGol_A"]:
    games_today[col] = np.nan

# 2. Rolling averages
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

# 3. Dynamic thresholds
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

# 5. Categories for today’s matches
def get_last_category(team, side, min_games=2, max_games=5):
    df = history[history["Home"] == team] if side == "H" else history[history["Away"] == team]
    df = df.sort_values("Date").tail(max_games)
    if len(df) < min_games or df.empty:
        return "—"
    return df.iloc[-1][f"Categoria_Gol_{side}"]

games_today["Categoria_Gol_H"] = games_today["Home"].apply(lambda t: get_last_category(t, "H"))
games_today["Categoria_Gol_A"] = games_today["Away"].apply(lambda t: get_last_category(t, "A"))

# 6. One-hot encoding for categories
cat_h = pd.get_dummies(history["Categoria_Gol_H"], prefix="Cat_H")
cat_a = pd.get_dummies(history["Categoria_Gol_A"], prefix="Cat_A")

cat_h_today = pd.get_dummies(games_today["Categoria_Gol_H"], prefix="Cat_H").reindex(columns=cat_h.columns, fill_value=0)
cat_a_today = pd.get_dummies(games_today["Categoria_Gol_A"], prefix="Cat_A").reindex(columns=cat_a.columns, fill_value=0)

# 7. Preencher games_today com as últimas médias reais do histórico
for idx, row in games_today.iterrows():
    home = row["Home"]
    away = row["Away"]

    last_home = history[history["Home"] == home].sort_values("Date").tail(1)
    last_away = history[history["Away"] == away].sort_values("Date").tail(1)

    games_today.at[idx, "Media_CustoGol_H"] = last_home["Media_CustoGol_H"].values[-1] if not last_home.empty else np.nan
    games_today.at[idx, "Media_ValorGol_H"] = last_home["Media_ValorGol_H"].values[-1] if not last_home.empty else np.nan

    games_today.at[idx, "Media_CustoGol_A"] = last_away["Media_CustoGol_A"].values[-1] if not last_away.empty else np.nan
    games_today.at[idx, "Media_ValorGol_A"] = last_away["Media_ValorGol_A"].values[-1] if not last_away.empty else np.nan



##################### BLOCO 5 – BASE FEATURES #####################
if "cat_h" not in locals() or "cat_a" not in locals():
    st.error("❌ Goal categories (BLOCK 4) were not calculated before BLOCK 5.")
    st.stop()

# Feature differences
history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]

# Feature sets (incluindo médias de custo/valor do gol)
features_1x2 = [
    "Odd_H", "Odd_D", "Odd_A",
    "Diff_Power", "M_H", "M_A", "Diff_M",
    "Diff_HT_P", "M_HT_H", "M_HT_A",
    "Media_CustoGol_H", "Media_ValorGol_H",
    "Media_CustoGol_A", "Media_ValorGol_A"
]

features_ou_btts = [
    "Odd_H", "Odd_D", "Odd_A",
    "Diff_Power", "M_H", "M_A", "Diff_M",
    "Diff_HT_P", "OU_Total",
    "Media_CustoGol_H", "Media_ValorGol_H",
    "Media_CustoGol_A", "Media_ValorGol_A"
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

    # Try to load model
    if not retrain:
        model = load_model(filename)
        if model:
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {
                "Model": name,
                "Accuracy": accuracy_score(y, preds),
                "LogLoss": log_loss(y, probs, labels=np.arange(num_classes)),
                "BrierScore": brier_score_loss(pd.get_dummies(y).values.ravel(), probs.ravel())
            }
            return res, (model, feature_cols)

    # Train new model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if "Random Forest" in ml_model_choice:
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    elif "XGBoost" in ml_model_choice:
        model = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            use_label_encoder=False, random_state=42
        )
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "LogLoss": log_loss(y_test, probs, labels=np.arange(num_classes)),
        "BrierScore": brier_score_loss(pd.get_dummies(y_test).values.ravel(), probs.ravel())
    }

    save_model(model, filename)
    return res, (model, feature_cols)


##################### BLOCO 8 – TRAIN MODELS #####################
stats = []
res, model_multi = train_and_evaluate(X_1x2, history["Target"], "1X2", 3); stats.append(res)
res, model_ou = train_and_evaluate(X_ou, history["Target_OU25"], "OverUnder25", 2); stats.append(res)
res, model_btts = train_and_evaluate(X_btts, history["Target_BTTS"], "BTTS", 2); stats.append(res)

df_stats = pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)

##################### FEATURE IMPORTANCE #####################
st.markdown("### 🔎 Feature Importance Analysis")

def plot_feature_importance(model_tuple, X, title):
    model, cols = model_tuple
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": cols, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
        st.markdown(f"#### {title}")
        st.bar_chart(fi_df.set_index("Feature"))
    else:
        st.warning(f"{title}: Modelo não fornece feature_importances_")

plot_feature_importance(model_multi, X_1x2, "1X2 Model – Top Features")
plot_feature_importance(model_ou, X_ou, "Over/Under 2.5 Model – Top Features")
plot_feature_importance(model_btts, X_btts, "BTTS Model – Top Features")


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
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({color}, {alpha:.2f})"

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
    "Media_CustoGol_H","Media_ValorGol_H",
    "Media_CustoGol_A","Media_ValorGol_A",
    "p_home","p_draw","p_away",
    "p_over25","p_under25",
    "p_btts_yes","p_btts_no"
]

styled_df = (
    games_today[cols_final]
    .style.format({
        "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
        "Media_CustoGol_H": "{:.2f}", "Media_ValorGol_H": "{:.2f}",
        "Media_CustoGol_A": "{:.2f}", "Media_ValorGol_A": "{:.2f}",
        "p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}",
        "p_over25": "{:.1%}", "p_under25": "{:.1%}",
        "p_btts_yes": "{:.1%}", "p_btts_no": "{:.1%}"
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
