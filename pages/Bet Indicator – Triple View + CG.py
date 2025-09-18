# Bet Indicator – Triple View + Custo/Valor do Gol
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator – Triple View", layout="wide")
st.title("📊 Bet Indicator – Triple View (1X2 + OU + BTTS + Goal Categories)")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ---------------- Helpers ----------------
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

# ---------------- Load Data ----------------
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

# ---------------- Targets ----------------
history["Target"] = history.apply(
    lambda row: 0 if row["Goals_H_FT"] > row["Goals_A_FT"]
    else (1 if row["Goals_H_FT"] == row["Goals_A_FT"] else 2),
    axis=1,
)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)

# ---------------- Extra Features: Custo & Valor do Gol ----------------
def rolling_stats(sub_df, col, window=5, min_periods=2):
    return sub_df.sort_values("Date")[col].rolling(window=window, min_periods=min_periods).mean()

# Histórico (tem resultados)
if "Goals_H_FT" in history.columns:
    history["Custo_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Odd_H"] / history["Goals_H_FT"], 0)
    history["Custo_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Odd_A"] / history["Goals_A_FT"], 0)

    if "Bet Result" in history.columns:
        history["Valor_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Bet Result"] / history["Goals_H_FT"], 0)
        history["Valor_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Bet Result"] / history["Goals_A_FT"], 0)
    else:
        history["Valor_Gol_H"] = 0
        history["Valor_Gol_A"] = 0


# Jogos do dia (sem resultados → apenas categorias herdadas)
games_today["Custo_Gol_H"] = 0
games_today["Custo_Gol_A"] = 0
games_today["Valor_Gol_H"] = 0
games_today["Valor_Gol_A"] = 0

# Rolling médias no histórico
history = history.sort_values("Date")
history["Media_CustoGol_H"] = history.groupby("Home", group_keys=False).apply(lambda x: rolling_stats(x, "Custo_Gol_H")).shift(1)
history["Media_ValorGol_H"] = history.groupby("Home", group_keys=False).apply(lambda x: rolling_stats(x, "Valor_Gol_H")).shift(1)
history["Media_CustoGol_A"] = history.groupby("Away", group_keys=False).apply(lambda x: rolling_stats(x, "Custo_Gol_A")).shift(1)
history["Media_ValorGol_A"] = history.groupby("Away", group_keys=False).apply(lambda x: rolling_stats(x, "Valor_Gol_A")).shift(1)

# Função de classificação
def classify_row(custo, valor, threshold_custo=1.5, threshold_valor=0):
    if pd.isna(custo) or pd.isna(valor):
        return "—"
    if custo <= threshold_custo and valor > threshold_valor:
        return "🟢"
    elif custo <= threshold_custo and valor <= threshold_valor:
        return "⚪"
    elif custo > threshold_custo and valor > threshold_valor:
        return "🟡"
    else:
        return "🔴"

# Classificação no histórico
history["Categoria_Gol_H"] = history.apply(lambda row: classify_row(row["Media_CustoGol_H"], row["Media_ValorGol_H"]), axis=1)
history["Categoria_Gol_A"] = history.apply(lambda row: classify_row(row["Media_CustoGol_A"], row["Media_ValorGol_A"]), axis=1)

# Propagar categorias para jogos do dia
def get_last_category(team, side):
    if side == "H":
        row = history[history["Home"] == team].sort_values("Date").tail(1)
        return row["Categoria_Gol_H"].iloc[0] if not row.empty else "—"
    else:
        row = history[history["Away"] == team].sort_values("Date").tail(1)
        return row["Categoria_Gol_A"].iloc[0] if not row.empty else "—"

games_today["Categoria_Gol_H"] = games_today["Home"].apply(lambda t: get_last_category(t, "H"))
games_today["Categoria_Gol_A"] = games_today["Away"].apply(lambda t: get_last_category(t, "A"))

# One-hot categorias
cat_h = pd.get_dummies(history["Categoria_Gol_H"], prefix="Cat_H")
cat_a = pd.get_dummies(history["Categoria_Gol_A"], prefix="Cat_A")
cat_h_today = pd.get_dummies(games_today["Categoria_Gol_H"], prefix="Cat_H")
cat_a_today = pd.get_dummies(games_today["Categoria_Gol_A"], prefix="Cat_A")
cat_h_today = cat_h_today.reindex(columns=cat_h.columns, fill_value=0)
cat_a_today = cat_a_today.reindex(columns=cat_a.columns, fill_value=0)

# ---------------- Features ----------------
history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]

features_1x2 = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A"]
features_ou_btts = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "OU_Total"]

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X_1x2 = pd.concat([history[features_1x2], history_leagues, cat_h, cat_a], axis=1)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1)

X_today_1x2 = pd.concat([games_today[features_1x2], games_today_leagues, cat_h_today, cat_a_today], axis=1)
X_today_ou = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)
X_today_btts = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)

# ---------------- Sidebar Config ----------------
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"])
retrain = st.sidebar.checkbox("Retrain models", value=False)

st.sidebar.markdown("""
**ℹ️ Usage recommendations:**
- 🔹 *Random Forest*: simple and fast baseline.  
- 🔹 *Random Forest Tuned*: suitable for market **1X2**.  
- 🔹 *XGBoost Tuned*: suitable for markets **Over/Under 2.5** e **BTTS**.  
""")

# ---------------- Train & Evaluate ----------------
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}.pkl"
    model = None

    if not retrain:
        model = load_model(filename)

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

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        save_model(model, filename)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

    metrics = {"Model": f"{ml_model_choice} - {name}", "Accuracy": f"{acc:.3f}", "LogLoss": f"{ll:.3f}", "Brier": bs}

    if num_classes == 3:
        metrics.update({
            "Winrate_Home": f"{(preds[y_val == 0] == 0).mean():.2%}",
            "Winrate_Draw": f"{(preds[y_val == 1] == 1).mean():.2%}",
            "Winrate_Away": f"{(preds[y_val == 2] == 2).mean():.2%}",
        })
    elif num_classes == 2:
        if name == "OverUnder25":
            metrics.update({
                "Winrate_Over25": f"{(preds[y_val == 1] == 1).mean():.2%}",
                "Winrate_Under25": f"{(preds[y_val == 0] == 0).mean():.2%}",
            })
        elif name == "BTTS":
            metrics.update({
                "Winrate_BTTS_Yes": f"{(preds[y_val == 1] == 1).mean():.2%}",
                "Winrate_BTTS_No": f"{(preds[y_val == 0] == 0).mean():.2%}",
            })

    return metrics, model

# ---------------- Train models ----------------
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

# ---------------- Predictions ----------------
games_today["p_home"], games_today["p_draw"], games_today["p_away"] = model_multi.predict_proba(X_today_1x2).T
games_today["p_over25"], games_today["p_under25"] = model_ou.predict_proba(X_today_ou).T
games_today["p_btts_yes"], games_today["p_btts_no"] = model_btts.predict_proba(X_today_btts).T

# ---------------- Styling ----------------
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

# ---------------- Display ----------------
cols_final = [
    "Date","Time","League","Home","Away",
    "Odd_H","Odd_D","Odd_A",
    "Categoria_Gol_H","Categoria_Gol_A",
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

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df, use_container_width=True, height=1000)

# ---------------- Legend ----------------
st.markdown("""
### 🟢⚪🟡🔴 Goal Categories – Legend

- 🟢 **(Baixo Custo, Alto Valor)** → Time eficiente e gols decisivos (perfil ideal).  
- ⚪ **(Baixo Custo, Baixo Valor)** → Time eficiente, mas gols pouco relevantes.  
- 🟡 **(Alto Custo, Alto Valor)** → Time ineficiente, mas quando marca, os gols decidem jogos.  
- 🔴 **(Alto Custo, Baixo Valor)** → Time ineficiente e gols pouco impactantes.  
- — Sem histórico suficiente para classificar.  
""")

