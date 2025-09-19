##################### BLOCK 1 – IMPORTS & CONFIG #####################
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


##################### BLOCK 2 – HELPERS #####################
def preprocess_df(df):
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    if "Bet Result" not in df.columns:
        df["Bet Result"] = np.nan
    return df

def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if set(["Date", "Home", "Away", "League"]).issubset(df.columns):
        df = df.drop_duplicates(subset=["Date", "Home", "Away", "League"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")
    return df

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files: return pd.DataFrame()
    today_file = files[-1]
    yesterday_file = files[-2] if len(files) >= 2 else None
    st.markdown("### 📂 Select matches to display")
    col1, col2 = st.columns(2)
    today_checked = col1.checkbox("Today Matches", value=True)
    yesterday_checked = col2.checkbox("Yesterday Matches", value=False)
    dfs = []
    if today_checked: dfs.append(preprocess_df(pd.read_csv(os.path.join(folder, today_file))))
    if yesterday_checked and yesterday_file: dfs.append(preprocess_df(pd.read_csv(os.path.join(folder, yesterday_file))))
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if set(["Date", "Home", "Away", "League"]).issubset(df.columns):
        df = df.drop_duplicates(subset=["Date", "Home", "Away", "League"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")
    return df

def filter_leagues(df):
    if df.empty or "League" not in df.columns: return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, filename):
    with open(os.path.join(MODELS_FOLDER, filename), "wb") as f: joblib.dump(model, f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f: return joblib.load(f)
    return None


##################### BLOCK 3 – LOAD DATA #####################
st.info("📂 Loading data...")
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()
if history.empty: st.stop()
games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()
if games_today.empty: st.stop()

# Targets
history["Target"] = history.apply(lambda r: 0 if r["Goals_H_FT"] > r["Goals_A_FT"] else (1 if r["Goals_H_FT"]==r["Goals_A_FT"] else 2), axis=1)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"]>0) & (history["Goals_A_FT"]>0)).astype(int)


##################### BLOCK 4 – EXTRA FEATURES (COST/VALUE + DYNAMIC CATEGORIES) #####################
history["Custo_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Odd_H"] / history["Goals_H_FT"], np.nan)
history["Custo_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Odd_A"] / history["Goals_A_FT"], np.nan)
history["Valor_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Bet Result"] / history["Goals_H_FT"], np.nan)
history["Valor_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Bet Result"] / history["Goals_A_FT"], np.nan)
for col in ["Custo_Gol_H","Custo_Gol_A","Valor_Gol_H","Valor_Gol_A"]:
    games_today[col] = np.nan

history = history.sort_values("Date")
# training → shift(1)
history["Media_CustoGol_H"] = history.groupby("Home")["Custo_Gol_H"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
history["Media_ValorGol_H"] = history.groupby("Home")["Valor_Gol_H"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
history["Media_CustoGol_A"] = history.groupby("Away")["Custo_Gol_A"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
history["Media_ValorGol_A"] = history.groupby("Away")["Valor_Gol_A"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())

# thresholds
t_c_h = history["Media_CustoGol_H"].quantile(0.6)
t_c_a = history["Media_CustoGol_A"].quantile(0.6)
t_v_h = history["Media_ValorGol_H"].quantile(0.4)
t_v_a = history["Media_ValorGol_A"].quantile(0.4)
st.sidebar.markdown(f"""
### 🔎 Dynamic thresholds (percentiles)
- Cost H (p60): {t_c_h:.2f}  
- Value H (p40): {t_v_h:.2f}  
- Cost A (p60): {t_c_a:.2f}  
- Value A (p40): {t_v_a:.2f}  
""")

def classify_row_dynamic(custo, valor, t_c, t_v):
    if pd.isna(custo) or pd.isna(valor): return "—"
    if custo <= t_c and valor > t_v: return "🟢"
    elif custo <= t_c and valor <= t_v: return "⚪"
    elif custo > t_c and valor > t_v: return "🟡"
    else: return "🔴"

history["Categoria_Gol_H"] = history.apply(lambda r: classify_row_dynamic(r["Media_CustoGol_H"], r["Media_ValorGol_H"], t_c_h, t_v_h), axis=1)
history["Categoria_Gol_A"] = history.apply(lambda r: classify_row_dynamic(r["Media_CustoGol_A"], r["Media_ValorGol_A"], t_c_a, t_v_a), axis=1)

# production → rolling without shift
def get_last_mean(team, side, col):
    df = history[history["Home"] == team] if side == "H" else history[history["Away"] == team]
    if df.empty: return np.nan
    return df[col].rolling(window=5, min_periods=2).mean().iloc[-1]

games_today["Media_CustoGol_H"] = games_today["Home"].apply(lambda t: get_last_mean(t, "H", "Custo_Gol_H"))
games_today["Media_ValorGol_H"] = games_today["Home"].apply(lambda t: get_last_mean(t, "H", "Valor_Gol_H"))
games_today["Media_CustoGol_A"] = games_today["Away"].apply(lambda t: get_last_mean(t, "A", "Custo_Gol_A"))
games_today["Media_ValorGol_A"] = games_today["Away"].apply(lambda t: get_last_mean(t, "A", "Valor_Gol_A"))

def get_last_category(team, side):
    df = history[history["Home"] == team] if side == "H" else history[history["Away"] == team]
    row = df.sort_values("Date").tail(1)
    return row[f"Categoria_Gol_{side}"].iloc[0] if not row.empty else "—"

games_today["Categoria_Gol_H"] = games_today["Home"].apply(lambda t: get_last_category(t, "H"))
games_today["Categoria_Gol_A"] = games_today["Away"].apply(lambda t: get_last_category(t, "A"))

expected_cats = ["🟢","⚪","🟡","🔴"]
cat_h = pd.get_dummies(history["Categoria_Gol_H"], prefix="Cat_H").reindex(columns=[f"Cat_H_{c}" for c in expected_cats], fill_value=0)
cat_a = pd.get_dummies(history["Categoria_Gol_A"], prefix="Cat_A").reindex(columns=[f"Cat_A_{c}" for c in expected_cats], fill_value=0)
cat_h_today = pd.get_dummies(games_today["Categoria_Gol_H"], prefix="Cat_H").reindex(columns=cat_h.columns, fill_value=0)
cat_a_today = pd.get_dummies(games_today["Categoria_Gol_A"], prefix="Cat_A").reindex(columns=cat_a.columns, fill_value=0)


##################### BLOCK 5 – FEATURES #####################
if "cat_h" not in locals() or "cat_a" not in locals():
    st.error("❌ Goal categories (BLOCK 4) not calculated.")
    st.stop()

history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]

features_1x2 = ["Odd_H","Odd_D","Odd_A","Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","M_HT_H","M_HT_A",
                "Media_CustoGol_H","Media_ValorGol_H","Media_CustoGol_A","Media_ValorGol_A"]
features_ou_btts = ["Odd_H","Odd_D","Odd_A","Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","OU_Total",
                    "Media_CustoGol_H","Media_ValorGol_H","Media_CustoGol_A","Media_ValorGol_A"]

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League").reindex(columns=history_leagues.columns, fill_value=0)

X_1x2 = pd.concat([history[features_1x2], history_leagues, cat_h, cat_a], axis=1).fillna(0)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1).fillna(0)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1).fillna(0)

X_today_1x2 = pd.concat([games_today[features_1x2], games_today_leagues, cat_h_today, cat_a_today], axis=1).reindex(columns=X_1x2.columns, fill_value=0).fillna(0)
X_today_ou = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1).reindex(columns=X_ou.columns, fill_value=0).fillna(0)
X_today_btts = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1).reindex(columns=X_btts.columns, fill_value=0).fillna(0)


##################### BLOCK 6 – SIDEBAR #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest","Random Forest Tuned","XGBoost Tuned"])
retrain = st.sidebar.checkbox("Retrain models", value=False)


##################### BLOCK 7 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name, num_classes):
    # Unique file name
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_CG.pkl"
    feature_cols = X.columns.tolist()  # freeze feature list

    model_bundle = None if retrain else load_model(filename)

    if model_bundle is None:
        # Create new model
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

        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Align features
        X_train = X_train.reindex(columns=feature_cols, fill_value=0)
        X_val   = X_val.reindex(columns=feature_cols, fill_value=0)
        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)

        model.fit(X_train, y_train)
        save_model((model, feature_cols), filename)

    else:
        if isinstance(model_bundle, tuple):
            model, feature_cols = model_bundle
        else:
            model = model_bundle
            feature_cols = X.columns.tolist()

        # Split again
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Align features
        X_train = X_train.reindex(columns=feature_cols, fill_value=0)
        X_val   = X_val.reindex(columns=feature_cols, fill_value=0)
        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)

    # Predictions
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

    metrics = {"Model": f"{ml_model_choice} - {name}", 
               "Accuracy": f"{acc:.3f}", 
               "LogLoss": f"{ll:.3f}", 
               "Brier": bs}
    return metrics, (model, feature_cols)



##################### BLOCK 8 – TRAIN MODELS #####################
stats=[]; 
res,model_multi=train_and_evaluate(X_1x2,history["Target"],"1X2",3); stats.append(res)
res,model_ou=train_and_evaluate(X_ou,history["Target_OU25"],"OverUnder25",2); stats.append(res)
res,model_btts=train_and_evaluate(X_btts,history["Target_BTTS"],"BTTS",2); stats.append(res)

df_stats=pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats,use_container_width=True)


##################### BLOCK 9 – PREDICTIONS #####################
model_multi,cols1=model_multi; model_ou,cols2=model_ou; model_btts,cols3=model_btts
X_today_1x2=X_today_1x2.reindex(columns=cols1,fill_value=0)
X_today_ou=X_today_ou.reindex(columns=cols2,fill_value=0)
X_today_btts=X_today_btts.reindex(columns=cols3,fill_value=0)

games_today["p_home"],games_today["p_draw"],games_today["p_away"]=model_multi.predict_proba(X_today_1x2).T
games_today["p_over25"],games_today["p_under25"]=model_ou.predict_proba(X_today_ou).T
games_today["p_btts_yes"],games_today["p_btts_no"]=model_btts.predict_proba(X_today_btts).T


##################### BLOCK 10 – DISPLAY #####################
def color_prob(val,color):
    alpha=int(val*255)
    return f"background-color: rgba({color}, {alpha/255:.2f})"

def style_probs(val,col):
    if col=="p_home": return color_prob(val,"0,200,0")
    if col=="p_draw": return color_prob(val,"150,150,150")
    if col=="p_away": return color_prob(val,"255,140,0")
    if col=="p_over25": return color_prob(val,"0,100,255")
    if col=="p_under25": return color_prob(val,"128,0,128")
    if col=="p_btts_yes": return color_prob(val,"0,200,200")
    if col=="p_btts_no": return color_prob(val,"200,0,0")
    return ""

cols_final=["Date","Time","League","Home","Away","Odd_H","Odd_D","Odd_A",
            "Categoria_Gol_H","Categoria_Gol_A",
            "p_home","p_draw","p_away","p_over25","p_under25","p_btts_yes","p_btts_no"]

styled_df=(games_today[cols_final]
    .style.format({"Odd_H":"{:.2f}","Odd_D":"{:.2f}","Odd_A":"{:.2f}",
                   "p_home":"{:.1%}","p_draw":"{:.1%}","p_away":"{:.1%}",
                   "p_over25":"{:.1%}","p_under25":"{:.1%}",
                   "p_btts_yes":"{:.1%}","p_btts_no":"{:.1%}"},na_rep="—")
    .applymap(lambda v: style_probs(v,"p_home"),subset=["p_home"])
    .applymap(lambda v: style_probs(v,"p_draw"),subset=["p_draw"])
    .applymap(lambda v: style_probs(v,"p_away"),subset=["p_away"])
    .applymap(lambda v: style_probs(v,"p_over25"),subset=["p_over25"])
    .applymap(lambda v: style_probs(v,"p_under25"),subset=["p_under25"])
    .applymap(lambda v: style_probs(v,"p_btts_yes"),subset=["p_btts_yes"])
    .applymap(lambda v: style_probs(v,"p_btts_no"),subset=["p_btts_no"]))

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df,use_container_width=True,height=1000)


##################### BLOCK 11 – CATEGORY ANALYSIS #####################
st.markdown("## 📊 Analysis of Results by Goal Category")

def match_result(row):
    if row["Goals_H_FT"] > row["Goals_A_FT"]: 
        return "Win_H"
    elif row["Goals_H_FT"] < row["Goals_A_FT"]: 
        return "Win_A"
    else: 
        return "Draw"

# Only for history
history["Result"] = history.apply(match_result, axis=1)

def stats_by_category(df, col_categoria, custo_col, valor_col):
    stats = (
        df.groupby(col_categoria)["Result"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    stats["Games"] = df.groupby(col_categoria)["Result"].count().values
    stats["Avg_Custo"] = df.groupby(col_categoria)[custo_col].mean().values
    stats["Avg_Valor"] = df.groupby(col_categoria)[valor_col].mean().values
    return stats

# Home teams
cat_home_stats = stats_by_category(history, "Categoria_Gol_H", "Media_CustoGol_H", "Media_ValorGol_H")
# Away teams
cat_away_stats = stats_by_category(history, "Categoria_Gol_A", "Media_CustoGol_A", "Media_ValorGol_A")

# Show in Streamlit
st.markdown("### 🏠 Home Teams")
st.dataframe(cat_home_stats.style.format({
    "Draw": "{:.1%}", "Win_H": "{:.1%}", "Win_A": "{:.1%}",
    "Avg_Custo": "{:.2f}", "Avg_Valor": "{:.2f}"
}))

st.markdown("### 🚗 Away Teams")
st.dataframe(cat_away_stats.style.format({
    "Draw": "{:.1%}", "Win_H": "{:.1%}", "Win_A": "{:.1%}",
    "Avg_Custo": "{:.2f}", "Avg_Valor": "{:.2f}"
}))

st.markdown("""
📌 Interpretation  
🟢 = efficient team and decisive goals → “winning” profile.  
⚪ = efficient team but low impact goals → tendency to draws.  
🟡 = inefficient team, but when they score it matters → unstable profile.  
🔴 = inefficient team and irrelevant goals → “loser” profile.  
""")