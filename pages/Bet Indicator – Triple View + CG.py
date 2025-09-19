##################### BLOCK 1 â€“ IMPORTS & CONFIG #####################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bet Indicator â€“ Triple View", layout="wide")
st.title("ðŸ“Š Bet Indicator â€“ Triple View (1X2 + OU + BTTS + Goal Categories)")

GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


##################### BLOCK 2 â€“ HELPERS #####################
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
    st.markdown("### ðŸ“‚ Select matches to display")
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


##################### BLOCK 3 â€“ LOAD DATA #####################
st.info("ðŸ“‚ Loading data...")
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


##################### BLOCK 4 â€“ EXTRA FEATURES (COST/VALUE + DYNAMIC CATEGORIES) #####################
history["Custo_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Odd_H"] / history["Goals_H_FT"], np.nan)
history["Custo_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Odd_A"] / history["Goals_A_FT"], np.nan)
history["Valor_Gol_H"] = np.where(history["Goals_H_FT"] > 0, history["Bet Result"] / history["Goals_H_FT"], np.nan)
history["Valor_Gol_A"] = np.where(history["Goals_A_FT"] > 0, history["Bet Result"] / history["Goals_A_FT"], np.nan)
for col in ["Custo_Gol_H","Custo_Gol_A","Valor_Gol_H","Valor_Gol_A"]:
    games_today[col] = np.nan

history = history.sort_values("Date")
history["Media_CustoGol_H"] = history.groupby("Home")["Custo_Gol_H"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
history["Media_ValorGol_H"] = history.groupby("Home")["Valor_Gol_H"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
history["Media_CustoGol_A"] = history.groupby("Away")["Custo_Gol_A"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())
history["Media_ValorGol_A"] = history.groupby("Away")["Valor_Gol_A"].transform(lambda x: x.shift().rolling(5, min_periods=2).mean())

t_c_h = history["Media_CustoGol_H"].quantile(0.6)
t_c_a = history["Media_CustoGol_A"].quantile(0.6)
t_v_h = history["Media_ValorGol_H"].quantile(0.4)
t_v_a = history["Media_ValorGol_A"].quantile(0.4)
st.sidebar.markdown(f"""
### ðŸ”Ž Dynamic thresholds (percentiles)
- Cost H (p60): {t_c_h:.2f}  
- Value H (p40): {t_v_h:.2f}  
- Cost A (p60): {t_c_a:.2f}  
- Value A (p40): {t_v_a:.2f}  
""")

def classify_row_dynamic(custo, valor, t_c, t_v):
    if pd.isna(custo) or pd.isna(valor): return "â€”"
    if custo <= t_c and valor > t_v: return "ðŸŸ¢"
    elif custo <= t_c and valor <= t_v: return "âšª"
    elif custo > t_c and valor > t_v: return "ðŸŸ¡"
    else: return "ðŸ”´"

history["Categoria_Gol_H"] = history.apply(lambda r: classify_row_dynamic(r["Media_CustoGol_H"], r["Media_ValorGol_H"], t_c_h, t_v_h), axis=1)
history["Categoria_Gol_A"] = history.apply(lambda r: classify_row_dynamic(r["Media_CustoGol_A"], r["Media_ValorGol_A"], t_c_a, t_v_a), axis=1)

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
    return row[f"Categoria_Gol_{side}"].iloc[0] if not row.empty else "â€”"

games_today["Categoria_Gol_H"] = games_today["Home"].apply(lambda t: get_last_category(t, "H"))
games_today["Categoria_Gol_A"] = games_today["Away"].apply(lambda t: get_last_category(t, "A"))

expected_cats = ["ðŸŸ¢","âšª","ðŸŸ¡","ðŸ”´"]
cat_h = pd.get_dummies(history["Categoria_Gol_H"], prefix="Cat_H").reindex(columns=[f"Cat_H_{c}" for c in expected_cats], fill_value=0)
cat_a = pd.get_dummies(history["Categoria_Gol_A"], prefix="Cat_A").reindex(columns=[f"Cat_A_{c}" for c in expected_cats], fill_value=0)
cat_h_today = pd.get_dummies(games_today["Categoria_Gol_H"], prefix="Cat_H").reindex(columns=cat_h.columns, fill_value=0)
cat_a_today = pd.get_dummies(games_today["Categoria_Gol_A"], prefix="Cat_A").reindex(columns=cat_a.columns, fill_value=0)