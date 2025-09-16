import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator v1.3b (Test Mode)", layout="wide")
st.title("📊 Bet Indicator – RF + OU/BTTS (Test Dataset Option)")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc"]

# ---------------- Helpers ----------------
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(os.path.join(folder, f)) for f in files], ignore_index=True)

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# ---------------- Dataset Selection ----------------
st.markdown("### ⚙️ Select Training Dataset")

train_mode = st.radio(
    "Choose training source:",
    ["All CSVs (full history)", "Single CSV (for test)"],
    index=0
)

if train_mode == "Single CSV (for test)":
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
    if not files:
        st.error("No CSV files found in GamesDay.")
        st.stop()
    selected_file = st.selectbox("Select training CSV:", files)
    history = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
else:
    history = load_all_games(GAMES_FOLDER)

history = filter_leagues(history)
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()

if history.empty:
    st.error("No valid historical data found.")
    st.stop()

# ---------------- Targets ----------------
history['Target'] = history.apply(
    lambda row: 0 if row['Goals_H_FT'] > row['Goals_A_FT']
    else (1 if row['Goals_H_FT'] == row['Goals_A_FT'] else 2),
    axis=1
)
history['Target_OU25'] = (history['Goals_H_FT'] + history['Goals_A_FT'] > 2.5).astype(int)
history['Target_BTTS'] = ((history['Goals_H_FT'] > 0) & (history['Goals_A_FT'] > 0)).astype(int)

# ---------------- Features ----------------
history['Diff_M'] = history['M_H'] - history['M_A']

features_1x2 = ["Odd_H","Odd_D","Odd_A","Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","M_HT_H","M_HT_A"]
features_ou_btts = ["Odd_H","Odd_D","Odd_A","Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","OU_Total"]

history_leagues = pd.get_dummies(history['League'], prefix="League")

X_1x2 = pd.concat([history[features_1x2], history_leagues], axis=1)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1)

# ---------------- Train & Evaluate ----------------
def train_and_evaluate_rf(X, y, name, show_class_report=False):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)  # aligned with local script
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)
    bs = brier_score_loss(y_val, probs[:,1]) if probs.shape[1] == 2 else "—"

    metrics = {
        "Model": name,
        "Accuracy": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
        "Brier": f"{bs:.3f}" if bs != "—" else "—"
    }

    if show_class_report:
        report = classification_report(y_val, preds, target_names=["Home","Draw","Away"], output_dict=True)
        metrics.update({
            "Winrate_Home": f"{(preds[y_val==0]==0).mean():.2%}",
            "Winrate_Draw": f"{(preds[y_val==1]==1).mean():.2%}",
            "Winrate_Away": f"{(preds[y_val==2]==2).mean():.2%}"
        })
    return metrics, model

stats = []
res, model_multi = train_and_evaluate_rf(X_1x2, history['Target'], "1X2", show_class_report=True)
stats.append(res)
res, model_ou = train_and_evaluate_rf(X_ou, history['Target_OU25'], "Over/Under 2.5")
stats.append(res)
res, model_btts = train_and_evaluate_rf(X_btts, history['Target_BTTS'], "BTTS")
stats.append(res)

df_stats = pd.DataFrame(stats)

st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)
