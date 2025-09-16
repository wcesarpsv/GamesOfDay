import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator v1.4 (RF + OU + BTTS)", layout="wide")
st.title("📊 Bet Indicator – Random Forest + OU/BTTS")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc","sudamericana"]

# ---------------- Helpers ----------------
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(os.path.join(folder, f)) for f in files], ignore_index=True)

def load_last_csv(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    latest_file = max(files)
    return pd.read_csv(os.path.join(folder, latest_file))

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# ---------------- Load Data ----------------
st.info("📂 Loading data...")

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=['Goals_H_FT', 'Goals_A_FT']).copy()

if history.empty:
    st.error("⚠️ No valid historical data found in GamesDay.")
    st.stop()

games_today = filter_leagues(load_last_csv(GAMES_FOLDER))
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

if games_today.empty:
    st.error("⚠️ No valid games today.")
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
games_today['Diff_M'] = games_today['M_H'] - games_today['M_A']

features_1x2 = ["Odd_H","Odd_D","Odd_A","Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","M_HT_H","M_HT_A"]
features_ou_btts = ["Odd_H","Odd_D","Odd_A","Diff_Power","M_H","M_A","Diff_M","Diff_HT_P","OU_Total"]

# One-hot encode leagues
history_leagues = pd.get_dummies(history['League'], prefix="League")
games_today_leagues = pd.get_dummies(games_today['League'], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

# Final datasets
X_1x2 = pd.concat([history[features_1x2], history_leagues], axis=1)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1)

X_today_1x2 = pd.concat([games_today[features_1x2], games_today_leagues], axis=1)
X_today_ou = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)
X_today_btts = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)

# ---------------- Train & Evaluate ----------------
def train_and_evaluate_rf(X, y, name, show_class_report=False):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)

    if probs.shape[1] == 2:
        bs = brier_score_loss(y_val, probs[:,1])
        bs = f"{bs:.3f}"
    else:
        y_onehot = pd.get_dummies(y_val).values
        bs_raw = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        bs = f"{bs_raw:.3f} (multi)"

    metrics = {
        "Model": name,
        "Accuracy": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
        "Brier": bs
    }

    if show_class_report:
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

# ---------------- Predictions ----------------
games_today['p_home'], games_today['p_draw'], games_today['p_away'] = model_multi.predict_proba(X_today_1x2).T
games_today['p_over25'], games_today['p_under25'] = model_ou.predict_proba(X_today_ou).T
games_today['p_btts_yes'], games_today['p_btts_no'] = model_btts.predict_proba(X_today_btts).T

# ---------------- Styling ----------------
def color_prob(val, color):
    alpha = int(val * 255)
    return f'background-color: rgba({color}, {alpha/255:.2f})'

def style_probs(val, col):
    if col == 'p_home': return color_prob(val, "0,200,0")
    elif col == 'p_draw': return color_prob(val, "150,150,150")
    elif col == 'p_away': return color_prob(val, "255,140,0")
    elif col == 'p_over25': return color_prob(val, "0,100,255")
    elif col == 'p_under25': return color_prob(val, "128,0,128")
    elif col == 'p_btts_yes': return color_prob(val, "0,200,200")
    elif col == 'p_btts_no': return color_prob(val, "200,0,0")
    return ''

# ---------------- Display ----------------
cols_final = [
    'Date','Time','League','Home','Away',
    'Odd_H','Odd_D','Odd_A',
    'p_home','p_draw','p_away',
    'p_over25','p_under25',
    'p_btts_yes','p_btts_no'
]

styled_df = (
    games_today[cols_final]
    .style.format({
        'Odd_H': '{:.2f}','Odd_D': '{:.2f}','Odd_A': '{:.2f}',
        'p_home': '{:.1%}','p_draw': '{:.1%}','p_away': '{:.1%}',
        'p_over25': '{:.1%}','p_under25': '{:.1%}',
        'p_btts_yes': '{:.1%}','p_btts_no': '{:.1%}',
    }, na_rep='—')
    .applymap(lambda v: style_probs(v, 'p_home'), subset=['p_home'])
    .applymap(lambda v: style_probs(v, 'p_draw'), subset=['p_draw'])
    .applymap(lambda v: style_probs(v, 'p_away'), subset=['p_away'])
    .applymap(lambda v: style_probs(v, 'p_over25'), subset=['p_over25'])
    .applymap(lambda v: style_probs(v, 'p_under25'), subset=['p_under25'])
    .applymap(lambda v: style_probs(v, 'p_btts_yes'), subset=['p_btts_yes'])
    .applymap(lambda v: style_probs(v, 'p_btts_no'), subset=['p_btts_no'])
)

st.markdown("### 📌 Predictions for Today's Games")
st.dataframe(styled_df, use_container_width=True, height=1000)
