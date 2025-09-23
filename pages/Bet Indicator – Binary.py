# ########################################################
# BLOCO 1 – Imports & Configurações
# ########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from collections import Counter

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report

# SMOTE para balanceamento
from imblearn.over_sampling import SMOTE

# ---------------- Configurações da Página ----------------
st.set_page_config(page_title="Bet Indicator – Home vs Away", layout="wide")
st.title("📊 AI-Powered Bet Indicator – Home vs Away (Binary)")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
MODELS_FOLDER = "Models"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa"]

os.makedirs(MODELS_FOLDER, exist_ok=True)


# ########################################################
# BLOCO 2 – Funções auxiliares
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


# ########################################################
# BLOCO 3 – Carregando dados históricos
# ########################################################
st.info("📂 Loading historical data...")
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
if all_games.empty:
    st.warning("No valid historical data found.")
    st.stop()

history = all_games.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
if history.empty:
    st.warning("No valid historical results found.")
    st.stop()


# ########################################################
# BLOCO 4 – Seleção do Matchday
# ########################################################
option = st.radio(
    "Select Matches",
    ("Today Matches", "Yesterday Matches"),
    horizontal=True
)

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No match files available.")
    st.stop()

if option == "Today Matches":
    selected_file = files[-1]  # latest file
elif option == "Yesterday Matches":
    if len(files) >= 2:
        selected_file = files[-2]  # second to last file
    else:
        st.warning("No yesterday matches available.")
        st.stop()

games_today = filter_leagues(pd.read_csv(os.path.join(GAMES_FOLDER, selected_file)))

# 🔹 Mantém apenas jogos futuros (sem placares ainda)
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

if games_today.empty:
    st.warning("No valid matches found for the selected day.")
    st.stop()


# ########################################################
# BLOCO 5 – Target binário e balanceamento inicial
# ########################################################
history = history[history['Goals_H_FT'] != history['Goals_A_FT']]  # remove draws
history['Target'] = history.apply(
    lambda row: 0 if row['Goals_H_FT'] > row['Goals_A_FT'] else 1,
    axis=1
)

# Ver distribuição de classes
class_counts = history['Target'].value_counts()
st.markdown("### ⚖️ Class Distribution (Home vs Away)")
st.write(pd.DataFrame({
    'Class': ['Home (0)', 'Away (1)'],
    'Count': [class_counts.get(0, 0), class_counts.get(1, 0)],
    'Percentage': [
        f"{class_counts.get(0, 0) / len(history) * 100:.1f}%",
        f"{class_counts.get(1, 0) / len(history) * 100:.1f}%"
    ]
}))


# ########################################################
# BLOCO 6 – Features básicas + Momentum
# ########################################################
history['Diff_M'] = history['M_H'] - history['M_A']
games_today['Diff_M'] = games_today['M_H'] - games_today['M_A']
history['Diff_Abs'] = (history['M_H'] - history['M_A']).abs()
games_today['Diff_Abs'] = (games_today['M_H'] - games_today['M_A']).abs()

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

base_features = [
    'Odd_H', 'Odd_D', 'Odd_A',
    'M_H', 'M_A', 'Diff_Power', 'Diff_M','Diff_Abs',
    'PesoMomentum_H', 'PesoMomentum_A',
    'CustoMomentum_H', 'CustoMomentum_A'
]


# ########################################################
# BLOCO 7 – One-hot Encoding + Duplicados tratados
# ########################################################
history_leagues = pd.get_dummies(history['League'], prefix="League")
games_today_leagues = pd.get_dummies(games_today['League'], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X = pd.concat([history[base_features], history_leagues], axis=1) \
        .fillna(0) \
        .join(history[["Date","Home","Away"]]) \
        .drop_duplicates(subset=["Date","Home","Away"], keep="first") \
        .drop(columns=["Date","Home","Away"])

y = history['Target']

X_today = pd.concat([games_today[base_features], games_today_leagues], axis=1) \
        .fillna(0) \
        .join(games_today[["Date","Home","Away"]]) \
        .drop_duplicates(subset=["Date","Home","Away"], keep="first") \
        .drop(columns=["Date","Home","Away"])
# ########################################################
# BLOCO 8 – Train / Validation + SMOTE
# ########################################################
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.info("Aplicando SMOTE para balancear as classes (Away)...")
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

st.write("Distribuição após SMOTE:", dict(Counter(y_train_res)))

scaler = StandardScaler()
X_train_scaled = X_train_res.copy()
X_val_scaled = X_val.copy()
X_today_scaled = X_today.copy()

X_train_scaled[base_features] = scaler.fit_transform(X_train_res[base_features].fillna(0))
X_val_scaled[base_features] = scaler.transform(X_val[base_features].fillna(0))
X_today_scaled[base_features] = scaler.transform(X_today[base_features].fillna(0))


# ########################################################
# BLOCO 9 – Treinando Modelos
# ########################################################
rf_tuned = RandomForestClassifier(
    n_estimators=500,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample"
)
rf_tuned.fit(X_train_res, y_train_res)

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
log_reg.fit(X_train_scaled, y_train_res)

model_choice = st.sidebar.radio(
    "Select Model",
    ("Random Forest (Tuned)", "Ensemble RF+Logistic"),
    index=0
)

# ########################################################
# BLOCO 10 – Validação e Métricas
# ########################################################
if model_choice == "Random Forest (Tuned)":
    preds = rf_tuned.predict(X_val)
    probs = rf_tuned.predict_proba(X_val)
elif model_choice == "Ensemble RF+Logistic":
    probs_rf = rf_tuned.predict_proba(X_val)
    probs_log = log_reg.predict_proba(X_val_scaled)
    probs = (0.7 * probs_rf) + (0.3 * probs_log)
    preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_val, preds)
ll = log_loss(y_val, probs)
bs = brier_score_loss(y_val, probs[:,1])

winrate_home = (preds[y_val==0] == 0).mean()
winrate_away = (preds[y_val==1] == 1).mean()

st.markdown("### 📊 Model Statistics (Validation)")
df_stats = pd.DataFrame([{
    "Model": model_choice,
    "Accuracy": f"{acc:.3f}",
    "LogLoss": f"{ll:.3f}",
    "Brier": f"{bs:.3f}",
    "Winrate_Home": f"{winrate_home:.2%}",
    "Winrate_Away": f"{winrate_away:.2%}"
}])
st.dataframe(df_stats, use_container_width=True)

report = classification_report(
    y_val, preds, target_names=["Home","Away"], output_dict=True
)
df_report = pd.DataFrame(report).transpose()
st.markdown("### 📑 Classification Report (Precision / Recall / F1)")
st.dataframe(df_report.style.format("{:.2f}"), use_container_width=True)


# ########################################################
# BLOCO 11 – Previsões para os jogos de hoje
# ########################################################
if model_choice == "Random Forest (Tuned)":
    probs_today = rf_tuned.predict_proba(X_today)
else:
    probs_rf_today = rf_tuned.predict_proba(X_today)
    probs_log_today = log_reg.predict_proba(X_today_scaled)
    probs_today = (0.7 * probs_rf_today) + (0.3 * probs_log_today)

games_today['p_home'] = probs_today[:,0]
games_today['p_away'] = probs_today[:,1]

cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Odd_H', 'Odd_A', 'PesoMomentum_H', 'PesoMomentum_A',
    'CustoMomentum_H', 'CustoMomentum_A',
    'p_home', 'p_away'
]

def color_prob(val, color):
    alpha = int((1 - val) * 255)
    return f'background-color: rgba({color}, {alpha/255:.2f})'

def style_probs(val, col):
    if col == 'p_home':
        return color_prob(val, "0,200,0")  # verde
    elif col == 'p_away':
        return color_prob(val, "255,140,0")  # laranja
    return ''

styled_df = (
    games_today[cols_to_show]
    .style.format({
        'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}',
        'PesoMomentum_H': '{:.2f}', 'PesoMomentum_A': '{:.2f}',
        'CustoMomentum_H': '{:.2f}', 'CustoMomentum_A': '{:.2f}',
        'p_home': '{:.1%}', 'p_away': '{:.1%}'
    }, na_rep='—')
    .applymap(lambda v: style_probs(v, 'p_home'), subset=['p_home'])
    .applymap(lambda v: style_probs(v, 'p_away'), subset=['p_away'])
)

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df, use_container_width=True, height=1000)



