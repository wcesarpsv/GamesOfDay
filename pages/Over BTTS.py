import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# ---------------- Page Config ----------------
st.set_page_config(page_title="Bet Indicator v1.1 (RF + OU + BTTS)", layout="wide")
st.title("📊 AI-Powered Bet Indicator – Random Forest + OU/BTTS")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc"]

# ---------------- Helpers ----------------
def load_all_games(folder):
    """Carrega todos os CSVs para montar histórico (com gols)."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Erro ao carregar {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def load_last_two_csvs(folder):
    """Carrega só o último ou penúltimo CSV (jogos do dia/ontem)."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    files = sorted(files)
    options = [files[-1]]
    if len(files) >= 2:
        options.insert(0, files[-2])
    selected_file = st.selectbox("📂 Escolha o arquivo (Hoje/ Ontem):", options=options, index=len(options)-1)
    return pd.read_csv(os.path.join(folder, selected_file))

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

# ---------------- Load Data ----------------
st.info("📂 Carregando dados...")

# Histórico: todos os CSVs (só com jogos já finalizados)
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = all_games.dropna(subset=['Goals_H_FT', 'Goals_A_FT']).copy()

if history.empty:
    st.error("⚠️ Nenhum histórico com gols encontrado em GamesDay.")
    st.stop()

# Jogos do dia: último ou penúltimo CSV (mesmo sem gols)
games_today = filter_leagues(load_last_two_csvs(GAMES_FOLDER))
if games_today.empty:
    st.error("⚠️ Nenhum jogo encontrado no arquivo selecionado.")
    st.stop()

# ---------------- Targets (só para treino) ----------------
# 1X2
history['Target'] = history.apply(
    lambda row: 0 if row['Goals_H_FT'] > row['Goals_A_FT']
    else (1 if row['Goals_H_FT'] == row['Goals_A_FT'] else 2),
    axis=1
)

# Over/Under 2.5
history['Target_OU25'] = history.apply(
    lambda row: 1 if (row['Goals_H_FT'] + row['Goals_A_FT']) > 2.5 else 0,
    axis=1
)

# BTTS
history['Target_BTTS'] = history.apply(
    lambda row: 1 if (row['Goals_H_FT'] > 0 and row['Goals_A_FT'] > 0) else 0,
    axis=1
)

# ---------------- Features ----------------
history['Diff_M'] = history['M_H'] - history['M_A']
games_today['Diff_M'] = games_today['M_H'] - games_today['M_A']

base_features = [
    'Odd_H', 'Odd_A', 'Odd_D',
    'M_H', 'M_A', 'Diff_Power', 'Diff_M',
    'Diff_HT_P', 'OU_Total'
]

# One-hot encode League
history_leagues = pd.get_dummies(history['League'], prefix="League")
games_today_leagues = pd.get_dummies(games_today['League'], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

# Montar features finais
X = pd.concat([history[base_features], history_leagues], axis=1)
X_today = pd.concat([games_today[base_features], games_today_leagues], axis=1)

# ---------------- Train models ----------------
def train_rf(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        class_weight="balanced_subsample"
    )
    model.fit(X, y)
    return model

# 1X2
y = history['Target']
model_multi = train_rf(X, y)

# Over/Under
y_ou = history['Target_OU25']
model_ou = train_rf(X, y_ou)

# BTTS
y_btts = history['Target_BTTS']
model_btts = train_rf(X, y_btts)

# ---------------- Evaluation ----------------
def evaluate_model(model, X, y, name):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    bs = brier_score_loss(y, probs[:,1]) if probs.shape[1] == 2 else "—"
    return {
        "Modelo": name,
        "Acurácia": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
        "Brier": f"{bs:.3f}" if bs != "—" else "—"
    }

stats = []
stats.append(evaluate_model(model_multi, X, y, "1X2"))
stats.append(evaluate_model(model_ou, X, y_ou, "Over/Under 2.5"))
stats.append(evaluate_model(model_btts, X, y_btts, "BTTS"))

df_stats = pd.DataFrame(stats)

# Exibir estatísticas
st.markdown("### 📊 Estatísticas do Modelo (Treino)")
st.dataframe(df_stats, use_container_width=True)

# ---------------- Predict ----------------
# 1X2
probs = model_multi.predict_proba(X_today)
games_today['p_home'] = probs[:, 0]
games_today['p_draw'] = probs[:, 1]
games_today['p_away'] = probs[:, 2]

# OU 2.5
probs_ou = model_ou.predict_proba(X_today)
games_today['p_over25'] = probs_ou[:, 1]
games_today['p_under25'] = probs_ou[:, 0]

# BTTS
probs_btts = model_btts.predict_proba(X_today)
games_today['p_btts_yes'] = probs_btts[:, 1]
games_today['p_btts_no'] = probs_btts[:, 0]

# ---------------- Styling ----------------
def color_prob(val, color):
    alpha = int(val * 255)
    return f'background-color: rgba({color}, {alpha/255:.2f})'

def style_probs(val, col):
    if col == 'p_home':
        return color_prob(val, "0,200,0")      # verde
    elif col == 'p_draw':
        return color_prob(val, "150,150,150")  # cinza
    elif col == 'p_away':
        return color_prob(val, "255,140,0")    # laranja
    elif col == 'p_over25':
        return color_prob(val, "0,100,255")    # azul
    elif col == 'p_under25':
        return color_prob(val, "128,0,128")    # roxo
    elif col == 'p_btts_yes':
        return color_prob(val, "0,200,200")    # ciano
    elif col == 'p_btts_no':
        return color_prob(val, "200,0,0")      # vermelho
    return ''

# ---------------- Display ----------------
cols_final = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Odd_H', 'Odd_D', 'Odd_A',
    'p_home', 'p_draw', 'p_away',
    'p_over25', 'p_under25',
    'p_btts_yes', 'p_btts_no'
]

styled_df = (
    games_today[cols_final]
    .style.format({
        'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
        'p_home': '{:.1%}', 'p_draw': '{:.1%}', 'p_away': '{:.1%}',
        'p_over25': '{:.1%}', 'p_under25': '{:.1%}',
        'p_btts_yes': '{:.1%}', 'p_btts_no': '{:.1%}',
    }, na_rep='—')
    .applymap(lambda v: style_probs(v, 'p_home'), subset=['p_home'])
    .applymap(lambda v: style_probs(v, 'p_draw'), subset=['p_draw'])
    .applymap(lambda v: style_probs(v, 'p_away'), subset=['p_away'])
    .applymap(lambda v: style_probs(v, 'p_over25'), subset=['p_over25'])
    .applymap(lambda v: style_probs(v, 'p_under25'), subset=['p_under25'])
    .applymap(lambda v: style_probs(v, 'p_btts_yes'), subset=['p_btts_yes'])
    .applymap(lambda v: style_probs(v, 'p_btts_no'), subset=['p_btts_no'])
)

st.markdown("### 📌 Probabilidades dos Jogos Selecionados")
st.dataframe(styled_df, use_container_width=True, height=1000)
