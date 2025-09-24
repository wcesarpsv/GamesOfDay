########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(page_title="ML Prototype – With Leagues", layout="wide")
st.title("🤖 ML Prototype – League-Aware Model")

GAMES_FOLDER = "GamesDay"


########################################
####### Bloco 3 – Carregar Histórico ####
########################################
@st.cache_data
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Erro carregando {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

history = load_all_games(GAMES_FOLDER)

if history.empty:
    st.warning("Nenhum dado histórico encontrado em GamesDay.")
    st.stop()

# Mantém apenas jogos com resultado
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])


########################################
####### Bloco 4 – Preparar Features #####
########################################
def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']:
        return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']:
        return "Away"
    else:
        return "Draw"

history['Result'] = history.apply(map_result, axis=1)
history['M_Diff'] = history['M_H'] - history['M_A']

features_base = ['M_H','M_A','Diff_Power','M_Diff']

# Bandas numéricas (se existirem)
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in history and 'Away_Band' in history:
    history['Home_Band_Num'] = history['Home_Band'].map(BAND_MAP)
    history['Away_Band_Num'] = history['Away_Band'].map(BAND_MAP)
    features_base += ['Home_Band_Num','Away_Band_Num']

# One-hot encoding da Liga
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
league_encoded = encoder.fit_transform(history[['League']])
league_df = pd.DataFrame(league_encoded, columns=encoder.get_feature_names_out(['League']))
history = pd.concat([history.reset_index(drop=True), league_df.reset_index(drop=True)], axis=1)

features = features_base + list(league_df.columns)


########################################
####### Bloco 5 – Train/Test Split #####
########################################
X = history[features]
y = history['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


########################################
######### Bloco 6 – Modelo ML ##########
########################################
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)


########################################
###### Bloco 7 – Métricas & Output #####
########################################
acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_proba)
br  = brier_score_loss(pd.get_dummies(y_test).values.ravel(), y_proba.ravel())

st.subheader("📊 Model Performance (with League Features)")
st.markdown(f"- **Accuracy:** {acc:.3f}")
st.markdown(f"- **Log Loss:** {ll:.3f}")
st.markdown(f"- **Brier Score:** {br:.3f}")

# Feature Importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
st.subheader("🔥 Feature Importances")
st.dataframe(importances.head(20))
