########################################
####### Bloco 1 – Imports & Config #####
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

st.set_page_config(page_title="ML Prototype – With Model Features", layout="wide")
st.title("🤖 ML Prototype – Using Features from Rules Model")

GAMES_FOLDER = "GamesDay"

########################################
####### Bloco 2 – Load Data ############
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
    st.warning("Nenhum histórico encontrado.")
    st.stop()

history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

########################################
####### Bloco 3 – Target & Features ####
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

features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band',
    'Dominant','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed'
]

features_raw = [f for f in features_raw if f in history.columns]
X = history[features_raw].copy()

# Bands -> numérico
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

# One-hot Dominant / League_Classification
cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

y = history['Result']

########################################
####### Bloco 4 – Train/Test Split #####
########################################
from sklearn.utils.class_weight import compute_class_weight
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

########################################
####### Bloco 5 – Model Training #######
########################################
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

########################################
####### Bloco 6 – Metrics Output #######
########################################
acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_proba)
br  = brier_score_loss(pd.get_dummies(y_test).values.ravel(), y_proba.ravel())

st.subheader("📊 Model Performance (using your features)")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Log Loss", f"{ll:.3f}")
col3.metric("Brier Score", f"{br:.3f}")

########################################
####### Bloco 7 – Feature Importance ###
########################################
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.subheader("🔥 Top Feature Importances")
st.dataframe(importances.head(20))




########################################
####### Bloco X – ML Training ##########
########################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Prepara histórico (já carregado antes)
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']:
        return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']:
        return "Away"
    else:
        return "Draw"

history['Result'] = history.apply(map_result, axis=1)
history['M_Diff'] = history['M_H'] - history['M_A']

# Features usadas no treino
features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band',
    'Dominant','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed'
]
features_raw = [f for f in features_raw if f in history.columns]

X = history[features_raw].copy()
y = history['Result']

# Bands → numérico
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

# One-hot para categóricos
cat_cols = [c for c in ['Dominant','League_Classification'] if c in X]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

# Treino modelo ML
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)


########################################
### Bloco Y – Aplicar no Games Today ###
########################################
# Prepara features iguais nos jogos de hoje
X_today = games_today[features_raw].copy()

if 'Home_Band' in X_today: X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                         encoded_today_df.reset_index(drop=True)], axis=1)

# Predições
ml_preds = model.predict(X_today)
ml_proba = model.predict_proba(X_today)

games_today["ML_Prediction"] = ml_preds
games_today["ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
games_today["ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]
games_today["ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]

########################################
####### Bloco Z – Exibição #############
########################################
cols_to_show = [
    'Date','Time','League','Home','Away',
    'Auto_Recommendation','Win_Probability',
    'ML_Prediction','ML_Proba_Home','ML_Proba_Away','ML_Proba_Draw'
]

st.subheader("📊 Regras vs ML")
st.dataframe(
    games_today[cols_to_show]
    .style.format({
        'Win_Probability':'{:.1f}%',
        'ML_Proba_Home':'{:.2f}',
        'ML_Proba_Away':'{:.2f}',
        'ML_Proba_Draw':'{:.2f}'
    }),
    use_container_width=True
)
