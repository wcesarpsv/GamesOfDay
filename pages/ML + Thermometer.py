########################################
####### Bloco 1 – Preparação ###########
########################################
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Carrega histórico
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = [pd.read_csv(os.path.join(folder, f)) for f in files]
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

GAMES_FOLDER = "GamesDay"
history = load_all_games(GAMES_FOLDER)

# Filtro: só jogos com resultado
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])


########################################
########## Bloco 2 – Target ############
########################################
def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']:
        return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']:
        return "Away"
    else:
        return "Draw"

history['Result'] = history.apply(map_result, axis=1)


########################################
##### Bloco 3 – Features Modelo ########
########################################
# Reuso do que já existe no seu código
features_raw = [
    'M_H','M_A','Diff_Power','M_Diff',
    'Home_Band','Away_Band',
    'Dominant','League_Classification',
    'Odd_H','Odd_D','Odd_A','Odd_1X','Odd_X2',
    'EV','Games_Analyzed'
]

# Filtra só as colunas disponíveis
features_raw = [f for f in features_raw if f in history.columns]

X = history[features_raw].copy()

# Bands -> numérico
BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X:
    X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X:
    X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

# One-hot para categóricos (Dominant, League_Classification)
cat_cols = []
for col in ['Dominant','League_Classification']:
    if col in X:
        cat_cols.append(col)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if cat_cols:
    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

y = history['Result']


########################################
######## Bloco 4 – Train/Test ##########
########################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


########################################
####### Bloco 5 – Treino ML ############
########################################
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",   # evita bias pró-Home
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)


########################################
####### Bloco 6 – Métricas #############
########################################
acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_proba)
br  = brier_score_loss(pd.get_dummies(y_test).values.ravel(), y_proba.ravel())

print("=== Modelo Aprendendo das Features do Código ===")
print(f"Accuracy: {acc:.3f}")
print(f"Log Loss: {ll:.3f}")
print(f"Brier Score: {br:.3f}")
