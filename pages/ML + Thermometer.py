########################################
########## Bloco 1 – Imports ############
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


########################################
########## Bloco 2 – Configs ############
########################################
st.set_page_config(page_title="Today's Picks - Momentum Thermometer + ML", layout="wide")
st.title("📊 Momentum Thermometer + ML Prototype")

GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "copa","afc"]

M_DIFF_MARGIN = 0.30
POWER_MARGIN = 10
DOMINANT_THRESHOLD = 0.90


########################################
####### Bloco 3 – Helper Functions #####
########################################
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def filter_leagues(df):
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def prepare_history(df):
    required = ['Goals_H_FT', 'Goals_A_FT', 'M_H', 'M_A', 'Diff_Power', 'League']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    return df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])


########################################
####### Bloco 4 – Carregar Dados #######
########################################
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select matchday file:", options, index=len(options)-1)

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Só jogos sem resultado
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# Carrega histórico
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()


########################################
####### Bloco 5 – Features Extras ######
########################################
# Criar colunas auxiliares
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']
history['M_Diff'] = history['M_H'] - history['M_A']

# Aproximação odds 1X e X2
def compute_double_chance_odds(df):
    probs = pd.DataFrame()
    probs['p_H'] = 1 / df['Odd_H']
    probs['p_D'] = 1 / df['Odd_D']
    probs['p_A'] = 1 / df['Odd_A']
    probs = probs.div(probs.sum(axis=1), axis=0)
    df['Odd_1X'] = 1 / (probs['p_H'] + probs['p_D'])
    df['Odd_X2'] = 1 / (probs['p_A'] + probs['p_D'])
    return df

games_today = compute_double_chance_odds(games_today)

# Aqui entraria sua lógica de bandas, dominant, EV etc. 
# (mantive resumido para não duplicar todo o seu código anterior)


########################################
####### Bloco 6 – Regras Híbridas ######
########################################
def auto_recommendation_dynamic_winrate(row, history,
                                        min_games=5,
                                        min_winrate=45.0):
    """
    Escolhe recomendação baseada no maior Winrate (automático).
    - Primeiro tenta Home / Away / Draw.
    - Se Winrate >= min_winrate e n >= min_games → aceita.
    - Caso contrário → tenta fallback 1X / X2.
    - Se nada válido → ❌ Avoid.
    """
    candidates_main = ["🟢 Back Home", "🟠 Back Away", "⚪ Back Draw"]
    candidates_fallback = ["🟦 1X (Home/Draw)", "🟪 X2 (Away/Draw)"]

    best_rec, best_prob, best_ev, best_n = None, None, None, None

    # 1) Checa vitórias puras
    for rec in candidates_main:
        row_copy = row.copy()
        row_copy["Auto_Recommendation"] = rec
        n, p = win_prob_for_recommendation(history, row_copy)

        if p is None or n < min_games:
            continue

        odd_ref = None
        if rec == "🟢 Back Home": odd_ref = row.get("Odd_H")
        elif rec == "🟠 Back Away": odd_ref = row.get("Odd_A")
        elif rec == "⚪ Back Draw": odd_ref = row.get("Odd_D")

        ev = (p/100.0) * odd_ref - 1 if odd_ref and odd_ref > 1.0 else None

        if (best_prob is None) or (p > best_prob):
            best_rec, best_prob, best_ev, best_n = rec, p, ev, n

    if best_prob is not None and best_prob >= min_winrate:
        return best_rec, best_prob, best_ev, best_n

    # 2) Se não, checa 1X/X2
    for rec in candidates_fallback:
        row_copy = row.copy()
        row_copy["Auto_Recommendation"] = rec
        n, p = win_prob_for_recommendation(history, row_copy)

        if p is None or n < min_games:
            continue

        odd_ref = None
        if rec == "🟦 1X (Home/Draw)" and row.get("Odd_H") and row.get("Odd_D"):
            odd_ref = 1 / (1/row["Odd_H"] + 1/row["Odd_D"])
        elif rec == "🟪 X2 (Away/Draw)" and row.get("Odd_A") and row.get("Odd_D"):
            odd_ref = 1 / (1/row["Odd_A"] + 1/row["Odd_D"])

        ev = (p/100.0) * odd_ref - 1 if odd_ref and odd_ref > 1.0 else None

        if (best_prob is None) or (p > best_prob):
            best_rec, best_prob, best_ev, best_n = rec, p, ev, n

    # 3) Se ainda não achar nada aceitável
    if best_prob is None or best_prob < min_winrate:
        return "❌ Avoid", best_prob, best_ev, best_n

    return best_rec, best_prob, best_ev, best_n


def auto_recommendation_hybrid(row, history,
                               min_games=5,
                               min_winrate=45.0):
    """
    Híbrido:
    1. Tenta pelas regras fixas (bands, dominant, thresholds).
    2. Valida com histórico (Win Probability + EV).
    3. Se cair em Avoid → chama fallback automático (dynamic winrate).
    """
    rec = None
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    odd_d     = row.get('Odd_D')
    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    league_cls= row.get('League_Classification','Medium Variation')

    # === Regras determinísticas (resumidas) ===
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        rec = '🟢 Back Home'
    elif band_home == 'Bottom 20%' and band_away == 'Top 20%':
        rec = '🟠 Back Away'
    elif dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong']:
        if diff_m is not None and diff_m >= 0.90:
            rec = '🟢 Back Home'
    elif dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong']:
        if diff_m is not None and diff_m <= -0.90:
            rec = '🟪 X2 (Away/Draw)'
    elif (odd_d is not None and 2.5 <= odd_d <= 6.0) and (diff_pow is not None and -10 <= diff_pow <= 10):
        rec = '⚪ Back Draw'

    if rec is None:
        rec = '❌ Avoid'

    # === Validação no histórico ===
    row_copy = row.copy()
    row_copy["Auto_Recommendation"] = rec
    n, p = win_prob_for_recommendation(history, row_copy)

    odd_ref = None
    if rec == "🟢 Back Home": odd_ref = row.get("Odd_H")
    elif rec == "🟠 Back Away": odd_ref = row.get("Odd_A")
    elif rec == "⚪ Back Draw": odd_ref = row.get("Odd_D")
    elif rec == "🟦 1X (Home/Draw)" and row.get("Odd_1X"): odd_ref = row["Odd_1X"]
    elif rec == "🟪 X2 (Away/Draw)" and row.get("Odd_X2"): odd_ref = row["Odd_X2"]

    ev = (p/100.0) * odd_ref - 1 if (odd_ref and p) else None

    # Se falhar nas regras → fallback automático
    if rec == "❌ Avoid" or (p is None) or (n < min_games) or (p < min_winrate):
        return auto_recommendation_dynamic_winrate(row, history, min_games, min_winrate)

    return rec, p, ev, n


# === Aplicar regras nos jogos do dia ===
recs = games_today.apply(lambda r: auto_recommendation_hybrid(r, history), axis=1)
games_today["Auto_Recommendation"] = [x[0] for x in recs]
games_today["Win_Probability"] = [x[1] for x in recs]
games_today["EV"] = [x[2] for x in recs]
games_today["Games_Analyzed"] = [x[3] for x in recs]



########################################
####### Bloco 7 – Train ML Model #######
########################################
history = history.dropna(subset=['Goals_H_FT','Goals_A_FT'])

def map_result(row):
    if row['Goals_H_FT'] > row['Goals_A_FT']:
        return "Home"
    elif row['Goals_H_FT'] < row['Goals_A_FT']:
        return "Away"
    else:
        return "Draw"

history['Result'] = history.apply(map_result, axis=1)

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

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)


########################################
####### Bloco 8 – Apply ML to Today ####
########################################
X_today = games_today[features_raw].copy()

if 'Home_Band' in X_today: X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                         encoded_today_df.reset_index(drop=True)], axis=1)

ml_preds = model.predict(X_today)
ml_proba = model.predict_proba(X_today)

games_today["ML_Prediction"] = ml_preds
games_today["ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
games_today["ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]
games_today["ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]


########################################
####### Bloco 9 – Exibição Final #######
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
