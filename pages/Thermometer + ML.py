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


########################################
####### Bloco 5B – Win Prob Helper #####
########################################
def event_side_for_winprob(auto_rec):
    if pd.isna(auto_rec): return None
    s = str(auto_rec)
    if 'Back Home' in s: return 'HOME'
    if 'Back Away' in s: return 'AWAY'
    if 'Back Draw' in s: return 'DRAW'
    if '1X' in s:       return '1X'
    if 'X2' in s:       return 'X2'
    return None

def win_prob_for_recommendation(history, row,
                                base_m_diff=0.30,
                                base_power=10,
                                min_games=10,
                                max_m_diff=1.0,
                                max_power=25):
    """
    Calcula Win Probability usando ranges de Diff_Power e M_Diff.
    Se não houver jogos suficientes, expande ranges até encontrar.
    Se ainda não houver → fallback usa odds implícitas.
    """
    m_h, m_a = row.get('M_H'), row.get('M_A')
    diff_m   = m_h - m_a if (m_h is not None and m_a is not None) else None
    diff_pow = row.get('Diff_Power')

    hist = history.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']

    # Inicializa ranges
    m_diff_margin = base_m_diff
    power_margin = base_power
    sample = pd.DataFrame()
    n = 0

    while n < min_games and (m_diff_margin <= max_m_diff and power_margin <= max_power):
        mask = (
            hist['M_Diff'].between(diff_m - m_diff_margin, diff_m + m_diff_margin) &
            hist['Diff_Power'].between(diff_pow - power_margin, diff_pow + power_margin)
        )
        sample = hist[mask]
        n = len(sample)

        if n < min_games:
            m_diff_margin += 0.20
            power_margin += 5

    if row.get('Auto_Recommendation') == '❌ Avoid':
        return n, None
    if n == 0:
        # fallback: usar odd implícita
        target = event_side_for_winprob(row['Auto_Recommendation'])
        if target == 'HOME' and row.get("Odd_H"):
            return 0, round(100 / row["Odd_H"], 1)
        if target == 'AWAY' and row.get("Odd_A"):
            return 0, round(100 / row["Odd_A"], 1)
        if target == 'DRAW' and row.get("Odd_D"):
            return 0, round(100 / row["Odd_D"], 1)
        return 0, None

    # ---- Calcula probabilidade real
    target = event_side_for_winprob(row['Auto_Recommendation'])
    if target == 'HOME':
        p = (sample['Goals_H_FT'] > sample['Goals_A_FT']).mean()
    elif target == 'AWAY':
        p = (sample['Goals_A_FT'] > sample['Goals_H_FT']).mean()
    elif target == 'DRAW':
        p = (sample['Goals_A_FT'] == sample['Goals_H_FT']).mean()
    elif target == '1X':
        p = (sample['Goals_H_FT'] >= sample['Goals_A_FT']).mean()
    elif target == 'X2':
        p = (sample['Goals_A_FT'] >= sample['Goals_H_FT']).mean()
    else:
        return n, None

    return n, (round(float(p)*100, 1) if p is not None else None)


########################################
####### Bloco 6 – Regras Híbridas ######
########################################
def auto_recommendation(row,
                        diff_mid_lo=0.20, diff_mid_hi=0.80,
                        diff_mid_hi_highvar=0.75, power_gate=1, power_gate_highvar=5):

    band_home = row.get('Home_Band')
    band_away = row.get('Away_Band')
    dominant  = row.get('Dominant')
    diff_m    = row.get('M_Diff')
    diff_pow  = row.get('Diff_Power')
    league_cls= row.get('League_Classification', 'Medium Variation')
    m_a       = row.get('M_A')
    m_h       = row.get('M_H')
    odd_d     = row.get('Odd_D')

    # 1) Strong edges -> Direct Back
    if band_home == 'Top 20%' and band_away == 'Bottom 20%':
        return '🟢 Back Home'
    if band_home == 'Bottom 20%' and band_away == 'Top 20%':
        return '🟠 Back Away'

    if dominant in ['Both extremes (Home↑ & Away↓)', 'Home strong'] and band_away != 'Top 20%':
        if diff_m is not None and diff_m >= 0.90:
            return '🟢 Back Home'
    if dominant in ['Both extremes (Away↑ & Home↓)', 'Away strong'] and band_home == 'Balanced':
        if diff_m is not None and diff_m <= -0.90:
            return '🟪 X2 (Away/Draw)'

    # 2) Both Balanced (with thresholds)
    if (band_home == 'Balanced') and (band_away == 'Balanced') and (diff_m is not None) and (diff_pow is not None):
        if league_cls == 'High Variation':
            if (diff_m >= 0.45 and diff_m < diff_mid_hi_highvar and diff_pow >= power_gate_highvar):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -0.45 and diff_m > -diff_mid_hi_highvar and diff_pow <= -power_gate_highvar):
                return '🟪 X2 (Away/Draw)'
        else:
            if (diff_m >= diff_mid_lo and diff_m < diff_mid_hi and diff_pow >= power_gate):
                return '🟦 1X (Home/Draw)'
            if (diff_m <= -diff_mid_lo and diff_m > -diff_mid_hi and diff_pow <= -power_gate):
                return '🟪 X2 (Away/Draw)'

    # 3) Balanced vs Bottom20%
    if (band_home == 'Balanced') and (band_away == 'Bottom 20%'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Balanced') and (band_home == 'Bottom 20%'):
        return '🟪 X2 (Away/Draw)'

    # 4) Top20% vs Balanced
    if (band_home == 'Top 20%') and (band_away == 'Balanced'):
        return '🟦 1X (Home/Draw)'
    if (band_away == 'Top 20%') and (band_home == 'Balanced'):
        return '🟪 X2 (Away/Draw)'

    # 5) Filtro Draw
    if (odd_d is not None and 2.5 <= odd_d <= 6.0) and (diff_pow is not None and -10 <= diff_pow <= 10):
        if (m_h is not None and 0 <= m_h <= 1) or (m_a is not None and 0 <= m_a <= 0.5):
            return '⚪ Back Draw'

    # 6) Fallback
    return '❌ Avoid'


def auto_recommendation_dynamic_winrate(row, history,
                                        min_games=5,
                                        min_winrate=45.0):
    candidates_main = ["🟢 Back Home", "🟠 Back Away", "⚪ Back Draw"]
    candidates_fallback = ["🟦 1X (Home/Draw)", "🟪 X2 (Away/Draw)"]

    best_rec, best_prob, best_ev, best_n = None, None, None, None

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

    if best_prob is None or best_prob < min_winrate:
        return "❌ Avoid", best_prob, best_ev, best_n

    return best_rec, best_prob, best_ev, best_n


def auto_recommendation_hybrid(row, history,
                               min_games=5,
                               min_winrate=45.0):
    rec = auto_recommendation(row)

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

BAND_MAP = {"Bottom 20%":1, "Balanced":2, "Top 20%":3}
if 'Home_Band' in X: X['Home_Band_Num'] = X['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X: X['Away_Band_Num'] = X['Away_Band'].map(BAND_MAP)

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
threshold = st.sidebar.slider(
    "ML Threshold for Direct Win (%)", 
    min_value=50, max_value=80, value=65, step=1
) / 100.0

def ml_recommendation_from_proba(p_home, p_draw, p_away, threshold=0.65):
    if p_home >= threshold:
        return "🟢 Back Home"
    elif p_away >= threshold:
        return "🟠 Back Away"
    else:
        sum_home_draw = p_home + p_draw
        sum_away_draw = p_away + p_draw
        if abs(p_home - p_away) < 0.05 and p_draw > 0.35:
            return "⚪ Back Draw"
        elif sum_home_draw > sum_away_draw:
            return "🟦 1X (Home/Draw)"
        elif sum_away_draw > sum_home_draw:
            return "🟪 X2 (Away/Draw)"
        else:
            return "❌ Avoid"

X_today = games_today[features_raw].copy()

if 'Home_Band' in X_today: 
    X_today['Home_Band_Num'] = X_today['Home_Band'].map(BAND_MAP)
if 'Away_Band' in X_today: 
    X_today['Away_Band_Num'] = X_today['Away_Band'].map(BAND_MAP)

if cat_cols:
    encoded_today = encoder.transform(X_today[cat_cols])
    encoded_today_df = pd.DataFrame(encoded_today, columns=encoder.get_feature_names_out(cat_cols))
    X_today = pd.concat([X_today.drop(columns=cat_cols).reset_index(drop=True),
                         encoded_today_df.reset_index(drop=True)], axis=1)

ml_preds = model.predict(X_today)
ml_proba = model.predict_proba(X_today)

games_today["ML_Proba_Home"] = ml_proba[:, list(model.classes_).index("Home")]
games_today["ML_Proba_Draw"] = ml_proba[:, list(model.classes_).index("Draw")]
games_today["ML_Proba_Away"] = ml_proba[:, list(model.classes_).index("Away")]

games_today["ML_Recommendation"] = [
    ml_recommendation_from_proba(row["ML_Proba_Home"], 
                                 row["ML_Proba_Draw"], 
                                 row["ML_Proba_Away"],
                                 threshold=threshold)
    for _, row in games_today.iterrows()
]


########################################
####### Bloco 9 – Exibição Final #######
########################################
cols_to_show = [
    'Date','Time','League','Home','Away',
    'Auto_Recommendation','Win_Probability',
    'ML_Recommendation',
    'ML_Proba_Home','ML_Proba_Draw','ML_Proba_Away'
]

available_cols = [c for c in cols_to_show if c in games_today.columns]

if "Auto_Recommendation" in games_today and "ML_Recommendation" in games_today:
    games_today["Agreement"] = np.where(
        games_today["Auto_Recommendation"] == games_today["ML_Recommendation"],
        "✅",
        "⚠️"
    )
    if "Agreement" not in available_cols:
        available_cols.insert(6, "Agreement")

st.subheader("📊 Regras vs ML")
st.dataframe(
    games_today[available_cols]
    .style.format({
        'Win_Probability':'{:.1f}%',
        'ML_Proba_Home':'{:.2f}',
        'ML_Proba_Draw':'{:.2f}',
        'ML_Proba_Away':'{:.2f}'
    }),
    use_container_width=True,
    height=1200  # tabela maior
)
