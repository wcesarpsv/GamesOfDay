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
####### Bloco 4B – LiveScore Merge #####
########################################
from datetime import datetime

# === Detectar data de hoje para nome do arquivo ===
today_str = datetime.today().strftime("%Y-%m-%d")
livescore_folder = "LiveScore"
livescore_file = os.path.join(livescore_folder, f"Resultados_RAW_{today_str}.csv")

# ===============================================
# 1) Garante que as colunas de gols existam
# ===============================================
if 'Goals_H_Today' not in games_today.columns:
    games_today['Goals_H_Today'] = np.nan
if 'Goals_A_Today' not in games_today.columns:
    games_today['Goals_A_Today'] = np.nan

# ===============================================
# 2) Carregar e integrar resultados do arquivo RAW
# ===============================================
if os.path.exists(livescore_file):
    st.info(f"Arquivo de resultados encontrado: {livescore_file}")
    
    # Carregar resultados
    results_df = pd.read_csv(livescore_file)

    # Conferir se as colunas essenciais existem
    required_cols = [
        'game_id', 'status', 'home_goal', 'away_goal',
        'home_ht_goal', 'away_ht_goal',
        'home_corners', 'away_corners',
        'home_yellow', 'away_yellow',
        'home_red', 'away_red'
    ]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if missing_cols:
        st.error(f"O arquivo {livescore_file} está faltando estas colunas: {missing_cols}")
    else:
        # === Merge seguro para evitar duplicatas ===
        games_today = games_today.merge(
            results_df,
            left_on='Id',
            right_on='game_id',
            how='left',
            suffixes=('', '_RAW')
        )

        # ===============================================
        # 3) Padronizar as colunas principais de gols
        # ===============================================
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']

        # ===============================================
        # 4) Garantir que só jogos finalizados tenham gols
        # ===============================================
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan

        # ===============================================
        # 5) Debug opcional – visualizar merge
        # ===============================================
        st.write("Amostra após merge LiveScore:",
                 games_today[['Id', 'status', 'Goals_H_Today', 'Goals_A_Today']].head(10))

else:
    st.warning(f"Nenhum arquivo de resultados encontrado em: {livescore_file}")



########################################
##### Bloco 4C – Avaliar Resultados ####
########################################

# 1) Determinar resultado real (apenas se já temos gols preenchidos)
def determine_result(row):
    if pd.isna(row['Goals_H_Today']) or pd.isna(row['Goals_A_Today']):
        return None
    if row['Goals_H_Today'] > row['Goals_A_Today']:
        return "Home"
    elif row['Goals_H_Today'] < row['Goals_A_Today']:
        return "Away"
    else:
        return "Draw"

games_today['Result_Today'] = games_today.apply(determine_result, axis=1)

# 2) Função para avaliar se a recomendação acertou
def check_recommendation(rec, result):
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return None  # Não considerar
    rec = str(rec)

    if 'Back Home' in rec:
        return result == "Home"
    elif 'Back Away' in rec:
        return result == "Away"
    elif 'Back Draw' in rec:
        return result == "Draw"
    elif '1X' in rec:
        return result in ["Home", "Draw"]
    elif 'X2' in rec:
        return result in ["Away", "Draw"]
    else:
        return None

# Avaliar acertos separadamente
games_today['Auto_Correct'] = games_today.apply(
    lambda r: check_recommendation(r['Auto_Recommendation'], r['Result_Today']), axis=1
)
games_today['ML_Correct'] = games_today.apply(
    lambda r: check_recommendation(r['ML_Recommendation'], r['Result_Today']), axis=1
)

# 3) Função para calcular profit de acordo com a odd usada
def calculate_profit(rec, result, odds_row):
    if pd.isna(rec) or result is None or rec == '❌ Avoid':
        return 0  # Não apostou, profit 0
    rec = str(rec)

    # Selecionar odd correta
    if 'Back Home' in rec:
        odd = odds_row.get('Odd_H', np.nan)
        return odd - 1 if result == "Home" else -1
    elif 'Back Away' in rec:
        odd = odds_row.get('Odd_A', np.nan)
        return odd - 1 if result == "Away" else -1
    elif 'Back Draw' in rec:
        odd = odds_row.get('Odd_D', np.nan)
        return odd - 1 if result == "Draw" else -1
    elif '1X' in rec:
        odd = odds_row.get('Odd_1X', np.nan)
        return odd - 1 if result in ["Home", "Draw"] else -1
    elif 'X2' in rec:
        odd = odds_row.get('Odd_X2', np.nan)
        return odd - 1 if result in ["Away", "Draw"] else -1
    else:
        return 0

# Calcular profit separadamente para Auto e ML
games_today['Profit_Auto'] = games_today.apply(
    lambda r: calculate_profit(r['Auto_Recommendation'], r['Result_Today'], r), axis=1
)
games_today['Profit_ML'] = games_today.apply(
    lambda r: calculate_profit(r['ML_Recommendation'], r['Result_Today'], r), axis=1
)

# 4) Resumo agregado
finished_games = games_today.dropna(subset=['Result_Today'])

def summary_stats(df, prefix):
    # Apenas apostas feitas (não considerar ❌ Avoid)
    bets = df[df[f'{prefix}_Correct'].notna()]
    total_bets = len(bets)
    correct_bets = bets[f'{prefix}_Correct'].sum()
    winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
    total_profit = bets[f'Profit_{prefix}'].sum()

    return {
        "Total Jogos": len(df),
        "Apostas Feitas": total_bets,
        "Acertos": correct_bets,
        "Winrate (%)": round(winrate, 2),
        "Profit Total": round(total_profit, 2)
    }

summary_auto = summary_stats(finished_games, "Auto")
summary_ml = summary_stats(finished_games, "ML")

# 5) Mostrar resumo no Streamlit
st.subheader("📈 Resumo do Dia")
st.markdown("### Performance Auto Recommendation (Regras)")
st.json(summary_auto)

st.markdown("### Performance Machine Learning (ML)")
st.json(summary_ml)



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
        target = event_side_for_winprob(row['Auto_Recommendation'])
        if target == 'HOME' and row.get("Odd_H"):
            return 0, round(100 / row["Odd_H"], 1)
        if target == 'AWAY' and row.get("Odd_A"):
            return 0, round(100 / row["Odd_A"], 1)
        if target == 'DRAW' and row.get("Odd_D"):
            return 0, round(100 / row["Odd_D"], 1)
        return 0, None

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
####### Bloco 5C – Bands & Dominant ####
########################################
def classify_leagues_variation(history_df):
    agg = (
        history_df.groupby('League')
        .agg(
            M_H_Min=('M_H','min'), M_H_Max=('M_H','max'),
            M_A_Min=('M_A','min'), M_A_Max=('M_A','max'),
            Hist_Games=('M_H','count')
        ).reset_index()
    )
    agg['Variation_Total'] = (agg['M_H_Max'] - agg['M_H_Min']) + (agg['M_A_Max'] - agg['M_A_Min'])
    def label(v):
        if v > 6.0: return "High Variation"
        if v >= 3.0: return "Medium Variation"
        return "Low Variation"
    agg['League_Classification'] = agg['Variation_Total'].apply(label)
    return agg[['League','League_Classification','Variation_Total','Hist_Games']]

def compute_league_bands(history_df):
    hist = history_df.copy()
    hist['M_Diff'] = hist['M_H'] - hist['M_A']
    diff_q = (
        hist.groupby('League')['M_Diff']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'P20_Diff', 0.8:'P80_Diff'})
            .reset_index()
    )
    home_q = (
        hist.groupby('League')['M_H']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Home_P20', 0.8:'Home_P80'})
            .reset_index()
    )
    away_q = (
        hist.groupby('League')['M_A']
            .quantile([0.20, 0.80]).unstack()
            .rename(columns={0.2:'Away_P20', 0.8:'Away_P80'})
            .reset_index()
    )
    out = diff_q.merge(home_q, on='League', how='inner').merge(away_q, on='League', how='inner')
    return out

def dominant_side(row, threshold=DOMINANT_THRESHOLD):
    m_h, m_a = row['M_H'], row['M_A']
    if (m_h >= threshold) and (m_a <= -threshold):
        return "Both extremes (Home↑ & Away↓)"
    if (m_a >= threshold) and (m_h <= -threshold):
        return "Both extremes (Away↑ & Home↓)"
    if m_h >= threshold:
        return "Home strong"
    if m_h <= -threshold:
        return "Home weak"
    if m_a >= threshold:
        return "Away strong"
    if m_a <= -threshold:
        return "Away weak"
    return "Mixed / Neutral"


# === aplicar nos jogos do dia ===
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

games_today = games_today.merge(league_class, on='League', how='left')
games_today = games_today.merge(league_bands, on='League', how='left')

games_today['Home_Band'] = np.where(
    games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
    np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
)
games_today['Away_Band'] = np.where(
    games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
    np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
)

games_today['Dominant'] = games_today.apply(dominant_side, axis=1)


########################################
####### Bloco 6 – Auto Recommendation ##
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

    # 5) Filtro Draw (novo)
    if (odd_d is not None and 2.5 <= odd_d <= 6.0) and (diff_pow is not None and -10 <= diff_pow <= 10):
        if (m_h is not None and 0 <= m_h <= 1) or (m_a is not None and 0 <= m_a <= 0.5):
            return '⚪ Back Draw'

    # 6) Fallback
    return '❌ Avoid'


# === Aplicar nos jogos do dia ===
games_today['Auto_Recommendation'] = games_today.apply(lambda r: auto_recommendation(r), axis=1)


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
    n_estimators=600,
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
##### Bloco 9 – Exibição com Cores #####
########################################

def highlight_row(row):
    """
    Define cor da linha baseada no status do jogo e acerto da aposta.
    - Verde: aposta correta
    - Vermelho: aposta errada
    - Cinza: jogo ainda não finalizado
    """
    # Jogo ainda sem resultado
    if pd.isna(row['Goals_H_Today']) or pd.isna(row['Goals_A_Today']):
        return ['background-color: #e2e3e5'] * len(row)  # Cinza claro

    # Avaliação para Auto_Recommendation
    if row['Auto_Correct'] is True:
        return ['background-color: #d4edda'] * len(row)  # Verde claro
    elif row['Auto_Correct'] is False:
        return ['background-color: #f8d7da'] * len(row)  # Vermelho claro

    return [''] * len(row)  # Sem destaque

# Colunas que queremos mostrar na tabela
cols_to_show = [
    'Date', 'Time', 'League', 'Home', 'Away',
    'Goals_H_Today', 'Goals_A_Today',
    'Auto_Recommendation', 'ML_Recommendation',
    'Auto_Correct', 'ML_Correct',
    'Profit_Auto', 'Profit_ML'
]

# Exibir tabela no Streamlit
st.subheader("📊 Jogos do Dia – Auto vs ML")
st.dataframe(
    games_today[cols_to_show]
    .style.apply(highlight_row, axis=1)
    .format({
        'Goals_H_Today': '{:.0f}',
        'Goals_A_Today': '{:.0f}',
        'Profit_Auto': '{:.2f}',
        'Profit_ML': '{:.2f}'
    }),
    use_container_width=True,
    height=800
)
