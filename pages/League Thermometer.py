import streamlit as st
import pandas as pd
import numpy as np
import os

# ================= Page Config =================
st.set_page_config(page_title="LeagueThermometer – Momentum & Value", layout="wide")
st.title("🔥 LeagueThermometer – Momentum & Value")

st.markdown(
    """
    **What this does**
    - Uses **league stability** (Baixa/Média/Alta variação) computed from historical M_H/M_A ranges.
    - Splits each league by **P20 / P80** of `Diff_M = M_H - M_A` to flag **Top 20%**, **Equilibrado (P20–P80)**, **Bottom 20%**.
    - Derives **historical outcome rates** for the relevant league + band to estimate fair probabilities & fair odds.
    - Checks **value (EV)** against posted odds when available (1X2 always; X2 or Away +1 if you add columns).

    **Optional odds columns** (if present in *today's* CSV):
    - `Odd_X2` → (Away or Draw)
    - `Odd_AwayPlus1` → Handicap +1 for Away (Asian)
    """
)

# ================= Configs =================
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

# ================= Helpers =================
@st.cache_data(show_spinner=False)
def load_csvs(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(os.path.join(folder, f)))
        except Exception as e:
            st.error(f"Error loading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df

@st.cache_data(show_spinner=False)
def load_latest_csv(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    latest = max(files)
    return pd.read_csv(os.path.join(folder, latest))


def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()


def require_cols(df: pd.DataFrame, cols: list) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return False
    return True


def classify_league_variation(history: pd.DataFrame) -> pd.DataFrame:
    # Compute league variation from historical M_H/M_A ranges
    agg = (
        history.groupby('League')
        .agg(M_H_Min=("M_H", "min"), M_H_Max=("M_H", "max"),
             M_A_Min=("M_A", "min"), M_A_Max=("M_A", "max"),
             Jogos=("M_H", "count"))
        .reset_index()
    )
    agg["Variation_Total"] = (agg["M_H_Max"] - agg["M_H_Min"]) + (agg["M_A_Max"] - agg["M_A_Min"]) 
    def _label(v):
        if v > 6.0:
            return "Alta Variação"
        elif v >= 3.0:
            return "Média Variação"
        return "Baixa Variação"
    agg['Classificação'] = agg['Variation_Total'].apply(_label)
    return agg


def add_diff_and_bands(df: pd.DataFrame, by_league: bool = True) -> pd.DataFrame:
    out = df.copy()
    out['Diff_M'] = out['M_H'] - out['M_A']

    if by_league:
        # Quantis por liga
        bands = (
            out.groupby('League')['Diff_M']
              .quantile([0.20, 0.80])
              .unstack()
              .rename(columns={0.2: 'P20_Diff', 0.8: 'P80_Diff'})
              .reset_index()
        )
        # (opcional) garantir amostra mínima por liga para quantis estáveis
        counts = out.groupby('League')['Diff_M'].size().rename('Hist_Jogos').reset_index()
        bands = bands.merge(counts, on='League', how='left')

        out = out.merge(bands, on='League', how='left')

        # Se algum caso tiver P20_Diff >= P80_Diff (amostra pequena), marque como NaN
        bad = out['P20_Diff'] >= out['P80_Diff']
        out.loc[bad, ['P20_Diff', 'P80_Diff']] = (np.nan, np.nan)

        # Classificação da banda (vectorizado, sem pd.cut)
        conds = [
            out['P20_Diff'].notna() & (out['Diff_M'] <= out['P20_Diff']),
            out['P80_Diff'].notna() & (out['Diff_M'] >= out['P80_Diff']),
        ]
        choices = ['Bottom 20%', 'Top 20%']
        out['Band'] = np.select(conds, choices, default='Equilibrado P80')
    else:
        # Quantis globais
        p20, p80 = out['Diff_M'].quantile(0.20), out['Diff_M'].quantile(0.80)
        out['P20_Diff'], out['P80_Diff'] = p20, p80
        out['Band'] = np.where(out['Diff_M'] <= p20, 'Bottom 20%',
                        np.where(out['Diff_M'] >= p80, 'Top 20%', 'Equilibrado P80'))

    return out



def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['Margin'] = d['Goals_H_FT'] - d['Goals_A_FT']
    return pd.Series({
        'p_Home': (d['Margin'] > 0).mean(),
        'p_Draw': (d['Margin'] == 0).mean(),
        'p_Away': (d['Margin'] < 0).mean(),
        'p_Home_by1': (d['Margin'] == 1).mean(),
        'p_Home_by2+': (d['Margin'] >= 2).mean(),
    })


def fair_odds(p):
    return np.inf if p == 0 else 1.0 / p


def ev_fraction(p, odd):
    # EV per 1 unit stake on binary bet
    # payoff = odd - 1 when win, -1 when loss
    return p * (odd - 1) - (1 - p)

# For Away +1 (Asian): win if Away or Draw, push if Home by 1, lose if Home by 2+
# EV uses only win/loss parts; pushes are neutral. Need p_win = p_away + p_draw, p_loss = p_home_by2+
# break-even odd = 1 + p_loss / p_win

def away_plus1_be_odd(p_away, p_draw, p_home_by2):
    p_win = p_away + p_draw
    if p_win <= 0:
        return np.nan
    return 1.0 + (p_home_by2 / p_win)

# ================= Load Data =================
all_games = load_csvs(GAMES_FOLDER)
all_games = filter_leagues(all_games)

required_hist = ['League','Home','Away','Goals_H_FT','Goals_A_FT','M_H','M_A','Diff_Power']
if not require_cols(all_games, required_hist):
    st.stop()

history = all_games.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
latest = load_latest_csv(GAMES_FOLDER)
latest = filter_leagues(latest)
if latest.empty:
    st.warning("No 'today' CSV found in GamesDay.")
    st.stop()

# If today's CSV has final scores, keep only upcoming (NaN FT)
if 'Goals_H_FT' in latest.columns:
    latest = latest[latest['Goals_H_FT'].isna()].copy()

# ================= Compute League Variation & Bands (from history) =================
variation = classify_league_variation(history)

history = add_diff_and_bands(history, by_league=True)
latest = latest.copy()
latest['Diff_M'] = latest['M_H'] - latest['M_A']
# attach league P20/P80 from history
league_bands = history[['League','P20_Diff','P80_Diff']].drop_duplicates()
latest = latest.merge(league_bands, on='League', how='left')

# Se alguma liga tiver P20 >= P80 (amostra ruim), invalida os limiares
bad = latest['P20_Diff'] >= latest['P80_Diff']
latest.loc[bad, ['P20_Diff', 'P80_Diff']] = (np.nan, np.nan)

latest['Band'] = np.where(
    latest['P20_Diff'].notna() & (latest['Diff_M'] <= latest['P20_Diff']), 'Bottom 20%',
    np.where(
        latest['P80_Diff'].notna() & (latest['Diff_M'] >= latest['P80_Diff']), 'Top 20%',
        'Equilibrado P80'
    )
)


# ================= Historical rates per league+band =================
# Build a table of outcome rates per (League, Band)
hist_rates = (
    history.groupby(['League','Band'])
    .apply(outcomes)
    .reset_index()
)

# Merge variation labels
hist_rates = hist_rates.merge(variation[['League','Classificação','Variation_Total','Jogos']], on='League', how='left')

# ================= Attach historical probabilities to today's games =================
show_cols = ['Date','Time','League','Home','Away','Odd_H','Odd_D','Odd_A','M_H','M_A','Diff_Power','Diff_M','Band']

preview = latest[show_cols].copy()
preview = preview.merge(hist_rates, on=['League','Band'], how='left')

# Compute fair odds (1X2) purely from history
for col_p in ['p_Home','p_Draw','p_Away']:
    preview[f'Fair_{col_p[2:]}'] = preview[col_p].apply(fair_odds)

# Composite markets from history only (no real odds used)
# X2 (Away or Draw)
preview['p_X2'] = preview['p_Away'] + preview['p_Draw']
preview['Fair_X2'] = preview['p_X2'].apply(fair_odds)

# ================= Handicap fair odds (mínimas) a partir da base histórica =================
# Away +1 (Asian): win = Away or Draw; push = Home by1; loss = Home by2+
preview['BE_AwayPlus1'] = preview.apply(
    lambda r: away_plus1_be_odd(r.get('p_Away',np.nan), r.get('p_Draw',np.nan), r.get('p_Home_by2+',np.nan)), axis=1
)

# Home -1 (Asian): win = Home by2+; push = Home by1; loss = Draw or Away
# Break-even odd = 1 + (p_loss / p_win) = 1 + ((p_Draw + p_Away) / p_Home_by2+)
def home_minus1_be_odd(p_home_by2, p_draw, p_away):
    p_win = p_home_by2
    p_loss = p_draw + p_away
    if p_win is None or np.isnan(p_win) or p_win <= 0:
        return np.nan
    return 1.0 + (p_loss / p_win)

preview['BE_HomeMinus1'] = preview.apply(
    lambda r: home_minus1_be_odd(r.get('p_Home_by2+',np.nan), r.get('p_Draw',np.nan), r.get('p_Away',np.nan)), axis=1
)

# Opcional: Home -1.5 e Away +1.5 (assumem vitórias/lições sem push)
# Para -1.5: win = Home by2+; loss = Draw ou Away. Fair = 1 / p_Home_by2+
# Para +1.5: win = Away/Draw/Home by1; loss = Home by2+. Fair = 1 / (1 - p_Home_by2+)
preview['Fair_HomeMinus1_5'] = preview['p_Home_by2+'].apply(fair_odds)
preview['Fair_AwayPlus1_5'] = (1 - preview['p_Home_by2+']).apply(fair_odds)

# ================= Ordering =================
# Priorize jogos com maior p_X2 (segurança) e maior p_Home_by2+ (força do mandante) conforme a banda
preview = preview.sort_values(by=['Band','p_X2','p_Home_by2+'], ascending=[True, False, False])

# ================= Display =================
st.subheader("Today's Card – Histórico → Fair odds & Handicaps mínimos")

disp_cols = [
    'Date','Time','League','Home','Away','Classificação','Band',
    'M_H','M_A','Diff_Power','Diff_M',
    'p_Home','p_Draw','p_Away','p_Home_by1','p_Home_by2+',
    'Fair_Home','Fair_Draw','Fair_Away','p_X2','Fair_X2',
    'BE_AwayPlus1','BE_HomeMinus1','Fair_AwayPlus1_5','Fair_HomeMinus1_5'
]

styled = (
    preview[disp_cols]
    .style
    .format({
        'M_H':'{:.3f}','M_A':'{:.3f}','Diff_Power':'{:.1f}','Diff_M':'{:.3f}',
        'p_Home':'{:.2%}','p_Draw':'{:.2%}','p_Away':'{:.2%}',
        'p_Home_by1':'{:.2%}','p_Home_by2+':'{:.2%}',
        'Fair_Home':'{:.2f}','Fair_Draw':'{:.2f}','Fair_Away':'{:.2f}',
        'p_X2':'{:.2%}','Fair_X2':'{:.2f}',
        'BE_AwayPlus1':'{:.2f}','BE_HomeMinus1':'{:.2f}',
        'Fair_AwayPlus1_5':'{:.2f}','Fair_HomeMinus1_5':'{:.2f}'
    })
)

st.dataframe(styled, use_container_width=True)


# ================= Simple Recommendation badges =================
st.subheader("Signals")
for _, row in preview.iterrows():
    league = row['League']
    match_name = f"{row['Home']} vs {row['Away']}"
    cls = row.get('Classificação', 'N/A')
    band = row.get('Band', 'N/A')
    msg = f"**{match_name}** — *{league}* | Liga: **{cls}**, Faixa: **{band}**"
    tips = []
    # Value checks
    if 'EV_X2' in row and pd.notna(row['EV_X2']) and row['EV_X2'] > 0:
        tips.append(f"X2 EV+: {row['EV_X2']:+.1%} (odds {row.get('Odd_X2','?')})")
    if 'Odd_AwayPlus1' in row and pd.notna(row['Odd_AwayPlus1']) and pd.notna(row['BE_AwayPlus1']):
        if row['Odd_AwayPlus1'] > row['BE_AwayPlus1']:
            tips.append(f"Away +1 EV+: odd {row['Odd_AwayPlus1']:.2f} > BE {row['BE_AwayPlus1']:.2f}")
    if not tips:
        tips.append("Sem valor óbvio pelas métricas históricas.")
    st.markdown(" • "+msg+"\n\n    → "+" | ".join(tips))
