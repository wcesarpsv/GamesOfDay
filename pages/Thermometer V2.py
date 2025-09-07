import streamlit as st
import pandas as pd
import numpy as np
import os

# ================= Page Config =================
st.set_page_config(page_title="LeagueThermometer - Momentum and Value", layout="wide")
st.title("LeagueThermometer - Momentum and Value")

st.markdown(
    """
    What this app does:
    - Uses league stability (Low/Medium/High Variation) computed from historical M_H/M_A ranges.
    - Splits each league by P20 / P80 of Diff_M = M_H - M_A into Bottom 20%, Balanced P20-80, Top 20%.
    - Derives historical outcome rates by (League, Band) to estimate fair probabilities and fair odds.
    - Suggests minimum fair odds for handicaps from history only (no real odds required).
    """
)

# ================= Configs =================
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]
MIN_HIST_GAMES_PER_LEAGUE = 10

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
    return pd.concat(frames, ignore_index=True)

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
        st.error("Missing required columns: " + ", ".join(missing))
        return False
    return True


def classify_league_variation(history: pd.DataFrame) -> pd.DataFrame:
    agg = (
        history.groupby('League')
        .agg(
            M_H_Min=("M_H", "min"), M_H_Max=("M_H", "max"),
            M_A_Min=("M_A", "min"), M_A_Max=("M_A", "max"),
            Games=("M_H", "count")
        )
        .reset_index()
    )
    agg["Variation_Total"] = (agg["M_H_Max"] - agg["M_H_Min"]) + (agg["M_A_Max"] - agg["M_A_Min"])
    def _label(v: float) -> str:
        if v > 6.0:
            return "High Variation"
        if v >= 3.0:
            return "Medium Variation"
        return "Low Variation"
    agg['Classification'] = agg['Variation_Total'].apply(_label)
    return agg


def add_diff_and_bands(df: pd.DataFrame, by_league: bool = True, min_hist_games_per_league: int = MIN_HIST_GAMES_PER_LEAGUE) -> pd.DataFrame:
    out = df.copy()
    out['Diff_M'] = out['M_H'] - out['M_A']

    if by_league:
        bands = (
            out.groupby('League')['Diff_M']
              .quantile([0.20, 0.80])
              .unstack()
              .rename(columns={0.2: 'P20_Diff', 0.8: 'P80_Diff'})
              .reset_index()
        )
        counts = out.groupby('League')['Diff_M'].size().rename('Hist_Games').reset_index()
        bands = bands.merge(counts, on='League', how='left')
        out = out.merge(bands, on='League', how='left')

        # Global fallback thresholds
        p20_global = out['Diff_M'].quantile(0.20)
        p80_global = out['Diff_M'].quantile(0.80)

        # Invalidate per-league thresholds when insufficient or inverted
        bad_mask = (out['Hist_Games'].fillna(0) < min_hist_games_per_league) | (
            out['P20_Diff'].notna() & out['P80_Diff'].notna() & (out['P20_Diff'] >= out['P80_Diff'])
        )
        out.loc[bad_mask, ['P20_Diff', 'P80_Diff']] = (np.nan, np.nan)

        # Fill with global thresholds
        out['P20_Diff'] = out['P20_Diff'].fillna(p20_global)
        out['P80_Diff'] = out['P80_Diff'].fillna(p80_global)

        # Vectorized banding
        out['Band'] = np.where(
            out['Diff_M'] <= out['P20_Diff'], 'Bottom 20%',
            np.where(out['Diff_M'] >= out['P80_Diff'], 'Top 20%', 'Balanced P20-80')
        )
    else:
        p20 = out['Diff_M'].quantile(0.20)
        p80 = out['Diff_M'].quantile(0.80)
        out['P20_Diff'] = p20
        out['P80_Diff'] = p80
        out['Band'] = np.where(
            out['Diff_M'] <= p20, 'Bottom 20%',
            np.where(out['Diff_M'] >= p80, 'Top 20%', 'Balanced P20-80')
        )

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


def fair_odds(p: float) -> float:
    if p is None or pd.isna(p) or p == 0:
        return np.inf
    return 1.0 / p


def away_plus1_be_odd(p_away: float, p_draw: float, p_home_by2: float) -> float:
    # Away +1: win = Away or Draw; push = Home by 1; loss = Home by 2+
    if any(pd.isna(x) for x in [p_away, p_draw, p_home_by2]):
        return np.nan
    p_win = p_away + p_draw
    if p_win <= 0:
        return np.nan
    return 1.0 + (p_home_by2 / p_win)


def home_minus1_be_odd(p_home_by2: float, p_draw: float, p_away: float) -> float:
    # Home -1: win = Home by 2+; push = Home by 1; loss = Draw or Away
    if any(pd.isna(x) for x in [p_home_by2, p_draw, p_away]):
        return np.nan
    p_win = p_home_by2
    p_loss = p_draw + p_away
    if p_win <= 0:
        return np.nan
    return 1.0 + (p_loss / p_win)

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

# Keep only upcoming games if FT columns exist
if 'Goals_H_FT' in latest.columns:
    latest = latest[latest['Goals_H_FT'].isna()].copy()

# ================= Compute League Variation and Bands (history) =================
variation = classify_league_variation(history)

history = add_diff_and_bands(history, by_league=True, min_hist_games_per_league=MIN_HIST_GAMES_PER_LEAGUE)
latest = latest.copy()
latest['Diff_M'] = latest['M_H'] - latest['M_A']

# Attach league P20/P80 from history with global fallback
league_bands = history[['League','P20_Diff','P80_Diff']].drop_duplicates()
latest = latest.merge(league_bands, on='League', how='left')

_hist_with_diff = history.copy()
_hist_with_diff['Diff_M'] = _hist_with_diff['M_H'] - _hist_with_diff['M_A']
p20_global = _hist_with_diff['Diff_M'].quantile(0.20)
p80_global = _hist_with_diff['Diff_M'].quantile(0.80)

bad_mask = latest['P20_Diff'].isna() | latest['P80_Diff'].isna() | (latest['P20_Diff'] >= latest['P80_Diff'])
latest.loc[bad_mask, 'P20_Diff'] = p20_global
latest.loc[bad_mask, 'P80_Diff'] = p80_global

latest['Band'] = np.where(
    latest['Diff_M'] <= latest['P20_Diff'], 'Bottom 20%',
    np.where(latest['Diff_M'] >= latest['P80_Diff'], 'Top 20%', 'Balanced P20-80')
)

# ================= Historical rates per (League, Band) =================
hist_rates = history.groupby(['League','Band']).apply(outcomes).reset_index()

# Merge variation labels
hist_rates = hist_rates.merge(variation[['League','Classification','Variation_Total','Games']], on='League', how='left')

# ================= Attach historical probabilities to today's games =================
show_cols = ['Date','Time','League','Home','Away','M_H','M_A','Diff_Power','Diff_M','Band']
preview = latest[show_cols].copy()
preview = preview.merge(hist_rates, on=['League','Band'], how='left')

# Compute fair odds (1X2) from history only
for col_p in ['p_Home','p_Draw','p_Away']:
    preview[f'Fair_{col_p[2:]}'] = preview[col_p].apply(fair_odds)

# Composite market: X2 (Away or Draw)
preview['p_X2'] = preview['p_Away'] + preview['p_Draw']
preview['Fair_X2'] = preview['p_X2'].apply(fair_odds)

# Handicap fair odds (minimums) from historical probabilities
preview['BE_AwayPlus1'] = preview.apply(lambda r: away_plus1_be_odd(r.get('p_Away'), r.get('p_Draw'), r.get('p_Home_by2+')), axis=1)
preview['BE_HomeMinus1'] = preview.apply(lambda r: home_minus1_be_odd(r.get('p_Home_by2+'), r.get('p_Draw'), r.get('p_Away')), axis=1)
preview['Fair_HomeMinus1_5'] = preview['p_Home_by2+'].apply(fair_odds)
preview['Fair_AwayPlus1_5'] = (1 - preview['p_Home_by2+']).apply(fair_odds)

# Order rows
preview = preview.sort_values(by=['Band','p_X2','p_Home_by2+'], ascending=[True, False, False])

# ================= Display =================
st.subheader("Today's Card - Historical fair odds and handicap minimums")

disp_cols = [
    'Date','Time','League','Home','Away','Classification','Band',
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

# ================= Signals =================
st.subheader("Signals")
for _, row in preview.iterrows():
    league = row['League']
    match_name = f"{row['Home']} vs {row['Away']}"
    cls = row.get('Classification', 'N/A')
    band = row.get('Band', 'N/A')
    msg = f"{match_name} — {league} | League: {cls}, Band: {band}"
    tips = []
    if pd.notna(row.get('Fair_X2')) and np.isfinite(row['Fair_X2']):
        tips.append(f"Min fair X2 >= {row['Fair_X2']:.2f}")
    if pd.notna(row.get('BE_AwayPlus1')) and np.isfinite(row['BE_AwayPlus1']):
        tips.append(f"Min fair Away +1 >= {row['BE_AwayPlus1']:.2f}")
    if pd.notna(row.get('BE_HomeMinus1')) and np.isfinite(row['BE_HomeMinus1']):
        tips.append(f"Min fair Home -1 >= {row['BE_HomeMinus1']:.2f}")
    if pd.notna(row.get('Fair_HomeMinus1_5')) and np.isfinite(row['Fair_HomeMinus1_5']):
        tips.append(f"Min fair Home -1.5 >= {row['Fair_HomeMinus1_5']:.2f}")
    if pd.notna(row.get('Fair_AwayPlus1_5')) and np.isfinite(row['Fair_AwayPlus1_5']):
        tips.append(f"Min fair Away +1.5 >= {row['Fair_AwayPlus1_5']:.2f}")
    if not tips:
        tips.append("No obvious value by historical metrics.")
    st.markdown(" - " + msg + "

    -> " + " | ".join(tips))
