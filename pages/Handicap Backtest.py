import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest – Asian Handicap", layout="wide")
st.title("📈 Strategy Backtest – Asian Handicap")

GAMES_FOLDER = "GamesDay/GamesAsian"

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
# ➜ Suas colunas Odd_H_Asi / Odd_A_Asi já estão em formato LÍQUIDO.
ODDS_ARE_NET = True  # se algum dia vierem brutas, mude para False

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _base_parse_asian_line(text: str):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    s = str(text).strip().lower().replace(' ', '')
    s = s.replace(',', '.')
    if s in ('pk', 'p.k.', 'level'):
        return [0.0]
    s_abs = re.sub(r'^[+-]', '', s)
    if '/' in s_abs:
        try:
            a, b = s_abs.split('/')
            return [float(a), float(b)]
        except Exception:
            return []
    try:
        x = float(s_abs)
    except Exception:
        return []
    # guard contra NaN (ex.: "nan" vira float('nan'))
    if pd.isna(x):
        return []
    frac = abs(x) - int(abs(x))
    base = int(abs(x))
    if abs(frac - 0.25) < 1e-9:
        return [base + 0.0, base + 0.5]
    if abs(frac - 0.75) < 1e-9:
        return [base + 0.5, base + 1.0]
    return [abs(x)]

def parse_asian_line(text: str):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    raw = str(text).strip().replace(' ', '')
    parts = _base_parse_asian_line(raw)
    if not parts:
        return []
    if raw.startswith('+'):
        sign = 1
    elif raw.startswith('-'):
        sign = -1
    else:
        sign = 1
    return [sign * p for p in parts]

def to_net_odds(x):
    """Retorna a odd líquida por stake=1.
       Se ODDS_ARE_NET=True, retorna o próprio valor (já líquido).
       Caso contrário, retorna odds_decimais - 1.0
    """
    if pd.isna(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if ODDS_ARE_NET else (v - 1.0)

def settle_ah_with_odds(goals_h, goals_a, ah_components_home, bet_on: str, net_odds: float) -> float:
    if (
        len(ah_components_home) == 0
        or pd.isna(goals_h) or pd.isna(goals_a)
        or net_odds is None
    ):
        return 0.0

    profits = []
    score_diff = goals_h - goals_a  # H - A
    for h_home in ah_components_home:
        if bet_on == "Home":
            margin = score_diff + h_home
        else:
            h_away = -h_home
            margin = (goals_a - goals_h) + h_away  # = -score_diff - h_home

        if margin > 0:
            profits.append(net_odds)   # full win
        elif abs(margin) < 1e-9:
            profits.append(0.0)        # push
        else:
            profits.append(-1.0)       # full loss

    return sum(profits) / len(profits)

# 🔧 número + slider + seletor de fonte
def range_filter_hibrido(label: str, data_min: float, data_max: float, step: float, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input("Min", value=float(data_min), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_min")
    max_val = c2.number_input("Max", value=float(data_max), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_max")

    # enforce order
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        st.session_state[f"{key_prefix}_min"] = min_val
        st.session_state[f"{key_prefix}_max"] = max_val

    slider_val = st.sidebar.slider("Drag to adjust",
                                   min_value=float(data_min),
                                   max_value=float(data_max),
                                   value=(float(min_val), float(max_val)),
                                   step=step,
                                   key=f"{key_prefix}_slider")

    source = st.sidebar.radio("Filter source", ["Slider", "Manual"], horizontal=True, key=f"{key_prefix}_src")
    st.sidebar.divider()

    if source == "Slider":
        return slider_val[0], slider_val[1]
    else:
        return float(min_val), float(max_val)

# ──────────────────────────────────────────────────────────────────────────────
# Load CSVs (tolerant)
# ──────────────────────────────────────────────────────────────────────────────
all_dfs = []
if not os.path.isdir(GAMES_FOLDER):
    st.error("❌ Folder 'GamesAsian' not found.")
    st.stop()

for file in sorted(os.listdir(GAMES_FOLDER)):
    if not file.lower().endswith(".csv"):
        continue

    df_path = os.path.join(GAMES_FOLDER, file)

    # 1) autodetect separator + encoding
    try:
        df = pd.read_csv(df_path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(df_path, sep="\t", encoding="utf-8-sig")
        except Exception as e:
            st.warning(f"⚠️ Failed to read {file}: {e}")
            continue

    # 2) normalize headers and some key text cols
    df.columns = df.columns.str.strip()
    for c in ["Asian_Line", "League", "Home", "Away"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    required = {
        "Goals_H_FT","Goals_A_FT","Asian_Line","Date",
        "Diff_Power","Diff_HT_P","League","Home","Away",
        "Odd_H_Asi","Odd_A_Asi"
    }
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        st.info(f"ℹ️ Skipping {file}: missing columns -> {missing}")
        continue

    # 3) coerce numeric (avoid NaN from strings)
    for c in ["Odd_H_Asi","Odd_A_Asi","Goals_H_FT","Goals_A_FT","Diff_Power","Diff_HT_P","Asian_Line"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) keep only rows with goals + odds present
    df = df.dropna(subset=["Goals_H_FT","Goals_A_FT","Odd_H_Asi","Odd_A_Asi"])
    if df.empty:
        st.info(f"ℹ️ Skipping {file}: no rows with goals + odds.")
        continue

    # 5) parse date (not blocking if NaT)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # 6) (NOVO) remover NaN explícito em Asian_Line antes do parser
    df = df.dropna(subset=["Asian_Line"])

    # 7) AH (Home reference) and clean average — com guards para NaN
    df["AH_components"] = df["Asian_Line"].apply(parse_asian_line)
    df = df[df["AH_components"].map(len) > 0].copy()
    if df.empty:
        st.info(f"ℹ️ Skipping {file}: invalid Asian_Line in all rows.")
        continue

    df["AH_clean"] = df["AH_components"].apply(lambda lst: sum(lst)/len(lst))

    # 8) net odds (identity if already net)
    df["Odd_H_liq"] = df["Odd_H_Asi"].apply(to_net_odds)
    df["Odd_A_liq"] = df["Odd_A_Asi"].apply(to_net_odds)

    # IMPORTANT: DO NOT filter by > 0 here, because 0.0 is valid for net odds = 1.0
    df = df.dropna(subset=["Odd_H_liq","Odd_A_liq"])
    if df.empty:
        st.info(f"ℹ️ Skipping {file}: all rows have invalid net odds.")
        continue

    all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data (with goals, Asian_Line and odds) found in GamesAsian.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

# bet side FIRST (Option 2)
bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# Diff_Power / Diff_HT_P
dp_min, dp_max = df_all["Diff_Power"].min(), df_all["Diff_Power"].max()
diff_power_sel = range_filter_hibrido("📊 Diff_Power", dp_min, dp_max, step=0.01, key_prefix="diff_power")

htp_min, htp_max = df_all["Diff_HT_P"].min(), df_all["Diff_HT_P"].max()
diff_ht_p_sel = range_filter_hibrido("📉 Diff_HT_P", htp_min, htp_max, step=0.01, key_prefix="diff_htp")

# Map AH to the bet side:
# Home → AH_clean_for_side = AH_clean; Away → -AH_clean
df_all["AH_clean_for_side"] = df_all["AH_clean"] if bet_on == "Home" else -df_all["AH_clean"]

ah_side_min = float(df_all["AH_clean_for_side"].min())
ah_side_max = float(df_all["AH_clean_for_side"].max())
ah_range_sel = range_filter_hibrido("⚖️ Asian Handicap (side line, AH_for_side)", ah_side_min, ah_side_max, step=0.25, key_prefix="ah_for_side")

# Apply filters
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power_sel[0]) & (df_all["Diff_Power"] <= diff_power_sel[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_all["Diff_HT_P"] <= diff_ht_p_sel[1]) &
    (df_all["AH_clean_for_side"] >= ah_range_sel[0]) & (df_all["AH_clean_for_side"] <= ah_range_sel[1])
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Profit calc (stake = 1)
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row):
    net_odds = row["Odd_H_liq"] if bet_on == "Home" else row["Odd_A_liq"]
    return settle_ah_with_odds(
        row["Goals_H_FT"], row["Goals_A_FT"], row["AH_components"], bet_on, net_odds
    )

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(filtered_df)), filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit by Bet (Asian Handicap, Stake=1)")
    st.pyplot(fig)

    n_matches = len(filtered_df)
    wins   = (filtered_df["Bet Result"] > 0).sum()
    pushes = (filtered_df["Bet Result"] == 0).sum()
    losses = (filtered_df["Bet Result"] < 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    mean_ah_home = filtered_df["AH_clean"].mean()              # Home reference
    mean_ah_side = filtered_df["AH_clean_for_side"].mean()     # Bet side reference
    mean_odd_liq = (filtered_df["Odd_H_liq"] if bet_on == "Home" else filtered_df["Odd_A_liq"]).mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader("📊 Backtest Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean AH (side)", f"{mean_ah_side:+.2f}")
    col4.metric("ROI (per bet)", f"{roi:.1%}")
    st.caption(f"Pushes: {pushes} · Losses: {losses} · Mean net-odds ({bet_on}): {mean_odd_liq:.2f} · Mean AH (home ref): {mean_ah_home:+.2f}")

    show_cols = [
        "Date", "League", "Home", "Away",
        "Asian_Line", "AH_clean", "AH_clean_for_side",
        "Diff_Power", "Diff_HT_P",
        "Odd_H_Asi", "Odd_A_Asi", "Odd_H_liq", "Odd_A_liq",
        "Goals_H_FT", "Goals_A_FT",
        "Bet Result", "Cumulative Profit"
    ]
    st.subheader("📝 Filtered Matches")
    st.dataframe(filtered_df[show_cols], use_container_width=True)

else:
    st.warning("⚠️ No matches found with selected filters.")
