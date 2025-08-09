import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest – Asian Handicap (Com Odds)", layout="wide")
st.title("📈 Strategy Backtest – Asian Handicap (Com Odds)")

GAMES_FOLDER = "GamesDay/GamesAsian"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _base_parse_asian_line(text: str):
    if text is None:
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
    frac = abs(x) - int(abs(x))
    base = int(abs(x))
    if abs(frac - 0.25) < 1e-9:
        return [base + 0.0, base + 0.5]
    if abs(frac - 0.75) < 1e-9:
        return [base + 0.5, base + 1.0]
    return [abs(x)]

def parse_asian_line(text: str):
    if text is None:
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
    if pd.isna(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v - 1.0 if v >= 1.5 else v

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
            profits.append(net_odds)   # vitória completa
        elif abs(margin) < 1e-9:
            profits.append(0.0)        # push
        else:
            profits.append(-1.0)       # derrota completa

    return sum(profits) / len(profits)

# 🔧 componente híbrido: número + slider + seletor de fonte
def range_filter_hibrido(label: str, data_min: float, data_max: float, step: float, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input("Min", value=float(data_min), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_min")
    max_val = c2.number_input("Max", value=float(data_max), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_max")

    # garante ordem
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        st.session_state[f"{key_prefix}_min"] = min_val
        st.session_state[f"{key_prefix}_max"] = max_val

    slider_val = st.sidebar.slider("Arraste para ajustar",
                                   min_value=float(data_min),
                                   max_value=float(data_max),
                                   value=(float(min_val), float(max_val)),
                                   step=step,
                                   key=f"{key_prefix}_slider")

    fonte = st.sidebar.radio("Fonte do filtro", ["Slider", "Manual"], horizontal=True, key=f"{key_prefix}_src")
    st.sidebar.divider()

    if fonte == "Slider":
        return slider_val[0], slider_val[1]
    else:
        return float(min_val), float(max_val)

# ⬇️ Carrega todos os CSVs válidos
all_dfs = []
if not os.path.isdir(GAMES_FOLDER):
    st.error("❌ Folder 'GamesAsian' not found.")
    st.stop()

for file in sorted(os.listdir(GAMES_FOLDER)):
    if file.endswith(".csv"):
        df_path = os.path.join(GAMES_FOLDER, file)
        try:
            df = pd.read_csv(df_path)
        except Exception:
            continue

        required = {
            "Goals_H_FT","Goals_A_FT","Asian_Line","Date",
            "Diff_Power","Diff_HT_P","League","Home","Away",
            "Odd_H_Asi","Odd_A_Asi"
        }
        if not required.issubset(df.columns):
            continue

        df = df.dropna(subset=["Goals_H_FT","Goals_A_FT"])
        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

        df["AH_components"] = df["Asian_Line"].apply(parse_asian_line)
        df = df[df["AH_components"].map(len) > 0].copy()
        df["AH_clean"] = df["AH_components"].apply(lambda lst: sum(lst)/len(lst))

        df["Odd_H_liq"] = df["Odd_H_Asi"].apply(to_net_odds)
        df["Odd_A_liq"] = df["Odd_A_Asi"].apply(to_net_odds)

        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data (with goals, Asian_Line and odds) found in GamesAsian.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# 🎚️ Filtros (híbridos)
st.sidebar.header("🎯 Filter Matches")

dp_min, dp_max = df_all["Diff_Power"].min(), df_all["Diff_Power"].max()
diff_power_sel = range_filter_hibrido("📊 Diff_Power", dp_min, dp_max, step=0.01, key_prefix="diff_power")

htp_min, htp_max = df_all["Diff_HT_P"].min(), df_all["Diff_HT_P"].max()
diff_ht_p_sel = range_filter_hibrido("📉 Diff_HT_P", htp_min, htp_max, step=0.01, key_prefix="diff_htp")

ah_min, ah_max = float(df_all["AH_clean"].min()), float(df_all["AH_clean"].max())
ah_range_sel = range_filter_hibrido("⚖️ Asian Handicap (Home line, AH_clean)", ah_min, ah_max, step=0.25, key_prefix="ah_clean")

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 🧮 Aplica filtros
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power_sel[0]) & (df_all["Diff_Power"] <= diff_power_sel[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_all["Diff_HT_P"] <= diff_ht_p_sel[1]) &
    (df_all["AH_clean"] >= ah_range_sel[0]) & (df_all["AH_clean"] <= ah_range_sel[1])
].copy()

# 🧠 Cálculo do lucro com odds (stake=1)
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
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit by Bet (Asian Handicap, Stake=1)")
    st.pyplot(fig)

    n_matches = len(filtered_df)
    wins = (filtered_df["Bet Result"] > 0).sum()
    pushes = (filtered_df["Bet Result"] == 0).sum()
    losses = (filtered_df["Bet Result"] < 0).sum()
    winrate = wins / n_matches if n_matches else 0.0
    mean_ah = filtered_df["AH_clean"].mean()
    mean_odd_liq = (filtered_df["Odd_H_liq"] if bet_on == "Home" else filtered_df["Odd_A_liq"]).mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader("📊 Backtest Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean AH (Home line)", f"{mean_ah:+.2f}")
    col4.metric("ROI (per bet)", f"{roi:.1%}")
    st.caption(f"Pushes: {pushes} · Losses: {losses} · Mean net-odds ({bet_on}): {mean_odd_liq:.2f}")

    show_cols = [
        "Date", "League", "Home", "Away",
        "Asian_Line", "AH_clean",
        "Diff_Power", "Diff_HT_P",
        "Odd_H_Asi", "Odd_A_Asi",
        "Goals_H_FT", "Goals_A_FT",
        "Bet Result", "Cumulative Profit"
    ]
    st.subheader("📝 Filtered Matches")
    st.dataframe(filtered_df[show_cols], use_container_width=True)

else:
    st.warning("⚠️ No matches found with selected filters.")
