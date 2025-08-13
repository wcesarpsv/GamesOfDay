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
# Suas odds Odd_H_Asi / Odd_A_Asi já são líquidas (lucro por stake=1).
ODDS_ARE_NET = True

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_net_odds(x):
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

# ---------- Parser da linha do AWAY (preserva slashes e trata casos especiais) ----------
def parse_away_line(raw: str):
    """
    Parser robusto para a linha Asian do AWAY.
    - Mantém formatos com '/', ex.: '0/0.5', '0/-0.5', '1/1.5', '-0.5/1'
    - 'pk', 'p.k.', 'level' → [0.0]
    - ±0.6666... → ±[1.0, 1.5]
    - 0.25 → [0, 0.5]; -0.75 → [-0.5, -1.0]
    Retorna lista de floats representando os componentes do AWAY.
    """
    if raw is None:
        return []
    s = str(raw).strip().lower().replace(" ", "")
    s = s.replace(",", ".")
    if s in ("", "nan"):
        return []
    if s in ("pk", "p.k.", "level"):
        return [0.0]

    if "/" in s:
        overall_sign = -1 if s.startswith("-") else (1 if s.startswith("+") else None)
        s_no_pref = s[1:] if s and s[0] in "+-" else s
        parts = s_no_pref.split("/")
        parsed = []
        for p in parts:
            if p in ("", "nan"):
                return []
            if p[0] in "+-":
                try:
                    parsed.append(float(p))
                except Exception:
                    return []
            else:
                try:
                    val = float(p)
                except Exception:
                    return []
                if overall_sign is not None:
                    val *= overall_sign
                parsed.append(val)
        return parsed

    try:
        x = float(s)
    except Exception:
        return []
    if pd.isna(x):
        return []

    # map ±2/3 → ±[1.0, 1.5]
    if abs(abs(x) - 2/3) < 1e-6:
        sign = -1 if x < 0 else 1
        return [sign*1.0, sign*1.5]

    # expand quarters 0.25 / 0.75
    frac = abs(x) - int(abs(x))
    base = int(abs(x))
    sign = -1 if x < 0 else 1
    if abs(frac - 0.25) < 1e-9:
        return [sign*(base + 0.0), sign*(base + 0.5)]
    if abs(frac - 0.75) < 1e-9:
        return [sign*(base + 0.5), sign*(base + 1.0)]
    return [x]

def canonical(parts):
    if not parts:
        return None
    def fmt(v):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "+0") else s
    if len(parts) == 1:
        return fmt(parts[0])
    a, b = sorted(parts)
    return f"{fmt(a)}/{fmt(b)}"

def range_filter_hibrido(label: str, data_min: float, data_max: float, step: float, key_prefix: str):
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input("Min", value=float(data_min), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_min")
    max_val = c2.number_input("Max", value=float(data_max), min_value=float(data_min), max_value=float(data_max),
                              step=step, key=f"{key_prefix}_max")
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

    fonte = st.sidebar.radio("Filter source", ["Slider", "Manual"], horizontal=True, key=f"{key_prefix}_src")
    st.sidebar.divider()

    if fonte == "Slider":
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

    try:
        df = pd.read_csv(df_path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(df_path, sep="\t", encoding="utf-8-sig")
        except Exception as e:
            st.warning(f"⚠️ Failed to read {file}: {e}")
            continue

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
        st.info(f"ℹ️ Skipping {file}: missing columns -> {sorted(required - set(df.columns))}")
        continue

    # numéricos (NÃO incluir Asian_Line aqui!)
    for c in ["Odd_H_Asi","Odd_A_Asi","Goals_H_FT","Goals_A_FT","Diff_Power","Diff_HT_P"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # manter linhas com gols + odds + Asian_Line presentes
    df["Asian_Line"] = df["Asian_Line"].astype(str).str.strip().replace({"": None, "nan": None})
    df = df.dropna(subset=["Goals_H_FT","Goals_A_FT","Odd_H_Asi","Odd_A_Asi","Asian_Line"])
    if df.empty:
        st.info(f"ℹ️ Skipping {file}: no rows with goals + odds + Asian_Line.")
        continue

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # a fonte traz a linha do AWAY
    df = df.rename(columns={"Asian_Line": "Asian_Line_Away_raw"})
    df["AH_components_away"] = df["Asian_Line_Away_raw"].apply(parse_away_line)
    df["Asian_Line_Away"] = df["AH_components_away"].apply(canonical)

    # componentes do HOME (inverte sinal)
    df["AH_components_home"] = df["AH_components_away"].apply(lambda lst: [-x for x in lst])
    df = df[df["AH_components_home"].map(len) > 0].copy()
    if df.empty:
        st.info(f"ℹ️ Skipping {file}: invalid Asian_Line_Away in all rows.")
        continue

    df["AH_clean_home"] = df["AH_components_home"].apply(lambda lst: sum(lst)/len(lst))

    # odds líquidas
    df["Odd_H_liq"] = df["Odd_H_Asi"].apply(to_net_odds)
    df["Odd_A_liq"] = df["Odd_A_Asi"].apply(to_net_odds)
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
# Sidebar Filters (inclui filtro de Ligas)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Filter Matches")

# 1) League filter (Include only / Exclude)
st.sidebar.subheader("Leagues")
league_mode = st.sidebar.radio("Mode", ["Include only", "Exclude"], horizontal=True, key="league_mode")
all_leagues = sorted([x for x in df_all["League"].dropna().unique().tolist()])
selected_leagues = st.sidebar.multiselect("Leagues", options=all_leagues, default=[])

if selected_leagues:
    if league_mode == "Include only":
        df_work = df_all[df_all["League"].isin(selected_leagues)].copy()
    else:  # Exclude
        df_work = df_all[~df_all["League"].isin(selected_leagues)].copy()
else:
    df_work = df_all.copy()

if df_work.empty:
    st.warning("⚠️ No data after league filter.")
    st.stop()

# 2) Bet side FIRST (Option 2)
bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 3) Numeric filters
dp_min, dp_max = df_work["Diff_Power"].min(), df_work["Diff_Power"].max()
diff_power_sel = range_filter_hibrido("📊 Diff_Power", dp_min, dp_max, step=0.01, key_prefix="diff_power")

htp_min, htp_max = df_work["Diff_HT_P"].min(), df_work["Diff_HT_P"].max()
diff_ht_p_sel = range_filter_hibrido("📉 Diff_HT_P", htp_min, htp_max, step=0.01, key_prefix="diff_htp")

# Mapear AH para o lado apostado:
# Home → usa AH_clean_home; Away → usa -AH_clean_home
df_work["AH_clean_for_side"] = df_work["AH_clean_home"] if bet_on == "Home" else -df_work["AH_clean_home"]

ah_side_min = float(df_work["AH_clean_for_side"].min())
ah_side_max = float(df_work["AH_clean_for_side"].max())
ah_range_sel = range_filter_hibrido("⚖️ Asian Handicap (side line, AH_for_side)", ah_side_min, ah_side_max, step=0.25, key_prefix="ah_for_side")

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
eps = 1e-6
filtered_df = df_work[
    (df_work["Diff_Power"] >= diff_power_sel[0] - eps) & (df_work["Diff_Power"] <= diff_power_sel[1] + eps) &
    (df_work["Diff_HT_P"] >= diff_ht_p_sel[0]) & (df_work["Diff_HT_P"] <= diff_ht_p_sel[1]) &
    (df_work["AH_clean_for_side"] >= ah_range_sel[0]) & (df_work["AH_clean_for_side"] <= ah_range_sel[1])
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Profit calc (stake = 1)
# ──────────────────────────────────────────────────────────────────────────────
def calculate_profit(row):
    net_odds = row["Odd_H_liq"] if bet_on == "Home" else row["Odd_A_liq"]
    return settle_ah_with_odds(
        row["Goals_H_FT"], row["Goals_A_FT"], row["AH_components_home"], bet_on, net_odds
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
    mean_ah_home = filtered_df["AH_clean_home"].mean()
    mean_ah_side = filtered_df["AH_clean_for_side"].mean()
    mean_odd_liq = (filtered_df["Odd_H_liq"] if bet_on == "Home" else filtered_df["Odd_A_liq"]).mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches if n_matches else 0.0

    st.subheader("📊 Backtest Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean AH (side)", f"{mean_ah_side:+.2f}")
    col4.metric("ROI (per bet)", f"{roi:.1%}")
    st.caption(
        "Obs.: 'Asian_Line_Away_raw' vem da fonte como handicap do visitante. "
        "Internamente convertida para a linha do mandante (AH_clean_home). "
        f"Pushes: {pushes} · Losses: {losses} · Mean net-odds ({bet_on}): {mean_odd_liq:.2f} "
        f"· Mean AH (home ref): {mean_ah_home:+.2f}"
    )

    show_cols = [
        "Date", "League", "Home", "Away",
        "Asian_Line_Away_raw", "Asian_Line_Away",
        "AH_clean_home", "AH_clean_for_side",
        "Diff_Power", "Diff_HT_P",
        "Odd_H_Asi", "Odd_A_Asi", "Odd_H_liq", "Odd_A_liq",
        "Goals_H_FT", "Goals_A_FT",
        "Bet Result", "Cumulative Profit"
    ]
    st.subheader("📝 Filtered Matches")
    st.dataframe(filtered_df[show_cols], use_container_width=True)

else:
    st.warning("⚠️ No matches found with selected filters.")
