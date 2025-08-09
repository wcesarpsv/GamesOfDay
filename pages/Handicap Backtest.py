
import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest – Asian Handicap (No Odds)", layout="wide")

st.title("📈 Strategy Backtest – Asian Handicap (No Odds)")

# 🔹 Folder containing match data CSVs (Asian Handicap)
GAMES_FOLDER = "GamesDay/GamesAsian"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _base_parse_asian_line(text: str):
    """Parse básico sem sinal. Retorna componentes (>=0) para suportar quartas/meias.
       Exemplos: '0.5/1' -> [0.5, 1.0], '-0.75' -> [0.75] (sinal tratado depois)
    """
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
    """Gera componentes do handicap **com sinal para o time da casa**.
       Regras: se não houver sinal explícito, assume POSITIVO para HOME.
    """
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
        sign = 1  # padrão: sem sinal ⇒ positivo para casa
    return [sign * p for p in parts]

def settle_ah_unit(goals_h, goals_a, asian_line_components, bet_on: str) -> float:
    """
    Lucro por stake=1 sem odds:
      - full win  -> +1.0
      - half win  -> +0.5  (acontece em linhas fracionadas via média)
      - push      ->  0.0
      - half loss -> -0.5
      - loss      -> -1.0
    Implementado via média dos componentes (ex.: [-0.5, 0.0] etc.).
    """
    if len(asian_line_components) == 0 or pd.isna(goals_h) or pd.isna(goals_a):
        return 0.0

    profits = []
    for h in asian_line_components:
        if bet_on == "Home":
            margin = (goals_h - goals_a) + h
        else:
            # para AWAY, o handicap do away é o inverso do home
            margin = (goals_a - goals_h) + (-h)

        if margin > 0:
            profits.append(1.0)      # win
        elif abs(margin) < 1e-9:
            profits.append(0.0)      # push
        else:
            profits.append(-1.0)     # loss

    # média dos componentes (gera +0.5/-0.5 automaticamente se split)
    return sum(profits) / len(profits)

# ⬇️ Load all valid CSVs with goal + AH data
all_dfs = []
if not os.path.isdir(GAMES_FOLDER):
    st.error("❌ Folder 'GamesAsian' not found.")
    st.stop()

for file in sorted(os.listdir(GAMES_FOLDER)):  # Sort files alphabetically (oldest first)
    if file.endswith(".csv"):
        df_path = os.path.join(GAMES_FOLDER, file)
        try:
            df = pd.read_csv(df_path)
        except Exception:
            continue
        # precisa ter colunas de gols e handicap + diffs básicos
        required = {"Goals_H_FT","Goals_A_FT","Asian_Line","Date",
                    "Diff_Power","Diff_HT_P","League","Home","Away"}
        if not required.issubset(df.columns):
            continue
        # remover linhas sem placar
        df = df.dropna(subset=["Goals_H_FT","Goals_A_FT"])
        if df.empty:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        # componentes do AH e versão limpa (média)
        df["AH_components"] = df["Asian_Line"].apply(parse_asian_line)
        df = df[df["AH_components"].map(len) > 0].copy()
        df["AH_clean"] = df["AH_components"].apply(lambda lst: sum(lst)/len(lst))
        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data (with goals & Asian_Line) found in GamesAsian.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# 🎚️ Filter sliders (iguais ao layout da sua página principal)
st.sidebar.header("🎯 Filter Matches")
diff_power = st.sidebar.slider("📊 Diff_Power",
                               float(df_all["Diff_Power"].min()),
                               float(df_all["Diff_Power"].max()),
                               (float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())))
diff_ht_p = st.sidebar.slider("📉 Diff_HT_P",
                              float(df_all["Diff_HT_P"].min()),
                              float(df_all["Diff_HT_P"].max()),
                              (float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())))
ah_min, ah_max = float(df_all["AH_clean"].min()), float(df_all["AH_clean"].max())
ah_range = st.sidebar.slider("⚖️ Asian Handicap (Home line, AH_clean)",
                             ah_min, ah_max, (ah_min, ah_max))

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 🧮 Apply filters
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power[0]) & (df_all["Diff_Power"] <= diff_power[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p[0]) & (df_all["Diff_HT_P"] <= diff_ht_p[1]) &
    (df_all["AH_clean"] >= ah_range[0]) & (df_all["AH_clean"] <= ah_range[1])
].copy()

# 🧠 Calculate bet result (sem odds: retorno unitário)
def calculate_profit(row):
    return settle_ah_unit(row["Goals_H_FT"], row["Goals_A_FT"], row["AH_components"], bet_on)

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # 📈 Plot profit by bet number
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(filtered_df)), filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit by Bet (Asian Handicap, Unit Stake)")
    st.pyplot(fig)

    # 🔢 Backtest Summary Metrics (iguais ao layout da sua página)
    n_matches = len(filtered_df)
    wins = (filtered_df["Bet Result"] > 0).sum()
    winrate = wins / n_matches
    mean_ah = filtered_df["AH_clean"].mean()
    total_profit = filtered_df["Bet Result"].sum()
    roi = total_profit / n_matches

    st.subheader("📊 Backtest Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Matches", f"{n_matches}")
    col2.metric("Winrate", f"{winrate:.1%}")
    col3.metric("Mean AH (Home line)", f"{mean_ah:+.2f}")
    col4.metric("ROI (per bet)", f"{roi:.1%}")

    # 📋 Show filtered table (mesmo padrão de colunas)
    st.subheader("📝 Filtered Matches")
    st.dataframe(
        filtered_df[[
            "Date", "League", "Home", "Away",
            "Asian_Line", "AH_clean",
            "Diff_Power", "Diff_HT_P",
            "Goals_H_FT", "Goals_A_FT",
            "Bet Result", "Cumulative Profit"
        ]],
        use_container_width=True
    )
else:
    st.warning("⚠️ No matches found with selected filters.")
