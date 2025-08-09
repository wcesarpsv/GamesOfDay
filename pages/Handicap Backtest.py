import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Backtest – Asian Handicap (Com Odds)", layout="wide")
st.title("📈 Strategy Backtest – Asian Handicap (Com Odds)")

# 🔹 Pasta com CSVs
GAMES_FOLDER = "GamesDay/GamesAsian"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _base_parse_asian_line(text: str):
    """Parse básico sem sinal. Retorna componentes (>=0) para suportar quartas/meias."""
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
    """Retorna componentes do handicap **com sinal para o time da casa**.
       Regra: sem sinal explícito ⇒ POSITIVO para HOME.
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
        sign = 1
    return [sign * p for p in parts]

def to_net_odds(x):
    """Normaliza odd: aceita formato líquido (~0.8–1.2) ou decimal (>=1.5).
       - Se x >= 1.5 → retorna (x - 1)
       - Caso contrário → assume que já é líquida
    """
    if pd.isna(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v - 1.0 if v >= 1.5 else v

def settle_ah_with_odds(goals_h, goals_a, ah_components, bet_on: str, net_odds: float) -> float:
    """Calcula o lucro por aposta (stake=1) usando **odd líquida**:
       - vitória componente → +net_odds
       - push → 0
       - derrota → -1
       Resultado final é a MÉDIA dos componentes (gera meia vitória/derrota automaticamente).
    """
    if (
        len(ah_components) == 0
        or pd.isna(goals_h) or pd.isna(goals_a)
        or net_odds is None
    ):
        return 0.0

    profits = []
    for h in ah_components:
        if bet_on == "Home":
            margin = (goals_h - goals_a) + h
        else:
            margin = (goals_a - goals_h) + (-h)

        if margin > 0:
            profits.append(net_odds)   # vitória
        elif abs(margin) < 1e-9:
            profits.append(0.0)        # push
        else:
            profits.append(-1.0)       # derrota

    return sum(profits) / len(profits)

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

        # componentes + AH limpo (média, para filtros)
        df["AH_components"] = df["Asian_Line"].apply(parse_asian_line)
        df = df[df["AH_components"].map(len) > 0].copy()
        df["AH_clean"] = df["AH_components"].apply(lambda lst: sum(lst)/len(lst))

        # normaliza odds para formato líquido
        df["Odd_H_liq"] = df["Odd_H_Asi"].apply(to_net_odds)
        df["Odd_A_liq"] = df["Odd_A_Asi"].apply(to_net_odds)

        all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid data (with goals, Asian_Line and odds) found in GamesAsian.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# 🎚️ Filtros
st.sidebar.header("🎯 Filter Matches")
diff_power = st.sidebar.slider(
    "📊 Diff_Power",
    float(df_all["Diff_Power"].min()),
    float(df_all["Diff_Power"].max()),
    (float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max()))
)
diff_ht_p = st.sidebar.slider(
    "📉 Diff_HT_P",
    float(df_all["Diff_HT_P"].min()),
    float(df_all["Diff_HT_P"].max()),
    (float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max()))
)
ah_min, ah_max = float(df_all["AH_clean"].min()), float(df_all["AH_clean"].max())
ah_range = st.sidebar.slider(
    "⚖️ Asian Handicap (Home line, AH_clean)",
    ah_min, ah_max, (ah_min, ah_max)
)

bet_on = st.sidebar.selectbox("🎯 Bet on", ["Home", "Away"])

# 🧮 Aplica filtros
filtered_df = df_all[
    (df_all["Diff_Power"] >= diff_power[0]) & (df_all["Diff_Power"] <= diff_power[1]) &
    (df_all["Diff_HT_P"] >= diff_ht_p[0]) & (df_all["Diff_HT_P"] <= diff_ht_p[1]) &
    (df_all["AH_clean"] >= ah_range[0]) & (df_all["AH_clean"] <= ah_range[1])
].copy()

# 🧠 Cálculo do lucro com odds
def calculate_profit(row):
    net_odds = row["Odd_H_liq"] if bet_on == "Home" else row["Odd_A_liq"]
    return settle_ah_with_odds(
        row["Goals_H_FT"], row["Goals_A_FT"], row["AH_components"], bet_on, net_odds
    )

if not filtered_df.empty:
    filtered_df["Bet Result"] = filtered_df.apply(calculate_profit, axis=1)
    filtered_df["Cumulative Profit"] = filtered_df["Bet Result"].cumsum()

    # 📈 Gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(filtered_df)), filtered_df["Cumulative Profit"], marker="o")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit by Bet (Asian Handicap, Stake=1)")
    st.pyplot(fig)

    # 🔢 Métricas
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

    # 📋 Tabela
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
