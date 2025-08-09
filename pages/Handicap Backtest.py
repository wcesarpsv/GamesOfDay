import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Asian Handicap Backtest", layout="wide")
st.title("⚖️ Asian Handicap Backtest")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATA_FOLDER = "GamesDay/GamesAsian"   # <- pasta com os CSVs de AH

REQUIRED_COLS = [
    "Date", "League", "Home", "Away",
    "Diff_Power", "Diff_HT_P",
    "Asian_Line", "Odd_H_Asi", "Odd_A_Asi",
    "Goals_H_FT", "Goals_A_FT",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_asian_line(raw: str) -> list[float]:
    """
    Converte 'Asian_Line' em lista de componentes.
    Exemplos:
      '0' -> [0.0], 'pk' -> [0.0], '+0.25' -> [0.0, 0.5],
      '-0.75' -> [-1.0, -0.5], '0.5/1' -> [0.5, 1.0], '-0.5/0' -> [-0.5, 0.0]
    """
    if raw is None:
        return []

    s = str(raw).strip().lower().replace(' ', '')
    if s in ('pk', 'p.k.', 'level'):
        return [0.0]

    # troca vírgula por ponto se vier '0,5/1'
    s = s.replace(',', '.')

    # formato a/b
    if '/' in s:
        try:
            a, b = s.split('/')
            return [float(a), float(b)]
        except Exception:
            return []

    # formato simples: +0.25, -0.75, 1.5, 0, etc.
    s = re.sub(r'^\+', '', s)
    try:
        x = float(s)
    except Exception:
        return []

    # se for quarto de gol, quebrar em duas metades vizinhas
    frac = abs(x) - int(abs(x))
    sign = 1 if x >= 0 else -1
    base = int(abs(x))

    if abs(frac - 0.25) < 1e-9:
        return [sign * (base + 0.0), sign * (base + 0.5)]
    if abs(frac - 0.75) < 1e-9:
        return [sign * (base + 0.5), sign * (base + 1.0)]

    # meia ou inteiro
    return [x]


def settle_ah_bet(goals_h, goals_a, asian_line_components, odds, bet_side: str) -> float:
    """
    Retorna o lucro por stake=1.
    bet_side: 'Home' ou 'Away'
    asian_line_components: lista de floats (ex.: [0.5, 1.0])
    """
    # validar inputs
    if len(asian_line_components) == 0:
        return 0.0
    if pd.isna(goals_h) or pd.isna(goals_a) or pd.isna(odds):
        return 0.0

    profits = []
    for h in asian_line_components:
        if bet_side == "Home":
            margin = (goals_h - goals_a) + h
        else:  # Away → linha invertida
            margin = (goals_a - goals_h) + (-h)

        if margin > 0:
            profits.append(odds - 1.0)   # win
        elif margin == 0:
            profits.append(0.0)          # push
        else:
            profits.append(-1.0)         # loss

    return sum(profits) / len(profits)


# ──────────────────────────────────────────────────────────────────────────────
# Carregar dados
# ──────────────────────────────────────────────────────────────────────────────
all_dfs = []
if not os.path.isdir(DATA_FOLDER):
    st.error(f"❌ Folder '{DATA_FOLDER}' not found.")
    st.stop()

for file in sorted(os.listdir(DATA_FOLDER)):
    if not file.endswith(".csv"):
        continue
    path = os.path.join(DATA_FOLDER, file)
    try:
        df = pd.read_csv(path)
    except Exception:
        continue

    # exigir colunas básicas
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        continue

    # descartar linhas sem gols (não dá para assentar AH)
    df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT"])

    # parse da data
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # manter
    all_dfs.append(df)

if not all_dfs:
    st.error("❌ No valid CSVs with required columns (including goals) were found in GamesAsian.")
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True)

# criar coluna com a versão "limpa" do handicap para filtro (média dos componentes)
df_all["AH_components"] = df_all["Asian_Line"].apply(parse_asian_line)
df_all = df_all[df_all["AH_components"].map(len) > 0].copy()
df_all["AH_clean"] = df_all["AH_components"].apply(lambda lst: sum(lst)/len(lst))

# ordenar cronologicamente
df_all = df_all.sort_values(by="Date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Controles / filtros
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎚️ Filters")

# ranges
dp_min, dp_max = float(df_all["Diff_Power"].min()), float(df_all["Diff_Power"].max())
dht_min, dht_max = float(df_all["Diff_HT_P"].min()), float(df_all["Diff_HT_P"].max())
oh_min, oh_max = float(df_all["Odd_H_Asi"].min()), float(df_all["Odd_H_Asi"].max())
oa_min, oa_max = float(df_all["Odd_A_Asi"].min()), float(df_all["Odd_A_Asi"].max())
ah_min, ah_max = float(df_all["AH_clean"].min()), float(df_all["AH_clean"].max())

diff_power = st.sidebar.slider("Diff_Power", dp_min, dp_max, (dp_min, dp_max))
diff_ht_p  = st.sidebar.slider("Diff_HT_P", dht_min, dht_max, (dht_min, dht_max))
odd_h      = st.sidebar.slider("Odd_H_Asi", oh_min, oh_max, (oh_min, oh_max))
odd_a      = st.sidebar.slider("Odd_A_Asi", oa_min, oa_max, (oa_min, oa_max))
ah_range   = st.sidebar.slider("Asian Handicap (Home line)", ah_min, ah_max, (ah_min, ah_max))

bet_side = st.sidebar.selectbox("Bet on", ["Home", "Away"])

# aplicar filtros
f = df_all[
    (df_all["Diff_Power"].between(*diff_power)) &
    (df_all["Diff_HT_P"].between(*diff_ht_p)) &
    (df_all["Odd_H_Asi"].between(*odd_h)) &
    (df_all["Odd_A_Asi"].between(*odd_a)) &
    (df_all["AH_clean"].between(*ah_range))
].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────────────────────
if f.empty:
    st.warning("⚠️ No matches found with selected filters.")
    st.stop()

# lucro por jogo
if bet_side == "Home":
    f["Bet Profit"] = f.apply(
        lambda r: settle_ah_bet(
            r["Goals_H_FT"], r["Goals_A_FT"], r["AH_components"], r["Odd_H_Asi"], "Home"
        ), axis=1
    )
else:
    f["Bet Profit"] = f.apply(
        lambda r: settle_ah_bet(
            r["Goals_H_FT"], r["Goals_A_FT"], r["AH_components"], r["Odd_A_Asi"], "Away"
        ), axis=1
    )

# acumulado por número de aposta (ordem cronológica já garantida)
f["Cumulative Profit"] = f["Bet Profit"].cumsum()

# ──────────────────────────────────────────────────────────────────────────────
# Visual
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"**Matches found:** {len(f)}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(f) + 1), f["Cumulative Profit"], marker="o")
ax.set_xlabel("Bet Number")
ax.set_ylabel("Cumulative Profit")
ax.set_title("Cumulative Profit (Asian Handicap)")
st.pyplot(fig)

st.subheader("📝 Filtered Matches")
show_cols = [
    "Date", "League", "Home", "Away",
    "Asian_Line", "AH_clean",
    "Odd_H_Asi", "Odd_A_Asi",
    "Goals_H_FT", "Goals_A_FT",
    "Diff_Power", "Diff_HT_P",
    "Bet Profit", "Cumulative Profit",
]
st.dataframe(f[show_cols], use_container_width=True)
