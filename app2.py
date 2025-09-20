import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

# ########################################################
# Bloco 1 – Configuração Inicial
# ########################################################
DATA_FOLDER = "GamesDay"

st.set_page_config(page_title="Data-Driven Football Insights", layout="wide")
st.title("🔮 Data-Driven Football Insights")

EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "uefa", "nordeste", "afc"]


# ########################################################
# Bloco 2 – Funções auxiliares
# ########################################################
def get_available_dates(folder):
    pattern = re.compile(r'jogosdodia_(\d{4}-\d{2}-\d{2})\.csv', re.IGNORECASE)
    dates = []
    for filename in os.listdir(folder):
        match = pattern.search(filename)
        if match:
            try:
                dates.append(datetime.strptime(match.group(1), '%Y-%m-%d').date())
            except:
                continue
    return sorted(dates)


def arrow_trend(val, mean, threshold=0.4):
    try:
        v = float(val)
    except:
        return val

    if v > mean + threshold:
        return f"🔵 {v:.2f}"
    elif v < mean - threshold:
        return f"🔴 {v:.2f}"
    else:
        return f"🟠 {v:.2f}"


# ########################################################
# Bloco 3 – Seleção de Data e Arquivo
# ########################################################
available_dates = get_available_dates(DATA_FOLDER)
if not available_dates:
    st.error("❌ No CSV files found in the game data folder.")
    st.stop()

latest_date = available_dates[-1]

if "last_seen_date" not in st.session_state:
    st.session_state.last_seen_date = latest_date

if latest_date and latest_date != st.session_state.last_seen_date:
    st.cache_data.clear()
    st.session_state.last_seen_date = latest_date
    st.rerun()

show_all = st.checkbox("🔓 Show all available dates", value=False)
dates_to_display = available_dates if show_all else available_dates[-7:]
default_index = dates_to_display.index(latest_date) if latest_date in dates_to_display else len(dates_to_display) - 1

selected_date = st.selectbox(
    "📅 Select a date:",
    dates_to_display,
    index=default_index
)

filename = f'Jogosdodia_{selected_date}.csv'
file_path = os.path.join(DATA_FOLDER, filename)


# ########################################################
# Bloco 4 – Perspective for the Day (Segmented by Diff_Power, M_H, M_A)
# ########################################################
try:
    all_dfs = []
    for f in os.listdir(DATA_FOLDER):
        if f.lower().endswith(".csv"):
            try:
                df_tmp = pd.read_csv(os.path.join(DATA_FOLDER, f))
                df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('^Unnamed')]
                df_tmp.columns = df_tmp.columns.str.strip()
                all_dfs.append(df_tmp)
            except:
                continue

    if all_dfs:
        df_history = pd.concat(all_dfs, ignore_index=True)

        # 🧹 Remove duplicates
        df_history = df_history.drop_duplicates(
            subset=["League", "Home", "Away", "Odd_H", "Odd_D", "Odd_A"],
            keep="first"
        )

        # Garantir coluna Date e excluir jogos do dia selecionado
        if "Date" in df_history.columns:
            df_history["Date"] = pd.to_datetime(df_history["Date"], errors="coerce").dt.date
            df_history = df_history[df_history["Date"] != selected_date]

        # Precisamos dessas colunas no histórico
        needed_cols = ["Diff_Power", "M_H", "M_A", "Goals_H_FT", "Goals_A_FT"]
        if all(col in df_history.columns for col in needed_cols):

            # Criar bins
            df_history["DiffPower_bin"] = pd.cut(df_history["Diff_Power"], bins=range(-50, 55, 5))
            df_history["M_H_bin"] = pd.cut(df_history["M_H"], bins=[i*0.5 for i in range(0, 41)])
            df_history["M_A_bin"] = pd.cut(df_history["M_A"], bins=[i*0.5 for i in range(0, 41)])

            # Resultado real
            def get_result(row):
                if row["Goals_H_FT"] > row["Goals_A_FT"]:
                    return "Home"
                elif row["Goals_H_FT"] < row["Goals_A_FT"]:
                    return "Away"
                else:
                    return "Draw"

            df_history["Result"] = df_history.apply(get_result, axis=1)

            # Agora para os jogos do dia
            df_day = df[df["Date"] == selected_date].copy()
            df_day = df_day.dropna(subset=["Diff_Power", "M_H", "M_A"])

            total_matches = 0
            home_wins, away_wins, draws = 0, 0, 0

            for _, game in df_day.iterrows():
                dp_bin = pd.IntervalIndex(df_history["DiffPower_bin"].cat.categories).get_loc(game["Diff_Power"])
                mh_bin = pd.IntervalIndex(df_history["M_H_bin"].cat.categories).get_loc(game["M_H"])
                ma_bin = pd.IntervalIndex(df_history["M_A_bin"].cat.categories).get_loc(game["M_A"])

                subset = df_history[
                    (df_history["DiffPower_bin"] == df_history["DiffPower_bin"].cat.categories[dp_bin]) &
                    (df_history["M_H_bin"] == df_history["M_H_bin"].cat.categories[mh_bin]) &
                    (df_history["M_A_bin"] == df_history["M_A_bin"].cat.categories[ma_bin])
                ]

                if not subset.empty:
                    total_matches += len(subset)
                    home_wins += (subset["Result"] == "Home").sum()
                    away_wins += (subset["Result"] == "Away").sum()
                    draws += (subset["Result"] == "Draw").sum()

            if total_matches > 0:
                pct_home = 100 * home_wins / total_matches
                pct_away = 100 * away_wins / total_matches
                pct_draw = 100 * draws / total_matches

                st.markdown("## ===== Perspective for the Day =====")
                st.write("*(Based on historical matches with similar Diff_Power, M_H, M_A)*")
                st.write(f"**Home Wins:** {pct_home:.1f}%")
                st.write(f"**Draws:** {pct_draw:.1f}%")
                st.write(f"**Away Wins:** {pct_away:.1f}%")
                st.write(f"*Based on {total_matches:,} similar matches in history (excluding {selected_date})*")
            else:
                st.info("No similar historical matches found for today's games.")

except Exception as e:
    st.warning(f"⚠️ Could not build segmented perspective: {e}")




# ########################################################
# Bloco 5 – Carregar e Filtrar Jogo do Dia
# ########################################################
try:
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = df["Date"].dt.date
    else:
        st.error("❌ File does not contain 'Date' column.")
        st.stop()

    df_filtered = df[df["Date"] == selected_date]

    if "League" in df_filtered.columns:
        df_filtered["League"] = df_filtered["League"].astype(str).str.strip()
        if EXCLUDED_LEAGUE_KEYWORDS:
            pattern = "|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS))
            df_filtered = df_filtered[~df_filtered["League"].str.contains(pattern, case=False, na=False)]

    selected_columns = [
        "Date", "Time", "League", "Home", "Away",
        "Diff_HT_P", "Diff_Power", "OU_Total",
        "M_HT_H", "M_HT_A", "M_H", "M_A",
        "Odd_H", "Odd_D", "Odd_A"
    ]
    existing_columns = [c for c in selected_columns if c in df_filtered.columns]
    df_display = df_filtered[existing_columns].copy()
    df_display.index = range(len(df_display))

    st.markdown(f"""
### 📊 Matchday Summary – *{selected_date.strftime('%Y-%m-%d')}*

- **Total matches:** {len(df_filtered)}
- **Total leagues:** {df_filtered['League'].nunique() if 'League' in df_filtered.columns else '—'}

---
""")

    if df_display.empty:
        st.warning("⚠️ No matches found for the selected date after applying the filters.")
    else:
        mean_cols = {col: df_display[col].mean() for col in ["M_HT_H", "M_HT_A", "M_H", "M_A"] if col in df_display.columns}

        styled = (
            df_display.style
            .format({
                "Odd_H": "{:.2f}" if "Odd_H" in df_display.columns else None,
                "Odd_D": "{:.2f}" if "Odd_D" in df_display.columns else None,
                "Odd_A": "{:.2f}" if "Odd_A" in df_display.columns else None,
                "Diff_HT_P": "{:.2f}" if "Diff_HT_P" in df_display.columns else None,
                "Diff_Power": "{:.2f}" if "Diff_Power" in df_display.columns else None,
                "OU_Total": (lambda x: f"{x:.2f}") if "OU_Total" in df_display.columns else None,
                "M_HT_H": (lambda x: arrow_trend(x, mean_cols["M_HT_H"])) if "M_HT_H" in mean_cols else None,
                "M_HT_A": (lambda x: arrow_trend(x, mean_cols["M_HT_A"])) if "M_HT_A" in mean_cols else None,
                "M_H": (lambda x: arrow_trend(x, mean_cols["M_H"])) if "M_H" in mean_cols else None,
                "M_A": (lambda x: arrow_trend(x, mean_cols["M_A"])) if "M_A" in mean_cols else None,
            })
            .background_gradient(cmap="RdYlGn", subset=[c for c in ["Diff_HT_P", "Diff_Power"] if c in df_display.columns])
            .background_gradient(cmap="Blues", subset=[c for c in ["OU_Total"] if c in df_display.columns])
        )

        st.dataframe(styled, height=1200, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ File `{filename}` not found.")
except pd.errors.EmptyDataError:
    st.error(f"❌ The file `{filename}` is empty or contains no valid data.")
except Exception as e:
    st.error(f"⚠️ Unexpected error: {e}")





