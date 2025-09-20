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
# Bloco 4 – Perspective for the Day (Segmented by Diff_Power, Diff_M, Diff_HT_P)
# ########################################################
import numpy as np

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
        needed_cols = ["Diff_Power", "M_H", "M_A", "Diff_HT_P", "Goals_H_FT", "Goals_A_FT"]
        if all(col in df_history.columns for col in needed_cols):

            # Criar coluna Diff_M (diferença de força ofensiva)
            df_history["Diff_M"] = df_history["M_H"] - df_history["M_A"]

            # Criar bins para as 3 métricas
            df_history["DiffPower_bin"] = pd.cut(df_history["Diff_Power"], bins=range(-50, 55, 10))
            df_history["DiffM_bin"] = pd.cut(df_history["Diff_M"], bins=np.arange(-10, 10.5, 1.0))
            df_history["DiffHTP_bin"] = pd.cut(df_history["Diff_HT_P"], bins=range(-30, 35, 5))

            # Resultado real
            def get_result(row):
                if row["Goals_H_FT"] > row["Goals_A_FT"]:
                    return "Home"
                elif row["Goals_H_FT"] < row["Goals_A_FT"]:
                    return "Away"
                else:
                    return "Draw"

            df_history["Result"] = df_history.apply(get_result, axis=1)

            # 🔹 Carregar jogos do dia diretamente
            df_day = pd.read_csv(file_path)
            df_day = df_day.loc[:, ~df_day.columns.str.contains('^Unnamed')]
            df_day.columns = df_day.columns.str.strip()
            if "Date" in df_day.columns:
                df_day["Date"] = pd.to_datetime(df_day["Date"], errors="coerce").dt.date
                df_day = df_day[df_day["Date"] == selected_date]

            # Criar coluna Diff_M no df_day
            df_day["Diff_M"] = df_day["M_H"] - df_day["M_A"]
            df_day = df_day.dropna(subset=["Diff_Power", "Diff_M", "Diff_HT_P"])

            total_matches = 0
            home_wins, away_wins, draws = 0, 0, 0

            # Intervalos dos bins
            dp_bins = pd.IntervalIndex(df_history["DiffPower_bin"].cat.categories)
            dm_bins = pd.IntervalIndex(df_history["DiffM_bin"].cat.categories)
            dhtp_bins = pd.IntervalIndex(df_history["DiffHTP_bin"].cat.categories)

            for _, game in df_day.iterrows():
                try:
                    # Garantir que os valores do jogo estão dentro dos bins
                    if (
                        dp_bins.contains(game["Diff_Power"]).any() and
                        dm_bins.contains(game["Diff_M"]).any() and
                        dhtp_bins.contains(game["Diff_HT_P"]).any()
                    ):
                        dp_bin = dp_bins.get_loc(game["Diff_Power"])
                        dm_bin = dm_bins.get_loc(game["Diff_M"])
                        dhtp_bin = dhtp_bins.get_loc(game["Diff_HT_P"])
                    else:
                        continue  # pula jogo fora dos ranges

                    # Filtrar histórico com base nos 3 bins
                    subset = df_history[
                        (df_history["DiffPower_bin"] == dp_bins[dp_bin]) &
                        (df_history["DiffM_bin"] == dm_bins[dm_bin]) &
                        (df_history["DiffHTP_bin"] == dhtp_bins[dhtp_bin])
                    ]

                    if not subset.empty:
                        total_matches += len(subset)
                        home_wins += (subset["Result"] == "Home").sum()
                        away_wins += (subset["Result"] == "Away").sum()
                        draws += (subset["Result"] == "Draw").sum()
                except Exception:
                    continue  # segurança extra

            # Exibir resultados
            if total_matches > 0:
                pct_home = 100 * home_wins / total_matches
                pct_away = 100 * away_wins / total_matches
                pct_draw = 100 * draws / total_matches

                st.markdown("## ===== Perspective for the Day =====")
                st.write("*(Based on historical matches with similar Diff_Power, Diff_M, Diff_HT_P)*")
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













