import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

# 📁 Pasta com os arquivos
DATA_FOLDER = "GamesDay"

st.set_page_config(page_title="Data-Driven Football Insights", layout="wide")
st.title("🔮 Data-Driven Football Insights")

# 🚫 Ligas a excluir
EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "UEFA","nordeste"]

# 🧠 Função para extrair datas dos arquivos
def get_available_dates(folder):
    pattern = r'Jogosdodia_(\d{4}-\d{2}-\d{2})\.csv'
    dates = []
    for filename in os.listdir(folder):
        match = re.match(pattern, filename)
        if match:
            try:
                dates.append(datetime.strptime(match.group(1), '%Y-%m-%d').date())
            except:
                continue
    return sorted(dates)

# 🔍 Busca as datas disponíveis
available_dates = get_available_dates(DATA_FOLDER)

if not available_dates:
    st.error("❌ No CSV files found in the game data folder.")
    st.stop()

# 🔓 Mostrar todas as datas ou só as últimas
show_all = st.checkbox("🔓 Show all available dates", value=False)

# 📅 Define as datas a exibir
if show_all:
    dates_to_display = available_dates
else:
    dates_to_display = available_dates[-7:]

# 📅 Seleciona data
selected_date = st.selectbox("📅 Select a date:", dates_to_display, index=len(dates_to_display)-1)

# 🛠️ Monta caminho do arquivo
filename = f'Jogosdodia_{selected_date}.csv'
file_path = os.path.join(DATA_FOLDER, filename)

try:
    # 📥 Carrega CSV
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    # 📆 Ajusta data
    df['Date'] = df['Date'].dt.date
    df_filtered = df[df['Date'].astype(str) == selected_date.strftime('%Y-%m-%d')]

    # 🚫 Filtro interno de ligas
    if 'League' in df_filtered.columns and EXCLUDED_LEAGUE_KEYWORDS:
        pattern = '|'.join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS))
        df_filtered = df_filtered[~df_filtered['League'].astype(str).str.contains(pattern, case=False, na=False)]

    # 👁️ Remove coluna Date
    df_display = df_filtered.drop(columns=['Date'], errors='ignore')
    df_display.index = range(len(df_display))

    # 📊 Resumo
    st.markdown(f"""
### 📊 Matchday Summary – *{selected_date.strftime('%Y-%m-%d')}*

- **Total matches:** {len(df_filtered)}
- **Total leagues:** {df_filtered['League'].nunique() if 'League' in df_filtered.columns else '—'}

---

### ℹ️ Column Descriptions:

- **`Diff HT`** – Difference in team strength for the **first half**  
- **`Diff FT`** – Overall team strength difference (full match)  
- **`OU %`** – Expected total goals probability

---

### 🎨 Color Guide:

- 🟦 **Blue**: Positive advantage  
- 🟥 **Red**: Negative disadvantage  
- 🟧 **Orange**: Neutral (~0)  
- 🔵 **Bars**: Higher = greater chance of Over line goals
""")

    # ⚠️ Se não houver jogos
    if df_filtered.empty:
        st.warning("⚠️ No matches found for the selected date after applying the internal league filter.")
    else:
        # 🔄 Reordena colunas
        column_order = [
            "League", "Home", "Away",
            "Diff_HT_P", "Diff_Power",
            "M_HT_H", "M_HT_A", "M_H", "M_A",
            "OU_Total",
            "Odd_H", "Odd_D", "Odd_A",
            "Goals_H_FT", "Goals_A_FT"
        ]
        df_display = df_display[[c for c in column_order if c in df_display.columns]]

        # 🔤 Renomeia
        rename_dict = {
            "League": "League",
            "Home": "Home",
            "Away": "Away",
            "Diff_HT_P": "Diff HT",
            "Diff_Power": "Diff FT",
            "M_HT_H": "HT Home",
            "M_HT_A": "HT Away",
            "M_H": "FT Home",
            "M_A": "FT Away",
            "OU_Total": "OU %",
            "Odd_H": "Odd H",
            "Odd_D": "Odd D",
            "Odd_A": "Odd A",
            "Goals_H_FT": "Gols H",
            "Goals_A_FT": "Gols A"
        }
        df_display = df_display.rename(columns=rename_dict)

        # 🎨 Função para cor do texto
        def color_arrows(val):
            try:
                v = float(val)
            except:
                return "color: black"
            if v > 0:
                return "color: blue"
            elif v < 0:
                return "color: red"
            else:
                return "color: orange"

        # ✅ Exibe estilizado
        st.dataframe(
            df_display.style
            .format({
                "Odd H": "{:.2f}", "Odd D": "{:.2f}", "Odd A": "{:.2f}",
                "Diff HT": "{:.2f}", "Diff FT": "{:.2f}",
                "OU %": "{:.2f}%",
                "Gols H": lambda x: f"{int(x)}",
                "Gols A": lambda x: f"{int(x)}",
                "HT Home": "{:.2f}",
                "HT Away": "{:.2f}",
                "FT Home": "{:.2f}",
                "FT Away": "{:.2f}",
            })
            .applymap(color_arrows, subset=["HT Home","HT Away","FT Home","FT Away"])
            .background_gradient(cmap="RdYlGn", subset=["Diff HT", "Diff FT"])
            .bar(subset=["OU %"], color="lightblue")
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            height=1200,
            use_container_width=True
        )

except FileNotFoundError:
    st.error(f"❌ File `{filename}` not found.")
except pd.errors.EmptyDataError:
    st.error(f"❌ The file `{filename}` is empty or contains no valid data.")
