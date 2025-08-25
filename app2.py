import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

# 📁 Folder containing the game files
DATA_FOLDER = "GamesDay"

st.set_page_config(page_title="Data-Driven Football Insights", layout="wide")
st.title("🔮 Data-Driven Football Insights")

# 🚫 Keywords (case-insensitive) that, if found in League, will EXCLUDE the row
EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "UEFA","nordeste"]

# 🧠 Helper function to extract available dates from filenames
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

# 🎯 Função para mostrar setas baseado na média da coluna
def arrow_trend(val, mean, threshold=0.6):
    try:
        v = float(val)
    except:
        return val
    
    if v > mean + threshold:
        return f"🔵 {v:.2f}"   # acima da média
    elif v < mean - threshold:
        return f"🔴 {v:.2f}"   # abaixo da média
    else:
        return f"🟠 {v:.2f}"   # próximo da média

# 🔍 Get available dates from CSV files
available_dates = get_available_dates(DATA_FOLDER)

if not available_dates:
    st.error("❌ No CSV files found in the game data folder.")
    st.stop()

# 🔓 Option to show all dates or only the most recent
show_all = st.checkbox("🔓 Show all available dates", value=False)

# 📅 Limit the list if the user doesn't want full history
if show_all:
    dates_to_display = available_dates
else:
    dates_to_display = available_dates[-7:]  # Show only the last 7 days

# 📅 Date selector
selected_date = st.selectbox("📅 Select a date:", dates_to_display, index=len(dates_to_display)-1)

# 🛠️ Build the file path for the selected date
filename = f'Jogosdodia_{selected_date}.csv'
file_path = os.path.join(DATA_FOLDER, filename)

try:
    # 📥 Load the CSV
    df = pd.read_csv(file_path, parse_dates=['Date'])  # já tenta parsear

    # 🧹 Clean up the data
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    # 📆 Ensure the 'Date' column is datetime.date
    df['Date'] = df['Date'].dt.date
    df_filtered = df[df['Date'].astype(str) == selected_date.strftime('%Y-%m-%d')]

    # 🚫 Apply internal league filter (case-insensitive), if coluna existir
    if 'League' in df_filtered.columns and EXCLUDED_LEAGUE_KEYWORDS:
        pattern = '|'.join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS))
        df_filtered = df_filtered[~df_filtered['League'].astype(str).str.contains(pattern, case=False, na=False)]

    # 👁️ Remove 'Date' column from display and reset index
    df_display = df_filtered.drop(columns=['Date'], errors='ignore')
    df_display.index = range(len(df_display))

    # 📊 Summary and explanation
    st.markdown(f"""
### 📊 Matchday Summary – *{selected_date.strftime('%Y-%m-%d')}*

- **Total matches:** {len(df_filtered)}
- **Total leagues:** {df_filtered['League'].nunique() if 'League' in df_filtered.columns else '—'}

---

### ℹ️ Column Descriptions:

- **`Diff_HT_P`** – Difference in team strength for the **first half**, based on Power Ratings  
- **`Diff_Power`** – Overall team strength difference for the full match (FT)  
- **`OU_Total`** – Expected total goals for the match (higher = greater chance of Over Line Goals)

---

### 🎨 Color Guide:

- 🔵 **Above average**  
- 🔴 **Below average**  
- 🟠 **Near average**  
- 🟩 **Green background**: Home advantage  
- 🟥 **Red background**: Away advantage  
- 🔵 **Blue background**: Higher expected goals
""")

    # ⚠️ Show warning if no matches found
    if df_filtered.empty:
        st.warning("⚠️ No matches found for the selected date after applying the internal league filter.")
    else:
        # 🔢 Calcula médias por coluna
        means = {
            "M_HT_H": df_display["M_HT_H"].mean(),
            "M_HT_A": df_display["M_HT_A"].mean(),
            "M_H": df_display["M_H"].mean(),
            "M_A": df_display["M_A"].mean()
        }

        # ✅ Cria estilo
        styled = (
            df_display.style
            .format({
                'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
                'Diff_HT_P': '{:.2f}', 'Diff_Power': '{:.2f}',
                'OU_Total': lambda x: f"{x * 100:.2f}",
                'Goals_H_FT': lambda x: f"{int(x)}",
                'Goals_A_FT': lambda x: f"{int(x)}",
                'M_HT_H': lambda x: arrow_trend(x, means["M_HT_H"]),
                'M_HT_A': lambda x: arrow_trend(x, means["M_HT_A"]),
                'M_H': lambda x: arrow_trend(x, means["M_H"]),
                'M_A': lambda x: arrow_trend(x, means["M_A"]),
            })
            .background_gradient(cmap='RdYlGn', subset=['Diff_HT_P','Diff_Power'])
            .background_gradient(cmap='Blues', subset=['OU_Total'])
        )

        # ✅ Show styled table
        st.dataframe(styled, height=1200, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ File `{filename}` not found.")
except pd.errors.EmptyDataError:
    st.error(f"❌ The file `{filename}` is empty or contains no valid data.")



