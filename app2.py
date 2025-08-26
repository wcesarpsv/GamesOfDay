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

# 🔍 Get available dates from CSV files
available_dates = get_available_dates(DATA_FOLDER)

# 🛠️ DEBUG: mostrar arquivos na pasta
st.write("📂 Arquivos encontrados na pasta:", os.listdir(DATA_FOLDER))
st.write("📅 Datas extraídas:", available_dates)

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
    df = pd.read_csv(file_path)

    # 🧹 Clean up the data
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    # 📆 Ensure the 'Date' column exists and convert properly
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = df["Date"].dt.date
    else:
        st.error("❌ O arquivo não contém a coluna 'Date'.")
        st.stop()

    # 📅 Filter by selected date
    df_filtered = df[df["Date"] == selected_date]

    # 🚫 Apply internal league filter (after strip/lower)
    if "League" in df_filtered.columns:
        df_filtered["League"] = df_filtered["League"].astype(str).str.strip()
        if EXCLUDED_LEAGUE_KEYWORDS:
            pattern = "|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS))
            df_filtered = df_filtered[~df_filtered["League"].str.contains(pattern, case=False, na=False)]

    # 👁️ Remove 'Date' column from display and reset index
    df_display = df_filtered.drop(columns=["Date"], errors="ignore")
    df_display.index = range(len(df_display))

    # 📊 Summary and explanation
    st.markdown(f"""
### 📊 Matchday Summary – *{selected_date.strftime('%Y-%m-%d')}*

- **Total matches:** {len(df_filtered)}
- **Total leagues:** {df_filtered['League'].nunique() if 'League' in df_filtered.columns else '—'}
""")

    if df_filtered.empty:
        st.warning("⚠️ No matches found for the selected date after applying the internal league filter.")
    else:
        st.dataframe(df_display, height=800, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ File `{filename}` not found.")
except pd.errors.EmptyDataError:
    st.error(f"❌ The file `{filename}` is empty or contains no valid data.")
except Exception as e:
    st.error(f"⚠️ Unexpected error: {e}")
