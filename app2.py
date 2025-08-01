import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

# 📁 Folder containing the game files
DATA_FOLDER = "GamesDay"

st.set_page_config(page_title="Data-Driven Football Insights", layout="wide")
st.title("🔮 Data-Driven Football Insights")

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
    dates_to_display = available_dates[-15:]  # Show only the last 15 days

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

    # 📆 Ensure the 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.date
    df_filtered = df[df['Date'] == selected_date]

    # 👁️ Remove 'Date' column from display and reset index
    df_display = df_filtered.drop(columns=['Date'])
    df_display.index = range(len(df_display))

    # 📊 Summary and explanation
    st.markdown(f"""
### 📊 Matchday Summary – *{selected_date.strftime('%Y-%m-%d')}*

- **Total matches:** {len(df_filtered)}
- **Total leagues:** {df_filtered['League'].nunique()}

---

### ℹ️ Column Descriptions:

- **`Diff_HT_P`** – Difference in team strength for the **first half**, based on Power Ratings  
- **`Diff_Power`** – Overall team strength difference for the full match (FT)  
- **`OU_Total`** – Expected total goals for the match (higher = greater chance of Over 2.5 goals)

---

### 🎨 Color Guide:

- 🟩 **Green**: Advantage for the **home team**  
- 🟥 **Red**: Advantage for the **away team**  
- 🔵 **Blue**: Higher expected total goals
""")

    # ⚠️ Show warning if no matches found
    if df_filtered.empty:
        st.warning("⚠️ No matches found for the selected date.")
    else:
        # ✅ Display styled table
        st.dataframe(
            df_display.style
            .format({
                'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
                'Diff_HT_P': '{:.2f}', 'Diff_Power': '{:.2f}', 'OU_Total': '{:.2f}'
            })
            .background_gradient(cmap='RdYlGn', subset=['Diff_HT_P', 'Diff_Power'])
            .background_gradient(cmap='Blues', subset=['OU_Total']),
            height=1200,
            use_container_width=True
        )

except FileNotFoundError:
    st.error(f"❌ File `{filename}` not found.")
except pd.errors.EmptyDataError:
    st.error(f"❌ The file `{filename}` is empty or contains no valid data.")
