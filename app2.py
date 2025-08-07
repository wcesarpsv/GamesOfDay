# app2.py (ou main.py)

import streamlit as st
import pandas as pd
import os, re
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
import urllib.parse
import streamlit.components.v1 as components

# ─── 1) Configuração da página ────────────────────────────────────────────────
st.set_page_config(page_title="Data-Driven Football Insights", layout="wide")
st.title("🔮 Data-Driven Football Insights")

# ─── 2) Helper para extrair datas dos arquivos ────────────────────────────────
def get_available_dates(folder):
    pattern = r'Jogosdodia_(\d{4}-\d{2}-\d{2})\.csv'
    dates = []
    for fn in os.listdir(folder):
        m = re.match(pattern, fn)
        if m:
            try:
                dates.append(datetime.strptime(m.group(1), '%Y-%m-%d').date())
            except:
                pass
    return sorted(dates)

# ─── 3) Carregamento de arquivos ───────────────────────────────────────────────
DATA_FOLDER = "GamesDay"
available_dates = get_available_dates(DATA_FOLDER)
if not available_dates:
    st.error("❌ No CSV files found in the game data folder.")
    st.stop()

show_all = st.checkbox("🔓 Show all available dates", value=False)
dates_to_display = available_dates if show_all else available_dates[-7:]
selected_date = st.selectbox("📅 Select a date:", dates_to_display, index=len(dates_to_display)-1)

file_path = os.path.join(DATA_FOLDER, f"Jogosdodia_{selected_date}.csv")
try:
    df = pd.read_csv(file_path, parse_dates=['Date'])
except FileNotFoundError:
    st.error(f"❌ File `{file_path}` not found.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error(f"❌ The file `{file_path}` is empty.")
    st.stop()

# ─── 4) Limpeza e preparo ─────────────────────────────────────────────────────
df = df.loc[:, ~df.columns.str.contains('^Unnamed')].dropna(axis=1, how='all')
df['Date'] = df['Date'].dt.date
df_filtered = df[df['Date'] == selected_date]

if df_filtered.empty:
    st.warning("⚠️ No matches found for the selected date.")
    st.stop()

df_display = df_filtered.drop(columns=['Date']).reset_index(drop=True)
# forçar tipos float nas colunas de interesse
for c in ['Odd_H','Odd_D','Odd_A','Diff_HT_P','Diff_Power','OU_Total']:
    df_display[c] = df_display[c].astype(float)

# ─── 5) Exibição com AgGrid ────────────────────────────────────────────────────
st.markdown("### 🎨 Color Guide:\n- 🟩 Green: home advantage\n- 🟥 Red: away advantage\n- 🔵 Blue: high expected goals")

gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_selection(selection_mode="single", use_checkbox=True)
grid_response = AgGrid(
    df_display,
    gridOptions=gb.build(),
    enable_enterprise_modules=False,
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True,
    height=400
)

selected_rows = grid_response["selected_rows"]  # sempre uma lista (possivelmente vazia)

# ─── 6) Seleção de colunas para filtrar ────────────────────────────────────────
cols = ["Diff_Power", "Diff_HT_P", "Odd_H", "Odd_D", "Odd_A"]
chosen = st.multiselect("Selecione ao menos 2 filtros para o BackTest:", options=cols)

# ─── 7) Botão de redirecionamento ──────────────────────────────────────────────
if st.button("🔎 BackTest Check"):
    if len(selected_rows) == 0:
        st.warning("❌ Selecione uma linha na tabela antes de rodar o BackTest.")
    elif len(chosen) < 2:
        st.warning("❌ Escolha pelo menos 2 colunas de filtro.")
    else:
        # extrai valores da linha selecionada
        row = selected_rows[0]
        params = {c: row[c] for c in chosen}
        query = urllib.parse.urlencode(params)
        target = f"/Strategy_Backtest?{query}"
        # redireciona via JS
        components.html(f"<script>window.location.href = '{target}';</script>", height=0)
