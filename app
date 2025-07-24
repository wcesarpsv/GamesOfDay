import pandas as pd
import streamlit as st
from datetime import date, datetime, timedelta
pip install streamlit pandas



day = date.today()

# Carregar o CSV com os dados dos jogos (coloque o caminho correto do seu arquivo)
csv_path = f'C:\\Users\\flavia\\FlashScore\\PyCaret\\Jogosdodia\\JogosDia\\Jogosdodia_{day}.csv'  # ou .xlsx se preferir
df = pd.read_csv(csv_path)

# Converter a data para formato datetime e filtrar jogos do dia
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
hoje = pd.to_datetime("today").date()
df_hoje = df[df['Date'] == hoje]

st.set_page_config(page_title="Previsões dos Jogos do Dia", layout="wide")
st.title("🔮 Previsões dos Jogos de Futebol")
st.markdown(f"📅 Jogos de **{hoje.strftime('%d-%m-%Y')}**")

if df_hoje.empty:
    st.warning("Nenhum jogo previsto para hoje.")
else:
    # Exibir os dados com estilo
    st.dataframe(
        df_hoje.style
        .format({
            'H': '{:.2f}', 'D': '{:.2f}', 'A': '{:.2f}',
            'Diff_HT_P': '{:.2f}', 'Diff_Power': '{:.2f}', 'OU_Total': '{:.2f}'
        })
        .background_gradient(cmap='RdYlGn', subset=['Diff_HT_P', 'Diff_Power'])
        .background_gradient(cmap='Blues', subset=['OU_Total']),
        use_container_width=True
    )
