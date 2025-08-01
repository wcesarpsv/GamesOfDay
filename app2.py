import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

# 📁 Pasta onde estão os arquivos
PASTA_ARQUIVOS = "GamesDay"

st.set_page_config(page_title="Data-Driven Football Insights", layout="wide")
st.title("🔮 Data-Driven Football Insights")

# 🧠 Função auxiliar para extrair datas disponíveis dos arquivos
def datas_disponiveis(pasta):
    padrao = r'Jogosdodia_(\d{4}-\d{2}-\d{2})\.csv'
    datas = []
    for nome in os.listdir(pasta):
        match = re.match(padrao, nome)
        if match:
            try:
                datas.append(datetime.strptime(match.group(1), '%Y-%m-%d').date())
            except:
                continue
    return sorted(datas)

# 🔍 Buscar as datas disponíveis
datas_csv = datas_disponiveis(PASTA_ARQUIVOS)

if not datas_csv:
    st.error("❌ No CSV files found in games folder.")
    st.stop()

# 📅 Caixa de seleção com datas válidas
data_escolhida = st.selectbox("📅 Choose a date with games available:", datas_csv, index=len(datas_csv)-1)

# 🛠️ Montar o caminho do arquivo
nome_arquivo = f'Jogosdodia_{data_escolhida}.csv'
caminho_arquivo = os.path.join(PASTA_ARQUIVOS, nome_arquivo)

try:
    # 📥 Carregar o CSV
    df = pd.read_csv(caminho_arquivo)

    # 🧹 Limpeza de colunas inúteis
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    # 📆 Garantir tipo de data
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.date
    df_filtrado = df[df['Date'] == data_escolhida]
    df_visual = df_filtrado.drop(columns=['Date'])  # Oculta a coluna 'Date'
    df_visual.index = range(len(df_visual))         # Remove o índice original

    st.markdown(f"### 📆 Jogos de **{data_escolhida.strftime('%Y-%m-%d')}**")

    st.markdown(f"""
### 📊 Matchday Summary – *{data_escolhida.strftime('%Y-%m-%d')}*

- **Total matches:** {len(df_filtrado)}
- **Total leagues:** {df_filtrado['League'].nunique()}

---

### ℹ️ Column Descriptions:

- **`Diff_HT_P`** – Difference in team strength for the **first half**, based on Power Ratings  
- **`Diff_Power`** – Overall team strength difference for the full match (FT)  
- **`OU_Total`** – Expected total goals for the match (higher = greater chance of Over 2.5)

---

### 🎨 Color Guide:

- 🟩 **Green**: Advantage for **home team**  
- 🟥 **Red**: Advantage for **away team**  
- 🔵 **Blue**: Higher expected total goals
""")


    if df_filtrado.empty:
        st.warning("⚠️ Nenhum jogo encontrado para esta data no arquivo.")
    else:
        # ✅ Tabela com scroll e estilo
        st.dataframe(
            df_visual.style
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
    st.error(f"❌ Arquivo `{nome_arquivo}` não encontrado.")
except pd.errors.EmptyDataError:
    st.error(f"❌ O arquivo `{nome_arquivo}` está vazio ou não contém dados válidos.")
