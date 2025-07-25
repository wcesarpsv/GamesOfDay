import pandas as pd
import streamlit as st
from datetime import datetime
import os
import re

# 📁 Pasta onde estão os arquivos
PASTA_ARQUIVOS = r'C:\Users\flavia\FlashScore\PyCaret\Jogosdodia\JogosDia'

st.set_page_config(page_title="Previsões dos Jogos do Dia", layout="wide")
st.title("🔮 Previsões para os Jogos do Dia")

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
    st.error("❌ Nenhum arquivo CSV encontrado na pasta de jogos.")
    st.stop()

# 📅 Caixa de seleção com datas válidas
data_escolhida = st.selectbox("📅 Escolha uma data com jogos disponíveis:", datas_csv, index=len(datas_csv)-1)

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

    st.markdown(f"### 📆 Jogos de **{data_escolhida.strftime('%Y-%m-%d')}**")

    st.markdown(
        f"Summario dos jogos do dia:\n"
        f"- **Total de jogos:** {len(df_filtrado)}\n"
        f"- **Total de ligas:** {df_filtrado['League'].nunique()}\n"
        f"- **Explicando as Colunas:**\n"
        f"- **Diff_HT_P:** = Diferença da Força do time Home menos a força do time Away, considerando o power rating no Primeiro tempo. HT\n"
        f"- **Diff_Power:** = Diferença da Força do time Home menos a força do time Away, considerando o Power Rating Geral. FT\n"
        f"- **Cores:** Quanto mais <span style='color:green;'>verde</span>, maior a probabilidade de vitória do time da casa. Quanto mais <span style='color:red;'>vermelho</span>, maior a probabilidade de vitória do time visitante.\n"
        f"- **OU_Total:** = Total de gols esperado para o jogo, considerando o Power Rating dos times. Quanto mais azul, maior a expectativa de over 2.5 gols.\n",
        unsafe_allow_html=True
    )


    if df_filtrado.empty:
        st.warning("⚠️ Nenhum jogo encontrado para esta data no arquivo.")
    else:
        # ✅ Tabela com scroll e estilo
        st.dataframe(
            df_filtrado.style
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