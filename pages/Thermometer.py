import streamlit as st
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Today's Picks – Power Thermometer", layout="wide")
st.title("🔥 Today's Betting Thermometer")

# Pasta com jogos do dia e histórico
GAMES_FOLDER = "GamesDay"

# Margens de tolerância
MARGIN_DIFF_POWER = 10.00
MARGIN_DIFF_HTP = 10.00
MARGIN_ODDS = 0.40

# ------------------- Funções auxiliares -------------------
def load_all_games(folder):
    """Carrega todos os CSVs da pasta e concatena em um DataFrame."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    df_list = []
    for file in files:
        path = os.path.join(folder, file)
        try:
            df = pd.read_csv(path)
            df_list.append(df)
        except Exception as e:
            st.error(f"Erro ao ler {file}: {e}")
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def prepare_historical(df):
    """Filtra apenas jogos com resultado e colunas necessárias."""
    required_cols = ['Goals_H_FT', 'Goals_A_FT', 'Diff_Power', 'Diff_HT_P', 'Odd_H', 'Odd_A']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Coluna necessária ausente no histórico: {col}")
            return pd.DataFrame()

    # Apenas jogos com resultados válidos
    df = df.dropna(subset=['Goals_H_FT', 'Goals_A_FT'])
    return df

def calculate_probability(history_df, diff_power, diff_htp, side, selected_odd):
    """Calcula a probabilidade de vitória com base no histórico."""
    if side == "Home":
        odd_col = 'Odd_H'
        win_mask = history_df['Goals_H_FT'] > history_df['Goals_A_FT']
    else:
        odd_col = 'Odd_A'
        win_mask = history_df['Goals_A_FT'] > history_df['Goals_H_FT']

    # Filtro com margens definidas
    mask = (
        (history_df['Diff_Power'].between(diff_power - MARGIN_DIFF_POWER, diff_power + MARGIN_DIFF_POWER)) &
        (history_df['Diff_HT_P'].between(diff_htp - MARGIN_DIFF_HTP, diff_htp + MARGIN_DIFF_HTP)) &
        (history_df[odd_col].between(selected_odd - MARGIN_ODDS, selected_odd + MARGIN_ODDS))
    )

    filtered = history_df[mask]
    total = len(filtered)
    if total == 0:
        return None

    wins = win_mask[mask].sum()
    return round((wins / total) * 100, 1)

def color_diff_power(val):
    """Colore o Diff_Power como termômetro."""
    if pd.isna(val):
        return ''
    if val > 0:
        return f'background-color: rgba(0, 255, 0, {min(1, val/5)})'
    else:
        return f'background-color: rgba(255, 0, 0, {min(1, abs(val)/5)})'

# ------------------- Main -------------------
# Carregar dados
all_games = load_all_games(GAMES_FOLDER)
if all_games.empty:
    st.warning("Nenhum jogo encontrado na pasta GamesDay.")
    st.stop()

# Histórico
history = prepare_historical(all_games)
if history.empty:
    st.warning("Nenhum jogo histórico válido encontrado (com resultados).")
    st.stop()

# Jogos do dia (sem resultado)
games_today = all_games[all_games['Goals_H_FT'].isna()].copy()

# Determinar lado e odd
games_today['Side'] = games_today['Diff_Power'].apply(lambda x: "Home" if x > 0 else "Away")
games_today['Selected_Odd'] = games_today.apply(lambda row: row['Odd_H'] if row['Side'] == "Home" else row['Odd_A'], axis=1)

# Calcular probabilidade
games_today['Win_Probability'] = games_today.apply(
    lambda row: calculate_probability(
        history,
        row['Diff_Power'],
        row['Diff_HT_P'],
        row['Side'],
        row['Selected_Odd']
    ), axis=1
)

# Ordenar por probabilidade
games_today = games_today.sort_values(by='Win_Probability', ascending=False)

# Exibir tabela formatada
st.dataframe(
    games_today[['Date', 'League', 'Home', 'Away', 'Diff_Power', 'Diff_HT_P', 'Side', 'Selected_Odd', 'Win_Probability']]
    .style.applymap(color_diff_power, subset=['Diff_Power'])
    .format({'Selected_Odd': '{:.2f}', 'Diff_Power': '{:.2f}', 'Diff_HT_P': '{:.2f}', 'Win_Probability': '{:.1f}%'})
)
