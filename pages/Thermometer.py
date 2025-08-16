import streamlit as st
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Today's Picks – Power Thermometer", layout="wide")
st.title("🔥 Today's Betting Thermometer")

# Legenda
st.markdown("""
**Legend – Diff_Power colors:**
- 🟩 **Green** → Higher values favor the **Home** team (stronger advantage).
- 🟥 **Red** → Lower values favor the **Away** team (stronger advantage).
- 🟨 **Yellow** → Values close to zero (-8 to +8) indicate balanced teams.
""")

# Pasta com jogos do dia e histórico
GAMES_FOLDER = "GamesDay"

# Margens de tolerância
MARGIN_DIFF_POWER = 10.00
MARGIN_DIFF_HTP = 10.00
MARGIN_ODDS = 0.50

# Ligas a excluir (não entram na análise)
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa"]

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

def load_last_csv(folder):
    """Carrega apenas o último arquivo CSV (mais recente) da pasta."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    latest_file = max(files)  # Assume formato com data no nome YYYY-MM-DD
    df = pd.read_csv(os.path.join(folder, latest_file))
    return df

def filter_excluded_leagues(df):
    """Remove ligas que contenham palavras-chave excluídas."""
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)]

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

def calculate_probability_and_count(history_df, diff_power, diff_htp, side, selected_odd):
    """Calcula a probabilidade de vitória e a quantidade de jogos na amostra."""
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
        return None, 0

    wins = win_mask[mask].sum()
    return round((wins / total) * 100, 1), total

def color_diff_power(val):
    """Colore o Diff_Power com degradê e zona neutra amarela."""
    if pd.isna(val):
        return ''
    if -8 <= val <= 8:
        intensity = 1 - (abs(val) / 8)
        return f'background-color: rgba(255, 255, 0, {0.3 + 0.4 * intensity})'
    if val > 0:
        intensity = min(1, val / 25)
        return f'background-color: rgba(0, 255, 0, {0.3 + 0.4 * intensity})'
    if val < 0:
        intensity = min(1, abs(val) / 25)
        return f'background-color: rgba(255, 0, 0, {0.3 + 0.4 * intensity})'

def color_probability(val):
    """Colore a Win_Probability com degradê de verde conforme aumenta a %."""
    if pd.isna(val):
        return ''
    intensity = min(1, val / 100)
    return f'background-color: rgba(0, 255, 0, {0.2 + 0.6 * intensity})'

# ------------------- Main -------------------
# Histórico: todos os arquivos (menos ligas excluídas)
all_games = load_all_games(GAMES_FOLDER)
all_games = filter_excluded_leagues(all_games)

history = prepare_historical(all_games)
if history.empty:
    st.warning("Nenhum jogo histórico válido encontrado (com resultados).")
    st.stop()

# Jogos do dia: apenas último arquivo (menos ligas excluídas)
games_today = load_last_csv(GAMES_FOLDER)
games_today = filter_excluded_leagues(games_today)
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# Determinar lado e odd
games_today['Side'] = games_today['Diff_Power'].apply(lambda x: "Home" if x > 0 else "Away")
games_today['Selected_Odd'] = games_today.apply(lambda row: row['Odd_H'] if row['Side'] == "Home" else row['Odd_A'], axis=1)

# Calcular probabilidade e quantidade de jogos
results = games_today.apply(
    lambda row: calculate_probability_and_count(
        history,
        row['Diff_Power'],
        row['Diff_HT_P'],
        row['Side'],
        row['Selected_Odd']
    ), axis=1
)

games_today['Win_Probability'] = [r[0] for r in results]
games_today['Games_Analyzed'] = [r[1] for r in results]

# Ordenar por probabilidade
games_today = games_today.sort_values(by='Win_Probability', ascending=False)

# Exibir tabela formatada
st.dataframe(
    games_today[['Date','Time,'League', 'Home', 'Away', 'Diff_Power', 'Diff_HT_P', 'Side', 'Selected_Odd', 'Games_Analyzed', 'Win_Probability']]
    .style
    .applymap(color_diff_power, subset=['Diff_Power'])
    .applymap(color_probability, subset=['Win_Probability'])
    .format({
        'Selected_Odd': '{:.2f}', 
        'Diff_Power': '{:.2f}', 
        'Diff_HT_P': '{:.2f}', 
        'Win_Probability': '{:.1f}%',
        'Games_Analyzed': '{:,.0f}'
    })
)
