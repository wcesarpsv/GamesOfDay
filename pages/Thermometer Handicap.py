########################################
######## Bloco 6 – Load Data ###########
########################################

files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select matchday file:", options, index=len(options)-1)

# Carrega jogos do dia
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Mantém apenas jogos futuros (sem resultado ainda)
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

# Carrega histórico
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
history = prepare_history(all_games)
if history.empty:
    st.warning("No valid historical data found.")
    st.stop()

# Calcula variação de ligas e bands por quantis
league_class = classify_leagues_variation(history)
league_bands = compute_league_bands(history)

# Cria coluna de diferença
games_today['M_Diff'] = games_today['M_H'] - games_today['M_A']

# Faz os merges (para trazer quantis)
games_today = games_today.merge(league_class, on='League', how='left')
games_today = games_today.merge(league_bands, on='League', how='left')

# ==== Criar bandas textuais nos jogos do dia ====
games_today['Home_Band'] = np.where(
    games_today['M_H'] <= games_today['Home_P20'], 'Bottom 20%',
    np.where(games_today['M_H'] >= games_today['Home_P80'], 'Top 20%', 'Balanced')
)
games_today['Away_Band'] = np.where(
    games_today['M_A'] <= games_today['Away_P20'], 'Bottom 20%',
    np.where(games_today['M_A'] >= games_today['Away_P80'], 'Top 20%', 'Balanced')
)

# ==== Mapear bandas para valores numéricos (1,2,3) ====
BAND_MAP = {"Bottom 20%": 1, "Balanced": 2, "Top 20%": 3}
games_today["Home_Band_Num"] = games_today["Home_Band"].map(BAND_MAP)
games_today["Away_Band_Num"] = games_today["Away_Band"].map(BAND_MAP)

# ==== Criar bandas também no histórico ====
history = history.merge(league_bands, on="League", how="left")

history['Home_Band'] = np.where(
    history['M_H'] <= history['Home_P20'], 'Bottom 20%',
    np.where(history['M_H'] >= history['Home_P80'], 'Top 20%', 'Balanced')
)
history['Away_Band'] = np.where(
    history['M_A'] <= history['Away_P20'], 'Bottom 20%',
    np.where(history['M_A'] >= history['Away_P80'], 'Top 20%', 'Balanced')
)
history["Home_Band_Num"] = history["Home_Band"].map(BAND_MAP)
history["Away_Band_Num"] = history["Away_Band"].map(BAND_MAP)

# ==== Continuar pipeline normal ====
games_today['Dominant'] = games_today.apply(dominant_side, axis=1)

# Aplica recomendação + métricas
recs = games_today.apply(lambda r: auto_recommendation_dynamic_winrate(r, history), axis=1)
games_today["Auto_Recommendation"] = [x[0] for x in recs]
games_today["Win_Probability"] = [x[1] for x in recs]
games_today["EV"] = [x[2] for x in recs]
games_today["Games_Analyzed"] = [x[3] for x in recs]
