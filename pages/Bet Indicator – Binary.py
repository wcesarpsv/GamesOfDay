# ########################################################
# BLOCO 1 – Imports & Configurações
# ########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import date, timedelta, datetime
from collections import Counter

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report

# SMOTE para balanceamento
from imblearn.over_sampling import SMOTE

# ---------------- Configurações da Página ----------------
st.set_page_config(page_title="Bet Indicator – Home vs Away", layout="wide")
st.title("📊 AI-Powered Bet Indicator – Home vs Away (Binary)")

# ---------------- Configs ----------------
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
MODELS_FOLDER = "Models"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa"]

os.makedirs(MODELS_FOLDER, exist_ok=True)


# ########################################################
# BLOCO 2 – Funções auxiliares
# ########################################################
def load_all_games(folder):
    """Carrega todos os CSVs da pasta e remove duplicados por (Date, Home, Away)."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    if not df_list:
        return pd.DataFrame()
    
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all.drop_duplicates(subset=["Date", "Home", "Away","Goals_H_FT","Goals_A_FT"], keep="first")

def filter_leagues(df):
    """Remove ligas indesejadas (Copa, UEFA, etc)."""
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()


# ########################################################
# BLOCO 3 – Carregando dados históricos
# ########################################################
st.info("📂 Loading historical data...")
all_games = filter_leagues(load_all_games(GAMES_FOLDER))
if all_games.empty:
    st.warning("No valid historical data found.")
    st.stop()

history = all_games.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
if history.empty:
    st.warning("No valid historical results found.")
    st.stop()


# ########################################################
# BLOCO 4 – NOVO: SELEÇÃO DE DATA PADRÃO + MERGE COM LIVESCORE
# ########################################################
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# Últimos dois arquivos (Hoje e Ontem) - igual ao código modelo
options = files[-2:] if len(files) >= 2 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Extrair a data do arquivo selecionado (YYYY-MM-DD)
date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
if date_match:
    selected_date_str = date_match.group(0)
else:
    selected_date_str = datetime.now().strftime("%Y-%m-%d")

# Carregar os jogos do dia selecionado
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# ========== MERGE COM LIVESCORE (IGUAL AO CÓDIGO MODELO) ==========
livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

# Ensure goal columns exist
if 'Goals_H_Today' not in games_today.columns:
    games_today['Goals_H_Today'] = np.nan
if 'Goals_A_Today' not in games_today.columns:
    games_today['Goals_A_Today'] = np.nan

# Merge with the correct LiveScore file
if os.path.exists(livescore_file):
    st.info(f"LiveScore file found: {livescore_file}")
    results_df = pd.read_csv(livescore_file)

    # FILTER OUT CANCELED AND POSTPONED GAMES
    results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
    
    required_cols = [
        'game_id', 'status', 'home_goal', 'away_goal',
        'home_ht_goal', 'away_ht_goal',
        'home_corners', 'away_corners',
        'home_yellow', 'away_yellow',
        'home_red', 'away_red'
    ]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if missing_cols:
        st.error(f"The file {livescore_file} is missing these columns: {missing_cols}")
    else:
        games_today = games_today.merge(
            results_df,
            left_on='Id',
            right_on='game_id',
            how='left',
            suffixes=('', '_RAW')
        )

        # Update goals only for finished games
        games_today['Goals_H_Today'] = games_today['home_goal']
        games_today['Goals_A_Today'] = games_today['away_goal']
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
        
        # ADD RED CARD COLUMNS
        games_today['Home_Red'] = games_today['home_red']
        games_today['Away_Red'] = games_today['away_red']
else:
    st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")

# 🔹 Mantém apenas jogos futuros (sem placares ainda) - baseado nos dados originais
if 'Goals_H_FT' in games_today.columns:
    games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

if games_today.empty:
    st.warning("No valid matches found for the selected day.")
    st.stop()


# ########################################################
# BLOCO 5 – Target binário e balanceamento inicial
# ########################################################
history = history[history['Goals_H_FT'] != history['Goals_A_FT']]  # remove draws
history['Target'] = history.apply(
    lambda row: 0 if row['Goals_H_FT'] > row['Goals_A_FT'] else 1,
    axis=1
)

# Ver distribuição de classes
class_counts = history['Target'].value_counts()
st.markdown("### ⚖️ Class Distribution (Home vs Away)")
st.write(pd.DataFrame({
    'Class': ['Home (0)', 'Away (1)'],
    'Count': [class_counts.get(0, 0), class_counts.get(1, 0)],
    'Percentage': [
        f"{class_counts.get(0, 0) / len(history) * 100:.1f}%",
        f"{class_counts.get(1, 0) / len(history) * 100:.1f}%"
    ]
}))


# ########################################################
# BLOCO 6 – Features básicas + Momentum
# ########################################################
history['Diff_M'] = history['M_H'] - history['M_A']
games_today['Diff_M'] = games_today['M_H'] - games_today['M_A']
history['Diff_Abs'] = (history['M_H'] - history['M_A']).abs()
games_today['Diff_Abs'] = (games_today['M_H'] - games_today['M_A']).abs()

def add_momentum_features(df):
    df['PesoMomentum_H'] = abs(df['M_H']) / (abs(df['M_H']) + abs(df['M_A']))
    df['PesoMomentum_A'] = abs(df['M_A']) / (abs(df['M_H']) + abs(df['M_A']))
    df['CustoMomentum_H'] = df.apply(
        lambda x: x['Odd_H'] / abs(x['M_H']) if abs(x['M_H']) > 0 else np.nan, axis=1
    )
    df['CustoMomentum_A'] = df.apply(
        lambda x: x['Odd_A'] / abs(x['M_A']) if abs(x['M_A']) > 0 else np.nan, axis=1
    )
    return df

history = add_momentum_features(history)
games_today = add_momentum_features(games_today)

base_features = [
    'Odd_H', 'Odd_D', 'Odd_A',
    'M_H', 'M_A', 'Diff_Power', 'Diff_M','Diff_Abs',
    'PesoMomentum_H', 'PesoMomentum_A',
    'CustoMomentum_H', 'CustoMomentum_A'
]


# ########################################################
# BLOCO 7 – One-hot Encoding + Duplicados tratados
# ########################################################
history_leagues = pd.get_dummies(history['League'], prefix="League")
games_today_leagues = pd.get_dummies(games_today['League'], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X = pd.concat([history[base_features], history_leagues], axis=1) \
        .fillna(0) \
        .join(history[["Date","Home","Away"]]) \
        .drop_duplicates(subset=["Date","Home","Away"], keep="first") \
        .drop(columns=["Date","Home","Away"])

y = history['Target']

X_today = pd.concat([games_today[base_features], games_today_leagues], axis=1) \
        .fillna(0) \
        .join(games_today[["Date","Home","Away"]]) \
        .drop_duplicates(subset=["Date","Home","Away"], keep="first") \
        .drop(columns=["Date","Home","Away"])

# ########################################################
# BLOCO 8 – Train / Validation + SMOTE
# ########################################################
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.info("Aplicando SMOTE para balancear as classes (Away)...")
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

st.write("Distribuição após SMOTE:", dict(Counter(y_train_res)))

scaler = StandardScaler()
X_train_scaled = X_train_res.copy()
X_val_scaled = X_val.copy()
X_today_scaled = X_today.copy()

X_train_scaled[base_features] = scaler.fit_transform(X_train_res[base_features].fillna(0))
X_val_scaled[base_features] = scaler.transform(X_val[base_features].fillna(0))
X_today_scaled[base_features] = scaler.transform(X_today[base_features].fillna(0))


# ########################################################
# BLOCO 9 – Treinando Modelos
# ########################################################
rf_tuned = RandomForestClassifier(
    n_estimators=500,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample"
)
rf_tuned.fit(X_train_res, y_train_res)

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
log_reg.fit(X_train_scaled, y_train_res)

model_choice = st.sidebar.radio(
    "Select Model",
    ("Random Forest (Tuned)", "Ensemble RF+Logistic"),
    index=0
)

# ########################################################
# BLOCO 10 – Validação e Métricas
# ########################################################
if model_choice == "Random Forest (Tuned)":
    preds = rf_tuned.predict(X_val)
    probs = rf_tuned.predict_proba(X_val)
elif model_choice == "Ensemble RF+Logistic":
    probs_rf = rf_tuned.predict_proba(X_val)
    probs_log = log_reg.predict_proba(X_val_scaled)
    probs = (0.7 * probs_rf) + (0.3 * probs_log)
    preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_val, preds)
ll = log_loss(y_val, probs)
bs = brier_score_loss(y_val, probs[:,1])

winrate_home = (preds[y_val==0] == 0).mean()
winrate_away = (preds[y_val==1] == 1).mean()

st.markdown("### 📊 Model Statistics (Validation)")
df_stats = pd.DataFrame([{
    "Model": model_choice,
    "Accuracy": f"{acc:.3f}",
    "LogLoss": f"{ll:.3f}",
    "Brier": f"{bs:.3f}",
    "Winrate_Home": f"{winrate_home:.2%}",
    "Winrate_Away": f"{winrate_away:.2%}"
}])
st.dataframe(df_stats, use_container_width=True)

report = classification_report(
    y_val, preds, target_names=["Home","Away"], output_dict=True
)
df_report = pd.DataFrame(report).transpose()
st.markdown("### 📑 Classification Report (Precision / Recall / F1)")
st.dataframe(df_report.style.format("{:.2f}"), use_container_width=True)

# ########################################################
# BLOCO 10.1 – ANÁLISE DAS FAIXAS DE PROBABILIDADE (HISTÓRICO)
# ########################################################

st.markdown("### 📈 Análise das Faixas de Probabilidade (Validação)")

# Usar as probabilidades do modelo escolhido
if model_choice == "Random Forest (Tuned)":
    probs_val = rf_tuned.predict_proba(X_val)
else:
    probs_rf_val = rf_tuned.predict_proba(X_val)
    probs_log_val = log_reg.predict_proba(X_val_scaled)
    probs_val = (0.7 * probs_rf_val) + (0.3 * probs_log_val)

# Probabilidade de Away para análise
prob_away_val = probs_val[:, 1]

# Criar faixas baseadas na sua observação visual
faixas = [
    (0, 0.25, "🟢 HIGH CONFIDENCE HOME"),
    (0.25, 0.35, "🟡 MEDIUM CONFIDENCE HOME"), 
    (0.35, 0.45, "⚪ LOW CONFIDENCE HOME"),
    (0.45, 0.55, "🔴 UNCERTAIN/AVOID"),
    (0.55, 0.65, "⚪ LOW CONFIDENCE AWAY"),
    (0.65, 0.75, "🟡 MEDIUM CONFIDENCE AWAY"),
    (0.75, 1.0, "🟢 HIGH CONFIDENCE AWAY")
]

resultados_faixas = []

for min_prob, max_prob, categoria in faixas:
    mask = (prob_away_val >= min_prob) & (prob_away_val < max_prob)
    
    if mask.sum() > 0:
        y_true_faixa = y_val[mask]
        y_pred_faixa = preds[mask]
        prob_faixa = prob_away_val[mask]
        
        # Estatísticas
        n_jogos = len(y_true_faixa)
        win_rate = accuracy_score(y_true_faixa, y_pred_faixa)
        home_wins = ((y_true_faixa == 0) & (y_pred_faixa == 0)).sum()
        away_wins = ((y_true_faixa == 1) & (y_pred_faixa == 1)).sum()
        
        # Calcular acurácia real vs esperada
        prob_media = prob_faixa.mean()
        if categoria in ["🟢 HIGH CONFIDENCE HOME", "🟡 MEDIUM CONFIDENCE HOME", "⚪ LOW CONFIDENCE HOME"]:
            acuracia_esperada = 1 - prob_media  # Probabilidade de Home vencer
            acuracia_real = home_wins / n_jogos if n_jogos > 0 else 0
        else:
            acuracia_esperada = prob_media  # Probabilidade de Away vencer
            acuracia_real = away_wins / n_jogos if n_jogos > 0 else 0
        
        resultados_faixas.append({
            'Faixa Away': f"{min_prob:.0%}-{max_prob:.0%}",
            'Categoria': categoria,
            'Jogos': n_jogos,
            'Win Rate': f"{win_rate:.1%}",
            'Home Wins': home_wins,
            'Away Wins': away_wins,
            'Prob Média': f"{prob_media:.1%}",
            'Acurácia Real': f"{acuracia_real:.1%}",
            'Acurácia Esperada': f"{acuracia_esperada:.1%}",
            'Performance': "✅ SUPERIOR" if acuracia_real > acuracia_esperada else "⚠️ INFERIOR"
        })

# DataFrame de resultados
df_faixas = pd.DataFrame(resultados_faixas)
st.dataframe(df_faixas, use_container_width=True)

# Gráfico de calibração
st.markdown("#### 📊 Gráfico de Calibração - Probabilidade Prevista vs Real")
calib_data = []
for i in range(0, 100, 5):
    min_p = i / 100
    max_p = (i + 5) / 100
    mask = (prob_away_val >= min_p) & (prob_away_val < max_p)
    
    if mask.sum() > 5:  # Mínimo de jogos para estatística
        y_true_calib = y_val[mask]
        away_win_rate = y_true_calib.mean()  # % de vitórias do Away
        prob_media_calib = prob_away_val[mask].mean()
        calib_data.append({
            'Probabilidade Prevista': prob_media_calib,
            'Win Rate Real': away_win_rate,
            'Jogos': mask.sum()
        })

df_calib = pd.DataFrame(calib_data)
if not df_calib.empty:
    st.line_chart(df_calib.set_index('Probabilidade Prevista')['Win Rate Real'])
    
    # Adicionar linha de referência perfeita
    st.caption("Linha de referência: quanto mais próxima da diagonal, melhor a calibração")

# ########################################################
# BLOCO 11 – Previsões para os jogos de hoje + Categorias Visuais
# ########################################################
if model_choice == "Random Forest (Tuned)":
    probs_today = rf_tuned.predict_proba(X_today)
else:
    probs_rf_today = rf_tuned.predict_proba(X_today)
    probs_log_today = log_reg.predict_proba(X_today_scaled)
    probs_today = (0.7 * probs_rf_today) + (0.3 * probs_log_today)

games_today['p_home'] = probs_today[:,0]
games_today['p_away'] = probs_today[:,1]

# ########################################################
# BLOCO 11.1 – CATEGORIAS VISUAIS PARA HOJE (SEM BACKGROUND)
# ########################################################

def categorizar_confianca(prob_away):
    """Categoriza a confiança baseado nas faixas analisadas"""
    if prob_away <= 0.25:
        return "🟢 HIGH CONFIDENCE HOME", "home_high"
    elif prob_away <= 0.35:
        return "🟡 MEDIUM CONFIDENCE HOME", "home_medium"
    elif prob_away <= 0.45:
        return "⚪ LOW CONFIDENCE HOME", "home_low"
    elif prob_away <= 0.55:
        return "🔴 UNCERTAIN/AVOID", "avoid"
    elif prob_away <= 0.65:
        return "⚪ LOW CONFIDENCE AWAY", "away_low"
    elif prob_away <= 0.75:
        return "🟡 MEDIUM CONFIDENCE AWAY", "away_medium"
    else:
        return "🟢 HIGH CONFIDENCE AWAY", "away_high"

# Aplicar categorias aos jogos de hoje
games_today['prob_away'] = probs_today[:, 1]
categorias = games_today['prob_away'].apply(categorizar_confianca)
games_today['Categoria'] = categorias.apply(lambda x: x[0])
games_today['Tipo_Confianca'] = categorias.apply(lambda x: x[1])

# Ordenar por confiança (melhores apostas primeiro)
ordem_confianca = {
    "🟢 HIGH CONFIDENCE HOME": 1,
    "🟢 HIGH CONFIDENCE AWAY": 2, 
    "🟡 MEDIUM CONFIDENCE HOME": 3,
    "🟡 MEDIUM CONFIDENCE AWAY": 4,
    "⚪ LOW CONFIDENCE HOME": 5,
    "⚪ LOW CONFIDENCE AWAY": 6,
    "🔴 UNCERTAIN/AVOID": 7
}
games_today['Ordem_Confianca'] = games_today['Categoria'].map(ordem_confianca)
games_today = games_today.sort_values('Ordem_Confianca')

# NOVAS COLUNAS PARA EXIBIÇÃO
cols_to_show_enhanced = [
    'Categoria', 'Date', 'Time', 'League', 'Home', 'Away',
    'Goals_H_Today', 'Goals_A_Today', 
    'Odd_H', 'Odd_A', 'prob_away', 'p_home', 'p_away'
]

# Exibir tabela categorizada - SEM BACKGROUND COLOR
st.markdown(f"### 🎯 Previsões Categorizadas para {selected_date_str}")

styled_enhanced = (
    games_today[cols_to_show_enhanced]
    .style.format({
        'Odd_H': '{:.2f}', 'Odd_A': '{:.2f}',
        'prob_away': '{:.1%}', 'p_home': '{:.1%}', 'p_away': '{:.1%}',
        'Goals_H_Today': '{:.0f}', 'Goals_A_Today': '{:.0f}'
    }, na_rep='—')
)

st.dataframe(styled_enhanced, use_container_width=True, height=1000)

# Resumo por categoria
st.markdown("#### 📋 Resumo por Categoria de Confiança")
resumo_categorias = games_today['Categoria'].value_counts().reset_index()
resumo_categorias.columns = ['Categoria', 'Quantidade']
st.dataframe(resumo_categorias, use_container_width=True)

# 🔹 Botão para download do CSV
import io
csv_buffer = io.BytesIO()
games_today.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
csv_buffer.seek(0)

st.download_button(
    label="📥 Download Predictions CSV",
    data=csv_buffer,
    file_name=f"Bet_Indicator_Binary_{selected_date_str}.csv",
    mime="text/csv"
)

# ########################################################
# BLOCO 12 – RECOMENDAÇÕES BASEADAS NA ANÁLISE
# ########################################################
st.markdown("### 💡 Recomendações Baseadas na Análise")

high_confidence_games = games_today[
    games_today['Categoria'].isin(["🟢 HIGH CONFIDENCE HOME", "🟢 HIGH CONFIDENCE AWAY"])
]

medium_confidence_games = games_today[
    games_today['Categoria'].isin(["🟡 MEDIUM CONFIDENCE HOME", "🟡 MEDIUM CONFIDENCE AWAY"])
]

if not high_confidence_games.empty:
    st.success(f"🎯 **MELHORES OPORTUNIDADES (Alta Confiança)**: {len(high_confidence_games)} jogos")
    
    for _, jogo in high_confidence_games.iterrows():
        if "HOME" in jogo['Categoria']:
            st.write(f"✅ **{jogo['Home']} vs {jogo['Away']}**")
            st.write(f"   🏠 **Home Win** | Odd: {jogo['Odd_H']:.2f} | Prob: {jogo['p_home']:.1%}")
        else:
            st.write(f"✅ **{jogo['Home']} vs {jogo['Away']}**")
            st.write(f"   ✈️ **Away Win** | Odd: {jogo['Odd_A']:.2f} | Prob: {jogo['p_away']:.1%}")
        st.write("---")

if not medium_confidence_games.empty:
    st.info(f"📊 **OPORTUNIDADES SECUNDÁRIAS (Média Confiança)**: {len(medium_confidence_games)} jogos")
    
    for _, jogo in medium_confidence_games.iterrows():
        if "HOME" in jogo['Categoria']:
            st.write(f"⚡ **{jogo['Home']} vs {jogo['Away']}**")
            st.write(f"   🏠 Home Win | Odd: {jogo['Odd_H']:.2f} | Prob: {jogo['p_home']:.1%}")
        else:
            st.write(f"⚡ **{jogo['Home']} vs {jogo['Away']}**")
            st.write(f"   ✈️ Away Win | Odd: {jogo['Odd_A']:.2f} | Prob: {jogo['p_away']:.1%}")

if not high_confidence_games.empty and not medium_confidence_games.empty:
    st.write("")
elif not high_confidence_games.empty:
    st.write("")
else:
    st.warning("⚠️ **ATENÇÃO**: Nenhum jogo com alta confiança identificado hoje. Considere as apostas de média confiança ou evite apostar.")

st.markdown("""
**🎯 LEGENDA DAS CATEGORIAS:**

- 🟢 **HIGH CONFIDENCE**: Melhores oportunidades (win rate histórico > 60%)
- 🟡 **MEDIUM CONFIDENCE**: Boas oportunidades (win rate histórico 55-60%)  
- ⚪ **LOW CONFIDENCE**: Oportunidades limitadas (win rate histórico 50-55%)
- 🔴 **UNCERTAIN/AVOID**: Evitar apostas (win rate histórico < 50%)
""")
