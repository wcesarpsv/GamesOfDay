##################### BLOCO 1 – IMPORTS & CONFIG #####################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (Home vs Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa","trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


##################### BLOCO 2 – HELPERS #####################
def preprocess_df(df):
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, feature_cols, filename):
    with open(os.path.join(MODELS_FOLDER, filename), "wb") as f:
        joblib.dump((model, feature_cols), f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None


##################### BLOCO 3 – LOAD DATA + HANDICAP TARGET #####################
st.info("📂 Loading data...")

# ========== SELEÇÃO DE DATA ==========
files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
files = sorted(files)

if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

# Últimos dois arquivos (Hoje e Ontem)
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

# ========== MERGE COM LIVESCORE ==========
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

# ========== CARREGAR HISTÓRICO ==========
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

if set(["Date", "Home", "Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home", "Away","Goals_H_FT", "Goals_A_FT"], keep="first")
else:
    history = history.drop_duplicates(keep="first")

if history.empty:
    st.stop()

# Filtrar apenas jogos sem placar final (para previsão)
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.warning("⚠️ No matches found for today (or yesterday, if selected).")
    st.stop()

def convert_asian_line(line_str):
    try:
        if pd.isna(line_str) or line_str == "":
            return None
        line_str = str(line_str).strip()
        if "/" not in line_str:
            return float(line_str)
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
    except:
        return None

history["Asian_Line_Display"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Display"] = games_today["Asian_Line"].apply(convert_asian_line)

def calc_handicap_result(margin, asian_line_str, invert=False):
    if pd.isna(asian_line_str):
        return np.nan
    if invert:
        margin = -margin
    try:
        parts = [float(x) for x in str(asian_line_str).split('/')]
    except:
        return np.nan
    results = []
    for line in parts:
        if margin > line:
            results.append(1.0)
        elif margin == line:
            results.append(0.5)
        else:
            results.append(0.0)
    return np.mean(results)

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Handicap_Home_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1)
history["Handicap_Away_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1)

history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)


##################### BLOCO 4 – FEATURE ENGINEERING OTIMIZADO #####################

# PRIMEIRO definir a lista de features
away_premium_features = [
    'Underdog_Indicator',      # Correlação 0.261 ✅
    'Handicap_Balance',        # Correlação -0.261 ✅
    'Aggression_Away',         # Correlação 0.209 ✅
    'Aggression_Home',         # Correlação -0.190 ✅
    'Odd_A',                   # Contexto de odds
    'Asian_Line_Display',      # Linha do handicap
    'Odds_Ratio',              # Relação de forças
    'Line_Abs'                 # Magnitude do handicap
]

def create_optimized_features(df):
    """
    Feature engineering focado nas variáveis com correlação comprovada
    """
    df = df.copy()
    
    # ✅ FEATURES COM CORRELAÇÃO FORTE (Foco Away)
    # Verificar se as colunas existem antes de criar features
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away']):
        df['Underdog_Indicator'] = df['Aggression_Away'] - df['Aggression_Home']
        df['Handicap_Balance'] = df['Aggression_Home'] - df['Aggression_Away']
        st.write("✅ Features de Aggression criadas")
    else:
        st.warning("⚠️ Colunas de Aggression não encontradas")
    
    # ✅ FEATURES COMPLEMENTARES
    if all(col in df.columns for col in ['Odd_H', 'Odd_A']):
        df['Odds_Ratio'] = df['Odd_A'] / df['Odd_H']
        st.write("✅ Odds_Ratio criada")
    
    # ✅ FEATURE DE LINHA (importante para handicap)
    if 'Asian_Line_Display' in df.columns:
        df['Line_Abs'] = abs(df['Asian_Line_Display'])
        st.write("✅ Line_Abs criada")
    else:
        st.warning("⚠️ Asian_Line_Display não encontrado")
    
    return df

# Aplicar feature engineering otimizado
st.info("🔄 Aplicando feature engineering otimizado...")

history = create_optimized_features(history)
games_today = create_optimized_features(games_today)

# AGORA filtrar apenas features que existem
away_premium_features = [f for f in away_premium_features if f in history.columns]

st.success(f"✅ Features premium para Away Handicap: {away_premium_features}")

if not away_premium_features:
    st.error("❌ Nenhuma feature premium disponível!")
    st.stop()

##################### BLOCO 4.1 – PREPARAR DADOS PARA MODELO AWAY PREMIUM #####################

# Garantir que temos o target para Away
if "Target_AH_Away" not in history.columns:
    st.error("Target_AH_Away não encontrado no histórico!")
    st.stop()

# VERIFICAÇÃO ROBUSTA DAS FEATURES
st.info("🔍 Verificando disponibilidade das features...")

# Mostrar todas as colunas disponíveis para debug
st.write("📋 **Colunas disponíveis no histórico:**", list(history.columns))
st.write("📋 **Colunas disponíveis hoje:**", list(games_today.columns))

# Features que realmente existem no histórico
available_away_features = [f for f in away_premium_features if f in history.columns]
st.write(f"✅ Features disponíveis no histórico: {available_away_features}")

# Features que existem nos dados de hoje
available_today_features = [f for f in away_premium_features if f in games_today.columns]
st.write(f"✅ Features disponíveis hoje: {available_today_features}")

# Usar apenas as features que existem em AMBOS
final_away_features = [f for f in available_away_features if f in available_today_features]
st.success(f"🎯 Features finais para o modelo: {final_away_features}")

if not final_away_features:
    st.error("❌ Nenhuma feature disponível em ambos histórico e dados de hoje!")
    
    # Mostrar quais features estão faltando
    missing_in_today = [f for f in available_away_features if f not in available_today_features]
    if missing_in_today:
        st.write(f"❌ Features faltando em games_today: {missing_in_today}")
    
    # Tentar uma abordagem alternativa com features básicas
    st.info("🔄 Tentando com features básicas...")
    basic_features = ['Odd_A', 'Asian_Line_Display']
    final_away_features = [f for f in basic_features if f in history.columns and f in games_today.columns]
    
    if final_away_features:
        st.success(f"🎯 Usando features básicas: {final_away_features}")
    else:
        st.stop()

# VERIFICAÇÃO FINAL - garantir que todas as features existem
missing_in_history = [f for f in final_away_features if f not in history.columns]
missing_in_today = [f for f in final_away_features if f not in games_today.columns]

if missing_in_history:
    st.error(f"❌ Features faltando no histórico: {missing_in_history}")
    final_away_features = [f for f in final_away_features if f not in missing_in_history]

if missing_in_today:
    st.error(f"❌ Features faltando em games_today: {missing_in_today}")
    final_away_features = [f for f in final_away_features if f not in missing_in_today]

if not final_away_features:
    st.error("❌ Nenhuma feature disponível após verificação!")
    st.stop()

st.success(f"✅ Features confirmadas: {final_away_features}")

# Preparar matriz de features para Away
try:
    X_away = history[final_away_features].copy()
    y_away = history["Target_AH_Away"].copy()
    st.success("✅ Dados históricos preparados com sucesso")
except KeyError as e:
    st.error(f"❌ Erro ao preparar dados históricos: {e}")
    st.stop()

# One-hot encoding para ligas
try:
    history_leagues = pd.get_dummies(history["League"], prefix="League")
    games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
    
    # Garantir que as colunas de liga são as mesmas
    common_league_cols = list(set(history_leagues.columns) & set(games_today_leagues.columns))
    if not common_league_cols:
        st.warning("⚠️ Nenhuma liga comum entre histórico e dados de hoje")
        history_leagues = pd.DataFrame()
        games_today_leagues = pd.DataFrame()
    else:
        games_today_leagues = games_today_leagues.reindex(columns=common_league_cols, fill_value=0)
        history_leagues = history_leagues[common_league_cols]
        st.success(f"✅ {len(common_league_cols)} ligas comuns encontradas")
        
except Exception as e:
    st.warning(f"⚠️ Erro no encoding de ligas: {e}")
    history_leagues = pd.DataFrame()
    games_today_leagues = pd.DataFrame()

# Adicionar ligas às features (se existirem)
if not history_leagues.empty:
    X_away = pd.concat([X_away, history_leagues], axis=1)
    league_cols = history_leagues.columns.tolist()
    final_away_features.extend(league_cols)
    st.success(f"✅ Adicionadas {len(league_cols)} colunas de liga")

# Preparar dados de hoje com VERIFICAÇÃO
try:
    X_today_away = games_today[final_away_features].copy()
    
    # Adicionar ligas se existirem
    if not games_today_leagues.empty:
        X_today_away = pd.concat([X_today_away, games_today_leagues], axis=1)
    
    # Garantir que as colunas são as mesmas
    X_today_away = X_today_away.reindex(columns=X_away.columns, fill_value=0)
    st.success("✅ Dados de hoje preparados com sucesso")
    
except KeyError as e:
    st.error(f"❌ Erro ao preparar dados de hoje: {e}")
    
    # Tentar alternativa: usar apenas as colunas que existem
    existing_cols = [col for col in X_away.columns if col in games_today.columns]
    if existing_cols:
        st.info(f"🔄 Usando apenas {len(existing_cols)} colunas disponíveis")
        X_today_away = games_today[existing_cols].copy()
        X_today_away = X_today_away.reindex(columns=X_away.columns, fill_value=0)
    else:
        st.error("❌ Nenhuma coluna comum disponível")
        st.stop()

# Normalização das features numéricas
numeric_away_features = [f for f in final_away_features if f in X_away.columns and 
                        X_away[f].dtype in ['float64', 'int64'] and 
                        not f.startswith('League_')]

st.info(f"🔢 Features numéricas para normalização: {numeric_away_features}")

# Mostrar estatísticas das features
st.write("📊 Estatísticas das features no histórico:")
st.dataframe(X_away.describe(), use_container_width=True)

# Mostrar shape dos dados
st.write(f"📐 Shape X_away: {X_away.shape}")
st.write(f"📐 Shape X_today_away: {X_today_away.shape}")
st.write(f"📐 Shape y_away: {y_away.shape}")

##################### BLOCO 5 – MODELO AWAY PREMIUM #####################

def train_away_premium_model(X, y, retrain=False):
    """
    Modelo especializado para Away Handicap
    """
    filename = f"AsianHandicap_Away_Premium_XGB_v2.pkl"
    
    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, feature_cols = loaded
            st.success("✅ Modelo Away Premium carregado do cache")
            return model, feature_cols
    
    # Split temporal (mais realista)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Normalização
    if normalize_features and numeric_away_features:
        scaler = StandardScaler()
        X_train[numeric_away_features] = scaler.fit_transform(X_train[numeric_away_features])
        X_test[numeric_away_features] = scaler.transform(X_test[numeric_away_features])
        
        # Salvar scaler para uso futuro
        joblib.dump(scaler, os.path.join(MODELS_FOLDER, "away_premium_scaler.pkl"))
    
    # Modelo XGBoost otimizado para Away
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Balanceamento
    )
    
    # Treinar com early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Avaliação
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    st.write("📊 **Performance do Modelo Away Premium:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Acurácia", f"{accuracy_score(y_test, preds):.1%}")
    with col2:
        st.metric("Log Loss", f"{log_loss(y_test, probs):.3f}")
    with col3:
        st.metric("Brier Score", f"{brier_score_loss(y_test, probs[:,1]):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.write("🎯 **Top 10 Features Mais Importantes:**")
    st.dataframe(feature_importance.head(10), use_container_width=True)
    
    # Salvar modelo
    save_model(model, X.columns.tolist(), filename)
    st.success("✅ Modelo Away Premium treinado e salvo")
    
    return model, X.columns.tolist()

##################### BLOCO 6 – SISTEMA DE CONFIANÇA INTEGRADO #####################

def calculate_confidence_system(row, prob_away):
    """
    Sistema de confiança integrado para Away + oportunidades Home
    """
    # 1. CONFIANÇA AWAY HANDICAP
    away_confidence = "BAIXA"
    away_reason = []
    
    # Usar get() com valores padrão para evitar KeyError
    underdog_indicator = row.get('Underdog_Indicator', 0)
    aggression_away = row.get('Aggression_Away', 0)
    aggression_home = row.get('Aggression_Home', 0)
    
    if (underdog_indicator > 0.3 and 
        prob_away > 0.60 and
        aggression_away > 0.2):
        away_confidence = "ALTA"
        away_reason.append("Underdog claro com aggression away alta")
    
    elif (underdog_indicator > 0.15 and 
          prob_away > 0.55 and
          aggression_away > 0.1):
        away_confidence = "MÉDIA"
        away_reason.append("Underdog moderado com probabilidade boa")
    
    else:
        away_confidence = "BAIXA"
        away_reason.append("Indicadores fracos para Away")
    
    # 2. OPORTUNIDADE HOME HANDICAP (baseado na fraqueza do Away)
    home_opportunity = "FRACA"
    home_reason = []
    
    if (prob_away < 0.35 and 
        underdog_indicator < -0.2 and
        aggression_home > aggression_away):
        home_opportunity = "FORTE"
        home_reason.append("Away muito fraco e Home favorito claro")
    
    elif (prob_away < 0.45 and 
          underdog_indicator < 0):
        home_opportunity = "MODERADA"
        home_reason.append("Away fraco e Home não é underdog")
    
    else:
        home_opportunity = "FRACA"
        home_reason.append("Away não está suficientemente fraco")
    
    return {
        'away_confidence': away_confidence,
        'away_reason': " | ".join(away_reason),
        'home_opportunity': home_opportunity,
        'home_reason': " | ".join(home_reason)
    }

##################### BLOCO 7 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features (odds + strength + aggression)", value=True)

# NOVA CONFIGURAÇÃO: Estratégia
strategy_choice = st.sidebar.selectbox(
    "🎯 Estratégia", 
    ["Away Premium + Home Opportunities", "Original Both Sides"]
)

##################### BLOCO 8 – APLICAÇÃO DO MODELO AWAY PREMIUM #####################

if strategy_choice == "Away Premium + Home Opportunities":
    st.info("🤖 Treinando modelo Away Premium...")
    away_model, away_feature_cols = train_away_premium_model(X_away, y_away, retrain)

    # Aplicar modelo aos jogos de hoje
    if not games_today.empty:
        # Garantir que temos as mesmas features
        X_today_away = X_today_away.reindex(columns=away_feature_cols, fill_value=0)
        
        # Normalizar dados de hoje
        if normalize_features and numeric_away_features:
            scaler_path = os.path.join(MODELS_FOLDER, "away_premium_scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                X_today_away[numeric_away_features] = scaler.transform(X_today_away[numeric_away_features])
        
        # Fazer previsões
        away_probs = away_model.predict_proba(X_today_away)
        games_today['p_ah_away_yes'] = away_probs[:, 1]  # Probabilidade classe 1 (win)
        
        # Aplicar sistema de confiança
        confidence_data = []
        for idx, row in games_today.iterrows():
            prob_away = row['p_ah_away_yes']
            confidence_info = calculate_confidence_system(row, prob_away)
            confidence_info['prob_away'] = prob_away
            confidence_info['prob_home'] = 1 - prob_away  # Oportunidade indireta
            confidence_data.append(confidence_info)
        
        # Adicionar dados de confiança ao dataframe
        confidence_df = pd.DataFrame(confidence_data)
        games_today = pd.concat([games_today, confidence_df], axis=1)
        
        # Calcular stakes recomendados
        games_today['stake_away'] = games_today['away_confidence'].apply(
            lambda x: calculate_stake_recommendation(x, 100)
        )
        games_today['stake_home'] = games_today['home_opportunity'].apply(
            lambda x: calculate_stake_recommendation(x, 100)
        )

    ##################### BLOCO 9 – VISUALIZAÇÃO DOS RESULTADOS PREMIUM #####################

    st.markdown(f"## 🎯 PREVISÕES AWAY HANDICAP + OPORTUNIDADES HOME - {selected_date_str}")

    # Função para colorir basedo na confiança
    def color_confidence(val):
        if val == 'ALTA' or val == 'FORTE':
            return 'background-color: #4CAF50; color: white; font-weight: bold;'
        elif val == 'MÉDIA' or val == 'MODERADA':
            return 'background-color: #FF9800; color: white; font-weight: bold;'
        else:
            return 'background-color: #F44336; color: white;'

    # DataFrame final otimizado
    display_df = games_today[['Home', 'Away', 'League', 'Asian_Line_Display']].copy()
    display_df['Match'] = display_df['Home'] + ' vs ' + display_df['Away']

    # Adicionar colunas calculadas
    display_df['Prob Away HC'] = games_today['p_ah_away_yes']
    display_df['Confiança Away'] = games_today['away_confidence']
    display_df['Stake Away'] = games_today['stake_away']
    display_df['Prob Home HC'] = games_today['prob_home']
    display_df['Oportunidade Home'] = games_today['home_opportunity']
    display_df['Stake Home'] = games_today['stake_home']

    # Ordenar por confiança Away (mais altos primeiro)
    display_df = display_df.sort_values(['Stake Away', 'Stake Home'], ascending=[False, False])

    # Exibir resultados
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📊 Previsões Detalhadas")
        
        # Formatação da tabela
        styled_df = display_df[[
            'Match', 'League', 'Asian_Line_Display',
            'Prob Away HC', 'Confiança Away', 'Stake Away',
            'Prob Home HC', 'Oportunidade Home', 'Stake Home'
        ]].style.format({
            'Prob Away HC': '{:.1%}',
            'Prob Home HC': '{:.1%}',
            'Asian_Line_Display': '{:.2f}',
            'Stake Away': 'R$ {:.0f}',
            'Stake Home': 'R$ {:.0f}'
        }).applymap(color_confidence, subset=['Confiança Away', 'Oportunidade Home'])
        
        st.dataframe(styled_df, use_container_width=True, height=600)

    with col2:
        st.markdown("### 🎯 Resumo de Oportunidades")
        
        # Estatísticas rápidas
        high_away = len(display_df[display_df['Confiança Away'] == 'ALTA'])
        strong_home = len(display_df[display_df['Oportunidade Home'] == 'FORTE'])
        
        st.metric("🎯 Away Alta Confiança", high_away)
        st.metric("🏠 Home Oportunidades Fortes", strong_home)
        st.metric("📈 Total de Jogos", len(display_df))
        
        # Stake total recomendado
        total_stake = display_df['Stake Away'].sum() + display_df['Stake Home'].sum()
        st.metric("💰 Stake Total Recomendado", f"R$ {total_stake:.0f}")
        
        # Top oportunidades
        st.markdown("#### 🔥 Melhores Oportunidades Away")
        top_away = display_df[display_df['Confiança Away'] == 'ALTA'][['Match', 'Prob Away HC']].head(3)
        for _, match in top_away.iterrows():
            st.write(f"**{match['Match']}** - {match['Prob Away HC']:.1%}")
        
        st.markdown("#### 🏠 Melhores Oportunidades Home")
        top_home = display_df[display_df['Oportunidade Home'] == 'FORTE'][['Match', 'Prob Home HC']].head(3)
        for _, match in top_home.iterrows():
            st.write(f"**{match['Match']}** - {match['Prob Home HC']:.1%}")

    ##################### BLOCO 10 – DOWNLOAD DOS RESULTADOS #####################

    # Opção para download
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download das Previsões",
        data=csv,
        file_name=f"previsoes_away_home_{selected_date_str}.csv",
        mime="text/csv"
    )

    st.success("✅ Sistema Away Premium executado com sucesso!")

else:
    ##################### BLOCO 11 – CÓDIGO ORIGINAL (AMBOS OS LADOS) #####################
    
    # [SEU CÓDIGO ORIGINAL AQUI - Blocos 4 a 8 do seu código inicial]
    # ... (manter todo o código original para a estratégia "Original Both Sides")

    st.info("🔁 Executando estratégia original (ambos os lados)...")
    
    # [TODO: Inserir aqui os blocos 4 a 8 do seu código original]
    # Por questões de espaço, mantive apenas a estrutura
    # Você pode copiar e colar seus blocos originais aqui

    st.warning("⚠️ Executando modo original - insira seus blocos 4-8 aqui")

# BLOCO EXTRA – ANÁLISE DAS CORRELAÇÕES (mantido para referência)
st.markdown("### 🔍 Análise: Aggression vs Asian Handicap")

if not history.empty and all(col in history.columns for col in ['Target_AH_Home', 'Target_AH_Away']):
    
    # Features disponíveis para análise
    analysis_features = [f for f in away_premium_features if f in history.columns]
    
    if analysis_features:
        # Correlação com target Home
        corr_home = history[analysis_features + ['Target_AH_Home']].corr()['Target_AH_Home'].drop('Target_AH_Home')
        
        # Correlação com target Away  
        corr_away = history[analysis_features + ['Target_AH_Away']].corr()['Target_AH_Away'].drop('Target_AH_Away')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlação com Handicap Home Win:**")
            corr_df_home = pd.DataFrame({
                'Feature': corr_home.index,
                'Correlation': corr_home.values
            }).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df_home.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
        
        with col2:
            st.write("**Correlação com Handicap Away Win:**")
            corr_df_away = pd.DataFrame({
                'Feature': corr_away.index, 
                'Correlation': corr_away.values
            }).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df_away.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
