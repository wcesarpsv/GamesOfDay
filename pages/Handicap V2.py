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


##################### BLOCO 3.5 – VERIFICAÇÃO DAS COLUNAS DE AGGRESSION #####################

st.info("🔍 Verificando colunas de Aggression disponíveis...")

# Listar colunas disponíveis
aggression_related_cols = [col for col in history.columns if any(keyword in col for keyword in 
                            ['Aggression', 'HandScore', 'OverScore'])]

st.write(f"**Colunas de Aggression encontradas:** {aggression_related_cols}")

# Verificar se temos as colunas essenciais
essential_cols = ['Aggression_Home', 'Aggression_Away']
missing_essential = [col for col in essential_cols if col not in history.columns]

if missing_essential:
    st.warning(f"⚠️ Colunas essenciais faltando: {missing_essential}")
else:
    st.success("✅ Todas as colunas essenciais de Aggression estão disponíveis!")


##################### BLOCO 3.6 – TRATAMENTO DE DADOS MISSING #####################

def handle_missing_aggression(df, df_name):
    """Trata dados missing nas colunas de Aggression"""
    aggression_cols = ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away', 'OverScore_Home', 'OverScore_Away']
    available_cols = [col for col in aggression_cols if col in df.columns]
    
    if not available_cols:
        return df
    
    st.write(f"**🔄 Tratando missing values para: {df_name}**")
    
    # Estratégia: Preencher com a mediana para colunas numéricas
    for col in available_cols:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            if missing_count > 0:
                st.info(f"  {col}: Preenchidos {missing_count} valores com mediana {median_val:.3f}")
    
    st.success(f"✅ Missing values tratados para {df_name}")
    return df

##################### BLOCO 3.6 – TRATAMENTO DE DADOS MISSING #####################

def handle_missing_aggression(df, df_name):
    """Trata dados missing nas colunas de Aggression"""
    aggression_cols = ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away', 'OverScore_Home', 'OverScore_Away']
    available_cols = [col for col in aggression_cols if col in df.columns]
    
    if not available_cols:
        st.warning(f"⚠️ Nenhuma coluna de Aggression encontrada em {df_name}")
        return df
    
    st.write(f"**🔄 Tratando missing values para: {df_name}**")
    
    # Estratégia: Preencher com a mediana para colunas numéricas
    for col in available_cols:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            if missing_count > 0:
                st.info(f"  {col}: Preenchidos {missing_count} valores com mediana {median_val:.3f}")
    
    st.success(f"✅ Missing values tratados para {df_name}")
    return df

# Aplicar tratamento APENAS se as colunas existirem
available_aggression_cols = [col for col in ['Aggression_Home', 'Aggression_Away'] if col in history.columns]
if available_aggression_cols:
    history = handle_missing_aggression(history, "Dados Históricos")
    games_today = handle_missing_aggression(games_today, "Dados de Hoje")
else:
    st.warning("⚠️ Pulando tratamento de missing values - colunas de Aggression não encontradas")


##################### BLOCO 3.7 – VERIFICAÇÃO E CORREÇÃO DOS TARGETS #####################

st.markdown("### 🔍 VERIFICAÇÃO DOS TARGETS DE HANDICAP")

def diagnose_handicap_targets(df):
    """Faz diagnóstico completo dos targets de handicap"""
    
    st.write("**📊 Diagnóstico dos Targets Atuais:**")
    
    # Análise dos targets atuais
    target_analysis = pd.DataFrame({
        'Target': ['Home Handicap Win', 'Away Handicap Win', 'Draw/Partial'],
        'Count': [
            df['Target_AH_Home'].sum(),
            df['Target_AH_Away'].sum(),
            len(df) - df['Target_AH_Home'].sum() - df['Target_AH_Away'].sum()
        ],
        'Percentage': [
            f"{df['Target_AH_Home'].mean():.1%}",
            f"{df['Target_AH_Away'].mean():.1%}",
            f"{(len(df) - df['Target_AH_Home'].sum() - df['Target_AH_Away'].sum()) / len(df):.1%}"
        ]
    })
    st.dataframe(target_analysis)
    
    # Verificar overlap (quando ambos são 1)
    overlap = ((df['Target_AH_Home'] == 1) & (df['Target_AH_Away'] == 1)).sum()
    st.write(f"**🚨 Jogos onde AMBOS targets são 1: {overcome}**")
    
    # Verificar quando ambos são 0
    both_zero = ((df['Target_AH_Home'] == 0) & (df['Target_AH_Away'] == 0)).sum()
    st.write(f"**Jogos onde AMBOS targets são 0: {both_zero}**")
    
    return overlap, both_zero

# Executar diagnóstico
overlap_count, both_zero_count = diagnose_handicap_targets(history)

def create_corrected_handicap_targets(df):
    """Cria targets corrigidos para handicap"""
    
    st.warning("🔄 Criando targets corrigidos...")
    
    # VERIFICAR O CÁLCULO ORIGINAL
    st.write("**Verificação do cálculo original:**")
    sample_check = df[['Goals_H_FT', 'Goals_A_FT', 'Asian_Line', 'Margin', 
                      'Handicap_Home_Result', 'Handicap_Away_Result',
                      'Target_AH_Home', 'Target_AH_Away']].head(5)
    st.dataframe(sample_check)
    
    # 🎯 CORREÇÃO: Targets mutuamente exclusivos
    # Home win: Handicap_Home_Result > 0.5
    # Away win: Handicap_Away_Result > 0.5  
    # Draw/Partial: ambos <= 0.5
    
    df['Target_AH_Home_Corrected'] = (df['Handicap_Home_Result'] > 0.5).astype(int)
    df['Target_AH_Away_Corrected'] = (df['Handicap_Away_Result'] > 0.5).astype(int)
    
    # Verificar a correção
    st.write("**✅ Targets Corrigidos:**")
    corrected_analysis = pd.DataFrame({
        'Target': ['Home Handicap Win', 'Away Handicap Win', 'Draw/Partial'],
        'Count': [
            df['Target_AH_Home_Corrected'].sum(),
            df['Target_AH_Away_Corrected'].sum(),
            len(df) - df['Target_AH_Home_Corrected'].sum() - df['Target_AH_Away_Corrected'].sum()
        ],
        'Percentage': [
            f"{df['Target_AH_Home_Corrected'].mean():.1%}",
            f"{df['Target_AH_Away_Corrected'].mean():.1%}",
            f"{(len(df) - df['Target_AH_Home_Corrected'].sum() - df['Target_AH_Away_Corrected'].sum()) / len(df):.1%}"
        ]
    })
    st.dataframe(corrected_analysis)
    
    # Verificar overlap nos corrigidos
    overlap_corrected = ((df['Target_AH_Home_Corrected'] == 1) & (df['Target_AH_Away_Corrected'] == 1)).sum()
    st.write(f"**Overlap nos targets corrigidos: {overlap_corrected}**")
    
    return df

# Aplicar correção se necessário
if overlap_count > 0 or both_zero_count > len(history) * 0.3:
    st.error("🚨 PROBLEMA DETECTADO: Targets com overlap ou muitos empates!")
    history = create_corrected_handicap_targets(history)
    
    # Usar os targets corrigidos
    history['Target_AH_Home'] = history['Target_AH_Home_Corrected']
    history['Target_AH_Away'] = history['Target_AH_Away_Corrected']
    
    st.success("✅ Targets corrigidos aplicados!")
else:
    st.success("✅ Targets parecem OK!")


##################### BLOCO 4 – FEATURE ENGINEERING SIMPLIFICADO #####################

# Usar apenas features mais importantes para evitar overfitting
feature_blocks_simplified = {
    "odds": ["Odd_H", "Odd_A"],
    "strength": ["Diff_Power", "M_H", "M_A", "Asian_Line_Display"],
    "aggression": ["Aggression_Home", "Aggression_Away", "Underdog_Indicator"],
    "categorical": []
}

# Filtrar apenas features que existem
for block_name, cols in feature_blocks_simplified.items():
    feature_blocks_simplified[block_name] = [c for c in cols if c in history.columns]

st.write("**🔧 Features Simplificadas:**", feature_blocks_simplified)

# Rebuild com features simplificadas
X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks_simplified)
X_ah_away = X_ah_home.copy()

X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks_simplified)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()

# Atualizar numeric_cols
numeric_cols = (feature_blocks_simplified["odds"] + feature_blocks_simplified["strength"] + 
                feature_blocks_simplified["aggression"])
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]


##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features", value=True)


##################### BLOCO 6 – TRAIN & EVALUATE (ATUALIZADO) #####################

def train_simple_model(X, y, name):
    """Treina modelo mais simples para evitar overfitting"""
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_SIMPLE.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            return {"Model": f"{name}_Simple", "Accuracy": "Loaded"}, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Modelo mais simples
    if ml_model_choice == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {
        "Model": f"{name}_Simple", 
        "Accuracy": f"{accuracy_score(y_test, preds):.3f}",
        "LogLoss": f"{log_loss(y_test, probs):.3f}",
        "BrierScore": f"{brier_score_loss(y_test, probs[:,1]):.3f}"
    }

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


##################### BLOCO 7 – TRAINING MODELS SIMPLIFICADOS #####################

st.info("🤖 Treinando modelos simplificados...")

stats_simple = []
res, model_ah_home_simple = train_simple_model(X_ah_home, history["Target_AH_Home"], "AH_Home")
stats_simple.append(res)
res, model_ah_away_simple = train_simple_model(X_ah_away, history["Target_AH_Away"], "AH_Away")
stats_simple.append(res)

stats_simple_df = pd.DataFrame(stats_simple)
st.markdown("### 📊 Model Statistics (Simplified)")
st.dataframe(stats_simple_df, use_container_width=True)


##################### BLOCO 8 – PREDICTIONS COM MODELOS SIMPLIFICADOS #####################

# Usar modelos simplificados
model_ah_home, cols1 = model_ah_home_simple
model_ah_away, cols2 = model_ah_away_simple

X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

if normalize_features:
    scaler = StandardScaler()
    scaler.fit(X_ah_home[numeric_cols])
    X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
    X_today_ah_away[numeric_cols] = scaler.transform(X_today_ah_away[numeric_cols])

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for cls, col in zip(model_ah_home.classes_, ["p_ah_home_no", "p_ah_home_yes"]):
        games_today[col] = probs_home[:, cls]

    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for cls, col in zip(model_ah_away.classes_, ["p_ah_away_no", "p_ah_away_yes"]):
        games_today[col] = probs_away[:, cls]

# VERIFICAÇÃO DAS PROBABILIDADES
st.markdown("### 🔍 VERIFICAÇÃO DAS PROBABILIDADES")

games_today['prob_sum'] = games_today['p_ah_home_yes'] + games_today['p_ah_away_yes']
prob_stats = games_today['prob_sum'].describe()

st.write("**Estatísticas da Soma das Probabilidades:**")
st.write(prob_stats)

# Identificar se ainda há problemas
problem_games = games_today[
    (games_today['p_ah_home_yes'] > 0.6) & 
    (games_today['p_ah_away_yes'] > 0.6)
]

if not problem_games.empty:
    st.error(f"🚨 {len(problem_games)} jogos ainda com probabilidades altas para AMBOS os lados!")
    st.dataframe(problem_games[['Home', 'Away', 'p_ah_home_yes', 'p_ah_away_yes', 'prob_sum']])
else:
    st.success("✅ Probabilidades estão bem distribuídas!")


##################### BLOCO 9 – SISTEMA DE RECOMENDAÇÕES INTELIGENTE #####################

def identify_team_trends(games_df):
    """Identifica times em alta/baixa baseado nas novas features"""
    
    trends = []
    
    for _, jogo in games_df.iterrows():
        home_trend = "NEUTRO"
        away_trend = "NEUTRO"
        
        # Analisar Home
        if 'Home_HandScore_Trend' in games_df.columns and 'Home_Positive_Streak' in games_df.columns:
            home_trend_score = jogo.get('Home_HandScore_Trend', 0)
            home_streak = jogo.get('Home_Positive_Streak', 0)
            
            if home_trend_score > 0.3 and home_streak >= 2:
                home_trend = "🔥 ALTA"
            elif home_trend_score < -0.3 and home_streak == 0:
                home_trend = "💀 BAIXA"
            elif jogo.get('Home_Reversion_Potential', 0) > 0.3:
                home_trend = "🔄 REVERSÃO"
        
        # Analisar Away
        if 'Away_HandScore_Trend' in games_df.columns and 'Away_Positive_Streak' in games_df.columns:
            away_trend_score = jogo.get('Away_HandScore_Trend', 0)
            away_streak = jogo.get('Away_Positive_Streak', 0)
            
            if away_trend_score > 0.3 and away_streak >= 2:
                away_trend = "🔥 ALTA"  
            elif away_trend_score < -0.3 and away_streak == 0:
                away_trend = "💀 BAIXA"
            elif jogo.get('Away_Reversion_Potential', 0) > 0.3:
                away_trend = "🔄 REVERSÃO"
        
        trends.append({
            'Home': jogo['Home'],
            'Away': jogo['Away'], 
            'Home_Trend': home_trend,
            'Away_Trend': away_trend,
            'Momentum_Advantage': jogo.get('Momentum_Advantage', 0),
            'Home_Form': jogo.get('Home_Goals_Form', 0),
            'Away_Form': jogo.get('Away_Goals_Form', 0)
        })
    
    return pd.DataFrame(trends)

def find_smart_away_handicap_bets(games_df):
    """Encontra apostas inteligentes de Away Handicap usando Aggression + Momentum"""
    
    conditions = (
        # Condições de Aggression (já testadas)
        (games_df['Underdog_Indicator'] > 0.3) &      # Home é underdog
        (games_df['Aggression_Away'] > 0.2) &         # Away é favorito
        (games_df['Aggression_Home'] < -0.2) &        # Home é underdog
        
        # Condições de Momentum
        (games_df.get('Away_HandScore_Trend', 0) > games_df.get('Home_HandScore_Trend', 0)) &  # Away em melhor momentum
        (games_df.get('Away_Positive_Streak', 0) >= 1) &  # Away com pelo menos 1 jogo positivo
        
        # Probabilidade do modelo
        (games_df['p_ah_away_yes'] > 0.55)
    )
    
    # Aplicar condições apenas para colunas que existem
    mask = pd.Series(True, index=games_df.index)
    for condition in conditions:
        if isinstance(condition, pd.Series):
            mask = mask & condition
    
    smart_bets = games_df[mask]
    
    return smart_bets

# Aplicar sistema de recomendações
st.markdown("### 🎯 Recomendações Inteligentes")

# Análise de tendências
if any(col in games_today.columns for col in ['Home_HandScore_Trend', 'Away_HandScore_Trend']):
    trends_df = identify_team_trends(games_today)
    st.write("**🔥 Análise de Tendências dos Times:**")
    st.dataframe(trends_df, use_container_width=True)

# Melhores apostas Away Handicap
smart_away_bets = find_smart_away_handicap_bets(games_today)

if not smart_away_bets.empty:
    st.success(f"🎯 **MELHORES OPORTUNIDADES - AWAY HANDICAP**: {len(smart_away_bets)} jogos")
    
    for _, jogo in smart_away_bets.iterrows():
        st.write(f"**{jogo['Home']} vs {jogo['Away']}**")
        st.write(f"🏠 Home Aggression: {jogo.get('Aggression_Home', 'N/A'):.2f} | Trend: {trends_df[trends_df['Home'] == jogo['Home']]['Home_Trend'].iloc[0] if not trends_df.empty else 'N/A'}")
        st.write(f"✈️ Away Aggression: {jogo.get('Aggression_Away', 'N/A'):.2f} | Trend: {trends_df[trends_df['Away'] == jogo['Away']]['Away_Trend'].iloc[0] if not trends_df.empty else 'N/A'}")
        st.write(f"⚖️ Underdog Indicator: {jogo.get('Underdog_Indicator', 'N/A'):.2f}")
        st.write(f"📊 Prob Away Handicap: {jogo['p_ah_away_yes']:.1%}")
        st.write("---")
else:
    st.info("ℹ️ Nenhuma oportunidade clara de Away Handicap identificada hoje.")

st.success("✅ Sistema executado com sucesso!")
