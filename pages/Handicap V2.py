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


##################### BLOCO 4 – FEATURE ENGINEERING COM AGGRESSION E MOMENTUM #####################

def add_aggression_features(df):
    """
    Aggression positivo = DÁ mais handicap (favorito)
    Aggression negativo = RECEBE mais handicap (underdog)
    """
    aggression_features = []
    
    # VERIFICAR COM NOMES CORRETOS
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away']):
        # Verificar qualidade dos dados
        valid_data = df['Aggression_Home'].notna() & df['Aggression_Away'].notna()
        
        if valid_data.any():
            # Criar features com nomes CORRETOS
            df['Handicap_Balance'] = df['Aggression_Home'] - df['Aggression_Away']
            df['Underdog_Indicator'] = -df['Handicap_Balance']  # Positivo = Home underdog
            
            # Power vs Market Perception
            if 'M_H' in df.columns and 'M_A' in df.columns:
                df['Power_Perception_Home'] = df['M_H'] - df['Aggression_Home']
                df['Power_Perception_Away'] = df['M_A'] - df['Aggression_Away']
                df['Power_Perception_Diff'] = df['Power_Perception_Home'] - df['Power_Perception_Away']
            
            aggression_features.extend(['Aggression_Home', 'Aggression_Away', 'Handicap_Balance', 
                                      'Underdog_Indicator', 'Power_Perception_Diff'])
        else:
            st.warning("⚠️ Colunas de Aggression existem mas não têm dados válidos")
    
    # HandScore com nomes CORRETOS
    if all(col in df.columns for col in ['HandScore_Home', 'HandScore_Away']):
        if df['HandScore_Home'].notna().any() and df['HandScore_Away'].notna().any():
            df['HandScore_Diff'] = df['HandScore_Home'] - df['HandScore_Away']
            aggression_features.append('HandScore_Diff')
    
    # OverScore com nomes CORRETOS  
    if all(col in df.columns for col in ['OverScore_Home', 'OverScore_Away']):
        if df['OverScore_Home'].notna().any() and df['OverScore_Away'].notna().any():
            df['OverScore_Diff'] = df['OverScore_Home'] - df['OverScore_Away']
            df['Total_OverScore'] = df['OverScore_Home'] + df['OverScore_Away']
            aggression_features.extend(['OverScore_Diff', 'Total_OverScore'])
    
    return df, aggression_features

def create_momentum_features(df, window=5):
    """Cria features que capturam momentum e tendências dos times"""
    
    momentum_features = []
    
    # Garantir que os dados estão ordenados por data
    if 'Date' not in df.columns:
        return df, momentum_features
    
    df = df.sort_values(['Home', 'Date']).reset_index(drop=True)
    
    # Features para Home teams
    if all(col in df.columns for col in ['HandScore_Home', 'Aggression_Home', 'Goals_H_FT']):
        
        # 1. MOMENTUM RECENTE DO HANDSCORE (últimos X jogos)
        df['Home_HandScore_Trend'] = df.groupby('Home')['HandScore_Home'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # 2. TENDÊNCIA DE AGGRESSION (está ficando mais ou menos agressivo?)
        df['Home_Aggression_Trend'] = df.groupby('Home')['Aggression_Home'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # 3. VOLATILIDADE DO PERFORMANCE
        df['Home_HandScore_Volatility'] = df.groupby('Home')['HandScore_Home'].transform(
            lambda x: x.rolling(window=window, min_periods=2).std()
        ).fillna(0)
        
        # 4. MOMENTUM OFENSIVO (gols recentes)
        df['Home_Goals_Form'] = df.groupby('Home')['Goals_H_FT'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 5. "HOT STREAK" - sequência de bons resultados
        df['Home_Positive_Streak'] = df.groupby('Home')['HandScore_Home'].transform(
            lambda x: (x > 0).rolling(window=3).sum()
        )
        
        momentum_features.extend([
            'Home_HandScore_Trend', 'Home_Aggression_Trend', 
            'Home_HandScore_Volatility', 'Home_Goals_Form',
            'Home_Positive_Streak'
        ])
    
    # Features para Away teams (mesma lógica)
    if all(col in df.columns for col in ['HandScore_Away', 'Aggression_Away', 'Goals_A_FT']):
        
        df['Away_HandScore_Trend'] = df.groupby('Away')['HandScore_Away'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        df['Away_Aggression_Trend'] = df.groupby('Away')['Aggression_Away'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        df['Away_HandScore_Volatility'] = df.groupby('Away')['HandScore_Away'].transform(
            lambda x: x.rolling(window=window, min_periods=2).std()
        ).fillna(0)
        
        df['Away_Goals_Form'] = df.groupby('Away')['Goals_A_FT'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        df['Away_Positive_Streak'] = df.groupby('Away')['HandScore_Away'].transform(
            lambda x: (x > 0).rolling(window=3).sum()
        )
        
        momentum_features.extend([
            'Away_HandScore_Trend', 'Away_Aggression_Trend',
            'Away_HandScore_Volatility', 'Away_Goals_Form', 
            'Away_Positive_Streak'
        ])
    
    # 6. FEATURES COMPARATIVAS DE MOMENTUM
    if all(col in df.columns for col in ['Home_HandScore_Trend', 'Away_HandScore_Trend']):
        df['Momentum_Advantage'] = df['Home_HandScore_Trend'] - df['Away_HandScore_Trend']
        df['Form_Difference'] = df['Home_Goals_Form'] - df['Away_Goals_Form']
        df['Trend_Consistency_Diff'] = df['Home_HandScore_Volatility'] - df['Away_HandScore_Volatility']
        
        momentum_features.extend([
            'Momentum_Advantage', 'Form_Difference', 'Trend_Consistency_Diff'
        ])
    
    return df, momentum_features

def create_contrarian_features(df):
    """Cria features que identificam oportunidades contrarianas"""
    
    contrarian_features = []
    
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'Odd_H', 'Odd_A']):
        
        # 1. "OVERVALUED" - Times com aggression alta mas odds altas (supervalorizados)
        df['Home_Overvalued'] = df['Aggression_Home'] * df['Odd_H']  # Alto = possivelmente supervalorizado
        df['Away_Overvalued'] = df['Aggression_Away'] * df['Odd_A']
        
        # 2. "UNDERVALUED" - Times com aggression baixa mas odds baixas (subvalorizados)  
        df['Home_Undervalued'] = (1 - df['Aggression_Home']) * (1/df['Odd_H'])  # Inverso
        df['Away_Undervalued'] = (1 - df['Aggression_Away']) * (1/df['Odd_A'])
        
        # 3. "REVERSION INDICATOR" - Times em baixa mas com potencial de reversão
        if 'Home_HandScore_Volatility' in df.columns and 'Home_Positive_Streak' in df.columns:
            df['Home_Reversion_Potential'] = df['Home_HandScore_Volatility'] * (1 - df['Home_Positive_Streak']/3)
        if 'Away_HandScore_Volatility' in df.columns and 'Away_Positive_Streak' in df.columns:
            df['Away_Reversion_Potential'] = df['Away_HandScore_Volatility'] * (1 - df['Away_Positive_Streak']/3)
        
        contrarian_features.extend([
            'Home_Overvalued', 'Away_Overvalued',
            'Home_Undervalued', 'Away_Undervalued'
        ])
        
        if 'Home_Reversion_Potential' in df.columns:
            contrarian_features.extend(['Home_Reversion_Potential', 'Away_Reversion_Potential'])
    
    return df, contrarian_features

# Aplicar todas as features
st.info("🔄 Aplicando feature engineering...")

history, aggression_features = add_aggression_features(history)
games_today, _ = add_aggression_features(games_today)

history, momentum_features = create_momentum_features(history)
games_today, _ = create_momentum_features(games_today)

history, contrarian_features = create_contrarian_features(history)
games_today, _ = create_contrarian_features(games_today)

# BLOCO DE FEATURES ATUALIZADO
feature_blocks = {
    "odds": ["Odd_H", "Odd_D", "Odd_A"],
    "strength": [
        "Diff_Power", "M_H", "M_A", "Diff_M",
        "Diff_HT_P", "M_HT_H", "M_HT_A",
        "Asian_Line_Display"
    ],
    "aggression": aggression_features,
    "momentum": momentum_features,
    "contrarian": contrarian_features,
    "categorical": []
}

# Filtrar apenas as features que existem no dataframe
for block_name in ["aggression", "momentum", "contrarian"]:
    feature_blocks[block_name] = [col for col in feature_blocks[block_name] if col in history.columns]

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)
feature_blocks["categorical"] = list(history_leagues.columns)

def build_feature_matrix(df, leagues, blocks):
    dfs = []
    for block_name, cols in blocks.items():
        if block_name == "categorical":
            dfs.append(leagues)
        elif cols:  # Só adiciona se a lista não estiver vazia
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                dfs.append(df[available_cols])
    return pd.concat(dfs, axis=1)

X_ah_home = build_feature_matrix(history, history_leagues, feature_blocks)
X_ah_away = X_ah_home.copy()

X_today_ah_home = build_feature_matrix(games_today, games_today_leagues, feature_blocks)
X_today_ah_home = X_today_ah_home.reindex(columns=X_ah_home.columns, fill_value=0)
X_today_ah_away = X_today_ah_home.copy()

# ATUALIZAR numeric_cols para incluir as novas features
numeric_cols = (feature_blocks["odds"] + feature_blocks["strength"] + 
                feature_blocks["aggression"] + feature_blocks["momentum"] +
                feature_blocks["contrarian"])
numeric_cols = [c for c in numeric_cols if c in X_ah_home.columns]

# BLOCO EXTRA – ANÁLISE DAS NOVAS FEATURES
st.markdown("### 🔍 Análise: Aggression vs Asian Handicap")

if not history.empty and all(col in history.columns for col in ['Target_AH_Home', 'Target_AH_Away'] + aggression_features):
    
    # Análise de correlação
    available_aggression_for_analysis = [f for f in aggression_features if f in history.columns]
    
    if available_aggression_for_analysis:
        # Correlação com target Home
        corr_home = history[available_aggression_for_analysis + ['Target_AH_Home']].corr()['Target_AH_Home'].drop('Target_AH_Home')
        
        # Correlação com target Away  
        corr_away = history[available_aggression_for_analysis + ['Target_AH_Away']].corr()['Target_AH_Away'].drop('Target_AH_Away')
        
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
        
        # Destacar features mais promissoras
        strong_features_home = corr_df_home[abs(corr_df_home['Correlation']) > 0.05]
        strong_features_away = corr_df_away[abs(corr_df_away['Correlation']) > 0.05]
        
        if not strong_features_home.empty or not strong_features_away.empty:
            st.success("🎯 **Features mais promissoras (correlação > |0.05|):**")
            if not strong_features_home.empty:
                st.write("Para Handicap Home:", ", ".join(strong_features_home['Feature'].tolist()))
            if not strong_features_away.empty:
                st.write("Para Handicap Away:", ", ".join(strong_features_away['Feature'].tolist()))


##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
ml_version_choice = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features", value=True)


##################### BLOCO 6 – TRAIN & EVALUATE #####################
def train_and_evaluate(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2C_v1.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {"Model": f"{name}_v1", "Accuracy": accuracy_score(y, preds),
                   "LogLoss": log_loss(y, probs), "BrierScore": brier_score_loss(y, probs[:,1])}
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if ml_model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    else:
        model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              use_label_encoder=False, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {"Model": f"{name}_v1", "Accuracy": accuracy_score(y_test, preds),
           "LogLoss": log_loss(y_test, probs), "BrierScore": brier_score_loss(y_test, probs[:,1])}

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


def train_and_evaluate_v2(X, y, name, use_calibration=True):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2C_v2.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {"Model": f"{name}_v2", "Accuracy": accuracy_score(y, preds),
                   "LogLoss": log_loss(y, probs), "BrierScore": brier_score_loss(y, probs[:,1])}
            return res, (model, cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if ml_model_choice == "Random Forest":
        base_model = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced",
                                            random_state=42, n_jobs=-1)
    else:
        base_model = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                   use_label_encoder=False, random_state=42,
                                   scale_pos_weight=(sum(y == 0) / sum(y == 1)) if sum(y == 1) > 0 else 1)

    if use_calibration:
        try:
            model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=2)
        except TypeError:
            model = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=2)
        model.fit(X_train, y_train)
    else:
        if ml_model_choice == "XGBoost":
            base_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)
        else:
            base_model.fit(X_train, y_train)
        model = base_model

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {"Model": f"{name}_v2", "Accuracy": accuracy_score(y_test, preds),
           "LogLoss": log_loss(y_test, probs), "BrierScore": brier_score_loss(y_test, probs[:,1])}

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols)


##################### BLOCO 7 – TRAINING MODELS #####################
stats = []
res, model_ah_home_v1 = train_and_evaluate(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v1 = train_and_evaluate(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)
res, model_ah_home_v2 = train_and_evaluate_v2(X_ah_home, history["Target_AH_Home"], "AH_Home"); stats.append(res)
res, model_ah_away_v2 = train_and_evaluate_v2(X_ah_away, history["Target_AH_Away"], "AH_Away"); stats.append(res)

stats_df = pd.DataFrame(stats)[["Model", "Accuracy", "LogLoss", "BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) – v1 vs v2")
st.dataframe(stats_df, use_container_width=True)


##################### BLOCO 8 – PREDICTIONS #####################
if ml_version_choice == "v1":
    model_ah_home, cols1 = model_ah_home_v1
    model_ah_away, cols2 = model_ah_away_v1
else:
    model_ah_home, cols1 = model_ah_home_v2
    model_ah_away, cols2 = model_ah_away_v2

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

def color_prob(val, color):
    if pd.isna(val): return ""
    alpha = float(np.clip(val, 0, 1))
    return f"background-color: rgba({color}, {alpha:.2f})"

# ========== EXIBIÇÃO DAS PREDIÇÕES ==========
styled_df = (
    games_today[[
        "Date","Time","League","Home","Away",
        "Goals_H_Today", "Goals_A_Today",
        "Odd_H","Odd_D","Odd_A",
        "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
        "p_ah_home_yes","p_ah_away_yes"
    ]]
    .style.format({
        "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
        "Asian_Line_Display": "{:.2f}",
        "Odd_H_Asi": "{:.2f}", "Odd_A_Asi": "{:.2f}",
        "p_ah_home_yes": "{:.1%}", "p_ah_away_yes": "{:.1%}",
        "Goals_H_Today": "{:.0f}", "Goals_A_Today": "{:.0f}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v, "0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v, "255,140,0"), subset=["p_ah_away_yes"])
)

st.markdown(f"### 📌 Predictions for {selected_date_str} – Asian Handicap ({ml_version_choice})")
st.dataframe(styled_df, use_container_width=True, height=800)


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
