from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import math
import plotly.graph_objects as go

st.set_page_config(page_title="Análise de Quadrantes 3D - Bet Indicator", layout="wide")
st.title("🎯 Análise 3D de 16 Quadrantes - ML Avançado (Home & Away)")

# ---------------- Configurações ----------------
PAGE_PREFIX = "QuadrantesML_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)




#################################



def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def convert_asian_line_to_decimal(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if "/" not in value:
        try:
            num = float(value)
            return -num
        except ValueError:
            return np.nan
    try:
        parts = [float(p) for p in value.split("/")]
        avg = np.mean(parts)
        if str(value).startswith("-"):
            result = -abs(avg)
        else:
            result = abs(avg)
        return -result
    except ValueError:
        return np.nan

def calculate_ah_home_target(margin, asian_line_str):
    line_home = convert_asian_line_to_decimal(asian_line_str)
    if pd.isna(line_home) or pd.isna(margin):
        return np.nan
    return 1 if margin > line_home else 0



###############################################



def setup_livescore_columns(df):
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

def load_and_merge_livescore(games_today, selected_date_str):
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

    if os.path.exists(livescore_file):
        st.info(f"📡 LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

        required_cols = [
            'Id', 'status', 'home_goal', 'away_goal',
            'home_ht_goal', 'away_ht_goal',
            'home_corners', 'away_corners', 
            'home_yellow', 'away_yellow',
            'home_red', 'away_red'
        ]

        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            st.error(f"❌ LiveScore file missing columns: {missing_cols}")
            return games_today
        else:
            games_today = games_today.merge(
                results_df,
                left_on='Id',
                right_on='Id',
                how='left',
                suffixes=('', '_RAW')
            )
            games_today['Goals_H_Today'] = games_today['home_goal']
            games_today['Goals_A_Today'] = games_today['away_goal']
            games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
            games_today['Home_Red'] = games_today['home_red']
            games_today['Away_Red'] = games_today['away_red']
            st.success(f"✅ LiveScore merged: {len(results_df)} games loaded")
            return games_today
    else:
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today



###########################################





@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
    return games_today, history

# Carregar dados
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")
files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)
date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

games_today, history = load_cached_data(selected_file)
games_today = load_and_merge_livescore(games_today, selected_date_str)




###########################################





QUADRANTES_16 = {
    1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
    2: {"nome": "Fav Forte Forte",       "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
    3: {"nome": "Fav Forte Moderado",    "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
    4: {"nome": "Fav Forte Neutro",      "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},
    5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
    6: {"nome": "Fav Moderado Forte",       "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
    7: {"nome": "Fav Moderado Moderado",    "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
    8: {"nome": "Fav Moderado Neutro",      "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},
    9: {"nome": "Under Moderado Neutro",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
    10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
    11: {"nome": "Under Moderado Forte",    "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
    12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},
    13: {"nome": "Under Forte Neutro",    "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
    14: {"nome": "Under Forte Moderado",  "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
    15: {"nome": "Under Forte Forte",     "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
    16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
}

def classificar_quadrante_16(agg, hs):
    if pd.isna(agg) or pd.isna(hs):
        return 0
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id
    return 0

# Aplicar classificação
games_today['Quadrante_Home'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
games_today['Quadrante_Away'] = games_today.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)
history['Quadrante_Home'] = history.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
)
history['Quadrante_Away'] = history.apply(
    lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
)





##############################



def calcular_momentum_time(df, window=6):
    df = df.copy()
    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_home, 'MT_H'] = zscore

        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            zscore = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            df.loc[mask_away, 'MT_A'] = zscore

    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)
    return df

def aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5):
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    history_clean = history[required_cols].fillna(0)
    games_today_clean = games_today[required_cols].fillna(0)
    
    history_clean['dx'] = history_clean['Aggression_Home'] - history_clean['Aggression_Away']
    history_clean['dy'] = history_clean['M_H'] - history_clean['M_A']
    history_clean['dz'] = history_clean['MT_H'] - history_clean['MT_A']
    
    games_today_clean['dx'] = games_today_clean['Aggression_Home'] - games_today_clean['Aggression_Away']
    games_today_clean['dy'] = games_today_clean['M_H'] - games_today_clean['M_A']
    games_today_clean['dz'] = games_today_clean['MT_H'] - games_today_clean['MT_A']
    
    X_train = history_clean[['dx', 'dy', 'dz']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=10)
    kmeans.fit(X_train)
    
    history['Cluster3D_Label'] = kmeans.predict(history_clean[['dx', 'dy', 'dz']].values)
    games_today['Cluster3D_Label'] = kmeans.predict(games_today_clean[['dx', 'dy', 'dz']].values)
    
    cluster_descriptions = {
        0: '⚡ Agressivos + Momentum Positivo',
        1: '💤 Reativos + Momentum Negativo', 
        2: '⚖️ Equilibrados',
        3: '🔥 Alta Variância',
        4: '🌪️ Caóticos / Transição'
    }
    
    history['Cluster3D_Desc'] = history['Cluster3D_Label'].map(cluster_descriptions).fillna('🌀 Outro')
    games_today['Cluster3D_Desc'] = games_today['Cluster3D_Label'].map(cluster_descriptions).fillna('🌀 Outro')
    
    return history, games_today

# Aplicar momentum e clusterização
history = calcular_momentum_time(history)
games_today = calcular_momentum_time(games_today)
history, games_today = aplicar_clusterizacao_3d_segura(history, games_today)




##################




def calcular_distancias_3d(df):
    df = df.copy()
    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in [
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Quadrant_Angle_XY', 'Quadrant_Angle_XZ', 'Quadrant_Angle_YZ',
            'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
            'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
            'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
            'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo', 'Vector_Sign',
            'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D'
        ]:
            df[col] = np.nan
        return df

    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    angle_xy = np.arctan2(dy, dx)
    angle_xz = np.arctan2(dz, dx)
    angle_yz = np.arctan2(dz, dy)

    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Angle_XZ'] = np.degrees(angle_xz)
    df['Quadrant_Angle_YZ'] = np.degrees(angle_yz)

    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Quadrant_Sin_XZ'] = np.sin(angle_xz)
    df['Quadrant_Cos_XZ'] = np.cos(angle_xz)
    df['Quadrant_Sin_YZ'] = np.sin(angle_yz)
    df['Quadrant_Cos_YZ'] = np.cos(angle_yz)

    df['Quadrant_Sin_Combo'] = np.sin(angle_xy + angle_xz + angle_yz)
    df['Quadrant_Cos_Combo'] = np.cos(angle_xy + angle_xz + angle_yz)
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df

# Aplicar cálculo 3D
games_today = calcular_distancias_3d(games_today)




###################################################



def treinar_modelo_3d_clusters_single(history, games_today):
    st.markdown("### ⚙️ Configuração do Treino 3D com Odds de Abertura")
    use_opening_odds = st.checkbox("📊 Incluir Odds de Abertura no Treino", value=True)

    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history, games_today = aplicar_clusterizacao_3d_segura(history, games_today, n_clusters=5)

    # Converter Asian Line e calcular targets
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    history = history.dropna(subset=['Asian_Line_Decimal'])
    
    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
        except Exception as e:
            st.error(f"Erro ao aplicar filtro temporal: {e}")

    history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
    history["Target_AH_Home"] = history.apply(
        lambda r: calculate_ah_home_target(r["Margin"], r["Asian_Line"]), axis=1
    )

    # Feature Engineering
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    features_3d = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Quadrant_Sin_XZ', 'Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ', 'Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo', 'Quadrant_Cos_Combo',
        'Vector_Sign', 'Magnitude_3D'
    ]

    extras_3d = history[features_3d].fillna(0)
    odds_features = pd.DataFrame()
    
    if use_opening_odds:
        for col in ['Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP']:
            if col not in history.columns:
                history[col] = np.nan

        history['Imp_H_OP'] = 1 / history['Odd_H_OP']
        history['Imp_D_OP'] = 1 / history['Odd_D_OP']
        history['Imp_A_OP'] = 1 / history['Odd_A_OP']
        history[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']] = history[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].replace([np.inf, -np.inf], np.nan)

        sum_probs = history[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].sum(axis=1).replace(0, np.nan)
        history['Imp_H_OP_Norm'] = history['Imp_H_OP'] / sum_probs
        history['Imp_D_OP_Norm'] = history['Imp_D_OP'] / sum_probs
        history['Imp_A_OP_Norm'] = history['Imp_A_OP'] / sum_probs

        odds_features = history[['Imp_H_OP_Norm', 'Imp_D_OP_Norm', 'Imp_A_OP_Norm']].fillna(0)

    if use_opening_odds:
        X = pd.concat([ligas_dummies, clusters_dummies, extras_3d, odds_features], axis=1)
    else:
        X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)

    y_home = history['Target_AH_Home'].astype(int)

    # Treinar modelo
    model_home = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    model_home.fit(X, y_home)

    # Previsões para jogos de hoje
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d].fillna(0)

    if use_opening_odds:
        for col in ['Odd_H_OP', 'Odd_D_OP', 'Odd_A_OP']:
            if col not in games_today.columns:
                games_today[col] = np.nan

        games_today['Imp_H_OP'] = 1 / games_today['Odd_H_OP']
        games_today['Imp_D_OP'] = 1 / games_today['Odd_D_OP']
        games_today['Imp_A_OP'] = 1 / games_today['Odd_A_OP']
        games_today[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']] = games_today[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].replace([np.inf, -np.inf], np.nan)

        sum_today = games_today[['Imp_H_OP', 'Imp_D_OP', 'Imp_A_OP']].sum(axis=1).replace(0, np.nan)
        games_today['Imp_H_OP_Norm'] = games_today['Imp_H_OP'] / sum_today
        games_today['Imp_D_OP_Norm'] = games_today['Imp_D_OP'] / sum_today
        games_today['Imp_A_OP_Norm'] = games_today['Imp_A_OP'] / sum_today

        odds_today = games_today[['Imp_H_OP_Norm', 'Imp_D_OP_Norm', 'Imp_A_OP_Norm']].fillna(0)
        X_today = pd.concat([ligas_today, clusters_today, extras_today, odds_today], axis=1)
    else:
        X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    # Fazer previsões
    proba_home = model_home.predict_proba(X_today)[:, 1]
    proba_away = 1 - proba_home

    games_today['Prob_Home'] = proba_home
    games_today['Prob_Away'] = proba_away
    games_today['ML_Side'] = np.where(proba_home > proba_away, 'HOME', 'AWAY')
    games_today['ML_Confidence'] = np.maximum(proba_home, proba_away)
    games_today['Quadrante_ML_Score_Home'] = games_today['Prob_Home']
    games_today['Quadrante_ML_Score_Away'] = games_today['Prob_Away']
    games_today['Quadrante_ML_Score_Main'] = games_today['ML_Confidence']

    # Avaliação
    accuracy = model_home.score(X, y_home)
    st.metric("Accuracy (Treino)", f"{accuracy:.2%}")
    st.write("📘 Features usadas:", len(X.columns))

    # Importância de features
    importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_feats = importances.head(25).to_frame("Importância")
    st.markdown("### 🔍 Top Features (Modelo Único – Home)")
    st.dataframe(top_feats, use_container_width=True)

    if use_opening_odds:
        odds_influentes = [f for f in top_feats.index if "Imp_" in f]
        if odds_influentes:
            st.success(f"💡 Variáveis de abertura influentes: {', '.join(odds_influentes)}")

    # Segurança final
    if "Quadrante_ML_Score_Home" not in games_today.columns:
        games_today["Quadrante_ML_Score_Home"] = np.nan
        games_today["Quadrante_ML_Score_Away"] = np.nan
        games_today["Quadrante_ML_Score_Main"] = np.nan
        games_today["ML_Side"] = "N/A"
        games_today["ML_Confidence"] = 0.0

    for col in ["League", "Home", "Away"]:
        if col not in games_today.columns:
            games_today[col] = "N/A"

    if games_today.empty:
        st.warning("⚠️ Nenhum jogo válido encontrado após o treino.")
    else:
        st.success(f"✅ {len(games_today)} jogos processados e prontos para análise 3D.")

    st.success("✅ Modelo 3D treinado (HOME) – com análise de viés integrada.")
    return model_home, games_today






################################





def calcular_features_mercado(df):
    df = df.copy()
    
    # Variação básica das odds
    df['Var_Odd_H'] = ((df['Odd_H_OP'] - df['Odd_H']) / df['Odd_H_OP'] * 100).fillna(0)
    df['Var_Odd_A'] = ((df['Odd_A_OP'] - df['Odd_A']) / df['Odd_A_OP'] * 100).fillna(0)
    
    # Volatilidade
    df['Volatilidade_Odd_H'] = abs(df['Var_Odd_H'])
    df['Volatilidade_Odd_A'] = abs(df['Var_Odd_A'])
    
    # Probabilidade implícita das odds
    df['Prob_Impl_Odd_H_OP'] = 1 / df['Odd_H_OP']
    df['Prob_Impl_Odd_A_OP'] = 1 / df['Odd_A_OP']
    df['Prob_Impl_Odd_H'] = 1 / df['Odd_H']
    df['Prob_Impl_Odd_A'] = 1 / df['Odd_A']
    
    # Gap entre probabilidade ML e probabilidade implícita das odds
    if 'Quadrante_ML_Score_Home' in df.columns:
        df['Gap_Prob_Odd_H'] = df['Quadrante_ML_Score_Home'] - df['Prob_Impl_Odd_H_OP']
        df['Gap_Prob_Odd_A'] = df['Quadrante_ML_Score_Away'] - df['Prob_Impl_Odd_A_OP']
    else:
        df['Gap_Prob_Odd_H'] = 0
        df['Gap_Prob_Odd_A'] = 0
    
    # Tendência de força (odds caindo = força)
    df['Tendencia_Forca_H'] = np.where(df['Var_Odd_H'] > 5, 1, 
                                      np.where(df['Var_Odd_H'] < -5, -1, 0))
    df['Tendencia_Forca_A'] = np.where(df['Var_Odd_A'] > 5, 1, 
                                      np.where(df['Var_Odd_A'] < -5, -1, 0))
    
    # Score de value baseado em múltiplos fatores
    df['Value_Score_H'] = (
        (df['Gap_Prob_Odd_H'].clip(lower=0) * 0.4) +
        (df['Var_Odd_H'].clip(lower=0) * 0.3) +
        (df['Tendencia_Forca_H'] * 0.3)
    )
    
    df['Value_Score_A'] = (
        (df['Gap_Prob_Odd_A'].clip(lower=0) * 0.4) +
        (df['Var_Odd_A'].clip(lower=0) * 0.3) +
        (df['Tendencia_Forca_A'] * 0.3)
    )
    
    return df

def calcular_target_profit(row):
    try:
        if row['ML_Side'] == 'HOME' and row['Quadrante_Correct']:
            return row.get('Profit_Quadrante', 0)
        elif row['ML_Side'] == 'AWAY' and row['Quadrante_Correct']:
            return row.get('Profit_Quadrante', 0)
        else:
            return row.get('Profit_Quadrante', -1)
    except:
        return 0

def gerar_recomendacao_mercado(row):
    prob_value = row.get('Prob_Value_Mercado', 0)
    confianca = row.get('Confianca_Mercado', 0)
    value_score_h = row.get('Value_Score_H', 0)
    value_score_a = row.get('Value_Score_A', 0)
    
    if prob_value >= 0.75 and confianca >= 0.7:
        if value_score_h > value_score_a:
            return f"🔥 VALUE MÁXIMO MERCADO HOME (Conf: {prob_value:.1%})"
        else:
            return f"🔥 VALUE MÁXIMO MERCADO AWAY (Conf: {prob_value:.1%})"
    
    elif prob_value >= 0.65 and confianca >= 0.6:
        if value_score_h > value_score_a:
            return f"🎯 VALUE ALTO MERCADO HOME (Conf: {prob_value:.1%})"
        else:
            return f"🎯 VALUE ALTO MERCADO AWAY (Conf: {prob_value:.1%})"
    
    elif prob_value >= 0.55:
        if value_score_h > value_score_a:
            return f"📈 VALUE MERCADO HOME (Conf: {prob_value:.1%})"
        else:
            return f"📈 VALUE MERCADO AWAY (Conf: {prob_value:.1%})"
    
    elif prob_value < 0.45:
        return f"⚖️ MERCADO NEUTRO (Conf: {prob_value:.1%})"
    
    else:
        return f"🔍 ANALISAR MERCADO (Conf: {prob_value:.1%})"

def treinar_ml_mercado(history, games_today):
    st.markdown("### 📊 ML de Análise de Mercado")
    
    # Preparar dados históricos para ML de mercado
    history_mercado = history.copy()
    
    # Garantir colunas de odds
    for col in ['Odd_H_OP', 'Odd_A_OP', 'Odd_H', 'Odd_A', 'Odd_H_Asi', 'Odd_A_Asi']:
        if col not in history_mercado.columns:
            history_mercado[col] = np.nan
    
    # Calcular features de mercado
    history_mercado = calcular_features_mercado(history_mercado)
    
    # Target: Profit real obtido
    history_mercado['Target_Profit'] = history_mercado.apply(calcular_target_profit, axis=1)
    history_mercado['Target_Value'] = (history_mercado['Target_Profit'] > 0).astype(int)
    
    # Features para ML de mercado
    features_mercado = [
        'Var_Odd_H', 'Var_Odd_A', 'Volatilidade_Odd_H', 'Volatilidade_Odd_A',
        'Odd_H_OP', 'Odd_A_OP', 'Gap_Prob_Odd_H', 'Gap_Prob_Odd_A',
        'Tendencia_Forca_H', 'Tendencia_Forca_A', 'Value_Score_H', 'Value_Score_A'
    ]
    
    # Filtrar dados válidos
    valid_data = history_mercado[features_mercado + ['Target_Value']].dropna()
    
    if len(valid_data) < 50:
        st.warning("⚠️ Dados insuficientes para treinar ML de mercado")
        return None, games_today
    
    X_mercado = valid_data[features_mercado]
    y_mercado = valid_data['Target_Value']
    
    # Treinar ML de mercado
    model_mercado = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model_mercado.fit(X_mercado, y_mercado)
    
    # Aplicar nos jogos de hoje
    games_today = calcular_features_mercado(games_today)
    X_today_mercado = games_today[features_mercado].fillna(0)
    
    # Previsões de value
    proba_value = model_mercado.predict_proba(X_today_mercado)[:, 1]
    games_today['Prob_Value_Mercado'] = proba_value
    games_today['Confianca_Mercado'] = np.maximum(proba_value, 1-proba_value)
    
    # Recomendações da ML de mercado
    games_today['Recomendacao_Mercado'] = games_today.apply(gerar_recomendacao_mercado, axis=1)
    
    st.success(f"✅ ML de Mercado treinada: {len(valid_data)} amostras")
    st.metric("Acurácia (Treino)", f"{model_mercado.score(X_mercado, y_mercado):.1%}")
    
    return model_mercado, games_today



##############################

#Bloco 10 - Sistema Híbrido e Recomendações


def gerar_recomendacao_3d_16_dual(row):
    home_q = row['Quadrante_Home_Label']
    away_q = row['Quadrante_Away_Label']
    score_home = row['Quadrante_ML_Score_Home']
    score_away = row['Quadrante_ML_Score_Away']
    ml_side = row['ML_Side']
    momentum_h = row.get('M_H', 0)
    momentum_a = row.get('M_A', 0)

    # Padrões 3D específicos incorporando momentum
    if 'Fav Forte' in home_q and 'Under Forte' in away_q and momentum_h > 1.0:
        return f'💪 FAVORITO HOME SUPER FORTE (+Momentum) ({score_home:.1%})'
    elif 'Under Forte' in home_q and 'Fav Forte' in away_q and momentum_a > 1.0:
        return f'💪 FAVORITO AWAY SUPER FORTE (+Momentum) ({score_away:.1%})'
    elif 'Fav Moderado' in home_q and 'Under Moderado' in away_q and momentum_h > 0.5:
        return f'🎯 VALUE NO HOME (+Momentum) ({score_home:.1%})'
    elif 'Under Moderado' in home_q and 'Fav Moderado' in away_q and momentum_a > 0.5:
        return f'🎯 VALUE NO AWAY (+Momentum) ({score_away:.1%})'
    elif ml_side == 'HOME' and score_home >= 0.60 and momentum_h > 0:
        return f'📈 MODELO CONFIA HOME (+Momentum) ({score_home:.1%})'
    elif ml_side == 'AWAY' and score_away >= 0.60 and momentum_a > 0:
        return f'📈 MODELO CONFIA AWAY (+Momentum) ({score_away:.1%})'
    elif momentum_h < -1.0 and score_away >= 0.55:
        return f'🔻 HOME EM MOMENTUM NEGATIVO → AWAY ({score_away:.1%})'
    elif momentum_a < -1.0 and score_home >= 0.55:
        return f'🔻 AWAY EM MOMENTUM NEGATIVO → HOME ({score_home:.1%})'
    elif 'Neutro' in home_q and score_away >= 0.58 and momentum_a > 0:
        return f'🔄 AWAY EM NEUTRO (+Momentum) ({score_away:.1%})'
    elif 'Neutro' in away_q and score_home >= 0.58 and momentum_h > 0:
        return f'🔄 HOME EM NEUTRO (+Momentum) ({score_home:.1%})'
    # NOVA CONDIÇÃO: Score alto mas momentum baixo
    elif score_home >= 0.75 and momentum_h >= -0.5:
        return f'🎯 VALUE HOME (Score Alto) ({score_home:.1%})'
    elif score_away >= 0.75 and momentum_a >= -0.5:
        return f'🎯 VALUE AWAY (Score Alto) ({score_away:.1%})'
    else:
        return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'

def sistema_hibrido_final(row):
    recomendacao_esportiva = row.get('Recomendacao', '')
    recomendacao_mercado = row.get('Recomendacao_Mercado', '')
    prob_esportiva = max(row.get('Quadrante_ML_Score_Home', 0), 
                        row.get('Quadrante_ML_Score_Away', 0))
    prob_mercado = row.get('Prob_Value_Mercado', 0)
    
    # CASO 1: Ambas concordam em VALUE MÁXIMO
    if ('🔥 VALUE MÁXIMO' in recomendacao_esportiva and 
        '🔥 VALUE MÁXIMO' in recomendacao_mercado):
        return f"🚀 CONVICÇÃO MÁXIMA: {recomendacao_esportiva}"
    
    # CASO 2: ML Esportiva forte + Mercado neutro
    elif ('🔥' in recomendacao_esportiva or '🎯' in recomendacao_esportiva) and prob_esportiva >= 0.75:
        return f"💪 FORÇA ESPORTIVA: {recomendacao_esportiva}"
    
    # CASO 3: Mercado forte + Esportiva neutra
    elif ('🔥' in recomendacao_mercado or '🎯' in recomendacao_mercado) and prob_mercado >= 0.75:
        return f"📊 INTELIGÊNCIA MERCADO: {recomendacao_mercado}"
    
    # CASO 4: Conflito (uma alerta, outra value)
    elif ('🔴' in recomendacao_esportiva and '🎯' in recomendacao_mercado):
        return f"⚖️ CONFLITO: Mercado vê value ({recomendacao_mercado})"
    
    elif ('🔴' in recomendacao_mercado and '🎯' in recomendacao_esportiva):
        return f"⚖️ CONFLITO: ML vê value ({recomendacao_esportiva})"
    
    # CASO 5: Ambas neutras
    elif '⚖️' in recomendacao_esportiva and '⚖️' in recomendacao_mercado:
        return "🤝 CONSENSO NEUTRO: Ambas MLs não detectam value claro"
    
    # Default: priorizar esportiva
    return recomendacao_esportiva



##############################################
#Bloco 11 - Sistema Live Score e Handicap

def handicap_favorito_v9(margin, line):
    line_abs = abs(line)
    
    if line_abs.is_integer():
        if margin > line_abs:
            return 1
        elif margin == line_abs:
            return 0
        else:
            return -1
    elif line == -0.25:
        if margin > 0:
            return 1
        elif margin == 0:
            return -0.5
        else:
            return -1
    elif line == -0.50:
        if margin > 0:
            return 1
        else:
            return -1
    elif line == -0.75:
        if margin >= 2:
            return 1
        elif margin == 1:
            return 0.5
        else:
            return -1
    elif line == -1.25:
        if margin >= 2:
            return 1
        elif margin == 1:
            return -0.5
        else:
            return -1
    elif line == -1.50:
        if margin >= 2:
            return 1
        else:
            return -1
    elif line == -1.75:
        if margin >= 3:
            return 1
        elif margin == 2:
            return 0.5
        else:
            return -1
    elif line == -2.00:
        if margin > 2:
            return 1
        elif margin == 2:
            return 0
        else:
            return -1
    return np.nan

def handicap_underdog_v9(margin, line):
    if line.is_integer():
        if margin >= -line:
            return 1
        elif margin == -(line + 1):
            return 0
        else:
            return -1
    elif line == 0.25:
        if margin > 0:
            return 1
        elif margin == 0:
            return 0.5
        else:
            return -1
    elif line == 0.50:
        if margin >= 0:
            return 1
        else:
            return -1
    elif line == 0.75:
        if margin >= 0:
            return 1
        elif margin == -1:
            return -0.5
        else:
            return -1
    elif line == 1.00:
        if margin >= -1:
            return 1
        else:
            return -1
    elif line == 1.25:
        if margin >= -1:
            return 1
        elif margin == -2:
            return 0.5
        else:
            return -1
    elif line == 1.50:
        if margin >= -1:
            return 1
        else:
            return -1
    elif line == 1.75:
        if margin >= -1:
            return 1
        elif margin == -2:
            return -0.5
        else:
            return -1
    elif line == 2.00:
        if margin >= -2:
            return 1
        elif margin == -3:
            return 0
        else:
            return -1
    return np.nan

def handicap_home_v9(row):
    margin = row['Goals_H_Today'] - row['Goals_A_Today']
    line = row['Asian_Line_Decimal']
    
    if line < 0:
        return handicap_favorito_v9(margin, line)
    else:
        return handicap_underdog_v9(margin, line)

def handicap_away_v9(row):
    margin = row['Goals_A_Today'] - row['Goals_H_Today']
    line = -row['Asian_Line_Decimal']
    
    if line < 0:
        return handicap_favorito_v9(margin, line)
    else:
        return handicap_underdog_v9(margin, line)

def determine_handicap_result_3d(row):
    try:
        gh = float(row['Goals_H_Today']) if pd.notna(row['Goals_H_Today']) else np.nan
        ga = float(row['Goals_A_Today']) if pd.notna(row['Goals_A_Today']) else np.nan
        asian_line = float(row['Asian_Line_Decimal'])
        recomendacao = str(row.get('Recomendacao', '')).upper()
    except (ValueError, TypeError):
        return None

    if pd.isna(gh) or pd.isna(ga) or pd.isna(asian_line):
        return None

    is_home_bet = any(k in recomendacao for k in [
        'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
        'MODELO CONFIA HOME', 'H:', 'HOME)'
    ])
    is_away_bet = any(k in recomendacao for k in [
        'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY',
        'MODELO CONFIA AWAY', 'A:', 'AWAY)'
    ])

    if not is_home_bet and not is_away_bet:
        return None

    if is_home_bet:
        outcome = handicap_home_v9(row)
    else:
        outcome = handicap_away_v9(row)

    if outcome == 1:
        return "FULL_WIN"
    elif outcome == 0.5:
        return "HALF_WIN" 
    elif outcome == 0:
        return "PUSH"
    elif outcome == -0.5:
        return "HALF_LOSS"
    elif outcome == -1:
        return "LOSS"
    else:
        return None


#####################################
#Bloco 12 - Sistema de Monitoramento e Execução Final

def check_handicap_recommendation_correct_3d(recomendacao, handicap_result):
    if pd.isna(recomendacao) or handicap_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
        return None
    return handicap_result in ["FULL_WIN", "HALF_WIN"]

def calculate_handicap_profit_3d(recomendacao, handicap_result, odds_row):
    if pd.isna(recomendacao) or handicap_result is None or '⚖️ ANALISAR' in str(recomendacao).upper():
        return 0

    recomendacao_str = str(recomendacao).upper()
    is_home_bet = any(k in recomendacao_str for k in [
        'HOME', '→ HOME', 'FAVORITO HOME', 'VALUE NO HOME',
        'MODELO CONFIA HOME', 'H:', 'HOME)'
    ])
    is_away_bet = any(k in recomendacao_str for k in [
        'AWAY', '→ AWAY', 'FAVORITO AWAY', 'VALUE NO AWAY', 
        'MODELO CONFIA AWAY', 'A:', 'AWAY)'
    ])

    if is_home_bet:
        odd = odds_row.get('Odd_H_Asi', np.nan)
    elif is_away_bet:
        odd = odds_row.get('Odd_A_Asi', np.nan)
    else:
        return 0

    if pd.isna(odd):
        return 0

    if handicap_result == "FULL_WIN":
        return odd - 1
    elif handicap_result == "HALF_WIN":
        return (odd - 1) / 2
    elif handicap_result == "PUSH":
        return 0
    elif handicap_result == "HALF_LOSS":
        return -0.5
    elif handicap_result == "LOSS":
        return -1
    else:
        return 0

def update_real_time_data_3d(df):
    df = df.copy()
    df['Handicap_Result'] = df.apply(determine_handicap_result_3d, axis=1)
    df['Quadrante_Correct'] = df.apply(
        lambda r: check_handicap_recommendation_correct_3d(
            r['Recomendacao'], r['Handicap_Result']
        ), axis=1
    )
    df['Profit_Quadrante'] = df.apply(
        lambda r: calculate_handicap_profit_3d(
            r['Recomendacao'], r['Handicap_Result'], r
        ), axis=1
    )
    return df

def generate_live_summary_3d(df):
    finished_games = df[df['Handicap_Result'].notna()]
    
    if finished_games.empty:
        return {
            "Total Jogos": len(df),
            "Jogos Finalizados": 0,
            "Apostas Quadrante 3D": 0,
            "Acertos Quadrante 3D": 0,
            "Winrate Quadrante 3D": "0%",
            "Profit Quadrante 3D": "0.00u",
            "ROI Quadrante 3D": "0%",
            "Full Wins": 0,
            "Half Wins": 0, 
            "Pushes": 0,
            "Half Losses": 0,
            "Losses": 0
        }
    
    quadrante_bets = finished_games[finished_games['Quadrante_Correct'].notna()]
    total_bets = len(quadrante_bets)
    correct_bets = quadrante_bets['Quadrante_Correct'].sum()
    winrate = (correct_bets / total_bets) * 100 if total_bets > 0 else 0
    total_profit = quadrante_bets['Profit_Quadrante'].sum()
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    
    full_wins = (finished_games['Handicap_Result'] == 'FULL_WIN').sum()
    half_wins = (finished_games['Handicap_Result'] == 'HALF_WIN').sum()
    pushes = (finished_games['Handicap_Result'] == 'PUSH').sum()
    half_losses = (finished_games['Handicap_Result'] == 'HALF_LOSS').sum()
    losses = (finished_games['Handicap_Result'] == 'LOSS').sum()
    
    return {
        "Total Jogos": len(df),
        "Jogos Finalizados": len(finished_games),
        "Apostas Quadrante 3D": total_bets,
        "Acertos Quadrante 3D": int(correct_bets),
        "Winrate Quadrante 3D": f"{winrate:.1f}%",
        "Profit Quadrante 3D": f"{total_profit:.2f}u",
        "ROI Quadrante 3D": f"{roi:.1f}%",
        "Full Wins": int(full_wins),
        "Half Wins": int(half_wins),
        "Pushes": int(pushes),
        "Half Losses": int(half_losses),
        "Losses": int(losses)
    }

# EXECUÇÃO PRINCIPAL
st.markdown("## 🧠 Sistema Híbrido: 2 MLs Especializadas")

# Executar ambas as MLs
modelo_esportivo, games_today = treinar_modelo_3d_clusters_single(history, games_today)
modelo_mercado, games_today = treinar_ml_mercado(history, games_today)

# Gerar recomendações iniciais
games_today['Recomendacao_Original'] = games_today.apply(gerar_recomendacao_3d_16_dual, axis=1)

# Aplicar sistema híbrido se ML de mercado foi treinada
if modelo_mercado is not None:
    games_today['Recomendacao_Hibrida'] = games_today.apply(sistema_hibrido_final, axis=1)
    games_today['Recomendacao'] = games_today['Recomendacao_Hibrida']
    st.success("✅ Sistema híbrido executado com sucesso!")
else:
    games_today['Recomendacao'] = games_today['Recomendacao_Original']
    st.info("ℹ️ Usando apenas ML Esportiva (ML Mercado não treinada)")

# Aplicar sistema live score
games_today = update_real_time_data_3d(games_today)

st.success("🎯 Sistema completo executado com sucesso!")



##########################################















