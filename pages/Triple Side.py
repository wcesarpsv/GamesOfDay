# ==============================================================
# 💸 ROI Focus 1X2 – Triple Side + Live Validation
#    (Home / Draw / Away Expected Value on 1X2 Market)
# ==============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# ============================ Config ============================
st.set_page_config(page_title="ROI Focus 1X2 – Triple Side + Live Validation", layout="wide")
st.title("💸 ROI Focus 1X2 – Triple Side (Home/Draw/Away) + Live Validation")

GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(GAMES_FOLDER, exist_ok=True)
os.makedirs(LIVESCORE_FOLDER, exist_ok=True)

# ============================ Helpers ============================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # normaliza nomes de gols FT (merge antigos)
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

# ---------------- 1X2 Helpers ----------------
def normalize_odds_value(odd: float) -> float:
    """
    Converte qualquer odd para LUCRO LÍQUIDO por stake=1.
    - Se odd <= 1.0: já é líquida (ex.: 0.90) => retorna 0.90
    - Se odd >  1.0: é decimal (ex.: 1.90)    => retorna 0.90
    """
    if pd.isna(odd):
        return np.nan
    return odd if odd <= 1.0 else (odd - 1.0)

def result_1x2_from_goals(gh, ga):
    if pd.isna(gh) or pd.isna(ga):
        return None
    gh = float(gh); ga = float(ga)
    if gh > ga: return "H"
    if gh < ga: return "A"
    return "D"

def profit_1x2(result: str, side: str, odd_value: float) -> float:
    """
    Retorna lucro líquido por stake=1 com base em resultado 1X2.
    side ∈ {"H","D","A"}; odd_value pode ser líquida (0.90) ou decimal (1.90).
    """
    if result is None or pd.isna(odd_value):
        return np.nan
    eff = normalize_odds_value(odd_value)
    return eff if side == result else -1.0

# ============== 3D features (diferenças, trig, magnitude) ==============
def calcular_distancias_3d(df):
    df = df.copy()
    req = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    for c in req:
        if c not in df.columns:
            df[c] = 0.0  # fallback neutro

    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3

    a_xy = np.arctan2(dy, dx)
    a_xz = np.arctan2(dz, dx)
    # evite divisão por zero (dy==0) na projeção YZ
    dy_safe = dy.copy()
    dy_safe = dy_safe.replace(0, 1e-9)
    a_yz = np.arctan2(dz, dy_safe)

    df['Quadrant_Sin_XY'] = np.sin(a_xy); df['Quadrant_Cos_XY'] = np.cos(a_xy)
    df['Quadrant_Sin_XZ'] = np.sin(a_xz); df['Quadrant_Cos_XZ'] = np.cos(a_xz)
    df['Quadrant_Sin_YZ'] = np.sin(a_yz); df['Quadrant_Cos_YZ'] = np.cos(a_yz)

    combo = a_xy + a_xz + a_yz
    df['Quadrant_Sin_Combo'] = np.sin(combo); df['Quadrant_Cos_Combo'] = np.cos(combo)
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    return df

def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    df = df.copy()
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']
    Xc = df[['dx','dy','dz']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df['Cluster3D_Label'] = kmeans.fit_predict(Xc)
    return df

# ====================== LiveScore merge ======================
def setup_livescore_columns(df):
    if 'Goals_H_Today' not in df.columns: df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns: df['Goals_A_Today'] = np.nan
    if 'Home_Red'      not in df.columns: df['Home_Red']      = np.nan
    if 'Away_Red'      not in df.columns: df['Away_Red']      = np.nan
    return df

def load_and_merge_livescore(games_today, selected_date_str):
    """
    Espera arquivo: LiveScore/Resultados_RAW_YYYY-MM-DD.csv (colunas padrão do teu scraper).
    """
    games_today = setup_livescore_columns(games_today)
    fp = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    if not os.path.exists(fp):
        st.warning(f"⚠️ LiveScore não encontrado para {selected_date_str} em {fp}")
        return games_today
    try:
        raw = pd.read_csv(fp)
    except Exception as e:
        st.error(f"Erro ao ler LiveScore: {e}")
        return games_today

    if 'status' in raw.columns:
        raw = raw[~raw['status'].isin(['Cancel', 'Postp.'])]

    left_key = 'Id' if 'Id' in games_today.columns else None
    right_key = 'game_id' if 'game_id' in raw.columns else None
    if left_key and right_key:
        games_today = games_today.merge(raw, left_on=left_key, right_on=right_key, how='left', suffixes=('', '_RAW'))

    # mapeia colunas comuns
    for src, dst in [('home_goal','Goals_H_Today'), ('away_goal','Goals_A_Today'),
                     ('home_red','Home_Red'), ('away_red','Away_Red')]:
        if src in games_today.columns:
            games_today[dst] = games_today[src]

    # se não finalizado, zera gols do dia (mantém NaN)
    if 'status' in games_today.columns:
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today','Goals_A_Today']] = np.nan

    st.success(f"✅ LiveScore mesclado ({len(raw)} linhas)")
    return games_today

# ============================ Carregamento ============================
st.info("📂 Carregando dados 1X2...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Selecione o arquivo do dia:", options, index=len(options)-1)

# data pela regex no nome
m = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = m.group(0) if m else datetime.now().strftime("%Y-%m-%d")

# jogos do dia
games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# histórico
history = filter_leagues(load_all_games(GAMES_FOLDER))
# Para 1X2, precisamos dos gols FT e odds 1X2
req_cols = ["Goals_H_FT", "Goals_A_FT", "Odd_H", "Odd_D", "Odd_A"]
history = history.dropna(subset=[c for c in req_cols if c in history.columns]).copy()

# filtro temporal
if "Date" in history.columns:
    try:
        sel_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < sel_date].copy()
    except Exception as e:
        st.warning(f"Erro ao aplicar filtro temporal: {e}")

# LiveScore (opcional) – para validação em tempo real
games_today = load_and_merge_livescore(games_today, selected_date_str)

# ===================== Targets EV 1X2 (histórico) =====================
st.markdown("### 🎯 Construindo targets de lucro esperado (1X2 histórico)")

# Resultado 1X2 no histórico
history['Result_1X2'] = history.apply(lambda r: result_1x2_from_goals(r.get('Goals_H_FT'), r.get('Goals_A_FT')), axis=1)

# Targets por lado (usando odds 1X2)
history['Target_EV_Home'] = history.apply(lambda r: profit_1x2(r['Result_1X2'], "H", r.get('Odd_H', np.nan)), axis=1)
history['Target_EV_Draw'] = history.apply(lambda r: profit_1x2(r['Result_1X2'], "D", r.get('Odd_D', np.nan)), axis=1)
history['Target_EV_Away'] = history.apply(lambda r: profit_1x2(r['Result_1X2'], "A", r.get('Odd_A', np.nan)), axis=1)

st.info(
    f"Histórico pronto: {len(history)} jogos | "
    f"EV médio H={pd.to_numeric(history['Target_EV_Home'], errors='coerce').mean():.3f} | "
    f"D={pd.to_numeric(history['Target_EV_Draw'], errors='coerce').mean():.3f} | "
    f"A={pd.to_numeric(history['Target_EV_Away'], errors='coerce').mean():.3f}"
)

# ===================== Feature engineering 3D =====================
def ensure_3d_features(df):
    df = calcular_distancias_3d(df)
    df = aplicar_clusterizacao_3d(df, n_clusters=5)
    return df

history = ensure_3d_features(history)
games_today = ensure_3d_features(games_today)

# ===================== Treinamento – regressão triple EV =====================
st.markdown("### 🤖 Treinando modelos de EV 1X2 (H / D / A)")

def train_triple_ev_models(history, games_today):
    # dummies categóricas
    ligas_d = pd.get_dummies(history['League'], prefix='League') if 'League' in history.columns else pd.DataFrame()
    clusters_d = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    feat3d = [
        'Quadrant_Dist_3D','Quadrant_Separation_3D',
        'Quadrant_Sin_XY','Quadrant_Cos_XY','Quadrant_Sin_XZ','Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ','Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo','Quadrant_Cos_Combo','Vector_Sign','Magnitude_3D'
    ]
    extras = history[feat3d].fillna(0)

    X = pd.concat([ligas_d, clusters_d, extras], axis=1)

    # targets (drop NaN para treinar limpo)
    y_H = history['Target_EV_Home']; mask_H = y_H.notna()
    y_D = history['Target_EV_Draw']; mask_D = y_D.notna()
    y_A = history['Target_EV_Away']; mask_A = y_A.notna()

    model_H = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_D = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_A = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)

    if mask_H.any(): model_H.fit(X.loc[mask_H], y_H.loc[mask_H])
    if mask_D.any(): model_D.fit(X.loc[mask_D], y_D.loc[mask_D])
    if mask_A.any(): model_A.fit(X.loc[mask_A], y_A.loc[mask_A])

    # preparar X_today
    ligas_today = pd.get_dummies(games_today['League'], prefix='League') if 'League' in games_today.columns else pd.DataFrame()
    ligas_today = ligas_today.reindex(columns=ligas_d.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_d.columns, fill_value=0)
    extras_today = games_today[feat3d].fillna(0)
    X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    # previsões
    if mask_H.any():
        games_today['Predicted_EV_H'] = model_H.predict(X_today)
    else:
        games_today['Predicted_EV_H'] = 0.0
    if mask_D.any():
        games_today['Predicted_EV_D'] = model_D.predict(X_today)
    else:
        games_today['Predicted_EV_D'] = 0.0
    if mask_A.any():
        games_today['Predicted_EV_A'] = model_A.predict(X_today)
    else:
        games_today['Predicted_EV_A'] = 0.0

    # escolha do lado
    preds = games_today[['Predicted_EV_H','Predicted_EV_D','Predicted_EV_A']].copy()
    games_today['Chosen_Side'] = preds.idxmax(axis=1).str[-1]  # "H","D","A"
    games_today['Predicted_EV'] = preds.max(axis=1)

    # Importâncias (Home) – se treinado
    try:
        importances = pd.Series(model_H.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        st.markdown("#### 🔍 Top Features (Modelo EV – HOME)")
        st.dataframe(importances.to_frame("Importância"), use_container_width=True)
    except Exception:
        pass

    st.success("✅ Modelos EV 1X2 treinados (H / D / A)")
    return (model_H, model_D, model_A), games_today

(models_HDA), games_today = train_triple_ev_models(history, games_today)

# ===================== Filtros de UI =====================
st.markdown("### 🎛️ Filtros")
leagues = sorted(games_today['League'].dropna().unique()) if 'League' in games_today.columns else []
col_f1, col_f2 = st.columns(2)
selected_league = col_f
