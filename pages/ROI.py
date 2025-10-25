# ==============================================================
# 💸 ROI Focus 3D – Expected Value Prediction (Home Handicap)
# ==============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# ==============================================================
# ⚙️ Configurações
# ==============================================================
st.set_page_config(page_title="ROI Focus 3D – Bet Indicator", layout="wide")
st.title("💸 ROI-Focused 3D Model – Expected Value Prediction (Home Handicap)")

PAGE_PREFIX = "ROI_3D"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(GAMES_FOLDER, exist_ok=True)
os.makedirs(LIVESCORE_FOLDER, exist_ok=True)

# ==============================================================
# 🧩 Helpers
# ==============================================================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
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

def convert_asian_line_to_decimal(line_str):
    if pd.isna(line_str) or line_str == "":
        return None
    try:
        line_str = str(line_str).strip()
        if "/" not in line_str:
            return float(line_str)
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
    except (ValueError, TypeError):
        return None

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

# ==============================================================
# 📂 Carregar dados
# ==============================================================
st.info("📂 Carregando dados para modelo ROI...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

date_match = datetime.now().strftime("%Y-%m-%d")
for pattern in ["%Y-%m-%d", "%d-%m-%Y"]:
    try:
        date_match = datetime.strptime(selected_file[:10], pattern).strftime("%Y-%m-%d")
        break
    except:
        pass

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

# Filtro temporal (só jogos anteriores à data atual)
try:
    selected_date = pd.to_datetime(date_match)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history = history[history["Date"] < selected_date].copy()
except Exception as e:
    st.warning(f"Erro ao aplicar filtro temporal: {e}")

# ==============================================================
# 🧮 Calcular Target baseado em ROI
# ==============================================================
history["Asian_Line_Decimal"] = history["Asian_Line"].apply(convert_asian_line_to_decimal)

def calc_profit_home(row):
    result = calc_handicap_result(row["Goals_H_FT"] - row["Goals_A_FT"], row["Asian_Line"], invert=False)
    odd_home = row.get("Odd_H_Asi", np.nan)
    if pd.isna(odd_home):
        return np.nan
    if result > 0.5:
        return odd_home
    elif result == 0.5:
        return 0
    else:
        return -1

history["Target_EV_Home"] = history.apply(calc_profit_home, axis=1)
st.info(f"🎯 Target ROI calculado – média: {history['Target_EV_Home'].mean():.3f}")

# ==============================================================
# 📊 Funções 3D (distâncias e clusterização)
# ==============================================================
def calcular_distancias_3d(df):
    df = df.copy()
    req = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    if any(c not in df.columns for c in req):
        for x in ['Quadrant_Dist_3D','Quadrant_Separation_3D','Magnitude_3D']:
            df[x] = np.nan
        return df

    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Quadrant_Sin_XY'] = np.sin(np.arctan2(dy, dx))
    df['Quadrant_Cos_XY'] = np.cos(np.arctan2(dy, dx))
    df['Quadrant_Sin_XZ'] = np.sin(np.arctan2(dz, dx))
    df['Quadrant_Cos_XZ'] = np.cos(np.arctan2(dz, dx))
    df['Quadrant_Sin_YZ'] = np.sin(np.arctan2(dz, dy))
    df['Quadrant_Cos_YZ'] = np.cos(np.arctan2(dz, dy))
    df['Quadrant_Sin_Combo'] = np.sin(np.arctan2(dy, dx) + np.arctan2(dz, dx) + np.arctan2(dz, dy))
    df['Quadrant_Cos_Combo'] = np.cos(np.arctan2(dy, dx) + np.arctan2(dz, dx) + np.arctan2(dz, dy))
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    return df

def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    df = df.copy()
    req = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    if any(c not in df.columns for c in req):
        df['Cluster3D_Label'] = -1
        return df
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']
    X = df[['dx','dy','dz']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df['Cluster3D_Label'] = kmeans.fit_predict(X)
    return df

# ==============================================================
# 🤖 Treinar modelo regressivo de ROI
# ==============================================================
def treinar_modelo_roi_3d(history, games_today):
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)
    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    clusters_dummies = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    features_3d = [
        'Quadrant_Dist_3D','Quadrant_Separation_3D',
        'Quadrant_Sin_XY','Quadrant_Cos_XY','Quadrant_Sin_XZ','Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ','Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo','Quadrant_Cos_Combo','Vector_Sign','Magnitude_3D'
    ]
    extras_3d = history[features_3d].fillna(0)

    X = pd.concat([ligas_dummies, clusters_dummies, extras_3d], axis=1)
    y = history['Target_EV_Home'].fillna(0)

    model_roi = RandomForestRegressor(
        n_estimators=500, max_depth=12, random_state=42, n_jobs=-1
    )
    model_roi.fit(X, y)

    # Dados de hoje
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=ligas_dummies.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_dummies.columns, fill_value=0)
    extras_today = games_today[features_3d].fillna(0)
    X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    games_today['Predicted_EV_Home'] = model_roi.predict(X_today)
    games_today['Predicted_EV_Away'] = -games_today['Predicted_EV_Home']
    games_today['ROI_Side'] = np.where(games_today['Predicted_EV_Home'] > 0, 'HOME', 'AWAY')
    games_today['Predicted_ROI'] = games_today['Predicted_EV_Home']

    importances = pd.Series(model_roi.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.markdown("### 🔍 Top Features (Modelo Regressivo ROI)")
    st.dataframe(importances.head(15).to_frame("Importância"), use_container_width=True)

    st.success("✅ Modelo Regressivo 3D de ROI treinado com sucesso!")
    return model_roi, games_today

# ==============================================================
# 🚀 Execução Principal
# ==============================================================
if not history.empty:
    model_roi, games_today = treinar_modelo_roi_3d(history, games_today)
else:
    st.warning("Histórico vazio – impossível treinar modelo ROI.")

# ==============================================================
# 📈 Simulação de ROI Previsto
# ==============================================================
roi_df = games_today.copy()
roi_df['Bet'] = np.where(roi_df['Predicted_EV_Home'] > 0, 1, 0)
roi_df['Profit_Sim'] = np.where(roi_df['Predicted_EV_Home'] > 0, roi_df['Predicted_EV_Home'], 0)

total_bets = roi_df['Bet'].sum()
total_profit = roi_df['Profit_Sim'].sum()
roi_mean = (total_profit / total_bets) if total_bets > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Total Bets", int(total_bets))
col2.metric("💰 Total Profit (u)", f"{total_profit:.2f}")
col3.metric("📊 ROI Médio Previsto", f"{roi_mean*100:.2f}%")

# ==============================================================
# 🏆 Ranking por Expected Value (ROI-Focused)
# ==============================================================
st.markdown("## 🏆 Ranking por Expected Value (ROI-Focused)")
ranking_roi = games_today.sort_values('Predicted_EV_Home', ascending=False).head(30)
st.dataframe(
    ranking_roi[['League','Home','Away','Predicted_EV_Home','ROI_Side','Quadrant_Dist_3D']]
    .style.background_gradient(subset=['Predicted_EV_Home'], cmap='RdYlGn')
    .format({'Predicted_EV_Home': '{:.2f}', 'Quadrant_Dist_3D': '{:.2f}'}),
    use_container_width=True
)

# ==============================================================
# 🧾 Conclusão
# ==============================================================
st.markdown("---")
st.success("💹 Modelo 3D focado em ROI implementado com sucesso!")
st.info("""
**Diferenciais deste modelo:**
- 🎯 Target contínuo baseado em lucro real (EV)
- 🤖 RandomForestRegressor otimizado para ROI
- 📊 Avaliação direta em Profit e ROI médio
- 🧩 Mantém todas as dimensões 3D (Aggression × M × MT)
- 🧠 Integra clusterização espacial (KMeans 3D)
""")
