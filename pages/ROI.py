# ==============================================================
# 💸 ROI Focus 3D – Dual Side + Live Validation (Asian Handicap)
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
st.set_page_config(page_title="ROI Focus 3D – Dual Side + Live Validation", layout="wide")
st.title("💸 ROI-Focused 3D – Dual Side (Home & Away) + Live Validation")

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

def convert_asian_line_to_decimal(line_str):
    if pd.isna(line_str) or line_str == "":
        return None
    try:
        s = str(line_str).strip()
        if "/" not in s:
            return float(s)
        parts = [float(x) for x in s.split("/")]
        return sum(parts) / len(parts)
    except (ValueError, TypeError):
        return None

def split_quarter_line(decimal_line: float):
    """
    Decompõe linha asiática decimal em 1 ou 2 componentes p/ avaliar meia vitória/derrota corretamente.
    Ex.: +0.25 -> [0, +0.5] | -0.75 -> [-0.5, -1.0]
    """
    if pd.isna(decimal_line):
        return []
    frac = abs(decimal_line) % 1
    sign = 1 if decimal_line >= 0 else -1
    base = math.floor(abs(decimal_line)) * sign
    if math.isclose(frac, 0.25, abs_tol=1e-9):
        return [base, base + 0.5 * sign]
    elif math.isclose(frac, 0.75, abs_tol=1e-9):
        return [base + 0.5 * sign, base + 1.0 * sign]
    else:
        return [decimal_line]

def get_odd(row: pd.Series, side: str):
    """
    Tenta buscar Odd_H_Asi / Odd_A_Asi, com fallback para Odd_H / Odd_A.
    """
    if side == "HOME":
        for c in ["Odd_H_Asi", "Odd_H"]:
            if c in row and pd.notna(row[c]): return float(row[c])
    else:
        for c in ["Odd_A_Asi", "Odd_A"]:
            if c in row and pd.notna(row[c]): return float(row[c])
    return np.nan

def profit_from_line(margin: float, line_component: float, odd: float, side: str):
    """
    Calcula o lucro em UMA componente de linha (sem média ainda).
    - margin = Goals_H - Goals_A
    - line_component = componente da linha em relação ao HOME
    - side = "HOME" ou "AWAY"
    """
    # Resultado em relação ao HOME
    if side == "HOME":
        if margin > line_component:   # HOME cobriu
            return odd - 1
        elif math.isclose(margin, line_component, abs_tol=1e-9):  # PUSH
            return 0
        else:
            return -1
    else:  # AWAY
        if margin < line_component:   # AWAY cobriu (HOME não cobriu)
            return odd - 1
        elif math.isclose(margin, line_component, abs_tol=1e-9):  # PUSH
            return 0
        else:
            return -1

def calc_ev_side(margin: float, asian_line_decimal: float, odd: float, side: str):
    """
    Lucro (unidade) da aposta em 'side' dado o placar (margin) e a linha decimal.
    Trata 0.25/0.75 via média das componentes.
    """
    if pd.isna(margin) or pd.isna(asian_line_decimal) or pd.isna(odd):
        return np.nan
    components = split_quarter_line(asian_line_decimal)
    if not components:
        return np.nan
    profits = [profit_from_line(margin, lc, odd, side) for lc in components]
    return float(np.mean(profits))

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
    # ângulos e projeções
    a_xy = np.arctan2(dy, dx)
    a_xz = np.arctan2(dz, dx)
    a_yz = np.arctan2(dz, dy.replace(0, 1e-9))
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

# ====================== LiveScore (opcional) ======================
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

    # remove cancelados/adiados, se existir coluna 'status'
    if 'status' in raw.columns:
        raw = raw[~raw['status'].isin(['Cancel', 'Postp.'])]

    # merge por Id ↔ game_id
    left_key = 'Id' if 'Id' in games_today.columns else None
    right_key = 'game_id' if 'game_id' in raw.columns else None
    if left_key and right_key:
        games_today = games_today.merge(raw, left_on=left_key, right_on=right_key, how='left', suffixes=('', '_RAW'))

    # mapeia colunas comuns
    for src, dst in [('home_goal','Goals_H_Today'), ('away_goal','Goals_A_Today'),
                     ('home_red','Home_Red'), ('away_red','Away_Red')]:
        if src in games_today.columns:
            games_today[dst] = games_today[src]

    # zera gols "não finalizados"
    if 'status' in games_today.columns:
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today','Goals_A_Today']] = np.nan

    st.success(f"✅ LiveScore mesclado ({len(raw)} linhas)")
    return games_today

# ============================ Carregamento ============================
st.info("📂 Carregando dados...")

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
games_today['Asian_Line_Decimal'] = games_today.get('Asian_Line', pd.Series([np.nan]*len(games_today))).apply(convert_asian_line_to_decimal)

# histórico
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()
history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)

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

# ===================== Targets EV (Home & Away) =====================
st.markdown("### 🎯 Construindo targets de lucro esperado (histórico)")

def compute_targets_ev(df_hist: pd.DataFrame):
    df = df_hist.copy()
    # margin true
    df['Margin'] = df['Goals_H_FT'].astype(float) - df['Goals_A_FT'].astype(float)

    # lucros reais do passado p/ cada lado
    ev_home = []
    ev_away = []
    for _, r in df.iterrows():
        odd_h = get_odd(r, "HOME")
        odd_a = get_odd(r, "AWAY")
        line = r.get('Asian_Line_Decimal', np.nan)
        margin = r['Margin']

        ev_home.append(calc_ev_side(margin, line, odd_h, "HOME"))
        ev_away.append(calc_ev_side(margin, line, odd_a, "AWAY"))

    df['Target_EV_Home'] = ev_home
    df['Target_EV_Away'] = ev_away
    return df

history = compute_targets_ev(history)

# Avisos básicos
st.info(
    f"Histórico pronto: {len(history)} jogos | "
    f"EV Home médio = {pd.to_numeric(history['Target_EV_Home'], errors='coerce').mean():.3f} | "
    f"EV Away médio = {pd.to_numeric(history['Target_EV_Away'], errors='coerce').mean():.3f}"
)

# ===================== Feature engineering 3D =====================
def ensure_3d_features(df):
    df = calcular_distancias_3d(df)
    df = aplicar_clusterizacao_3d(df, n_clusters=5)
    return df

history = ensure_3d_features(history)
games_today = ensure_3d_features(games_today)

# ===================== Treinamento – regressão dual =====================
st.markdown("### 🤖 Treinando modelos de EV (Home & Away)")

def train_dual_ev_models(history, games_today):
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
    y_home = history['Target_EV_Home']
    y_away = history['Target_EV_Away']
    mask_home = y_home.notna()
    mask_away = y_away.notna()

    model_home = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_away = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_home.fit(X.loc[mask_home], y_home.loc[mask_home])
    model_away.fit(X.loc[mask_away], y_away.loc[mask_away])

    # preparar X_today
    ligas_today = pd.get_dummies(games_today['League'], prefix='League') if 'League' in games_today.columns else pd.DataFrame()
    ligas_today = ligas_today.reindex(columns=ligas_d.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_d.columns, fill_value=0)
    extras_today = games_today[feat3d].fillna(0)
    X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    # previsões
    games_today['Predicted_EV_Home'] = model_home.predict(X_today)
    games_today['Predicted_EV_Away'] = model_away.predict(X_today)

    # melhor lado por EV
    games_today['Chosen_Side'] = np.where(
        games_today['Predicted_EV_Home'] >= games_today['Predicted_EV_Away'], 'HOME', 'AWAY'
    )
    games_today['Predicted_EV'] = games_today[['Predicted_EV_Home','Predicted_EV_Away']].max(axis=1)

    # Importâncias (Home)
    try:
        importances = pd.Series(model_home.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        st.markdown("#### 🔍 Top Features (Modelo EV – HOME)")
        st.dataframe(importances.to_frame("Importância"), use_container_width=True)
    except Exception:
        pass

    st.success("✅ Modelos EV treinados (Home & Away)")
    return model_home, model_away, games_today

model_home, model_away, games_today = train_dual_ev_models(history, games_today)

# ===================== Ranking por EV previsto =====================
st.markdown("## 🏆 Ranking por Expected Value (unidades por aposta)")
top_n = st.slider("Quantos confrontos exibir:", 10, min(200, len(games_today)), 40, step=5)
rank = games_today.sort_values('Predicted_EV', ascending=False).head(top_n)

st.dataframe(
    rank[['League','Home','Away','Chosen_Side','Predicted_EV_Home','Predicted_EV_Away','Predicted_EV','Quadrant_Dist_3D']]
    .style.background_gradient(subset=['Predicted_EV','Predicted_EV_Home','Predicted_EV_Away'], cmap='RdYlGn')
    .format({'Predicted_EV':'{:.2f}','Predicted_EV_Home':'{:.2f}','Predicted_EV_Away':'{:.2f}','Quadrant_Dist_3D':'{:.2f}'}),
    use_container_width=True
)

# ===================== Simulação – carteira prevista =====================
st.markdown("### 💼 Simulação de carteira prevista (aposta quando EV>0)")
sim = games_today.copy()
sim['Place_Bet'] = sim['Predicted_EV'] > 0
sim['Profit_Predicted'] = np.where(sim['Place_Bet'], sim['Predicted_EV'], 0.0)

tot_bets = int(sim['Place_Bet'].sum())
tot_pred_profit = sim['Profit_Predicted'].sum()
roi_pred_mean = (tot_pred_profit / tot_bets) if tot_bets > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Apostas previstas", tot_bets)
c2.metric("Profit Previsto (u)", f"{tot_pred_profit:.2f}")
c3.metric("ROI Médio Previsto", f"{roi_pred_mean*100:.2f}%")

# ===================== Live Validation – ROI real do dia =====================
st.markdown("## 📡 Live Validation – ROI real (gols do dia)")

def real_profit_for_row(row):
    gh = row.get('Goals_H_Today', np.nan)
    ga = row.get('Goals_A_Today', np.nan)
    line = row.get('Asian_Line_Decimal', np.nan)
    if pd.isna(gh) or pd.isna(ga) or pd.isna(line):
        return np.nan
    margin = float(gh) - float(ga)
    side = row['Chosen_Side']
    odd = get_odd(row, side)
    return calc_ev_side(margin, line, odd, side)

# lucro real apenas para jogos onde decidimos apostar
sim['Real_Profit'] = sim.apply(real_profit_for_row, axis=1)
finished_bets = sim[sim['Place_Bet'] & sim['Real_Profit'].notna()]

if finished_bets.empty:
    st.info("Ainda não há jogos finalizados com aposta prevista para calcular ROI real.")
else:
    real_total = finished_bets['Real_Profit'].sum()
    real_bets = len(finished_bets)
    real_roi = (real_total / real_bets) if real_bets > 0 else 0.0

    # por lado
    by_side = finished_bets.groupby('Chosen_Side')['Real_Profit'].agg(['count','sum'])
    by_side['ROI_mean'] = by_side['sum'] / by_side['count']

    c1, c2, c3 = st.columns(3)
    c1.metric("Apostas finalizadas", real_bets)
    c2.metric("Profit Real (u)", f"{real_total:.2f}")
    c3.metric("ROI Médio Real", f"{real_roi*100:.2f}%")

    st.markdown("### 🔍 Comparativo Previsto vs Real (jogos finalizados)")
    comp_cols = ['League','Home','Away','Chosen_Side','Predicted_EV','Goals_H_Today','Goals_A_Today','Real_Profit','Asian_Line_Decimal']
    show_cols = [c for c in comp_cols if c in finished_bets.columns]
    st.dataframe(
        finished_bets.sort_values('Predicted_EV', ascending=False)[show_cols]
        .style.background_gradient(subset=['Real_Profit','Predicted_EV'], cmap='RdYlGn')
        .format({'Predicted_EV':'{:.2f}','Real_Profit':'{:.2f}','Asian_Line_Decimal':'{:.2f}'})
    )

    st.markdown("#### 📊 Quebra por lado")
    st.dataframe(by_side.rename(columns={'count':'Apostas','sum':'Profit'}))

st.markdown("---")
st.success("💹 Pipeline ROI Dual + Live Validation pronto!")
st.info("""
**O que esta versão faz:**
- 🎯 Targets contínuos de lucro esperado para HOME e AWAY (historico)
- 🤖 Dois regressors (EV Home e EV Away) e escolha automática do melhor lado
- 📈 Ranking por EV previsto e simulação de carteira (EV>0)
- 📡 Validação em tempo real com gols do dia (Profit Real & ROI por lado)
- 🧩 Mantém todas as features 3D (Aggression × M × MT) + clusters KMeans
""")
