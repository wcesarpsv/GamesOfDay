# ==============================================================
# 💸 ROI Focus 1X2 – Triple Side + Live Validation + EV Ótimo Automático
# ==============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

# ============================ Config ============================
st.set_page_config(page_title="ROI Focus 1X2 – Triple Side + Live Validation", layout="wide")
st.title("💸 ROI Focus 1X2 – Triple Side (Home / Draw / Away) + Live Validation + EV Ótimo Automático")

GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "coppa", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(GAMES_FOLDER, exist_ok=True)
os.makedirs(LIVESCORE_FOLDER, exist_ok=True)

# ============================ Helpers ============================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def remove_duplicados_inteligente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicados de forma inteligente baseado em múltiplas chaves
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Contar duplicados antes
    total_antes = len(df_clean)
    
    # Tentar diferentes combinações de chaves para identificar duplicados
    chaves_duplicacao = [
        ['Id'],  # Chave principal
        ['Home', 'Away', 'League', 'Goals_H_FT','Goals_A_FT'],  # Chave mais específica
        ['Id'] if 'Id' in df_clean.columns else []  # Se tiver ID
    ]
    
    for chaves in chaves_duplicacao:
        if chaves and all(col in df_clean.columns for col in chaves):
            duplicados = df_clean.duplicated(subset=chaves, keep='first')
            if duplicados.any():
                st.warning(f"⚠️ Removidos {duplicados.sum()} duplicados por: {chaves}")
                df_clean = df_clean[~duplicados]
    
    # Remover linhas com dados essenciais missing
    colunas_essenciais = ['Home', 'Away', 'Goals_H_FT', 'Goals_A_FT']
    colunas_presentes = [col for col in colunas_essenciais if col in df_clean.columns]
    
    if colunas_presentes:
        mask_completos = df_clean[colunas_presentes].notna().all(axis=1)
        if not mask_completos.all():
            st.warning(f"⚠️ Removidas {(~mask_completos).sum()} linhas com dados essenciais missing")
            df_clean = df_clean[mask_completos]
    
    total_depois = len(df_clean)
    removidos = total_antes - total_depois
    
    if removidos > 0:
        st.success(f"✅ Limpeza concluída: {removidos} registros removidos | Restantes: {total_depois}")
    
    return df_clean

def load_all_games(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    df_completo = pd.concat(dfs, ignore_index=True)
    
    # Aplicar limpeza de duplicados
    df_completo = remove_duplicados_inteligente(df_completo)
    
    return df_completo

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

# --- Odds utilities ---
def effective_profit_from_odd(odd: float) -> float:
    """
    Converte uma odd 1X2 para lucro por unidade apostada.
    - Se odd > 1.0, assume odd decimal (ex: 1.90) -> retorna 0.90
    - Se odd <= 1.0, assume odd líquida (ex: 0.90) -> retorna 0.90
    """
    if pd.isna(odd):
        return np.nan
    return odd if odd <= 1.0 else (odd - 1.0)

def remove_juice_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera odds 'fair' (sem juice) a partir de Odd_H, Odd_D, Odd_A.
    Mantém proporções do mercado.
    """
    req = ['Odd_H','Odd_D','Odd_A']
    if not all(c in df.columns for c in req):
        return df

    df = df.copy()
    # prob. implícitas brutas
    pH = 1 / df['Odd_H']
    pD = 1 / df['Odd_D']
    pA = 1 / df['Odd_A']
    total = pH + pD + pA
    # normaliza para 1
    pH_f = pH / total
    pD_f = pD / total
    pA_f = pA / total
    # fair odds
    df['Odd_H_fair'] = 1 / pH_f
    df['Odd_D_fair'] = 1 / pD_f
    df['Odd_A_fair'] = 1 / pA_f
    return df

# --- Resultados 1X2 ---
def result_1x2_from_ft(gh, ga):
    if pd.isna(gh) or pd.isna(ga):
        return np.nan
    if gh > ga:
        return "H"
    elif gh < ga:
        return "A"
    return "D"

def result_1x2_from_today(row):
    gh = row.get('Goals_H_Today', np.nan)
    ga = row.get('Goals_A_Today', np.nan)
    if pd.isna(gh) or pd.isna(ga):
        return None
    return result_1x2_from_ft(gh, ga)

# --- Profit 1X2 ---
def calc_profit_1x2(result: str, side: str, odd: float) -> float:
    """
    Lucro por unidade no mercado 1X2.
    - odd pode ser decimal (>1) ou líquida (<=1) – tratamos automaticamente.
    """
    if pd.isna(result) or pd.isna(odd):
        return np.nan
    eff = effective_profit_from_odd(odd)
    return eff if side == result else -1.0

# ====================== 3D features (diferenças, trig, magnitude) ======================
def calcular_distancias_3d(df):
    df = df.copy()
    for c in ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']:
        if c not in df.columns:
            df[c] = 0.0
    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    # ângulos e projeções
    a_xy = np.arctan2(dy, dx)
    a_xz = np.arctan2(dz, dx)
    a_yz = np.arctan2(dz, np.where(dy==0, 1e-9, dy))
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

# ====================== LiveScore (merge) ======================
def setup_livescore_columns(df):
    if 'Goals_H_Today' not in df.columns: df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns: df['Goals_A_Today'] = np.nan
    if 'Home_Red'      not in df.columns: df['Home_Red']      = np.nan
    if 'Away_Red'      not in df.columns: df['Away_Red']      = np.nan
    return df

def load_and_merge_livescore(games_today, selected_date_str):
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
    right_key = 'Id' if 'Id' in raw.columns else None
    if left_key and right_key:
        games_today = games_today.merge(raw, left_on=left_key, right_on=right_key, how='left', suffixes=('', '_RAW'))

    # mapeia colunas comuns
    for src, dst in [('home_goal','Goals_H_Today'), ('away_goal','Goals_A_Today'),
                     ('home_red','Home_Red'), ('away_red','Away_Red')]:
        if src in games_today.columns:
            games_today[dst] = games_today[src]

    if 'status' in games_today.columns:
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today','Goals_A_Today']] = np.nan

    st.success(f"✅ LiveScore mesclado ({len(raw)} linhas)")
    return games_today

# ===================== EV ÓTIMO AUTOMÁTICO =====================
def calcular_ev_otimo_por_liga_lado(history, min_apostas=5, step=0.05, ev_range=(0.0, 1.0)):
    """
    Calcula EV ótimo por liga e por lado (H/D/A) com intervalos de confiança
    """
    resultados = []
    
    # Para cada combinação de liga e lado
    ligas = history['League'].unique() if 'League' in history.columns else ['Todas']
    lados = ['H', 'D', 'A']
    
    for liga in ligas:
        if liga == 'Todas':
            df_liga = history.copy()
        else:
            df_liga = history[history['League'] == liga].copy()
        
        for lado in lados:
            # Colunas relevantes para este lado - USAR TARGETS REAIS DO HISTÓRICO
            target_col = f'Target_EV_{"Home" if lado == "H" else "Draw" if lado == "D" else "Away"}'
            odd_col = f'Odd_{"H" if lado == "H" else "D" if lado == "D" else "A"}'
            
            if target_col not in df_liga.columns or odd_col not in df_liga.columns:
                continue
            
            # Para cada threshold, simular apostas baseadas no EV real
            for threshold in np.arange(ev_range[0], ev_range[1] + step, step):
                # CORREÇÃO: usar .values para evitar ambiguidade do pandas
                mask = (df_liga[target_col].values >= threshold)
                jogos_apostados = df_liga[mask]
                
                if len(jogos_apostados) < min_apostas:
                    continue
                
                # Calcular ROI real dessas apostas
                profits = jogos_apostados[target_col]
                roi_medio = profits.mean()
                n_apostas = len(profits)
                
                # Intervalo de confiança 95% (bootstrap) - apenas se houver dados suficientes
                if n_apostas >= 10:
                    try:
                        bootstraps = []
                        for _ in range(100):  # Reduzido para performance
                            sample = np.random.choice(profits.values, size=n_apostas, replace=True)
                            bootstraps.append(sample.mean())
                        ci_lower = np.percentile(bootstraps, 2.5)
                        ci_upper = np.percentile(bootstraps, 97.5)
                        confianca = (ci_upper - ci_lower) / 2
                    except:
                        ci_lower = ci_upper = confianca = roi_medio
                else:
                    ci_lower = ci_upper = confianca = roi_medio
                
                resultados.append({
                    'liga': liga,
                    'lado': lado,
                    'threshold': threshold,
                    'roi': roi_medio,
                    'apostas': n_apostas,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'confianca': confianca
                })
    
    return pd.DataFrame(resultados)

def encontrar_melhores_thresholds(df_analise, min_apostas=10):
    """
    Encontra os melhores thresholds por liga e lado
    """
    melhores = []
    
    if df_analise.empty:
        return pd.DataFrame(melhores)
    
    for liga in df_analise['liga'].unique():
        for lado in ['H', 'D', 'A']:
            df_filtrado = df_analise[
                (df_analise['liga'] == liga) & 
                (df_analise['lado'] == lado) & 
                (df_analise['apostas'] >= min_apostas)
            ]
            
            if len(df_filtrado) == 0:
                continue
            
            # Penalizar thresholds com poucas apostas e alta variância
            df_filtrado = df_filtrado.copy()
            # CORREÇÃO: usar .values para evitar ambiguidade
            roi_vals = df_filtrado['roi'].values
            confianca_vals = df_filtrado['confianca'].values
            
            scores = []
            for i in range(len(roi_vals)):
                if abs(roi_vals[i]) > 0:
                    score = roi_vals[i] * (1 - confianca_vals[i] / abs(roi_vals[i]))
                else:
                    score = roi_vals[i]
                scores.append(score)
            
            df_filtrado['score'] = scores
            melhor_idx = df_filtrado['score'].idxmax()
            melhor = df_filtrado.loc[melhor_idx]
            
            melhores.append({
                'liga': liga,
                'lado': lado,
                'threshold_otimo': melhor['threshold'],
                'roi_esperado': melhor['roi'],
                'apostas': melhor['apostas'],
                'confianca': melhor['confianca'],
                'ci_lower': melhor['ci_lower'],
                'ci_upper': melhor['ci_upper']
            })
    
    return pd.DataFrame(melhores)

# ============================ Carregamento ============================
st.info("📂 Carregando dados 1X2...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Selecione o arquivo do dia:", options, index=len(options)-1)

m = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = m.group(0) if m else datetime.now().strftime("%Y-%m-%d")

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

# Adicionar análise de qualidade antes do processamento
st.markdown("### 🔍 Análise de Qualidade dos Dados")

col_q1, col_q2, col_q3 = st.columns(3)

with col_q1:
    st.metric("Jogos Hoje (Raw)", len(games_today))

# Carregar histórico com limpeza
with st.spinner("Limpando e carregando histórico..."):
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    
    with col_q2:
        st.metric("Histórico (Limpo)", len(history))
    
    # Análise de completude dos dados
    colunas_importantes = ['Goals_H_FT', 'Goals_A_FT', 'Odd_H', 'Odd_D', 'Odd_A']
    colunas_presentes = [col for col in colunas_importantes if col in history.columns]
    
    completude = {}
    for col in colunas_presentes:
        completude[col] = history[col].notna().sum() / len(history) * 100
    
    with col_q3:
        completude_media = np.mean(list(completude.values())) if completude else 0
        st.metric("Completude Média", f"{completude_media:.1f}%")

# Mostrar detalhes da completude
if completude:
    st.markdown("#### 📊 Completude por Coluna")
    for col, perc in completude.items():
        st.progress(perc/100, text=f"{col}: {perc:.1f}%")

history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

# filtro temporal
if "Date" in history.columns:
    try:
        sel_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < sel_date].copy()
        st.info(f"📅 Histórico filtrado: {len(history)} jogos antes de {selected_date_str}")
    except Exception as e:
        st.warning(f"Erro ao aplicar filtro temporal: {e}")

# Verifica se odds 1X2 existem
for df, name in [(history, "history"), (games_today, "games_today")]:
    missing = [c for c in ['Odd_H','Odd_D','Odd_A'] if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas de odds 1X2 ausentes em {name}: {missing}")

# LiveScore para validação em tempo real
games_today = load_and_merge_livescore(games_today, selected_date_str)

# ===================== Targets EV (Home / Draw / Away) – histórico =====================
st.markdown("### 🎯 Construindo targets de lucro 1X2 (histórico)")

def build_1x2_targets(df_hist: pd.DataFrame):
    df = df_hist.copy()

    # Verificar duplicados finais
    chaves_finais = ['Home', 'Away', 'Date'] if all(col in df.columns for col in ['Home', 'Away', 'Date']) else ['Home', 'Away']
    if all(col in df.columns for col in chaves_finais):
        duplicados_finais = df.duplicated(subset=chaves_finais, keep='first')
        if duplicados_finais.any():
            st.error(f"🚨 ATENÇÃO: {duplicados_finais.sum()} duplicados ainda presentes após limpeza!")
            st.dataframe(df[duplicados_finais][chaves_finais + ['League']].head(10))
            # Remover os duplicados finais
            df = df[~duplicados_finais]

    # resultado real 1X2 (usando FT)
    df['Result_1X2'] = [result_1x2_from_ft(h, a) for h, a in zip(df['Goals_H_FT'], df['Goals_A_FT'])]

    # odds (podem ser decimais ou líquidas; função trata ambos)
    df['Target_EV_Home'] = df.apply(lambda r: calc_profit_1x2(r['Result_1X2'], 'H', r.get('Odd_H', np.nan)), axis=1)
    df['Target_EV_Draw'] = df.apply(lambda r: calc_profit_1x2(r['Result_1X2'], 'D', r.get('Odd_D', np.nan)), axis=1)
    df['Target_EV_Away'] = df.apply(lambda r: calc_profit_1x2(r['Result_1X2'], 'A', r.get('Odd_A', np.nan)), axis=1)

    return df

history = build_1x2_targets(history)

# ===================== Opções de normalização na UI =====================
st.markdown("### ⚙️ Normalizações de odds / targets (opcional)")
col_opt1, col_opt2 = st.columns(2)
use_fair = col_opt1.checkbox("Usar odds fair (remover juice 1X2)", value=False, help="Reestima odds sem vig pelo método de prob. normalizadas.")
center_targets = col_opt2.checkbox("Recentrar targets (EV relativo ao mercado)", value=True, help="Subtrai a média do EV histórico por lado.")

if use_fair:
    history = remove_juice_1x2(history)
    games_today = remove_juice_1x2(games_today)
    # troca as odds para as 'fair' quando existirem
    for side, fair_col in [('H','Odd_H_fair'), ('D','Odd_D_fair'), ('A','Odd_A_fair')]:
        col = {'H':'Odd_H','D':'Odd_D','A':'Odd_A'}[side]
        if fair_col in history.columns:
            history[col] = history[fair_col]
        if fair_col in games_today.columns:
            games_today[col] = games_today[fair_col]

if center_targets:
    for tgt in ['Target_EV_Home','Target_EV_Draw','Target_EV_Away']:
        if tgt in history.columns:
            mean_tgt = pd.to_numeric(history[tgt], errors='coerce').mean()
            history[tgt] = history[tgt] - mean_tgt

st.info(
    f"Histórico: {len(history)} jogos | "
    f"EV_Home médio={pd.to_numeric(history.get('Target_EV_Home'), errors='coerce').mean():.3f} | "
    f"EV_Draw médio={pd.to_numeric(history.get('Target_EV_Draw'), errors='coerce').mean():.3f} | "
    f"EV_Away médio={pd.to_numeric(history.get('Target_EV_Away'), errors='coerce').mean():.3f}"
)

# ===================== Feature engineering 3D =====================
def ensure_3d_features(df):
    df = calcular_distancias_3d(df)
    df = aplicar_clusterizacao_3d(df, n_clusters=5)
    return df

history = ensure_3d_features(history)
games_today = ensure_3d_features(games_today)

# ===================== Treinamento – três regressões (H/D/A) =====================
st.markdown("### 🤖 Treinando modelos de EV 1X2 (Home / Draw / Away)")

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
    yH = history['Target_EV_Home']; maskH = yH.notna()
    yD = history['Target_EV_Draw']; maskD = yD.notna()
    yA = history['Target_EV_Away']; maskA = yA.notna()

    model_H = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_D = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_A = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)

    if maskH.any(): model_H.fit(X.loc[maskH], yH.loc[maskH])
    if maskD.any(): model_D.fit(X.loc[maskD], yD.loc[maskD])
    if maskA.any(): model_A.fit(X.loc[maskA], yA.loc[maskA])

    # preparar X_today
    ligas_today = pd.get_dummies(games_today['League'], prefix='League') if 'League' in games_today.columns else pd.DataFrame()
    ligas_today = ligas_today.reindex(columns=ligas_d.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_d.columns, fill_value=0)
    extras_today = games_today[feat3d].fillna(0)
    X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    # previsões
    games_today['Predicted_EV_H'] = model_H.predict(X_today) if maskH.any() else 0.0
    games_today['Predicted_EV_D'] = model_D.predict(X_today) if maskD.any() else 0.0
    games_today['Predicted_EV_A'] = model_A.predict(X_today) if maskA.any() else 0.0

    # melhor lado por EV
    preds = games_today[['Predicted_EV_H','Predicted_EV_D','Predicted_EV_A']]
    idxmax = preds.values.argmax(axis=1)
    map_side = {0:'H', 1:'D', 2:'A'}
    games_today['Chosen_Side'] = [map_side[i] for i in idxmax]
    games_today['Predicted_EV'] = preds.max(axis=1)

    # Importâncias (Home) – best-effort
    try:
        importances = pd.Series(model_H.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        st.markdown("#### 🔍 Top Features (Modelo EV – HOME)")
        st.dataframe(importances.to_frame("Importância"), use_container_width=True)
    except Exception:
        pass

    st.success("✅ Modelos EV 1X2 treinados (H/D/A)")
    return model_H, model_D, model_A, games_today

model_H, model_D, model_A, games_today = train_triple_ev_models(history, games_today)

# # ===================== SLIDER INTELIGENTE =====================
# st.markdown("### ⚖️ EV Mínimo - Sistema Inteligente")

# col_auto, col_manual = st.columns(2)

# with col_auto:
#     usar_ev_auto = st.checkbox("🎯 Usar EV Ótimo Automático", value=True, 
#                               help="Calcula o melhor EV threshold baseado no histórico")

# with col_manual:
#     ev_manual = st.slider("EV mínimo manual", -0.50, 1.0, 0.00, 0.01, 
#                          help="Use se preferir controle manual")

# # Calcular EV ótimo se selecionado
# ev_threshold = ev_manual
# df_melhores = pd.DataFrame()

# if usar_ev_auto:
#     with st.spinner("🔍 Calculando EV ótimo por liga e lado..."):
#         try:
#             df_analise = calcular_ev_otimo_por_liga_lado(history)
            
#             if not df_analise.empty:
#                 df_melhores = encontrar_melhores_thresholds(df_analise)
                
#                 # Encontrar threshold global (média ponderada por apostas)
#                 if not df_melhores.empty:
#                     total_apostas = df_melhores['apostas'].sum()
#                     ev_global = (df_melhores['threshold_otimo'] * df_melhores['apostas']).sum() / total_apostas
#                     ev_threshold = ev_global
                    
#                     st.success(f"**EV Ótimo Global:** {ev_global:.3f} | **Baseado em {len(df_melhores)} combinações liga/lado**")
                    
#                     # Mostrar tabela de thresholds por liga/lado
#                     st.markdown("#### 📊 Thresholds por Liga e Lado")
#                     display_cols = ['liga', 'lado', 'threshold_otimo', 'roi_esperado', 'apostas', 'confianca']
#                     st.dataframe(
#                         df_melhores[display_cols]
#                         .round({'threshold_otimo': 3, 'roi_esperado': 3, 'confianca': 3})
#                         .sort_values('roi_esperado', ascending=False)
#                         .style.background_gradient(subset=['roi_esperado'], cmap='RdYlGn')
#                         .format({
#                             'threshold_otimo': '{:.3f}',
#                             'roi_esperado': '{:.1%}',
#                             'confianca': '{:.3f}'
#                         })
#                     )
                    
#                     # Gráfico de análise (apenas se houver dados suficientes)
#                     ligas_com_dados = df_analise[df_analise['apostas'] >= 10]['liga'].unique()
#                     if len(ligas_com_dados) > 0:
#                         st.markdown("#### 📈 Análise EV vs ROI")
#                         fig = px.line(df_analise[df_analise['apostas'] >= 10], 
#                                      x='threshold', y='roi', color='lado',
#                                      facet_col='liga', facet_col_wrap=2,
#                                      title='ROI vs EV Threshold por Liga e Lado')
#                         st.plotly_chart(fig, use_container_width=True)
                    
#                 else:
#                     st.warning("⚠️ Não foram encontrados thresholds ótimos com dados suficientes.")
#                     ev_threshold = ev_manual
#             else:
#                 st.warning("⚠️ Dados insuficientes para cálculo automático. Verifique se há histórico suficiente.")
#                 ev_threshold = ev_manual
                
#         except Exception as e:
#             st.error(f"❌ Erro no cálculo automático: {e}")
#             st.info("📝 Usando EV manual como fallback")
#             ev_threshold = ev_manual



# ===================== SLIDER INTELIGENTE =====================
st.markdown("### ⚖️ EV Mínimo - Sistema Inteligente")

col_auto, col_manual = st.columns(2)

with col_auto:
    usar_ev_auto = st.checkbox("🎯 Usar EV Ótimo Automático", value=True, 
                              help="Calcula o melhor EV threshold baseado no histórico")

with col_manual:
    ev_manual = st.slider("EV mínimo manual", -0.50, 1.0, 0.00, 0.01, 
                         help="Use se preferir controle manual")

# Calcular EV ótimo se selecionado
ev_threshold = ev_manual
df_melhores = pd.DataFrame()

if usar_ev_auto:
    with st.spinner("🔍 Calculando EV ótimo por liga e lado..."):
        try:
            df_analise = calcular_ev_otimo_por_liga_lado(history)
            
            if not df_analise.empty:
                df_melhores = encontrar_melhores_thresholds(df_analise)
                
                # Encontrar threshold global (média ponderada por apostas)
                if not df_melhores.empty:
                    total_apostas = df_melhores['apostas'].sum()
                    ev_global = (df_melhores['threshold_otimo'] * df_melhores['apostas']).sum() / total_apostas
                    ev_threshold = ev_global
                    
                    st.success(f"**EV Ótimo Global:** {ev_global:.3f} | **Baseado em {len(df_melhores)} combinações liga/lado**")
                    
                    # Mostrar tabela de thresholds por liga/lado
                    st.markdown("#### 📊 Thresholds por Liga e Lado")
                    display_cols = ['liga', 'lado', 'threshold_otimo', 'roi_esperado', 'apostas', 'confianca']
                    st.dataframe(
                        df_melhores[display_cols]
                        .round({'threshold_otimo': 3, 'roi_esperado': 3, 'confianca': 3})
                        .sort_values('roi_esperado', ascending=False)
                        .style.background_gradient(subset=['roi_esperado'], cmap='RdYlGn')
                        .format({
                            'threshold_otimo': '{:.3f}',
                            'roi_esperado': '{:.1%}',
                            'confianca': '{:.3f}'
                        })
                    )
                    
                    # Gráfico de análise (apenas se houver dados suficientes)
                    ligas_com_dados = df_analise[df_analise['apostas'] >= 10]['liga'].unique()
                    if len(ligas_com_dados) > 0:
                        st.markdown("#### 📈 Análise EV vs ROI")
                        
                        # Limitar o número de ligas no gráfico para evitar erro
                        ligas_para_grafico = ligas_com_dados[:6]  # Máximo 6 ligas
                        
                        if len(ligas_para_grafico) > 0:
                            df_grafico = df_analise[
                                (df_analise['apostas'] >= 10) & 
                                (df_analise['liga'].isin(ligas_para_grafico))
                            ]
                            
                            if not df_grafico.empty:
                                try:
                                    fig = px.line(df_grafico, 
                                                 x='threshold', y='roi', color='lado',
                                                 facet_col='liga', facet_col_wrap=3,
                                                 title='ROI vs EV Threshold (Top Ligas)',
                                                 facet_col_spacing=0.08)
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning("⚠️ Gráfico simplificado devido a muitas ligas")
                                    # Fallback: gráfico agregado
                                    fig = px.line(df_analise[df_analise['apostas'] >= 10], 
                                                 x='threshold', y='roi', color='lado',
                                                 title='ROI vs EV Threshold (Todas as Ligas Agregadas)')
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar estatísticas das outras ligas
                        if len(ligas_com_dados) > len(ligas_para_grafico):
                            st.info(f"📊 +{len(ligas_com_dados) - len(ligas_para_grafico)} ligas analisadas mas não mostradas no gráfico")
                    
                    # Mostrar as top 5 ligas com melhor ROI
                    st.markdown("#### 🏆 Top 5 Ligas por ROI Máximo")
                    top_ligas = df_melhores.groupby('liga')['roi_esperado'].max().nlargest(5).round(4)
                    st.dataframe(top_ligas.rename('ROI Máximo').apply(lambda x: f"{x:.1%}"))
                    
                else:
                    st.warning("⚠️ Não foram encontrados thresholds ótimos com dados suficientes.")
                    ev_threshold = ev_manual
            else:
                st.warning("⚠️ Dados insuficientes para cálculo automático. Verifique se há histórico suficiente.")
                ev_threshold = ev_manual
                
        except Exception as e:
            st.error(f"❌ Erro no cálculo automático: {e}")
            st.info("📝 Usando EV manual como fallback")
            ev_threshold = ev_manual




# ===================== Filtros e Ranking =====================
st.markdown("## 🏆 Ranking por Expected Value (1X2) - COM PROBABILIDADES")

# ===================== SUBSECÇÃO: Filtros =====================
st.markdown("### 🔧 Filtros")

col_f1, col_f2 = st.columns(2)
leagues = sorted(games_today['League'].dropna().unique()) if 'League' in games_today.columns else []
sel_league = col_f1.selectbox("Filtrar por liga", options=["Todas"] + leagues, index=0)

df_rank = games_today.copy()

# Aplicar filtro de liga
if sel_league != "Todas" and 'League' in df_rank.columns:
    df_rank = df_rank[df_rank['League'] == sel_league]

# Aplicar filtro EV - versão inteligente
if usar_ev_auto and not df_melhores.empty and sel_league != "Todas":
    # Aplicar threshold específico para a liga selecionada
    thresholds_liga = df_melhores[df_melhores['liga'] == sel_league]
    if not thresholds_liga.empty:
        # Criar máscara com thresholds específicos por lado para esta liga
        masks = []
        for _, row in thresholds_liga.iterrows():
            lado_mask = (df_rank['Chosen_Side'] == row['lado'])
            ev_mask = (df_rank['Predicted_EV'] >= row['threshold_otimo'])
            masks.append(lado_mask & ev_mask)
        
        # Combinar todas as máscaras
        if masks:
            mask_final = masks[0]
            for mask in masks[1:]:
                mask_final = mask_final | mask
            df_rank = df_rank[mask_final]
else:
    # Filtro global tradicional
    df_rank = df_rank[df_rank['Predicted_EV'] >= ev_threshold]

# ===================== SUBSECÇÃO: Controle de Exibição =====================
st.markdown("### 📊 Controle de Exibição")

# CORREÇÃO: Verificar se há dados antes do slider
if len(df_rank) == 0:
    st.warning(f"⚠️ Nenhum jogo encontrado com os filtros atuais")
    rank = pd.DataFrame()
    top_n = 0
else:
    # Slider seguro - garantir que max seja maior que min
    max_games = min(200, len(df_rank))
    default_games = min(40, max_games)
    top_n = st.slider("Quantos confrontos exibir:", 1, max_games, default_games, step=5)
    rank = df_rank.sort_values('Predicted_EV', ascending=False).head(top_n)

# ===================== SUBSECÇÃO: Tabela de Resultados =====================
st.markdown("### 📋 Resultados Filtrados")

if not rank.empty:
    # Calcular probabilidades implícitas das odds
    def implied_probability(odd):
        if pd.isna(odd) or odd <= 0:
            return np.nan
        return 1 / odd

    rank_with_probs = rank.copy()
    
    # Probabilidades implícitas
    rank_with_probs['Prob_H'] = rank_with_probs['Odd_H'].apply(implied_probability)
    rank_with_probs['Prob_D'] = rank_with_probs['Odd_D'].apply(implied_probability)  
    rank_with_probs['Prob_A'] = rank_with_probs['Odd_A'].apply(implied_probability)

    # Probabilidades previstas
    def ev_to_prob(ev, odd):
        if pd.isna(ev) or pd.isna(odd) or odd <= 1.0:
            return np.nan
        prob = (ev + 1) / odd
        return max(0.0, min(1.0, prob))

    rank_with_probs['Prob_Pred_H'] = [ev_to_prob(ev, odd) for ev, odd in zip(rank_with_probs['Predicted_EV_H'], rank_with_probs['Odd_H'])]
    rank_with_probs['Prob_Pred_D'] = [ev_to_prob(ev, odd) for ev, odd in zip(rank_with_probs['Predicted_EV_D'], rank_with_probs['Odd_D'])]
    rank_with_probs['Prob_Pred_A'] = [ev_to_prob(ev, odd) for ev, odd in zip(rank_with_probs['Predicted_EV_A'], rank_with_probs['Odd_A'])]

    # Calcular Edge
    rank_with_probs['Edge_H'] = rank_with_probs['Prob_Pred_H'] - rank_with_probs['Prob_H']
    rank_with_probs['Edge_D'] = rank_with_probs['Prob_Pred_D'] - rank_with_probs['Prob_D']
    rank_with_probs['Edge_A'] = rank_with_probs['Prob_Pred_A'] - rank_with_probs['Prob_A']

    def get_chosen_edge(row):
        side = row['Chosen_Side']
        if side == 'H': return row['Edge_H']
        elif side == 'D': return row['Edge_D']
        else: return row['Edge_A']

    rank_with_probs['Edge_Chosen'] = rank_with_probs.apply(get_chosen_edge, axis=1)

    # Exibir tabela
    display_columns = [
        'League', 'Home', 'Away', 'Chosen_Side', 'Predicted_EV',
        'Odd_H', 'Odd_D', 'Odd_A',
        'Prob_H', 'Prob_D', 'Prob_A',
        'Prob_Pred_H', 'Prob_Pred_D', 'Prob_Pred_A', 
        'Edge_H', 'Edge_D', 'Edge_A', 'Edge_Chosen'
    ]

    available_columns = [col for col in display_columns if col in rank_with_probs.columns]

    st.dataframe(
        rank_with_probs[available_columns]
        .style
        .background_gradient(subset=['Predicted_EV', 'Edge_Chosen', 'Edge_H', 'Edge_D', 'Edge_A'], cmap='RdYlGn', vmin=-0.1, vmax=0.1)
        .background_gradient(subset=['Prob_Pred_H', 'Prob_Pred_D', 'Prob_Pred_A'], cmap='Blues', vmin=0, vmax=1)
        .format({
            'Predicted_EV': '{:.3f}',
            'Odd_H': '{:.2f}', 'Odd_D': '{:.2f}', 'Odd_A': '{:.2f}',
            'Prob_H': '{:.1%}', 'Prob_D': '{:.1%}', 'Prob_A': '{:.1%}',
            'Prob_Pred_H': '{:.1%}', 'Prob_Pred_D': '{:.1%}', 'Prob_Pred_A': '{:.1%}',
            'Edge_H': '{:.2%}', 'Edge_D': '{:.2%}', 'Edge_A': '{:.2%}', 'Edge_Chosen': '{:.2%}'
        }),
        width='stretch'  # CORREÇÃO: use_container_width -> width='stretch'
    )

    # ===================== SUBSECÇÃO: Estatísticas =====================
    st.markdown("### 📈 Estatísticas das Probabilidades")
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_edge = rank_with_probs['Edge_Chosen'].mean() * 100
        st.metric("Edge Médio do Lado Escolhido", f"{avg_edge:.2f}%")

    with col2:
        positive_edges = (rank_with_probs['Edge_Chosen'] > 0).sum()
        total_games = len(rank_with_probs)
        st.metric("Jogos com Edge Positivo", f"{positive_edges}/{total_games}")

    with col3:
        max_edge = rank_with_probs['Edge_Chosen'].max() * 100
        st.metric("Maior Edge Encontrado", f"{max_edge:.2f}%")

else:
    st.info("ℹ️ Ajuste os filtros para ver jogos recomendados")

# ===================== SUBSECÇÃO: Legenda =====================
st.markdown("### 📖 Legenda das Colunas")
st.markdown("""
- **`Prob_H/D/A`**: Probabilidade implícita das odds atuais
- **`Prob_Pred_H/D/A`**: Probabilidade prevista pelo modelo  
- **`Edge_H/D/A`**: Vantagem = Prob_Prevista - Prob_Implícita
- **`Edge_Chosen`**: Edge do lado escolhido para aposta
""")

# ===================== Simulação – carteira prevista =====================
st.markdown("### 💼 Simulação de carteira prevista (aposta quando EV ≥ threshold)")
sim = df_rank.copy()
sim['Place_Bet'] = sim['Predicted_EV'] >= ev_threshold
sim['Profit_Predicted'] = np.where(sim['Place_Bet'], sim['Predicted_EV'], 0.0)

tot_bets = int(sim['Place_Bet'].sum())
tot_pred_profit = sim['Profit_Predicted'].sum()
roi_pred_mean = (tot_pred_profit / tot_bets) if tot_bets > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Apostas previstas", tot_bets)
c2.metric("Profit Previsto (u)", f"{tot_pred_profit:.2f}")
c3.metric("ROI Médio Previsto", f"{roi_pred_mean*100:.2f}%")

# ===================== Live Validation – ROI real do dia =====================
st.markdown("## 📡 Live Validation – ROI real 1X2 (gols do dia)")

def real_profit_1x2_row(row):
    res = result_1x2_from_today(row)
    if res is None:
        return np.nan
    side = row['Chosen_Side']
    odd_map = {'H':'Odd_H','D':'Odd_D','A':'Odd_A'}
    odd_col = odd_map[side]
    odd = row.get(odd_col, np.nan)
    return calc_profit_1x2(res, side, odd)

sim['Real_Profit'] = sim.apply(real_profit_1x2_row, axis=1)
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
    comp_cols = ['League','Home','Away','Odd_H','Odd_D','Odd_A','Chosen_Side','Predicted_EV','Goals_H_Today','Goals_A_Today','Real_Profit']
    show_cols = [c for c in comp_cols if c in finished_bets.columns]
    st.dataframe(
        finished_bets.sort_values('Predicted_EV', ascending=False)[show_cols]
        .style.background_gradient(subset=['Real_Profit','Predicted_EV'], cmap='RdYlGn')
        .format({'Predicted_EV':'{:.2f}','Real_Profit':'{:.2f}','Goals_H_Today':'{:.0f}','Goals_A_Today':'{:.0f}','Odd_H':'{:.2f}','Odd_D':'{:.2f}','Odd_A':'{:.2f}'})
    )

    st.markdown("#### 📊 Quebra por lado (H/D/A)")
    st.dataframe(by_side.rename(columns={'count':'Apostas','sum':'Profit'}))

st.markdown("---")
st.success("💹 Pipeline ROI 1X2 – Triple Side + Live Validation + EV Ótimo Automático pronto!")
st.info("""
**O que esta versão faz:**
- 🎯 Targets contínuos de lucro esperado para Home / Draw / Away (1X2)
- 🤖 Três regressões de EV (H/D/A) e escolha automática do melhor lado
- ⚙️ Opções: odds 'fair' sem juice + recenter de targets para EV relativo
- 🎯 EV ÓTIMO AUTOMÁTICO por liga e lado com intervalos de confiança
- 📈 Ranking inteligente com thresholds específicos por contexto
- 📡 Validação em tempo real com gols do dia (Profit Real & ROI por lado)
- 🧩 Mantém features 3D (Aggression × M × MT) + clusters KMeans
""")
