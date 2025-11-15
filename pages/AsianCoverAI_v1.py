# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

# ========================= CONFIG STREAMLIT =========================
st.set_page_config(page_title="AsianCoverAI v1 - Probabilidade de Cobrir Handicap", layout="wide")
st.title("🎯 AsianCoverAI v1 – Probabilidade de Cobrir o Handicap (Home & Away)")

# ========================= CONFIG GERAIS =========================
PAGE_PREFIX = "AsianCoverAI_v1"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

# ============================================================
# 🔧 FUNÇÕES AUXILIARES BÁSICAS
# ============================================================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Goals_H_Today', 'Goals_A_Today', 'Home_Red', 'Away_Red', 'status_ls']:
        if col not in df.columns:
            df[col] = np.nan
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        return pd.DataFrame()
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(preprocess_df(pd.read_csv(os.path.join(folder, f))))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def convert_asian_line_to_decimal(value):
    """
    Converte Asian Line textual em decimal referência HOME.
    Mantemos convenção: linha negativa favorece o HOME.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan

    # Ex.: "0", "-0.5", "1.25", "0/0.5", "-0.5/1"
    # Atenção: sua base pode vir com "H" / "A" etc. Aqui assumimos que
    # a linha já está em formato numérico ou split tipo "-0.5/1.0".
    if "/" not in s:
        try:
            num = float(s)
            # Convenção: para HOME, negativo favorece casa
            return -num
        except:
            return np.nan

    try:
        raw = s
        sign = -1 if raw.strip().startswith("-") else 1
        clean = raw.replace("+", "").replace("-", "")
        parts = [float(p) for p in clean.split("/")]
        avg = np.mean(parts)
        result = sign * avg
        return -result
    except:
        return np.nan

# ============================================================
# ⚙️ Z-SCORES (M / MT) A PARTIR DO HANDSCORE
# ============================================================
def calcular_zscores_detalhados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Z-scores a partir do HandScore:
    - M_H, M_A: Z-score do time em relação à liga (performance relativa)
    - MT_H, MT_A: Z-score do time em relação a si mesmo (consistência)
    """
    df = df.copy()

    # 1) Z-score por liga
    if 'League' in df.columns and 'HandScore_Home' in df.columns and 'HandScore_Away' in df.columns:
        league_stats = df.groupby('League').agg({
            'HandScore_Home': ['mean', 'std'],
            'HandScore_Away': ['mean', 'std']
        }).round(3)
        league_stats.columns = ['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std']

        league_stats['HS_H_std'] = league_stats['HS_H_std'].replace(0, 1)
        league_stats['HS_A_std'] = league_stats['HS_A_std'].replace(0, 1)

        df = df.merge(league_stats, on='League', how='left')

        df['M_H'] = (df['HandScore_Home'] - df['HS_H_mean']) / df['HS_H_std']
        df['M_A'] = (df['HandScore_Away'] - df['HS_A_mean']) / df['HS_A_std']

        df['M_H'] = np.clip(df['M_H'], -5, 5)
        df['M_A'] = np.clip(df['M_A'], -5, 5)
    else:
        df['M_H'] = 0.0
        df['M_A'] = 0.0

    # 2) Z-score por time
    if 'Home' in df.columns and 'Away' in df.columns:
        home_team_stats = df.groupby('Home').agg({
            'HandScore_Home': ['mean', 'std']
        }).round(3)
        home_team_stats.columns = ['HT_mean', 'HT_std']

        away_team_stats = df.groupby('Away').agg({
            'HandScore_Away': ['mean', 'std']
        }).round(3)
        away_team_stats.columns = ['AT_mean', 'AT_std']

        home_team_stats['HT_std'] = home_team_stats['HT_std'].replace(0, 1)
        away_team_stats['AT_std'] = away_team_stats['AT_std'].replace(0, 1)

        df = df.merge(home_team_stats, left_on='Home', right_index=True, how='left')
        df = df.merge(away_team_stats, left_on='Away', right_index=True, how='left')

        df['MT_H'] = (df['HandScore_Home'] - df['HT_mean']) / df['HT_std']
        df['MT_A'] = (df['HandScore_Away'] - df['AT_mean']) / df['AT_std']

        df['MT_H'] = np.clip(df['MT_H'], -5, 5)
        df['MT_A'] = np.clip(df['MT_A'], -5, 5)

        df = df.drop(['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std',
                      'HT_mean', 'HT_std', 'AT_mean', 'AT_std'], axis=1, errors='ignore')
    else:
        df['MT_H'] = 0.0
        df['MT_A'] = 0.0

    return df

# ============================================================
# 🧼 LIMPEZA DE FEATURES (SEM PERDER LINHA)
# ============================================================
def clean_features_for_training(X: pd.DataFrame) -> pd.DataFrame:
    X_clean = X.copy()

    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean, columns=X.columns if hasattr(X, 'columns') else range(X.shape[1]))

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)

    for col in X_clean.columns:
        if pd.api.types.is_float_dtype(X_clean[col]):
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower, upper)

    X_clean = X_clean.replace([np.inf, -np.inf], 0).fillna(0)
    return X_clean

# ============================================================
# ⚽ WEIGHTED GOALS (WG) – PONDERADOS PELA PROBABILIDADE REAL
# ============================================================
def odds_to_probs(odd_h, odd_d, odd_a):
    try:
        odd_h = float(odd_h); odd_d = float(odd_d); odd_a = float(odd_a)
        if odd_h <= 0 or odd_d <= 0 or odd_a <= 0:
            return 1/3, 1/3, 1/3
        inv_sum = (1/odd_h) + (1/odd_d) + (1/odd_a)
        p_h = (1/odd_h) / inv_sum
        p_d = (1/odd_d) / inv_sum
        p_a = (1/odd_a) / inv_sum
        return p_h, p_d, p_a
    except:
        return 1/3, 1/3, 1/3

def weighted_goals_home(row):
    gf = row.get('Goals_H_FT', 0)
    ga = row.get('Goals_A_FT', 0)
    p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 0), row.get('Odd_D', 0), row.get('Odd_A', 0))
    gf_weighted = gf * (1 - p_h)
    ga_weighted = ga * p_h
    return gf_weighted - ga_weighted

def weighted_goals_away(row):
    gf = row.get('Goals_A_FT', 0)
    ga = row.get('Goals_H_FT', 0)
    p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 0), row.get('Odd_D', 0), row.get('Odd_A', 0))
    gf_weighted = gf * (1 - p_a)
    ga_weighted = ga * p_a
    return gf_weighted - ga_weighted

# ============================================================
# 🧮 ESPAÇO 3D (AGGRESSION / MOMENTUM)
# ============================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        for c in ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT']:
            df[c] = 0.0
        return df

    dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
    dy = (df['M_H'] - df['M_A']) / 2
    dz = (df['MT_H'] - df['MT_A']) / 2
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = df['Quadrant_Dist_3D']
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    return df

from sklearn.cluster import KMeans

def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters=4, random_state=42) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        df['Cluster3D_Label'] = 0
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']
    X = df[['dx','dy','dz']].fillna(0).to_numpy()
    n_samples = X.shape[0]
    k = max(1, min(n_clusters, n_samples))
    try:
        km = KMeans(n_clusters=k, random_state=random_state, init='k-means++', n_init=10)
        df['Cluster3D_Label'] = km.fit_predict(X)
    except Exception:
        df['Cluster3D_Label'] = 0
    return df

# ============================================================
# ⚖️ LIQUIDAÇÃO DE HANDICAP (Para targets Cover)
# ============================================================
def _split_quarter(line):
    # ex: -0.75 -> [-0.5, -1.0]; +0.25 -> [0.0, +0.5]
    if (abs(line) * 100) % 50 == 0:
        return [line]
    base = math.floor(abs(line)*2)/2.0
    if abs(abs(line) - base) < 1e-9:
        return [line]
    sign = 1 if line >= 0 else -1
    lower = sign * base
    upper = sign * (base + 0.5)
    return [lower, upper]

def settle_ah_bet(margin, line, side):
    """
    margin = Goals_H_FT - Goals_A_FT
    line   = Asian_Line_Decimal (referência HOME)
    side   = 'HOME' ou 'AWAY'
    Retorna ganho unitário médio considerando .25/.75
    """
    bet_line = line if side == 'HOME' else -line
    parts = _split_quarter(bet_line)
    scores = []
    for p in parts:
        adj = margin + p
        if adj > 0:
            scores.append(1.0)
        elif adj == 0:
            scores.append(0.0)
        else:
            scores.append(-1.0)
    return sum(scores)/len(scores)

def criar_targets_cover(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Cover_Home'] = np.nan
    df['Cover_Away'] = np.nan

    mask = df['Goals_H_FT'].notna() & df['Goals_A_FT'].notna() & df['Asian_Line_Decimal'].notna()
    sub = df.loc[mask].copy()
    if sub.empty:
        return df

    margin = sub['Goals_H_FT'] - sub['Goals_A_FT']
    line = sub['Asian_Line_Decimal']

    cover_home = []
    cover_away = []

    for m, l in zip(margin, line):
        u_home = settle_ah_bet(m, l, 'HOME')
        u_away = settle_ah_bet(m, l, 'AWAY')
        cover_home.append(1 if u_home > 0 else 0)
        cover_away.append(1 if u_away > 0 else 0)

    df.loc[mask, 'Cover_Home'] = cover_home
    df.loc[mask, 'Cover_Away'] = cover_away
    return df

# ============================================================
# 📡 LIVE SCORE INTEGRATION (apenas FT, inteiros)
# ============================================================
def load_and_merge_livescore(games_today, selected_date_str):
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

    if not os.path.exists(livescore_file):
        return games_today

    results_df = pd.read_csv(livescore_file)
    results_df['status'] = results_df['status'].astype(str).str.upper().str.strip()
    df_ft = results_df[results_df['status'] == 'FT'].copy()

    for c in ['home_goal','away_goal','home_red','away_red']:
        df_ft[c] = pd.to_numeric(df_ft[c], errors='coerce').fillna(0).astype(int)

    if 'Id' in games_today.columns and 'Id' in df_ft.columns:
        games_today = games_today.merge(
            df_ft[['Id','status','home_goal','away_goal','home_red','away_red']],
            on='Id',
            how='left',
            suffixes=('', '_ls')
        )

        mask_ft = games_today['status_ls'] == 'FT'
        games_today.loc[mask_ft, 'Goals_H_Today'] = games_today.loc[mask_ft, 'home_goal']
        games_today.loc[mask_ft, 'Goals_A_Today'] = games_today.loc[mask_ft, 'away_goal']
        games_today.loc[mask_ft, 'Home_Red']      = games_today.loc[mask_ft, 'home_red']
        games_today.loc[mask_ft, 'Away_Red']      = games_today.loc[mask_ft, 'away_red']

    return games_today

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ============================================================
def main():
    st.info("📂 Carregando dados para AsianCoverAI...")

    # Seleção do arquivo do dia
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
    if not files:
        st.warning("Nenhum CSV encontrado na pasta GamesDay.")
        return

    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.selectbox("Selecione o arquivo do dia:", options, index=len(options)-1)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)

        # Fillna geral para NÃO perder linha
        games_today = games_today.fillna(0)
        history = history.fillna(0)

        # excluir copas
        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        # classificar tier
        def classificar_league_tier(league_name: str) -> int:
            if pd.isna(league_name):
                return 3
            name = str(league_name).lower()
            if any(x in name for x in ['premier','la liga','serie a','bundesliga','ligue 1','eredivisie','primeira liga','brasileirão','super league','mls','championship','liga pro','a-league']):
                return 1
            if any(x in name for x in ['serie b','segunda','league 1','liga ii','liga 2','division 2','bundesliga 2','ligue 2','j-league','k-league','superettan','1st division','national league','liga nacional']):
                return 2
            return 3

        def aplicar_filtro_tier(df: pd.DataFrame, max_tier=3) -> pd.DataFrame:
            if 'League' not in df.columns:
                df['League_Tier'] = 3
                return df
            df = df.copy()
            df['League_Tier'] = df['League'].apply(classificar_league_tier)
            filtrado = df[df['League_Tier'] <= max_tier].copy()
            return filtrado

        history = aplicar_filtro_tier(history, max_tier=3)
        games_today = aplicar_filtro_tier(games_today, max_tier=3)

        # OneHot de 10 ligas mais comuns
        top_ligas = history['League'].value_counts().head(10).index
        history['League_Clean'] = history['League'].where(history['League'].isin(top_ligas), 'Other')
        games_today['League_Clean'] = games_today['League'].where(games_today['League'].isin(top_ligas), 'Other')

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc_hist = encoder.fit_transform(history[['League_Clean']])
        enc_hist_df = pd.DataFrame(enc_hist, columns=encoder.get_feature_names_out(['League_Clean']))
        history = pd.concat([history.reset_index(drop=True), enc_hist_df.reset_index(drop=True)], axis=1)

        enc_today = encoder.transform(games_today[['League_Clean']])
        enc_today_df = pd.DataFrame(enc_today, columns=encoder.get_feature_names_out(['League_Clean']))
        games_today = pd.concat([games_today.reset_index(drop=True), enc_today_df.reset_index(drop=True)], axis=1)

        return games_today, history

    games_today, history = load_cached_data(selected_file)

    if games_today.empty:
        st.warning("Nenhum jogo encontrado após filtro.")
        return
    if history.empty:
        st.warning("Histórico vazio após filtro.")
        return

    # Converter Asian Line
    history['Asian_Line_Decimal'] = history.get('Asian_Line', np.nan).apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today.get('Asian_Line', np.nan).apply(convert_asian_line_to_decimal)

    # Filtro temporal: histórico < selected_date
    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
            games_today["Date"] = pd.to_datetime(games_today.get("Date", selected_date_str), errors="coerce")
        except Exception:
            pass

    # LiveScore (opcional, apenas para mostrar FT se existir)
    games_today = load_and_merge_livescore(games_today, selected_date_str)

    # Garantir colunas de gols FT no histórico
    if 'Goals_H_FT' not in history.columns:
        history['Goals_H_FT'] = 0
    if 'Goals_A_FT' not in history.columns:
        history['Goals_A_FT'] = 0

    # Weighted Goals no histórico
    if all(c in history.columns for c in ['Odd_H','Odd_D','Odd_A']):
        history = history.copy()
        history['WG_Home'] = history.apply(weighted_goals_home, axis=1)
        history['WG_Away'] = history.apply(weighted_goals_away, axis=1)
    else:
        history['WG_Home'] = 0.0
        history['WG_Away'] = 0.0

    # Rolling WG por time (5 jogos), time-safe (shift(1))
    if "Date" in history.columns:
        history = history.sort_values("Date")
    else:
        history = history.reset_index(drop=True)

    group_home = history.groupby('Home', group_keys=False)
    history['WG_Home_Team'] = group_home['WG_Home'].apply(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1)
    )

    group_away = history.groupby('Away', group_keys=False)
    history['WG_Away_Team'] = group_away['WG_Away'].apply(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1)
    )

    history[['WG_Home_Team','WG_Away_Team']] = history[['WG_Home_Team','WG_Away_Team']].fillna(0)
    history['WG_Diff'] = history['WG_Home_Team'] - history['WG_Away_Team']

    # Mapear WG para os jogos do dia usando último valor conhecido de cada time
    map_wg_home = history.groupby('Home')['WG_Home_Team'].last()
    map_wg_away = history.groupby('Away')['WG_Away_Team'].last()

    games_today['WG_Home_Team'] = games_today['Home'].map(map_wg_home).fillna(0)
    games_today['WG_Away_Team'] = games_today['Away'].map(map_wg_away).fillna(0)
    games_today['WG_Diff'] = games_today['WG_Home_Team'] - games_today['WG_Away_Team']

    # Recalcular Z-scores M / MT para history e today
    history = calcular_zscores_detalhados(history)
    games_today = calcular_zscores_detalhados(games_today)

    # Espaço 3D + cluster
    history = aplicar_clusterizacao_3d(calcular_distancias_3d(history))
    games_today = aplicar_clusterizacao_3d(calcular_distancias_3d(games_today))

    # Criar targets (apenas NO HISTÓRICO com FT)
    history = criar_targets_cover(history)

    # Exportar base final de treino para debug
    export_cols = [
        'League','Date','Home','Away',
        'Goals_H_FT','Goals_A_FT',
        'Asian_Line','Asian_Line_Decimal',
        'Odd_H','Odd_D','Odd_A',
        'Aggression_Home','Aggression_Away',
        'HandScore_Home','HandScore_Away',
        'M_H','M_A','MT_H','MT_A',
        'Quadrant_Dist_3D','Quadrant_Separation_3D',
        'Vector_Sign','Magnitude_3D',
        'Momentum_Diff','Momentum_Diff_MT',
        'WG_Home','WG_Away','WG_Home_Team','WG_Away_Team','WG_Diff',
        'Cover_Home','Cover_Away'
    ]
    for c in export_cols:
        if c not in history.columns:
            history[c] = np.nan

    df_export = history[export_cols].copy()
    file_name = f"TrainingBase_Final_{selected_date_str}.csv"
    df_export_csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download da Base Final de Treino (CSV)",
        data=df_export_csv,
        file_name=file_name,
        mime="text/csv"
    )

    st.markdown("## 🧠 Treinando modelos de Cobertura de Handicap")
    if st.button("🚀 Treinar & Gerar Probabilidades", type="primary"):
        with st.spinner("Treinando modelos..."):

            # FEATURES
            feature_cols = [
                'Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D',
                'Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label',
                'M_H','M_A','MT_H','MT_A',
                'WG_Home_Team','WG_Away_Team','WG_Diff'
            ]
            feature_cols = [f for f in feature_cols if f in history.columns]

            train_mask = history['Cover_Home'].notna() & history['Cover_Away'].notna()
            hist_train = history[train_mask].copy()
            if hist_train.empty:
                st.error("❌ Histórico sem targets válidos de cobertura. Verifique se Goals_FT e Asian_Line estão corretos.")
                return

            X = hist_train[feature_cols].astype(float)
            X = clean_features_for_training(X)

            y_home = hist_train['Cover_Home'].astype(int)
            y_away = hist_train['Cover_Away'].astype(int)

            model_home = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                class_weight='balanced',
                min_samples_leaf=20
            )
            model_away = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                class_weight='balanced',
                min_samples_leaf=20
            )

            model_home.fit(X, y_home)
            model_away.fit(X, y_away)

            # Métricas simples (AUC)
            try:
                ph = model_home.predict_proba(X)[:,1]
                pa = model_away.predict_proba(X)[:,1]
                auc_h = roc_auc_score(y_home, ph)
                auc_a = roc_auc_score(y_away, pa)
                st.success(f"✅ AUC HOME_COVER: {auc_h:.3f} | AUC AWAY_COVER: {auc_a:.3f}")
            except Exception:
                pass

            # Predições nos jogos do dia
            X_today = games_today[feature_cols].astype(float)
            X_today = clean_features_for_training(X_today)

            probs_home = model_home.predict_proba(X_today)[:,1]
            probs_away = model_away.predict_proba(X_today)[:,1]

            games_today['P_Home_Cover'] = probs_home
            games_today['P_Away_Cover'] = probs_away

            # Regras de recomendação simples
            recs = []
            confs = []
            for phc, pac in zip(probs_home, probs_away):
                best_side = None
                best_prob = 0.0
                if phc > pac and phc >= 0.55:
                    best_side = 'BET HOME'
                    best_prob = phc
                elif pac > phc and pac >= 0.55:
                    best_side = 'BET AWAY'
                    best_prob = pac
                else:
                    best_side = 'NO CLEAR EDGE'
                    best_prob = max(phc, pac)

                if best_side == 'NO CLEAR EDGE':
                    conf = 'LOW'
                elif best_prob >= 0.65:
                    conf = 'HIGH'
                else:
                    conf = 'MEDIUM'

                recs.append(best_side)
                confs.append(conf)

            games_today['Recomendacao'] = recs
            games_today['Confidence'] = confs

            # Live score string
            live_info = []
            for _, r in games_today.iterrows():
                g_h = r.get('Goals_H_Today', np.nan)
                g_a = r.get('Goals_A_Today', np.nan)
                h_r = r.get('Home_Red', 0)
                a_r = r.get('Away_Red', 0)
                info = ""
                if pd.notna(g_h) and pd.notna(g_a):
                    info = f"⚽ {int(g_h)}-{int(g_a)}"
                    if h_r and h_r > 0:
                        info += f" 🟥H{int(h_r)}"
                    if a_r and a_r > 0:
                        info += f" 🟥A{int(a_r)}"
                live_info.append(info)

            games_today['Live_Score'] = live_info

            # Mostrar apenas colunas relevantes
            cols_show = [
                'League','Home','Away',
                'Asian_Line','Asian_Line_Decimal',
                'P_Home_Cover','P_Away_Cover',
                'Recomendacao','Confidence',
                'Live_Score'
            ]
            for c in cols_show:
                if c not in games_today.columns:
                    games_today[c] = np.nan

            df_show = games_today[cols_show].copy()
            st.markdown("## 📊 Recomendações de Aposta (Cobertura do Handicap)")
            st.dataframe(df_show, use_container_width=True)

            # Export CSV com resultados de hoje
            out_name = f"AsianCoverAI_Results_{selected_date_str}.csv"
            csv_today = df_show.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download dos Resultados de Hoje (CSV)",
                data=csv_today,
                file_name=out_name,
                mime="text/csv"
            )

            st.success("✅ Análise concluída!")
            st.balloons()

if __name__ == "__main__":
    main()
