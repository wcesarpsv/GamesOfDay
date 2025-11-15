# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# ========================= CONFIG STREAMLIT =========================
st.set_page_config(page_title="Asian Handicap Cover – Dual Model", layout="wide")
st.title("🎯 Asian Handicap Cover – Dual Model (Home + Away)")

# ========================= CONFIGURAÇÕES GERAIS =========================
PAGE_PREFIX = "AsianCoverAI_v2"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

# ============================================================
# 🔧 FUNÇÕES AUXILIARES
# ============================================================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Goals_H_Today','Goals_A_Today','Home_Red','Away_Red','status']:
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
    Converte string de handicap para decimal, sempre na referência HOME.
    Ex.: '-0.5/1' -> média dos módulos -> -0.75 e depois inverte sinal.
    """
    if pd.isna(value): 
        return np.nan
    s = str(value).strip()
    if "/" not in s:
        try:
            num = float(s)
            return -num  # manter convenção: negativo favorece HOME
        except:
            return np.nan
    try:
        parts = [float(p) for p in s.replace("+","").replace("-","").split("/")]
        avg = np.mean(parts)
        sign = -1 if s.startswith("-") else 1
        result = sign * avg
        return -result
    except:
        return np.nan

# ============================================================
# 🔢 Z-SCORES M / MT (recalcular a partir do HandScore_Home/Away)
# ============================================================
def calcular_zscores_detalhados(df):
    """
    Calcula Z-scores a partir do HandScore:
    - M_H, M_A: Z-score do time em relação à liga (performance relativa)
    - MT_H, MT_A: Z-score do time em relação a si mesmo (consistência)
    Usa: HandScore_Home / HandScore_Away
    """
    df = df.copy()
    
    st.info("📊 Calculando Z-scores a partir do HandScore (Home/Away)...")
    
    # 1. Z-SCORE POR LIGA (M_H, M_A)
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
        
        st.success(f"✅ Z-score por liga calculado para {len(df)} jogos")
    else:
        st.warning("⚠️ Colunas 'League' ou 'HandScore_Home/HandScore_Away' não encontradas para Z-score por liga")
        df['M_H'] = 0
        df['M_A'] = 0
    
    # 2. Z-SCORE POR TIME (MT_H, MT_A)
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
        
        st.success(f"✅ Z-score por time calculado para {len(df)} jogos")
        
        df = df.drop(['HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std', 
                      'HT_mean', 'HT_std', 'AT_mean', 'AT_std'], axis=1, errors='ignore')
    else:
        st.warning("⚠️ Colunas 'Home' ou 'Away' não encontradas para Z-score por time")
        df['MT_H'] = 0
        df['MT_A'] = 0
    
    return df

def clean_features_for_training(X):
    """
    Remove infinitos, NaNs e valores extremos das features.
    """
    X_clean = X.copy()
    
    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean, columns=X.columns if hasattr(X, 'columns') else range(X.shape[1]))
    
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    inf_count = (X_clean == np.inf).sum().sum() + (X_clean == -np.inf).sum().sum()
    nan_count = X_clean.isna().sum().sum()
    
    if inf_count > 0 or nan_count > 0:
        st.warning(f"⚠️ Encontrados {inf_count} infinitos e {nan_count} NaNs nas features")
    
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)
    
    for col in X_clean.columns:
        if X_clean[col].dtype in [np.float64, np.float32]:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)
    
    final_inf_count = (X_clean == np.inf).sum().sum() + (X_clean == -np.inf).sum().sum()
    final_nan_count = X_clean.isna().sum().sum()
    
    if final_inf_count > 0 or final_nan_count > 0:
        st.error(f"❌ Ainda existem {final_inf_count} infinitos e {final_nan_count} NaNs — forçando preenchimento 0")
        X_clean = X_clean.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
    
    st.success(f"✅ Features limpas: shape={X_clean.shape}")
    return X_clean

# ============== Espaço 3D (Aggression/Momentum) ==============
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing}")
        for c in ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign',
                  'Magnitude_3D','Momentum_Diff','Momentum_Diff_MT']:
            df[c] = np.nan
        return df
    dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
    dy = (df['M_H'] - df['M_A']) / 2
    dz = (df['MT_H'] - df['MT_A']) / 2
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    return df

def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters=4, random_state=42) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = 0
        return df
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']
    X = df[['dx','dy','dz']].fillna(0).to_numpy()
    n_samples = X.shape[0]
    k = max(1, min(n_clusters, n_samples))
    if n_samples < n_clusters:
        st.info(f"🔧 Ajustando n_clusters: {n_clusters} → {k} (amostras={n_samples})")
    try:
        km = KMeans(n_clusters=k, random_state=random_state, init='k-means++', n_init=10)
        df['Cluster3D_Label'] = km.fit_predict(X)
    except Exception as e:
        st.error(f"❌ Erro no clustering: {e}")
        df['Cluster3D_Label'] = 0
    return df

# ============================================================
# 🆕 WEIGHTED GOALS (WG) BASEADO EM GOLS E ODDS
# ============================================================
def odds_to_probs(odd_h, odd_d, odd_a):
    try:
        odd_h = float(odd_h)
        odd_d = float(odd_d)
        odd_a = float(odd_a)
        if odd_h <= 0 or odd_d <= 0 or odd_a <= 0:
            return 0.33, 0.33, 0.33
        inv_sum = (1/odd_h) + (1/odd_d) + (1/odd_a)
        return (1/odd_h)/inv_sum, (1/odd_d)/inv_sum, (1/odd_a)/inv_sum
    except:
        return 0.33, 0.33, 0.33

def wg_home(row):
    gf = row.get('Goals_H_FT', 0)
    ga = row.get('Goals_A_FT', 0)
    p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 2.5), row.get('Odd_D', 3.0), row.get('Odd_A', 2.5))
    # gols feitos valem mais quando o mercado não esperava tanto (1 - p_h)
    # gols sofridos doem mais quando time era favorito (p_h)
    return (gf * (1 - p_h)) - (ga * p_h)

def wg_away(row):
    gf = row.get('Goals_A_FT', 0)
    ga = row.get('Goals_H_FT', 0)
    p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 2.5), row.get('Odd_D', 3.0), row.get('Odd_A', 2.5))
    # similar, mas na ótica do away
    return (gf * (1 - p_a)) - (ga * p_a)

def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = ['Home','Away','Date','Goals_H_FT','Goals_A_FT','Odd_H','Odd_D','Odd_A']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para WG (usar 0): {missing}")
        df['WG_Home'] = 0.0
        df['WG_Away'] = 0.0
        df['WG_Home_Team'] = 0.0
        df['WG_Away_Team'] = 0.0
        df['WG_Diff'] = 0.0
        return df

    # linha a linha
    df['WG_Home'] = df.apply(wg_home, axis=1)
    df['WG_Away'] = df.apply(wg_away, axis=1)

    # garantir tipo datetime para ordem temporal
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    # rolling 5 jogos por time
    df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']

    st.success("🔥 Weighted Goals (WG_Home_Team / WG_Away_Team / WG_Diff) calculados!")
    return df

# ============================================================
# 🎯 TARGETS: COVER_HOME / COVER_AWAY
# ============================================================
def criar_targets_cobertura(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria dois targets binários:
    - Cover_Home: 1 se o time da casa cobriu o handicap (referência HOME)
    - Cover_Away: 1 se o visitante cobriu o handicap (linha invertida)
    
    margin = Goals_H_FT - Goals_A_FT
    adj = margin + Asian_Line_Decimal
    
    adj > 0 → HOME cobre
    adj < 0 → AWAY cobre
    adj == 0 → PUSH (0 para ambos)
    """
    hist = df.dropna(subset=['Goals_H_FT','Goals_A_FT','Asian_Line_Decimal']).copy()
    if hist.empty:
        return hist
    margin = hist['Goals_H_FT'] - hist['Goals_A_FT']
    adj = margin + hist['Asian_Line_Decimal']
    hist['Cover_Home'] = (adj > 0).astype(int)
    hist['Cover_Away'] = (adj < 0).astype(int)
    return hist

# ============================================================
# ⚖️ UTILS: Liquidação AH + Thresholds Dinâmicos
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
        if adj > 0: scores.append(1.0)
        elif adj == 0: scores.append(0.0)
        else: scores.append(-1.0)
    return sum(scores)/len(scores)

def bucket_line(asian_line_decimal: float) -> str:
    if asian_line_decimal <= -1.0: return "HOME_heavy"
    if -1.0 < asian_line_decimal <= -0.25: return "HOME_light"
    if -0.25 < asian_line_decimal < 0.25: return "EVEN"
    if 0.25 <= asian_line_decimal < 1.0: return "AWAY_light"
    return "AWAY_heavy"

def adjust_threshold_by_line(thr_base: float, asian_line_decimal: float) -> float:
    bl = bucket_line(asian_line_decimal)
    if bl == "EVEN":
        return max(0.05, round(thr_base - 0.05, 2))
    if bl in ("HOME_heavy","AWAY_heavy"):
        return round(thr_base + 0.05, 2)
    return round(thr_base, 2)

def _evaluate_threshold_side(df, side, thr):
    vg_col = 'VG_HOME' if side == 'HOME' else 'VG_AWAY'
    pick = df[df[vg_col].abs() >= thr].copy()
    if pick.empty: 
        return 0.0, 0, 0.0
    margin = pick['Goals_H_FT'] - pick['Goals_A_FT']
    res = [settle_ah_bet(m, l, side) for m,l in zip(margin, pick['Asian_Line_Decimal'])]
    pick['unit'] = res
    n = len(pick)
    win = (pick['unit'] > 0).mean() if n else 0.0
    roi = pick['unit'].mean() if n else 0.0
    return roi, n, win

def find_league_thresholds(history: pd.DataFrame, min_bets=60):
    """
    history deve conter: League, VG_HOME, VG_AWAY, Goals_H_FT, Goals_A_FT, Asian_Line_Decimal
    VG_* = P(Cover) - 0.5
    """
    thr_norm_grid = np.arange(0.10, 0.55, 0.05)
    thr_strong_grid = np.arange(0.25, 0.90, 0.05)

    leagues = sorted(history['League'].dropna().unique().tolist())

    def _best_global(df, grid):
        best_thr, best_roi, best_n = None, -1, 0
        for t in grid:
            roi_h, n_h, _ = _evaluate_threshold_side(df,'HOME',t)
            roi_a, n_a, _ = _evaluate_threshold_side(df,'AWAY',t)
            roi = np.nanmean([roi_h, roi_a])
            if (n_h+n_a) >= min_bets and roi > best_roi:
                best_thr, best_roi, best_n = t, roi, (n_h+n_a)
        return best_thr if best_thr is not None else 0.15

    best_global = {
        'HOME': _best_global(history, thr_norm_grid),
        'AWAY': _best_global(history, thr_norm_grid),
        'HOME_STRONG': _best_global(history, thr_strong_grid),
        'AWAY_STRONG': _best_global(history, thr_strong_grid)
    }

    out = {}
    for lg in leagues:
        df_lg = history[history['League']==lg].copy()
        def _pick(grid, global_thr):
            cands = []
            for t in grid:
                roi_h, n_h, _ = _evaluate_threshold_side(df_lg,'HOME',t)
                roi_a, n_a, _ = _evaluate_threshold_side(df_lg,'AWAY',t)
                roi, n = np.nanmean([roi_h,roi_a]), (n_h+n_a)
                cands.append((t, roi, n))
            if not cands: return global_thr
            score = [(t, roi*np.log1p(max(n,1))) for (t,roi,n) in cands]
            t_local = max(score, key=lambda x:x[1])[0]
            n_tot = max([n for _,_,n in cands])
            alpha = min(1.0, n_tot / max(min_bets,1))
            return round(alpha*t_local + (1-alpha)*global_thr, 2)

        out[lg] = {
            'HOME': _pick(thr_norm_grid, best_global['HOME']),
            'AWAY': _pick(thr_norm_grid, best_global['AWAY']),
            'HOME_STRONG': _pick(thr_strong_grid, best_global['HOME_STRONG']),
            'AWAY_STRONG': _pick(thr_strong_grid, best_global['AWAY_STRONG']),
            'N': len(df_lg)
        }
    out['_GLOBAL'] = best_global | {'N': len(history)}
    return out

# ============================================================
# 📡 LIVE SCORE INTEGRATION (apenas FT, inteiros)
# ============================================================
def load_and_merge_livescore(games_today, selected_date_str):

    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    games_today = setup_livescore_columns(games_today)

    if not os.path.exists(livescore_file):
        st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
        return games_today

    results_df = pd.read_csv(livescore_file)

    results_df['status'] = (
        results_df['status']
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_ft = results_df[results_df['status'] == 'FT'].copy()

    for c in ['home_goal','away_goal','home_red','away_red']:
        df_ft[c] = pd.to_numeric(df_ft[c], errors='coerce').fillna(0).astype(int)

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
# 🧠 MODELOS: P_Home_Cover e P_Away_Cover
# ============================================================
def treinar_modelo_cover_home(history, games_today, features_3d_common):
    st.markdown("### 🏠 Modelo HOME - Probabilidade de Cobrir Handicap")

    hist = criar_targets_cobertura(history)
    if hist.empty or hist['Cover_Home'].nunique() < 2:
        st.error("❌ Target Cover_Home sem variação suficiente para treinar.")
        return None, games_today

    X = hist[features_3d_common].copy()
    X = clean_features_for_training(X)
    y = hist['Cover_Home']

    modelo_home = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    modelo_home.fit(X, y)

    acc = (modelo_home.predict(X) == y).mean()
    st.success(f"✅ HOME Cover - Accuracy (treino): {acc:.3f}")

    X_today = games_today[features_3d_common].copy()
    X_today = clean_features_for_training(X_today)
    proba_home = modelo_home.predict_proba(X_today)[:, 1]

    games_today['P_Home_Cover'] = proba_home
    games_today['Value_Gap_HOME'] = games_today['P_Home_Cover'] - 0.5

    return modelo_home, games_today

def treinar_modelo_cover_away(history, games_today, features_3d_common):
    st.markdown("### ✈️ Modelo AWAY - Probabilidade de Cobrir Handicap")

    hist = criar_targets_cobertura(history)
    if hist.empty or hist['Cover_Away'].nunique() < 2:
        st.error("❌ Target Cover_Away sem variação suficiente para treinar.")
        return None, games_today

    X = hist[features_3d_common].copy()
    X = clean_features_for_training(X)
    y = hist['Cover_Away']

    modelo_away = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    modelo_away.fit(X, y)

    acc = (modelo_away.predict(X) == y).mean()
    st.success(f"✅ AWAY Cover - Accuracy (treino): {acc:.3f}")

    X_today = games_today[features_3d_common].copy()
    X_today = clean_features_for_training(X_today)
    proba_away = modelo_away.predict_proba(X_today)[:, 1]

    games_today['P_Away_Cover'] = proba_away
    games_today['Value_Gap_AWAY'] = games_today['P_Away_Cover'] - 0.5

    return modelo_away, games_today

# ============================================================
# 💎 ANÁLISE DUAL - HOME + AWAY (com thresholds híbridos)
# ============================================================
def analisar_value_bets_dual_modelos(games_today: pd.DataFrame, league_thresholds: dict):
    st.markdown("## 💎 Análise DUAL - Home & Away Models (Cover Probabilities)")
    results = []

    for _, row in games_today.iterrows():
        asian_line = float(row.get('Asian_Line_Decimal', 0) or 0.0)

        p_home = float(row.get('P_Home_Cover', 0) or 0.0)
        p_away = float(row.get('P_Away_Cover', 0) or 0.0)

        value_gap_home = p_home - 0.5
        value_gap_away = p_away - 0.5

        league = row.get('League')
        thr_pack = league_thresholds.get(league, league_thresholds.get('_GLOBAL', {}))
        thr_home = adjust_threshold_by_line(float(thr_pack.get('HOME', 0.15)), asian_line)
        thr_home_str = adjust_threshold_by_line(float(thr_pack.get('HOME_STRONG', 0.30)), asian_line)
        thr_away = adjust_threshold_by_line(float(thr_pack.get('AWAY', 0.15)), asian_line)
        thr_away_str = adjust_threshold_by_line(float(thr_pack.get('AWAY_STRONG', 0.30)), asian_line)

        forca_relativa = p_home - p_away  # diferença de prob de cobrir
        equilibrio = abs(forca_relativa) < 0.05

        recomendacao_final, confidence = "NO CLEAR EDGE", "LOW"

        # CENÁRIO 1: Equilíbrio + mercado puxando forte para um lado
        if equilibrio and abs(asian_line) > 0.25:
            if asian_line < -0.25:  # mercado puxa casa → valor AWAY
                if value_gap_away >= thr_away_str:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
                elif value_gap_away >= thr_away:
                    recomendacao_final, confidence = "BET AWAY", "MEDIUM"
            elif asian_line > 0.25:  # mercado puxa away → valor HOME
                if value_gap_home >= thr_home_str:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
                elif value_gap_home >= thr_home:
                    recomendacao_final, confidence = "BET HOME", "MEDIUM"

        # CENÁRIO 2: Um lado bem mais provável de cobrir + linha até razoável
        elif not equilibrio and abs(asian_line) < 0.75:
            if forca_relativa > 0.10 and asian_line <= 0:
                if value_gap_home >= thr_home_str:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
                elif value_gap_home >= thr_home:
                    recomendacao_final, confidence = "BET HOME", "MEDIUM"
            elif forca_relativa < -0.10 and asian_line >= 0:
                if value_gap_away >= thr_away_str:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
                elif value_gap_away >= thr_away:
                    recomendacao_final, confidence = "BET AWAY", "MEDIUM"

        # CENÁRIO 3: Linhas extremas → exigir edge bem alto
        elif abs(asian_line) >= 1.0:
            if forca_relativa > 0.15 and asian_line < -1.0 and value_gap_home >= thr_home_str:
                recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
            elif forca_relativa < -0.15 and asian_line > 1.0 and value_gap_away >= thr_away_str:
                recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"

        # Live score
        g_h = row.get('Goals_H_Today'); g_a = row.get('Goals_A_Today')
        h_r = row.get('Home_Red'); a_r = row.get('Away_Red')
        live_score_info = ""
        if pd.notna(g_h) and pd.notna(g_a):
            live_score_info = f"⚽ {int(g_h)}-{int(g_a)}"
            if pd.notna(h_r) and int(h_r) > 0: live_score_info += f" 🟥H{int(h_r)}"
            if pd.notna(a_r) and int(a_r) > 0: live_score_info += f" 🟥A{int(a_r)}"

        results.append({
            'League': league,
            'Home': row.get('Home'),
            'Away': row.get('Away'),
            'Asian_Line': row.get('Asian_Line'),
            'Asian_Line_Decimal': asian_line,

            'P_Home_Cover': round(p_home, 3),
            'Value_Gap_HOME': round(value_gap_home, 3),

            'P_Away_Cover': round(p_away, 3),
            'Value_Gap_AWAY': round(value_gap_away, 3),

            'Recomendacao': recomendacao_final,
            'Confidence': confidence,
            'Edge_Difference': round(abs(value_gap_home - value_gap_away), 3),
            'Live_Score': live_score_info
        })

    df_results = pd.DataFrame(results)
    bets_validos = df_results[df_results['Recomendacao'] != 'NO CLEAR EDGE']
    return df_results, bets_validos

# ============================================================
# 📈 VISUALIZAÇÕES
# ============================================================
def plot_analise_dual_modelos(games_today: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    value_gaps_home, value_gaps_away = [], []
    for _, row in games_today.iterrows():
        value_gaps_home.append(float(row.get('Value_Gap_HOME', 0) or 0.0))
        value_gaps_away.append(float(row.get('Value_Gap_AWAY', 0) or 0.0))

    x_pos = list(range(len(value_gaps_home)))
    ax1.bar([x - 0.2 for x in x_pos], value_gaps_home, 0.4, label='HOME Value Gap', alpha=0.7)
    ax1.bar([x + 0.2 for x in x_pos], value_gaps_away, 0.4, label='AWAY Value Gap', alpha=0.7)
    ax1.axhline(y=0, linestyle='-', alpha=0.5)
    ax1.axhline(y=0.15, linestyle='--', alpha=0.5, label='Threshold ~ médio')
    ax1.axhline(y=-0.15, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Jogos'); ax1.set_ylabel('Value Gap (P - 0.5)')
    ax1.set_title('Value Gaps: HOME vs AWAY')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    asian_lines, p_home_list, p_away_list = [], [], []
    for _, row in games_today.iterrows():
        asian_lines.append(float(row.get('Asian_Line_Decimal', 0) or 0.0))
        p_home_list.append(float(row.get('P_Home_Cover', 0) or 0.0))
        p_away_list.append(float(row.get('P_Away_Cover', 0) or 0.0))

    ax2.scatter(asian_lines, p_home_list, alpha=0.7, s=60, label='P(Home Cover)')
    ax2.scatter(asian_lines, p_away_list, alpha=0.7, s=60, label='P(Away Cover)')
    ax2.set_xlabel('Asian Line (Mercado - HOME ref)')
    ax2.set_ylabel('Probabilidade de Cobrir')
    ax2.set_title('P(Cover) vs Asian Line')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL - DUAL MODEL
# ============================================================
def main_calibrado():
    st.info("📂 Carregando dados para Análise DUAL MODEL...")

    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
    if not files:
        st.warning("No CSV files found in GamesDay folder.")
        return
    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)

        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        def classificar_league_tier(league_name: str) -> int:
            if pd.isna(league_name): return 3
            name = league_name.lower()
            if any(x in name for x in ['premier','la liga','serie a','bundesliga','ligue 1','eredivisie',
                                       'primeira liga','brasileirão','super league','mls','championship',
                                       'liga pro','a-league']):
                return 1
            if any(x in name for x in ['serie b','segunda','league 1','liga ii','liga 2','division 2',
                                       'bundesliga 2','ligue 2','j-league','k-league','superettan',
                                       '1st division','national league','liga nacional']):
                return 2
            return 3

        def aplicar_filtro_tier(df: pd.DataFrame, max_tier=3) -> pd.DataFrame:
            if 'League' not in df.columns:
                st.warning("⚠️ Coluna 'League' ausente — filtro de tier não aplicado.")
                df['League_Tier'] = 3
                return df
            df = df.copy()
            df['League_Tier'] = df['League'].apply(classificar_league_tier)
            filtrado = df[df['League_Tier'] <= max_tier].copy()
            st.info(f"🎯 Ligas filtradas (Tier ≤ {max_tier}): {len(filtrado)}/{len(df)} jogos mantidos")
            return filtrado

        history = aplicar_filtro_tier(history, max_tier=3)
        games_today = aplicar_filtro_tier(games_today, max_tier=3)

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
        st.warning("⚠️ Nenhum jogo encontrado após filtro de ligas.")
        return
    if history.empty:
        st.warning("⚠️ Histórico vazio após filtro de ligas.")
        return

    # 🔒 garantir que não perdemos linhas por NaN
    for df in (history, games_today):
        df.fillna(0, inplace=True)

    # Converter Asian Line
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    history = history.dropna(subset=['Asian_Line_Decimal'])
    games_today = games_today.dropna(subset=['Asian_Line_Decimal'])

    # ================= TIME-SAFE: Z-SCORE + WG E SPLIT POR DATA =================
    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")

            if "Date" in games_today.columns:
                games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").fillna(selected_date)
            else:
                games_today["Date"] = selected_date

            history_past = history[history["Date"] < selected_date].copy()
            if history_past.empty:
                st.error("❌ Nenhum jogo passado encontrado para treinar antes da data selecionada.")
                return

            full_df = pd.concat([history_past, games_today], ignore_index=True)

            # M/MT primeiro
            full_df = calcular_zscores_detalhados(full_df)
            # WG em cima de full_df (time-safe, usando Date)
            full_df = adicionar_weighted_goals(full_df)

            # split de volta
            history = full_df[full_df["Date"] < selected_date].copy()
            games_today = full_df[full_df["Date"] >= selected_date].copy()

            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str} (M/MT + WG atualizados)")
        except Exception as e:
            st.warning(f"⚠️ Erro no filtro temporal/Z-score/WG: {e}")
            history = calcular_zscores_detalhados(history)
            history = adicionar_weighted_goals(history)
            games_today = calcular_zscores_detalhados(games_today)
            games_today = adicionar_weighted_goals(games_today)
    else:
        st.warning("⚠️ Coluna 'Date' ausente — Z-score/WG calculados sem controle temporal.")
        history = calcular_zscores_detalhados(history)
        history = adicionar_weighted_goals(history)
        games_today = calcular_zscores_detalhados(games_today)
        games_today = adicionar_weighted_goals(games_today)

    if history.empty:
        st.error("❌ Histórico ficou vazio após aplicação de M/MT + WG e filtro temporal.")
        return

    # LiveScore (apenas FT)
    games_today = load_and_merge_livescore(games_today, selected_date_str)

    # Espaço 3D + clusters
    required_cols = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    missing_history = [c for c in required_cols if c not in history.columns]
    missing_today = [c for c in required_cols if c not in games_today.columns]
    if missing_history or missing_today:
        st.error(f"❌ Colunas necessárias faltando: History={missing_history}, Today={missing_today}")
        return

    history = aplicar_clusterizacao_3d(calcular_distancias_3d(history))
    games_today = aplicar_clusterizacao_3d(calcular_distancias_3d(games_today))

    # Features 3D + WG
    features_3d_common = [
        'Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign',
        'Magnitude_3D','Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label',
        'WG_Home_Team','WG_Away_Team','WG_Diff'
    ]

    st.markdown("## 🧠 Treinando Modelos DUAL (HOME + AWAY Cover)")
    if st.button("🚀 Executar Análise DUAL", type="primary"):
        with st.spinner("Treinando modelos..."):
            hist_for_train = history.dropna(subset=['Goals_H_FT','Goals_A_FT','Asian_Line_Decimal']).copy()
            if hist_for_train.empty:
                st.error("❌ Histórico sem FT/Asian_Line_Decimal suficiente para treinar.")
                return

            modelo_home, games_today = treinar_modelo_cover_home(hist_for_train, games_today, features_3d_common)
            modelo_away, games_today = treinar_modelo_cover_away(hist_for_train, games_today, features_3d_common)
            if modelo_home is None or modelo_away is None:
                st.error("❌ Falha ao treinar um dos modelos (HOME/AWAY).")
                return

            hist_for_pred = criar_targets_cobertura(hist_for_train)
            if hist_for_pred.empty:
                st.error("❌ Histórico sem targets de cobertura válidos.")
                return

            X_hist = clean_features_for_training(hist_for_pred[features_3d_common].copy())
            hist_for_pred['P_Home_Cover'] = modelo_home.predict_proba(X_hist)[:, 1]
            hist_for_pred['P_Away_Cover'] = modelo_away.predict_proba(X_hist)[:, 1]
            hist_for_pred['VG_HOME'] = hist_for_pred['P_Home_Cover'] - 0.5
            hist_for_pred['VG_AWAY'] = hist_for_pred['P_Away_Cover'] - 0.5

            league_thresholds = find_league_thresholds(hist_for_pred, min_bets=60)

            df_value_bets_dual, bets_validos_dual = analisar_value_bets_dual_modelos(games_today, league_thresholds)

            st.markdown("## 📊 Resultados - Análise DUAL")
            if bets_validos_dual.empty:
                st.warning("⚠️ Nenhuma recomendação de value bet encontrada")
            else:
                st.dataframe(bets_validos_dual, use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("🏠 HOME Bets", int((bets_validos_dual['Recomendacao'].str.contains('HOME')).sum()))
                with c2:
                    st.metric("✈️ AWAY Bets", int((bets_validos_dual['Recomendacao'].str.contains('AWAY')).sum()))
                with c3:
                    st.metric("🎯 Strong Bets", int((bets_validos_dual['Confidence'].eq('HIGH')).sum()))
                with c4:
                    st.metric("📊 Total Recomendações", int(len(bets_validos_dual)))

            st.pyplot(plot_analise_dual_modelos(games_today))
            st.success("✅ Análise DUAL concluída com sucesso!")
            st.balloons()

if __name__ == "__main__":
    main_calibrado()
