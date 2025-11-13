# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math, joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

# ========================= CONFIG STREAMLIT =========================
st.set_page_config(page_title="Analisador de Handicap Ótimo - DUAL MODEL", layout="wide")
st.title("🎯 Analisador de Handicap Ótimo - Dual Model (Home + Away)")

# ========================= CONFIGURAÇÕES GERAIS =========================
PAGE_PREFIX = "HandicapOptimizer_DualModel"
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
    if pd.isna(value): return np.nan
    s = str(value).strip()
    if "/" not in s:
        try:
            num = float(s)
            return -num  # manter convenção HOME (negativo favorece casa)
        except:
            return np.nan
    try:
        # média de split (ex.: "-0.5/1" -> [-0.5, -1] -> média=-0.75), preserva sinal e depois inverte p/ HOME
        parts = [float(p) for p in s.replace("+","").replace("-","").split("/")]
        avg = np.mean(parts)
        sign = -1 if s.startswith("-") else 1
        result = sign * avg
        return -result
    except:
        return np.nan

# ============== Espaço 3D (Aggression/Momentum) ==============
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing}")
        for c in ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT']:
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
        if k > 1:
            cents = pd.DataFrame(km.cluster_centers_, columns=['dx','dy','dz'])
            cents['Cluster'] = range(k)
            # st.markdown("### 🧭 Clusters 3D (KMeans)")
            # st.dataframe(cents.style.format({'dx':'{:.2f}','dy':'{:.2f}','dz':'{:.2f}'}))
    except Exception as e:
        st.error(f"❌ Erro no clustering: {e}")
        df['Cluster3D_Label'] = 0
    return df

# ============== Handicap ótimo (HOME/AWAY) ===================
def calcular_handicap_otimo_calibrado_v2(row):
    gh, ga = row.get('Goals_H_FT', 0), row.get('Goals_A_FT', 0)
    margin = gh - ga
    handicaps = [-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0,+0.25,+0.5,+0.75,+1.0,+1.25,+1.5]
    best_h, best_score = 0, -10
    for h in handicaps:
        adj = margin + h
        if adj > 0:
            base = 1.5
            if abs(h) > 1.0: base -= 0.8
            elif abs(h) > 0.75: base -= 0.4
            elif abs(h) > 0.5: base -= 0.2
            score = base - abs(h)*0.1
        elif adj == 0:
            score = 0.3
        else:
            score = -0.5 - abs(h)*0.15
        if score > best_score:
            best_score, best_h = score, h
    if abs(best_h) > 1.0:
        best_h *= 0.6
    elif abs(best_h) > 0.75:
        best_h *= 0.8
    return best_h

def criar_target_handicap_discreto_calibrado_v2(row):
    h = calcular_handicap_otimo_calibrado_v2(row)
    if h <= -0.75: return 'MODERATE_HOME'
    elif h <= -0.25: return 'LIGHT_HOME'
    elif h == 0: return 'NEUTRAL'
    elif h < 0.5: return 'LIGHT_AWAY'
    else: return 'MODERATE_AWAY'

def calcular_handicap_otimo_away(row):
    gh, ga = row.get('Goals_H_FT', 0), row.get('Goals_A_FT', 0)
    margin = ga - gh  # invertido p/ AWAY
    handicaps = [-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0,+0.25,+0.5,+0.75,+1.0,+1.25,+1.5]
    best_h, best_score = 0, -10
    for h in handicaps:
        adj = margin + h
        if adj > 0:
            base = 1.5
            if abs(h) > 1.0: base -= 0.8
            elif abs(h) > 0.75: base -= 0.4
            elif abs(h) > 0.5: base -= 0.2
            score = base - abs(h)*0.1
        elif adj == 0:
            score = 0.3
        else:
            score = -0.5 - abs(h)*0.15
        if score > best_score:
            best_score, best_h = score, h
    if abs(best_h) > 1.0:
        best_h *= 0.6
    elif abs(best_h) > 0.75:
        best_h *= 0.8
    return best_h

def criar_target_handicap_away_discreto_calibrado(row):
    h = calcular_handicap_otimo_away(row)
    if h <= -0.75: return 'MODERATE_AWAY'
    elif h <= -0.25: return 'LIGHT_AWAY'
    elif h == 0: return 'NEUTRAL'
    elif h < 0.5: return 'LIGHT_HOME'
    else: return 'MODERATE_HOME'

# ============================================================
# 🧠 MODELOS HOME (Regressão e Classificação)
# ============================================================
def treinar_modelo_handicap_regressao_calibrado_v2(history, games_today):
    st.markdown("### 📈 Modelo HOME Regressão Calibrado")
    history = history.copy()
    history['Handicap_Otimo_Calibrado'] = history.apply(calcular_handicap_otimo_calibrado_v2, axis=1)
    history_cal = history[(history['Handicap_Otimo_Calibrado']>=-1.25)&(history['Handicap_Otimo_Calibrado']<=1.25)].copy()
    st.info(f"📊 Dados HOME calibrados: {len(history_cal)} jogos")

    features = ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label']
    avail = [f for f in features if f in history_cal.columns]
    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento HOME (regressão)")
        return None, games_today, None

    X = history_cal[avail].fillna(0)
    y = history_cal['Handicap_Otimo_Calibrado']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_leaf=20, max_features=0.6, random_state=42
    )
    model.fit(Xs, y)
    mae = mean_absolute_error(y, model.predict(Xs))
    st.success(f"✅ MAE HOME: {mae:.3f}")

    # hoje
    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns: X_today[f] = 0
    X_today = X_today[avail].fillna(0)
    preds = np.clip(model.predict(scaler.transform(X_today)), -1.25, 1.25)
    games_today['Handicap_Predito_Regressao_Calibrado'] = preds
    return model, games_today, scaler

def treinar_modelo_handicap_classificacao_calibrado_v2(history, games_today):
    st.markdown("### 🎯 Modelo HOME Classificação Calibrado")
    history = history.copy()
    history['Handicap_Categoria_Calibrado'] = history.apply(criar_target_handicap_discreto_calibrado_v2, axis=1)
    features = ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label']
    avail = [f for f in features if f in history.columns]
    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento HOME (classificação)")
        return None, games_today, None
    X = history[avail].fillna(0)
    y = history['Handicap_Categoria_Calibrado']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', min_samples_leaf=15
    )
    model.fit(X, y_enc)

    # hoje
    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns: X_today[f] = 0
    X_today = X_today[avail].fillna(0)
    preds_enc = model.predict(X_today)
    probas = model.predict_proba(X_today)
    games_today['Handicap_Categoria_Predito_Calibrado'] = le.inverse_transform(preds_enc)
    games_today['Confianca_Categoria_Calibrado'] = np.max(probas, axis=1)

    map_cat = {'MODERATE_HOME': -0.75, 'LIGHT_HOME': -0.25, 'NEUTRAL': 0, 'LIGHT_AWAY': +0.25, 'MODERATE_AWAY': +0.75}
    games_today['Handicap_Predito_Classificacao_Calibrado'] = games_today['Handicap_Categoria_Predito_Calibrado'].map(map_cat)
    st.info(f"📊 Distribuição HOME (hist): {dict(history['Handicap_Categoria_Calibrado'].value_counts())}")
    return model, games_today, le

# ============================================================
# 🧠 MODELOS AWAY (Regressão e Classificação)
# ============================================================
def treinar_modelo_away_handicap_regressao_calibrado(history, games_today):
    st.markdown("### 📈 Modelo AWAY Regressão Calibrado")
    history = history.copy()
    history['Handicap_Otimo_AWAY_Calibrado'] = history.apply(calcular_handicap_otimo_away, axis=1)
    history_cal = history[(history['Handicap_Otimo_AWAY_Calibrado']>=-1.25)&(history['Handicap_Otimo_AWAY_Calibrado']<=1.25)].copy()
    st.info(f"📊 Dados AWAY calibrados: {len(history_cal)} jogos")

    features = ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label']
    avail = [f for f in features if f in history_cal.columns]
    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento AWAY (regressão)")
        return None, games_today, None
    X = history_cal[avail].fillna(0)
    y = history_cal['Handicap_Otimo_AWAY_Calibrado']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=20, max_features=0.6, random_state=42
    )
    model.fit(Xs, y)
    mae = mean_absolute_error(y, model.predict(Xs))
    st.success(f"✅ MAE AWAY: {mae:.3f}")

    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns: X_today[f] = 0
    X_today = X_today[avail].fillna(0)
    preds = np.clip(model.predict(scaler.transform(X_today)), -1.25, 1.25)
    games_today['Handicap_AWAY_Predito_Regressao_Calibrado'] = preds
    return model, games_today, scaler

def treinar_modelo_away_handicap_classificacao_calibrado(history, games_today):
    st.markdown("### 🎯 Modelo AWAY Classificação Calibrado")
    history = history.copy()
    history['Handicap_Categoria_AWAY_Calibrado'] = history.apply(criar_target_handicap_away_discreto_calibrado, axis=1)
    features = ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label']
    avail = [f for f in features if f in history.columns]
    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento AWAY (classificação)")
        return None, games_today, None
    X = history[avail].fillna(0)
    y = history['Handicap_Categoria_AWAY_Calibrado']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=5, random_state=42, class_weight='balanced', min_samples_leaf=15
    )
    model.fit(X, y_enc)

    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns: X_today[f] = 0
    X_today = X_today[avail].fillna(0)
    preds_enc = model.predict(X_today)
    probas = model.predict_proba(X_today)
    games_today['Handicap_Categoria_AWAY_Predito_Calibrado'] = le.inverse_transform(preds_enc)
    games_today['Confianca_Categoria_AWAY_Calibrado'] = np.max(probas, axis=1)

    map_cat = {'MODERATE_AWAY': -0.75, 'LIGHT_AWAY': -0.25, 'NEUTRAL': 0, 'LIGHT_HOME': +0.25, 'MODERATE_HOME': +0.75}
    games_today['Handicap_AWAY_Predito_Classificacao_Calibrado'] = games_today['Handicap_Categoria_AWAY_Predito_Calibrado'].map(map_cat)
    st.info(f"📊 Distribuição AWAY (hist): {dict(history['Handicap_Categoria_AWAY_Calibrado'].value_counts())}")
    return model, games_today, le

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

def _predict_block(history, features, scaler, model_reg, model_cls, label_enc, map_cat_num):
    """
    history        : DataFrame com linhas para prever (histórico com FT).
    features       : lista de features numéricas usadas no modelo.
    scaler         : scaler usado no modelo de regressão (StandardScaler).
    model_reg      : modelo de regressão treinado (HOME ou AWAY).
    model_cls      : modelo de classificação treinado (HOME ou AWAY).
    label_enc      : LabelEncoder correspondente ao modelo de classificação.
    map_cat_num    : dicionário label -> valor numérico (no referencial NATIVO desse modelo).
    """
    Xh = history.copy()
    for f in features:
        if f not in Xh.columns:
            Xh[f] = 0
    X = Xh[features].fillna(0)

    # regressão no espaço escalado
    Xs = scaler.transform(X)
    pred_reg = np.clip(model_reg.predict(Xs), -1.25, 1.25)

    # classificação → labels → valores numéricos
    pred_cls_enc = model_cls.predict(X)
    pred_cls_labels = label_enc.inverse_transform(pred_cls_enc)
    pred_cls_num = np.array([map_cat_num.get(lbl, 0.0) for lbl in pred_cls_labels])

    # combinação (mesma lógica: 70% reg, 30% cls)
    return 0.7 * pred_reg + 0.3 * pred_cls_num


def bucket_line(asian_line_decimal: float) -> str:
    # buckets largos e estáveis
    if asian_line_decimal <= -1.0: return "HOME_heavy"
    if -1.0 < asian_line_decimal <= -0.25: return "HOME_light"
    if -0.25 < asian_line_decimal < 0.25: return "EVEN"
    if 0.25 <= asian_line_decimal < 1.0: return "AWAY_light"
    return "AWAY_heavy"

def adjust_threshold_by_line(thr_base: float, asian_line_decimal: float) -> float:
    # ajuste fino (híbrido): linhas equilibradas pedem menor threshold; linhas extremas pedem maior
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

    # ============================
    # 🧭 MODO MODERADO – calibrado
    # ============================

    # Bets normais: sensibilidade média (≥ 0.10)
    thr_norm_grid = np.arange(0.10, 0.25, 0.05)

    # Strong bets: exigência maior (≥ 0.20)
    thr_strong_grid = np.arange(0.20, 0.40, 0.05)



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

    # Normalizar status
    results_df['status'] = (
        results_df['status']
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Filtrar FT
    df_ft = results_df[results_df['status'] == 'FT'].copy()

    # Garantir inteiros
    for c in ['home_goal','away_goal','home_red','away_red']:
        df_ft[c] = pd.to_numeric(df_ft[c], errors='coerce').fillna(0).astype(int)

    # MERGE — mas salvando status com outro nome para evitar conflito
    games_today = games_today.merge(
        df_ft[['Id','status','home_goal','away_goal','home_red','away_red']],
        on='Id',
        how='left',
        suffixes=('', '_ls')
    )

    # Agora a coluna correta é: status_ls
    mask_ft = games_today['status_ls'] == 'FT'

    # Preencher apenas FT
    games_today.loc[mask_ft, 'Goals_H_Today'] = games_today.loc[mask_ft, 'home_goal']
    games_today.loc[mask_ft, 'Goals_A_Today'] = games_today.loc[mask_ft, 'away_goal']
    games_today.loc[mask_ft, 'Home_Red']        = games_today.loc[mask_ft, 'home_red']
    games_today.loc[mask_ft, 'Away_Red']        = games_today.loc[mask_ft, 'away_red']

    return games_today




def analisar_value_bets_dual_modelos(games_today: pd.DataFrame, league_thresholds: dict):
    st.markdown("## 💎 Análise DUAL - Home & Away Models")
    results = []

    for _, row in games_today.iterrows():
        asian_line = float(row.get('Asian_Line_Decimal', 0) or 0.0)

        # ===== PREDIÇÕES COMBINADAS (referenciais distintos e corretos)
        pred_home = (
            0.7 * float(row.get('Handicap_Predito_Regressao_Calibrado', 0))
            + 0.3 * float(row.get('Handicap_Predito_Classificacao_Calibrado', 0))
        )

        pred_away = (
            0.7 * float(row.get('Handicap_AWAY_Predito_Regressao_Calibrado', 0))
            + 0.3 * float(row.get('Handicap_AWAY_Predito_Classificacao_Calibrado', 0))
        )

        # ===== VALUE GAPS (NÃO usar abs aqui!)
        # HOME refere-se a Asian_Line (HOME)
        value_gap_home = pred_home - asian_line
        # AWAY refere-se a Asian_Line invertido
        value_gap_away = pred_away - (-asian_line)

        # ===== THRESHOLDS POR LIGA
        league = row.get('League')
        thr_pack = league_thresholds.get(league, league_thresholds.get('_GLOBAL', {}))

        thr_home      = adjust_threshold_by_line(float(thr_pack.get('HOME',        0.15)), asian_line)
        thr_home_str  = adjust_threshold_by_line(float(thr_pack.get('HOME_STRONG', 0.30)), asian_line)
        thr_away      = adjust_threshold_by_line(float(thr_pack.get('AWAY',        0.15)), asian_line)
        thr_away_str  = adjust_threshold_by_line(float(thr_pack.get('AWAY_STRONG', 0.30)), asian_line)

        # ===== FORÇA RELATIVA (diferença entre modelos)
        forca_relativa = pred_home - pred_away
        equilibrio = abs(forca_relativa) < 0.20  # equilíbrio moderado

        recomendacao_final = "NO CLEAR EDGE"
        confidence = "LOW"

        # ====================================================
        # 🔥 CENÁRIO 1: Equilíbrio + Mercado tendenciando um lado
        # ====================================================
        if equilibrio and abs(asian_line) > 0.25:

            # Mercado puxou HOME → line negativa → valor tende a AWAY
            if asian_line < -0.25:
                # AWAY só entra se value_gap_away >= threshold (SEM ABS)
                if value_gap_away >= thr_away_str:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
                elif value_gap_away >= thr_away:
                    recomendacao_final, confidence = "BET AWAY", "MEDIUM"

            # Mercado puxou AWAY → line positiva → valor tende a HOME
            elif asian_line > 0.25:
                if value_gap_home >= thr_home_str:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
                elif value_gap_home >= thr_home:
                    recomendacao_final, confidence = "BET HOME", "MEDIUM"

        # ====================================================
        # 🔥 CENÁRIO 2: Um lado mais forte + linha moderada (<0.75)
        # ====================================================
        elif not equilibrio and abs(asian_line) < 0.75:

            # Modelo HOME muito superior → handicap negativo → valor em HOME
            if forca_relativa > 0.30 and asian_line < 0:
                cobertura = forca_relativa - abs(asian_line)
                if cobertura >= (thr_home_str / 2):
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"
                elif cobertura >= (thr_home / 2):
                    recomendacao_final, confidence = "BET HOME", "MEDIUM"

            # Modelo AWAY muito superior → handicap positivo → valor em AWAY
            elif forca_relativa < -0.30 and asian_line > 0:
                cobertura = abs(forca_relativa) - abs(asian_line)
                if cobertura >= (thr_away_str / 2):
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"
                elif cobertura >= (thr_away / 2):
                    recomendacao_final, confidence = "BET AWAY", "MEDIUM"

        # ====================================================
        # 🔥 CENÁRIO 3: Linhas extremas (>= 1.0)
        # ====================================================
        elif abs(asian_line) >= 1.0:
            # Linha muito negativa → mercado puxando fortemente HOME → só entra se gap HOME for grande
            if forca_relativa > 0.50 and asian_line < -1.0:
                if value_gap_home >= thr_home_str:
                    recomendacao_final, confidence = "STRONG BET HOME", "HIGH"

            # Linha muito positiva → mercado puxando AWAY forte → só entra se gap AWAY for grande
            elif forca_relativa < -0.50 and asian_line > 1.0:
                if value_gap_away >= thr_away_str:
                    recomendacao_final, confidence = "STRONG BET AWAY", "HIGH"

        # ====================================================
        # LIVE SCORE
        # ====================================================
        g_h = row.get('Goals_H_Today'); g_a = row.get('Goals_A_Today')
        h_r = row.get('Home_Red'); a_r = row.get('Away_Red')
        live_score_info = ""
        if pd.notna(g_h) and pd.notna(g_a):
            live_score_info = f"⚽ {int(g_h)}-{int(g_a)}"
            if pd.notna(h_r) and int(h_r) > 0: live_score_info += f" 🟥H{int(h_r)}"
            if pd.notna(a_r) and int(a_r) > 0: live_score_info += f" 🟥A{int(a_r)}"

        # ====================================================
        # REGISTRO FINAL DA RECOMENDAÇÃO
        # ====================================================
        results.append({
            'League': row.get('League'),
            'Home': row.get('Home'),
            'Away': row.get('Away'),
            'Asian_Line': row.get('Asian_Line'),
            'Asian_Line_Decimal': asian_line,

            'Handicap_HOME_Predito': round(pred_home, 2),
            'Value_Gap_HOME': round(value_gap_home, 2),

            'Handicap_AWAY_Predito': round(pred_away, 2),
            'Value_Gap_AWAY': round(value_gap_away, 2),

            'Recomendacao': recomendacao_final,
            'Confidence': confidence,
            'Edge_Difference': round(abs(value_gap_home - value_gap_away), 2),

            'Live_Score': live_score_info
        })

    df_results = pd.DataFrame(results)
    bets_validos = df_results[df_results['Recomendacao'] != 'NO CLEAR EDGE']
    return df_results, bets_validos



# ============================================================
# 📈 VISUALIZAÇÕES DUAL (eixo HOME unificado)
# ============================================================
def plot_analise_dual_modelos(games_today: pd.DataFrame):
    """
    Gráfico revisado:
    - Subplot 1: Value Gap HOME vs AWAY (já convertendo AWAY para eixo HOME)
    - Subplot 2: Predito HOME e Predito AWAY (em eixo HOME) vs linha do mercado
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    value_gaps_home, value_gaps_away_home = [], []
    asian_lines = []
    fair_lines_dual = []

    for _, row in games_today.iterrows():
        asian_line = float(row.get('Asian_Line_Decimal', 0) or 0.0)

        pred_home = (
            0.7 * float(row.get('Handicap_Predito_Regressao_Calibrado', 0) or 0.0) +
            0.3 * float(row.get('Handicap_Predito_Classificacao_Calibrado', 0) or 0.0)
        )
        pred_away_raw = (
            0.7 * float(row.get('Handicap_AWAY_Predito_Regressao_Calibrado', 0) or 0.0) +
            0.3 * float(row.get('Handicap_AWAY_Predito_Classificacao_Calibrado', 0) or 0.0)
        )
        pred_away_home_axis = -pred_away_raw

        vg_home = pred_home - asian_line
        vg_away_home = pred_away_home_axis - asian_line
        value_gaps_home.append(vg_home)
        value_gaps_away_home.append(vg_away_home)

        asian_lines.append(asian_line)
        fair_lines_dual.append((pred_home + pred_away_home_axis) / 2.0)

    # ---------------- Subplot 1: Value Gaps ----------------
    x_pos = list(range(len(value_gaps_home)))
    ax1.bar([x - 0.2 for x in x_pos], value_gaps_home, 0.4, label='HOME Value Gap', alpha=0.7)
    ax1.bar([x + 0.2 for x in x_pos], value_gaps_away_home, 0.4, label='AWAY Value Gap (eixo HOME)', alpha=0.7)
    ax1.axhline(y=0, linestyle='-', alpha=0.5)
    ax1.set_xlabel('Jogos')
    ax1.set_ylabel('Value Gap (eixo HOME)')
    ax1.set_title('Value Gaps: HOME vs AWAY (em eixo HOME)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---------------- Subplot 2: Predito vs Mercado ----------------
    ax2.scatter(asian_lines, fair_lines_dual, alpha=0.7, s=60, label='Fair Line DUAL (HOME-axis)')
    ax2.scatter(asian_lines, [ph for ph in [  # lista com os preds HOME
        0.7 * float(row.get('Handicap_Predito_Regressao_Calibrado', 0) or 0.0) +
        0.3 * float(row.get('Handicap_Predito_Classificacao_Calibrado', 0) or 0.0)
        for _, row in games_today.iterrows()
    ]], alpha=0.6, s=40, label='HOME Predito (modelo HOME)')
    ax2.scatter(asian_lines, [ -(
        0.7 * float(row.get('Handicap_AWAY_Predito_Regressao_Calibrado', 0) or 0.0) +
        0.3 * float(row.get('Handicap_AWAY_Predito_Classificacao_Calibrado', 0) or 0.0)
    ) for _, row in games_today.iterrows()], alpha=0.6, s=40, label='AWAY Predito (convertido p/ HOME)')

    ax2.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', alpha=0.3, label='Linha Mercado (y=x)')
    ax2.set_xlabel('Asian Line (Mercado, eixo HOME)')
    ax2.set_ylabel('Handicap Predito (eixo HOME)')
    ax2.set_title('Predito vs Mercado (HOME & AWAY em eixo HOME)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# 🚀 EXECUÇÃO PRINCIPAL - DUAL MODEL
# ============================================================
def main_calibrado():
    st.info("📂 Carregando dados para Análise DUAL MODEL...")

    # Seleção do arquivo do dia
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

        # excluir copas/troféus logo no início
        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        # filtro tier (opcional suave)
        def classificar_league_tier(league_name: str) -> int:
            if pd.isna(league_name): return 3
            name = league_name.lower()
            if any(x in name for x in ['premier','la liga','serie a','bundesliga','ligue 1','eredivisie','primeira liga','brasileirão','super league','mls','championship','liga pro','a-league']):
                return 1
            if any(x in name for x in ['serie b','segunda','league 1','liga ii','liga 2','division 2','bundesliga 2','ligue 2','j-league','k-league','superettan','1st division','national league','liga nacional']):
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
        st.warning("⚠️ Nenhum jogo encontrado após filtro de ligas.")
        return
    if history.empty:
        st.warning("⚠️ Histórico vazio após filtro de ligas.")
        return

    # Converter Asian Line
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    history = history.dropna(subset=['Asian_Line_Decimal'])
    games_today = games_today.dropna(subset=['Asian_Line_Decimal'])

    # Filtro temporal: histórico < selected_date
    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
        except Exception as e:
            st.warning(f"⚠️ Erro no filtro temporal: {e}")

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

    st.markdown("## 🧠 Treinando Modelos DUAL (HOME + AWAY)")
    if st.button("🚀 Executar Análise DUAL", type="primary"):
        with st.spinner("Treinando modelos..."):
    
            # TREINO DOS MODELOS
            modelo_home_reg, games_today, scaler_home = treinar_modelo_handicap_regressao_calibrado_v2(history, games_today)
            modelo_home_cls, games_today, le_home = treinar_modelo_handicap_classificacao_calibrado_v2(history, games_today)
    
            modelo_away_reg, games_today, scaler_away = treinar_modelo_away_handicap_regressao_calibrado(history, games_today)
            modelo_away_cls, games_today, le_away = treinar_modelo_away_handicap_classificacao_calibrado(history, games_today)
    
            # ===== Features comuns (mesmo pipeline aplicado nas previsões)
            features_3d_common = [
                'Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign',
                'Magnitude_3D','Momentum_Diff','Momentum_Diff_MT','Cluster3D_Label'
            ]
    
            # HISTÓRICO para calibrar thresholds
            hist_for_pred = history.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
            if hist_for_pred.empty:
                st.error("❌ Histórico sem FT para calibrar thresholds.")
                return
    
            # MAPEAMENTOS DE CLASSES PARA HOME e AWAY
            map_cat_home_num = {
                'MODERATE_HOME': -0.75,
                'LIGHT_HOME': -0.25,
                'NEUTRAL': 0.0,
                'LIGHT_AWAY': +0.25,
                'MODERATE_AWAY': +0.75,
            }
    
            map_cat_away_num = {
                'MODERATE_AWAY': -0.75,
                'LIGHT_AWAY': -0.25,
                'NEUTRAL': 0.0,
                'LIGHT_HOME': +0.25,
                'MODERATE_HOME': +0.75,
            }
    
            # ===== PREVER O HISTÓRICO (HOME e AWAY SEPARADOS)
            pred_home_hist = _predict_block(
                hist_for_pred,
                features_3d_common,
                scaler_home,
                modelo_home_reg,
                modelo_home_cls,
                le_home,
                map_cat_home_num
            )
    
            pred_away_hist = _predict_block(
                hist_for_pred,
                features_3d_common,
                scaler_away,
                modelo_away_reg,
                modelo_away_cls,
                le_away,
                map_cat_away_num
            )
    
            # Armazenar
            hist_for_pred['Pred_HOME'] = pred_home_hist
            hist_for_pred['Pred_AWAY'] = pred_away_hist
    
            # VG no referencial natural dos modelos (HOME e AWAY)
            hist_for_pred['VG_HOME'] = hist_for_pred['Pred_HOME'] - hist_for_pred['Asian_Line_Decimal']
            hist_for_pred['VG_AWAY'] = hist_for_pred['Pred_AWAY'] - (-hist_for_pred['Asian_Line_Decimal'])
    
            # ===== CALCULAR THRESHOLDS POR LIGA
            league_thresholds = find_league_thresholds(hist_for_pred, min_bets=60)
    
            # ===== ANALISAR VALUE BETS COM O DUAL
            df_value_bets_dual, bets_validos_dual = analisar_value_bets_dual_modelos(games_today, league_thresholds)
    
            st.markdown("## 📊 Resultados - Análise DUAL")
            if bets_validos_dual.empty:
                st.warning("⚠️ Nenhuma recomendação de value bet encontrada")
            else:
                st.dataframe(bets_validos_dual, use_container_width=True)
    
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("🏠 HOME Bets", int(bets_validos_dual['Recomendacao'].str.contains('HOME').sum()))
                with c2: st.metric("✈️ AWAY Bets", int(bets_validos_dual['Recomendacao'].str.contains('AWAY').sum()))
                with c3: st.metric("🎯 Strong Bets", int(bets_validos_dual['Confidence'].eq('HIGH').sum()))
                with c4: st.metric("📊 Total Recomendações", len(bets_validos_dual))
    
            st.pyplot(plot_analise_dual_modelos(games_today))
            st.success("✅ Análise DUAL concluída com sucesso!")
            st.balloons()


if __name__ == "__main__":
    main_calibrado()
