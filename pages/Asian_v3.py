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

MAX_ABS_HANDICAP = 1.5  # limite global de clipping p/ alvo contínuo


# ============================================================
# 🔧 FUNÇÕES AUXILIARES BÁSICAS
# ============================================================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Goals_H_Today', 'Goals_A_Today', 'Home_Red', 'Away_Red', 'status']:
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
    Converte a linha asiática original (que pode vir como string, ex: '-0.5/1')
    para um float no EIXO HOME:
    - negativo favorece HOME
    - positivo favorece AWAY
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if "/" not in s:
        try:
            num = float(s)
            # book normalmente mostra linha no referencial do favorito.
            # Aqui mantemos convenção: valor NEGATIVO favorece HOME.
            return -num
        except Exception:
            return np.nan
    try:
        # split ex: "-0.5/1" → partes sem sinal → [0.5, 1.0]
        parts = [float(p) for p in s.replace("+", "").replace("-", "").split("/")]
        avg = np.mean(parts)
        sign = -1 if s.startswith("-") else 1
        result = sign * avg
        return -result  # converte p/ eixo HOME
    except Exception:
        return np.nan


# ============================================================
# ============== Espaço 3D (Aggression/Momentum) =============
# ============================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing}")
        for c in ['Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
                  'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT']:
            df[c] = np.nan
        return df

    dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
    dy = (df['M_H'] - df['M_A']) / 2
    dz = (df['MT_H'] - df['MT_A']) / 2

    df['Quadrant_Dist_3D'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    return df


def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters=4, random_state=42) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Colunas ausentes para clusterização 3D: {missing}")
        df['Cluster3D_Label'] = 0
        return df

    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    X = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()
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
# 🔵 Handicap Ótimo Contínuo (HOME / AWAY / Conversão)
# ============================================================
def calcular_handicap_otimo_home_continuo(row):
    """
    Alvo contínuo do modelo HOME (eixo HOME).
    margin_home = G_H - G_A
    h_cont = -margin_home
    Clip: [-MAX_ABS_HANDICAP, +MAX_ABS_HANDICAP]
    """
    gh = float(row.get("Goals_H_FT", 0))
    ga = float(row.get("Goals_A_FT", 0))

    margin_home = gh - ga
    h_cont = -margin_home
    return float(np.clip(h_cont, -MAX_ABS_HANDICAP, MAX_ABS_HANDICAP))


def calcular_handicap_otimo_away_continuo(row):
    """
    Alvo contínuo do modelo AWAY (eixo AWAY).
    margin_away = G_A - G_H
    h_cont = -margin_away
    Clip: [-MAX_ABS_HANDICAP, +MAX_ABS_HANDICAP]
    """
    gh = float(row.get("Goals_H_FT", 0))
    ga = float(row.get("Goals_A_FT", 0))

    margin_away = ga - gh
    h_cont = -margin_away
    return float(np.clip(h_cont, -MAX_ABS_HANDICAP, MAX_ABS_HANDICAP))


def converter_handicap_away_para_eixo_home(h_away):
    """
    Converte handicap AWAY (eixo AWAY) para eixo HOME.
    """
    if pd.isna(h_away):
        return np.nan
    return -float(h_away)


# ============================================================
# 🟢 Categorização em 5 Classes (HOME / AWAY)
# ============================================================
def categorizar_handicap_home(h):
    """
    Classes no eixo HOME:
    <= -0.75 → MODERATE_HOME
    <= -0.25 → LIGHT_HOME
     ==  0.0 → NEUTRAL
     < +0.75 → LIGHT_AWAY
    >= +0.75 → MODERATE_AWAY
    """
    if pd.isna(h):
        return "NEUTRAL"

    if h <= -0.75:
        return "MODERATE_HOME"
    if h <= -0.25:
        return "LIGHT_HOME"
    if abs(h) < 1e-9:
        return "NEUTRAL"
    if h < 0.75:
        return "LIGHT_AWAY"
    return "MODERATE_AWAY"


def categorizar_handicap_away(h):
    """
    Classes no eixo AWAY:
    <= -0.75 → MODERATE_AWAY
    <= -0.25 → LIGHT_AWAY
     ==  0.0 → NEUTRAL
     < +0.75 → LIGHT_HOME
    >= +0.75 → MODERATE_HOME
    """
    if pd.isna(h):
        return "NEUTRAL"

    if h <= -0.75:
        return "MODERATE_AWAY"
    if h <= -0.25:
        return "LIGHT_AWAY"
    if abs(h) < 1e-9:
        return "NEUTRAL"
    if h < 0.75:
        return "LIGHT_HOME"
    return "MODERATE_HOME"


# ============================================================
# 🧮 Combinação dinâmica Regressão + Classificação
# ============================================================
def combinar_reg_class(reg_values, cls_values, confidences):
    """
    3B — Combinação dinâmica usando confiança do classificador.

    peso_cls = 0.2 + 0.3 * conf
    peso_reg = 1 - peso_cls
    """
    reg = np.asarray(reg_values, dtype=float)
    cls = np.asarray(cls_values, dtype=float)
    conf = np.asarray(confidences, dtype=float)

    conf = np.clip(conf, 0.0, 1.0)
    w_cls = 0.2 + 0.3 * conf
    w_reg = 1.0 - w_cls

    return w_reg * reg + w_cls * cls


# ============================================================
# 🧠 MODELOS HOME — Regressão e Classificação
# ============================================================
def treinar_modelo_handicap_regressao_calibrado_v2(history, games_today):
    """
    Modelo HOME (regressão) com alvo contínuo matemático.
    """
    st.markdown("### 📈 Modelo HOME – Regressão (alvo contínuo novo)")
    history = history.copy()

    # alvo contínuo
    history['Handicap_Home_Cont'] = history.apply(
        calcular_handicap_otimo_home_continuo, axis=1
    )

    history_reg = history.copy()
    st.info(f"📊 Dados HOME (regressão contínua): {len(history_reg)} jogos")

    features = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    avail = [f for f in features if f in history_reg.columns]

    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento HOME (regressão)")
        return None, games_today, None

    X = history_reg[avail].fillna(0.0)
    y = history_reg['Handicap_Home_Cont']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=25,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xs, y)

    preds_in_sample = model.predict(Xs)
    mae = mean_absolute_error(y, preds_in_sample)
    st.success(f"✅ MAE HOME (contínuo): {mae:.3f}")

    # previsões hoje
    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns:
            X_today[f] = 0.0
    X_today = X_today[avail].fillna(0.0)

    preds_today = model.predict(scaler.transform(X_today))
    preds_today = np.clip(preds_today, -MAX_ABS_HANDICAP, MAX_ABS_HANDICAP)

    games_today['Handicap_Predito_Regressao_Calibrado'] = preds_today

    return model, games_today, scaler


def treinar_modelo_handicap_classificacao_calibrado_v2(history, games_today):
    """
    Modelo HOME (classificação) com 5 classes em eixo HOME.
    """
    st.markdown("### 🎯 Modelo HOME – Classificação (5 classes)")
    history = history.copy()

    # garante alvo contínuo (se ainda não tiver)
    if 'Handicap_Home_Cont' not in history.columns:
        history['Handicap_Home_Cont'] = history.apply(
            calcular_handicap_otimo_home_continuo, axis=1
        )

    history['Handicap_Home_Class'] = history['Handicap_Home_Cont'].apply(
        categorizar_handicap_home
        )

    features = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    avail = [f for f in features if f in history.columns]

    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento HOME (classificação)")
        return None, games_today, None

    X = history[avail].fillna(0.0)
    y = history['Handicap_Home_Class']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X, y_enc)

    # previsões hoje
    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns:
            X_today[f] = 0.0
    X_today = X_today[avail].fillna(0.0)

    preds_enc = model.predict(X_today)
    probas = model.predict_proba(X_today)
    preds_labels = le.inverse_transform(preds_enc)
    confs = np.max(probas, axis=1)

    # mapeia classes → valor numérico no eixo HOME
    map_cat_home_num = {
        'MODERATE_HOME': -1.0,
        'LIGHT_HOME': -0.5,
        'NEUTRAL': 0.0,
        'LIGHT_AWAY': 0.5,
        'MODERATE_AWAY': 1.0,
    }
    preds_num = np.array([map_cat_home_num.get(lbl, 0.0) for lbl in preds_labels])

    games_today['Handicap_Categoria_Predito_Calibrado'] = preds_labels
    games_today['Confianca_Categoria_Calibrado'] = confs
    games_today['Handicap_Predito_Classificacao_Calibrado'] = preds_num

    st.info(f"📊 Distribuição HOME (hist): {dict(history['Handicap_Home_Class'].value_counts())}")

    return model, games_today, le


# ============================================================
# 🧠 MODELOS AWAY — Regressão e Classificação
# ============================================================
def treinar_modelo_away_handicap_regressao_calibrado(history, games_today):
    """
    Modelo AWAY (regressão) com alvo contínuo matemático (eixo AWAY).
    """
    st.markdown("### 📈 Modelo AWAY – Regressão (alvo contínuo novo)")
    history = history.copy()

    history['Handicap_Away_Cont'] = history.apply(
        calcular_handicap_otimo_away_continuo, axis=1
    )

    history_reg = history.copy()
    st.info(f"📊 Dados AWAY (regressão contínua): {len(history_reg)} jogos")

    features = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    avail = [f for f in features if f in history_reg.columns]

    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento AWAY (regressão)")
        return None, games_today, None

    X = history_reg[avail].fillna(0.0)
    y = history_reg['Handicap_Away_Cont']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=25,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xs, y)

    preds_in_sample = model.predict(Xs)
    mae = mean_absolute_error(y, preds_in_sample)
    st.success(f"✅ MAE AWAY (contínuo): {mae:.3f}")

    # previsões hoje (eixo AWAY)
    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns:
            X_today[f] = 0.0
    X_today = X_today[avail].fillna(0.0)

    preds_today = model.predict(scaler.transform(X_today))
    preds_today = np.clip(preds_today, -MAX_ABS_HANDICAP, MAX_ABS_HANDICAP)

    games_today['Handicap_AWAY_Predito_Regressao_Calibrado'] = preds_today

    return model, games_today, scaler


def treinar_modelo_away_handicap_classificacao_calibrado(history, games_today):
    """
    Modelo AWAY (classificação) com 5 classes em eixo AWAY.
    """
    st.markdown("### 🎯 Modelo AWAY – Classificação (5 classes)")
    history = history.copy()

    if 'Handicap_Away_Cont' not in history.columns:
        history['Handicap_Away_Cont'] = history.apply(
            calcular_handicap_otimo_away_continuo, axis=1
        )

    history['Handicap_Away_Class'] = history['Handicap_Away_Cont'].apply(
        categorizar_handicap_away
    )

    features = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
        'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
    ]
    avail = [f for f in features if f in history.columns]

    if len(avail) < 3:
        st.error("❌ Features insuficientes para treinamento AWAY (classificação)")
        return None, games_today, None

    X = history[avail].fillna(0.0)
    y = history['Handicap_Away_Class']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X, y_enc)

    # previsões hoje (eixo AWAY)
    X_today = games_today.copy()
    for f in avail:
        if f not in X_today.columns:
            X_today[f] = 0.0
    X_today = X_today[avail].fillna(0.0)

    preds_enc = model.predict(X_today)
    probas = model.predict_proba(X_today)
    preds_labels = le.inverse_transform(preds_enc)
    confs = np.max(probas, axis=1)

    map_cat_away_num = {
        'MODERATE_AWAY': -1.0,
        'LIGHT_AWAY': -0.5,
        'NEUTRAL': 0.0,
        'LIGHT_HOME': 0.5,
        'MODERATE_HOME': 1.0,
    }
    preds_num = np.array([map_cat_away_num.get(lbl, 0.0) for lbl in preds_labels])

    games_today['Handicap_Categoria_AWAY_Predito_Calibrado'] = preds_labels
    games_today['Confianca_Categoria_AWAY_Calibrado'] = confs
    games_today['Handicap_AWAY_Predito_Classificacao_Calibrado'] = preds_num

    st.info(f"📊 Distribuição AWAY (hist): {dict(history['Handicap_Away_Class'].value_counts())}")

    return model, games_today, le


# ============================================================
# ⚖️ UTILS: Liquidação AH + VG + Thresholds
# ============================================================
def _split_quarter(line):
    # ex: -0.75 -> [-0.5, -1.0]; +0.25 -> [0.0, +0.5]
    if (abs(line) * 100) % 50 == 0:
        return [line]
    base = math.floor(abs(line) * 2) / 2.0
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
    return sum(scores) / len(scores)


def bucket_line(asian_line_decimal: float) -> str:
    if asian_line_decimal <= -1.0:
        return "HOME_heavy"
    if -1.0 < asian_line_decimal <= -0.25:
        return "HOME_light"
    if -0.25 < asian_line_decimal < 0.25:
        return "EVEN"
    if 0.25 <= asian_line_decimal < 1.0:
        return "AWAY_light"
    return "AWAY_heavy"


def adjust_threshold_by_line(thr_base: float, asian_line_decimal: float) -> float:
    bl = bucket_line(asian_line_decimal)
    if bl == "EVEN":
        return max(0.05, round(thr_base - 0.05, 2))
    if bl in ("HOME_heavy", "AWAY_heavy"):
        return round(thr_base + 0.05, 2)
    return round(thr_base, 2)


def _evaluate_threshold_side(df, side, thr):
    """
    df   : DataFrame com VG_HOME / VG_AWAY (eixo HOME)
    side : 'HOME' ou 'AWAY'
    thr  : threshold de |VG| para considerar aposta
    """
    vg_col = 'VG_HOME' if side == 'HOME' else 'VG_AWAY'
    pick = df[df[vg_col].abs() >= thr].copy()
    if pick.empty:
        return 0.0, 0, 0.0

    margin = pick['Goals_H_FT'] - pick['Goals_A_FT']
    res = [settle_ah_bet(m, l, side) for m, l in zip(margin, pick['Asian_Line_Decimal'])]
    pick['unit'] = res

    n = len(pick)
    win = (pick['unit'] > 0).mean() if n else 0.0
    roi = pick['unit'].mean() if n else 0.0
    return roi, n, win


def classify_league_stability(df_league):
    """
    Classifica liga em TIER_1 / TIER_2 / TIER_3 (estabilidade).
    """
    n = len(df_league)
    if n == 0:
        return "TIER_3"

    var_goals = df_league[['Goals_H_FT', 'Goals_A_FT']].fillna(0).values.var()
    margin = (df_league['Goals_H_FT'] - df_league['Goals_A_FT']).abs().mean()
    blowouts = ((df_league['Goals_H_FT'] - df_league['Goals_A_FT']).abs() >= 3).mean()

    instability = (
        (var_goals * 0.5) +
        (margin * 0.3) +
        (blowouts * 2.0) +
        (1.0 / max(n, 1)) * 10
    )

    if instability < 2.0:
        return "TIER_1"
    elif instability < 4.0:
        return "TIER_2"
    else:
        return "TIER_3"


def find_league_thresholds(history: pd.DataFrame, min_bets: int = 60):
    """
    history deve conter: League, VG_HOME, VG_AWAY, Goals_H_FT, Goals_A_FT, Asian_Line_Decimal
    Usa TIER + blending com thresholds globais.
    """
    leagues = sorted(history['League'].dropna().unique().tolist())

    # TIER das ligas
    league_stability = {}
    for lg in leagues:
        df_lg = history[history['League'] == lg]
        league_stability[lg] = classify_league_stability(df_lg)

    TIER_THRESHOLDS = {
        "TIER_1": {
            "HOME": 0.10,
            "AWAY": 0.10,
            "HOME_STRONG": 0.25,
            "AWAY_STRONG": 0.25,
        },
        "TIER_2": {
            "HOME": 0.15,
            "AWAY": 0.15,
            "HOME_STRONG": 0.30,
            "AWAY_STRONG": 0.30,
        },
        "TIER_3": {
            "HOME": 0.22,
            "AWAY": 0.22,
            "HOME_STRONG": 0.35,
            "AWAY_STRONG": 0.35,
        },
    }

    global_tier = "TIER_2"
    global_pack = TIER_THRESHOLDS[global_tier]

    out = {}

    for lg in leagues:
        df_lg = history[history['League'] == lg].copy()
        tier = league_stability.get(lg, "TIER_2")
        base = TIER_THRESHOLDS[tier]
        n_games = len(df_lg)

        if n_games < min_bets:
            alpha = n_games / max(min_bets, 1)

            def blend(local, glob):
                return round(alpha * local + (1 - alpha) * glob, 3)

            out[lg] = {
                "TIER": tier,
                "HOME": blend(base["HOME"], global_pack["HOME"]),
                "AWAY": blend(base["AWAY"], global_pack["AWAY"]),
                "HOME_STRONG": blend(base["HOME_STRONG"], global_pack["HOME_STRONG"]),
                "AWAY_STRONG": blend(base["AWAY_STRONG"], global_pack["AWAY_STRONG"]),
                "N": n_games,
            }
        else:
            out[lg] = {
                "TIER": tier,
                "HOME": base["HOME"],
                "AWAY": base["AWAY"],
                "HOME_STRONG": base["HOME_STRONG"],
                "AWAY_STRONG": base["AWAY_STRONG"],
                "N": n_games,
            }

    out["_GLOBAL"] = {
        "TIER": global_tier,
        "HOME": global_pack["HOME"],
        "AWAY": global_pack["AWAY"],
        "HOME_STRONG": global_pack["HOME_STRONG"],
        "AWAY_STRONG": global_pack["AWAY_STRONG"],
        "N": len(history),
    }
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

    for c in ['home_goal', 'away_goal', 'home_red', 'away_red']:
        df_ft[c] = pd.to_numeric(df_ft[c], errors='coerce').fillna(0).astype(int)

    games_today = games_today.merge(
        df_ft[['Id', 'status', 'home_goal', 'away_goal', 'home_red', 'away_red']],
        on='Id',
        how='left',
        suffixes=('', '_ls')
    )

    mask_ft = games_today['status_ls'] == 'FT'

    games_today.loc[mask_ft, 'Goals_H_Today'] = games_today.loc[mask_ft, 'home_goal']
    games_today.loc[mask_ft, 'Goals_A_Today'] = games_today.loc[mask_ft, 'away_goal']
    games_today.loc[mask_ft, 'Home_Red'] = games_today.loc[mask_ft, 'home_red']
    games_today.loc[mask_ft, 'Away_Red'] = games_today.loc[mask_ft, 'away_red']

    return games_today


# ============================================================
# 🔥 MÓDULO DUAL – Eixo HOME Unificado
# ============================================================
def analisar_value_bets_dual_modelos(games_today: pd.DataFrame, league_thresholds: dict):
    st.markdown("## 💎 Análise DUAL – Eixo HOME Unificado")

    results = []

    for _, row in games_today.iterrows():
        asian_line = float(row.get("Asian_Line_Decimal", 0.0))

        # Predições já combinadas em eixo HOME
        pred_home = float(row.get('Pred_HOME_Combined', 0.0))
        pred_away_home_axis = float(row.get('Pred_AWAY_Combined_HOME_AXIS', 0.0))

        vg_home = pred_home - asian_line
        vg_away = pred_away_home_axis - asian_line

        league = row.get("League")
        thr_pack = league_thresholds.get(league, league_thresholds.get("_GLOBAL", {}))

        thr_home = float(thr_pack.get("HOME", 0.15))
        thr_home_str = float(thr_pack.get("HOME_STRONG", 0.30))
        thr_away = float(thr_pack.get("AWAY", 0.15))
        thr_away_str = float(thr_pack.get("AWAY_STRONG", 0.30))

        thr_home = adjust_threshold_by_line(thr_home, asian_line)
        thr_home_str = adjust_threshold_by_line(thr_home_str, asian_line)
        thr_away = adjust_threshold_by_line(thr_away, asian_line)
        thr_away_str = adjust_threshold_by_line(thr_away_str, asian_line)

        recomendacao = "NO CLEAR EDGE"
        confidence = "LOW"

        # Prioridade HOME fortíssimo
        if vg_home >= thr_home_str:
            recomendacao = "STRONG BET HOME"
            confidence = "HIGH"
        elif vg_home >= thr_home:
            recomendacao = "BET HOME"
            confidence = "MEDIUM"
        elif vg_away >= thr_away_str:
            recomendacao = "STRONG BET AWAY"
            confidence = "HIGH"
        elif vg_away >= thr_away:
            recomendacao = "BET AWAY"
            confidence = "MEDIUM"

        g_h = row.get('Goals_H_Today')
        g_a = row.get('Goals_A_Today')
        h_r = row.get('Home_Red')
        a_r = row.get('Away_Red')

        live_info = ""
        if pd.notna(g_h) and pd.notna(g_a):
            live_info = f"{int(g_h)}-{int(g_a)}"
        if pd.notna(h_r) and int(h_r) > 0:
            live_info += f" 🟥H{int(h_r)}"
        if pd.notna(a_r) and int(a_r) > 0:
            live_info += f" 🟥A{int(a_r)}"

        results.append({
            "League": league,
            "Home": row.get("Home"),
            "Away": row.get("Away"),
            "Asian_Line": row.get("Asian_Line"),
            "Asian_Line_Decimal": asian_line,

            "Pred_HOME": round(pred_home, 3),
            "Pred_AWAY_HOME_AXIS": round(pred_away_home_axis, 3),

            "VG_HOME": round(vg_home, 3),
            "VG_AWAY": round(vg_away, 3),

            "Recomendacao": recomendacao,
            "Confidence": confidence,
            "Live_Score": live_info,
        })

    df = pd.DataFrame(results)
    bets = df[df["Recomendacao"] != "NO CLEAR EDGE"]
    return df, bets


# ============================================================
# 📈 VISUALIZAÇÕES DUAL (eixo HOME unificado)
# ============================================================
def plot_analise_dual_modelos(games_today: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))

    asian = games_today["Asian_Line_Decimal"].tolist()

    pred_home = games_today['Pred_HOME_Combined'].tolist()
    pred_away_home = games_today['Pred_AWAY_Combined_HOME_AXIS'].tolist()

    fair_line_dual = (np.array(pred_home) + np.array(pred_away_home)) / 2.0

    ax.scatter(asian, pred_home, s=70, label="Pred HOME (eixo HOME)", alpha=0.75)
    ax.scatter(asian, pred_away_home, s=70, label="Pred AWAY (eixo HOME)", alpha=0.75)
    ax.scatter(asian, fair_line_dual, s=90, label="Fair Line DUAL", marker="x")

    ax.plot([-1.5, 1.5], [-1.5, 1.5], "k--", alpha=0.3, label="Mercado (y = x)")

    ax.set_title("Predições DUAL – Tudo no eixo HOME")
    ax.set_xlabel("Asian Line (Mercado – HOME)")
    ax.set_ylabel("Handicap Predito (Eixo HOME)")
    ax.grid(alpha=0.3)
    ax.legend()

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
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options) - 1)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)

        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        def classificar_league_tier(league_name: str) -> int:
            if pd.isna(league_name):
                return 3
            name = league_name.lower()
            if any(x in name for x in ['premier', 'la liga', 'serie a', 'bundesliga', 'ligue 1',
                                       'eredivisie', 'primeira liga', 'brasileirão', 'super league',
                                       'mls', 'championship', 'liga pro', 'a-league']):
                return 1
            if any(x in name for x in ['serie b', 'segunda', 'league 1', 'liga ii', 'liga 2',
                                       'division 2', 'bundesliga 2', 'ligue 2', 'j-league', 'k-league',
                                       'superettan', '1st division', 'national league', 'liga nacional']):
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

    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    history = history.dropna(subset=['Asian_Line_Decimal'])
    games_today = games_today.dropna(subset=['Asian_Line_Decimal'])

    if "Date" in history.columns:
        try:
            selected_date = pd.to_datetime(selected_date_str)
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < selected_date].copy()
            st.info(f"📊 Treinando com {len(history)} jogos anteriores a {selected_date_str}")
        except Exception as e:
            st.warning(f"⚠️ Erro no filtro temporal: {e}")

    games_today = load_and_merge_livescore(games_today, selected_date_str)

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
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
            # HOME
            modelo_home_reg, games_today, scaler_home = treinar_modelo_handicap_regressao_calibrado_v2(history, games_today)
            modelo_home_cls, games_today, le_home = treinar_modelo_handicap_classificacao_calibrado_v2(history, games_today)

            # AWAY
            modelo_away_reg, games_today, scaler_away = treinar_modelo_away_handicap_regressao_calibrado(history, games_today)
            modelo_away_cls, games_today, le_away = treinar_modelo_away_handicap_classificacao_calibrado(history, games_today)

            # ===== COMBINAÇÃO DINÂMICA PARA OS JOGOS DO DIA =====
            # HOME
            reg_home_today = games_today['Handicap_Predito_Regressao_Calibrado'].values
            cls_home_today = games_today['Handicap_Predito_Classificacao_Calibrado'].values
            conf_home_today = games_today['Confianca_Categoria_Calibrado'].values

            pred_home_combined = combinar_reg_class(reg_home_today, cls_home_today, conf_home_today)
            games_today['Pred_HOME_Combined'] = pred_home_combined

            # AWAY (nativo e eixo HOME)
            reg_away_today = games_today['Handicap_AWAY_Predito_Regressao_Calibrado'].values
            cls_away_today = games_today['Handicap_AWAY_Predito_Classificacao_Calibrado'].values
            conf_away_today = games_today['Confianca_Categoria_AWAY_Calibrado'].values

            pred_away_combined_native = combinar_reg_class(reg_away_today, cls_away_today, conf_away_today)
            pred_away_combined_home_axis = -pred_away_combined_native

            games_today['Pred_AWAY_Combined_NATIVE'] = pred_away_combined_native
            games_today['Pred_AWAY_Combined_HOME_AXIS'] = pred_away_combined_home_axis

            # ===== HISTÓRICO PARA THRESHOLDS (APENAS COM FT) =====
            hist_for_pred = history.dropna(subset=['Goals_H_FT', 'Goals_A_FT']).copy()
            if hist_for_pred.empty:
                st.error("❌ Histórico sem FT para calibrar thresholds.")
                return

            features_3d_common = [
                'Quadrant_Dist_3D', 'Quadrant_Separation_3D', 'Vector_Sign',
                'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT', 'Cluster3D_Label'
            ]

            # garantir colunas
            for f in features_3d_common:
                if f not in hist_for_pred.columns:
                    hist_for_pred[f] = 0.0

            X_hist = hist_for_pred[features_3d_common].fillna(0.0)

            # HOME histórico
            Xs_home_hist = scaler_home.transform(X_hist)
            reg_home_hist = modelo_home_reg.predict(Xs_home_hist)

            proba_home_hist = modelo_home_cls.predict_proba(X_hist)
            cls_home_idx_hist = np.argmax(proba_home_hist, axis=1)
            labels_home_hist = le_home.inverse_transform(cls_home_idx_hist)

            map_cat_home_num = {
                'MODERATE_HOME': -1.0,
                'LIGHT_HOME': -0.5,
                'NEUTRAL': 0.0,
                'LIGHT_AWAY': 0.5,
                'MODERATE_AWAY': 1.0,
            }
            cls_home_num_hist = np.array([map_cat_home_num.get(lbl, 0.0) for lbl in labels_home_hist])
            conf_home_hist = np.max(proba_home_hist, axis=1)

            pred_home_hist_final = combinar_reg_class(reg_home_hist, cls_home_num_hist, conf_home_hist)

            hist_for_pred['Pred_HOME'] = pred_home_hist_final

            # AWAY histórico (nativo e eixo HOME)
            Xs_away_hist = scaler_away.transform(X_hist)
            reg_away_hist = modelo_away_reg.predict(Xs_away_hist)

            proba_away_hist = modelo_away_cls.predict_proba(X_hist)
            cls_away_idx_hist = np.argmax(proba_away_hist, axis=1)
            labels_away_hist = le_away.inverse_transform(cls_away_idx_hist)

            map_cat_away_num = {
                'MODERATE_AWAY': -1.0,
                'LIGHT_AWAY': -0.5,
                'NEUTRAL': 0.0,
                'LIGHT_HOME': 0.5,
                'MODERATE_HOME': 1.0,
            }
            cls_away_num_hist = np.array([map_cat_away_num.get(lbl, 0.0) for lbl in labels_away_hist])
            conf_away_hist = np.max(proba_away_hist, axis=1)

            pred_away_hist_native = combinar_reg_class(reg_away_hist, cls_away_num_hist, conf_away_hist)
            pred_away_hist_home_axis = -pred_away_hist_native

            hist_for_pred['Pred_AWAY_NATIVE'] = pred_away_hist_native
            hist_for_pred['Pred_AWAY_HOME_AXIS'] = pred_away_hist_home_axis

            asian_hist = hist_for_pred['Asian_Line_Decimal']

            hist_for_pred['VG_HOME'] = hist_for_pred['Pred_HOME'] - asian_hist
            hist_for_pred['VG_AWAY'] = hist_for_pred['Pred_AWAY_HOME_AXIS'] - asian_hist

            league_thresholds = find_league_thresholds(hist_for_pred, min_bets=60)

            # ===== ANALISAR VALUE BETS COM O DUAL =====
            df_value_bets_dual, bets_validos_dual = analisar_value_bets_dual_modelos(games_today, league_thresholds)

            st.markdown("## 📊 Resultados - Análise DUAL")
            if bets_validos_dual.empty:
                st.warning("⚠️ Nenhuma recomendação de value bet encontrada")
            else:
                st.dataframe(bets_validos_dual, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("🏠 HOME Bets", int(bets_validos_dual['Recomendacao'].str.contains('HOME').sum()))
                with c2:
                    st.metric("✈️ AWAY Bets", int(bets_validos_dual['Recomendacao'].str.contains('AWAY').sum()))
                with c3:
                    st.metric("🎯 Strong Bets", int(bets_validos_dual['Confidence'].eq('HIGH').sum()))
                with c4:
                    st.metric("📊 Total Recomendações", len(bets_validos_dual))

            st.pyplot(plot_analise_dual_modelos(games_today))

            st.markdown("### 🔎 Debug: Value Gaps reais do dia")

            df_debug = games_today.copy()
            df_debug['VG_HOME'] = df_debug['Pred_HOME_Combined'] - df_debug['Asian_Line_Decimal']
            df_debug['VG_AWAY'] = df_debug['Pred_AWAY_Combined_HOME_AXIS'] - df_debug['Asian_Line_Decimal']

            st.dataframe(df_debug[['Home', 'Away', 'Asian_Line', 'Asian_Line_Decimal',
                                   'Pred_HOME_Combined', 'Pred_AWAY_Combined_HOME_AXIS',
                                   'VG_HOME', 'VG_AWAY']])

            st.success("✅ Análise DUAL concluída com sucesso!")
            st.balloons()


if __name__ == "__main__":
    main_calibrado()
