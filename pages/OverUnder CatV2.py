# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import math

from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
import plotly.graph_objects as go

# ==========================================================
# CONFIGURAÇÕES BÁSICAS
# ==========================================================
st.set_page_config(page_title="Over/Under ML Inteligente", layout="wide")
st.title("🎯 ML Avançado - Over/Under na Linha do Mercado")

PAGE_PREFIX = "OU_ML"
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "coppa", "copas", "uefa", "sudamericana", "copa", "trophy"]

# ==========================================================
# HELPERS BÁSICOS (LOAD + FILTER)
# ==========================================================
def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def load_all_games(folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        return pd.DataFrame()
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(folder, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def load_and_filter_history(selected_date_str: str) -> pd.DataFrame:
    """Carrega histórico APENAS com jogos anteriores à data selecionada."""
    st.info("📊 Carregando histórico de jogos anteriores...")
    history = filter_leagues(load_all_games(GAMES_FOLDER))

    if history.empty:
        st.warning("⚠️ Histórico vazio")
        return history

    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        selected_date = pd.to_datetime(selected_date_str)
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📅 Histórico filtrado: {len(history)} jogos anteriores a {selected_date_str}")
    else:
        st.warning("⚠️ Coluna Date não encontrada no histórico.")

    return history

# ==========================================================
# OVER/UNDER LINE → DECIMAL (2, 2.25, 2.5, 2.75, 3...)
# ==========================================================
def convert_ou_line_to_decimal(line_str):
    if pd.isna(line_str):
        return np.nan
    s = str(line_str).strip()
    if s == "":
        return np.nan

    # Split tipo 2/2.5, 2.5/3
    if "/" in s:
        try:
            parts = [float(x) for x in s.split("/")]
            return sum(parts) / len(parts)
        except Exception:
            return np.nan
    else:
        try:
            return float(s)
        except Exception:
            return np.nan

def adicionar_ou_line_decimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "OverLine" in df.columns:
        df["OU_Line_Dec"] = df["OverLine"].apply(convert_ou_line_to_decimal)
    else:
        df["OU_Line_Dec"] = np.nan
    return df

# ==========================================================
# TARGET O/U GENÉRICO (REMOVE PUSH)
# ==========================================================
def create_ou_target_generico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria target para Over/Under baseado na linha real OverLine (convertida em OU_Line_Dec).
    - Usa Goals_H_FT + Goals_A_FT
    - Usa OU_Line_Dec (2.0, 2.25, 2.5, 2.75, 3.0, ...)
    - Remove PUSH (Total_Goals == OU_Line_Dec)
    """
    df = df.copy()
    df = adicionar_ou_line_decimal(df)

    # Garante gols FT
    if "Goals_H_FT" not in df.columns or "Goals_A_FT" not in df.columns:
        st.error("❌ Faltam colunas Goals_H_FT / Goals_A_FT para criar target O/U.")
        return pd.DataFrame()

    df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT", "OU_Line_Dec"]).copy()

    df["Total_Goals"] = df["Goals_H_FT"] + df["Goals_A_FT"]
    df["OU_Margin"] = df["Total_Goals"] - df["OU_Line_Dec"]

    # 1.0 = Over ganhou; 0.5 = PUSH; 0.0 = Under ganhou
    def classificar_ou_result(margin):
        if pd.isna(margin):
            return np.nan
        if abs(margin) < 1e-9:
            return 0.5   # PUSH
        elif margin > 0:
            return 1.0   # Over ganhou
        else:
            return 0.0   # Under ganhou

    df["OU_Result"] = df["OU_Margin"].apply(classificar_ou_result)

    total = len(df)
    df_treino = df[df["OU_Result"] != 0.5].copy()  # remove PUSH
    clean = len(df_treino)

    df_treino["Target_OU_Over"] = (df_treino["OU_Result"] > 0.5).astype(int)
    df_treino["Target_OU_Under"] = (df_treino["OU_Result"] < 0.5).astype(int)

    over_rate = df_treino["Target_OU_Over"].mean() if clean > 0 else 0.0
    under_rate = 1 - over_rate if clean > 0 else 0.0
    push_pct = (total - clean) / total if total > 0 else 0.0

    st.info(f"🎯 Total analisado: {total} jogos")
    st.info(f"🗑️ Removidos por PUSH na linha: {total-clean} jogos ({push_pct:.1%})")
    st.info(f"📊 Treino com: {clean} jogos válidos (sem PUSH)")
    st.info(f"⚽ Over (cobriu linha): {over_rate:.1%}")
    st.info(f"🛡️ Under (não cobriu linha): {under_rate:.1%}")

    return df_treino

# ==========================================================
# FUNÇÃO DE LIMPEZA DE FEATURES
# ==========================================================
def clean_features_for_training(X: pd.DataFrame) -> pd.DataFrame:
    X_clean = X.copy()
    if isinstance(X_clean, np.ndarray):
        X_clean = pd.DataFrame(X_clean)

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(0)

    for col in X_clean.columns:
        if np.issubdtype(X_clean[col].dtype, np.number):
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            if pd.isna(IQR) or IQR == 0:
                continue
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)

    X_clean = X_clean.fillna(0)
    X_clean = X_clean.replace([np.inf, -np.inf], 0)

    return X_clean



# ==========================================================
# LIGA: TAXA MÉDIA DE OVER POR LINHA (GENÉRICO)
# ==========================================================
def adicionar_liga_over_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "League" not in df.columns or "Target_OU_Over" not in df.columns:
        df["Liga_OverRate"] = 0.5
        return df

    df["Liga_OverRate"] = df.groupby("League")["Target_OU_Over"].transform("mean")
    df["Liga_OverRate"] = df["Liga_OverRate"].fillna(0.5)
    return df

# ==========================================================
# OverScore_DIFF (Home - Away)
# ==========================================================
def adicionar_overscore_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "OverScore_Home" in df.columns and "OverScore_Away" in df.columns:
        df["OverScore_Diff"] = df["OverScore_Home"] - df["OverScore_Away"]
    else:
        df["OverScore_Diff"] = 0.0
    return df

# ==========================================================
# FEATURE SET FINAL
# ==========================================================
def create_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # OverScore
    df = adicionar_overscore_diff(df)

    # Liga Rate (se já tiver Target_OU_Over)
    if "Target_OU_Over" in df.columns:
        df = adicionar_liga_over_rate(df)

    # Lista de possíveis features (serão filtradas se não existirem)
    basic_features = [
        "Liga_OverRate",
        "OverScore_Diff",
    ]

    momentum_features = [
        "M_H", "M_A", "MT_H", "MT_A",
        "M_Total", "Momentum_Advantage"
    ]

    # Se M_H / M_A existirem, gera derivados
    if "M_H" in df.columns and "M_A" in df.columns:
        df["M_Total"] = df["M_H"] + df["M_A"]
        df["Momentum_Advantage"] = df["M_H"] - df["M_A"]

    vector_features = [
        "Quadrant_Dist_3D", "Magnitude_3D",
        "Quadrant_Sin_XY", "Quadrant_Cos_XY",
        "Quadrant_Sin_XZ", "Quadrant_Cos_XZ",
        "Quadrant_Sin_YZ", "Quadrant_Cos_YZ",
    ]

    wg_features = [
        "WG_Home_Team_Last", "WG_Away_Team_Last", "WG_Diff",
        "WG_Def_Home_Team_Last", "WG_Def_Away_Team_Last", "WG_Def_Diff",
        "WG_Total_Home_Team_Last", "WG_Total_Away_Team_Last",
        "WG_Net_Home_Team_Last", "WG_Net_Away_Team_Last",
        "WG_Net_Diff",
    ]

    ges_features = [
        "GES_Of_H_Roll", "GES_Of_A_Roll", "GES_Of_Diff",
        "GES_Def_H_Roll", "GES_Def_A_Roll", "GES_Def_Diff",
        "GES_Total_Diff",
    ]

    odds_features = [
        "Odd_Over25", "Odd_Under25"
    ]

    all_features = (
        basic_features
        + momentum_features
        + vector_features
        + wg_features
        + ges_features
        + odds_features
    )

    available_features = [f for f in all_features if f in df.columns]

    st.info(f"📋 Features disponíveis para ML: {len(available_features)}/{len(all_features)}")

    if not available_features:
        st.error("❌ Nenhuma feature disponível para ML.")
        return pd.DataFrame()

    X = df[available_features].copy().fillna(0)
    return X

# ==========================================================
# TREINO CATBOOST
# ==========================================================
def train_catboost_model(X: pd.DataFrame, y: pd.Series, feature_names):
    st.info("🤖 Treinando modelo CatBoost para Over (cobrir linha)...")

    X_clean = clean_features_for_training(X)
    y_clean = y.copy()

    if hasattr(y_clean, "isna") and y_clean.isna().any():
        st.warning(f"⚠️ {y_clean.isna().sum()} NaNs no target - removendo")
        valid_mask = ~y_clean.isna()
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]

    # class_weights com base no balanceamento
    pos_rate = y_clean.mean()
    neg_rate = 1 - pos_rate
    if pos_rate > 0 and neg_rate > 0:
        class_weights = [neg_rate, pos_rate]
    else:
        class_weights = [1.0, 1.0]

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=400,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        l2_leaf_reg=3.0,
        verbose=False,
        class_weights=class_weights
    )

    try:
        scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring="accuracy")
        st.write(f"📊 Validação Cruzada (Accuracy): {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        if scores.mean() < 0.55:
            st.warning("⚠️ Modelo abaixo do esperado - revisar dados/features.")
        elif scores.mean() > 0.62:
            st.success("🎯 Modelo com boa performance!")
    except Exception as e:
        st.warning(f"⚠️ Validação cruzada falhou: {e}")

    model.fit(X_clean, y_clean)

    # Importância das features
    importances = pd.Series(model.get_feature_importance(), index=feature_names).sort_values(ascending=False)
    st.write("🔍 **Top Features mais importantes:**")
    st.dataframe(importances.head(15).to_frame("Importância"))

    return model

# ==========================================================
# MIN CONF POR ODD + EV SELEÇÃO OVER/UNDER
# ==========================================================
def min_conf_by_odd(odd):
    if pd.isna(odd):
        return 0.60
    if odd <= 1.80:
        return 0.61
    if odd <= 2.05:
        return 0.57
    if odd <= 2.30:
        return 0.54
    return 0.50

def decidir_over_under_com_ev(games_today: pd.DataFrame) -> pd.DataFrame:
    df = games_today.copy()

    if "Prob_OU_Over" not in df.columns or "Prob_OU_Under" not in df.columns:
        return df

    # Garante odds
    for col in ["Odd_Over25", "Odd_Under25"]:
        if col not in df.columns:
            df[col] = np.nan

    df["EV_Over"] = df["Prob_OU_Over"] * df["Odd_Over25"] - 1
    df["EV_Under"] = df["Prob_OU_Under"] * df["Odd_Under25"] - 1

    bet_side = []
    bet_conf = []
    bet_ev = []
    bet_approved = []

    for _, r in df.iterrows():
        odd_over = r["Odd_Over25"]
        odd_under = r["Odd_Under25"]
        p_over = r["Prob_OU_Over"]
        p_under = r["Prob_OU_Under"]
        ev_over = r["EV_Over"]
        ev_under = r["EV_Under"]

        # Se odds inválidas → no bet
        if (not isinstance(odd_over, (int, float))) or odd_over <= 1.01 \
           or (not isinstance(odd_under, (int, float))) or odd_under <= 1.01:
            bet_side.append("NONE")
            bet_conf.append(0.0)
            bet_ev.append(0.0)
            bet_approved.append(False)
            continue

        # Escolhe maior EV
        if ev_over >= ev_under:
            side = "OVER"
            conf = p_over
            ev = ev_over
            min_conf = min_conf_by_odd(odd_over)
        else:
            side = "UNDER"
            conf = p_under
            ev = ev_under
            min_conf = min_conf_by_odd(odd_under)

        approved = (ev > 0) and (conf >= min_conf)

        bet_side.append(side)
        bet_conf.append(conf)
        bet_ev.append(ev)
        bet_approved.append(approved)

    df["Bet_Side"] = bet_side          # OVER / UNDER / NONE
    df["Bet_Confidence"] = bet_conf    # prob do lado escolhido
    df["Bet_EV"] = bet_ev              # EV do lado escolhido
    df["Bet_Approved"] = bet_approved  # bool

    return df



# ==========================================================
# CARREGAR GAMESDAY + HISTORY
# ==========================================================
st.info("📂 Carregando arquivos de jogos...")

if not os.path.exists(GAMES_FOLDER):
    st.error(f"❌ Pasta '{GAMES_FOLDER}' não encontrada.")
    st.stop()

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.error("❌ Nenhum CSV encontrado em GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Selecione o arquivo de Matchday:", options, index=len(options)-1)

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

history = load_and_filter_history(selected_date_str)

# ==========================================================
# TREINO CATBOOST (HISTÓRICO)
# ==========================================================
model_ou = None

if not history.empty:
    # Cria target genérico O/U
    history_target = create_ou_target_generico(history)

    if not history_target.empty and "Target_OU_Over" in history_target.columns and len(history_target) > 50:
        X_hist = create_robust_features(history_target)
        if not X_hist.empty:
            y_over = history_target["Target_OU_Over"]
            model_ou = train_catboost_model(X_hist, y_over, X_hist.columns)
        else:
            st.error("❌ X_hist vazio após create_robust_features.")
    else:
        st.warning("⚠️ Histórico insuficiente para treinar o modelo O/U (linha OverLine).")
else:
    st.warning("⚠️ Histórico vazio, não foi possível treinar o modelo O/U.")

# ==========================================================
# PREDIÇÃO NOS JOGOS DE HOJE
# ==========================================================
if model_ou is not None and not games_today.empty:
    st.markdown("## 📡 Previsões O/U para os jogos de hoje na linha OverLine")

    games_today = adicionar_ou_line_decimal(games_today)
    X_today = create_robust_features(games_today)

    if not X_today.empty:
        # Garante mesmas features do treino
        required_features = model_ou.feature_names_
        X_today = X_today.reindex(columns=required_features, fill_value=0)

        proba_over = model_ou.predict_proba(X_today)[:, 1]
        proba_under = 1 - proba_over

        games_today["Prob_OU_Over"] = proba_over
        games_today["Prob_OU_Under"] = proba_under

        # Decisão EV Over x Under
        games_today = decidir_over_under_com_ev(games_today)

        # ==================================================
        # TABELA COMPLETA
        # ==================================================
        st.markdown("### 🏆 Ranking de jogos por EV do lado escolhido")

        cols_rank = [
            "League", "Home", "Away",
            "OverLine", "OU_Line_Dec",
            "Odd_Over25", "Odd_Under25",
            "Prob_OU_Over", "Prob_OU_Under",
            "Bet_Side", "Bet_Confidence", "Bet_EV", "Bet_Approved",
        ]
        cols_rank = [c for c in cols_rank if c in games_today.columns]

        ranking = games_today.sort_values("Bet_EV", ascending=False).copy()
        st.dataframe(ranking[cols_rank].head(50))

        # ==================================================
        # SINAIS APROVADOS
        # ==================================================
        aprovados = ranking[ranking["Bet_Approved"]].copy()
        if not aprovados.empty:
            st.markdown("### ✅ Sinais aprovados (EV > 0 e Conf ≥ min)")

            cols_aprov = [
                "League", "Home", "Away",
                "OverLine", "Odd_Over25", "Odd_Under25",
                "Prob_OU_Over", "Prob_OU_Under",
                "Bet_Side", "Bet_Confidence", "Bet_EV",
            ]
            cols_aprov = [c for c in cols_aprov if c in aprovados.columns]

            st.dataframe(aprovados[cols_aprov].head(50))
        else:
            st.info("Nenhum sinal aprovado hoje (com base em EV e confiança mínima).")
    else:
        st.error("❌ X_today vazio após create_robust_features para jogos de hoje.")
else:
    if games_today.empty:
        st.warning("⚠️ Nenhum jogo em games_today.")
    else:
        st.warning("⚠️ Modelo O/U não treinado - sem previsões para hoje.")


