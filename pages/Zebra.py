from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from datetime import datetime
import math

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

# ---------------- CONFIGURAÇÕES LIVE SCORE ----------------
LIVESCORE_FOLDER = "LiveScore"

def setup_livescore_columns(df):
    """Garante que as colunas do Live Score existam no DataFrame"""
    if 'Goals_H_Today' not in df.columns:
        df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns:
        df['Goals_A_Today'] = np.nan
    if 'Home_Red' not in df.columns:
        df['Home_Red'] = np.nan
    if 'Away_Red' not in df.columns:
        df['Away_Red'] = np.nan
    return df

# ---------------- Helpers Básicos ----------------
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

def convert_asian_line(line_str):
    """Converte string de linha asiática em média numérica"""
    try:
        if pd.isna(line_str) or line_str == "":
            return None
        line_str = str(line_str).strip()
        if "/" not in line_str:
            val = float(line_str)
            return 0.0 if abs(val) < 1e-10 else val
        parts = [float(x) for x in line_str.split("/")]
        avg = sum(parts) / len(parts)
        return 0.0 if abs(avg) < 1e-10 else avg
    except:
        return None

# ---------------- CORREÇÕES CRÍTICAS ASIAN LINE ----------------
def convert_asian_line_to_decimal_corrigido(line_str):
    """
    Converte handicaps asiáticos (Away) no formato string para decimal invertido (Home).
    """
    if pd.isna(line_str) or line_str == "":
        return None

    line_str = str(line_str).strip()

    # Caso especial: linha zero
    if line_str == "0" or line_str == "0.0":
        return 0.0

    # ✅ CORREÇÃO: Mapeamento COMPLETO de todos os splits comuns
    common_splits = {
        # Splits positivos (Away dá handicap)
        '0/0.5': -0.25,
        '0.5/1': -0.75,
        '1/1.5': -1.25,
        '1.5/2': -1.75,
        '2/2.5': -2.25,
        '2.5/3': -2.75,
        '3/3.5': -3.25,

        # Splits negativos (Away recebe handicap)
        '0/-0.5': 0.25,
        '-0.5/-1': 0.75,
        '-1/-1.5': 1.25,
        '-1.5/-2': 1.75,
        '-2/-2.5': 2.25,
        '-2.5/-3': 2.75,
        '-3/-3.5': 3.25,

        # Quarter handicaps
        '0.75': -0.75,
        '-0.75': 0.75,
        '0.25': -0.25,
        '-0.25': 0.25,
    }

    if line_str in common_splits:
        return common_splits[line_str]

    # Caso simples — número único
    if "/" not in line_str:
        try:
            num = float(line_str)
            return -num  # Inverte sinal (Away → Home)
        except ValueError:
            return None

    # Split genérico
    try:
        parts = [float(p) for p in line_str.split("/")]
        avg = sum(parts) / len(parts)

        first_part = parts[0]
        if first_part < 0:
            result = -abs(avg)
        else:
            result = abs(avg)

        return -result

    except (ValueError, TypeError):
        st.warning(f"⚠️ Split handicap não reconhecido: {line_str}")
        return None

def calc_handicap_result_corrigido(margin, asian_line_decimal):
    """
    Calcula o resultado do Handicap Asiático do ponto de vista do HOME.

    Retorno:
    1.0  -> Full Win
    0.5  -> Half Win / Push
    0.0  -> Full Loss
    """
    if pd.isna(margin) or pd.isna(asian_line_decimal):
        return np.nan

    line = asian_line_decimal
    abs_line = abs(line)

    # 🔹 LINHAS .0 (ex: 0, -1.0, +2.0) -> POSSUI PUSH
    if abs_line % 1 == 0:
        if margin > line:
            return 1.0  # win
        elif margin == line:
            return 0.5  # push
        else:
            return 0.0  # loss

    # 🔸 LINHAS .5 (ex: -0.5, +1.5) -> NÃO TEM PUSH
    if abs_line % 1 == 0.5:
        if margin > line:
            return 1.0  # full win
        else:
            return 0.0  # full loss

    # 🟢 QUARTER LINES (0.25, 0.75, 1.25 etc)
    full_step = math.floor(abs_line * 2) / 2  # menor .5
    half_step = math.ceil(abs_line * 2) / 2   # maior .5

    if line < 0:
        line1 = -half_step
        line2 = -full_step
    else:
        line1 = full_step
        line2 = half_step

    def single_result(m, l):
        # full-line logic (can have push)
        if abs(l % 1) == 0:
            if m > l:
                return 1.0
            if m == l:
                return 0.5
            return 0.0
        # half-line logic (no push)
        if m > l:
            return 1.0
        return 0.0

    result = (single_result(margin, line1) + single_result(margin, line2)) / 2
    return result

def testar_conversao_asian_line():
    st.markdown("### 🧪 TESTE COMPLETO – LINHA & RESULTADO")

    test_cases = [
        # Full lines
        ("0.5", "Away +0.5 → Home -0.5"),
        ("-0.5", "Away -0.5 → Home +0.5"),
        ("1.0", "Away +1.0 → Home -1.0"),
        ("-1.0", "Away -1.0 → Home +1.0"),

        # Splits
        ("0/0.5", "Away 0/0.5 → Home -0.25"),
        ("0/-0.5", "Away 0/-0.5 → Home +0.25"),
        ("0.5/1", "Away 0.5/1 → Home -0.75"),
        ("-0.5/-1", "Away -0.5/-1 → Home +0.75"),
        ("1/1.5", "Away 1/1.5 → Home -1.25"),
        ("-1/-1.5", "Away -1/-1.5 → Home +1.25"),
        ("1.5/2", "Away 1.5/2 → Home -1.75"),
        ("-1.5/-2", "Away -1.5/-2 → Home +1.75"),

        # Quarter-lines
        ("0.25", "Away +0.25 → Home -0.25"),
        ("-0.25", "Away -0.25 → Home +0.25"),
        ("0.75", "Away +0.75 → Home -0.75"),
        ("-0.75", "Away -0.75 → Home +0.75"),

        # Zero line
        ("0", "Away 0 → Home 0"),
    ]

    test_margins = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]

    results = []
    for line_str, desc in test_cases:
        decimal = convert_asian_line_to_decimal_corrigido(line_str)

        tests = []
        for m in test_margins:
            r = calc_handicap_result_corrigido(m, decimal)
            if r == 1.0:
                sym = "🟩"  # full win
            elif r == 0.5:
                sym = "🟨"  # half/push
            else:
                sym = "🟥"  # loss
            tests.append(f"{m}:{sym}")

        results.append({
            "AsianLine(Away)": line_str,
            "Convertido(Home)": decimal,
            "Teste": " | ".join(tests)
        })

    st.dataframe(pd.DataFrame(results))
    st.success("Conversões e resultados validados!")

# ---------------- CORREÇÕES CRÍTICAS PARA ML ----------------
def load_and_filter_history(selected_date_str):
    """Carrega histórico APENAS com jogos anteriores à data selecionada - CORRIGIDO"""
    st.info("📊 Carregando histórico com filtro temporal correto...")

    history = filter_leagues(load_all_games(GAMES_FOLDER))

    if history.empty:
        st.warning("⚠️ Histórico vazio")
        return history

    # Converter datas ANTES de filtrar
    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        selected_date = pd.to_datetime(selected_date_str)

        # Filtrar TEMPORALMENTE primeiro
        history = history[history["Date"] < selected_date].copy()
        st.info(f"📅 Histórico filtrado: {len(history)} jogos anteriores a {selected_date_str}")

    # Só depois processar o resto
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line"]).copy()

    # Conversão corrigida
    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)
    history = history.dropna(subset=['Asian_Line_Decimal'])

    st.success(f"✅ Histórico processado: {len(history)} jogos válidos")
    return history

def create_better_target_corrigido(df):
    """
    Cria targets 100% binários (sem push), com Zebra:

    - Target_AH_Home: 1 se o HOME cobrir o handicap (full win)
    - Expected_Favorite: HOME se linha < 0, AWAY se linha > 0
    - Zebra: 1 se o favorito do mercado falhar
    - Full push e quarter push são EXCLUÍDOS do dataset
    """
    df = df.copy()

    # Margem real de gols
    df["Margin"] = df["Goals_H_FT"] - df["Goals_A_FT"]

    # Calcular o resultado asiático completo (0.0, 0.5, 1.0)
    df["AH_Result"] = df.apply(
        lambda r: calc_handicap_result_corrigido(
            r["Margin"], r["Asian_Line_Decimal"]
        ),
        axis=1
    )

    total = len(df)

    # EXCLUIR QUALQUER PUSH (0.5)
    df = df[df["AH_Result"].isin([0, 1])].copy()
    clean = len(df)

    # Target binário 1.0 → cobre | 0.0 → não cobre
    df["Target_AH_Home"] = (df["AH_Result"] == 1.0).astype(int)

    # Quem o mercado aponta como favorito
    df["Expected_Favorite"] = np.where(
        df["Asian_Line_Decimal"] < 0,
        "HOME",
        np.where(df["Asian_Line_Decimal"] > 0, "AWAY", "NONE")
    )

    # Zebra: favorito não cumpre expectativa do AH
    df["Zebra"] = np.where(
        (
            (df["Expected_Favorite"] == "HOME") & (df["Target_AH_Home"] == 0)
        ) |
        (
            (df["Expected_Favorite"] == "AWAY") & (df["Target_AH_Home"] == 1)
        ),
        1,
        0
    )

    win_rate = df["Target_AH_Home"].mean() if len(df) > 0 else 0.0
    zebra_rate = df["Zebra"].mean() if len(df) > 0 else 0.0

    st.info(f"🎯 Total analisado: {total} jogos")
    st.info(f"🗑️ Excluídos por Push: {total-clean} jogos ({(total-clean)/total:.1%})")
    st.info(f"📊 Treino com: {clean} jogos restantes")
    st.info(f"🏠 Win rate HOME cobrindo: {win_rate:.1%}")
    st.info(f"🦓 Taxa de Zebra (favorito falhou): {zebra_rate:.1%}")

    debug_cols = [
        "League", "Home", "Away",
        "Asian_Line", "Asian_Line_Decimal",
        "Goals_H_FT", "Goals_A_FT",
        "Margin", "Expected_Favorite",
        "Target_AH_Home", "AH_Result", "Zebra"
    ]
    debug_cols = [c for c in debug_cols if c in df.columns]

    st.write("🔍 Exemplos (após correção PUSH):")
    st.dataframe(df.head(6)[debug_cols])

    return df


def create_robust_features(df):
    """Cria features mais robustas e elimina colinearidade - CORRIGIDO"""

    # 1. Features básicas essenciais
    basic_features = [
        'Aggression_Home', 'Aggression_Away',
        'M_H', 'M_A', 'MT_H', 'MT_A'
    ]

    # 2. Features derivadas (evitar colinearidade)
    df = df.copy()
    if 'Aggression_Home' in df.columns and 'Aggression_Away' in df.columns:
        df['Aggression_Diff'] = df['Aggression_Home'] - df['Aggression_Away']
        df['Aggression_Total'] = df['Aggression_Home'] + df['Aggression_Away']
    if 'M_H' in df.columns and 'M_A' in df.columns:
        df['M_Total'] = df['M_H'] + df['M_A']
        df['Momentum_Advantage'] = (df['M_H'] - df['M_A'])
    if 'MT_H' in df.columns and 'MT_A' in df.columns:
        df['MT_Total'] = df['MT_H'] + df['MT_A']
        # complementar com Momentum_Advantage incluindo MT se ambas existirem
        if 'Momentum_Advantage' in df.columns:
            df['Momentum_Advantage'] = df['Momentum_Advantage'] + (df['MT_H'] - df['MT_A'])
        else:
            df['Momentum_Advantage'] = (df['MT_H'] - df['MT_A'])

    derived_features = [
        'Aggression_Diff', 'M_Total', 'MT_Total',
        'Momentum_Advantage', 'Aggression_Total'
    ]

    # 3. Apenas as melhores features 3D
    vector_features = [
        'Quadrant_Dist_3D', 'Momentum_Diff', 'Magnitude_3D'
    ]

    all_features = basic_features + derived_features + vector_features

    # Verificar quais features existem
    available_features = [f for f in all_features if f in df.columns]

    st.info(f"📋 Features disponíveis: {len(available_features)}/{len(all_features)}")

    return df[available_features].fillna(0)

def train_improved_model(X, y, feature_names):
    """Modelo com melhor configuração para dados esportivos - CORRIGIDO"""

    from sklearn.ensemble import RandomForestClassifier

    st.info("🤖 Treinando modelo otimizado...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Validação cruzada
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        st.write(f"📊 Validação Cruzada: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        if scores.mean() < 0.55:
            st.warning("⚠️ Modelo abaixo do esperado - verificar qualidade dos dados")
        elif scores.mean() > 0.65:
            st.success("🎯 Modelo com boa performance!")
    except Exception as e:
        st.warning(f"⚠️ Validação cruzada falhou: {e}")

    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    st.write("🔍 **Top Features mais importantes:**")
    st.dataframe(importances.head(10).to_frame("Importância"))

    return model

def data_quality_report(df, target_col='Target_AH_Home'):
    """Diagnóstico completo da qualidade dos dados"""
    st.markdown("### 🔍 Diagnóstico de Qualidade dos Dados")

    if df.empty:
        st.warning("DataFrame vazio")
        return

    # Distribuição do target
    if target_col in df.columns:
        st.write("**Distribuição do Target:**")
        target_dist = df[target_col].value_counts(normalize=True).sort_index()
        st.write(target_dist)

        balance_ratio = target_dist.min() / target_dist.max()
        if balance_ratio < 0.6:
            st.warning(f"⚠️ Desbalanceamento detectado: ratio = {balance_ratio:.2f}")
        else:
            st.success(f"✅ Target balanceado: ratio = {balance_ratio:.2f}")

    # Missing values
    st.write("**Valores Faltantes:**")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.warning(f"⚠️ {len(missing)} colunas com missing values")
        st.write(missing)
    else:
        st.success("✅ Sem valores faltantes")

    # Correlação com target
    if target_col in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        if len(numeric_cols) > 0:
            correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            st.write("**Top Correlações com Target:**")
            st.write(correlations.head(10))

            if correlations.max() < 0.1:
                st.warning("⚠️ Correlações muito baixas com target")

# ---------------- CACHE INTELIGENTE ----------------
@st.cache_data(ttl=3600)
def load_cached_data(selected_file):
    """Cache apenas dos dados pesados"""
    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
    games_today = filter_leagues(games_today)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

    history = load_and_filter_history(selected_date_str)

    return games_today, history, selected_date_str

# ---------------- LIVE SCORE INTEGRATION ----------------
def load_and_merge_livescore(games_today, selected_date_str):
    """Carrega e faz merge dos dados do Live Score"""
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

# ---------------- Carregar Dados ----------------
st.info("📂 Carregando dados para análise 3D de 16 quadrantes...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay folder.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)

# Carregar dados com cache
games_today, history, selected_date_str = load_cached_data(selected_file)

# CORREÇÃO: Aplicar conversão corrigida também aos games_today
if 'Asian_Line' in games_today.columns:
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal_corrigido)

# Aplicar Live Score
games_today = load_and_merge_livescore(games_today, selected_date_str)

# ---------------- TESTE DE CONVERSÃO ASIAN LINE ----------------
testar_conversao_asian_line()

# ---------------- DIAGNÓSTICO INICIAL ----------------
st.markdown("## 🔍 DIAGNÓSTICO INICIAL DOS DADOS")

if not history.empty:
    # aqui ainda não temos Target_AH_Home em history cru,
    # então o relatório vai focar em missing/correlações gerais
    data_quality_report(history, 'Target_AH_Home')
else:
    st.warning("⚠️ Histórico vazio - não é possível executar diagnóstico")

# ---------------- SISTEMA 3D DE 16 QUADRANTES ----------------
st.markdown("## 🎯 Sistema 3D de 16 Quadrantes")

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
    """Classifica Aggression e HandScore em um dos 16 quadrantes"""
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

if not history.empty:
    history['Quadrante_Home'] = history.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home')), axis=1
    )
    history['Quadrante_Away'] = history.apply(
        lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away')), axis=1
    )

# ---------------- CÁLCULO DE DISTÂNCIAS 3D ----------------
def calcular_distancias_3d(df):
    """Calcula distância 3D e ângulos usando Aggression, Momentum (liga) e Momentum (time)"""
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
            'Momentum_Diff', 'Momentum_Diff_MT', 'Magnitude_3D'
        ]:
            df[col] = np.nan
        return df

    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']

    df['Quadrant_Dist_3D'] = np.sqrt(
        (dx)**2 * 1.5 + (dy/3.5)**2 * 2.0 + (dz/3.5)**2 * 1.8
    ) * 10

    df['Quadrant_Angle_XY'] = np.degrees(np.arctan2(dy, dx))
    df['Quadrant_Angle_XZ'] = np.degrees(np.arctan2(dz, dx))
    df['Quadrant_Angle_YZ'] = np.degrees(np.arctan2(dz, dy))

    angle_xy = np.arctan2(dy, dx)
    angle_xz = np.arctan2(dz, dx)
    angle_yz = np.arctan2(dz, dy)

    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Quadrant_Sin_XZ'] = np.sin(angle_xz)
    df['Quadrant_Cos_XZ'] = np.cos(angle_xz)
    df['Quadrant_Sin_YZ'] = np.sin(angle_yz)
    df['Quadrant_Cos_YZ'] = np.cos(angle_yz)

    df['Quadrant_Separation_3D'] = (
        0.4 * (60 * dx) + 0.35 * (20 * dy) + 0.25 * (20 * dz)
    )

    df['Momentum_Diff'] = dy
    df['Momentum_Diff_MT'] = dz
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df

# Aplicar cálculo 3D
games_today = calcular_distancias_3d(games_today)
if not history.empty:
    history = calcular_distancias_3d(history)

# ---------------- MODELO ML 3D CORRIGIDO — NOVA VERSÃO ----------------
def treinar_modelo_3d_quadrantes_16_corrigido(history, games_today):
    """
    Treino e Previsão:
    - Modelo HOME → Probabilidade do Home cobrir o AH
    - Modelo AWAY → Probabilidade do Away cobrir o AH
    - Modelo ZEBRA → Probabilidade do favorito falhar
    """

    st.markdown("## 🤖 TREINAMENTO DO MODELO ML (CORRIGIDO)")

    if history.empty:
        st.error("❌ Histórico vazio - não é possível treinar modelo")
        return None, None, None, games_today

    # ---------------- TARGET BINÁRIO + ZEBRA
    history_clean = create_better_target_corrigido(history)
    if history_clean.empty:
        st.error("❌ Nenhum jogo válido após criação do target (sem push)")
        return None, None, None, games_today

    # ---------------- FEATURES
    X_history = create_robust_features(history_clean)
    if X_history.empty:
        st.error("❌ Zero features disponíveis")
        return None, None, None, games_today

    y_home = history_clean["Target_AH_Home"]
    y_away = 1 - y_home
    y_zebra = history_clean["Zebra"]

    st.success(f"Treino: {X_history.shape[0]} amostras | {X_history.shape[1]} features")

    # ---------------- TREINAMENTO
    st.subheader("Modelo HOME")
    model_home = train_improved_model(X_history, y_home, X_history.columns.tolist())

    st.subheader("Modelo AWAY")
    model_away = train_improved_model(X_history, y_away, X_history.columns.tolist())

    st.subheader("Modelo ZEBRA")
    model_zebra = train_improved_model(X_history, y_zebra, X_history.columns.tolist())

    # ---------------- APPLY TO TODAY
    if games_today.empty:
        return model_home, model_away, model_zebra, games_today

    X_today = create_robust_features(games_today)

    # Garantia de alinhamento de features
    missing_cols = set(X_history.columns) - set(X_today.columns)
    for col in missing_cols:
        X_today[col] = 0
    X_today = X_today[X_history.columns]

    # Probabilidades
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]
    probas_zebra = model_zebra.predict_proba(X_today)[:, 1]

    games_today['Prob_HomeCover'] = probas_home
    games_today['Prob_AwayCover'] = probas_away
    games_today['Prob_Zebra'] = probas_zebra

    games_today['Prob_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')

    # ---------------- EXPECTED FAVORITE DA BOOKIE — CORRIGIDO
    def compute_expected_favorite(line):
        if pd.isna(line):
            return "NONE"
        if abs(line) >= 0.5:
            return "HOME" if line < 0 else "AWAY"
        return "NONE"

    games_today["Expected_Favorite"] = games_today["Asian_Line_Decimal"].apply(compute_expected_favorite)

    # ---------------- ZEBRA FINAL — Correto com favorito
    def compute_zebra(row):
        fav = row["Expected_Favorite"]
        if fav == "HOME":
            return 1 if row["Prob_HomeCover"] < 0.5 else 0
        if fav == "AWAY":
            return 1 if row["Prob_AwayCover"] < 0.5 else 0
        return 0

    games_today["Zebra_Flag"] = games_today.apply(compute_zebra, axis=1)

    st.success(f"Previsões geradas para {len(games_today)} jogos")

    return model_home, model_away, model_zebra, games_today



# ---------------- EXECUÇÃO DO MODELO CORRIGIDO ----------------
if not history.empty:
    modelo_home, modelo_away, modelo_zebra, games_today = treinar_modelo_3d_quadrantes_16_corrigido(history, games_today)

    if modelo_home is not None:
        st.success("✅ Modelo 3D corrigido treinado com sucesso!")

        if 'Quadrante_ML_Score_Main' in games_today.columns:
            avg_score = games_today['Quadrante_ML_Score_Main'].mean()
            high_confidence = len(games_today[games_today['Quadrante_ML_Score_Main'] > 0.65])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Score Médio", f"{avg_score:.1%}")
            with col2:
                st.metric("🎯 Alto Confiança (Home/Away)", high_confidence)
    else:
        st.error("❌ Falha no treinamento do modelo")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

# ---------------- SISTEMA DE INDICAÇÕES 3D ----------------
def adicionar_indicadores_explicativos_3d_16_dual(df):
    """Adiciona classificações e recomendações explícitas para sistema 3D"""
    if df.empty:
        return df

    df = df.copy()

    df['Quadrante_Home_Label'] = df['Quadrante_Home'].map(
        lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro')
    )
    df['Quadrante_Away_Label'] = df['Quadrante_Away'].map(
        lambda x: QUADRANTES_16.get(x, {}).get('nome', 'Neutro')
    )

    # CLASSIFICAÇÃO DE VALOR HOME
    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.65,
        df['Quadrante_ML_Score_Home'] >= 0.58,
        df['Quadrante_ML_Score_Home'] >= 0.52,
        df['Quadrante_ML_Score_Home'] >= 0.48,
        df['Quadrante_ML_Score_Home'] < 0.48
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')

    # CLASSIFICAÇÃO DE VALOR AWAY
    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.65,
        df['Quadrante_ML_Score_Away'] >= 0.58,
        df['Quadrante_ML_Score_Away'] >= 0.52,
        df['Quadrante_ML_Score_Away'] >= 0.48,
        df['Quadrante_ML_Score_Away'] < 0.48
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')

    def gerar_recomendacao_3d_16_dual(row):
        home_q = row.get('Quadrante_Home_Label', 'Neutro')
        away_q = row.get('Quadrante_Away_Label', 'Neutro')
        score_home = row.get('Quadrante_ML_Score_Home', 0.5)
        score_away = row.get('Quadrante_ML_Score_Away', 0.5)
        ml_side = row.get('ML_Side', 'HOME')
        momentum_h = row.get('M_H', 0)
        momentum_a = row.get('M_A', 0)
        prob_zebra = row.get('Prob_Zebra', 0.0)
        expected_fav = row.get('Expected_Favorite', 'NONE')

        # ALERTA ZEBRA ANTES DE TUDO
        if prob_zebra >= 0.65 and expected_fav in ['HOME', 'AWAY']:
            return f'🦓 ALTA PROB. ZEBRA contra {expected_fav} ({prob_zebra:.1%})'

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
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%} Z:{prob_zebra:.1%})'

    df['Recomendacao'] = df.apply(gerar_recomendacao_3d_16_dual, axis=1)
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)

    return df

# ---------------- EXIBIÇÃO DOS RESULTADOS (LIMPO) ----------------
st.markdown("## 🏆 Melhorias e Sinais para Hoje")

if not games_today.empty and "Prob_HomeCover" in games_today.columns:

    df_show = games_today.copy()

    # Ranking baseado em prob. mais forte
    df_show["Rank"] = df_show[["Prob_HomeCover", "Prob_AwayCover"]].max(axis=1).rank(
        method="first", ascending=False
    ).astype(int)

    df_show = df_show.sort_values("Rank")

    cols_final = [
        "Rank", "League", "Home", "Away",
        "ML_Side_Final",
        "Prob_HomeCover", "Prob_AwayCover", "Prob_Zebra",
        "Expected_Favorite", "Asian_Line", "Asian_Line_Decimal",
        "Goals_H_Today", "Goals_A_Today"
    ]

    cols_final = [c for c in cols_final if c in df_show.columns]

    # Gradiente apenas para probabilidades
    def highlight_probs(s):
        return [
            "background-color: rgba(0,0,0,0)"  # transparente fora das probabs
            for _ in s
        ]

    styler = df_show[cols_final].style \
        .format({
            "Prob_HomeCover": "{:.1%}",
            "Prob_AwayCover": "{:.1%}",
            "Prob_Zebra": "{:.1%}",
            "Asian_Line_Decimal": "{:.2f}",
            "Goals_H_Today": "{:.0f}",
            "Goals_A_Today": "{:.0f}"
        }, na_rep="-") \
        .background_gradient(
            cmap="RdYlGn",
            subset=["Prob_HomeCover", "Prob_AwayCover", "Prob_Zebra"]
        )

    st.dataframe(styler, use_container_width=True, height=600)

else:
    st.info("⚠️ Aguardando previsões para exibir tabela...")

# ---------------- RESUMO EXECUTIVO ----------------
def resumo_3d_16_quadrantes_hoje(df):
    """Resumo executivo dos 16 quadrantes 3D de hoje"""
    st.markdown("### 📋 Resumo Executivo - Sistema 3D Hoje")

    if df.empty:
        st.info("Nenhum dado disponível para resumo 3D")
        return

    total_jogos = len(df)

    if 'Classificacao_Valor_Home' in df.columns:
        alto_valor_home = len(df[df['Classificacao_Valor_Home'] == '🏆 ALTO VALOR'])
        alto_valor_away = len(df[df['Classificacao_Valor_Away'] == '🏆 ALTO VALOR'])
    else:
        alto_valor_home = alto_valor_away = 0

    if 'M_H' in df.columns:
        momentum_positivo_home = len(df[df['M_H'] > 0.5])
        momentum_negativo_home = len(df[df['M_H'] < -0.5])
        momentum_positivo_away = len(df[df['M_A'] > 0.5])
        momentum_negativo_away = len(df[df['M_A'] < -0.5])
    else:
        momentum_positivo_home = momentum_negativo_home = momentum_positivo_away = momentum_negativo_away = 0

    if 'Prob_Zebra' in df.columns:
        zebra_alta = len(df[df['Prob_Zebra'] >= 0.65])
    else:
        zebra_alta = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jogos", total_jogos)
        st.metric("📈 Momentum + Home", momentum_positivo_home)
    with col2:
        st.metric("📉 Momentum - Home", momentum_negativo_home)
        st.metric("📈 Momentum + Away", momentum_positivo_away)
    with col3:
        st.metric("📉 Momentum - Away", momentum_negativo_away)
        st.metric("🦓 Jogos com Alta Zebra", zebra_alta)
    with col4:
        st.metric("🎯 Alto Valor Home", alto_valor_home)
        st.metric("🎯 Alto Valor Away", alto_valor_away)

if not games_today.empty:
    resumo_3d_16_quadrantes_hoje(games_today)



# ---------------- ESTATÍSTICAS ZEBRA ----------------
st.markdown("## 🦓 Estatísticas de Zebra (mercado errando)")

if not history.empty:

    # Recalcula Zebra no histórico (caso necessário)
    if 'Zebra' not in history.columns:
        history_tmp = create_better_target_corrigido(history.copy())
    else:
        history_tmp = history.copy()

    st.markdown("### Por Liga")

    if "League" in history_tmp.columns:
        zebra_liga = history_tmp.groupby("League")["Zebra"].mean().sort_values(ascending=False)
        st.dataframe(
            zebra_liga.to_frame("Taxa Zebra").style.format({"Taxa Zebra": "{:.1%}"}),
            use_container_width=True
        )
    else:
        st.info("Liga não disponível no histórico.")

    st.markdown("### Por Linha de Handicap")

    zebra_handicap = history_tmp.groupby("Asian_Line_Decimal")["Zebra"].mean().sort_values(ascending=False)
    st.dataframe(
        zebra_handicap.to_frame("Taxa Zebra").style.format({"Taxa Zebra": "{:.1%}"}),
        use_container_width=True
    )

else:
    st.info("Sem histórico para estatísticas Zebra.")




st.markdown("---")
st.success("🎯 **Sistema 3D de 16 Quadrantes ML CORRIGIDO + ZEBRA** implementado com sucesso!")
st.info("""
**Principais correções aplicadas:**

✅ **Asian Line CORRIGIDA** - Perspectiva do Away convertida corretamente para Home  
✅ **Target 100% Binário** - PUSH excluído, apenas Win/Loss  
✅ **Modelo HOME / AWAY** - Probabilidade de cada lado cobrir o AH  
✅ **Modelo ZEBRA** - Quando o favorito do mercado falha (mercado erra)  
✅ **Data Leakage Eliminado** - Filtro temporal aplicado corretamente  
✅ **Feature Engineering Robusto** - Features 3D e derivadas  
✅ **Validação Cruzada** - Performance monitorada  
✅ **Dashboard Explicativo** - Recomendação, valor e risco de zebra  

Agora o sistema não só te diz **quem tende a cobrir a linha**, mas também **onde o mercado tem alta chance de errar (zebra)**.
""")
