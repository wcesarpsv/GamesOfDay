# ########################################################
# BLOCO 1 – CONFIGURAÇÃO E IMPORTS
# ########################################################
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from catboost import CatBoostClassifier
import shap
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from datetime import datetime
import math

st.set_page_config(page_title="Bet Indicator – Forecast V2 + Quadrantes", layout="wide")
st.title("🎯 Forecast V2 + Sistema de 16 Quadrantes")


# Paths
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "coppa" ,"copas", "uefa", "nordeste", "afc", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ########################################################
# BLOCO 2 – FUNÇÕES AUXILIARES BÁSICAS
# ########################################################
def load_all_games(folder: str) -> pd.DataFrame:
    """Carrega todos os arquivos CSV do folder especificado"""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(os.path.join(folder, f))
            dfs.append(df)
        except Exception as e:
            st.warning(f"Erro ao carregar {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_selected_csvs(folder: str) -> pd.DataFrame:
    """Carrega arquivos selecionados (hoje e ontem)"""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    
    today_file = files[-1]
    yesterday_file = files[-2] if len(files) >= 2 else None

    st.markdown("### 📂 Select matches to display")
    col1, col2 = st.columns(2)
    today_checked = col1.checkbox("Today Matches", value=True)
    yesterday_checked = col2.checkbox("Yesterday Matches", value=False)

    selected_dfs = []
    if today_checked:
        selected_dfs.append(pd.read_csv(os.path.join(folder, today_file)))
    if yesterday_checked and yesterday_file:
        selected_dfs.append(pd.read_csv(os.path.join(folder, yesterday_file)))

    return pd.concat(selected_dfs, ignore_index=True) if selected_dfs else pd.DataFrame()

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra ligas excluindo copas e torneios"""
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, filename: str):
    """Salva modelo no diretório de modelos"""
    path = os.path.join(MODELS_FOLDER, filename)
    with open(path, "wb") as f:
        joblib.dump(model, f)

def load_model(filename: str):
    """Carrega modelo do diretório de modelos"""
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None

def compute_double_chance_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula odds 1X e X2 a partir das odds 1X2"""
    df = df.copy()
    if set(["Odd_H", "Odd_D", "Odd_A"]).issubset(df.columns):
        probs = pd.DataFrame()
        probs["p_H"] = 1 / df["Odd_H"]
        probs["p_D"] = 1 / df["Odd_D"]
        probs["p_A"] = 1 / df["Odd_A"]
        probs = probs.div(probs.sum(axis=1), axis=0)
        df["Odd_1X"] = 1 / (probs["p_H"] + probs["p_D"])
        df["Odd_X2"] = 1 / (probs["p_A"] + probs["p_D"])
    return df

# ########################################################
# BLOCO 3 – CARREGAMENTO E PREPARAÇÃO DE DADOS
# ########################################################
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e prepara dados históricos e do dia"""
    st.info("📂 Carregando e preparando dados...")
    
    # Carregar dados históricos
    history = filter_leagues(load_all_games(GAMES_FOLDER))
    history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()
    
    if history.empty:
        st.error("⚠️ No valid historical data found in GamesDay.")
        st.stop()
    
    # Carregar jogos de hoje
    games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
    if "Goals_H_FT" in games_today.columns:
        games_today = games_today[games_today["Goals_H_FT"].isna()].copy()
    
    if games_today.empty:
        st.error("⚠️ No valid matches selected.")
        st.stop()
    
    # Aplicar odds duplas
    history = compute_double_chance_odds(history)
    games_today = compute_double_chance_odds(games_today)
    
    return history, games_today

# ########################################################
# BLOCO 4 – SISTEMA DE 16 QUADRANTES
# ########################################################
def setup_quadrantes_system():
    """Configura o sistema de 16 quadrantes"""
    QUADRANTES_16 = {
        1: {"nome": "Fav Forte Muito Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 45, "hs_max": 60},
        2: {"nome": "Fav Forte Forte", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 30, "hs_max": 45},
        3: {"nome": "Fav Forte Moderado", "agg_min": 0.75, "agg_max": 1.0, "hs_min": 15, "hs_max": 30},
        4: {"nome": "Fav Forte Neutro", "agg_min": 0.75, "agg_max": 1.0, "hs_min": -15, "hs_max": 15},
        5: {"nome": "Fav Moderado Muito Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 45, "hs_max": 60},
        6: {"nome": "Fav Moderado Forte", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 30, "hs_max": 45},
        7: {"nome": "Fav Moderado Moderado", "agg_min": 0.25, "agg_max": 0.75, "hs_min": 15, "hs_max": 30},
        8: {"nome": "Fav Moderado Neutro", "agg_min": 0.25, "agg_max": 0.75, "hs_min": -15, "hs_max": 15},
        9: {"nome": "Under Moderado Neutro", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -15, "hs_max": 15},
        10: {"nome": "Under Moderado Moderado", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -30, "hs_max": -15},
        11: {"nome": "Under Moderado Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -45, "hs_max": -30},
        12: {"nome": "Under Moderado Muito Forte", "agg_min": -0.75, "agg_max": -0.25, "hs_min": -60, "hs_max": -45},
        13: {"nome": "Under Forte Neutro", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -15, "hs_max": 15},
        14: {"nome": "Under Forte Moderado", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -30, "hs_max": -15},
        15: {"nome": "Under Forte Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -45, "hs_max": -30},
        16: {"nome": "Under Forte Muito Forte", "agg_min": -1.0, "agg_max": -0.75, "hs_min": -60, "hs_max": -45}
    }
    return QUADRANTES_16

def classificar_quadrante_16(agg: float, hs: float, QUADRANTES_16: dict) -> int:
    """Classifica Aggression e HandScore em um dos 16 quadrantes"""
    if pd.isna(agg) or pd.isna(hs):
        return 0
    
    for quadrante_id, config in QUADRANTES_16.items():
        agg_ok = (config['agg_min'] <= agg <= config['agg_max'])
        hs_ok = (config['hs_min'] <= hs <= config['hs_max'])
        if agg_ok and hs_ok:
            return quadrante_id
    
    return 0

def calcular_distancias_quadrantes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distância, separação e ângulo entre quadrantes Home e Away"""
    df = df.copy()
    if all(col in df.columns for col in ['Aggression_Home', 'Aggression_Away', 'HandScore_Home', 'HandScore_Away']):
        dx = df['Aggression_Home'] - df['Aggression_Away']
        dy = df['HandScore_Home'] - df['HandScore_Away']
        df['Quadrant_Dist'] = np.sqrt(dx**2 + (dy/60)**2 * 2.5) * 10
        df['Quadrant_Separation'] = 0.5 * (dy + 60 * dx)
        df['Quadrant_Angle'] = np.degrees(np.arctan2(dy, dx))
    else:
        st.warning("⚠️ Colunas Aggression/HandScore não encontradas")
        df['Quadrant_Dist'] = np.nan
        df['Quadrant_Separation'] = np.nan
        df['Quadrant_Angle'] = np.nan
    return df

def aplicar_sistema_quadrantes(history: pd.DataFrame, games_today: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica sistema de quadrantes aos dataframes"""
    QUADRANTES_16 = setup_quadrantes_system()
    
    # Aplicar classificação de quadrantes
    for df in [history, games_today]:
        if all(col in df.columns for col in ['Aggression_Home', 'HandScore_Home']):
            df['Quadrante_Home'] = df.apply(
                lambda x: classificar_quadrante_16(x.get('Aggression_Home'), x.get('HandScore_Home'), QUADRANTES_16), 
                axis=1
            )
            df['Quadrante_Away'] = df.apply(
                lambda x: classificar_quadrante_16(x.get('Aggression_Away'), x.get('HandScore_Away'), QUADRANTES_16), 
                axis=1
            )
        else:
            st.warning("⚠️ Colunas Aggression/HandScore não encontradas - pulando quadrantes")
            df['Quadrante_Home'] = 0
            df['Quadrante_Away'] = 0
    
    # Calcular distâncias
    history = calcular_distancias_quadrantes(history)
    games_today = calcular_distancias_quadrantes(games_today)
    
    return history, games_today


def show_shap_analysis(model, X, title: str):
    st.markdown(f"### 🔍 SHAP Explainability – {title}")

    # SHAP Explainer para CatBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Sumário geral (todas classes)
    st.markdown("📌 **Feature Importance Global**")
    fig1 = shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig1)

    # Se classificação multi-classe (1X2)
    if isinstance(shap_values, list) and len(shap_values) == 3:
        st.markdown("📌 **Impacto nas Classes (Home, Draw, Away)**")
        class_labels = ["Home", "Draw", "Away"]
        
        for idx, lbl in enumerate(class_labels):
            st.write(f"#### 🎯 Classe: {lbl}")
            fig_class = shap.summary_plot(shap_values[idx], X, show=False)
            st.pyplot(fig_class)


# ########################################################
# BLOCO 5 – CONSTRUÇÃO DO DATASET DE FEATURES
# ########################################################
def setup_feature_blocks(use_odds: bool) -> dict:
    """Configura os blocos de features baseado na escolha de usar odds"""
    base_blocks = {
        "strength": [
           
        ],
        "quadrantes": [
            "Quadrante_Home", "Quadrante_Away", 
            "Quadrant_Dist", "Quadrant_Separation", "Quadrant_Angle",
            "Aggression_Home", "Aggression_Away", 
            "HandScore_Home", "HandScore_Away"
        ],
        "categorical": [
            
        ]
    }
    
    if use_odds:
        base_blocks["odds"] = ["Odd_H", "Odd_D", "Odd_A", "Odd_1X", "Odd_X2"]
    
    return base_blocks

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features avançadas como bandas de liga e dominância"""
    df = df.copy()
    
    # Momentum Difference
    df["M_Diff"] = df["M_H"] - df["M_A"]
    
    # Classificação de ligas
    def classify_leagues_variation(history_df):
        agg = (
            history_df.groupby("League")
            .agg(
                M_H_Min=("M_H","min"), M_H_Max=("M_H","max"),
                M_A_Min=("M_A","min"), M_A_Max=("M_A","max"),
                Hist_Games=("M_H","count")
            ).reset_index()
        )
        agg["Variation_Total"] = (agg["M_H_Max"] - agg["M_H_Min"]) + (agg["M_A_Max"] - agg["M_A_Min"])
        def label(v):
            if v > 6.0: return "High Variation"
            if v >= 3.0: return "Medium Variation"
            return "Low Variation"
        agg["League_Classification"] = agg["Variation_Total"].apply(label)
        return agg[["League","League_Classification","Variation_Total","Hist_Games"]]
    
    # Compute league bands
    def compute_league_bands(history_df):
        hist = history_df.copy()
        hist["M_Diff"] = hist["M_H"] - hist["M_A"]
        diff_q = (
            hist.groupby("League")["M_Diff"]
                .quantile([0.20,0.80]).unstack()
                .rename(columns={0.2:"P20_Diff",0.8:"P80_Diff"})
                .reset_index()
        )
        home_q = (
            hist.groupby("League")["M_H"]
                .quantile([0.20,0.80]).unstack()
                .rename(columns={0.2:"Home_P20",0.8:"Home_P80"})
                .reset_index()
        )
        away_q = (
            hist.groupby("League")["M_A"]
                .quantile([0.20,0.80]).unstack()
                .rename(columns={0.2:"Away_P20",0.8:"Away_P80"})
                .reset_index()
        )
        out = diff_q.merge(home_q,on="League",how="inner").merge(away_q,on="League",how="inner")
        return out

    # Dominância
    def dominant_side(row, threshold=0.90):
        m_h, m_a = row["M_H"], row["M_A"]
        if (m_h >= threshold) and (m_a <= -threshold):
            return "Both extremes (Home↑ & Away↓)"
        if (m_a >= threshold) and (m_h <= -threshold):
            return "Both extremes (Away↑ & Home↓)"
        if m_h >= threshold: return "Home strong"
        if m_h <= -threshold: return "Home weak"
        if m_a >= threshold: return "Away strong"
        if m_a <= -threshold: return "Away weak"
        return "Mixed / Neutral"
    
    # Aplicar às ligas (apenas para history)
    if "League" in df.columns and not df.empty:
        league_class = classify_leagues_variation(df)
        league_bands = compute_league_bands(df)
        
        df = df.merge(league_class, on="League", how="left")
        df = df.merge(league_bands, on="League", how="left")
        
        # Bandas Home/Away
        df["Home_Band"] = np.where(
            df["M_H"] <= df["Home_P20"], "Bottom 20%",
            np.where(df["M_H"] >= df["Home_P80"], "Top 20%", "Balanced")
        )
        df["Away_Band"] = np.where(
            df["M_A"] <= df["Away_P20"], "Bottom 20%",
            np.where(df["M_A"] >= df["Away_P80"], "Top 20%", "Balanced")
        )
        df["Dominant"] = df.apply(dominant_side, axis=1)
        df["Home_Band_Num"] = df["Home_Band"].map({"Bottom 20%":1,"Balanced":2,"Top 20%":3})
        df["Away_Band_Num"] = df["Away_Band"].map({"Bottom 20%":1,"Balanced":2,"Top 20%":3})
        df["Band_Diff"] = df["Home_Band_Num"] - df["Away_Band_Num"]
    
    return df

def build_feature_matrix(df: pd.DataFrame, leagues_df: pd.DataFrame, feature_blocks: dict, 
                        fit_encoder: bool = False, encoder: OneHotEncoder = None) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Constrói matriz de features baseada nos blocos definidos"""
    dfs = []
    
    # Features numéricas e de odds
    for block_name, cols in feature_blocks.items():
        if block_name == "categorical": 
            continue
        available_cols = [c for c in cols if c in df.columns]
        if available_cols:
            dfs.append(df[available_cols])
    
    # Features de liga
    if leagues_df is not None and not leagues_df.empty:
        dfs.append(leagues_df)
    
    # Features categóricas (One-Hot Encoding)
    cat_cols = [c for c in ["Dominant","League_Classification"] if c in df.columns]
    if cat_cols:
        if fit_encoder:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        dfs.append(encoded_df)

    # Features numéricas categóricas
    for col in ["Home_Band_Num","Away_Band_Num"]:
        if col in df.columns:
            dfs.append(df[[col]])
    
    X = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
    return X, encoder

# ########################################################
# BLOCO 6 – CONFIGURAÇÕES DO MODELO (SIDEBAR)
# ########################################################
def setup_sidebar_config() -> dict:
    st.sidebar.header("⚙️ Configurações do Modelo")

    config = {
        "ml_model": "CatBoost",
        "use_odds": st.sidebar.checkbox("Usar Odds como Features", value=True),
        "retrain": st.sidebar.checkbox("Retreinar Modelos", value=False)
    }

    return config


# ########################################################
# BLOCO 7 – TREINAMENTO E AVALIAÇÃO
# ########################################################
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, name: str, num_classes: int, config: dict) -> tuple[dict, any]:
    """Treina e avalia modelo com CatBoost"""
    model_name = "CatBoost"
    filename = f"{model_name}_{name}_fc_v2_quadrantes.pkl"
    model = None

    # Tentar carregar modelo salvo
    if not config.get("retrain", False):
        model = load_model(filename)

    # Treinar modelo caso não exista salvo
    if model is None:
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function='MultiClass' if num_classes > 2 else 'Logloss',
            eval_metric='MultiClass' if num_classes > 2 else 'Logloss',
            random_seed=42,
            verbose=False
        )

        model.fit(X_train, y_train)
        save_model(model, filename)

    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Avaliação
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)

    if num_classes == 2:
        bs = brier_score_loss(y_val, probs[:, 1])
        bs = f"{bs:.3f}"
    else:
        y_onehot = pd.get_dummies(y_val).values
        bs_raw = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        bs = f"{bs_raw:.3f} (multi)"

    metrics = {
        "Modelo": f"{model_name} - {name}",
        "Odds": "Sim" if config["use_odds"] else "Não",
        "Acurácia": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
        "Brier": bs,
        "Features": f"{X.shape[1]}"
    }

    return metrics, model

# ########################################################
# BLOCO 8 – PREVISÕES E RESULTADOS
# ########################################################
def make_predictions(games_today: pd.DataFrame, models: dict, X_today: dict) -> pd.DataFrame:
    """Faz previsões para todos os modelos"""
    df_pred = games_today.copy()
    
    # Previsões 1X2
    if "1X2" in models and "1X2" in X_today:
        probs_1x2 = models["1X2"].predict_proba(X_today["1X2"])
        df_pred["p_home"], df_pred["p_draw"], df_pred["p_away"] = probs_1x2.T
    
    # Previsões Over/Under
    if "OverUnder25" in models and "OverUnder25" in X_today:
        probs_ou = models["OverUnder25"].predict_proba(X_today["OverUnder25"])
        df_pred["p_over25"], df_pred["p_under25"] = probs_ou.T
    
    # Previsões BTTS
    if "BTTS" in models and "BTTS" in X_today:
        probs_btts = models["BTTS"].predict_proba(X_today["BTTS"])
        df_pred["p_btts_yes"], df_pred["p_btts_no"] = probs_btts.T
    
    return df_pred

# ########################################################
# BLOCO 9 – VISUALIZAÇÃO E ANÁLISE
# ########################################################
def style_probabilities(val, col):
    """Aplica estilo colorido às probabilidades"""
    def color_prob(v, color):
        alpha = min(int(v * 255), 200)
        return f"background-color: rgba({color}, {alpha/255:.2f})"
    
    color_map = {
        "p_home": "0,200,0",
        "p_draw": "150,150,150", 
        "p_away": "255,140,0",
        "p_over25": "0,100,255",
        "p_under25": "128,0,128",
        "p_btts_yes": "0,200,200",
        "p_btts_no": "200,0,0"
    }
    
    return color_prob(val, color_map.get(col, "150,150,150")) if pd.notna(val) else ""

def display_results(games_today: pd.DataFrame, metrics_df: pd.DataFrame, config: dict):
    """Exibe resultados finais formatados"""
    st.markdown("### 📊 Estatísticas dos Modelos")
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("### 📌 Previsões para os Jogos Selecionados")
    
    # Colunas para exibição
    cols_final = [
        "Date", "Time", "League", "Home", "Away",
        "Odd_H", "Odd_D", "Odd_A",
        "p_home", "p_draw", "p_away",
        "p_over25", "p_under25", 
        "p_btts_yes", "p_btts_no"
    ]
    
    # Aplicar estilos
    styled_df = (
        games_today[cols_final]
        .style.format({
            "Odd_H": "{:.2f}", "Odd_D": "{:.2f}", "Odd_A": "{:.2f}",
            "p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}",
            "p_over25": "{:.1%}", "p_under25": "{:.1%}",
            "p_btts_yes": "{:.1%}", "p_btts_no": "{:.1%}",
        }, na_rep="—")
        .applymap(lambda v: style_probabilities(v, "p_home"), subset=["p_home"])
        .applymap(lambda v: style_probabilities(v, "p_draw"), subset=["p_draw"])
        .applymap(lambda v: style_probabilities(v, "p_away"), subset=["p_away"])
        .applymap(lambda v: style_probabilities(v, "p_over25"), subset=["p_over25"])
        .applymap(lambda v: style_probabilities(v, "p_under25"), subset=["p_under25"])
        .applymap(lambda v: style_probabilities(v, "p_btts_yes"), subset=["p_btts_yes"])
        .applymap(lambda v: style_probabilities(v, "p_btts_no"), subset=["p_btts_no"])
    )
    
    st.dataframe(styled_df, use_container_width=True, height=800)
    
    # Resumo da configuração
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Configuração Atual")
    st.sidebar.write(f"**Modelo:** {config['ml_model']}")
    st.sidebar.write(f"**Usando Odds:** {'Sim' if config['use_odds'] else 'Não'}")
    st.sidebar.write(f"**Retreinar:** {'Sim' if config['retrain'] else 'Não'}")

# ########################################################
# BLOCO PRINCIPAL – EXECUÇÃO
# ########################################################
def main():
    """Função principal que orquestra toda a execução"""
    
    # Carregar dados
    history, games_today = load_and_prepare_data()
    
    # Configurar sistema de quadrantes
    history, games_today = aplicar_sistema_quadrantes(history, games_today)
    
    # Calcular features avançadas
    history = compute_advanced_features(history)
    games_today = compute_advanced_features(games_today)
    
    # Definir targets
    history["Target"] = history.apply(
        lambda row: 0 if row["Goals_H_FT"] > row["Goals_A_FT"]
        else (1 if row["Goals_H_FT"] == row["Goals_A_FT"] else 2),
        axis=1,
    )
    history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
    history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)
    
    # Configurações do sidebar
    config = setup_sidebar_config()
    
    # Configurar blocos de features
    feature_blocks = setup_feature_blocks(config["use_odds"])
    
    # Preparar dados de liga
    history_leagues = pd.get_dummies(history["League"], prefix="League")
    games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
    games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)
    
    # Construir matrizes de features
    X_1x2, encoder_cat = build_feature_matrix(history, history_leagues, feature_blocks, fit_encoder=True)
    X_today_1x2, _ = build_feature_matrix(games_today, games_today_leagues, feature_blocks, fit_encoder=False, encoder=encoder_cat)
    X_today_1x2 = X_today_1x2.reindex(columns=X_1x2.columns, fill_value=0)
    
    # Mesmo conjunto para OU e BTTS
    X_ou = X_1x2.copy()
    X_today_ou = X_today_1x2.copy()
    X_btts = X_1x2.copy()
    X_today_btts = X_today_1x2.copy()
    
    # Treinar modelos
    st.info("🎯 Treinando modelos...")
    stats = []
    models = {}
    
    # Modelo 1X2
    metrics_1x2, model_1x2 = train_and_evaluate(X_1x2, history["Target"], "1X2", 3, config)
    stats.append(metrics_1x2)
    models["1X2"] = model_1x2

    with st.expander("🧠 SHAP Analysis – Modelo 1X2", expanded=False):
        show_shap_analysis(model_1x2, X_1x2.sample(min(1000, len(X_1x2))), "1X2")

    
    # Modelo Over/Under
    metrics_ou, model_ou = train_and_evaluate(X_ou, history["Target_OU25"], "OverUnder25", 2, config)
    stats.append(metrics_ou)
    models["OverUnder25"] = model_ou
    
    # Modelo BTTS
    metrics_btts, model_btts = train_and_evaluate(X_btts, history["Target_BTTS"], "BTTS", 2, config)
    stats.append(metrics_btts)
    models["BTTS"] = model_btts
    
    # Fazer previsões
    X_today_dict = {
        "1X2": X_today_1x2,
        "OverUnder25": X_today_ou, 
        "BTTS": X_today_btts
    }
    
    games_today_pred = make_predictions(games_today, models, X_today_dict)
    
    # Exibir resultados
    metrics_df = pd.DataFrame(stats)
    display_results(games_today_pred, metrics_df, config)
    
    st.success("✅ Análise concluída com sucesso!")

# Executar aplicação
if __name__ == "__main__":
    main()
