from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

# =================== CONFIG STREAMLIT ===================
st.set_page_config(page_title="Sistema Calibrado - HOME Perspective", layout="wide")
st.title("🎯 Sistema de Handicap Calibrado (HOME Perspective)")

# =================== CONFIGURAÇÕES GERAIS ===================
PAGE_PREFIX = "AsianLine_Calibrado"
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_DIR = os.path.join(BASE_DIR, GAMES_FOLDER)
os.makedirs(GAMES_DIR, exist_ok=True)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def convert_asian_line_to_decimal(value):
    """Converte linha Away para decimal Home."""
    if pd.isna(value):
        return np.nan
    try:
        value = str(value).strip()
        if "/" in value:
            parts = [float(p) for p in value.split("/")]
            val = np.mean(parts)
        else:
            val = float(value)
        return -val  # inverter (mercado é Away)
    except:
        return np.nan


def calcular_handicap_otimo(row):
    gh, ga = row.get("Goals_H_FT", 0), row.get("Goals_A_FT", 0)
    margin = gh - ga
    handicaps = np.arange(-2, 2.25, 0.25)
    best_h, best_score = 0, -999
    for h in handicaps:
        adj = margin + h
        score = 1.5 - abs(h)*0.2 if adj > 0 else (-0.5 - abs(h)*0.1 if adj < 0 else 0.3)
        if score > best_score:
            best_score, best_h = score, h
    return best_h


def criar_target_discreto(row):
    h = row["Handicap_Otimo"]
    if h <= -1.25: return "STRONG_HOME"
    elif h <= -0.5: return "MODERATE_HOME"
    elif h < 0: return "LIGHT_HOME"
    elif h == 0: return "NEUTRAL"
    elif h < 0.5: return "LIGHT_AWAY"
    elif h < 1.25: return "MODERATE_AWAY"
    else: return "STRONG_AWAY"


def carregar_ultimo_csv():
    """Carrega o CSV mais recente da pasta GamesDay"""
    arquivos = [f for f in os.listdir(GAMES_DIR) if f.endswith(".csv")]
    if not arquivos:
        st.error("❌ Nenhum arquivo encontrado em GamesDay/")
        st.stop()
    arquivos.sort(reverse=True)
    arquivo_recente = arquivos[0]
    path = os.path.join(GAMES_DIR, arquivo_recente)
    st.success(f"📂 Carregado: {arquivo_recente}")
    return pd.read_csv(path)

# ============================================================
# TREINAMENTO E PREDIÇÃO
# ============================================================

def treinar_e_prever(history):
    st.subheader("🧠 Treinando Modelos Calibrados (HOME Perspective)")

    # Corrigir sinal do target
    history["Asian_Line_Decimal"] = history["Asian_Line"].apply(convert_asian_line_to_decimal)
    history["Target_Handicap_Home"] = -history["Asian_Line_Decimal"]

    # Criar labels supervisionadas
    history["Handicap_Otimo"] = history.apply(calcular_handicap_otimo, axis=1)
    history["Handicap_Categoria"] = history.apply(criar_target_discreto, axis=1)

    # Selecionar features principais
    feats = ["Quadrant_Dist_3D", "Quadrant_Separation_3D", "Magnitude_3D", "Momentum_Diff", "Momentum_Diff_MT"]
    feats = [f for f in feats if f in history.columns]

    if len(feats) < 3:
        st.error("⚠️ Features insuficientes para treino.")
        return None

    X = history[feats].fillna(0)
    y_reg = history["Target_Handicap_Home"]
    y_cls = history["Handicap_Categoria"]

    # Escalamento
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Regressão
    model_reg = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model_reg.fit(Xs, y_reg)
    mae = mean_absolute_error(y_reg, model_reg.predict(Xs))
    st.info(f"MAE Regressão: {mae:.3f}")

    # Classificação
    le = LabelEncoder()
    y_enc = le.fit_transform(y_cls)
    model_cls = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    model_cls.fit(Xs, y_enc)

    return model_reg, model_cls, scaler, le, feats

# ============================================================
# VALUE GAP + AUDITORIA
# ============================================================

def analisar_value(df, model_reg, model_cls, scaler, le, feats):
    st.subheader("💎 Análise de Value e Auditoria")

    X = df[feats].fillna(0)
    Xs = scaler.transform(X)
    df["Pred_Reg"] = model_reg.predict(Xs)
    preds_cls = model_cls.predict(Xs)
    df["Pred_Cls_Label"] = le.inverse_transform(preds_cls)

    map_cls = {"STRONG_HOME": -1.5, "MODERATE_HOME": -0.75, "LIGHT_HOME": -0.25, "NEUTRAL": 0,
               "LIGHT_AWAY": 0.25, "MODERATE_AWAY": 0.75, "STRONG_AWAY": 1.5}
    df["Pred_Cls_Val"] = df["Pred_Cls_Label"].map(map_cls)

    # Gap (mercado - predito)
    df["gap_reg"] = df["Asian_Line_Decimal"] - df["Pred_Reg"]
    df["gap_cls"] = df["Asian_Line_Decimal"] - df["Pred_Cls_Val"]
    df["Home_Value_Gap"] = 0.7 * df["gap_reg"] + 0.3 * df["gap_cls"]

    def classificar(g):
        if g > 0.4: return "STRONG HOME VALUE"
        elif g > 0.2: return "HOME VALUE"
        elif g < -0.4: return "STRONG AWAY VALUE"
        elif g < -0.2: return "AWAY VALUE"
        else: return "NO CLEAR VALUE"

    df["Recomendacao"] = df["Home_Value_Gap"].apply(classificar)

    # ===================== GRÁFICOS =====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Histograma Mercado vs Predito
    axes[0].hist(df["Asian_Line_Decimal"], bins=20, alpha=0.6, label="Mercado")
    axes[0].hist(df["Pred_Reg"], bins=20, alpha=0.6, label="Predito")
    axes[0].set_title("Distribuição: Mercado vs Predito")
    axes[0].legend()

    # Distribuição de Value Gap
    axes[1].hist(df["Home_Value_Gap"], bins=20, color="gray", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Distribuição do Home Value Gap")

    # Scatter Predito vs Mercado
    colors = df["Home_Value_Gap"].apply(lambda x: "green" if x > 0.3 else ("blue" if x < -0.3 else "gray"))
    axes[2].scatter(df["Asian_Line_Decimal"], df["Pred_Reg"], c=colors, edgecolors="k", alpha=0.7)
    axes[2].plot([-2, 2], [-2, 2], "k--", alpha=0.3)
    axes[2].set_xlabel("Mercado (Home Line)")
    axes[2].set_ylabel("Predito (Home)")
    axes[2].set_title("Predito vs Mercado (HOME Perspective)")

    plt.tight_layout()
    st.pyplot(fig)

    # ===================== AUDITORIA =====================
    st.markdown("### 📈 Estatísticas de Auditoria")
    resumo = {
        "Média Handicap Mercado": df["Asian_Line_Decimal"].mean(),
        "Média Handicap Predito": df["Pred_Reg"].mean(),
        "Média Home Value Gap": df["Home_Value_Gap"].mean(),
        "% HOME Value (>+0.2)": (df["Home_Value_Gap"] > 0.2).mean(),
        "% AWAY Value (<-0.2)": (df["Home_Value_Gap"] < -0.2).mean(),
    }
    st.dataframe(pd.DataFrame(resumo, index=["Auditoria"]).T)

    st.markdown("### 🔍 Amostra de Jogos e Recomendações")
    st.dataframe(df[["League", "Home", "Away", "Asian_Line_Decimal", "Pred_Reg", "Pred_Cls_Val", "Home_Value_Gap", "Recomendacao"]].head(20))

    return df

# ============================================================
# MAIN
# ============================================================

def main():
    df = carregar_ultimo_csv()
    df = df[~df["League"].str.lower().str.contains("|".join(EXCLUDED_LEAGUE_KEYWORDS), na=False)]

    if st.button("🚀 Executar Análise HOME Perspective"):
        model_reg, model_cls, scaler, le, feats = treinar_e_prever(df)
        if model_reg is not None:
            analisar_value(df, model_reg, model_cls, scaler, le, feats)

if __name__ == "__main__":
    main()
