from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Analisador de Handicap Ótimo - Calibrado (Home Perspective)", layout="wide")
st.title("🎯 Analisador de Handicap Ótimo – Calibrado (Home Perspective)")

# ============================================================
# 🔧 FUNÇÕES AUXILIARES
# ============================================================

def convert_asian_line_to_decimal(value):
    """Converte a linha do mercado (Away) para decimal na perspectiva HOME"""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    try:
        if "/" in value:
            parts = [float(p) for p in value.split("/")]
            avg = np.mean(parts)
        else:
            avg = float(value)
        # Inverter o sinal (mercado é Away)
        return -avg
    except Exception:
        return np.nan


def calcular_handicap_otimo_calibrado(row):
    """Calcula handicap ótimo (com limites realistas)"""
    gh, ga = row.get("Goals_H_FT", 0), row.get("Goals_A_FT", 0)
    margin = gh - ga
    handicaps = np.arange(-2, 2.25, 0.25)
    best_hcap, best_score = 0, -999
    for h in handicaps:
        adj = margin + h
        score = 1.5 - abs(h) * 0.2 if adj > 0 else (-0.5 - abs(h) * 0.1 if adj < 0 else 0.3)
        if score > best_score:
            best_score, best_hcap = score, h
    return best_hcap


def criar_target_handicap_discreto_calibrado(row):
    """Mapeia handicap ótimo para categoria"""
    h = row["Handicap_Otimo_Calibrado"]
    if h <= -1.25: return "STRONG_HOME"
    elif h <= -0.5: return "MODERATE_HOME"
    elif h < 0: return "LIGHT_HOME"
    elif h == 0: return "NEUTRAL"
    elif h < 0.5: return "LIGHT_AWAY"
    elif h < 1.25: return "MODERATE_AWAY"
    else: return "STRONG_AWAY"

# ============================================================
# 🧠 TREINAMENTO
# ============================================================

def treinar_modelo(history, games_today):
    """Treina regressão e classificação calibradas (Home Perspective)"""
    st.subheader("🧠 Treinando Modelos (Home Perspective)")

    # Corrigir sinal do target (Home perspective)
    history["Asian_Line_Decimal"] = history["Asian_Line"].apply(convert_asian_line_to_decimal)
    history["Target_Handicap_Home"] = -history["Asian_Line_Decimal"]

    # Gerar handicap ótimo calibrado
    history["Handicap_Otimo_Calibrado"] = history.apply(calcular_handicap_otimo_calibrado, axis=1)
    history["Handicap_Categoria_Calibrado"] = history.apply(criar_target_handicap_discreto_calibrado, axis=1)

    # Features principais (3D simplificado)
    feats = ["Quadrant_Dist_3D", "Quadrant_Separation_3D", "Magnitude_3D", "Momentum_Diff", "Momentum_Diff_MT"]
    feats = [f for f in feats if f in history.columns]
    if len(feats) < 3:
        st.error("❌ Faltam features para treino")
        return None, None

    # =============== Regressão ===============
    X, y = history[feats].fillna(0), history["Target_Handicap_Home"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model_reg = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model_reg.fit(Xs, y)
    mae = mean_absolute_error(y, model_reg.predict(Xs))
    st.success(f"✅ MAE Regressão: {mae:.3f}")

    # =============== Classificação ===============
    le = LabelEncoder()
    y_cls = le.fit_transform(history["Handicap_Categoria_Calibrado"])
    model_cls = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight="balanced")
    model_cls.fit(Xs, y_cls)

    # Aplicar aos jogos do dia
    Xt = games_today[feats].fillna(0)
    Xt_s = scaler.transform(Xt)
    games_today["Pred_Reg"] = model_reg.predict(Xt_s)
    pred_cls = model_cls.predict(Xt_s)
    probas = model_cls.predict_proba(Xt_s)
    games_today["Pred_Cls"] = le.inverse_transform(pred_cls)
    games_today["Confidencia"] = np.max(probas, axis=1)
    map_cls = {
        "STRONG_HOME": -1.5, "MODERATE_HOME": -0.75, "LIGHT_HOME": -0.25, "NEUTRAL": 0,
        "LIGHT_AWAY": 0.25, "MODERATE_AWAY": 0.75, "STRONG_AWAY": 1.5
    }
    games_today["Pred_Cls_Val"] = games_today["Pred_Cls"].map(map_cls)

    return model_reg, model_cls, scaler, le, games_today

# ============================================================
# 💎 VALUE ANALYSIS + AUDITORIA
# ============================================================

def analisar_value_e_auditar(df):
    """Calcula gaps e mostra distribuição"""
    df["Asian_Line_Decimal"] = df["Asian_Line_Decimal"].astype(float)
    df["gap_reg"] = df["Asian_Line_Decimal"] - df["Pred_Reg"]
    df["gap_cls"] = df["Asian_Line_Decimal"] - df["Pred_Cls_Val"]
    df["Home_Value_Gap"] = 0.7 * df["gap_reg"] + 0.3 * df["gap_cls"]

    def classificar(gap):
        if gap > 0.4: return "STRONG HOME VALUE"
        elif gap > 0.2: return "HOME VALUE"
        elif gap < -0.4: return "STRONG AWAY VALUE"
        elif gap < -0.2: return "AWAY VALUE"
        else: return "NO CLEAR VALUE"

    df["Recomendacao_Corrigida"] = df["Home_Value_Gap"].apply(classificar)

    st.subheader("📊 Distribuição – Mercado vs Predições")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(df["Asian_Line_Decimal"], bins=20, alpha=0.6, label="Mercado (Asian_Line_Decimal)")
    axes[0].hist(df["Pred_Reg"], bins=20, alpha=0.6, label="Predito (Regressão)")
    axes[0].set_title("Distribuição: Mercado vs Predito (Regressão)")
    axes[0].legend()

    axes[1].hist(df["Home_Value_Gap"], bins=20, color="gray", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", label="Ponto Neutro")
    axes[1].set_title("Distribuição do Home Value Gap")
    axes[1].legend()
    st.pyplot(fig)

    # Estatísticas
    st.markdown("### 📈 Estatísticas de Auditoria")
    summary = {
        "Média Handicap Mercado": df["Asian_Line_Decimal"].mean(),
        "Média Handicap Predito": df["Pred_Reg"].mean(),
        "Média Home Value Gap": df["Home_Value_Gap"].mean(),
        "% HOME Value (>+0.2)": (df["Home_Value_Gap"] > 0.2).mean(),
        "% AWAY Value (<-0.2)": (df["Home_Value_Gap"] < -0.2).mean(),
    }
    st.dataframe(pd.DataFrame(summary, index=["Auditoria"]).T)

    return df

# ============================================================
# 🚀 EXECUÇÃO STREAMLIT
# ============================================================

def main():
    st.info("📂 Selecione o arquivo CSV para análise")
    uploaded = st.file_uploader("Carregar histórico de jogos", type=["csv"])
    if uploaded is None:
        st.stop()

    df = pd.read_csv(uploaded)
    st.success(f"✅ {len(df)} jogos carregados")

    if st.button("🚀 Executar Análise Calibrada (HOME Perspective)"):
        model_reg, model_cls, scaler, le, df_out = treinar_modelo(df, df)
        if df_out is not None:
            resultado = analisar_value_e_auditar(df_out)
            st.markdown("### ✅ Amostra das Recomendações Corrigidas")
            st.dataframe(resultado[["League","Home","Away","Asian_Line_Decimal","Pred_Reg","Pred_Cls_Val","Home_Value_Gap","Recomendacao_Corrigida"]].head(20))

if __name__ == "__main__":
    main()
