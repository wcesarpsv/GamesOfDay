# ============================================================
# 🎯 Análise 3D de 16 Quadrantes – Versão 2 (Z-Score + Handicap Inteligente)
# ============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
import math
from datetime import datetime
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# ⚙️ CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------
st.set_page_config(page_title="Análise 3D de 16 Quadrantes – V2", layout="wide")
st.title("🎯 Análise 3D – Versão 2 (Z-Score + Handicap Inteligente)")

PAGE_PREFIX = "QuadrantesML_3D_V2"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "coppa", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# 🧩 LEITURA DOS DADOS
# ------------------------------------------------------------
st.sidebar.header("📂 Configurações de Análise")

files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
selected_file = st.sidebar.selectbox("Selecione o arquivo de jogos:", files)

if selected_file:
    df = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))

    st.write(f"📄 **Arquivo carregado:** `{selected_file}`")
    st.markdown("---")

    # ------------------------------------------------------------
    # 🧠 FILTROS E AJUSTES INICIAIS
    # ------------------------------------------------------------
    df = df[~df["League"].str.lower().str.contains("|".join(EXCLUDED_LEAGUE_KEYWORDS))]
    leagues = sorted(df["League"].unique())
    selected_leagues = st.sidebar.multiselect("Filtrar ligas:", leagues, default=leagues)
    df = df[df["League"].isin(selected_leagues)]

    st.subheader("📊 Confrontos Carregados")
    st.dataframe(df[["League", "Home", "Away", "Asian_Line_Decimal"]].head(10), use_container_width=True)

    # ============================================================
    # 🧮 1️⃣ CÁLCULO DAS VARIÁVEIS DE DESEQUILÍBRIO
    # ============================================================
    st.markdown("### ⚙️ Cálculos Vetoriais e Desequilíbrios")

    df["Delta_M"] = df["M_H"] - df["M_A"]       # força estrutural
    df["Delta_MT"] = df["MT_H"] - df["MT_A"]    # forma recente
    df["Quadrant_Angle"] = np.degrees(np.arctan2(df["Delta_MT"], df["Delta_M"]))
    df["Quadrant_Dist"] = np.sqrt(df["Delta_M"]**2 + df["Delta_MT"]**2)

    # Classificação do tipo de desequilíbrio
    def classify_desequilibrio(angle, dist):
        if abs(angle) > 60:
            return "Forma-Recente"
        elif abs(angle) < 30:
            return "Força-Liga"
        elif 30 <= abs(angle) <= 60 and dist > 0.5:
            return "Consistente"
        else:
            return "Equilibrado"

    df["Tipo_Desequilibrio"] = df.apply(
        lambda x: classify_desequilibrio(x["Quadrant_Angle"], x["Quadrant_Dist"]), axis=1
    )

    # ------------------------------------------------------------
    # 🔍 LADO PROVÁVEL (quem tende a cobrir o handicap)
    # ------------------------------------------------------------
    def predict_side(row):
        line = row["Asian_Line_Decimal"]
        dM, dMT = row["Delta_M"], row["Delta_MT"]
        if line < 0:
            if dMT < 0:
                return "Away"
            elif dM > 0 and dMT > 0:
                return "Home"
            else:
                return "Equilibrado"
        elif line > 0:
            if dMT > 0:
                return "Away"
            else:
                return "Home"
        else:
            return "Equilibrado"

    df["Valor_Sugerido"] = df.apply(predict_side, axis=1)

    # ------------------------------------------------------------
    # 🔢 CLASSIFICAÇÃO DE CONFIANÇA
    # ------------------------------------------------------------
    def classify_confidence(row):
        if row["Quadrant_Dist"] >= 1.2:
            return "Alta"
        elif row["Quadrant_Dist"] >= 0.6:
            return "Moderada"
        else:
            return "Baixa"

    df["Confiança_Modelo"] = df.apply(classify_confidence, axis=1)

    # ------------------------------------------------------------
    # 📈 MÉTRICA CONTÍNUA PARA ML
    # ------------------------------------------------------------
    df["Cover_Tendency"] = (
        (df["Delta_M"] * np.sign(-df["Asian_Line_Decimal"])) +
        (df["Delta_MT"] * np.sign(-df["Asian_Line_Decimal"]))
    )

    # ============================================================
    # 🧭 2️⃣ GRÁFICO – VETORES DOS CONFRONTOS
    # ============================================================
    st.markdown("### 📈 Vetores dos Confrontos (M × MT)")

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in df.iterrows():
        x_home, y_home = row["M_H"], row["MT_H"]
        x_away, y_away = row["M_A"], row["MT_A"]

        # Vetor Home → Away
        ax.arrow(
            x_home, y_home,
            x_away - x_home, y_away - y_home,
            head_width=0.07, length_includes_head=True,
            color="gray", alpha=0.4
        )
        ax.text(x_home, y_home, row["Home"], color="blue", fontsize=8, weight="bold")
        ax.text(x_away, y_away, row["Away"], color="orange", fontsize=8, weight="bold")

    # Linhas dos eixos
    ax.axhline(0, color="green", linestyle="--", alpha=0.4)
    ax.axvline(0, color="green", linestyle="--", alpha=0.4)
    ax.set_xlabel("M (z-score da Liga)")
    ax.set_ylabel("MT (z-score sobre o próprio padrão)")
    ax.set_title("Mapa Vetorial – Força x Forma")
    st.pyplot(fig, use_container_width=True)

    # ============================================================
    # 🧩 3️⃣ TABELA DE ANÁLISE ESTRATÉGICA
    # ============================================================
    st.markdown("### 🧩 Tabela Estratégica – Z-Scores + Handicap")

    cols_show = [
        "League", "Home", "Away", "Asian_Line_Decimal",
        "Tipo_Desequilibrio", "Valor_Sugerido",
        "Confiança_Modelo", "Cover_Tendency"
    ]

    st.dataframe(
        df[cols_show].style
        .apply(lambda s: ["background-color: #e0ffe0" if v=="Home" else
                          "background-color: #ffe0e0" if v=="Away" else ""
                          for v in s], subset=["Valor_Sugerido"])
        .highlight_max(subset=["Cover_Tendency"], color="#c1f0c1")
        .highlight_min(subset=["Cover_Tendency"], color="#f0c1c1"),
        use_container_width=True
    )

    # ============================================================
    # 🏆 4️⃣ RANKING AUTOMÁTICO DE OPORTUNIDADES
    # ============================================================
    st.markdown("### 🏆 Ranking de Oportunidades (Classificação Automática)")

    resumo = (
        df.groupby(["Valor_Sugerido", "Tipo_Desequilibrio", "Confiança_Modelo"])
          .size().reset_index(name="Jogos")
          .sort_values("Jogos", ascending=False)
    )
    st.dataframe(resumo, use_container_width=True)

    # ============================================================
    # 📘 INFORMAÇÕES
    # ============================================================
    st.info("""
    **Lógica utilizada:**
    - `Delta_M` → força relativa na liga  
    - `Delta_MT` → forma atual comparada ao padrão próprio  
    - `Tipo_Desequilibrio` → Forma-Recente, Força-Liga, Consistente, Equilibrado  
    - `Valor_Sugerido` → lado com maior probabilidade de cobrir o handicap  
    - `Cover_Tendency` → métrica contínua (positivo = Home, negativo = Away)
    """)

else:
    st.warning("📂 Nenhum arquivo encontrado na pasta GamesDay.")
