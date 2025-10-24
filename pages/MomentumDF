########################################
#### 🧮 Recalcular Z-Scores (M_ por Liga | MT_ por Time)
########################################
import streamlit as st
import pandas as pd
import numpy as np
import os

st.markdown("## 🧮 Recalcular Z-Scores (M_ por Liga | MT_ por Time)")
st.info("""
Esta rotina:
1️⃣ Lê todos os arquivos da pasta **GamesDay**  
2️⃣ Remove duplicados por **League + Home + Away + Placar Final**  
3️⃣ Recalcula:
- `M_H`, `M_A` → Z-score do time **perante a média da liga**
- `MT_H`, `MT_A` → Z-score do time **perante o próprio histórico**  
4️⃣ Garante **ordem cronológica** e **zero leakage**  
5️⃣ Salva a base como **Base_GamesDay_zscore_recalculada.csv**
""")

# Caminho base
GAMES_FOLDER = "GamesDay"
OUTPUT_FILE = "Base_GamesDay_zscore_recalculada.csv"

if st.button("🚀 Recalcular agora"):
    try:
        # ------------------------------
        # Carregar e consolidar todos os arquivos
        # ------------------------------
        all_files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
        if not all_files:
            st.error("❌ Nenhum arquivo CSV encontrado na pasta GamesDay.")
            st.stop()

        dfs = []
        for f in all_files:
            try:
                df_tmp = pd.read_csv(os.path.join(GAMES_FOLDER, f))
                df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('^Unnamed')]
                dfs.append(df_tmp)
            except Exception as e:
                st.warning(f"Erro ao ler {f}: {e}")
                continue

        df = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Total de registros carregados: {len(df):,}")

        # ------------------------------
        # Limpeza e ordenação
        # ------------------------------
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(['League', 'Date'])
        df = df.drop_duplicates(
            subset=["League", "Home", "Away", "Goals_H_FT", "Goals_A_FT"],
            keep="first"
        ).reset_index(drop=True)
        st.info(f"🧹 Após remoção de duplicatas: {len(df):,} jogos únicos.")

        # ------------------------------
        # Função genérica de z-score temporal
        # ------------------------------
        def compute_zscore_temporal(df, col, group_cols, new_col):
            df = df.copy()
            df[new_col] = np.nan
            for keys, sub in df.groupby(group_cols):
                sub = sub.sort_values('Date')
                mean_roll = sub[col].expanding().mean().shift(1)
                std_roll = sub[col].expanding().std(ddof=0).shift(1)
                df.loc[sub.index, new_col] = (sub[col] - mean_roll) / std_roll
            return df

        # Garantir colunas necessárias
        if 'HandScore_Home' not in df.columns or 'HandScore_Away' not in df.columns:
            st.error("❌ Colunas HandScore_Home e HandScore_Away não encontradas.")
            st.stop()

        # ------------------------------
        # Cálculos
        # ------------------------------
        st.info("📈 Calculando z-scores por liga (M_)...")
        df = compute_zscore_temporal(df, 'HandScore_Home', ['League'], 'M_H')
        df = compute_zscore_temporal(df, 'HandScore_Away', ['League'], 'M_A')

        st.info("📊 Calculando z-scores por time dentro da liga (MT_)...")
        df = compute_zscore_temporal(df, 'HandScore_Home', ['League', 'Home'], 'MT_H')
        df = compute_zscore_temporal(df, 'HandScore_Away', ['League', 'Away'], 'MT_A')

        # ------------------------------
        # Salvar resultado consolidado
        # ------------------------------
        df.to_csv(OUTPUT_FILE, index=False)
        st.success(f"💾 Base recalculada salva com sucesso como `{OUTPUT_FILE}`")

        # ------------------------------
        # Visualização final
        # ------------------------------
        preview_cols = [
            'Date', 'League', 'Home', 'Away',
            'HandScore_Home', 'HandScore_Away',
            'M_H', 'M_A', 'MT_H', 'MT_A'
        ]
        preview_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(df[preview_cols].tail(20), use_container_width=True)

        # ------------------------------
        # Botão para download direto
        # ------------------------------
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar Base Recalculada (CSV)",
            data=csv_bytes,
            file_name=OUTPUT_FILE,
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Erro ao recalcular: {e}")
