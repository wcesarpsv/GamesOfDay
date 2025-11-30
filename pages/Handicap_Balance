# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# ========================= GET HANDICAP V1 =========================
def main_handicap_v1():
    st.set_page_config(page_title="GetHandicap V1 - Handicap-Specific Analysis", layout="wide")
    st.title("🎯 GetHandicap V1 - Análise por Handicap Específico (Modo Híbrido AIL)")

    # Configurações
    GAMES_FOLDER = "GamesDay"
    EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

    # ============================================================
    # 🔧 FUNÇÕES AUXILIARES (do código original)
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
        Converte a linha original para a perspectiva do HOME.
        Ex: '+0.5' (Away) -> -0.5 (Home)
        """
        if pd.isna(value):
            return np.nan
        s = str(value).strip()
        if "/" not in s:
            try:
                num = float(s)
                return -num
            except:
                return np.nan
        try:
            parts = [float(p) for p in s.replace("+", "").replace("-", "").split("/")]
            avg = np.mean(parts)
            sign = -1 if s.startswith("-") else 1
            result = sign * avg
            return -result
        except:
            return np.nan

    def calcular_zscores_detalhados(df):
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

    def odds_to_probs(odd_h, odd_d, odd_a):
        try:
            odd_h = float(odd_h)
            odd_d = float(odd_d)
            odd_a = float(odd_a)
            if odd_h <= 0 or odd_d <= 0 or odd_a <= 0:
                return 0.33, 0.33, 0.33
            inv_sum = (1 / odd_h) + (1 / odd_d) + (1 / odd_a)
            return (1 / odd_h) / inv_sum, (1 / odd_d) / inv_sum, (1 / odd_a) / inv_sum
        except:
            return 0.33, 0.33, 0.33

    def wg_home(row):
        gf = row.get('Goals_H_FT', 0)
        ga = row.get('Goals_A_FT', 0)
        p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 2.5), row.get('Odd_D', 3.0), row.get('Odd_A', 2.5))
        return (gf * (1 - p_h)) - (ga * p_h)

    def wg_away(row):
        gf = row.get('Goals_A_FT', 0)
        ga = row.get('Goals_H_FT', 0)
        p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 2.5), row.get('Odd_D', 3.0), row.get('Odd_A', 2.5))
        return (gf * (1 - p_a)) - (ga * p_a)

    def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required_cols = ['Home', 'Away', 'Date', 'Goals_H_FT', 'Goals_A_FT', 'Odd_H', 'Odd_D', 'Odd_A']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.warning(f"⚠️ Colunas ausentes para WG (usar 0): {missing}")
            df['WG_Home'] = 0.0
            df['WG_Away'] = 0.0
            df['WG_Home_Team'] = 0.0
            df['WG_Away_Team'] = 0.0
            df['WG_Diff'] = 0.0
            return df

        df['WG_Home'] = df.apply(wg_home, axis=1)
        df['WG_Away'] = df.apply(wg_away, axis=1)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

        df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())

        df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']

        st.success("🔥 Weighted Goals (WG_Home_Team / WG_Away_Team / WG_Diff) calculados!")
        return df

    def criar_targets_cobertura(df: pd.DataFrame) -> pd.DataFrame:
        hist = df.dropna(subset=['Goals_H_FT', 'Goals_A_FT', 'Asian_Line_Decimal']).copy()
        if hist.empty:
            return hist
        margin = hist['Goals_H_FT'] - hist['Goals_A_FT']
        adj = margin + hist['Asian_Line_Decimal']
        hist['Cover_Home'] = (adj > 0).astype(int)
        hist['Cover_Away'] = (adj < 0).astype(int)
        return hist

    # ============================================================
    # 🆕 FUNÇÕES ESPECÍFICAS GET HANDICAP V1
    # ============================================================

    def segmentar_por_handicap(df: pd.DataFrame, handicap_alvo: float, tolerancia: float = 0.25) -> pd.DataFrame:
        """
        Filtra jogos com handicap (Home) próximo ao alvo
        Ex: handicap_alvo = -0.5, tolerancia=0.25 → pega -0.75, -0.5, -0.25
        """
        if df.empty or 'Asian_Line_Decimal' not in df.columns:
            return pd.DataFrame()

        mask = df['Asian_Line_Decimal'].sub(handicap_alvo).abs() <= tolerancia
        df_segmento = df[mask].copy()

        st.info(f"🎯 Handicap {handicap_alvo}: {len(df_segmento)} jogos (tolerância: ±{tolerancia})")
        return df_segmento

    def analisar_patterns_handicap(df_segmento: pd.DataFrame, handicap_nome: str, min_amostras: int = 30):
        """
        Analisa quais features correlacionam com sucesso para um handicap específico
        """
        if len(df_segmento) < min_amostras:
            st.warning(f"⚠️ Amostras insuficientes para {handicap_nome}: {len(df_segmento)} < {min_amostras}")
            return None, None

        st.markdown(f"### 📊 Padrões - {handicap_nome}")

        # Features para análise
        features_analise = [
            'WG_Home_Team', 'WG_Away_Team', 'WG_Diff',
            'M_H', 'M_A', 'MT_H', 'MT_A',
            'Aggression_Home', 'Aggression_Away',
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Vector_Sign', 'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT',
            'Cluster3D_Label'
        ]

        # Garantir que as features existem
        features_disponiveis = [f for f in features_analise if f in df_segmento.columns]

        if 'Cover_Home' not in df_segmento.columns:
            st.error("❌ Target Cover_Home não encontrado no segmento")
            return None, None

        # Calcular correlações
        correlations_home = {}
        correlations_away = {}

        for feature in features_disponiveis:
            corr_home = df_segmento[feature].corr(df_segmento['Cover_Home'])
            corr_away = df_segmento[feature].corr(df_segmento['Cover_Away'])
            correlations_home[feature] = corr_home
            correlations_away[feature] = corr_away

        # Ordenar por correlação absoluta
        home_sorted = sorted(correlations_home.items(), key=lambda x: abs(x[1]), reverse=True)
        away_sorted = sorted(correlations_away.items(), key=lambda x: abs(x[1]), reverse=True)

        # Top 5 features para cada lado
        top_home = home_sorted[:5]
        top_away = away_sorted[:5]

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🏠 HOME Cover Patterns")
            for feature, corr in top_home:
                corr_color = "🟢" if corr > 0.1 else "🟡" if corr > 0.05 else "🔴"
                st.write(f"{corr_color} **{feature}**: {corr:.3f}")

        with col2:
            st.subheader(f"✈️ AWAY Cover Patterns")
            for feature, corr in top_away:
                corr_color = "🟢" if corr > 0.1 else "🟡" if corr > 0.05 else "🔴"
                st.write(f"{corr_color} **{feature}**: {corr:.3f}")

        return top_home, top_away

    def criar_heatmap_handicap_features(history: pd.DataFrame):
        """
        Cria heatmap: Handicap vs Features vs Performance
        """
        st.markdown("### 🔥 Heatmap - Correlações por Handicap")

        handicaps = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        features_analise = [
            'WG_Home_Team', 'WG_Away_Team', 'M_H', 'M_A',
            'Aggression_Home', 'Aggression_Away', 'Cluster3D_Label'
        ]

        # Filtrar features disponíveis
        features_disponiveis = [f for f in features_analise if f in history.columns]

        results = []

        for handicap in handicaps:
            df_seg = segmentar_por_handicap(history, handicap, 0.15)
            if len(df_seg) > 20:  # mínimo de amostras
                for feature in features_disponiveis:
                    if feature in df_seg.columns and 'Cover_Home' in df_seg.columns:
                        corr = df_seg[feature].corr(df_seg['Cover_Home'])
                        results.append({
                            'Handicap': handicap,
                            'Feature': feature,
                            'Correlacao': corr,
                            'Amostras': len(df_seg)
                        })

        if not results:
            st.warning("❌ Dados insuficientes para heatmap")
            return None

        df_heatmap = pd.DataFrame(results)

        # Pivot para heatmap
        heatmap_data = df_heatmap.pivot_table(
            index='Feature',
            columns='Handicap',
            values='Correlacao',
            aggfunc='mean'
        ).fillna(0)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt='.3f',
            ax=ax
        )
        ax.set_title('Correlação: Features vs Cover_Home por Handicap\n(Valores Positivos = Favorecem HOME)')
        plt.tight_layout()

        st.pyplot(fig)
        return df_heatmap

    # ===================== SPLIT TEMPORAL =====================
    def split_temporal(df: pd.DataFrame, test_size: float = 0.2):
        """
        Faz split temporal (train = jogos antigos, val = jogos recentes)
        Requer coluna 'Date' já convertida para datetime.
        """
        df = df.copy()
        if 'Date' not in df.columns:
            st.warning("⚠️ Coluna 'Date' não encontrada para split temporal. Usando tudo como treino.")
            df['__split'] = 'train'
            return df, df, df

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')

        n = len(df)
        if n < 10:
            st.warning(f"⚠️ Amostras insuficientes ({n}) para split temporal. Usando tudo como treino.")
            df['__split'] = 'train'
            return df, df, df

        split_idx = int(n * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        train_df['__split'] = 'train'
        val_df['__split'] = 'val'

        full = pd.concat([train_df, val_df], axis=0)
        return full, train_df, val_df

    def treinar_modelo_handicap_especifico(history: pd.DataFrame, handicap_alvo: float, features: list):
        """
        Treina modelo RandomForest específico para um handicap
        com validação temporal (time-safe).
        """
        df_segmento = segmentar_por_handicap(history, handicap_alvo, 0.25)

        if len(df_segmento) < 50:
            st.warning(f"⚠️ Amostras insuficientes para modelo {handicap_alvo}: {len(df_segmento)}")
            return None, None

        # Garantir targets
        if 'Cover_Home' not in df_segmento.columns or 'Cover_Away' not in df_segmento.columns:
            df_segmento = criar_targets_cobertura(df_segmento)

        # Garantir coluna de data para split temporal
        if 'Date' not in df_segmento.columns:
            st.warning(f"⚠️ Coluna 'Date' ausente no segmento {handicap_alvo}. Sem validação temporal.")
            df_segmento['Date'] = pd.to_datetime('1900-01-01')

        # Aplicar split temporal
        df_segmento, df_train, df_val = split_temporal(df_segmento, test_size=0.2)

        # Features disponíveis
        features_disponiveis = [f for f in features if f in df_segmento.columns]

        if not features_disponiveis:
            st.error(f"❌ Nenhuma feature disponível para treino no handicap {handicap_alvo}")
            return None, None

        X_train = df_train[features_disponiveis].copy()
        y_train = df_train['Cover_Home'].astype(int)

        X_val = df_val[features_disponiveis].copy()
        y_val = df_val['Cover_Home'].astype(int)

        X_train = clean_features_for_training(X_train)
        X_val = clean_features_for_training(X_val)

        # Treinar modelo
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        modelo.fit(X_train, y_train)

        # Métricas
        y_pred_train = modelo.predict(X_train)
        y_pred_val = modelo.predict(X_val)

        acc_train = (y_pred_train == y_train).mean() if len(y_train) > 0 else np.nan
        acc_val = (y_pred_val == y_val).mean() if len(y_val) > 0 else np.nan

        st.success(
            f"✅ Modelo Handicap {handicap_alvo} "
            f"- Train Acc: {acc_train:.3f} (n={len(y_train)}) | "
            f"Val Acc (time-safe): {acc_val:.3f} (n={len(y_val)})"
        )

        return modelo, features_disponiveis

    # ===================== LÓGICA HÍBRIDA AIL (ML + WG) =====================

    def gerar_ail_value_score_hibrido(df: pd.DataFrame, profile: str = "moderado") -> pd.DataFrame:
        """
        Gera:
          - AIL_Value_Score (score contínuo)
          - AIL_Pick (texto)
          - AIL_Confidence (ALTA/MEDIA/BAIXA/PASS)

        Modo 'moderado':
          - thresholds ajustados para equilíbrio risco/retorno
        """
        df = df.copy()

        # Thresholds para perfil moderado
        thr_signal = 0.15     # mínimo para virar pick
        thr_conf_alta = 0.30  # score forte
        thr_conf_media = 0.15

        # Componentes
        ml_component = (df['P_Cover_Home_Especifico'] - 0.5) * 2
        ml_component = ml_component.fillna(0)  # onde não há modelo

        wg_component = df.get('WG_Diff', 0).fillna(0)
        mom_component = df.get('Momentum_Diff', 0).fillna(0)
        quad_component = df.get('Quadrant_Dist_3D', 0).fillna(0)

        # Combinação híbrida (moderado)
        df['AIL_Value_Score'] = (
            0.6 * ml_component +
            0.3 * wg_component +
            0.1 * mom_component
        )

        # Geração de picks
        picks = []
        confs = []
        sides = []
        hc_display_list = []

        for _, row in df.iterrows():
            score = row['AIL_Value_Score']
            hc_home = row.get('Asian_Line_Decimal', 0.0)
            asian_raw = row.get('Asian_Line', "")

            # Sem handicap útil
            if pd.isna(hc_home) or abs(hc_home) < 0.25:
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            mag = abs(score)

            if mag < thr_signal:
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            # Direção: score > 0 favorece HOME cobrir
            if score > 0:
                side = "HOME"
                hc_side = hc_home  # handicap do mandante
            else:
                side = "AWAY"
                hc_side = -hc_home  # handicap do visitante

            # Confiança
            if mag >= thr_conf_alta:
                conf = "ALTA"
            elif mag >= thr_conf_media:
                conf = "MEDIA"
            else:
                conf = "BAIXA"

            # Display do handicap (formato +0.5 / -0.75)
            if pd.isna(hc_side):
                hc_str = ""
            else:
                hc_str = f"{hc_side:+.2f}".rstrip("0").rstrip(".")

            label = "🏠 HOME" if side == "HOME" else "✈️ AWAY"
            pick_text = f"{label} {hc_str}"

            picks.append(pick_text)
            confs.append(conf)
            sides.append(side)
            hc_display_list.append(hc_str)

        df['AIL_Pick'] = picks
        df['AIL_Confidence'] = confs
        df['AIL_Pick_Side'] = sides
        df['AIL_Handicap_Display'] = hc_display_list

        return df

    # ===================== APLICAR MODELOS NOS JOGOS DE HOJE =====================

    def aplicar_modelos_handicap(games_today: pd.DataFrame, modelos_handicap: dict) -> pd.DataFrame:
        """
        Aplica todos os modelos de handicap específico nos jogos de hoje.
        Retorna um DF com TODOS os jogos_today + colunas de ML onde existirem modelos.
        """
        st.markdown("### 🎯 Previsões por Handicap Específico (ML + AIL)")

        df_all = games_today.copy()

        # Inicializar colunas de modelo
        df_all['P_Cover_Home_Especifico'] = np.nan
        df_all['Value_Gap_Especifico'] = np.nan
        df_all['Handicap_Modelo'] = np.nan
        df_all['Modelo_Confianca'] = ""

        for handicap, model_pack in modelos_handicap.items():
            modelo, features = model_pack
            if modelo is None:
                continue

            jogos_alvo = segmentar_por_handicap(df_all, handicap, 0.25)
            if len(jogos_alvo) == 0:
                continue

            # Features podem não existir em todos os jogos (segurança)
            features_validas = [f for f in features if f in df_all.columns]
            if not features_validas:
                st.warning(f"⚠️ Nenhuma feature válida encontrada nos jogos de hoje para handicap {handicap}")
                continue

            X_today = jogos_alvo[features_validas].copy()
            X_today = clean_features_for_training(X_today)

            probas = modelo.predict_proba(X_today)[:, 1]  # P(Cover_Home)

            value_gap = probas - 0.5
            conf = np.where(np.abs(value_gap) > 0.15, "ALTA", "MEDIA")

            df_all.loc[jogos_alvo.index, 'P_Cover_Home_Especifico'] = probas
            df_all.loc[jogos_alvo.index, 'Value_Gap_Especifico'] = value_gap
            df_all.loc[jogos_alvo.index, 'Handicap_Modelo'] = handicap
            df_all.loc[jogos_alvo.index, 'Modelo_Confianca'] = conf

        return df_all

    # ============================================================
    # 🚀 EXECUÇÃO PRINCIPAL GET HANDICAP V1
    # ============================================================

    st.info("📂 Carregando dados para Análise GetHandicap V1...")

    # Carregar dados
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
    if not files:
        st.warning("No CSV files found in GamesDay folder.")
        return

    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options) - 1)

    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)

        # Aplicar filtros e pré-processamento
        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        # Converter Asian Line para decimal (perspectiva HOME)
        history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
        games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

        # Calcular features (Z-scores, WG, 3D, etc.)
        history = calcular_zscores_detalhados(history)
        history = adicionar_weighted_goals(history)
        history = aplicar_clusterizacao_3d(calcular_distancias_3d(history))

        games_today = calcular_zscores_detalhados(games_today)
        games_today = aplicar_clusterizacao_3d(calcular_distancias_3d(games_today))

        # ============================================
        # 🆕 Merge WG do histórico → games_today
        # ============================================
        wg_cols = ['Home', 'WG_Home_Team', 'WG_Diff']
        wg_home = history[['Home', 'WG_Home_Team']].dropna().drop_duplicates(subset=['Home'], keep='last')
        wg_away = history[['Away', 'WG_Away_Team']].dropna().drop_duplicates(subset=['Away'], keep='last')
        
        # Merge para trazer WG histórico
        games_today = games_today.merge(wg_home.rename(columns={'Home':'Team'}),
                                        left_on='Home', right_on='Team', how='left').drop(columns=['Team'])
        
        games_today = games_today.merge(wg_away.rename(columns={'Away':'Team'}),
                                        left_on='Away', right_on='Team', how='left').drop(columns=['Team'])
        
        # Calcular WG_Diff para jogos de hoje
        games_today['WG_Diff'] = games_today['WG_Home_Team'] - games_today['WG_Away_Team']
        
        # Substituir NaN por 0 (caso time novo)
        games_today[['WG_Home_Team','WG_Away_Team','WG_Diff']] = \
            games_today[['WG_Home_Team','WG_Away_Team','WG_Diff']].fillna(0)
        
        st.success("🔥 Features WG trazidas do histórico para os jogos do dia!")


        # Criar targets
        history = criar_targets_cobertura(history)

        return games_today, history

    games_today, history = load_cached_data(selected_file)

    if history.empty or games_today.empty:
        st.error("❌ Dados insuficientes para análise")
        return

    # ============================================================
    # 🎯 INTERFACE GET HANDICAP V1
    # ============================================================

    st.sidebar.markdown("## 🎯 GetHandicap V1 - Configurações")

    analise_modo = st.sidebar.selectbox(
        "Modo de Análise:",
        ["📊 Análise Exploratória", "🤖 Modelos Específicos", "🎯 Previsões Hoje"]
    )

    if analise_modo == "📊 Análise Exploratória":
        st.header("📊 Análise Exploratória por Handicap")

        handicap_selecionado = st.selectbox(
            "Selecione o Handicap para Análise (Home):",
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            index=2  # Default -0.5
        )

        if st.button("🔍 Analisar Patterns", type="primary"):
            df_segmento = segmentar_por_handicap(history, handicap_selecionado, 0.25)
            top_home, top_away = analisar_patterns_handicap(df_segmento, f"Handicap {handicap_selecionado}")

        # Heatmap geral
        st.markdown("---")
        criar_heatmap_handicap_features(history)

    elif analise_modo == "🤖 Modelos Específicos":
        st.header("🤖 Treinar Modelos por Handicap (Time-Safe)")

        handicaps_treinar = st.multiselect(
            "Handicaps para Treinar Modelos (Home):",
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            default=[-0.5, 0.0, 0.5]
        )

        features_base = [
            'WG_Home_Team', 'WG_Away_Team', 'WG_Diff',
            'M_H', 'M_A', 'MT_H', 'MT_A',
            'Aggression_Home', 'Aggression_Away',
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Vector_Sign', 'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT',
            'Cluster3D_Label'
        ]

        modelos_treinados = {}

        if st.button("🚀 Treinar Modelos Específicos", type="primary"):
            for handicap in handicaps_treinar:
                with st.spinner(f"Treinando modelo para handicap {handicap}..."):
                    modelo, features = treinar_modelo_handicap_especifico(history, handicap, features_base)
                    modelos_treinados[handicap] = (modelo, features)

            st.session_state['modelos_handicap'] = modelos_treinados
            st.success("✅ Todos os modelos específicos treinados (com validação temporal)!")

    elif analise_modo == "🎯 Previsões Hoje":
        st.header("🎯 Previsões para Jogos de Hoje (Modo Híbrido AIL)")

        if 'modelos_handicap' not in st.session_state:
            st.warning("⚠️ Treine os modelos específicos primeiro na aba 'Modelos Específicos'")
        else:
            df_all = aplicar_modelos_handicap(games_today, st.session_state['modelos_handicap'])

            # Gerar AIL híbrido (ML + WG)
            df_all = gerar_ail_value_score_hibrido(df_all, profile="moderado")

            # Somente picks (tirar PASS)
            df_picks = df_all[df_all['AIL_Pick'] != "⚪ PASS"].copy()

            if df_picks.empty:
                st.warning("⚠️ Nenhuma pick gerada com os thresholds atuais.")
                return

            # Filtros interativos
            st.sidebar.markdown("## 🔍 Filtros Previsões")

            min_score = st.sidebar.slider("Score mínimo (AIL_Value_Score):", 0.0, 0.6, 0.15, 0.05)
            conf_sel = st.sidebar.multiselect(
                "Confiança AIL:",
                ['ALTA', 'MEDIA', 'BAIXA'],
                default=['ALTA', 'MEDIA']
            )

            df_filtrado = df_picks[
                (df_picks['AIL_Value_Score'].abs() >= min_score) &
                (df_picks['AIL_Confidence'].isin(conf_sel))
            ].copy()

            # Ordenar por valor absoluto do score
            df_filtrado = df_filtrado.sort_values('AIL_Value_Score', key=lambda s: s.abs(), ascending=False)

            st.metric("🎯 Apostas Sugeridas", len(df_filtrado))
            st.dataframe(
                df_filtrado[[
                    'League', 'Home', 'Away',
                    'Asian_Line', 'Asian_Line_Decimal',
                    'AIL_Pick', 'AIL_Confidence', 'AIL_Value_Score',
                    'P_Cover_Home_Especifico', 'Value_Gap_Especifico',
                    'WG_Home_Team', 'WG_Away_Team', 'WG_Diff'
                ]],
                use_container_width=True
            )

# ============================================================
# 🚀 EXECUTAR
# ============================================================
if __name__ == "__main__":
    main_handicap_v1()
