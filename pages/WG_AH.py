# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ========================= GET HANDICAP V1 =========================
def main_handicap_v1():
    st.set_page_config(page_title="GetHandicap V1 - Handicap-Specific Analysis", layout="wide")
    st.title("🎯 GetHandicap V1 - Análise por Handicap Específico (Modo Híbrido AIL)")

    # ---------------- Configurações gerais ----------------
    GAMES_FOLDER = "GamesDay"
    EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

    # ============================================================
    # 🔧 FUNÇÕES AUXILIARES
    # ============================================================
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
                df_tmp = pd.read_csv(os.path.join(folder, f))
                dfs.append(preprocess_df(df_tmp))
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
        Converte a linha original (perspectiva AWAY) para decimal na perspectiva do HOME.
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

    # ============================================================
    # 📊 Z-SCORES (M_H, M_A, MT_H, MT_A)
    # ============================================================
    def calcular_zscores_detalhados(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1) Z-score por liga
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
        else:
            df['M_H'] = 0
            df['M_A'] = 0

        # 2) Z-score por time
        if 'Home' in df.columns and 'Away' in df.columns and 'HandScore_Home' in df.columns and 'HandScore_Away' in df.columns:
            home_team_stats = df.groupby('Home').agg({'HandScore_Home': ['mean', 'std']}).round(3)
            home_team_stats.columns = ['HT_mean', 'HT_std']

            away_team_stats = df.groupby('Away').agg({'HandScore_Away': ['mean', 'std']}).round(3)
            away_team_stats.columns = ['AT_mean', 'AT_std']

            home_team_stats['HT_std'] = home_team_stats['HT_std'].replace(0, 1)
            away_team_stats['AT_std'] = away_team_stats['AT_std'].replace(0, 1)

            df = df.merge(home_team_stats, left_on='Home', right_index=True, how='left')
            df = df.merge(away_team_stats, left_on='Away', right_index=True, how='left')

            df['MT_H'] = (df['HandScore_Home'] - df['HT_mean']) / df['HT_std']
            df['MT_A'] = (df['HandScore_Away'] - df['AT_mean']) / df['AT_std']

            df['MT_H'] = np.clip(df['MT_H'], -5, 5)
            df['MT_A'] = np.clip(df['MT_A'], -5, 5)

            df = df.drop([
                'HS_H_mean', 'HS_H_std', 'HS_A_mean', 'HS_A_std',
                'HT_mean', 'HT_std', 'AT_mean', 'AT_std'
            ], axis=1, errors='ignore')
        else:
            df['MT_H'] = 0
            df['MT_A'] = 0

        return df

    # ============================================================
    # ⚽ WEIGHTED GOALS + PROBABILIDADES
    # ============================================================
    def odds_to_probs(odd_h, odd_d, odd_a):
        """
        Odds 1X2 -> probabilidades "fair" (removendo vig por normalização simples)
        """
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

    def odds_to_market_probs(row):
        """
        Apenas HOME vs AWAY, ignorando empate (útil p/ WG)
        """
        try:
            odd_h = float(row['Odd_H'])
            odd_a = float(row['Odd_A'])
            if odd_h <= 0 or odd_a <= 0:
                return 0.50, 0.50
            inv_h = 1 / odd_h
            inv_a = 1 / odd_a
            total = inv_h + inv_a
            return inv_h / total, inv_a / total
        except:
            return 0.50, 0.50

    def wg_home(row):
        p_h, p_a = odds_to_market_probs(row)
        return (row.get('Goals_H_FT', 0) * (1 - p_h)) - (row.get('Goals_A_FT', 0) * p_h)

    def wg_away(row):
        p_h, p_a = odds_to_market_probs(row)
        return (row.get('Goals_A_FT', 0) * (1 - p_a)) - (row.get('Goals_H_FT', 0) * p_a)

    def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula WG base + WG ponderado por handicap (WG_AH) no HISTÓRICO.
        """
        df = df.copy()

        required_cols = ['Home', 'Away', 'Date', 'Goals_H_FT', 'Goals_A_FT', 'Odd_H', 'Odd_D', 'Odd_A']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.warning(f"⚠️ Colunas ausentes para Weighted Goals, usando 0: {missing}")
            df['WG_Home'] = 0.0
            df['WG_Away'] = 0.0
            df['WG_Home_Team'] = 0.0
            df['WG_Away_Team'] = 0.0
            df['WG_Diff'] = 0.0
            df['WG_Confidence'] = 0
            df['WG_AH_Home'] = 0.0
            df['WG_AH_Away'] = 0.0
            df['WG_AH_Home_Team'] = 0.0
            df['WG_AH_Away_Team'] = 0.0
            df['WG_AH_Diff'] = 0.0
            return df

        df['WG_Home'] = df.apply(wg_home, axis=1)
        df['WG_Away'] = df.apply(wg_away, axis=1)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

        # Rolling WG base
        df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']
        df['WG_Confidence'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).count())

        # WG ponderado pelo handicap (por jogo)
        if 'Asian_Line_Decimal' in df.columns:
            factor = 1 + df['Asian_Line_Decimal'].abs().fillna(0)
            df['WG_AH_Home'] = df['WG_Home'] * factor
            df['WG_AH_Away'] = df['WG_Away'] * factor
        else:
            df['WG_AH_Home'] = df['WG_Home']
            df['WG_AH_Away'] = df['WG_Away']

        # Rolling WG_AH por time
        df['WG_AH_Home_Team'] = df.groupby('Home')['WG_AH_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_AH_Away_Team'] = df.groupby('Away')['WG_AH_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_AH_Diff'] = df['WG_AH_Home_Team'] - df['WG_AH_Away_Team']

        return df

    # ============================================================
    # WG HISTÓRICO PARA JOGOS DO DIA (SEM FUTURO)
    # ============================================================
    def aplicar_weighted_goals_today(history: pd.DataFrame, games_today: pd.DataFrame) -> pd.DataFrame:
        df = games_today.copy()
        history = history.sort_values('Date')

        for col in ['WG_Home_Team', 'WG_Away_Team', 'WG_Diff', 'WG_Confidence',
                    'WG_AH_Home_Team', 'WG_AH_Away_Team', 'WG_AH_Diff']:
            if col not in df.columns:
                df[col] = np.nan

        for idx, row in df.iterrows():
            home_team = row['Home']
            away_team = row['Away']

            last_home_games = history[history['Home'] == home_team].tail(5)
            last_away_games = history[history['Away'] == away_team].tail(5)

            df.at[idx, 'WG_Home_Team'] = last_home_games['WG_Home'].mean() if not last_home_games.empty else 0
            df.at[idx, 'WG_Away_Team'] = last_away_games['WG_Away'].mean() if not last_away_games.empty else 0

            df.at[idx, 'WG_Diff'] = df.at[idx, 'WG_Home_Team'] - df.at[idx, 'WG_Away_Team']
            df.at[idx, 'WG_Confidence'] = len(last_home_games) + len(last_away_games)

            if 'WG_AH_Home' in history.columns and 'WG_AH_Away' in history.columns:
                df.at[idx, 'WG_AH_Home_Team'] = last_home_games['WG_AH_Home'].mean() if not last_home_games.empty else 0
                df.at[idx, 'WG_AH_Away_Team'] = last_away_games['WG_AH_Away'].mean() if not last_away_games.empty else 0
            else:
                df.at[idx, 'WG_AH_Home_Team'] = df.at[idx, 'WG_Home_Team']
                df.at[idx, 'WG_AH_Away_Team'] = df.at[idx, 'WG_Away_Team']

            df.at[idx, 'WG_AH_Diff'] = df.at[idx, 'WG_AH_Home_Team'] - df.at[idx, 'WG_AH_Away_Team']

        return df

    # ============================================================
    # 🎯 TARGETS: Cover_Home / Cover_Away
    # ============================================================
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
    # 🧬 DISTÂNCIAS 3D + CLUSTER
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
        df['Magnitude_3D'] = df['Quadrant_Dist_3D']
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
            st.error(f"❌ Erro no clustering 3D: {e}")
            df['Cluster3D_Label'] = 0

        return df

    # ============================================================
    # 🧼 LIMPEZA DE FEATURES
    # ============================================================
    def clean_features_for_training(X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

        for col in X_clean.columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(median_val)
                if X_clean[col].isna().any():
                    X_clean[col] = X_clean[col].fillna(0)

        for col in X_clean.columns:
            if X_clean[col].dtype in [np.float64, np.float32, float, int]:
                Q1 = X_clean[col].quantile(0.25)
                Q3 = X_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)

        X_clean = X_clean.replace([np.inf, -np.inf], 0).fillna(0)
        return X_clean

    # ============================================================
    # 🕒 SPLIT TEMPORAL
    # ============================================================
    def split_temporal(df: pd.DataFrame, test_size: float = 0.2):
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

    # ============================================================
    # 🎯 SEGMENTAÇÃO POR HANDICAP
    # ============================================================
    def segmentar_por_handicap(df: pd.DataFrame, handicap_alvo: float, tolerancia: float = 0.25) -> pd.DataFrame:
        if df.empty or 'Asian_Line_Decimal' not in df.columns:
            return pd.DataFrame()
        mask = df['Asian_Line_Decimal'].sub(handicap_alvo).abs() <= tolerancia
        df_segmento = df[mask].copy()
        st.info(f"🎯 Handicap {handicap_alvo}: {len(df_segmento)} jogos (tolerância: ±{tolerancia})")
        return df_segmento

    # ============================================================
    # 🔍 ANÁLISE DE PATTERNS POR HANDICAP + HEATMAP
    # ============================================================
    def analisar_patterns_handicap(df_segmento: pd.DataFrame, handicap_nome: str, min_amostras: int = 30):
        if len(df_segmento) < min_amostras:
            st.warning(f"⚠️ Amostras insuficientes para {handicap_nome}: {len(df_segmento)} < {min_amostras}")
            return None, None

        st.markdown(f"### 📊 Padrões - {handicap_nome}")

        features_analise = [
            'WG_Home_Team', 'WG_Away_Team', 'WG_Diff',
            'M_H', 'M_A', 'MT_H', 'MT_A',
            'Aggression_Home', 'Aggression_Away',
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
            'Vector_Sign', 'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT',
            'Cluster3D_Label'
        ]

        features_disponiveis = [f for f in features_analise if f in df_segmento.columns]

        if 'Cover_Home' not in df_segmento.columns or 'Cover_Away' not in df_segmento.columns:
            st.error("❌ Targets 'Cover_Home' ou 'Cover_Away' não encontrados.")
            return None, None

        correlations_home = {}
        correlations_away = {}

        for feature in features_disponiveis:
            corr_home = df_segmento[feature].corr(df_segmento['Cover_Home'])
            corr_away = df_segmento[feature].corr(df_segmento['Cover_Away'])
            correlations_home[feature] = corr_home
            correlations_away[feature] = corr_away

        home_sorted = sorted(correlations_home.items(), key=lambda x: abs(x[1]), reverse=True)
        away_sorted = sorted(correlations_away.items(), key=lambda x: abs(x[1]), reverse=True)

        top_home = home_sorted[:5]
        top_away = away_sorted[:5]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏠 HOME Cover Patterns")
            for feature, corr in top_home:
                corr_color = "🟢" if corr > 0.1 else "🟡" if corr > 0.05 else "🔴"
                st.write(f"{corr_color} {feature}: {corr:.3f}")
        with col2:
            st.subheader("✈️ AWAY Cover Patterns")
            for feature, corr in top_away:
                corr_color = "🟢" if corr > 0.1 else "🟡" if corr > 0.05 else "🔴"
                st.write(f"{corr_color} {feature}: {corr:.3f}")

        return top_home, top_away

    def criar_heatmap_handicap_features(history: pd.DataFrame):
        st.markdown("### 🔥 Heatmap - Correlações por Handicap")

        handicaps = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        features_analise = [
            'WG_Home_Team', 'WG_Away_Team', 'M_H', 'M_A',
            'Aggression_Home', 'Aggression_Away', 'Cluster3D_Label'
        ]
        features_disponiveis = [f for f in features_analise if f in history.columns]

        results = []
        for handicap in handicaps:
            df_seg = segmentar_por_handicap(history, handicap, 0.15)
            if len(df_seg) > 20 and 'Cover_Home' in df_seg.columns:
                for feature in features_disponiveis:
                    if feature in df_seg.columns:
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
        heatmap_data = df_heatmap.pivot_table(
            index='Feature',
            columns='Handicap',
            values='Correlacao',
            aggfunc='mean'
        ).fillna(0)

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

    # ============================================================
    # 🤖 MODELOS POR HANDICAP
    # ============================================================
    def treinar_modelo_handicap_especifico(history: pd.DataFrame, handicap_alvo: float, features: list):
        df_segmento = segmentar_por_handicap(history, handicap_alvo, 0.25)

        if len(df_segmento) < 50:
            st.warning(f"⚠️ Amostras insuficientes para modelo {handicap_alvo}: {len(df_segmento)}")
            return None, None

        if 'Cover_Home' not in df_segmento.columns or 'Cover_Away' not in df_segmento.columns:
            df_segmento = criar_targets_cobertura(df_segmento)

        if 'Date' not in df_segmento.columns:
            st.warning(f"⚠️ Coluna 'Date' ausente no segmento {handicap_alvo}. Sem validação temporal.")
            df_segmento['Date'] = pd.to_datetime('1900-01-01')

        df_segmento, df_train, df_val = split_temporal(df_segmento, test_size=0.2)

        features_disponiveis = [f for f in features if f in df_segmento.columns]
        if not features_disponiveis:
            st.error(f"❌ Nenhuma feature disponível para treino no handicap {handicap_alvo}")
            return None, None

        X_train = clean_features_for_training(df_train[features_disponiveis])
        y_train = df_train['Cover_Home'].astype(int)

        X_val = clean_features_for_training(df_val[features_disponiveis])
        y_val = df_val['Cover_Home'].astype(int)

        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        modelo.fit(X_train, y_train)

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

    # ============================================================
    # 📈 EDGE: MERCADO vs MODELO
    # ============================================================
    def market_cover_prob(row):
        """
        Probabilidade de o HOME cobrir o handicap, segundo o MERCADO,
        usando Asian_Line_Decimal (perspectiva HOME) e odds 1X2 com vig removido.
        """
        hc = row.get('Asian_Line_Decimal', np.nan)
        if pd.isna(hc):
            return np.nan

        p_h, p_d, p_a = odds_to_probs(
            row.get('Odd_H', 0),
            row.get('Odd_D', 0),
            row.get('Odd_A', 0)
        )

        if hc < 0:
            return p_h
        elif hc == 0:
            return p_h
        else:
            return 1 - p_a  # Home recebe gols

    def calcular_edge_mercado(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Market_Cover_Prob'] = df.apply(market_cover_prob, axis=1)

        df['Edge_Model_vs_Market'] = (
            df.get('P_Cover_Home_Especifico', np.nan) - df['Market_Cover_Prob']
        )

        df['Edge_Pick'] = np.where(
            df.get('AIL_Pick_Side', "") == "HOME",
            df['Edge_Model_vs_Market'],
            -df['Edge_Model_vs_Market']
        )
        return df

    # ============================================================
    # 🧠 AIL VALUE SCORE HÍBRIDO (ML + WG + MOMENTUM)
    # ============================================================
    def gerar_ail_value_score_hibrido(df: pd.DataFrame, profile: str = "moderado") -> pd.DataFrame:
        df = df.copy()

        thr_signal = 0.15
        thr_conf_alta = 0.30
        thr_conf_media = 0.15

        ml_component = (df.get('P_Cover_Home_Especifico', 0) - 0.5) * 2
        ml_component = ml_component.fillna(0)

        wg_component = df.get('WG_Diff', 0).fillna(0)
        mom_component = df.get('Momentum_Diff', 0).fillna(0)

        df['AIL_Value_Score'] = (
            0.6 * ml_component +
            0.3 * wg_component +
            0.1 * mom_component
        )

        picks = []
        confs = []
        sides = []
        hc_display_list = []

        for _, row in df.iterrows():
            score = row['AIL_Value_Score']
            hc_home = row.get('Asian_Line_Decimal', np.nan)
            p_home = row.get('P_Cover_Home_Especifico', np.nan)

            if pd.isna(hc_home) or pd.isna(p_home):
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            if abs(hc_home) < 0.25:
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            if abs(score) < thr_signal:
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            if p_home > 0.5:
                side = "HOME"
                hc_side = hc_home
            else:
                side = "AWAY"
                hc_side = -hc_home

            mag = abs(score)
            if mag >= thr_conf_alta:
                conf = "ALTA"
            elif mag >= thr_conf_media:
                conf = "MEDIA"
            else:
                conf = "BAIXA"

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

    # ============================================================
    # 🧠 AIL VALUE SCORE HÍBRIDO FOCADO EM HANDICAP (WG_AH)
    # ============================================================
    def gerar_ail_value_score_hibrido_ah(df: pd.DataFrame, profile: str = "moderado") -> pd.DataFrame:
        df = df.copy()

        thr_signal = 0.15
        thr_conf_alta = 0.30
        thr_conf_media = 0.15

        ml_component = (df.get('P_Cover_Home_Especifico', 0) - 0.5) * 2
        ml_component = ml_component.fillna(0)

        wg_ah_component = df.get('WG_AH_Diff', 0).fillna(0)
        mom_component = df.get('Momentum_Diff', 0).fillna(0)

        df['AIL_Value_Score_AH'] = (
            0.5 * ml_component +
            0.4 * wg_ah_component +
            0.1 * mom_component
        )

        picks = []
        confs = []
        sides = []
        hc_display_list = []

        for _, row in df.iterrows():
            score = row['AIL_Value_Score_AH']
            hc_home = row.get('Asian_Line_Decimal', np.nan)
            p_home = row.get('P_Cover_Home_Especifico', np.nan)

            if pd.isna(hc_home) or pd.isna(p_home):
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            if abs(hc_home) < 0.25:
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            if abs(score) < thr_signal:
                picks.append("⚪ PASS")
                confs.append("PASS")
                sides.append("")
                hc_display_list.append("")
                continue

            if p_home > 0.5:
                side = "HOME"
                hc_side = hc_home
            else:
                side = "AWAY"
                hc_side = -hc_home

            mag = abs(score)
            if mag >= thr_conf_alta:
                conf = "ALTA"
            elif mag >= thr_conf_media:
                conf = "MEDIA"
            else:
                conf = "BAIXA"

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

        df['AIL_Pick_AH'] = picks
        df['AIL_Confidence_AH'] = confs
        df['AIL_Pick_Side_AH'] = sides
        df['AIL_Handicap_Display_AH'] = hc_display_list

        return df

    # ============================================================
    # 🔷 STAMP PREMIUM
    # ============================================================
    def aplicar_stamp(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['AIL_Stamp'] = ""
        cond = (
            (df.get('AIL_Confidence', "") == "ALTA") &
            (df.get('AIL_Value_Score', 0).abs() > 0.25) &
            (df.get('Edge_Pick', 0).abs() > 0.10)
        )
        df.loc[cond, 'AIL_Stamp'] = "🔷"
        return df

    def aplicar_stamp_ah(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['AIL_Stamp_AH'] = ""
        cond = (
            (df.get('AIL_Confidence_AH', "") == "ALTA") &
            (df.get('AIL_Value_Score_AH', 0).abs() > 0.25) &
            (df.get('Edge_Pick', 0).abs() > 0.10)
        )
        df.loc[cond, 'AIL_Stamp_AH'] = "🔷"
        return df

    # ============================================================
    # 📊 GRÁFICO MODELO vs MERCADO (UX)
    # ============================================================
    def plot_model_vs_market(df: pd.DataFrame):
        df_plot = df.dropna(subset=['P_Cover_Home_Especifico', 'Market_Cover_Prob']).copy()
        if df_plot.empty:
            st.info("📉 Sem dados suficientes para gráfico Modelo vs Mercado.")
            return

        df_plot['Abs_Edge'] = df_plot['Edge_Model_vs_Market'].abs()
        top = df_plot.nlargest(12, 'Abs_Edge')

        labels = top['Home'] + " x " + top['Away']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, top['Market_Cover_Prob'], alpha=0.4, label="Mercado (HOME cover)")
        ax.barh(labels, top['P_Cover_Home_Especifico'], alpha=0.7, label="Modelo (HOME cover)")
        ax.set_title("📈 Modelo vs Mercado — Prob. HOME cobrir o Handicap")
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # 🎯 APLICAR MODELOS NOS JOGOS DE HOJE
    # ============================================================
    def aplicar_modelos_handicap(games_today: pd.DataFrame, modelos_handicap: dict) -> pd.DataFrame:
        st.markdown("### 🎯 Previsões por Handicap Específico (ML + AIL)")

        df_all = games_today.copy()

        df_all['P_Cover_Home_Especifico'] = np.nan
        df_all['Value_Gap_Específico'] = np.nan
        df_all['Handicap_Modelo'] = np.nan
        df_all['Modelo_Confianca'] = ""

        for handicap, model_pack in modelos_handicap.items():
            modelo, features = model_pack
            if modelo is None:
                continue

            jogos_alvo = segmentar_por_handicap(df_all, handicap, 0.25)
            if len(jogos_alvo) == 0:
                continue

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
    # 🧩 INTERPRETAÇÃO DO HANDICAP (UX)
    # ============================================================
    def interpretar_handicap(hc):
        if pd.isna(hc):
            return ""
        if hc < 0:
            return f"🏠 Home cede {abs(hc)} gols (precisa vencer por {abs(hc)+0.01:.1f}+ para cobrir)"
        elif hc > 0:
            return f"✈️ Home recebe +{hc} gols (pode empatar ou perder por pouco e ainda cobrir)"
        else:
            return "Linha 0: Home precisa vencer para cobrir"

    # ============================================================
    # HIGHLIGHT VISUAL (CONFIRMED PICKS)
    # ============================================================
    def highlight_confirmed(row):
        return ['background-color: #dbeafe' if row.get('Confirmed', '') == '🔷' else '' for _ in row]

    # ============================================================
    # 🚀 CARREGAR DADOS (CACHE)
    # ============================================================
    st.info("📂 Carregando dados para Análise GetHandicap V1...")

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

        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        if 'Asian_Line' in history.columns:
            history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
        if 'Asian_Line' in games_today.columns:
            games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

        history = calcular_zscores_detalhados(history)
        history = adicionar_weighted_goals(history)
        history = aplicar_clusterizacao_3d(calcular_distancias_3d(history))

        games_today = calcular_zscores_detalhados(games_today)
        games_today = aplicar_weighted_goals_today(history, games_today)
        games_today = aplicar_clusterizacao_3d(calcular_distancias_3d(games_today))

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

    # ---------------- MODO 1: ANÁLISE EXPLORATÓRIA ----------------
    if analise_modo == "📊 Análise Exploratória":
        st.header("📊 Análise Exploratória por Handicap")

        handicap_selecionado = st.selectbox(
            "Selecione o Handicap para Análise (Home):",
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            index=2
        )

        if st.button("🔍 Analisar Patterns", type="primary"):
            df_segmento = segmentar_por_handicap(history, handicap_selecionado, 0.25)
            top_home, top_away = analisar_patterns_handicap(df_segmento, f"Handicap {handicap_selecionado}")

            st.markdown("---")
            criar_heatmap_handicap_features(history)

    # ---------------- MODO 2: TREINO DE MODELOS ----------------
    elif analise_modo == "🤖 Modelos Específicos":
        st.header("🤖 Treinar Modelos por Handicap (Time-Safe)")

        handicaps_treinar = st.multiselect(
            "Handicaps para Treinar Modelos (Home):",
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            default=[-0.5, 0.0, 0.5]
        )

        features_base = [
            'WG_Home_Team', 'WG_Away_Team', 'WG_Diff',
            'WG_AH_Home_Team', 'WG_AH_Away_Team', 'WG_AH_Diff',
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

    # ---------------- MODO 3: PREVISÕES HOJE (AIL HÍBRIDO) ----------------
    elif analise_modo == "🎯 Previsões Hoje":
        st.header("🎯 Previsões para Jogos de Hoje (Modo Híbrido AIL)")

        if 'modelos_handicap' not in st.session_state:
            st.warning("⚠️ Treine os modelos específicos primeiro na aba 'Modelos Específicos'")
            return

        df_all = aplicar_modelos_handicap(games_today, st.session_state['modelos_handicap'])

        df_all = gerar_ail_value_score_hibrido(df_all, profile="moderado")
        df_all = gerar_ail_value_score_hibrido_ah(df_all, profile="moderado")
        df_all = calcular_edge_mercado(df_all)
        df_all = aplicar_stamp(df_all)
        df_all = aplicar_stamp_ah(df_all)

        df_all['Handicap_Info'] = df_all['Asian_Line_Decimal'].apply(interpretar_handicap)

        # Consenso entre sistemas (Base + AH)
        df_all['Confirmed'] = np.where(
            (df_all['AIL_Pick'] != "⚪ PASS") &
            (df_all['AIL_Pick_AH'] != "⚪ PASS") &
            (df_all['AIL_Pick_Side'] == df_all['AIL_Pick_Side_AH']),
            "🔷",
            ""
        )

        # Apenas picks (sem PASS)
        df_picks_base = df_all[df_all['AIL_Pick'] != "⚪ PASS"].copy()
        df_picks_ah = df_all[df_all['AIL_Pick_AH'] != "⚪ PASS"].copy()

        if df_picks_base.empty and df_picks_ah.empty:
            st.warning("⚠️ Nenhuma pick gerada com os thresholds atuais.")
            return

        # Filtros UX
        st.sidebar.markdown("## 🔍 Filtros Previsões")
        min_score = st.sidebar.slider("Score mínimo (AIL_Value_Score):", 0.0, 0.6, 0.15, 0.05)
        conf_sel = st.sidebar.multiselect(
            "Confiança AIL:",
            ['ALTA', 'MEDIA', 'BAIXA'],
            default=['ALTA', 'MEDIA']
        )

        # -------- TABELA A - Geral (WG base) --------
        df_filtrado_base = df_picks_base[
            (df_picks_base['AIL_Value_Score'].abs() >= min_score) &
            (df_picks_base['AIL_Confidence'].isin(conf_sel))
        ].copy()

        df_filtrado_base = df_filtrado_base.sort_values('AIL_Value_Score', key=lambda s: s.abs(), ascending=False)

        st.subheader("🟦 Tabela A – Recomendação Geral (WG base)")
        st.metric("🎯 Apostas Sugeridas (Geral)", len(df_filtrado_base))

        if not df_filtrado_base.empty:
            cols_base = [
                'League', 'Time', 'Home', 'Away',
                'Asian_Line_Decimal',
                'AIL_Pick', 'AIL_Confidence', 'AIL_Stamp', 'Confirmed',
                'AIL_Value_Score',
                'P_Cover_Home_Especifico', 'Market_Cover_Prob', 'Edge_Pick',
                'WG_Home_Team', 'WG_Away_Team', 'WG_Diff'
            ]
            cols_base = [c for c in cols_base if c in df_filtrado_base.columns]

            st.dataframe(
                df_filtrado_base[cols_base].style,
                use_container_width=True
            )
        else:
            st.info("Nenhuma aposta na Tabela A com os filtros atuais.")

        st.markdown("---")

        # -------- TABELA B - Handicap (WG_AH) --------
        df_filtrado_ah = df_picks_ah[
            (df_picks_ah['AIL_Value_Score_AH'].abs() >= min_score) &
            (df_picks_ah['AIL_Confidence_AH'].isin(conf_sel))
        ].copy()

        df_filtrado_ah = df_filtrado_ah.sort_values('AIL_Value_Score_AH', key=lambda s: s.abs(), ascending=False)

        st.subheader("🟥 Tabela B – Recomendação Focada em Handicap (WG_AH)")
        st.metric("🎯 Apostas Sugeridas (Handicap)", len(df_filtrado_ah))

        if not df_filtrado_ah.empty:
            cols_ah = [
                'League', 'Time', 'Home', 'Away',
                'Asian_Line_Decimal',
                'AIL_Pick_AH', 'AIL_Confidence_AH', 'AIL_Stamp_AH', 'Confirmed',
                'AIL_Value_Score_AH',
                'P_Cover_Home_Especifico', 'Market_Cover_Prob', 'Edge_Pick',
                'WG_AH_Home_Team', 'WG_AH_Away_Team', 'WG_AH_Diff'
            ]
            cols_ah = [c for c in cols_ah if c in df_filtrado_ah.columns]

            st.dataframe(
                df_filtrado_ah[cols_ah].style,
                use_container_width=True
            )
        else:
            st.info("Nenhuma aposta na Tabela B com os filtros atuais.")

        # Seção de análise visual Modelo vs Mercado
        with st.expander("📊 Análise de Valor: Modelo vs Mercado"):
            plot_model_vs_market(df_filtrado_base if not df_filtrado_base.empty else df_all)


# ============================================================
# ▶️ EXECUTAR
# ============================================================
if __name__ == "__main__":
    main_handicap_v1()
