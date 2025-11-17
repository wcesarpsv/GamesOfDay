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

    # ============================================================ #
    # 🔧 FUNÇÕES AUXILIARES
    # ============================================================ #
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
        """ Converte a linha (Away perspective) para decimal Home perspective """
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

    # ============================================================ #
    # 📊 Calcular Z-scores e métricas
    # ============================================================ #
    def calcular_zscores_detalhados(df):
        df = df.copy()
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

        if 'Home' in df.columns and 'Away' in df.columns:
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

    # ============================================================ #
    # ⭐ Weighted Goals e outras features
    # (continua abaixo — limite de caracteres)
    # ============================================================ #

# ============================================================ #
    # ⚽ Weighted Goals (WG)
    # ============================================================ #
    def odds_to_market_probs(row):
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
        df = df.copy()
        df['WG_Home'] = df.apply(wg_home, axis=1)
        df['WG_Away'] = df.apply(wg_away, axis=1)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

        df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())

        df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']
        df['WG_Confidence'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).count())
        return df

    # ============================================================ #
    # 🎯 Alvo: Home cobrir o handicap
    # ============================================================ #
    def criar_targets_cobertura(df: pd.DataFrame) -> pd.DataFrame:
        hist = df.dropna(subset=['Goals_H_FT', 'Goals_A_FT', 'Asian_Line_Decimal']).copy()
        if hist.empty:
            return hist
        margin = hist['Goals_H_FT'] - hist['Goals_A_FT']
        adj = margin + hist['Asian_Line_Decimal']
        hist['Cover_Home'] = (adj > 0).astype(int)
        hist['Cover_Away'] = (adj < 0).astype(int)
        return hist

    # ============================================================ #
    # 🧬 Features 3D
    # ============================================================ #
    def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
        dy = (df['M_H'] - df['M_A']) / 2
        dz = (df['MT_H'] - df['MT_A']) / 2
        df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
        df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
        df['Vector_Sign'] = np.sign(dx * dy * dz)
        df['Magnitude_3D'] = df['Quadrant_Dist_3D']
        df['Momentum_Diff'] = dy
        df['Momentum_Diff_MT'] = dz
        return df

    def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters=4, random_state=42) -> pd.DataFrame:
        df = df.copy()
        df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
        df['dy'] = df['M_H'] - df['M_A']
        df['dz'] = df['MT_H'] - df['MT_A']
        X = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()
        try:
            km = KMeans(n_clusters=min(n_clusters, len(df)), random_state=random_state, n_init=10)
            df['Cluster3D_Label'] = km.fit_predict(X)
        except:
            df['Cluster3D_Label'] = 0
        return df

    # ============================================================ #
    # 📌 Carregamento histórico + jogos do dia
    # ============================================================ #
    st.info("📂 Carregando dados para Análise GetHandicap V1...")
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
    if not files:
        st.warning("❌ Nenhum CSV na pasta GamesDay.")
        return

    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.selectbox("Arquivo do Matchday:", options, index=len(options)-1)

    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)

        history = filter_leagues(history)
        games_today = filter_leagues(games_today)

        history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
        games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

        history = calcular_zscores_detalhados(history)
        history = adicionar_weighted_goals(history)
        history = aplicar_clusterizacao_3d(calcular_distancias_3d(history))

        games_today = calcular_zscores_detalhados(games_today)
        games_today = adicionar_weighted_goals(games_today)
        games_today = aplicar_clusterizacao_3d(calcular_distancias_3d(games_today))

        history = criar_targets_cobertura(history)
        return games_today, history

    games_today, history = load_cached_data(selected_file)

    if history.empty or games_today.empty:
        st.error("❌ Dados insuficientes para análise")
        return

    # ============================================================ #
    # 🎛️ Interface lateral
    # ============================================================ #
    st.sidebar.markdown("## 🎯 GetHandicap V1 - Configurações")
    analise_modo = st.sidebar.selectbox(
        "Modo:",
        ["📊 Análise Exploratória", "🤖 Modelos Específicos", "🎯 Previsões Hoje"]
    )


# ============================================================ #
    # 🤖 Modelos específicos por linha de handicap
    # ============================================================ #
    def split_temporal(df: pd.DataFrame, test_size: float = 0.2):
        df = df.copy()
        if 'Date' not in df.columns:
            df['__split'] = 'train'
            return df, df, df

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')

        n = len(df)
        if n < 10:
            df['__split'] = 'train'
            return df, df, df

        split_idx = int(n * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        train_df['__split'] = 'train'
        val_df['__split'] = 'val'
        full = pd.concat([train_df, val_df], axis=0)
        return full, train_df, val_df

    def clean_features_for_training(X):
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median()).fillna(0)
        for col in X_clean.columns:
            if X_clean[col].dtype in [float, int]:
                Q1 = X_clean[col].quantile(0.25)
                Q3 = X_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)
        return X_clean

    def segmentar_por_handicap(df: pd.DataFrame, handicap_alvo: float, tolerancia: float = 0.25) -> pd.DataFrame:
        if df.empty or 'Asian_Line_Decimal' not in df.columns:
            return pd.DataFrame()
        mask = df['Asian_Line_Decimal'].sub(handicap_alvo).abs() <= tolerancia
        return df[mask].copy()

    def treinar_modelo_handicap_especifico(history: pd.DataFrame, handicap_alvo: float, features: list):
        df_seg = segmentar_por_handicap(history, handicap_alvo, 0.25)
        if len(df_seg) < 50:
            st.warning(f"⚠️ Poucos dados para handicap {handicap_alvo} ({len(df_seg)})")
            return None, None

        df_seg = criar_targets_cobertura(df_seg)
        df_seg, df_train, df_val = split_temporal(df_seg, test_size=0.2)

        features_ok = [f for f in features if f in df_seg.columns]
        if not features_ok:
            return None, None

        X_train = clean_features_for_training(df_train[features_ok])
        y_train = df_train['Cover_Home'].astype(int)
        X_val = clean_features_for_training(df_val[features_ok])
        y_val = df_val['Cover_Home'].astype(int)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)

        acc_train = (model.predict(X_train) == y_train).mean()
        acc_val = (model.predict(X_val) == y_val).mean()

        st.success(f"Handicap {handicap_alvo}: Train={acc_train:.3f} | Val={acc_val:.3f}")
        return model, features_ok

    # ============================================================ #
    # 🎯 Aplicação dos modelos nos jogos do dia
    # ============================================================ #
    def aplicar_modelos_handicap(games_today: pd.DataFrame, modelos_handicap: dict) -> pd.DataFrame:
        df_all = games_today.copy()
        df_all['P_Cover_Home_Especifico'] = np.nan

        for handicap, model_pack in modelos_handicap.items():
            modelo, features = model_pack
            if modelo is None: 
                continue

            jogos_alvo = segmentar_por_handicap(df_all, handicap, 0.25)
            if len(jogos_alvo) == 0:
                continue

            X_today = clean_features_for_training(jogos_alvo[features])
            probas = modelo.predict_proba(X_today)[:, 1]
            df_all.loc[jogos_alvo.index, 'P_Cover_Home_Especifico'] = probas

        return df_all

    # ============================================================ #
    # 🚀 EXECUÇÃO: modos da interface
    # ============================================================ #
    if analise_modo == "📊 Análise Exploratória":
        st.header("📊 Análise Exploratória por Handicap")
        hc_sel = st.selectbox("Handicap (Home):", [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], index=2)

        df_seg = segmentar_por_handicap(history, hc_sel)
        if len(df_seg) == 0:
            st.warning("Sem jogos!")
        else:
            st.dataframe(df_seg.head())

    elif analise_modo == "🤖 Modelos Específicos":
        st.header("Treinar Modelos")
        handicaps_treinar = st.multiselect(
            "Handicaps:",
            [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
            default=[-0.5, 0.25]
        )

        features_base = [
            'WG_Home_Team', 'WG_Away_Team', 'WG_Diff',
            'M_H', 'M_A', 'MT_H', 'MT_A',
            'Aggression_Home', 'Aggression_Away',
            'Quadrant_Dist_3D', 'Quadrant_Separation_3D'
        ]

        modelos_treinados = {}
        if st.button("🚀 Treinar"):
            for hc in handicaps_treinar:
                modelos_treinados[hc] = treinar_modelo_handicap_especifico(history, hc, features_base)
            st.session_state['modelos_handicap'] = modelos_treinados
            st.success("Modelos prontos!")

    elif analise_modo == "🎯 Previsões Hoje":
        st.header("🎯 Picks para Hoje")
        if 'modelos_handicap' not in st.session_state:
            st.warning("Treine primeiro!")
            return

        df_pred = aplicar_modelos_handicap(games_today, st.session_state['modelos_handicap'])
        st.dataframe(df_pred)

# ============================================================ #
# ▶️ EXECUTAR SCRIPT
# ============================================================ #
if __name__ == "__main__":
    main_handicap_v1()
