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

# ========================= GET HANDICAP V1 - DUAL MODEL =========================
def main_handicap_v1_dual():
    st.set_page_config(page_title="GetHandicap V1 DUAL - Home & Away Models", layout="wide")
    st.title("🎯 GetHandicap V1 DUAL - Modelos HOME & AWAY Específicos")
    
    # Configurações
    GAMES_FOLDER = "GamesDay"
    LIVESCORE_FOLDER = "LiveScore"
    EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]
    
    # ============================================================
    # 🔧 FUNÇÕES AUXILIARES (mantidas do código anterior)
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
            parts = [float(p) for p in s.replace("+","").replace("-","").split("/")]
            avg = np.mean(parts)
            sign = -1 if s.startswith("-") else 1
            result = sign * avg
            return -result
        except:
            return np.nan

    def load_and_merge_livescore(games_today, selected_date_str):
        livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
        games_today = setup_livescore_columns(games_today)

        if not os.path.exists(livescore_file):
            st.warning(f"⚠️ No LiveScore file found for: {selected_date_str}")
            return games_today

        results_df = pd.read_csv(livescore_file)
        results_df['status'] = results_df['status'].astype(str).str.upper().str.strip()
        df_ft = results_df[results_df['status'] == 'FT'].copy()

        for c in ['home_goal','away_goal','home_red','away_red']:
            df_ft[c] = pd.to_numeric(df_ft[c], errors='coerce').fillna(0).astype(int)

        games_today = games_today.merge(
            df_ft[['Id','status','home_goal','away_goal','home_red','away_red']],
            on='Id',
            how='left',
            suffixes=('', '_ls')
        )

        mask_ft = games_today['status_ls'] == 'FT'
        games_today.loc[mask_ft, 'Goals_H_Today'] = games_today.loc[mask_ft, 'home_goal']
        games_today.loc[mask_ft, 'Goals_A_Today'] = games_today.loc[mask_ft, 'away_goal']
        games_today.loc[mask_ft, 'Home_Red'] = games_today.loc[mask_ft, 'home_red']
        games_today.loc[mask_ft, 'Away_Red'] = games_today.loc[mask_ft, 'away_red']

        st.success(f"✅ LiveScore integrado: {mask_ft.sum()} jogos com resultado final")
        return games_today

    def calcular_zscores_detalhados(df):
        df = df.copy()
        st.info("📊 Calculando Z-scores a partir do HandScore (Home/Away)...")

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

        return df

    def clean_features_for_training(X):
        X_clean = X.copy()
        if isinstance(X_clean, np.ndarray):
            X_clean = pd.DataFrame(X_clean, columns=X.columns if hasattr(X, 'columns') else range(X.shape[1]))

        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
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

        X_clean = X_clean.fillna(0).replace([np.inf, -np.inf], 0)
        return X_clean

    def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
        missing = [c for c in required if c not in df.columns]
        if missing:
            for c in ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D','Momentum_Diff','Momentum_Diff_MT']:
                df[c] = np.nan
            return df
        dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
        dy = (df['M_H'] - df['M_A']) / 2
        dz = (df['MT_H'] - df['MT_A']) / 2
        df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
        df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
        df['Vector_Sign'] = np.sign(dx * dy * dz)
        df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
        df['Momentum_Diff'] = dy
        df['Momentum_Diff_MT'] = dz
        return df

    def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters=4, random_state=42) -> pd.DataFrame:
        df = df.copy()
        required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
        missing = [c for c in required if c not in df.columns]
        if missing:
            df['Cluster3D_Label'] = 0
            return df
        df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
        df['dy'] = df['M_H'] - df['M_A']
        df['dz'] = df['MT_H'] - df['MT_A']
        X = df[['dx','dy','dz']].fillna(0).to_numpy()
        n_samples = X.shape[0]
        k = max(1, min(n_clusters, n_samples))
        try:
            km = KMeans(n_clusters=k, random_state=random_state, init='k-means++', n_init=10)
            df['Cluster3D_Label'] = km.fit_predict(X)
        except Exception as e:
            df['Cluster3D_Label'] = 0
        return df

    def odds_to_probs(odd_h, odd_d, odd_a):
        try:
            odd_h = float(odd_h); odd_d = float(odd_d); odd_a = float(odd_a)
            if odd_h <= 0 or odd_d <= 0 or odd_a <= 0: return 0.33, 0.33, 0.33
            inv_sum = (1/odd_h) + (1/odd_d) + (1/odd_a)
            return (1/odd_h)/inv_sum, (1/odd_d)/inv_sum, (1/odd_a)/inv_sum
        except: return 0.33, 0.33, 0.33

    def wg_home(row):
        gf = row.get('Goals_H_FT', 0); ga = row.get('Goals_A_FT', 0)
        p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 2.5), row.get('Odd_D', 3.0), row.get('Odd_A', 2.5))
        return (gf * (1 - p_h)) - (ga * p_h)

    def wg_away(row):
        gf = row.get('Goals_A_FT', 0); ga = row.get('Goals_H_FT', 0)
        p_h, p_d, p_a = odds_to_probs(row.get('Odd_H', 2.5), row.get('Odd_D', 3.0), row.get('Odd_A', 2.5))
        return (gf * (1 - p_a)) - (ga * p_a)

    def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required_cols = ['Home','Away','Date','Goals_H_FT','Goals_A_FT','Odd_H','Odd_D','Odd_A']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            for col in ['WG_Home','WG_Away','WG_Home_Team','WG_Away_Team','WG_Diff']:
                df[col] = 0.0
            return df

        df['WG_Home'] = df.apply(wg_home, axis=1)
        df['WG_Away'] = df.apply(wg_away, axis=1)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')
        df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']
        return df

    def criar_targets_cobertura(df: pd.DataFrame) -> pd.DataFrame:
        hist = df.dropna(subset=['Goals_H_FT','Goals_A_FT','Asian_Line_Decimal']).copy()
        if hist.empty: return hist
        margin = hist['Goals_H_FT'] - hist['Goals_A_FT']
        adj = margin + hist['Asian_Line_Decimal']
        hist['Cover_Home'] = (adj > 0).astype(int)
        hist['Cover_Away'] = (adj < 0).astype(int)
        return hist

    # ============================================================
    # 🆕 FUNÇÕES DUAL MODEL - HOME & AWAY ESPECÍFICOS
    # ============================================================
    
    def segmentar_por_handicap(df: pd.DataFrame, handicap_alvo: float, tolerancia: float = 0.25) -> pd.DataFrame:
        if df.empty or 'Asian_Line_Decimal' not in df.columns: return pd.DataFrame()
        mask = abs(df['Asian_Line_Decimal'] - handicap_alvo) <= tolerancia
        df_segmento = df[mask].copy()
        return df_segmento
    
    def treinar_modelo_dual_handicap(history: pd.DataFrame, handicap_alvo: float, features: list):
        """
        🆕 TREINA DOIS MODELOS: HOME e AWAY para o mesmo handicap
        """
        df_segmento = segmentar_por_handicap(history, handicap_alvo, 0.25)
        
        if len(df_segmento) < 50:
            st.warning(f"⚠️ Amostras insuficientes para modelo {handicap_alvo}: {len(df_segmento)}")
            return None, None, None
        
        # Garantir targets
        if 'Cover_Home' not in df_segmento.columns:
            df_segmento = criar_targets_cobertura(df_segmento)
        
        # Features disponíveis
        features_disponiveis = [f for f in features if f in df_segmento.columns]
        
        X = df_segmento[features_disponiveis].copy()
        X = clean_features_for_training(X)
        y_home = df_segmento['Cover_Home']
        y_away = df_segmento['Cover_Away']
        
        # 🏠 Modelo HOME
        modelo_home = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42
        )
        modelo_home.fit(X, y_home)
        acc_home = (modelo_home.predict(X) == y_home).mean()
        
        # ✈️ Modelo AWAY  
        modelo_away = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=43  # Diferente seed
        )
        modelo_away.fit(X, y_away)
        acc_away = (modelo_away.predict(X) == y_away).mean()
        
        st.success(f"✅ Handicap {handicap_alvo} - HOME: {acc_home:.3f}, AWAY: {acc_away:.3f} (n={len(df_segmento)})")
        
        return modelo_home, modelo_away, features_disponiveis
    

    def aplicar_modelos_dual_handicap(games_today: pd.DataFrame, modelos_dual_handicap: dict):
        """
        🆕 VERSÃO NORMALIZADA - Garante P_Home + P_Away ≈ 1
        """
        st.markdown("### 🎯 Previsões DUAL NORMALIZADAS - HOME & AWAY")
        
        resultados = []
        
        for handicap, (modelo_home, modelo_away, features) in modelos_dual_handicap.items():
            if modelo_home is None or modelo_away is None:
                continue
                    
            # Filtrar jogos com handicap próximo
            jogos_alvo = segmentar_por_handicap(games_today, handicap, 0.25)
            
            if len(jogos_alvo) == 0:
                continue
            
            # Fazer previsões DUAL
            X_today = jogos_alvo[features].copy()
            X_today = clean_features_for_training(X_today)
            
            # Probabilidades brutas de ambos os modelos
            probas_home_bruto = modelo_home.predict_proba(X_today)[:, 1]  # P(Cover_Home) bruto
            probas_away_bruto = modelo_away.predict_proba(X_today)[:, 1]  # P(Cover_Away) bruto
            
            for idx, (_, jogo) in enumerate(jogos_alvo.iterrows()):
                # 🆕 NORMALIZAÇÃO CRÍTICA: garantir soma ≈ 1
                proba_home_bruto = probas_home_bruto[idx]
                proba_away_bruto = probas_away_bruto[idx]
                
                soma = proba_home_bruto + proba_away_bruto
                
                # Se soma for muito diferente de 1, normalizar
                if abs(soma - 1.0) > 0.05:  # Se diferença > 5%
                    proba_home = proba_home_bruto / soma
                    proba_away = proba_away_bruto / soma
                    normalized = True
                else:
                    proba_home = proba_home_bruto
                    proba_away = proba_away_bruto  
                    normalized = False
                
                # 🆕 DEBUG: Mostrar quando normalizou
                if normalized and (idx == 0 or abs(soma - 1.0) > 0.2):
                    st.info(f"🔧 Normalizado: {proba_home_bruto:.3f} + {proba_away_bruto:.3f} = {soma:.3f} → {proba_home:.3f} + {proba_away:.3f}")
                
                # Value Gaps com probabilidades normalizadas
                value_gap_home = proba_home - 0.5
                value_gap_away = proba_away - 0.5
                
                # 🎯 DECISÃO DINÂMICA BASEADA NO HANDICAP
                def get_dynamic_threshold(asian_line_decimal: float) -> tuple:
                    abs_line = abs(asian_line_decimal)
                    if abs_line <= 0.25:  # Jogos equilibrados
                        return 0.15, 0.25
                    elif abs_line <= 0.75:  # Handicaps médios
                        return 0.20, 0.30
                    else:  # Handicaps pesados
                        return 0.25, 0.35
    
                threshold_bet, threshold_high = get_dynamic_threshold(jogo.get('Asian_Line_Decimal', 0))
                
                if value_gap_home > value_gap_away and value_gap_home > threshold_bet:
                    recomendacao = "BET HOME"
                    value_gap_utilizado = value_gap_home
                    confidence = 'ALTA' if value_gap_home > threshold_high else 'MEDIA'
                elif value_gap_away > value_gap_home and value_gap_away > threshold_bet:
                    recomendacao = "BET AWAY" 
                    value_gap_utilizado = value_gap_away
                    confidence = 'ALTA' if value_gap_away > threshold_high else 'MEDIA'
                else:
                    recomendacao = "NO BET"
                    value_gap_utilizado = max(value_gap_home, value_gap_away)
                    confidence = 'BAIXA'
                
                # Live Score
                g_h = jogo.get('Goals_H_Today'); g_a = jogo.get('Goals_A_Today')
                h_r = jogo.get('Home_Red'); a_r = jogo.get('Away_Red')
                live_score_info = ""
                if pd.notna(g_h) and pd.notna(g_a):
                    live_score_info = f"⚽ {int(g_h)}-{int(g_a)}"
                    if pd.notna(h_r) and int(h_r) > 0: live_score_info += f" 🟥H{int(h_r)}"
                    if pd.notna(a_r) and int(a_r) > 0: live_score_info += f" 🟥A{int(a_r)}"
                
                resultados.append({
                    'League': jogo.get('League', ''),
                    'Home': jogo.get('Home', ''),
                    'Away': jogo.get('Away', ''),
                    'Asian_Line': jogo.get('Asian_Line', ''),
                    'Asian_Line_Decimal': jogo.get('Asian_Line_Decimal', 0),
                    'Handicap_Modelo': handicap,
                    
                    # 🆕 PROBABILIDADES NORMALIZADAS
                    'P_Home_Cover': proba_home,
                    'P_Away_Cover': proba_away,
                    'Value_Gap_HOME': value_gap_home,
                    'Value_Gap_AWAY': value_gap_away,
                    'Soma_Probabilidades': proba_home + proba_away,  # 🆕 DEBUG
                    'Normalizado': normalized,  # 🆕 DEBUG
                    
                    'Recomendacao': recomendacao,
                    'Value_Gap_Utilizado': value_gap_utilizado,
                    'Confianca': confidence,
                    'Live_Score': live_score_info
                })
        
        if resultados:
            df_resultados = pd.DataFrame(resultados)
            
            # 🆕 ESTATÍSTICAS DE NORMALIZAÇÃO
            total_jogos = len(df_resultados)
            normalizados = df_resultados['Normalizado'].sum()
            st.info(f"🔧 Normalização: {normalizados}/{total_jogos} jogos ajustados")
            
            # Mostrar soma média das probabilidades
            soma_media = df_resultados['Soma_Probabilidades'].mean()
            st.info(f"📊 Soma média P_Home + P_Away: {soma_media:.3f}")
            
            # Ordenar por Value Gap utilizado
            df_resultados = df_resultados.sort_values('Value_Gap_Utilizado', ascending=False)
            
            # Estilo para destacar recomendações
            def color_recomendacao(val):
                if 'BET HOME' in str(val): return 'font-weight: bold'
                if 'BET AWAY' in str(val): return 'font-weight: bold'
                return ''
            
            # Mostrar apenas colunas principais (ocultar debug)
            cols_principais = ['League', 'Home', 'Away', 'Asian_Line', 'P_Home_Cover', 'P_Away_Cover', 
                              'Value_Gap_HOME', 'Value_Gap_AWAY', 'Recomendacao', 'Confianca', 'Live_Score']
            df_display = df_resultados[cols_principais]
            
            styled_df = df_display.style.applymap(color_recomendacao, subset=['Recomendacao'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Estatísticas DUAL
            bets_home = df_resultados[df_resultados['Recomendacao'] == 'BET HOME']
            bets_away = df_resultados[df_resultados['Recomendacao'] == 'BET AWAY']
            no_bets = df_resultados[df_resultados['Recomendacao'] == 'NO BET']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("🎯 Total Previsões", len(df_resultados))
            with col2: st.metric("🏠 BET HOME", len(bets_home))
            with col3: st.metric("✈️ BET AWAY", len(bets_away))  
            with col4: st.metric("📊 NO BET", len(no_bets))
            
            return df_resultados
        else:
            st.warning("⚠️ Nenhuma previsão DUAL gerada")
            return pd.DataFrame()

    # ============================================================
    # 🚀 EXECUÇÃO PRINCIPAL
    # ============================================================
    
    st.info("📂 Carregando dados para GetHandicap V1 DUAL...")
    
    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]) if os.path.exists(GAMES_FOLDER) else []
    if not files: return
    
    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)
    
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    
    @st.cache_data(ttl=3600)
    def load_cached_data(selected_file, selected_date_str):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        history = load_all_games(GAMES_FOLDER)
        
        history = filter_leagues(history)
        games_today = filter_leagues(games_today)
        
        history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
        games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
        
        games_today = load_and_merge_livescore(games_today, selected_date_str)
        
        history = calcular_zscores_detalhados(history)
        history = adicionar_weighted_goals(history)
        history = aplicar_clusterizacao_3d(calcular_distancias_3d(history))
        
        games_today = calcular_zscores_detalhados(games_today)
        games_today = adicionar_weighted_goals(games_today)
        games_today = aplicar_clusterizacao_3d(calcular_distancias_3d(games_today))
        
        history = criar_targets_cobertura(history)
        
        return games_today, history
    
    games_today, history = load_cached_data(selected_file, selected_date_str)
    
    if history.empty or games_today.empty: return
    
    # ============================================================
    # 🎯 INTERFACE DUAL MODEL
    # ============================================================
    
    st.sidebar.markdown("## 🎯 GetHandicap V1 DUAL - Config")
    
    analise_modo = st.sidebar.selectbox("Modo de Análise:", ["🤖 Modelos DUAL", "🎯 Previsões DUAL Hoje"])
    
    features_base = [
        'WG_Home_Team', 'WG_Away_Team', 'WG_Diff',
        'M_H', 'M_A', 'MT_H', 'MT_A',
        'Aggression_Home', 'Aggression_Away',
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Vector_Sign', 'Magnitude_3D', 'Momentum_Diff', 'Momentum_Diff_MT',
        'Cluster3D_Label'
    ]
    
    if analise_modo == "🤖 Modelos DUAL":
        st.header("🤖 Treinar Modelos DUAL por Handicap")
        
        handicaps_treinar = st.multiselect("Handicaps para Treinar:", [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], default=[-0.5, 0.0, 0.5])
        
        if st.button("🚀 Treinar Modelos DUAL", type="primary"):
            modelos_dual_treinados = {}
            
            for handicap in handicaps_treinar:
                with st.spinner(f"Treinando modelos DUAL para handicap {handicap}..."):
                    modelo_home, modelo_away, features = treinar_modelo_dual_handicap(history, handicap, features_base)
                    modelos_dual_treinados[handicap] = (modelo_home, modelo_away, features)
            
            st.session_state['modelos_dual_handicap'] = modelos_dual_treinados
            st.success("✅ Todos os modelos DUAL treinados!")
    
    elif analise_modo == "🎯 Previsões DUAL Hoje":
        st.header("🎯 Previsões DUAL para Hoje")
        
        if 'modelos_dual_handicap' not in st.session_state:
            st.warning("⚠️ Treine os modelos DUAL primeiro!")
        else:
            df_previsoes_dual = aplicar_modelos_dual_handicap(games_today, st.session_state['modelos_dual_handicap'])
            
            if not df_previsoes_dual.empty:
                # Filtros para bets válidos
                st.sidebar.markdown("## 🔍 Filtros DUAL")
                min_value_gap = st.sidebar.slider("Value Gap Mínimo:", 0.0, 0.3, 0.1, 0.05)
                confianca_filtro = st.sidebar.multiselect("Confiança:", ['ALTA', 'MEDIA', 'BAIXA'], default=['ALTA', 'MEDIA'])
                
                df_filtrado = df_previsoes_dual[
                    (df_previsoes_dual['Value_Gap_Utilizado'] >= min_value_gap) &
                    (df_previsoes_dual['Confianca'].isin(confianca_filtro)) &
                    (df_previsoes_dual['Recomendacao'] != 'NO BET')
                ]
                
                st.metric("🎯 Apostas DUAL Filtradas", len(df_filtrado))
                
                if len(df_filtrado) > 0:
                    def color_recomendacao_filtrada(val):
                        if 'BET HOME' in str(val): return 'font-weight: bold'
                        if 'BET AWAY' in str(val): return 'font-weight: bold'
                        return ''
                    
                    styled_filtrado = df_filtrado.style.applymap(color_recomendacao_filtrada, subset=['Recomendacao'])
                    st.dataframe(styled_filtrado, use_container_width=True)
                else:
                    st.info("📭 Nenhuma aposta DUAL encontrada com os filtros atuais")

# ============================================================
# 🚀 EXECUTAR
# ============================================================
if __name__ == "__main__":
    main_handicap_v1_dual()
