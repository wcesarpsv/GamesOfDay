# =====================================================================
# 🎯 SISTEMA ESPACIAL INTELIGENTE – V3 MARKET JUDGMENT
# =====================================================================
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistema Espacial Inteligente – Market Judgment V3", layout="wide")
st.title("🎯 Sistema Espacial Inteligente com Market Judgment V3")

# ------------------- CONFIG -------------------
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup","copas","uefa","afc","sudamericana","copa","trophy"]
np.random.seed(42)

# =====================================================================
# 🔧 FUNÇÕES BÁSICAS (mantendo sua lógica)
# =====================================================================
def filter_leagues(df):
    if df.empty or "League" not in df.columns: return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern,na=False)].copy()

def convert_asian_line_to_decimal(v):
    if pd.isna(v): return np.nan
    v=str(v).strip()
    if "/" not in v:
        try: return -float(v)
        except: return np.nan
    try:
        parts=[float(p) for p in v.split("/")]
        avg=np.mean(parts)
        return -avg if str(v).startswith("-") else avg*-1
    except: return np.nan

def calculate_ah_home_target(row):
    gh,ga,line=row.get("Goals_H_FT"),row.get("Goals_A_FT"),row.get("Asian_Line_Decimal")
    if pd.isna(gh) or pd.isna(ga) or pd.isna(line): return np.nan
    return 1 if (gh+line-ga)>0 else 0

def calcular_momentum_time(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Calcula MT_H e MT_A - versão simplificada"""
    df = df.copy()
    
    if 'MT_H' not in df.columns:
        df['MT_H'] = 0.0
    if 'MT_A' not in df.columns:
        df['MT_A'] = 0.0
        
    # Se não tem HandScore, retorna zeros
    if 'HandScore_Home' not in df.columns or 'HandScore_Away' not in df.columns:
        df['MT_H'] = 0.0
        df['MT_A'] = 0.0
        return df
        
    return df

# =====================================================================
# 📊 CÁLCULO ESPACIAL (seguindo a lógica do código que funciona)
# =====================================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas espaciais 3D com normalização por liga
    """
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]

    # Se faltar algo, inicializa colunas de saída
    out_cols = [
        'dx', 'dy', 'dz',
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Angle_XY', 'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Vector_Sign', 'Magnitude_3D'
    ]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando: {missing_cols}")
        for col in out_cols:
            if col not in df.columns:
                df[col] = 0
        return df

    # Normalização por liga (z-score) - MESMA LÓGICA DO CÓDIGO QUE FUNCIONA
    cols_norm = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    if 'League' in df.columns:
        df[cols_norm] = df.groupby('League')[cols_norm].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) or 1)
        )
    else:
        for c in cols_norm:
            mu = df[c].mean()
            sigma = df[c].std(ddof=0) or 1
            df[c] = (df[c] - mu) / sigma

    try:
        # Diferenciais
        df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
        df['dy'] = df['M_H'] - df['M_A']
        df['dz'] = df['MT_H'] - df['MT_A']

        dx = df['dx'].fillna(0)
        dy = df['dy'].fillna(0)
        dz = df['dz'].fillna(0)

        # Distância 3D
        df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Ângulo XY
        angle_xy = np.arctan2(dy, dx)
        df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
        df['Quadrant_Sin_XY'] = np.sin(angle_xy)
        df['Quadrant_Cos_XY'] = np.cos(angle_xy)

        # Sinal do vetor
        df['Vector_Sign'] = np.sign(dx * dy * dz).fillna(0)

        # Separação média e magnitude
        df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3.0
        df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

        for col in out_cols:
            df[col] = df[col].fillna(0)

    except Exception as e:
        st.error(f"❌ Erro no cálculo 3D: {e}")
        for col in out_cols:
            df[col] = 0

    return df

# =====================================================================
# ⚡ CLUSTERIZAÇÃO 3D 
# =====================================================================
def aplicar_clusterizacao_3d(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Clusterização 3D usando (dx, dy, dz)
    """
    df = df.copy()

    required_cols = ['dx', 'dy', 'dz']
    if not all(c in df.columns for c in required_cols):
        df = calcular_distancias_3d(df)

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    # Poucos dados -> cluster único
    if len(df) < 10:
        df['Cluster3D_Label'] = 0
        return df

    try:
        best_k = 4  # Default
        max_k = min(6, len(df) - 1)
        
        if len(df) >= 20:
            best_score = -1
            for k in range(2, max_k + 1):
                try:
                    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    labels = km.fit_predict(X_cluster)
                    score = silhouette_score(X_cluster, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except Exception:
                    continue

        kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        df['Cluster3D_Label'] = labels

    except Exception as e:
        st.warning(f"⚠️ Clusterização simplificada: {e}")
        df['Cluster3D_Label'] = 0

    return df

# =====================================================================
# 🎯 OTIMIZAÇÃO DE ÂNGULO (igual ao código que funciona)
# =====================================================================
def encontrar_angulo_otimo(history: pd.DataFrame, target_col: str = 'Target_AH_Home', min_samples: int = 100) -> int:
    """
    Encontra o melhor ângulo-limite para separar zonas estáveis/instáveis
    """
    if 'Quadrant_Angle_XY' not in history.columns:
        return 40

    if target_col not in history.columns:
        return 40

    if len(history) < min_samples:
        return 40

    resultados = []
    angulos_testar = range(10, 80, 5)

    for ang in angulos_testar:
        mask_estavel = history['Quadrant_Angle_XY'].abs() < ang
        mask_instavel = ~mask_estavel

        # Estável
        if mask_estavel.sum() >= min_samples:
            estavel = history[mask_estavel]
            acc_e = estavel[target_col].mean()
            vol_e = len(estavel)
        else:
            acc_e = vol_e = 0

        # Instável
        if mask_instavel.sum() >= min_samples:
            instavel = history[mask_instavel]
            acc_i = instavel[target_col].mean()
            vol_i = len(instavel)
        else:
            acc_i = vol_i = 0

        if vol_e >= min_samples and vol_i >= min_samples:
            diff_acc = acc_e - acc_i
            score = diff_acc * ((vol_e + vol_i) / 2000)
        else:
            diff_acc = 0
            score = -1

        resultados.append({
            'angulo_limite': ang,
            'acuracia_estavel': acc_e,
            'volume_estavel': vol_e,
            'acuracia_instavel': acc_i,
            'volume_instavel': vol_i,
            'diferenca_acuracia': diff_acc,
            'score_qualidade': score
        })

    df_res = pd.DataFrame(resultados)
    df_validos = df_res[df_res['score_qualidade'] > 0]

    if not df_validos.empty:
        idx = df_validos['score_qualidade'].idxmax()
        ang_otimo = int(df_validos.loc[idx, 'angulo_limite'])
        return ang_otimo

    return 40

# =====================================================================
# 🧮 SCORE ESPACIAL INTELIGENTE 
# =====================================================================
def calcular_score_espacial_inteligente(row, angulo_limite: float) -> float:
    """
    Score contínuo [0,1] baseado em geometria
    """
    dx = row.get('dx', 0)
    dy = row.get('dy', 0)
    dz = row.get('dz', 0)
    ang_xy = row.get('Quadrant_Angle_XY', 0)
    cluster = row.get('Cluster3D_Label', 0)

    if any(pd.isna(v) for v in [dx, dy, dz, ang_xy]):
        return 0.5

    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    ang_estavel = abs(ang_xy) < angulo_limite

    score = 0.5

    # Peso direção principal (dx)
    if dx > 0:
        score += 0.12
    elif dx < 0:
        score -= 0.12

    # Momentum 3D (dz)
    if dz > 0:
        score += 0.10
    elif dz < 0:
        score -= 0.10

    # Ângulo estável favorece quem está "empurrando"
    if ang_estavel and dx > 0:
        score += 0.08
    if (not ang_estavel) and dx < 0:
        score -= 0.08

    # Distância mínima: se muito perto, puxa p/ neutro
    if dist < 0.4:
        score = 0.5 + (score - 0.5) * 0.4

    # Cluster confiável puxa levemente
    if cluster in [0]:
        score += 0.04
    elif cluster in [1]:
        score -= 0.04

    return float(np.clip(score, 0.05, 0.95))

# =====================================================================
# 🔥 DETECÇÃO DE JULGAMENTO DE MERCADO (NOVO)
# =====================================================================
def calcular_julgamento_mercado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de julgamento de mercado baseado na discrepância
    entre agressividade e momentum
    """
    df = df.copy()
    
    # Discrepância de julgamento (quando agressividade não condiz com momentum)
    df['Judgment_Discrepancy_H'] = (df['Aggression_Home'] * -1) * (df['M_H'] + df['MT_H'])
    df['Judgment_Discrepancy_A'] = (df['Aggression_Away'] * -1) * (df['M_A'] + df['MT_A'])
    df['Diff_Judgment'] = df['Judgment_Discrepancy_H'] - df['Judgment_Discrepancy_A']
    
    # Valor relativo (quem está sub/sobrevalorizado)
    df['Market_Value_H'] = df['Aggression_Home'] - df['M_H']
    df['Market_Value_A'] = df['Aggression_Away'] - df['M_A']
    df['Value_Gap'] = df['Market_Value_H'] - df['Market_Value_A']
    
    return df

# =====================================================================
# 🧠 TREINAMENTO COM MARKET JUDGMENT V3 (CORRIGIDO)
# =====================================================================
def treinar_modelo_espacial_inteligente(history: pd.DataFrame, games_today: pd.DataFrame):
    st.markdown("## 🧠 Treinando Modelo Market Judgment V3")

    # 1) Verificar se temos dados históricos suficientes
    if history.empty:
        st.error("❌ Nenhum dado histórico com resultados para treinar!")
        return None, games_today

    st.success(f"📚 Dados de treino: {len(history)} jogos históricos com resultados")

    # 2) Métricas 3D e clusterização
    st.info("📐 Calculando métricas 3D e clusters...")
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)

    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    # 3) Cálculo de julgamento de mercado
    st.info("🎯 Calculando julgamento de mercado...")
    history = calcular_julgamento_mercado(history)
    games_today = calcular_julgamento_mercado(games_today)

    # 4) Otimizar ângulo usando Target_AH_Home (base real)
    angulo_otimo = encontrar_angulo_otimo(history, target_col='Target_AH_Home')

    # 5) Score & Target Espacial (APENAS PARA HISTORY - TREINO)
    st.info(f"🎯 Aplicando Score Espacial com ângulo {angulo_otimo}°")
    history['Score_Espacial'] = history.apply(
        lambda x: calcular_score_espacial_inteligente(x, angulo_otimo), axis=1
    )
    history['Target_Espacial'] = (history['Score_Espacial'] >= 0.5).astype(int)

    # 6) Features para o modelo (EXCLUINDO Score_Espacial das features de treino)
    features_espaciais = [
        'dx', 'dy', 'dz', 'Diff_Judgment', 'Value_Gap',
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Angle_XY',
        'Vector_Sign', 'Magnitude_3D',
        'Cluster3D_Label'  # REMOVIDO: 'Score_Espacial'
    ]

    features_espaciais = [f for f in features_espaciais if f in history.columns]

    # Adicionar dummies de liga se disponível
    if 'League' in history.columns:
        ligas_dummies = pd.get_dummies(history['League'], prefix='League')
        X = pd.concat([ligas_dummies, history[features_espaciais]], axis=1).fillna(0)
    else:
        X = history[features_espaciais].fillna(0)
        
    y = history['Target_Espacial'].astype(int)

    if len(history) < 50:
        st.error("❌ Dados insuficientes para treinamento (<50 jogos).")
        return None, games_today

    # 7) Modelo
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(X, y)

    # 8) CALCULAR SCORE ESPACIAL PARA HOJE (APÓS O TREINAMENTO)
    games_today['Score_Espacial'] = games_today.apply(
        lambda x: calcular_score_espacial_inteligente(x, angulo_otimo), axis=1
    )

    # 9) Preparar dados de hoje para previsão (AGORA INCLUINDO Score_Espacial)
    features_para_previsao = features_espaciais + ['Score_Espacial']  # ADICIONADO para previsão
    
    if 'League' in games_today.columns:
        ligas_today = pd.get_dummies(games_today['League'], prefix='League')
        # Garantir mesmas colunas
        for col in X.columns:
            if col.startswith("League_") and col not in ligas_today.columns:
                ligas_today[col] = 0
        
        # Usar features atualizadas que incluem Score_Espacial
        X_today = pd.concat([ligas_today, games_today[features_para_previsao]], axis=1)
    else:
        X_today = games_today[features_para_previsao].copy()
    
    # Garantir que X_today tenha as mesmas colunas que X (do treino)
    X_today = X_today.reindex(columns=X.columns, fill_value=0)

    # 10) Previsão
    try:
        proba = model.predict_proba(X_today)[:, 1]
        proba = np.clip(proba, 0.05, 0.95)
    except Exception as e:
        st.error(f"❌ Erro nas previsões: {e}")
        proba = np.full(len(games_today), 0.5)

    games_today['Prob_Espacial'] = proba
    games_today['ML_Side_Espacial'] = np.where(proba >= 0.5, 'HOME', 'AWAY')
    games_today['Confidence_Espacial'] = np.round(np.maximum(proba, 1 - proba), 3)
    games_today['Angulo_Otimizado'] = angulo_otimo

    st.success(f"✅ Modelo Market Judgment V3 treinado em {len(history)} jogos históricos")

    return model, games_today

# =====================================================================
# 🚀 MAIN (LÓGICA CORRIGIDA)
# =====================================================================
def main():
    st.sidebar.markdown("## ⚙️ Configurações Market Judgment V3")
    
    # Carregar arquivos
    files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith('.csv')]
    
    if not files: 
        st.error(f"❌ Nenhum CSV encontrado na pasta '{GAMES_FOLDER}'")
        return
    
    # Selecionar arquivo mais recente
    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.sidebar.selectbox("Selecionar arquivo:", options, index=len(options)-1)
    
    try:
        # Carregar dados do dia
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
        games_today = filter_leagues(games_today)
        
        # Carregar histórico de vários arquivos
        history_dfs = []
        for file in files:
            try:
                df_hist = pd.read_csv(os.path.join(GAMES_FOLDER, file))
                df_hist = filter_leagues(df_hist)
                history_dfs.append(df_hist)
            except:
                continue
        
        if history_dfs:
            history = pd.concat(history_dfs, ignore_index=True)
        else:
            history = pd.DataFrame()
            
        st.sidebar.success(f"✅ {len(games_today)} jogos de hoje | {len(history)} jogos históricos")
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return

    # Preparar colunas necessárias
    if 'Asian_Line' in history.columns:
        history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    if 'Asian_Line' in games_today.columns:
        games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

    # Garantir colunas de goals no histórico
    if 'Goals_H_FT' not in history.columns: 
        history['Goals_H_FT'] = np.nan
        history['Goals_A_FT'] = np.nan

    # Calcular target APENAS para jogos históricos que têm resultados
    history['Target_AH_Home'] = history.apply(calculate_ah_home_target, axis=1)
    
    # Filtrar apenas jogos com resultados para treinamento
    history_with_results = history.dropna(subset=['Target_AH_Home']).copy()
    
    # Aplicar momentum time
    history_with_results = calcular_momentum_time(history_with_results)
    games_today = calcular_momentum_time(games_today)

    st.sidebar.info(f"""
    📊 **Estatísticas:**
    - 📚 Histórico com resultados: **{len(history_with_results)}** jogos
    - 🎯 Jogos para prever hoje: **{len(games_today)}** jogos
    """)

    # Debug
    with st.expander("🔍 Ver detalhes dos dados"):
        st.write("**Colunas disponíveis:**", list(games_today.columns))
        if not history_with_results.empty:
            st.write("**Exemplo histórico:**")
            st.dataframe(history_with_results[['Home', 'Away', 'Goals_H_FT', 'Goals_A_FT', 'Target_AH_Home']].head(3), width='stretch')

    st.markdown("""
    ## 🎯 Market Judgment V3
    - 🔍 **Detecção de julgamento de mercado** (sub/sobrevalorização)
    - 📐 **Análise espacial 3D** com otimização de ângulo
    - 🧠 **ML com features de discrepância** entre agressividade e momentum
    - ⚡ **Clusterização dinâmica** para padrões de mercado
    """)

    if st.sidebar.button("🚀 Executar Market Judgment V3", type="primary"):
        with st.spinner("Processando dados e detectando julgamento de mercado..."):
            model, resultados = treinar_modelo_espacial_inteligente(history_with_results, games_today)
            
            if model is not None and not resultados.empty:
                # Mostrar resultados
                st.markdown("## 📊 Previsões com Market Judgment")
                
                # Tabela principal
                cols_principal = ['League', 'Home', 'Away', 'Prob_Espacial', 'ML_Side_Espacial', 'Confidence_Espacial']
                cols_principal = [c for c in cols_principal if c in resultados.columns]
                
                df_show = resultados[cols_principal].copy()
                df_show = df_show.sort_values('Confidence_Espacial', ascending=False)
                
                st.dataframe(df_show, width='stretch')
                
                # Tabela de julgamento de mercado
                st.markdown("### 🧭 Top 10 - Julgamento de Mercado")
                cols_julgamento = ['League', 'Home', 'Away', 'Diff_Judgment', 'Value_Gap', 'ML_Side_Espacial']
                cols_julgamento = [c for c in cols_julgamento if c in resultados.columns]
                
                if cols_julgamento:
                    top_julgamento = resultados[cols_julgamento].copy()
                    top_julgamento['Tipo_Julgamento'] = np.where(
                        top_julgamento['Diff_Judgment'] > 0, 
                        '⚡ Home Subestimado', 
                        '🔻 Home Superestimado'
                    )
                    st.dataframe(
                        top_julgamento.sort_values('Diff_Judgment', ascending=False).head(10), 
                        width='stretch'
                    )
                
                # Estatísticas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏠 Previsões HOME", 
                             len(resultados[resultados['ML_Side_Espacial'] == 'HOME']))
                with col2:
                    st.metric("✈️ Previsões AWAY", 
                             len(resultados[resultados['ML_Side_Espacial'] == 'AWAY']))
                with col3:
                    avg_conf = resultados['Confidence_Espacial'].mean()
                    st.metric("🎯 Confiança Média", f"{avg_conf:.1%}")
                    
            else:
                st.error("❌ Não foi possível gerar previsões")

    else:
        st.info("👆 Clique em **'Executar Market Judgment V3'** para detectar oportunidades de mercado")

if __name__ == "__main__":
    main()
