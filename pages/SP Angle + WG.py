from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# ========================= CONFIG STREAMLIT =========================
st.set_page_config(page_title="Sistema Espacial Inteligente - Bet Indicator V2", layout="wide")
st.title("🎯 Sistema Espacial Inteligente com Otimização Automática (V2)")

# ========================= CONFIGURAÇÕES GERAIS =========================
PAGE_PREFIX = "Espacial_Inteligente_V2"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)

np.random.seed(42)

# ========================= LIVE SCORE =========================
def setup_livescore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas do Live Score existam no DataFrame"""
    df = df.copy()
    for col in ['Goals_H_Today', 'Goals_A_Today', 'Home_Red', 'Away_Red']:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ========================= HELPERS BÁSICOS =========================
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
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def convert_asian_line_to_decimal(value):
    """
    Converte Asian_Line (referência AWAY) para decimal na perspectiva HOME.
    Exemplo:
        - "0"     -> 0.0
        - "-0.5"  -> +0.5 (Home recebe vantagem)
        - "0/0.5" -> +0.25 (Home)
        - "-0.5/1"-> +0.75 (Home)
    """
    if pd.isna(value):
        return np.nan
    value = str(value).strip()

    # Linha cheia
    if "/" not in value:
        try:
            num = float(value)
            return -num  # inverter perspectiva away -> home
        except ValueError:
            return np.nan

    # Linha fracionada
    try:
        parts = [float(p) for p in value.replace("+", "").split("/")]
        avg = np.mean(parts)

        # Verifica se é negativa (ex: "-0.5/1" ou "-0.5/-1")
        negative = value.startswith("-")

        # Aplica inversão apenas se for negativa
        if negative:
            return abs(avg)
        else:
            return -abs(avg)
    except ValueError:
        return np.nan



def calcular_wg_para_jogos_do_dia(history: pd.DataFrame, games_today: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features WG para jogos do dia usando médias históricas dos times
    """
    if games_today.empty:
        return games_today
    
    games_today_wg = games_today.copy()
    
    # Para cada time nos jogos de hoje, buscar sua média WG histórica
    all_teams_today = set(games_today_wg['Home'].tolist() + games_today_wg['Away'].tolist())
    
    # Calcular as últimas médias WG para cada time do histórico
    wg_home_means = {}
    wg_away_means = {}
    wg_ah_home_means = {}
    wg_ah_away_means = {}
    
    for team in all_teams_today:
        # Últimos 5 jogos como mandante
        home_games = history[history['Home'] == team].tail(5)
        if not home_games.empty:
            wg_home_means[team] = home_games['WG_Home'].mean()
            wg_ah_home_means[team] = home_games['WG_AH_Home'].mean()
        else:
            wg_home_means[team] = 0.0
            wg_ah_home_means[team] = 0.0
            
        # Últimos 5 jogos como visitante
        away_games = history[history['Away'] == team].tail(5)
        if not away_games.empty:
            wg_away_means[team] = away_games['WG_Away'].mean()
            wg_ah_away_means[team] = away_games['WG_AH_Away'].mean()
        else:
            wg_away_means[team] = 0.0
            wg_ah_away_means[team] = 0.0
    
    # Aplicar aos jogos de hoje
    games_today_wg['WG_Home_Team'] = games_today_wg['Home'].map(wg_home_means).fillna(0.0)
    games_today_wg['WG_Away_Team'] = games_today_wg['Away'].map(wg_away_means).fillna(0.0)
    games_today_wg['WG_AH_Home_Team'] = games_today_wg['Home'].map(wg_ah_home_means).fillna(0.0)
    games_today_wg['WG_AH_Away_Team'] = games_today_wg['Away'].map(wg_ah_away_means).fillna(0.0)
    
    # Calcular diferenças
    games_today_wg['WG_Diff'] = games_today_wg['WG_Home_Team'] - games_today_wg['WG_Away_Team']
    games_today_wg['WG_AH_Diff'] = games_today_wg['WG_AH_Home_Team'] - games_today_wg['WG_AH_Away_Team']
    
    # Confiança baseada em quantos dados históricos temos
    games_today_wg['WG_Confidence'] = (
        games_today_wg['WG_Home_Team'].notna().astype(int) + 
        games_today_wg['WG_Away_Team'].notna().astype(int)
    )
    
    return games_today_wg



# ========================= WEIGHTED GOALS SYSTEM =========================
def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Weighted Goals (WG) para o DataFrame
    Penaliza marcar menos do que o esperado
    Premia marcar mais do que o mercado esperava
    """
    df = df.copy()

    for col in ['WG_Home', 'WG_Away']:
        if col not in df.columns:
            df[col] = 0.0

    # Converte odds em probabilidades sem vig
    def odds_to_market_probs(row):
        try:
            odd_h = float(row.get('Odd_H', 0))
            odd_a = float(row.get('Odd_A', 0))

            if odd_h <= 0 or odd_a <= 0:
                return 0.50, 0.50

            inv_h = 1 / odd_h
            inv_a = 1 / odd_a
            total = inv_h + inv_a
            return inv_h / total, inv_a / total

        except:
            return 0.50, 0.50

    # Fórmula base do WG
    def wg_home(row):
        p_h, p_a = odds_to_market_probs(row)
        goals_h = row.get('Goals_H_FT', 0)
        goals_a = row.get('Goals_A_FT', 0)
        return (goals_h * (1 - p_h)) - (goals_a * p_h)

    def wg_away(row):
        p_h, p_a = odds_to_market_probs(row)
        goals_h = row.get('Goals_H_FT', 0)
        goals_a = row.get('Goals_A_FT', 0)
        return (goals_a * (1 - p_a)) - (goals_h * p_a)

    df['WG_Home'] = df.apply(wg_home, axis=1)
    df['WG_Away'] = df.apply(wg_away, axis=1)

    return df

def adicionar_weighted_goals_ah(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta o WG com base na dificuldade do handicap do mercado.
    Handicaps altos = mercado espera goleada
    • Se superar -> WG deve pesar mais
    • Se frustrar -> WG deve punir fortemente
    """
    df = df.copy()

    if 'Asian_Line_Decimal' not in df.columns:
        df['WG_AH_Home'] = 0.0
        df['WG_AH_Away'] = 0.0
        return df

    # Peso baseado na magnitude do handicap
    df['WG_AH_Home'] = df['WG_Home'] * (1 + df['Asian_Line_Decimal'].abs())
    df['WG_AH_Away'] = df['WG_Away'] * (1 + df['Asian_Line_Decimal'].abs())

    return df

def calcular_rolling_wg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula médias rolling dos WG para cada time
    """
    df = df.copy()
    
    # Garantir que Date está em datetime e ordenado
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    # Média dos últimos 5 jogos em casa e fora
    df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    df['WG_AH_Home_Team'] = df.groupby('Home')['WG_AH_Home'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['WG_AH_Away_Team'] = df.groupby('Away')['WG_AH_Away'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Diferenças para serem usadas como features no modelo
    df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']
    df['WG_AH_Diff'] = df['WG_AH_Home_Team'] - df['WG_AH_Away_Team']

    # Contagem dos jogos históricos utilizados
    df['WG_Confidence'] = (
        df['WG_Home_Team'].notna().astype(int) +
        df['WG_Away_Team'].notna().astype(int)
    )
    
    return df


# ========================= TARGET AH BASE =========================
def calculate_ah_home_target(row):
    """Target binário: 1 se HOME cobre o handicap (Asian_Line_Decimal pró-casa), 0 caso contrário"""
    gh = row.get("Goals_H_FT")
    ga = row.get("Goals_A_FT")
    line_home = row.get("Asian_Line_Decimal")

    if pd.isna(gh) or pd.isna(ga) or pd.isna(line_home):
        return np.nan

    adjusted = (gh + line_home) - ga
    return 1 if adjusted > 0 else 0  # push entra como 0 (conservador)

# ========================= MOMENTUM TIME =========================
def calcular_momentum_time(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Calcula MT_H e MT_A com base no HandScore_Home e HandScore_Away (rolling + z-score).
    Se não tiver HandScore, só retorna colunas preenchidas com 0.
    """
    df = df.copy()

    if 'MT_H' not in df.columns:
        df['MT_H'] = np.nan
    if 'MT_A' not in df.columns:
        df['MT_A'] = np.nan

    if not {'Home', 'Away'}.issubset(df.columns):
        df['MT_H'] = 0
        df['MT_A'] = 0
        return df

    if 'HandScore_Home' not in df.columns or 'HandScore_Away' not in df.columns:
        # Sem HandScore, MT = 0
        df['MT_H'] = 0
        df['MT_A'] = 0
        return df

    all_teams = pd.unique(df[['Home', 'Away']].values.ravel())

    for team in all_teams:
        # Mandante
        mask_home = df['Home'] == team
        if mask_home.sum() > 2:
            series = df.loc[mask_home, 'HandScore_Home'].astype(float).rolling(window, min_periods=2).mean()
            mu = series.mean()
            sigma = series.std(ddof=0) or 1
            zscore = (series - mu) / sigma
            df.loc[mask_home, 'MT_H'] = zscore

        # Visitante
        mask_away = df['Away'] == team
        if mask_away.sum() > 2:
            series = df.loc[mask_away, 'HandScore_Away'].astype(float).rolling(window, min_periods=2).mean()
            mu = series.mean()
            sigma = series.std(ddof=0) or 1
            zscore = (series - mu) / sigma
            df.loc[mask_away, 'MT_A'] = zscore

    df['MT_H'] = df['MT_H'].fillna(0)
    df['MT_A'] = df['MT_A'].fillna(0)
    return df

# ========================= MÉTRICAS 3D + NORMALIZAÇÃO =========================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas espaciais 3D com:
    - normalização por liga (z-score) para Aggression, M, MT
    - dx, dy, dz entre Home e Away
    - distância, ângulo, seno, cosseno, magnitude
    """
    df = df.copy()

    required_cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    missing_cols = [c for c in required_cols if c not in df.columns]

    # Se faltar algo, inicializa colunas de saída e retorna
    out_cols = [
        'dx', 'dy', 'dz',
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Angle_XY', 'Quadrant_Sin_XY', 'Quadrant_Cos_XY',
        'Vector_Sign', 'Magnitude_3D'
    ]

    if missing_cols:
        st.warning(f"⚠️ Colunas faltando para cálculo 3D: {missing_cols}")
        for col in out_cols:
            if col not in df.columns:
                df[col] = 0
        return df

    # Normalização por liga (z-score) para estabilizar espaço
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

        # Separação média e magnitude (redundantes mas úteis)
        df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3.0
        df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)

        for col in out_cols:
            df[col] = df[col].fillna(0)

    except Exception as e:
        st.error(f"❌ Erro no cálculo 3D: {e}")
        for col in out_cols:
            df[col] = 0

    return df

# ========================= CLUSTERIZAÇÃO 3D DINÂMICA =========================
def aplicar_clusterizacao_3d(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Clusterização 3D usando (dx, dy, dz) com escolha dinâmica de K via Silhouette.
    """
    df = df.copy()

    required_cols = ['dx', 'dy', 'dz']
    if not all(c in df.columns for c in required_cols):
        df = calcular_distancias_3d(df)

    X_cluster = df[['dx', 'dy', 'dz']].fillna(0).to_numpy()

    # Poucos dados -> cluster único
    if len(df) < 20:
        df['Cluster3D_Label'] = 0
        df['Cluster3D_Desc'] = '📉 Poucos dados'
        return df

    try:
        best_k = None
        best_score = -1

        max_k = min(8, len(df) - 1)
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

        if best_k is None:
            best_k = 4  # fallback

        kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        df['Cluster3D_Label'] = labels

        # Mapeamento descritivo simples por posição do cluster
        desc_map = {}
        for c in range(best_k):
            mask = df['Cluster3D_Label'] == c
            if not mask.any():
                desc_map[c] = '🌀 Outro'
                continue
            mean_dx = df.loc[mask, 'dx'].mean()
            mean_dz = df.loc[mask, 'dz'].mean()
            if mean_dx > 0.5 and mean_dz >= 0:
                desc = '⚡ Agressivo + Momentum Positivo'
            elif mean_dx < -0.5 and mean_dz < 0:
                desc = '🔻 Dominado + Momentum Negativo'
            elif abs(mean_dx) < 0.3 and abs(mean_dz) < 0.3:
                desc = '⚖️ Equilibrado'
            else:
                desc = '🔥 Alta Variância'
            desc_map[c] = desc

        df['Cluster3D_Desc'] = df['Cluster3D_Label'].map(desc_map).fillna('🌀 Outro')
        st.info(f"📦 Clusterização 3D concluída com K = {best_k} (Silhouette: {best_score:.3f})")

    except Exception as e:
        st.error(f"❌ Erro na clusterização 3D: {e}")
        df['Cluster3D_Label'] = 0
        df['Cluster3D_Desc'] = 'Erro'

    return df

# ========================= OTIMIZAÇÃO DE ÂNGULO =========================
def encontrar_angulo_otimo(history: pd.DataFrame,
                           target_col: str = 'Target_AH_Home',
                           min_samples: int = 150) -> int:
    """
    Encontra o melhor ângulo-limite para separar zonas estáveis/instáveis,
    maximizando diferença de acurácia com volume mínimo.
    """
    st.markdown("### 🔍 Otimizando Ângulo Espacial")

    if 'Quadrant_Angle_XY' not in history.columns:
        history = calcular_distancias_3d(history)

    if 'Quadrant_Angle_XY' not in history.columns:
        st.warning("⚠️ Não foi possível calcular ângulo. Usando 40° como padrão.")
        return 40

    if target_col not in history.columns:
        st.warning(f"⚠️ Target {target_col} não encontrado. Usando 40°.")
        return 40

    if len(history) < min_samples:
        st.warning(f"⚠️ Dados insuficientes para otimização ({len(history)} < {min_samples}). Usando 40°.")
        return 40

    resultados = []
    angulos_testar = range(10, 80, 5)
    progress_bar = st.progress(0)
    total = len(angulos_testar)

    for i, ang in enumerate(angulos_testar):
        progress_bar.progress((i + 1) / total)

        mask_estavel = history['Quadrant_Angle_XY'].abs() < ang
        mask_instavel = ~mask_estavel

        # Estável
        if mask_estavel.sum() >= min_samples:
            estavel = history[mask_estavel]
            acc_e = estavel[target_col].mean()
            vol_e = len(estavel)
            roi_e = (acc_e * 0.90 - (1 - acc_e)) * 100
        else:
            acc_e = vol_e = roi_e = 0

        # Instável
        if mask_instavel.sum() >= min_samples:
            instavel = history[mask_instavel]
            acc_i = instavel[target_col].mean()
            vol_i = len(instavel)
            roi_i = (acc_i * 0.90 - (1 - acc_i)) * 100
        else:
            acc_i = vol_i = roi_i = 0

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
            'roi_estavel': roi_e,
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
        score = df_validos.loc[idx, 'score_qualidade']

        st.success(f"🎯 Ângulo ótimo encontrado: {ang_otimo}° (Score {score:.4f})")

        top_5 = df_validos.nlargest(5, 'score_qualidade')[
            ['angulo_limite', 'acuracia_estavel', 'volume_estavel',
             'acuracia_instavel', 'volume_instavel', 'diferenca_acuracia']
        ].round(4)

        st.markdown("#### 📊 Top 5 Ângulos por Performance")
        st.dataframe(
            top_5.style.format({
                'acuracia_estavel': '{:.1%}',
                'acuracia_instavel': '{:.1%}',
                'diferenca_acuracia': '{:.1%}'
            }),
            use_container_width=True
        )

        # Gráfico simples (sem frescura)
        fig, ax = plt.subplots()
        ax.plot(df_res['angulo_limite'], df_res['acuracia_estavel'], marker='o', label='Zona Estável')
        ax.plot(df_res['angulo_limite'], df_res['acuracia_instavel'], marker='s', label='Zona Instável')
        ax.axvline(x=ang_otimo, linestyle='--', label=f'Ângulo Ótimo {ang_otimo}°')
        ax.set_xlabel("Ângulo Limite (°)")
        ax.set_ylabel("Acurácia")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        return ang_otimo

    st.warning("⚠️ Nenhum ângulo consistente encontrado. Usando 40°.")
    return 40

# ========================= SCORE ESPACIAL INTELIGENTE =========================
def calcular_score_espacial_inteligente(row, angulo_limite: float) -> float:
    """
    Score contínuo [0,1] baseado em geometria:
    >0.5 tende HOME, <0.5 tende AWAY.
    Usa dx, dz, ângulo e cluster como reforço.
    """
    dx = row.get('dx', 0)
    dy = row.get('dy', 0)
    dz = row.get('dz', 0)
    ang_xy = row.get('Quadrant_Angle_XY', 0)
    cluster = row.get('Cluster3D_Label', 0)

    # Falhas -> neutro
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

    # Cluster confiável puxa levemente pró-padrão
    if cluster in [0]:  # cluster agressivo positivo
        score += 0.04
    elif cluster in [1]:  # cluster negativo
        score -= 0.04

    return float(np.clip(score, 0.05, 0.95))

# ========================= TREINAMENTO ESPACIAL V2 =========================
def treinar_modelo_espacial_inteligente(history: pd.DataFrame,
                                        games_today: pd.DataFrame):
    st.markdown("## 🧠 Treinando Modelo Espacial Inteligente (V2)")

    # 1) Métricas 3D e clusterização
    st.info("📐 Calculando métricas 3D e clusters...")
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)

    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    # 2) Otimizar ângulo usando Target_AH_Home (base real)
    if 'Target_AH_Home' not in history.columns:
        st.error("❌ Target_AH_Home não encontrado no histórico.")
        return None, games_today

    angulo_otimo = encontrar_angulo_otimo(history, target_col='Target_AH_Home')

    # 3) Score & Target Espacial
    st.info(f"🎯 Aplicando Score Espacial com ângulo {angulo_otimo}°")
    history['Score_Espacial'] = history.apply(
        lambda x: calcular_score_espacial_inteligente(x, angulo_otimo), axis=1
    )
    history['Target_Espacial'] = (history['Score_Espacial'] >= 0.5).astype(int)

    dist_target = history['Target_Espacial'].value_counts(normalize=True).to_dict()
    st.info(f"📊 Distribuição Target_Espacial: { {k: f'{v:.1%}' for k,v in dist_target.items()} }")

    # 4) Features
    # 4) Features - ATUALIZADA COM WG
    features_espaciais = [
        'Quadrant_Dist_3D', 'Quadrant_Separation_3D',
        'Quadrant_Sin_XY', 'Quadrant_Cos_XY', 'Quadrant_Angle_XY',
        'Vector_Sign', 'Magnitude_3D',
        'dx', 'dy', 'dz',
        'Score_Espacial',
        'Cluster3D_Label',
        # NOVAS FEATURES WG
        'WG_Home_Team', 'WG_Away_Team', 
        'WG_AH_Home_Team', 'WG_AH_Away_Team',
        'WG_Diff', 'WG_AH_Diff', 'WG_Confidence'
    ]
    
    features_espaciais = [f for f in features_espaciais if f in history.columns]

    if 'League' in history.columns:
        ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    else:
        ligas_dummies = pd.DataFrame(index=history.index)

    X = pd.concat([ligas_dummies, history[features_espaciais]], axis=1).fillna(0)
    y = history['Target_Espacial'].astype(int)

    if len(history) < 100:
        st.error("❌ Dados insuficientes para treinamento consistente (<100 jogos).")
        return None, games_today

    # 5) Balanceamento simples (upsample da minoria)
    counts = y.value_counts()
    if len(counts) == 2:
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        df_major = history[history['Target_Espacial'] == majority_class]
        df_minor = history[history['Target_Espacial'] == minority_class]

        if len(df_minor) > 20:
            df_minor_up = resample(df_minor,
                                   replace=True,
                                   n_samples=len(df_major),
                                   random_state=42)
            hist_bal = pd.concat([df_major, df_minor_up], ignore_index=True)

            if 'League' in hist_bal.columns:
                ligas_bal = pd.get_dummies(hist_bal['League'], prefix='League')
            else:
                ligas_bal = pd.DataFrame(index=hist_bal.index)

            X_bal = pd.concat([ligas_bal, hist_bal[features_espaciais]], axis=1).fillna(0)
            y_bal = hist_bal['Target_Espacial'].astype(int)

            st.info(f"⚖️ Dataset balanceado: {len(df_major)} x {len(df_minor_up)}")
        else:
            st.warning("⚠️ Minoria muito pequena, usando dataset original.")
            X_bal, y_bal = X, y
    else:
        X_bal, y_bal = X, y

    # Alinhar colunas (caso tenha diferença entre X e X_bal)
    X_cols = X_bal.columns

    # 6) Modelo
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(X_bal, y_bal)

    # 7) Importância das features (top 15)
    try:
        importances = pd.Series(model.feature_importances_, index=X_cols).sort_values(ascending=False)
        st.markdown("### 🧩 Principais Features do Modelo")
        st.dataframe(importances.head(15).round(4).to_frame("Importância"), use_container_width=True)
    except Exception:
        pass

    # 8) Preparar dados de hoje
    if 'League' in games_today.columns:
        ligas_today = pd.get_dummies(games_today['League'], prefix='League')
    else:
        ligas_today = pd.DataFrame(index=games_today.index)

    # Garantir mesmas colunas
    for col in X_cols:
        if col not in ligas_today.columns and col not in games_today.columns:
            # feature ausente -> cria 0
            if col.startswith("League_"):
                ligas_today[col] = 0

    X_today = pd.concat(
        [ligas_today.reindex(columns=[c for c in X_cols if c.startswith("League_")], fill_value=0),
         games_today.reindex(columns=[c for c in X_cols if c not in ligas_today.columns], fill_value=0)],
        axis=1
    )

    # Reordenar
    X_today = X_today.reindex(columns=X_cols, fill_value=0)

    # Score espacial de hoje
    games_today['Score_Espacial'] = games_today.apply(
        lambda x: calcular_score_espacial_inteligente(x, angulo_otimo), axis=1
    )

    # 9) Previsão
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

    mean_conf = games_today['Confidence_Espacial'].mean()
    home_preds = (games_today['ML_Side_Espacial'] == 'HOME').sum()
    away_preds = (games_today['ML_Side_Espacial'] == 'AWAY').sum()

    st.success(f"✅ Modelo treinado em {len(history)} jogos históricos.")
    st.success(f"🏟️ {len(games_today)} jogos hoje | 🎯 Confiança média: {mean_conf:.1%} | 🏠 HOME: {home_preds} | ✈️ AWAY: {away_preds}")

    return model, games_today

# ========================= EXIBIÇÃO RESULTADOS =========================
def exibir_resultados_espaciais(games_today: pd.DataFrame):
    st.markdown("## 📊 Resultados do Modelo Espacial Inteligente (V2)")

    cols_display = [
        'League', 'Home', 'Away', 'Asian_Line_Decimal',
        'Goals_H_Today', 'Goals_A_Today',
        'Prob_Espacial', 'ML_Side_Espacial',
        'Confidence_Espacial', 'Score_Espacial', 'Angulo_Otimizado',
        # NOVAS COLUNAS WG
        'WG_Home_Team', 'WG_Away_Team', 'WG_Diff', 'WG_Confidence'
    ]
    cols_display = [c for c in cols_display if c in games_today.columns]

    df_show = games_today[cols_display].copy()

    # Formatação
    if 'Prob_Espacial' in df_show.columns:
        df_show['Prob_Espacial'] = df_show['Prob_Espacial'].apply(lambda x: f"{x:.1%}")
    if 'Confidence_Espacial' in df_show.columns:
        df_show['Confidence_Espacial'] = df_show['Confidence_Espacial'].apply(lambda x: f"{x:.1%}")
    if 'Score_Espacial' in df_show.columns:
        df_show['Score_Espacial'] = df_show['Score_Espacial'].apply(lambda x: f"{x:.3f}")
    
    # Formatar colunas WG
    wg_cols = ['WG_Home_Team', 'WG_Away_Team', 'WG_Diff']
    for col in wg_cols:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(lambda x: f"{x:.2f}")

    if 'Confidence_Espacial' in games_today.columns:
        df_show = df_show.sort_values('Confidence_Espacial', ascending=False)

    st.dataframe(df_show, use_container_width=True)

    # Métricas adicionais com WG
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Jogos", len(games_today))
    with col2:
        st.metric("🏠 Recomendações HOME", int((games_today['ML_Side_Espacial'] == 'HOME').sum()))
    with col3:
        st.metric("✈️ Recomendações AWAY", int((games_today['ML_Side_Espacial'] == 'AWAY').sum()))
    with col4:
        if 'Confidence_Espacial' in games_today.columns:
            st.metric("🎯 Confiança Média", f"{games_today['Confidence_Espacial'].mean():.1%}")
    with col5:
        if 'WG_Confidence' in games_today.columns:
            avg_wg_conf = games_today['WG_Confidence'].mean()
            st.metric("📈 Conf. WG Média", f"{avg_wg_conf:.1f}")

# ========================= MAIN =========================
def main():
    st.sidebar.markdown("## ⚙️ Configurações")

    if not os.path.exists(GAMES_FOLDER):
        st.error(f"Pasta {GAMES_FOLDER} não encontrada.")
        return

    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    if not files:
        st.error("❌ Nenhum arquivo CSV encontrado na pasta GamesDay")
        return

    options = files[-7:] if len(files) >= 7 else files
    selected_file = st.sidebar.selectbox("Selecionar Arquivo:", options, index=len(options) - 1)

    @st.cache_data(ttl=3600)
    def load_cached_data(sel_file: str):
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, sel_file))
        games_today = filter_leagues(games_today)
    
        history = filter_leagues(load_all_games(GAMES_FOLDER))
        history = preprocess_df(history)
    
        # Garantir colunas essenciais
        if 'Asian_Line' in history.columns:
            history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
        if 'Asian_Line' in games_today.columns:
            games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)
    
        history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT", "Asian_Line_Decimal"]).copy()
    
        # Target base AH Home
        history["Target_AH_Home"] = history.apply(calculate_ah_home_target, axis=1)
        history = history.dropna(subset=["Target_AH_Home"]).copy()
        history["Target_AH_Home"] = history["Target_AH_Home"].astype(int)
    
        # ============ SISTEMA WEIGHTED GOALS ============
        st.info("📊 Calculando Weighted Goals...")
        
        # Aplicar WG no histórico (com resultados conhecidos)
        history = adicionar_weighted_goals(history)
        history = adicionar_weighted_goals_ah(history)
        history = calcular_rolling_wg(history)
        
        # ============ MERGE PARA JOGOS DO DIA ============
        st.info("🔄 Aplicando WG nos jogos de hoje...")
        games_today = calcular_wg_para_jogos_do_dia(history, games_today)
        # ============ FIM MERGE WG ============
    
        # Momentum (se tiver HandScore)
        history_mt = calcular_momentum_time(history)
        games_today_mt = calcular_momentum_time(games_today)
    
        return games_today_mt, history_mt

    games_today, history = load_cached_data(selected_file)

    # LiveScore
    def load_and_merge_livescore(games_today_df: pd.DataFrame, selected_date_str: str):
        games_today_df = setup_livescore_columns(games_today_df)
        livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

        if os.path.exists(livescore_file):
            try:
                results_df = pd.read_csv(livescore_file)
                results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]

                required_cols = ['Id', 'status', 'home_goal', 'away_goal', 'home_red', 'away_red']
                if all(c in results_df.columns for c in required_cols):
                    merged = games_today_df.merge(
                        results_df[['Id', 'status', 'home_goal', 'away_goal', 'home_red', 'away_red']],
                        on='Id', how='left'
                    )
                    merged.loc[merged['status'] == 'FT', 'Goals_H_Today'] = merged['home_goal']
                    merged.loc[merged['status'] == 'FT', 'Goals_A_Today'] = merged['away_goal']
                    merged['Home_Red'] = merged['home_red']
                    merged['Away_Red'] = merged['away_red']
                    st.success("✅ LiveScore integrado")
                    return merged
            except Exception:
                st.warning("⚠️ Erro ao integrar LiveScore.")
        return games_today_df

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
    games_today = load_and_merge_livescore(games_today, selected_date_str)

    st.info(f"📊 Carregados: {len(games_today)} jogos de hoje | {len(history)} jogos históricos válidos")


    st.markdown("""
    ## 🎯 Sistema Espacial Inteligente V2
    - 🧠 Otimização automática do ângulo espacial (baseado no histórico real AH)
    - 📐 Métricas 3D normalizadas por liga (Aggression, M, MT)
    - 📦 Clusterização 3D dinâmica (K otimizado por Silhouette)
    - 🎯 Target Espacial derivado de **score contínuo geométrico**
    - 📊 **Sistema Weighted Goals (WG)** - performance vs expectativa de mercado
    - 🔄 **WG em Tempo Real** - médias históricas aplicadas aos jogos do dia
    - ⚖️ Balanceamento automático de dados
    """)

    if st.sidebar.button("🚀 Treinar Modelo Espacial", type="primary"):
        with st.spinner("Treinando modelo espacial inteligente V2..."):
            model, resultados = treinar_modelo_espacial_inteligente(history, games_today)
            if model is not None:
                exibir_resultados_espaciais(resultados)
                out_path = f"resultados_espacial_V2_{selected_date_str}.csv"
                resultados.to_csv(out_path, index=False)
                st.success(f"💾 Resultados salvos em: {out_path}")
                st.balloons()
            else:
                st.error("❌ Falha no treinamento do modelo.")
    else:
        st.info("👆 Clique em **'Treinar Modelo Espacial'** para gerar as recomendações do dia.")

# ========================= RUN =========================
if __name__ == "__main__":
    main()
