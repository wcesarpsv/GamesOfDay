# ==============================================================
# 🎯 Previsão Over/Under 2.5 & BTTS + Live Validation
# ==============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ============================ Config ============================
st.set_page_config(page_title="Previsão Over/Under 2.5 & BTTS", layout="wide")
st.title("🎯 Previsão Over/Under 2.5 & BTTS + Live Validation")

GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(GAMES_FOLDER, exist_ok=True)
os.makedirs(LIVESCORE_FOLDER, exist_ok=True)

# ============================ Helpers ============================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

# --- Resultados Over/Under e BTTS ---
def result_over25_from_ft(gh, ga):
    if pd.isna(gh) or pd.isna(ga):
        return np.nan
    return 1 if (gh + ga) > 2.5 else 0

def result_btts_from_ft(gh, ga):
    if pd.isna(gh) or pd.isna(ga):
        return np.nan
    return 1 if (gh > 0 and ga > 0) else 0

def result_over25_from_today(row):
    gh = row.get('Goals_H_Today', np.nan)
    ga = row.get('Goals_A_Today', np.nan)
    if pd.isna(gh) or pd.isna(ga):
        return None
    return 1 if (gh + ga) > 2.5 else 0

def result_btts_from_today(row):
    gh = row.get('Goals_H_Today', np.nan)
    ga = row.get('Goals_A_Today', np.nan)
    if pd.isna(gh) or pd.isna(ga):
        return None
    return 1 if (gh > 0 and ga > 0) else 0

# ====================== 3D features (diferenças, trig, magnitude) ======================
def calcular_distancias_3d(df):
    df = df.copy()
    for c in ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']:
        if c not in df.columns:
            df[c] = 0.0
    dx = df['Aggression_Home'] - df['Aggression_Away']
    dy = df['M_H'] - df['M_A']
    dz = df['MT_H'] - df['MT_A']
    df['Quadrant_Dist_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    # ângulos e projeções
    a_xy = np.arctan2(dy, dx)
    a_xz = np.arctan2(dz, dx)
    a_yz = np.arctan2(dz, np.where(dy==0, 1e-9, dy))
    df['Quadrant_Sin_XY'] = np.sin(a_xy); df['Quadrant_Cos_XY'] = np.cos(a_xy)
    df['Quadrant_Sin_XZ'] = np.sin(a_xz); df['Quadrant_Cos_XZ'] = np.cos(a_xz)
    df['Quadrant_Sin_YZ'] = np.sin(a_yz); df['Quadrant_Cos_YZ'] = np.cos(a_yz)
    combo = a_xy + a_xz + a_yz
    df['Quadrant_Sin_Combo'] = np.sin(combo); df['Quadrant_Cos_Combo'] = np.cos(combo)
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = np.sqrt(dx**2 + dy**2 + dz**2)
    return df

def aplicar_clusterizacao_3d(df, n_clusters=5, random_state=42):
    df = df.copy()
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']
    Xc = df[['dx','dy','dz']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df['Cluster3D_Label'] = kmeans.fit_predict(Xc)
    return df

# ====================== LiveScore (merge) ======================
def setup_livescore_columns(df):
    if 'Goals_H_Today' not in df.columns: df['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in df.columns: df['Goals_A_Today'] = np.nan
    if 'Home_Red'      not in df.columns: df['Home_Red']      = np.nan
    if 'Away_Red'      not in df.columns: df['Away_Red']      = np.nan
    return df

def load_and_merge_livescore(games_today, selected_date_str):
    games_today = setup_livescore_columns(games_today)
    fp = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")
    if not os.path.exists(fp):
        st.warning(f"⚠️ LiveScore não encontrado para {selected_date_str} em {fp}")
        return games_today
    try:
        raw = pd.read_csv(fp)
    except Exception as e:
        st.error(f"Erro ao ler LiveScore: {e}")
        return games_today

    if 'status' in raw.columns:
        raw = raw[~raw['status'].isin(['Cancel', 'Postp.'])]

    left_key = 'Id' if 'Id' in games_today.columns else None
    right_key = 'Id' if 'Id' in raw.columns else None
    if left_key and right_key:
        games_today = games_today.merge(raw, left_on=left_key, right_on=right_key, how='left', suffixes=('', '_RAW'))

    # mapeia colunas comuns
    for src, dst in [('home_goal','Goals_H_Today'), ('away_goal','Goals_A_Today'),
                     ('home_red','Home_Red'), ('away_red','Away_Red')]:
        if src in games_today.columns:
            games_today[dst] = games_today[src]

    if 'status' in games_today.columns:
        games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today','Goals_A_Today']] = np.nan

    st.success(f"✅ LiveScore mesclado ({len(raw)} linhas)")
    return games_today

# ============================ Carregamento ============================
st.info("📂 Carregando dados para Over/Under 2.5 & BTTS...")

files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
if not files:
    st.warning("No CSV files found in GamesDay.")
    st.stop()

options = files[-7:] if len(files) >= 7 else files
selected_file = st.selectbox("Selecione o arquivo do dia:", options, index=len(options)-1)

m = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
selected_date_str = m.group(0) if m else datetime.now().strftime("%Y-%m-%d")

games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected_file))
games_today = filter_leagues(games_today)

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

# filtro temporal
if "Date" in history.columns:
    try:
        sel_date = pd.to_datetime(selected_date_str)
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < sel_date].copy()
    except Exception as e:
        st.warning(f"Erro ao aplicar filtro temporal: {e}")

# LiveScore para validação em tempo real
games_today = load_and_merge_livescore(games_today, selected_date_str)

# ===================== Targets Over/Under 2.5 & BTTS =====================
st.markdown("### 🎯 Construindo targets para Over/Under 2.5 & BTTS")

def build_overunder_btts_targets(df_hist: pd.DataFrame):
    df = df_hist.copy()
    
    # Targets Over/Under 2.5
    df['Target_Over25'] = [result_over25_from_ft(h, a) for h, a in zip(df['Goals_H_FT'], df['Goals_A_FT'])]
    df['Target_Under25'] = 1 - df['Target_Over25']
    
    # Targets BTTS
    df['Target_BTTS_Yes'] = [result_btts_from_ft(h, a) for h, a in zip(df['Goals_H_FT'], df['Goals_A_FT'])]
    df['Target_BTTS_No'] = 1 - df['Target_BTTS_Yes']
    
    return df

history = build_overunder_btts_targets(history)

st.info(
    f"Histórico: {len(history)} jogos | "
    f"Over 2.5: {history['Target_Over25'].mean():.1%} | "
    f"BTTS Yes: {history['Target_BTTS_Yes'].mean():.1%}"
)

# ===================== Feature engineering 3D =====================
def ensure_3d_features(df):
    df = calcular_distancias_3d(df)
    df = aplicar_clusterizacao_3d(df, n_clusters=5)
    return df

history = ensure_3d_features(history)
games_today = ensure_3d_features(games_today)

# ===================== Treinamento – Dois Modelos (Over25 & BTTS) =====================
st.markdown("### 🤖 Treinando modelos para Over/Under 2.5 & BTTS")

def train_overunder_btts_models(history, games_today):
    # dummies categóricas
    ligas_d = pd.get_dummies(history['League'], prefix='League') if 'League' in history.columns else pd.DataFrame()
    clusters_d = pd.get_dummies(history['Cluster3D_Label'], prefix='C3D')

    feat3d = [
        'Quadrant_Dist_3D','Quadrant_Separation_3D',
        'Quadrant_Sin_XY','Quadrant_Cos_XY','Quadrant_Sin_XZ','Quadrant_Cos_XZ',
        'Quadrant_Sin_YZ','Quadrant_Cos_YZ',
        'Quadrant_Sin_Combo','Quadrant_Cos_Combo','Vector_Sign','Magnitude_3D','OverScore_Home','OverScore_Away'
    ]
    extras = history[feat3d].fillna(0)

    X = pd.concat([ligas_d, clusters_d, extras], axis=1)

    # targets
    y_over = history['Target_Over25']; mask_over = y_over.notna()
    y_btts = history['Target_BTTS_Yes']; mask_btts = y_btts.notna()

    # Modelos de classificação
    model_over = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    model_btts = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)

    if mask_over.any(): 
        model_over.fit(X.loc[mask_over], y_over.loc[mask_over])
    if mask_btts.any(): 
        model_btts.fit(X.loc[mask_btts], y_btts.loc[mask_btts])

    # preparar X_today
    ligas_today = pd.get_dummies(games_today['League'], prefix='League') if 'League' in games_today.columns else pd.DataFrame()
    ligas_today = ligas_today.reindex(columns=ligas_d.columns, fill_value=0)
    clusters_today = pd.get_dummies(games_today['Cluster3D_Label'], prefix='C3D').reindex(columns=clusters_d.columns, fill_value=0)
    extras_today = games_today[feat3d].fillna(0)
    X_today = pd.concat([ligas_today, clusters_today, extras_today], axis=1)

    # previsões de probabilidade
    if mask_over.any():
        prob_over = model_over.predict_proba(X_today)[:, 1]
        games_today['Prob_Over'] = prob_over
        games_today['Prob_Under'] = 1 - prob_over
        games_today['Odd_Justa_Over'] = 1 / prob_over
        games_today['Odd_Justa_Under'] = 1 / (1 - prob_over)
    else:
        games_today['Prob_Over'] = 0.5
        games_today['Prob_Under'] = 0.5
        games_today['Odd_Justa_Over'] = 2.0
        games_today['Odd_Justa_Under'] = 2.0

    if mask_btts.any():
        prob_btts = model_btts.predict_proba(X_today)[:, 1]
        games_today['Prob_BTTS_Yes'] = prob_btts
        games_today['Prob_BTTS_No'] = 1 - prob_btts
        games_today['Odd_Justa_BTTS_Yes'] = 1 / prob_btts
        games_today['Odd_Justa_BTTS_No'] = 1 / (1 - prob_btts)
    else:
        games_today['Prob_BTTS_Yes'] = 0.5
        games_today['Prob_BTTS_No'] = 0.5
        games_today['Odd_Justa_BTTS_Yes'] = 2.0
        games_today['Odd_Justa_BTTS_No'] = 2.0

    # Importâncias dos modelos
    try:
        importances_over = pd.Series(model_over.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        st.markdown("#### 🔍 Top Features (Modelo Over/Under 2.5)")
        st.dataframe(importances_over.to_frame("Importância"), use_container_width=True)
    except Exception:
        pass

    st.success("✅ Modelos Over/Under 2.5 & BTTS treinados")
    return model_over, model_btts, games_today

model_over, model_btts, games_today = train_overunder_btts_models(history, games_today)

# ===================== Tabela Principal =====================
st.markdown("## 📊 Previsões Over/Under 2.5 & BTTS")

# Ordenar por Time (horário)
if 'Time' in games_today.columns:
    games_today = games_today.sort_values('Time')

# Selecionar e formatar colunas
display_cols = [
    'League', 'Time', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
    'Prob_Over', 'Prob_Under', 'Odd_Justa_Over', 'Odd_Justa_Under',
    'Prob_BTTS_Yes', 'Prob_BTTS_No', 'Odd_Justa_BTTS_Yes', 'Odd_Justa_BTTS_No'
]

available_cols = [col for col in display_cols if col in games_today.columns]
df_display = games_today[available_cols].copy()

# Formatar percentuais e odds
format_dict = {}
for col in df_display.columns:
    if col.startswith('Prob_'):
        df_display[col] = df_display[col] * 100
        format_dict[col] = '{:.1f}%'
    elif col.startswith('Odd_Justa_'):
        format_dict[col] = '{:.2f}'

st.dataframe(
    df_display.style.format(format_dict),
    use_container_width=True
)

# ===================== Live Validation =====================
st.markdown("## 📡 Live Validation - Resultados Reais")

# Calcular resultados reais
games_today['Real_Over25'] = games_today.apply(result_over25_from_today, axis=1)
games_today['Real_BTTS'] = games_today.apply(result_btts_from_today, axis=1)

# Filtrar jogos finalizados
finished_games = games_today[games_today['Real_Over25'].notna() & games_today['Real_BTTS'].notna()]

if not finished_games.empty:
    # WINRATE INDIVIDUAL PARA CADA OPÇÃO
    st.markdown("### 🎯 Winrate por Tipo de Aposta")
    
    # Over/Under
    over_bets = finished_games[finished_games['Prob_Over'] > 0.5]
    under_bets = finished_games[finished_games['Prob_Under'] > 0.5]
    
    over_winrate = (over_bets['Real_Over25'] == 1).mean() if len(over_bets) > 0 else 0
    under_winrate = (under_bets['Real_Over25'] == 0).mean() if len(under_bets) > 0 else 0
    
    # BTTS
    btts_yes_bets = finished_games[finished_games['Prob_BTTS_Yes'] > 0.5]
    btts_no_bets = finished_games[finished_games['Prob_BTTS_No'] > 0.5]
    
    btts_yes_winrate = (btts_yes_bets['Real_BTTS'] == 1).mean() if len(btts_yes_bets) > 0 else 0
    btts_no_winrate = (btts_no_bets['Real_BTTS'] == 0).mean() if len(btts_no_bets) > 0 else 0
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Over 2.5 Winrate", 
                f"{over_winrate:.1%}", 
                f"{len(over_bets)} apostas")
    col2.metric("🎯 Under 2.5 Winrate", 
                f"{under_winrate:.1%}", 
                f"{len(under_bets)} apostas")
    col3.metric("🎯 BTTS Yes Winrate", 
                f"{btts_yes_winrate:.1%}", 
                f"{len(btts_yes_bets)} apostas")
    col4.metric("🎯 BTTS No Winrate", 
                f"{btts_no_winrate:.1%}", 
                f"{len(btts_no_bets)} apostas")
    
    # Tabela detalhada de performance
    st.markdown("### 📊 Performance Detalhada por Tipo")
    
    performance_data = {
        'Tipo': ['Over 2.5', 'Under 2.5', 'BTTS Yes', 'BTTS No'],
        'Apostas': [len(over_bets), len(under_bets), len(btts_yes_bets), len(btts_no_bets)],
        'Winrate': [over_winrate, under_winrate, btts_yes_winrate, btts_no_winrate],
        'Acertos': [
            (over_bets['Real_Over25'] == 1).sum() if len(over_bets) > 0 else 0,
            (under_bets['Real_Over25'] == 0).sum() if len(under_bets) > 0 else 0,
            (btts_yes_bets['Real_BTTS'] == 1).sum() if len(btts_yes_bets) > 0 else 0,
            (btts_no_bets['Real_BTTS'] == 0).sum() if len(btts_no_bets) > 0 else 0
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(
        performance_df.style.format({
            'Winrate': '{:.1%}',
            'Apostas': '{:.0f}',
            'Acertos': '{:.0f}'
        }).background_gradient(subset=['Winrate'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Tabela de resultados individuais
    st.markdown("### 🔍 Resultados Individuais dos Jogos")
    validation_cols = [
        'League', 'Time', 'Home', 'Away', 'Goals_H_Today', 'Goals_A_Today',
        'Prob_Over', 'Real_Over25', 'Prob_BTTS_Yes', 'Real_BTTS'
    ]
    available_val_cols = [col for col in validation_cols if col in finished_games.columns]
    
    st.dataframe(
        finished_games[available_val_cols].style.format({
            'Prob_Over': '{:.1f}%',
            'Prob_BTTS_Yes': '{:.1f}%'
        }),
        use_container_width=True
    )
else:
    st.info("⏳ Aguardando jogos finalizados para validação...")

st.markdown("---")
st.success("🎯 Sistema Over/Under 2.5 & BTTS implementado com sucesso!")
st.info("""
**Funcionalidades:**
- 📊 Previsões de probabilidade para Over/Under 2.5 e BTTS
- 💰 Cálculo de odds justas baseadas nas probabilidades
- 🕒 Ordenação por horário dos jogos
- 📡 Validação em tempo real com gols do dia
- 🔍 Features 3D (Aggression × M × MT) + clusters KMeans
- 📈 Estatísticas de acerto para ambos os mercados
""")
