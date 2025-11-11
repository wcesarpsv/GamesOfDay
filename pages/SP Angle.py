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
# 🔧 FUNÇÕES BÁSICAS (idênticas às da V2 exceto onde indicado)
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


# =====================================================================
# 📊 CÁLCULO ESPACIAL COM JULGAMENTO DE MERCADO (VERSÃO ROBUSTA V3.1)
# =====================================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas espaciais 3D + detecção de julgamento de mercado
    - Normalização robusta por liga (z-score)
    - Cálculo de dx, dy, dz e métricas geométricas
    - Cálculo de Diff_Judgment (gap entre percepção e momento real)
    """
    df = df.copy()

    # ------------------ Garantir colunas básicas ------------------
    cols = ['Aggression_Home', 'Aggression_Away', 'M_H', 'M_A', 'MT_H', 'MT_A']
    for c in cols:
        if c not in df.columns:
            df[c] = 0

    # ------------------ Normalização robusta por liga ------------------
    cols_norm = cols.copy()

    if 'League' in df.columns:
        normed_blocks = []

        for league, g in df.groupby('League', group_keys=False):
            # Se a liga tem só NaN ou 1 jogo, não normaliza
            if len(g) < 2 or g[cols_norm].dropna(how='all').empty:
                g_norm = g[cols_norm].fillna(0)
            else:
                g_norm = g[cols_norm].apply(
                    lambda x: (x - x.mean()) / (x.std(ddof=0) or 1)
                )
            normed_blocks.append(g_norm)

        # Concatenar resultados e restaurar ordem original
        df[cols_norm] = pd.concat(normed_blocks, axis=0).sort_index()

    else:
        # Sem coluna de liga — normalização global
        df[cols_norm] = df[cols_norm].apply(
            lambda x: (x - x.mean()) / (x.std(ddof=0) or 1)
        )

    # ------------------ Cálculo vetorial 3D ------------------
    df['dx'] = df['Aggression_Home'] - df['Aggression_Away']
    df['dy'] = df['M_H'] - df['M_A']
    df['dz'] = df['MT_H'] - df['MT_A']

    df['Quadrant_Dist_3D'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2 + df['dz'] ** 2)

    angle_xy = np.arctan2(df['dy'], df['dx'])
    df['Quadrant_Angle_XY'] = np.degrees(angle_xy)
    df['Quadrant_Sin_XY'] = np.sin(angle_xy)
    df['Quadrant_Cos_XY'] = np.cos(angle_xy)
    df['Vector_Sign'] = np.sign(df['dx'] * df['dy'] * df['dz']).fillna(0)
    df['Quadrant_Separation_3D'] = (df['dx'] + df['dy'] + df['dz']) / 3.0
    df['Magnitude_3D'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2 + df['dz'] ** 2)

    # ------------------ Cálculo de distorção de julgamento ------------------
    df['Judgment_Discrepancy_H'] = (df['Aggression_Home'] * -1) * (
        df['M_H'] + df['MT_H']
    )
    df['Judgment_Discrepancy_A'] = (df['Aggression_Away'] * -1) * (
        df['M_A'] + df['MT_A']
    )
    df['Diff_Judgment'] = (
        df['Judgment_Discrepancy_H'] - df['Judgment_Discrepancy_A']
    )

    # ------------------ Segurança final ------------------
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


# =====================================================================
# ⚡ CLUSTERIZAÇÃO 3D (idêntica à V2 resumida)
# =====================================================================
def aplicar_clusterizacao_3d(df,random_state=42):
    df=df.copy()
    X=df[['dx','dy','dz']].fillna(0)
    best_k,best_score=2,-1
    for k in range(2,min(8,len(df))):
        try:
            sc=silhouette_score(X,KMeans(k,random_state=42,n_init=10).fit_predict(X))
            if sc>best_score: best_k, best_score=k, sc
        except: continue
    km=KMeans(best_k,random_state=42,n_init=10)
    df['Cluster3D_Label']=km.fit_predict(X)
    return df

# =====================================================================
# 🧮 SCORE ESPACIAL AJUSTADO POR JULGAMENTO
# =====================================================================
def calcular_score_espacial_inteligente(row,angulo):
    dx,dy,dz=row.get('dx',0),row.get('dy',0),row.get('dz',0)
    ang_xy=row.get('Quadrant_Angle_XY',0)
    diff=row.get('Diff_Judgment',0)
    cluster=row.get('Cluster3D_Label',0)
    dist=math.sqrt(dx**2+dy**2+dz**2)
    score=0.5
    # pesos simétricos
    score+=np.sign(dx)*0.12
    score+=np.sign(dz)*0.10
    if abs(ang_xy)<angulo: score+=np.sign(dx)*0.08
    else: score-=np.sign(dx)*0.08
    # ajuste por julgamento
    if diff>0: score+=0.05
    elif diff<0: score-=0.05
    # cluster refinamento
    if cluster==0: score+=0.02
    elif cluster==1: score-=0.02
    # distancia amortecida
    if dist<0.4: score=0.5+(score-0.5)*0.4
    return float(np.clip(score,0.05,0.95))

# =====================================================================
# 🎯 TREINAMENTO E EXIBIÇÃO (resumo)
# =====================================================================
def treinar_modelo_espacial_inteligente(history,games_today):
    st.subheader("Treinando Modelo Market Judgment V3")
    history=calcular_distancias_3d(history)
    games_today=calcular_distancias_3d(games_today)
    history=aplicar_clusterizacao_3d(history)
    games_today=aplicar_clusterizacao_3d(games_today)
    ang=40
    history['Score_Espacial']=history.apply(lambda x:calcular_score_espacial_inteligente(x,ang),axis=1)
    history['Target_Espacial']=(history['Score_Espacial']>=0.5).astype(int)
    features=['dx','dy','dz','Diff_Judgment','Quadrant_Dist_3D','Magnitude_3D','Score_Espacial','Cluster3D_Label']
    X=history[features].fillna(0); y=history['Target_Espacial']
    model=RandomForestClassifier(n_estimators=200,max_depth=8,class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(X,y)
    proba=np.clip(model.predict_proba(games_today[features].fillna(0))[:,1],0.05,0.95)
    games_today['Prob_Espacial']=proba
    games_today['ML_Side_Espacial']=np.where(proba>=0.5,'HOME','AWAY')
    games_today['Confidence_Espacial']=np.maximum(proba,1-proba)
    # ---- tabela de julgamento invertido ----
    st.markdown("### 🧭 Top 10 Confrontos de Julgamento Invertido")
    top=games_today[['League','Home','Away','Diff_Judgment','ML_Side_Espacial','Confidence_Espacial']].copy()
    top['Tipo']=np.where(top['Diff_Judgment']>0,'⚡ Home Subestimado','🔻 Home Overvalued')
    st.dataframe(top.sort_values('Diff_Judgment',ascending=False).head(10),use_container_width=True)
    st.success("✅ Modelo Market Judgment V3 treinado!")
    return model,games_today

# =====================================================================
# 🚀 MAIN
# =====================================================================
def main():
    st.sidebar.markdown("## Configurações V3")
    files=[f for f in os.listdir(GAMES_FOLDER) if f.endswith('.csv')]
    if not files: st.error("Nenhum CSV encontrado em GamesDay"); return
    fsel=st.sidebar.selectbox("Arquivo:",sorted(files),index=len(files)-1)
    df=pd.read_csv(os.path.join(GAMES_FOLDER,fsel))
    df=filter_leagues(df)
    if 'Asian_Line' in df.columns:
        df['Asian_Line_Decimal']=df['Asian_Line'].apply(convert_asian_line_to_decimal)
    if 'Goals_H_FT' not in df.columns: df['Goals_H_FT']=np.nan; df['Goals_A_FT']=np.nan
    df['Target_AH_Home']=df.apply(calculate_ah_home_target,axis=1)
    history=df.dropna(subset=['Target_AH_Home']).copy()
    games_today=df.copy()
    if st.sidebar.button("🚀 Treinar V3"):
        model,res=treinar_modelo_espacial_inteligente(history,games_today)
        st.dataframe(res[['Home','Away','Prob_Espacial','ML_Side_Espacial','Confidence_Espacial']].sort_values('Confidence_Espacial',ascending=False),use_container_width=True)
    else:
        st.info("👆 Clique em Treinar V3 para rodar o detector de julgamento de mercado.")

if __name__=="__main__":
    main()
