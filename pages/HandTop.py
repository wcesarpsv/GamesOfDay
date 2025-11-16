# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ========================= CONFIG GLOBAL =========================
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copas", "uefa", "afc", "sudamericana", "copa", "trophy"]


# ============================================================
# 🔧 FUNÇÕES AUXILIARES
# ============================================================

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "Goals_H_FT_x": "Goals_H_FT",
        "Goals_A_FT_x": "Goals_A_FT",
        "Goals_H_FT_y": "Goals_H_FT",
        "Goals_A_FT_y": "Goals_A_FT",
    }
    for c_old, c_new in rename_map.items():
        if c_old in df.columns:
            df.rename(columns={c_old: c_new}, inplace=True)
    return df


def load_all_games(folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        return pd.DataFrame()
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            try:
                dfs.append(preprocess_df(pd.read_csv(os.path.join(folder, f))))
            except:
                pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()


# ============================================================
# ⚽ Asian Line → Perspectiva HOME
# ============================================================

def convert_asian_line_to_decimal(value):
    if pd.isna(value): return np.nan
    s = str(value).strip()
    if "/" not in s:
        try:
            return -float(s)
        except:
            return np.nan
    try:
        parts = [float(p) for p in s.replace("+", "").replace("-", "").split("/")]
        avg = np.mean(parts)
        sign = -1 if s.startswith("-") else 1
        return -(sign * avg)
    except:
        return np.nan


# ============================================================
# 📌 Weighted Goals (FULL + ROLLING)
# ============================================================

def adicionar_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    required = ['Home','Away','Date','Goals_H_FT','Goals_A_FT','Odd_H','Odd_D','Odd_A']
    for c in required:
        if c not in df.columns:
            df[c] = 0

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    def odds2(row):
        try:
            oh, od, oa = float(row['Odd_H']), float(row['Odd_D']), float(row['Odd_A'])
            inv = (1/oh)+(1/od)+(1/oa)
            return (1/oh)/inv, (1/od)/inv, (1/oa)/inv
        except:
            return 0.33,0.33,0.33

    def wg_home(row):
        p_h,_,_ = odds2(row)
        return (row['Goals_H_FT']*(1-p_h)) - (row['Goals_A_FT']*p_h)

    def wg_away(row):
        _,_,p_a = odds2(row)
        return (row['Goals_A_FT']*(1-p_a)) - (row['Goals_H_FT']*p_a)

    df['WG_Home'] = df.apply(wg_home, axis=1)
    df['WG_Away'] = df.apply(wg_away, axis=1)

    df['WG_Home_Team'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['WG_Away_Team'] = df.groupby('Away')['WG_Away'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df['WG_Diff'] = df['WG_Home_Team'] - df['WG_Away_Team']

    df['WG_Confidence'] = df.groupby('Home')['WG_Home'].transform(lambda x: x.rolling(5, min_periods=1).count())

    return df


# ============================================================
# 🎯 Targets (Handicap Home)
# ============================================================

def criar_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Asian_Line_Decimal' not in df.columns:
        return df
    margin = df['Goals_H_FT'] - df['Goals_A_FT']
    adj = margin + df['Asian_Line_Decimal']
    df['Cover_Home'] = (adj > 0).astype(int)
    df['Cover_Away'] = (adj < 0).astype(int)
    return df


# ============================================================
# 🤖 RF para Handicap
# ============================================================

def clean_features(X):
    return X.replace([np.inf,-np.inf], np.nan).fillna(0)


def treinar_hist(df, hc):
    seg = df[df['Asian_Line_Decimal'].sub(hc).abs() <= 0.25].copy()
    if len(seg) < 40: return None, None

    seg = seg.sort_values('Date')
    split = int(len(seg)*0.8)
    train = seg.iloc[:split]
    val = seg.iloc[split:]

    feats = ['WG_Home_Team','WG_Away_Team','WG_Diff']
    Xtr = clean_features(train[feats]); ytr=train['Cover_Home']
    Xv  = clean_features(val[feats]);   yv =val['Cover_Home']

    model = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=8, random_state=42)
    model.fit(Xtr, ytr)

    return model, feats


def aplicar_modelos(df, modelos):
    df = df.copy()
    df['P_Cover_Home'] = np.nan

    for hc, (model, feats) in modelos.items():
        seg = df[df['Asian_Line_Decimal'].sub(hc).abs() <= 0.25]
        if len(seg)==0: continue
        df.loc[seg.index, 'P_Cover_Home'] = model.predict_proba(clean_features(seg[feats]))[:,1]
    return df


# ============================================================
# 🚀 AIL Final Score
# ============================================================

def aplicar_AIL(df):
    df = df.copy()
    df['AIL_Value_Score'] = (df['P_Cover_Home'].fillna(0) - 0.5)*2 + df['WG_Diff'].fillna(0)*0.5

    picks=[]
    for _, r in df.iterrows():
        score=r['AIL_Value_Score']
        hc=r['Asian_Line_Decimal']
        if pd.isna(hc) or abs(score)<0.15: picks.append("PASS"); continue
        if score>0: picks.append(f"🏠 HOME {hc}")
        else: picks.append(f"✈️ AWAY {abs(hc)}")
    df['AIL_Pick']=picks
    return df


# ============================================================
# 📌 APP STREAMLIT
# ============================================================

def main():
    st.set_page_config(page_title="GetHandicap - V1", layout="wide")
    st.title("🎯 GetHandicap V1 — WG + ML + AIL")

    if not os.path.exists(GAMES_FOLDER):
        st.error("Pasta GamesDay vazia.")
        return

    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    selected = st.selectbox("Selecione o arquivo do dia", files, index=len(files)-1)

    games_today = pd.read_csv(os.path.join(GAMES_FOLDER, selected))
    history = load_all_games(GAMES_FOLDER)

    history = filter_leagues(history)
    games_today = filter_leagues(games_today)

    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

    history = adicionar_weighted_goals(history)
    games_today = adicionar_weighted_goals(games_today)

    history = criar_targets(history)

    st.success("Dados carregados com sucesso! 🚀")

    # Treinamento modelos
    handicaps = [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75]
    modelos={}
    for hc in handicaps:
        m, feats = treinar_hist(history, hc)
        if m is not None:
            modelos[hc]=(m,feats)

    # Aplicar nos jogos de hoje
    df=aplicar_modelos(games_today, modelos)
    df=aplicar_AIL(df)

    st.dataframe(
        df[['League','Home','Away','Asian_Line_Decimal','WG_Home_Team','WG_Away_Team','WG_Diff','P_Cover_Home','AIL_Value_Score','AIL_Pick']],
        use_container_width=True
    )


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()