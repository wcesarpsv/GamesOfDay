# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import os, re, math, joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="DUAL Asian Handicap Predictor", layout="wide")
st.title("🎯 Dual Perspective Asian Handicap — HOME vs AWAY")

# ========================= CONSTANTES =========================
PAGE_PREFIX = "DUAL_AH_V4"
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup","copas","coppa","uefa","afc","sudamericana","copa","trophy"]

# ============================================================
# 🔧 FUNÇÕES AUXILIARES
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
        df = df.rename(columns={"Goals_H_FT_x":"Goals_H_FT","Goals_A_FT_x":"Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y":"Goals_H_FT","Goals_A_FT_y":"Goals_A_FT"})
    return df

def load_all_games(folder: str) -> pd.DataFrame:
    if not os.path.exists(folder):
        return pd.DataFrame()
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []
    for f in files:
        try:
            dfs.append(preprocess_df(pd.read_csv(os.path.join(folder,f))))
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

# ============================================================
# 🔢 CONVERSÃO DE LINHA ASIÁTICA PARA DECIMAL (PADRÃO HOME)
# ============================================================
def convert_asian_line_to_decimal(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()

    # se não é split
    if "/" not in s:
        try:
            num = float(s)
            # padroniza: negativo = favorece HOME
            return -num
        except:
            return np.nan

    # split ex: "-0.5/1"
    try:
        parts = [float(p) for p in s.replace("+","").replace("-","").split("/")]
        avg = np.mean(parts)
        sign = -1 if s.startswith("-") else 1
        result = sign * avg
        return -result
    except:
        return np.nan

# ============================================================
# 🧭 TARGET HÍBRIDO (C) PARA REGRESSÃO
# ============================================================
def continuous_target_home(row):
    """ h = -(GH - GA), com clipping e suavização """
    margin = row['Goals_H_FT'] - row['Goals_A_FT']
    h = -margin  # home dá/tira gols
    # clipping
    h = np.clip(h, -1.75, 1.75)
    # suavização leve
    if abs(h) > 1.25:
        h *= 0.75
    elif abs(h) > 1.0:
        h *= 0.85
    return h

def continuous_target_away(row):
    """ h = -(GA - GH), com clipping e suavização """
    margin = row['Goals_A_FT'] - row['Goals_H_FT']
    h = -margin
    h = np.clip(h, -1.75, 1.75)
    if abs(h) > 1.25:
        h *= 0.75
    elif abs(h) > 1.0:
        h *= 0.85
    return h

# ============================================================
# 🟩 CATEGORIZAÇÃO (5 CLASSES)
# ============================================================
def categorize_home_target(h):
    if h <= -0.75: return "MODERATE_HOME"
    elif h <= -0.25: return "LIGHT_HOME"
    elif h < 0.25: return "NEUTRAL"
    elif h < 0.75: return "LIGHT_AWAY"
    else: return "MODERATE_AWAY"

def categorize_away_target(h):
    if h <= -0.75: return "MODERATE_AWAY"
    elif h <= -0.25: return "LIGHT_AWAY"
    elif h < 0.25: return "NEUTRAL"
    elif h < 0.75: return "LIGHT_HOME"
    else: return "MODERATE_HOME"

# ============================================================
# 🧮 MATRIZ 3D (DX, DY, DZ)
# ============================================================
def calcular_distancias_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['Aggression_Home','Aggression_Away','M_H','M_A','MT_H','MT_A']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Faltam colunas para 3D: {missing}")
        for c in ['Quadrant_Dist_3D','Quadrant_Separation_3D','Vector_Sign','Magnitude_3D']:
            df[c] = np.nan
        return df

    dx = (df['Aggression_Home'] - df['Aggression_Away']) / 2
    dy = (df['M_H'] - df['M_A']) / 2
    dz = (df['MT_H'] - df['MT_A']) / 2

    df['Quadrant_Dist_3D'] = np.sqrt(dx*dx + dy*dy + dz*dz)
    df['Quadrant_Separation_3D'] = (dx + dy + dz) / 3
    df['Vector_Sign'] = np.sign(dx * dy * dz)
    df['Magnitude_3D'] = df['Quadrant_Dist_3D']

    df['dx'] = dx
    df['dy'] = dy
    df['dz'] = dz
    return df

# ============================================================
# 🔵 CLUSTERIZAÇÃO 3D
# ============================================================
def aplicar_clusterizacao_3d(df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    df = df.copy()
    required = ['dx','dy','dz']
    if any(c not in df.columns for c in required):
        df['Cluster3D_Label'] = 0
        return df

    X = df[['dx','dy','dz']].fillna(0).to_numpy()
    k = max(1, min(n_clusters, len(X)))

    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['Cluster3D_Label'] = km.fit_predict(X)
    except:
        df['Cluster3D_Label'] = 0

    return df


# ============================================================
# 🟢 HOME MODEL — Regressão (híbrido contínuo)
# ============================================================
def treinar_modelo_home_regressao(history, games_today):
    st.markdown("### 🟢 Modelo HOME — Regressão")

    hist = history.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
    hist['Target_Home_Cont'] = hist.apply(continuous_target_home, axis=1)

    # features do HOME
    features = [
        'Aggression_Home','Aggression_Away',
        'M_H','M_A','MT_H','MT_A',
        'dx','dy','dz','Cluster3D_Label'
    ]
    features = [f for f in features if f in hist.columns]

    X = hist[features].fillna(0)
    y = hist['Target_Home_Cont']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=42
    )
    model.fit(Xs, y)

    mae = mean_absolute_error(y, model.predict(Xs))
    st.success(f"MAE HOME (treino in-sample): {mae:.3f}")

    # aplicar em games_today
    X_today = games_today.copy()
    for f in features:
        if f not in X_today.columns:
            X_today[f] = 0

    preds = model.predict(scaler.transform(X_today[features].fillna(0)))
    preds = np.clip(preds, -1.75, 1.75)

    games_today['Pred_HOME_REG'] = preds

    return model, scaler, games_today


# ============================================================
# 🟢 HOME MODEL — Classificação (5 classes)
# ============================================================
def treinar_modelo_home_classificacao(history, games_today):
    st.markdown("### 🟢 Modelo HOME — Classificação")

    hist = history.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
    hist['Target_Home_Cont'] = hist.apply(continuous_target_home, axis=1)
    hist['Target_Home_Class'] = hist['Target_Home_Cont'].apply(categorize_home_target)

    features = [
        'Aggression_Home','Aggression_Away',
        'M_H','M_A','MT_H','MT_A',
        'dx','dy','dz','Cluster3D_Label'
    ]
    features = [f for f in features if f in hist.columns]

    X = hist[features].fillna(0)
    y = hist['Target_Home_Class']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y_enc)

    # aplicar em games_today
    X_today = games_today.copy()
    for f in features:
        if f not in X_today.columns:
            X_today[f] = 0

    X_td = X_today[features].fillna(0)
    pred_enc = model.predict(X_td)
    probas = model.predict_proba(X_td)

    games_today['Pred_HOME_Class_Label'] = le.inverse_transform(pred_enc)
    games_today['Pred_HOME_Class_Conf'] = np.max(probas, axis=1)

    # converter classes → valores AH
    map_class_to_num_home = {
        "MODERATE_HOME": -0.75,
        "LIGHT_HOME": -0.25,
        "NEUTRAL": 0.0,
        "LIGHT_AWAY": 0.25,
        "MODERATE_AWAY": 0.75
    }
    games_today['Pred_HOME_CLS'] = games_today['Pred_HOME_Class_Label'].map(map_class_to_num_home)

    return model, le, games_today



# ============================================================
# 🔵 AWAY MODEL — Regressão (com features invertidas)
# ============================================================
def treinar_modelo_away_regressao(history, games_today):
    st.markdown("### 🔵 Modelo AWAY — Regressão")

    hist = history.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
    hist['Target_Away_Cont'] = hist.apply(continuous_target_away, axis=1)

    # features espelhadas
    features = [
        'Aggression_Away','Aggression_Home',
        'M_A','M_H','MT_A','MT_H',
        # dx invertido
        # dy invertido
        # dz invertido
        # cluster igual (mas usada no contexto away)
        'Cluster3D_Label'
    ]

    # criar dx_AWY, dy_AWY, dz_AWY
    hist['dx_AWY'] = hist['Aggression_Away'] - hist['Aggression_Home']
    hist['dy_AWY'] = hist['M_A'] - hist['M_H']
    hist['dz_AWY'] = hist['MT_A'] - hist['MT_H']

    games_today['dx_AWY'] = games_today['Aggression_Away'] - games_today['Aggression_Home']
    games_today['dy_AWY'] = games_today['M_A'] - games_today['M_H']
    games_today['dz_AWY'] = games_today['MT_A'] - games_today['MT_H']

    features += ['dx_AWY','dy_AWY','dz_AWY']
    features = [f for f in features if f in hist.columns]

    X = hist[features].fillna(0)
    y = hist['Target_Away_Cont']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=42
    )
    model.fit(Xs, y)

    mae = mean_absolute_error(y, model.predict(Xs))
    st.success(f"MAE AWAY (treino in-sample): {mae:.3f}")

    # aplicar no today
    for f in features:
        if f not in games_today.columns:
            games_today[f] = 0

    preds = model.predict(scaler.transform(games_today[features].fillna(0)))
    preds = np.clip(preds, -1.75, 1.75)

    games_today['Pred_AWAY_REG_RAW'] = preds  # AWAY nativo
    games_today['Pred_AWAY_REG'] = -preds     # convertido para eixo HOME

    return model, scaler, games_today


# ============================================================
# 🔵 AWAY MODEL — Classificação (5 classes, features invertidas)
# ============================================================
def treinar_modelo_away_classificacao(history, games_today):
    st.markdown("### 🔵 Modelo AWAY — Classificação")

    hist = history.dropna(subset=['Goals_H_FT','Goals_A_FT']).copy()
    hist['Target_Away_Cont'] = hist.apply(continuous_target_away, axis=1)
    hist['Target_Away_Class'] = hist['Target_Away_Cont'].apply(categorize_away_target)

    # features espelhadas
    features = [
        'Aggression_Away','Aggression_Home',
        'M_A','M_H','MT_A','MT_H',
        'Cluster3D_Label'
    ]

    # dx/dy/dz invertidos
    hist['dx_AWY'] = hist['Aggression_Away'] - hist['Aggression_Home']
    hist['dy_AWY'] = hist['M_A'] - hist['M_H']
    hist['dz_AWY'] = hist['MT_A'] - hist['MT_H']

    games_today['dx_AWY'] = games_today['Aggression_Away'] - games_today['Aggression_Home']
    games_today['dy_AWY'] = games_today['M_A'] - games_today['M_H']
    games_today['dz_AWY'] = games_today['MT_A'] - games_today['MT_H']

    features += ['dx_AWY','dy_AWY','dz_AWY']
    features = [f for f in features if f in hist.columns]

    X = hist[features].fillna(0)
    y = hist['Target_Away_Class']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y_enc)

    # aplicar no today
    for f in features:
        if f not in games_today.columns:
            games_today[f] = 0

    X_td = games_today[features].fillna(0)
    pred_enc = model.predict(X_td)
    probas = model.predict_proba(X_td)

    games_today['Pred_AWAY_Class_Label_RAW'] = le.inverse_transform(pred_enc)
    games_today['Pred_AWAY_Class_Conf'] = np.max(probas, axis=1)

    # map invertido → eixo HOME
    map_class_home_axis = {
        "MODERATE_AWAY": -0.75,
        "LIGHT_AWAY": -0.25,
        "NEUTRAL": 0.0,
        "LIGHT_HOME": 0.25,
        "MODERATE_HOME": 0.75
    }

    games_today['Pred_AWAY_CLS'] = games_today['Pred_AWAY_Class_Label_RAW'].map(map_class_home_axis)

    return model, le, games_today


# ============================================================
# 🧮 DUAL COMBINATION
# ============================================================
def combinar_preds(row):
    # HOME
    w_cls_home = 0.2 + 0.3 * row['Pred_HOME_Class_Conf']
    w_reg_home = 1 - w_cls_home
    pred_home = (
        w_reg_home * row['Pred_HOME_REG'] +
        w_cls_home * row['Pred_HOME_CLS']
    )

    # AWAY
    w_cls_away = 0.2 + 0.3 * row['Pred_AWAY_Class_Conf']
    w_reg_away = 1 - w_cls_away
    pred_away_home_axis = (
        w_reg_away * row['Pred_AWAY_REG'] +
        w_cls_away * row['Pred_AWAY_CLS']
    )

    return pred_home, pred_away_home_axis


# ============================================================
# 📊 ENGINE DE RECOMENDAÇÃO FINAL
# ============================================================
def analisar_value_bets_dual(games_today, league_thresholds):
    results = []
    for _, row in games_today.iterrows():

        pred_home, pred_away = combinar_preds(row)

        asian = float(row['Asian_Line_Decimal'])

        vg_home = pred_home - asian
        vg_away = pred_away - asian

        lg = row['League']
        thr_pack = league_thresholds.get(lg, league_thresholds['_GLOBAL'])

        thrH = thr_pack['HOME']
        thrHS = thr_pack['HOME_STRONG']
        thrA = thr_pack['AWAY']
        thrAS = thr_pack['AWAY_STRONG']

        rec = "NO BET"
        conf = "LOW"

        if vg_home >= thrHS:
            rec, conf = "STRONG HOME", "HIGH"
        elif vg_home >= thrH:
            rec, conf = "BET HOME", "MEDIUM"
        elif vg_away >= thrAS:
            rec, conf = "STRONG AWAY", "HIGH"
        elif vg_away >= thrA:
            rec, conf = "BET AWAY", "MEDIUM"

        results.append({
            'League': lg,
            'Home': row['Home'],
            'Away': row['Away'],
            'Asian_Line': row['Asian_Line'],
            'Asian_Line_Decimal': asian,
            'Pred_HOME': pred_home,
            'Pred_AWAY_HOME_AXIS': pred_away,
            'VG_HOME': vg_home,
            'VG_AWAY': vg_away,
            'Rec': rec,
            'Confidence': conf
        })

    return pd.DataFrame(results)


# ============================================================
# 📈 Plot Dual
# ============================================================
def plot_dual(g):
    fig, ax = plt.subplots(figsize=(12,6))
    asian = g['Asian_Line_Decimal'].tolist()
    pred_home = g['Pred_HOME'].tolist()
    pred_away = g['Pred_AWAY_HOME_AXIS'].tolist()

    fair_line = (g['Pred_HOME'] + g['Pred_AWAY_HOME_AXIS'])/2

    ax.scatter(asian, pred_home, s=70, label="HOME")
    ax.scatter(asian, pred_away, s=70, label="AWAY→HOME")
    ax.scatter(asian, fair_line, s=80, marker='x', label="Fair Line Dual")

    ax.plot([-2,2],[-2,2],'k--',alpha=0.3)

    ax.set_title("Dual Handicap Predictions")
    ax.set_xlabel("Market Asian Line (HOME)")
    ax.set_ylabel("Predicted Handicap (HOME Axis)")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


# ============================================================
# 🚀 MAIN COMPLETO
# ============================================================
def main_calibrado():

    st.header("⚙️ Preparando Dados...")

    if not os.path.exists(GAMES_FOLDER):
        st.error("GamesDay folder not found.")
        return

    files = sorted([f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")])
    selected_file = st.selectbox("Selecione o arquivo do dia:", files, index=len(files)-1)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    selected_date_str = date_match.group(0)

    games_today = preprocess_df(pd.read_csv(os.path.join(GAMES_FOLDER, selected_file)))
    history = load_all_games(GAMES_FOLDER)

    games_today = filter_leagues(games_today)
    history = filter_leagues(history)

    history['Asian_Line_Decimal'] = history['Asian_Line'].apply(convert_asian_line_to_decimal)
    games_today['Asian_Line_Decimal'] = games_today['Asian_Line'].apply(convert_asian_line_to_decimal)

    history.dropna(subset=['Asian_Line_Decimal'], inplace=True)
    games_today.dropna(subset=['Asian_Line_Decimal'], inplace=True)

    # filtrar passado
    history['Date'] = pd.to_datetime(history['Date'], errors='coerce')
    cutoff = pd.to_datetime(selected_date_str)
    history = history[history['Date'] < cutoff].copy()

    # espaço 3d
    history = calcular_distancias_3d(history)
    games_today = calcular_distancias_3d(games_today)

    history = aplicar_clusterizacao_3d(history)
    games_today = aplicar_clusterizacao_3d(games_today)

    st.subheader("🧠 Treinando Modelos...")

    # HOME
    model_home_reg, scaler_home, games_today = treinar_modelo_home_regressao(history, games_today)
    model_home_cls, le_home, games_today = treinar_modelo_home_classificacao(history, games_today)

    # AWAY
    model_away_reg, scaler_away, games_today = treinar_modelo_away_regressao(history, games_today)
    model_away_cls, le_away, games_today = treinar_modelo_away_classificacao(history, games_today)

    st.subheader("📊 Gerando Valor Final")

    # calcular thresholds
    league_thresholds = {
        "_GLOBAL": {
            "HOME": 0.15,
            "HOME_STRONG": 0.30,
            "AWAY": 0.15,
            "AWAY_STRONG": 0.30
        }
    }
    # (você pode adicionar a função de thresholds customizada aqui)

    df_final = analisar_value_bets_dual(games_today, league_thresholds)
    st.dataframe(df_final, use_container_width=True)

    st.pyplot(plot_dual(df_final))

    st.success("Finalizado!")

if __name__ == "__main__":
    main_calibrado()

