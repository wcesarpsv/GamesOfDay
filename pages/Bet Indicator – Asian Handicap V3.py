##################### BLOCO 1 – IMPORTS & CONFIG #####################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import io
import zipfile
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta

st.set_page_config(page_title="Bet Indicator – Asian Handicap V3", layout="wide")
st.title("📊 Bet Indicator – Asian Handicap (V3 – Enriched Features)")

PAGE_PREFIX = "AsianHandicap"
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup","copas","uefa","afc","sudamericana","copa"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


##################### BLOCO 2 – HELPERS #####################
def preprocess_df(df):
    df = df.copy()
    if "Goals_H_FT_x" in df.columns:
        df = df.rename(columns={"Goals_H_FT_x": "Goals_H_FT", "Goals_A_FT_x": "Goals_A_FT"})
    elif "Goals_H_FT_y" in df.columns:
        df = df.rename(columns={"Goals_H_FT_y": "Goals_H_FT", "Goals_A_FT_y": "Goals_A_FT"})
    return df

def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    dfs = [preprocess_df(pd.read_csv(os.path.join(folder, f))) for f in files]
    return pd.concat(dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, feature_cols, filename):
    with open(os.path.join(MODELS_FOLDER, filename), "wb") as f:
        joblib.dump((model, feature_cols), f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None

def offer_models_download(model_files):
    """
    Cria um botão no sidebar para baixar todos os modelos (.pkl) em um único ZIP.
    Pega direto da pasta Models para evitar ZIP vazio.
    """
    if not model_files:
        st.sidebar.warning("Nenhum modelo disponível para download ainda.")
        return

    # Garante que todos os arquivos apontem para a pasta Models
    files_to_zip = [os.path.join(MODELS_FOLDER, os.path.basename(f)) for f in model_files]

    # Filtra apenas os arquivos que realmente existem
    files_to_zip = [f for f in files_to_zip if os.path.exists(f)]

    if not files_to_zip:
        st.sidebar.warning("Nenhum modelo salvo encontrado na pasta Models.")
        return

    # Cria ZIP em memória
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for file in files_to_zip:
            zf.write(file, os.path.basename(file))
    zip_buffer.seek(0)

    # Botão de download
    st.sidebar.download_button(
        label="⬇️ Download all models (ZIP)",
        data=zip_buffer,
        file_name="asian_handicap_models_v3c.zip",
        mime="application/zip"
    )


##################### BLOCO 3 – LOAD DATA + HANDICAP TARGET #####################
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT","Goals_A_FT","Asian_Line"]).copy()

if set(["Date","Home","Away"]).issubset(history.columns):
    history = history.drop_duplicates(subset=["Home","Away","Goals_H_FT","Goals_A_FT"], keep="first")

today = datetime.now().strftime("%Y-%m-%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Date" in games_today.columns:
    games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
games_today = games_today[games_today["Date"] == today].copy()

include_yesterday = st.sidebar.checkbox("Include yesterday's matches", value=False)
if include_yesterday:
    games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
    if "Date" in games_today.columns:
        games_today["Date"] = pd.to_datetime(games_today["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games_today = games_today[games_today["Date"].isin([today,yesterday])].copy()

if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

def convert_asian_line(line_str):
    try:
        if pd.isna(line_str) or line_str == "":
            return None
        line_str = str(line_str).strip()
        if "/" not in line_str:
            return float(line_str)
        parts = [float(x) for x in line_str.split("/")]
        return sum(parts) / len(parts)
    except:
        return None

history["Asian_Line_Display"] = history["Asian_Line"].apply(convert_asian_line)
games_today["Asian_Line_Display"] = games_today["Asian_Line"].apply(convert_asian_line)

def calc_handicap_result(margin, asian_line_str, invert=False):
    if pd.isna(asian_line_str): return np.nan
    if invert: margin = -margin
    try:
        parts = [float(x) for x in str(asian_line_str).split("/")]
    except:
        return np.nan
    results = []
    for line in parts:
        if margin > line: results.append(1.0)
        elif margin == line: results.append(0.5)
        else: results.append(0.0)
    return np.mean(results)

history["Margin"] = history["Goals_H_FT"] - history["Goals_A_FT"]
history["Handicap_Home_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=False), axis=1)
history["Handicap_Away_Result"] = history.apply(lambda r: calc_handicap_result(r["Margin"], r["Asian_Line"], invert=True), axis=1)

# --- Targets ---
history["Target_AH_Home"] = history["Handicap_Home_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Away"] = history["Handicap_Away_Result"].apply(lambda x: 1 if x >= 0.5 else 0)
history["Target_AH_Home_strict"] = (history["Handicap_Home_Result"] == 1.0).astype(int)


##################### BLOCO 4 – FEATURE ENGINEERING #####################
# (mantém o mesmo código de feature engineering que você já tinha: odds, momentum, bands, weights...)


##################### BLOCO 5 – SIDEBAR CONFIG #####################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
retrain = st.sidebar.checkbox("Retrain models", value=False)
normalize_features = st.sidebar.checkbox("Normalize features (odds + strength)", value=False)

calibration_choice = st.sidebar.selectbox(
    "Calibration method",
    ["sigmoid", "isotonic", "none"],
    index=0
)


##################### BLOCO 6 – TRAIN & EVALUATE #####################
def train_and_evaluate_v2(X, y, name):
    safe_name = name.replace(" ", "")
    safe_model = ml_model_choice.replace(" ", "")
    filename = f"{PAGE_PREFIX}_{safe_model}_{safe_name}_2C_v3c.pkl"
    feature_cols = X.columns.tolist()

    if not retrain:
        loaded = load_model(filename)
        if loaded:
            model, cols = loaded
            preds = model.predict(X)
            probs = model.predict_proba(X)
            res = {
                "Model": f"{name}_v3c (loaded)",
                "Accuracy": accuracy_score(y, preds),
                "LogLoss": log_loss(y, probs),
                "BrierScore": brier_score_loss(y, probs[:, 1])
            }
            return res, (model, cols, filename)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if normalize_features:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if ml_model_choice == "Random Forest":
        base_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    else:
        base_model = XGBClassifier(
            n_estimators=300,
            tree_method="hist",
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            scale_pos_weight=(sum(y == 0) / sum(y == 1)) if sum(y == 1) > 0 else 1
        )

    if calibration_choice == "none":
        base_model.fit(X_train, y_train)
        model = base_model
    else:
        try:
            model = CalibratedClassifierCV(
                estimator=base_model,
                method=calibration_choice,
                cv=2
            )
        except TypeError:
            model = CalibratedClassifierCV(
                base_estimator=base_model,
                method=calibration_choice,
                cv=2
            )
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    res = {
        "Model": f"{name}_v3c ({calibration_choice})",
        "Accuracy": accuracy_score(y_test, preds),
        "LogLoss": log_loss(y_test, probs),
        "BrierScore": brier_score_loss(y_test, probs[:, 1])
    }

    save_model(model, feature_cols, filename)
    return res, (model, feature_cols, filename)


##################### BLOCO 7 – TRAINING MODELS #####################
stats = []
all_model_files = []

# --- Home model (sempre strict) ---
y_home = history["Target_AH_Home_strict"]
res, model_ah_home_v3c = train_and_evaluate_v2(X_ah_home, y_home, "AH_Home_v3")
stats.append(res); all_model_files.append(model_ah_home_v3c[2])

# --- Away model ---
res, model_ah_away_v3c = train_and_evaluate_v2(X_ah_away, history["Target_AH_Away"], "AH_Away_v3")
stats.append(res); all_model_files.append(model_ah_away_v3c[2])

stats_df = pd.DataFrame(stats)[["Model","Accuracy","LogLoss","BrierScore"]]
st.markdown("### 📊 Model Statistics (Validation) – v3c (Calibrated)")
st.dataframe(stats_df, use_container_width=True)

offer_models_download(all_model_files)


##################### BLOCO 8 – PREDICTIONS #####################
model_ah_home, cols1, _ = model_ah_home_v3c
model_ah_away, cols2, _ = model_ah_away_v3c

X_today_ah_home = X_today_ah_home.reindex(columns=cols1, fill_value=0)
X_today_ah_away = X_today_ah_away.reindex(columns=cols2, fill_value=0)

if normalize_features:
    scaler = StandardScaler()
    scaler.fit(X_ah_home[numeric_cols])
    X_today_ah_home[numeric_cols] = scaler.transform(X_today_ah_home[numeric_cols])
    X_today_ah_away[numeric_cols] = scaler.transform(X_today_ah_away[numeric_cols])

if not games_today.empty:
    probs_home = model_ah_home.predict_proba(X_today_ah_home)
    for idx, col in enumerate(["p_ah_home_no","p_ah_home_yes"]):
        games_today[col] = probs_home[:, idx]

    probs_away = model_ah_away.predict_proba(X_today_ah_away)
    for idx, col in enumerate(["p_ah_away_no","p_ah_away_yes"]):
        games_today[col] = probs_away[:, idx]

def color_prob(val, color):
    if pd.isna(val): return ""
    alpha = float(np.clip(val,0,1))
    return f"background-color: rgba({color},{alpha:.2f})"

# Seleção de colunas para exibição
cols_to_show = [
    "Date","Time","League","Home","Away",
    "Odd_H","Odd_D","Odd_A",
    "Asian_Line_Display","Odd_H_Asi","Odd_A_Asi",
    "p_ah_home_yes","p_ah_away_yes",
    "Band_Diff","Band_Weight","Band_Weight_Dynamic","Weight_Source"
]

styled_df = (
    games_today[cols_to_show]
    .style.format({
        "Odd_H":"{:.2f}","Odd_D":"{:.2f}","Odd_A":"{:.2f}",
        "Asian_Line_Display":"{:.2f}",
        "Odd_H_Asi":"{:.2f}","Odd_A_Asi":"{:.2f}",
        "p_ah_home_yes":"{:.1%}","p_ah_away_yes":"{:.1%}",
        "Band_Diff":"{:.0f}","Band_Weight":"{:.2f}","Band_Weight_Dynamic":"{:.2f}"
    }, na_rep="—")
    .applymap(lambda v: color_prob(v,"0,200,0"), subset=["p_ah_home_yes"])
    .applymap(lambda v: color_prob(v,"255,140,0"), subset=["p_ah_away_yes"])
)

# --- Mostrar previsões (segundo na tela) ---
st.markdown("### 📌 Predictions for Today's Matches – Asian Handicap (v3c Calibrated + Band Weights)")
st.dataframe(styled_df, use_container_width=True, height=800)


##################### BLOCO 9 – DISTRIBUIÇÃO DOS TARGETS #####################
st.markdown("### 📊 Distribuição dos Targets (Home vs Away)")

dist_home_strict = history["Target_AH_Home_strict"].value_counts(normalize=True).rename("Target_AH_Home_strict")
dist_away = history["Target_AH_Away"].value_counts(normalize=True).rename("Target_AH_Away")

dist_df = pd.concat([dist_home_strict, dist_away], axis=1).fillna(0).T
st.dataframe(dist_df.style.format("{:.2%}"), use_container_width=True)


##################### BLOCO 10 – HISTORICAL VALIDATION #####################
st.markdown("### 📊 Historical Validation – Band Cross vs Handicap Result (per League)")

band_eval = history.dropna(subset=["Target_AH_Home","Target_AH_Away"]).copy()
band_eval["Band_Cross"] = band_eval["Home_Band"] + " vs " + band_eval["Away_Band"]

league_band_summary = (
    band_eval.groupby(["League","Band_Cross"])
    .agg(
        Games=("Target_AH_Home","count"),
        Home_AH_Winrate=("Target_AH_Home","mean"),
        Away_AH_Winrate=("Target_AH_Away","mean"),
        Avg_BandDiff=("Band_Diff","mean")
    )
    .reset_index()
)

league_band_summary["Home_AH_Winrate"] = (league_band_summary["Home_AH_Winrate"]*100).round(1)
league_band_summary["Away_AH_Winrate"] = (league_band_summary["Away_AH_Winrate"]*100).round(1)

league_band_summary["Dynamic_Band_Weight"] = np.where(
    league_band_summary["Games"] >= 10,
    (league_band_summary["Home_AH_Winrate"] - league_band_summary["Away_AH_Winrate"]) / 100.0,
    np.nan
).round(2)

st.dataframe(
    league_band_summary.sort_values(["League","Games"], ascending=[True,False]),
    use_container_width=True
)
