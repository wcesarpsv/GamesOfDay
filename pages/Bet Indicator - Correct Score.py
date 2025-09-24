# ########################################################
# Bloco 1 – Imports & Config
# ########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bet Indicator – Correct Score", layout="wide")
st.title("📊 Bet Indicator – Correct Score (0x0 - 3x3)")

# Paths
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "copas", "uefa", "nordeste", "afc"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


# ########################################################
# Bloco 2 – Funções auxiliares
# ########################################################
def load_all_games(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(os.path.join(folder, f)) for f in files], ignore_index=True)

def load_selected_csvs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    
    today_file = files[-1]
    yesterday_file = files[-2] if len(files) >= 2 else None

    st.markdown("### 📂 Select matches to display")
    col1, col2 = st.columns(2)
    today_checked = col1.checkbox("Today Matches", value=True)
    yesterday_checked = col2.checkbox("Yesterday Matches", value=False)

    selected_dfs = []
    if today_checked:
        selected_dfs.append(pd.read_csv(os.path.join(folder, today_file)))
    if yesterday_checked and yesterday_file:
        selected_dfs.append(pd.read_csv(os.path.join(folder, yesterday_file)))

    if not selected_dfs:
        return pd.DataFrame()
    return pd.concat(selected_dfs, ignore_index=True)

def filter_leagues(df):
    if df.empty or "League" not in df.columns:
        return df
    pattern = "|".join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df["League"].str.lower().str.contains(pattern, na=False)].copy()

def save_model(model, filename):
    path = os.path.join(MODELS_FOLDER, filename)
    with open(path, "wb") as f:
        joblib.dump(model, f)

def load_model(filename):
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return joblib.load(f)
    return None


# ########################################################
# Bloco 3 – Carregar Dados
# ########################################################
st.info("📂 Loading data...")

history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

if history.empty:
    st.error("⚠️ No valid historical data found in GamesDay.")
    st.stop()

games_today = filter_leagues(load_selected_csvs(GAMES_FOLDER))
if "Goals_H_FT" in games_today.columns:
    games_today = games_today[games_today["Goals_H_FT"].isna()].copy()

if games_today.empty:
    st.error("⚠️ No valid matches selected.")
    st.stop()


# ########################################################
# Bloco 4 – Target Score
# ########################################################
# 🔹 Criar target com placar exato (0x0 até 3x3)
history["Target_Score"] = (
    history["Goals_H_FT"].astype(int).astype(str) + "-" + history["Goals_A_FT"].astype(int).astype(str)
)
history = history[(history["Goals_H_FT"] <= 3) & (history["Goals_A_FT"] <= 3)].copy()


# ########################################################
# Bloco 5 – Features & One-Hot Leagues
# ########################################################
history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]

features_score = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P"]

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X_score = pd.concat([history[features_score], history_leagues], axis=1)
X_today_score = pd.concat([games_today[features_score], games_today_leagues], axis=1)


# ########################################################
# Bloco 6 – Configurações ML (Sidebar)
# ########################################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Random Forest", "XGBoost Tuned"]
)
retrain = st.sidebar.checkbox("Retrain model", value=False)


# ########################################################
# Bloco 7 – Treino & Avaliação 
# ########################################################

from sklearn.preprocessing import LabelEncoder

def train_and_evaluate(X, y, name):
    # 🔹 Nome do arquivo inclui tipo de modelo + target
    filename = f"{ml_model_choice.replace(' ', '').replace('-', '')}_{name}CC.pkl"
    model = None

    # 🔹 LabelEncoder para padronizar classes
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Se não for para re-treinar, tenta carregar o modelo já salvo
    if not retrain:
        model = load_model(filename)

    if model is None:
        if ml_model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced_subsample"
            )
            # split de treino/validação
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )
            model.fit(X_train, y_train)

        elif ml_model_choice == "XGBoost Tuned":
            model = XGBClassifier(
                n_estimators=300,          # reduzido para ser mais rápido
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=42,
                use_label_encoder=False
            )

            # split de treino/validação
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )

            # treino normal (sem early stopping)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # 🔹 Sempre salvar como tuple (model, le)
        save_model((model, le), filename)

    else:
        try:
            model, le = model  # garantir tuple
        except:
            model = model
            le = LabelEncoder().fit(y)

        # re-divisão para validação
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

    # 🔹 Avaliação
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)

    metrics = {
        "Model": f"{ml_model_choice} - {name}",
        "Accuracy": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
    }

    return metrics, (model, le)




# ########################################################
# Bloco 8 – Treinar Modelo Score
# ########################################################
stats = []
res, model_score = train_and_evaluate(X_score, history["Target_Score"], "ExactScore")
stats.append(res)

df_stats = pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)


# ########################################################
# Bloco 9 – Previsões de Placar com estilo tabular
# ########################################################
model, le = model_score
probas_score = model.predict_proba(X_today_score)
score_classes = le.inverse_transform(np.arange(probas_score.shape[1]))

# Criar DataFrame com odds + probabilidades por placar
df_preds_full = games_today[["Date","Time","League","Home","Away","Odd_H","Odd_D","Odd_A"]].copy()

# adicionar uma coluna para cada placar
for idx, score in enumerate(score_classes):
    df_preds_full[score] = probas_score[:, idx]

# 🔹 Função de cor para probabilidades
def color_prob(val):
    if pd.isna(val):
        return ""
    return f"background-color: rgba(0,200,0,{val:.2f})"

# 🔹 Formatação separada: odds (2 decimais), placares (%)
fmt_dict = {col: "{:.2f}" for col in ["Odd_H","Odd_D","Odd_A"]}
fmt_dict.update({col: "{:.1%}" for col in score_classes})

# 🔹 Aplicar estilo
styled_df = (
    df_preds_full.style
    .format(fmt_dict)
    .applymap(color_prob, subset=score_classes)  # só aplica cores nos placares
)

st.markdown("### 📌 Predictions for Selected Matches (Exact Score Probabilities)")
st.dataframe(styled_df, use_container_width=True, height=800)


