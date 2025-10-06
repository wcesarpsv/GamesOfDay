# ########################################################
# Bloco 1 – Imports & Config
# ########################################################
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from datetime import date, timedelta, datetime
from collections import Counter

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SMOTE para balanceamento
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Bet Indicator – Triple View", layout="wide")
st.title("📊 Bet Indicator – Triple View (1X2 + OU + BTTS)")

# Paths
GAMES_FOLDER = "GamesDay"
LIVESCORE_FOLDER = "LiveScore"
EXCLUDED_LEAGUE_KEYWORDS = ["cup", "copa", "copas", "uefa", "nordeste", "afc","trophy"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


# ########################################################
# Bloco 2 – Funções auxiliares (ATUALIZADAS DO BINARY)
# ########################################################
def load_all_games(folder):
    """Carrega todos os CSVs da pasta e remove duplicados por (Date, Home, Away)."""
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df_list.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    if not df_list:
        return pd.DataFrame()
    
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all.drop_duplicates(subset=["Date", "Home", "Away","Goals_H_FT","Goals_A_FT"], keep="first")

def filter_leagues(df):
    """Remove ligas indesejadas (Copa, UEFA, etc)."""
    if df.empty or 'League' not in df.columns:
        return df
    pattern = '|'.join(EXCLUDED_LEAGUE_KEYWORDS)
    return df[~df['League'].str.lower().str.contains(pattern, na=False)].copy()

def load_selected_csvs(folder):
    """Carrega CSVs selecionados com sistema de data do Binary"""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not files:
        return pd.DataFrame()
    
    # Últimos dois arquivos (Hoje e Ontem) - igual ao código Binary
    options = files[-2:] if len(files) >= 2 else files
    selected_file = st.selectbox("Select Matchday File:", options, index=len(options)-1)
    
    # Extrair a data do arquivo selecionado (YYYY-MM-DD)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", selected_file)
    if date_match:
        selected_date_str = date_match.group(0)
    else:
        selected_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Carregar o arquivo selecionado
    games_today = pd.read_csv(os.path.join(folder, selected_file))
    games_today = filter_leagues(games_today)
    
    # ========== MERGE COM LIVESCORE (DO BINARY) ==========
    livescore_file = os.path.join(LIVESCORE_FOLDER, f"Resultados_RAW_{selected_date_str}.csv")

    # Ensure goal columns exist
    if 'Goals_H_Today' not in games_today.columns:
        games_today['Goals_H_Today'] = np.nan
    if 'Goals_A_Today' not in games_today.columns:
        games_today['Goals_A_Today'] = np.nan

    # Merge with the correct LiveScore file
    if os.path.exists(livescore_file):
        st.info(f"LiveScore file found: {livescore_file}")
        results_df = pd.read_csv(livescore_file)

        # FILTER OUT CANCELED AND POSTPONED GAMES
        results_df = results_df[~results_df['status'].isin(['Cancel', 'Postp.'])]
        
        required_cols = [
            'game_id', 'status', 'home_goal', 'away_goal',
            'home_ht_goal', 'away_ht_goal',
            'home_corners', 'away_corners',
            'home_yellow', 'away_yellow',
            'home_red', 'away_red'
        ]
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            st.error(f"The file {livescore_file} is missing these columns: {missing_cols}")
        else:
            games_today = games_today.merge(
                results_df,
                left_on='Id',
                right_on='game_id',
                how='left',
                suffixes=('', '_RAW')
            )

            # Update goals only for finished games
            games_today['Goals_H_Today'] = games_today['home_goal']
            games_today['Goals_A_Today'] = games_today['away_goal']
            games_today.loc[games_today['status'] != 'FT', ['Goals_H_Today', 'Goals_A_Today']] = np.nan
            
            # ADD RED CARD COLUMNS
            games_today['Home_Red'] = games_today['home_red']
            games_today['Away_Red'] = games_today['away_red']
    else:
        st.warning(f"No LiveScore results file found for selected date: {selected_date_str}")

    # 🔹 Mantém apenas jogos futuros (sem placares ainda) - baseado nos dados originais
    if 'Goals_H_FT' in games_today.columns:
        games_today = games_today[games_today['Goals_H_FT'].isna()].copy()

    return games_today

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
# Bloco 3 – Carregar Dados (ATUALIZADO)
# ########################################################
st.info("📂 Loading data...")

# Carregar dados históricos com função melhorada do Binary
history = filter_leagues(load_all_games(GAMES_FOLDER))
history = history.dropna(subset=["Goals_H_FT", "Goals_A_FT"]).copy()

if history.empty:
    st.error("⚠️ No valid historical data found in GamesDay.")
    st.stop()

# Carregar jogos de hoje com sistema melhorado
games_today = load_selected_csvs(GAMES_FOLDER)

if games_today.empty:
    st.error("⚠️ No valid matches selected.")
    st.stop()


# ########################################################
# Bloco 4 – Targets
# ########################################################
history["Target"] = history.apply(
    lambda row: 0 if row["Goals_H_FT"] > row["Goals_A_FT"]
    else (1 if row["Goals_H_FT"] == row["Goals_A_FT"] else 2),
    axis=1,
)
history["Target_OU25"] = (history["Goals_H_FT"] + history["Goals_A_FT"] > 2.5).astype(int)
history["Target_BTTS"] = ((history["Goals_H_FT"] > 0) & (history["Goals_A_FT"] > 0)).astype(int)


# ########################################################
# Bloco 5 – Features & One-Hot Leagues
# ########################################################
# Adicionar features de momentum do Binary
def add_momentum_features(df):
    df['PesoMomentum_H'] = abs(df['M_H']) / (abs(df['M_H']) + abs(df['M_A']))
    df['PesoMomentum_A'] = abs(df['M_A']) / (abs(df['M_H']) + abs(df['M_A']))
    df['CustoMomentum_H'] = df.apply(
        lambda x: x['Odd_H'] / abs(x['M_H']) if abs(x['M_H']) > 0 else np.nan, axis=1
    )
    df['CustoMomentum_A'] = df.apply(
        lambda x: x['Odd_A'] / abs(x['M_A']) if abs(x['M_A']) > 0 else np.nan, axis=1
    )
    return df

history = add_momentum_features(history)
games_today = add_momentum_features(games_today)

history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]
history['Diff_Abs'] = (history['M_H'] - history['M_A']).abs()
games_today['Diff_Abs'] = (games_today['M_H'] - games_today['M_A']).abs()

# Features atualizadas com momentum
features_1x2 = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A", 
                "Diff_Abs", "PesoMomentum_H", "PesoMomentum_A", "CustoMomentum_H", "CustoMomentum_A"]
features_ou_btts = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "OU_Total",
                   "Diff_Abs", "PesoMomentum_H", "PesoMomentum_A", "CustoMomentum_H", "CustoMomentum_A"]

history_leagues = pd.get_dummies(history["League"], prefix="League")
games_today_leagues = pd.get_dummies(games_today["League"], prefix="League")
games_today_leagues = games_today_leagues.reindex(columns=history_leagues.columns, fill_value=0)

X_1x2 = pd.concat([history[features_1x2], history_leagues], axis=1)
X_ou = pd.concat([history[features_ou_btts], history_leagues], axis=1)
X_btts = pd.concat([history[features_ou_btts], history_leagues], axis=1)

X_today_1x2 = pd.concat([games_today[features_1x2], games_today_leagues], axis=1)
X_today_ou = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)
X_today_btts = pd.concat([games_today[features_ou_btts], games_today_leagues], axis=1)


# ########################################################
# Bloco 6 – Configurações ML (Sidebar) - ATUALIZADO
# ########################################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"]
)
use_smote = st.sidebar.checkbox("Use SMOTE for balancing", value=True)
retrain = st.sidebar.checkbox("Retrain models", value=False)

st.sidebar.markdown("""
**ℹ️ Usage recommendations:**
- 🔹 *Random Forest*: simple and fast baseline.  
- 🔹 *Random Forest Tuned*: suitable for market **1X2**.  
- 🔹 *XGBoost Tuned*: suitable for markets **Over/Under 2.5** e **BTTS**.  
- 🔹 *SMOTE*: recommended for imbalanced datasets
""")


# ########################################################
# Bloco 7 – Treino & Avaliação (COM SMOTE - CORRIGIDO)
# ########################################################
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_fc.pkl"
    model = None

    if not retrain:
        model = load_model(filename)

    # 🔥 CORREÇÃO: Limpeza de dados antes do split
    # Combinar X e y para limpeza consistente
    data_clean = X.copy()
    data_clean['target'] = y
    
    # Remover linhas com NaN ou infinitos
    data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
    data_clean = data_clean.dropna()
    
    if data_clean.empty:
        st.error(f"❌ No valid data after cleaning for {name}")
        return {}, None
        
    # Separar novamente
    X_clean = data_clean.drop('target', axis=1)
    y_clean = data_clean['target']
    
    st.info(f"📊 Dataset {name}: {len(X_clean)} samples after cleaning")

    # Split dos dados LIMPOS
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Aplicar SMOTE se selecionado
    if use_smote:
        st.info(f"🔄 Applying SMOTE for {name} (before: {dict(Counter(y_train))})")
        try:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.info(f"📊 After SMOTE: {dict(Counter(y_train))}")
        except Exception as e:
            st.error(f"❌ SMOTE failed for {name}: {e}")
            st.warning("🔄 Continuing without SMOTE...")

    if model is None:
        if ml_model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")

        elif ml_model_choice == "Random Forest Tuned":
            rf_params = {
                "1X2": {'n_estimators': 600, 'max_depth': 14, 'min_samples_split': 10,
                        'min_samples_leaf': 1, 'max_features': 'sqrt'},
                "OverUnder25": {'n_estimators': 600, 'max_depth': 5, 'min_samples_split': 9,
                                'min_samples_leaf': 3, 'max_features': 'sqrt'},
                "BTTS": {'n_estimators': 400, 'max_depth': 18, 'min_samples_split': 4,
                         'min_samples_leaf': 5, 'max_features': 'sqrt'},
            }
            model = RandomForestClassifier(random_state=42, class_weight="balanced_subsample", **rf_params[name])

        elif ml_model_choice == "XGBoost Tuned":
            xgb_params = {
                "1X2": {'n_estimators': 219, 'max_depth': 9, 'learning_rate': 0.05,
                        'subsample': 0.9, 'colsample_bytree': 0.8,
                        'eval_metric': 'mlogloss', 'use_label_encoder': False},
                "OverUnder25": {'n_estimators': 488, 'max_depth': 10, 'learning_rate': 0.03,
                                'subsample': 0.9, 'colsample_bytree': 0.7,
                                'eval_metric': 'logloss', 'use_label_encoder': False},
                "BTTS": {'n_estimators': 695, 'max_depth': 6, 'learning_rate': 0.04,
                         'subsample': 0.8, 'colsample_bytree': 0.8,
                         'eval_metric': 'logloss', 'use_label_encoder': False},
            }
            model = XGBClassifier(random_state=42, **xgb_params[name])

        model.fit(X_train, y_train)
        save_model(model, filename)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs)

    if num_classes == 2:
        bs = brier_score_loss(y_val, probs[:, 1])
        bs = f"{bs:.3f}"
    else:
        y_onehot = pd.get_dummies(y_val).values
        bs_raw = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
        bs = f"{bs_raw:.3f} (multi)"

    metrics = {
        "Model": f"{ml_model_choice} - {name}",
        "Accuracy": f"{acc:.3f}",
        "LogLoss": f"{ll:.3f}",
        "Brier": bs,
        "SMOTE": "Yes" if use_smote else "No",
        "Samples": len(X_clean)
    }

    return metrics, model


# ########################################################
# Bloco 8 – Treinar Modelos
# ########################################################
stats = []
res, model_multi = train_and_evaluate(X_1x2, history["Target"], "1X2", 3)
stats.append(res)
res, model_ou = train_and_evaluate(X_ou, history["Target_OU25"], "OverUnder25", 2)
stats.append(res)
res, model_btts = train_and_evaluate(X_btts, history["Target_BTTS"], "BTTS", 2)
stats.append(res)

df_stats = pd.DataFrame(stats)
st.markdown("### 📊 Model Statistics (Validation)")
st.dataframe(df_stats, use_container_width=True)


# ########################################################
# Bloco 9 – Previsões (COM TRATAMENTO DE NaN)
# ########################################################

# 🔥 CORREÇÃO: Preencher NaN nos dados de hoje antes da previsão
def safe_predict_proba(model, X_data, default_value=0.33):
    """Previsão segura com tratamento de NaN"""
    X_filled = X_data.fillna(0)  # Preencher NaN com 0
    
    try:
        return model.predict_proba(X_filled)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        # Retornar probabilidades uniformes em caso de erro
        n_samples = len(X_data)
        if hasattr(model, 'classes_'):
            n_classes = len(model.classes_)
            return np.full((n_samples, n_classes), default_value)
        else:
            return np.full((n_samples, 2), 0.5)  # Fallback para binário

# Previsões com tratamento seguro
probs_1x2 = safe_predict_proba(model_multi, X_today_1x2)
probs_ou = safe_predict_proba(model_ou, X_today_ou) 
probs_btts = safe_predict_proba(model_btts, X_today_btts)

games_today["p_home"], games_today["p_draw"], games_today["p_away"] = probs_1x2.T
games_today["p_over25"], games_today["p_under25"] = probs_ou.T
games_today["p_btts_yes"], games_today["p_btts_no"] = probs_btts.T


# ########################################################
# Bloco 10 – Styling e Display
# ########################################################
def color_prob(val, color):
    alpha = int(val * 255)
    return f"background-color: rgba({color}, {alpha/255:.2f})"

def style_probs(val, col):
    if col == "p_home": return color_prob(val, "0,200,0")
    elif col == "p_draw": return color_prob(val, "150,150,150")
    elif col == "p_away": return color_prob(val, "255,140,0")
    elif col == "p_over25": return color_prob(val, "0,100,255")
    elif col == "p_under25": return color_prob(val, "128,0,128")
    elif col == "p_btts_yes": return color_prob(val, "0,200,200")
    elif col == "p_btts_no": return color_prob(val, "200,0,0")
    return ""

cols_final = [
    "Date","Time","League","Home","Away",
    "Odd_H","Odd_D","Odd_A",
    "p_home","p_draw","p_away",
    "p_over25","p_under25",
    "p_btts_yes","p_btts_no"
]

styled_df = (
    games_today[cols_final]
    .style.format({
        "Odd_H": "{:.2f}","Odd_D": "{:.2f}","Odd_A": "{:.2f}",
        "p_home": "{:.1%}","p_draw": "{:.1%}","p_away": "{:.1%}",
        "p_over25": "{:.1%}","p_under25": "{:.1%}",
        "p_btts_yes": "{:.1%}","p_btts_no": "{:.1%}",
    }, na_rep="—")
    .applymap(lambda v: style_probs(v, "p_home"), subset=["p_home"])
    .applymap(lambda v: style_probs(v, "p_draw"), subset=["p_draw"])
    .applymap(lambda v: style_probs(v, "p_away"), subset=["p_away"])
    .applymap(lambda v: style_probs(v, "p_over25"), subset=["p_over25"])
    .applymap(lambda v: style_probs(v, "p_under25"), subset=["p_under25"])
    .applymap(lambda v: style_probs(v, "p_btts_yes"), subset=["p_btts_yes"])
    .applymap(lambda v: style_probs(v, "p_btts_no"), subset=["p_btts_no"])
)

st.markdown("### 📌 Predictions for Selected Matches")
st.dataframe(styled_df, use_container_width=True, height=1000)

# 🔹 Botão para download do CSV (do Binary)
import io
csv_buffer = io.BytesIO()
games_today.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
csv_buffer.seek(0)

st.download_button(
    label="📥 Download Predictions CSV",
    data=csv_buffer,
    file_name=f"Bet_Indicator_Triple_View_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)


# ########################################################
# Block 11 – Hybrid Forecast (Historical vs ML)
# ########################################################
st.markdown("## 🔮 Hybrid Forecast – Perspective vs ML")

try:
    import numpy as np

    # 🔹 Ensure we have a reference date
    if not games_today.empty and "Date" in games_today.columns:
        selected_date = pd.to_datetime(games_today["Date"], errors="coerce").dt.date.iloc[0]
    else:
        selected_date = None

    # ===== Historical Perspective =====
    all_dfs = []
    for f in os.listdir(GAMES_FOLDER):
        if f.lower().endswith(".csv"):
            try:
                df_tmp = pd.read_csv(os.path.join(GAMES_FOLDER, f))
                df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('^Unnamed')]
                df_tmp.columns = df_tmp.columns.str.strip()
                all_dfs.append(df_tmp)
            except:
                continue

    if all_dfs and selected_date is not None:
        df_history = pd.concat(all_dfs, ignore_index=True)

        # 🧹 Remove duplicates (usando função melhorada do Binary)
        df_history = df_history.drop_duplicates(
            subset=["Date", "Home", "Away", "Goals_H_FT", "Goals_A_FT"],
            keep="first"
        )

        # Normalize dates and exclude today's matches
        if "Date" in df_history.columns:
            df_history["Date"] = pd.to_datetime(df_history["Date"], errors="coerce").dt.date
            df_history = df_history[df_history["Date"] != selected_date]

        # Create Diff_M and bins
        df_history["Diff_M"] = df_history["M_H"] - df_history["M_A"]
        df_history["DiffPower_bin"] = pd.cut(df_history["Diff_Power"], bins=range(-50, 55, 10))
        df_history["DiffM_bin"] = pd.cut(df_history["Diff_M"], bins=np.arange(-10, 10.5, 1.0))
        df_history["DiffHTP_bin"] = pd.cut(df_history["Diff_HT_P"], bins=range(-30, 35, 5))

        # Real match outcome
        def get_result(row):
            if row["Goals_H_FT"] > row["Goals_A_FT"]:
                return "Home"
            elif row["Goals_H_FT"] < row["Goals_A_FT"]:
                return "Away"
            else:
                return "Draw"

        df_history["Result"] = df_history.apply(get_result, axis=1)

        # Prepare today's matches (using games_today)
        df_day = games_today.copy()
        df_day = df_day.loc[:, ~df_day.columns.str.contains('^Unnamed')]
        df_day.columns = df_day.columns.str.strip()
        df_day["Date"] = pd.to_datetime(df_day["Date"], errors="coerce").dt.date
        df_day = df_day[df_day["Date"] == selected_date]
        df_day["Diff_M"] = df_day["M_H"] - df_day["M_A"]
        df_day = df_day.dropna(subset=["Diff_Power", "Diff_M", "Diff_HT_P"])

        # Bin intervals
        dp_bins = pd.IntervalIndex(df_history["DiffPower_bin"].cat.categories)
        dm_bins = pd.IntervalIndex(df_history["DiffM_bin"].cat.categories)
        dhtp_bins = pd.IntervalIndex(df_history["DiffHTP_bin"].cat.categories)

        # Counters
        total_matches, home_wins, away_wins, draws = 0, 0, 0, 0

        for _, game in df_day.iterrows():
            try:
                if (
                    dp_bins.contains(game["Diff_Power"]).any() and
                    dm_bins.contains(game["Diff_M"]).any() and
                    dhtp_bins.contains(game["Diff_HT_P"]).any()
                ):
                    dp_bin = dp_bins.get_loc(game["Diff_Power"])
                    dm_bin = dm_bins.get_loc(game["Diff_M"])
                    dhtp_bin = dhtp_bins.get_loc(game["Diff_HT_P"])
                else:
                    continue

                subset = df_history[
                    (df_history["DiffPower_bin"] == dp_bins[dp_bin]) &
                    (df_history["DiffM_bin"] == dm_bins[dm_bin]) &
                    (df_history["DiffHTP_bin"] == dhtp_bins[dhtp_bin])
                ]

                if not subset.empty:
                    total_matches += len(subset)
                    home_wins += (subset["Result"] == "Home").sum()
                    away_wins += (subset["Result"] == "Away").sum()
                    draws += (subset["Result"] == "Draw").sum()
            except:
                continue

        if total_matches > 0:
            pct_home = 100 * home_wins / total_matches
            pct_away = 100 * away_wins / total_matches
            pct_draw = 100 * draws / total_matches
        else:
            pct_home, pct_away, pct_draw = 0, 0, 0

    # ===== ML Forecast =====
    if not games_today.empty:
        ml_probs = model_multi.predict_proba(X_today_1x2)
        df_preds = pd.DataFrame(ml_probs, columns=["p_home", "p_draw", "p_away"])

        ml_home = df_preds["p_home"].mean() * 100
        ml_draw = df_preds["p_draw"].mean() * 100
        ml_away = df_preds["p_away"].mean() * 100
    else:
        ml_home, ml_draw, ml_away = 0, 0, 0

    # ===== Side by side display =====
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 📊 Historical Perspective")
        st.write(f"**Home Wins:** {pct_home:.1f}%")
        st.write(f"**Draws:** {pct_draw:.1f}%")
        st.write(f"**Away Wins:** {pct_away:.1f}%")
        st.caption(f"Based on {total_matches:,} similar historical matches (excluding today)")
    with cols[1]:
        st.markdown("### 🤖 ML Forecast (Trained Model)")
        st.write(f"**Home Wins:** {ml_home:.1f}%")
        st.write(f"**Draws:** {ml_draw:.1f}%")
        st.write(f"**Away Wins:** {ml_away:.1f}%")
        st.caption(f"Based on {len(games_today)} matches today")

except Exception as e:
    st.warning(f"⚠️ Hybrid Forecast could not be generated: {e}")



# ########################################################
# Block 12 – Divergence Index with Gauge
# ########################################################
try:
    import plotly.graph_objects as go

    # ===== Divergence Index =====
    divergence = abs(ml_home - pct_home) + abs(ml_draw - pct_draw) + abs(ml_away - pct_away)

    if divergence < 10:
        status_icon, status_text = "🟢", "High confidence (ML aligned with historical)"
    elif divergence < 25:
        status_icon, status_text = "🟡", "Medium confidence (some divergence)"
    else:
        status_icon, status_text = "🔴", "Low confidence (ML diverges strongly from historical)"

    # Detailed differences
    st.markdown("### 🔍 Difference: Historical vs ML")
    st.write(f"- Home: {ml_home - pct_home:+.1f} pp")
    st.write(f"- Draw: {ml_draw - pct_draw:+.1f} pp")
    st.write(f"- Away: {ml_away - pct_away:+.1f} pp")

    # Global index
    st.markdown("### 📈 Global Divergence Index")
    st.write(f"{status_icon} {status_text}")
    st.caption(f"Total divergence index: {divergence:.1f} percentage points")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=divergence,
        title={'text': "Divergence Index"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 25], 'color': "khaki"},
                {'range': [25, 50], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': divergence
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Divergence Block could not be generated: {e}")
