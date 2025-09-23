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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bet Indicator – Triple View", layout="wide")
st.title("📊 Bet Indicator – Triple View (1X2 + OU + BTTS)")

# Paths
GAMES_FOLDER = "GamesDay"
EXCLUDED_LEAGUE_KEYWORDS = ["Cup", "Copa", "Copas", "uefa", "nordeste", "afc"]

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
history["Diff_M"] = history["M_H"] - history["M_A"]
games_today["Diff_M"] = games_today["M_H"] - games_today["M_A"]

features_1x2 = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "M_HT_H", "M_HT_A"]
features_ou_btts = ["Odd_H", "Odd_D", "Odd_A", "Diff_Power", "M_H", "M_A", "Diff_M", "Diff_HT_P", "OU_Total"]

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
# Bloco 6 – Configurações ML (Sidebar)
# ########################################################
st.sidebar.header("⚙️ Settings")
ml_model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Random Forest", "Random Forest Tuned", "XGBoost Tuned"]
)
retrain = st.sidebar.checkbox("Retrain models", value=False)

st.sidebar.markdown("""
**ℹ️ Usage recommendations:**
- 🔹 *Random Forest*: simple and fast baseline.  
- 🔹 *Random Forest Tuned*: suitable for market **1X2**.  
- 🔹 *XGBoost Tuned*: suitable for markets **Over/Under 2.5** e **BTTS**.  
""")


# ########################################################
# Bloco 7 – Treino & Avaliação
# ########################################################
def train_and_evaluate(X, y, name, num_classes):
    filename = f"{ml_model_choice.replace(' ', '')}_{name}_fc.pkl"
    model = None

    if not retrain:
        model = load_model(filename)

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

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        save_model(model, filename)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
# Bloco 9 – Previsões
# ########################################################
games_today["p_home"], games_today["p_draw"], games_today["p_away"] = model_multi.predict_proba(X_today_1x2).T
games_today["p_over25"], games_today["p_under25"] = model_ou.predict_proba(X_today_ou).T
games_today["p_btts_yes"], games_today["p_btts_no"] = model_btts.predict_proba(X_today_btts).T


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



# ########################################################
# Bloco 11 – Forecast Híbrido (Estatístico vs ML) – Final
# ########################################################
st.markdown("## 🔮 Forecast Híbrido – Perspective vs ML")

try:
    import numpy as np

    # ===== Forecast Estatístico (Perspective) =====
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

    if all_dfs:
        df_history = pd.concat(all_dfs, ignore_index=True)

        # 🧹 Remove duplicados (mesmo critério da página principal)
        df_history = df_history.drop_duplicates(
            subset=["League", "Home", "Away", "Goals_H_FT", "Goals_A_FT"],
            keep="first"
        )

        # Normalizar Date e excluir o dia selecionado
        if "Date" in df_history.columns:
            df_history["Date"] = pd.to_datetime(df_history["Date"], errors="coerce").dt.date
            df_history = df_history[df_history["Date"] != selected_date]

        # Criar Diff_M e bins
        df_history["Diff_M"] = df_history["M_H"] - df_history["M_A"]
        df_history["DiffPower_bin"] = pd.cut(df_history["Diff_Power"], bins=range(-50, 55, 10))
        df_history["DiffM_bin"] = pd.cut(df_history["Diff_M"], bins=np.arange(-10, 10.5, 1.0))
        df_history["DiffHTP_bin"] = pd.cut(df_history["Diff_HT_P"], bins=range(-30, 35, 5))

        # Resultado real
        def get_result(row):
            if row["Goals_H_FT"] > row["Goals_A_FT"]:
                return "Home"
            elif row["Goals_H_FT"] < row["Goals_A_FT"]:
                return "Away"
            else:
                return "Draw"

        df_history["Result"] = df_history.apply(get_result, axis=1)

        # Preparar jogos do dia
        df_day = pd.read_csv(file_path)
        df_day = df_day.loc[:, ~df_day.columns.str.contains('^Unnamed')]
        df_day.columns = df_day.columns.str.strip()
        df_day["Date"] = pd.to_datetime(df_day["Date"], errors="coerce").dt.date
        df_day = df_day[df_day["Date"] == selected_date]
        df_day["Diff_M"] = df_day["M_H"] - df_day["M_A"]
        df_day = df_day.dropna(subset=["Diff_Power", "Diff_M", "Diff_HT_P"])

        # Intervalos dos bins
        dp_bins = pd.IntervalIndex(df_history["DiffPower_bin"].cat.categories)
        dm_bins = pd.IntervalIndex(df_history["DiffM_bin"].cat.categories)
        dhtp_bins = pd.IntervalIndex(df_history["DiffHTP_bin"].cat.categories)

        # Contadores
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

    # ===== Forecast ML =====
    if not games_today.empty:
        ml_probs = model_multi.predict_proba(X_today_1x2)
        df_preds = pd.DataFrame(ml_probs, columns=["p_home", "p_draw", "p_away"])

        ml_home = df_preds["p_home"].mean() * 100
        ml_draw = df_preds["p_draw"].mean() * 100
        ml_away = df_preds["p_away"].mean() * 100
    else:
        ml_home, ml_draw, ml_away = 0, 0, 0

    # ===== Índice de Divergência =====
    divergence = abs(ml_home - pct_home) + abs(ml_draw - pct_draw) + abs(ml_away - pct_away)

    if divergence < 10:
        status_icon, status_text = "🟢", "Alta confiança (ML e histórico alinhados)"
    elif divergence < 25:
        status_icon, status_text = "🟡", "Confiança média (alguma divergência)"
    else:
        status_icon, status_text = "🔴", "Baixa confiança (ML difere muito do histórico)"

    # ===== Mostrar lado a lado =====
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 📊 Histórico (Perspective)")
        st.write(f"**Home Wins:** {pct_home:.1f}%")
        st.write(f"**Draws:** {pct_draw:.1f}%")
        st.write(f"**Away Wins:** {pct_away:.1f}%")
        st.caption(f"Baseado em {total_matches:,} jogos históricos similares (excluindo o dia atual)")
    with cols[1]:
        st.markdown("### 🤖 ML (Modelo Treinado)")
        st.write(f"**Home Wins:** {ml_home:.1f}%")
        st.write(f"**Draws:** {ml_draw:.1f}%")
        st.write(f"**Away Wins:** {ml_away:.1f}%")
        st.caption(f"Baseado em {len(games_today)} jogos de hoje")

    # Diferença e Divergência
    st.markdown("### 🔍 Diferença Estatística vs ML")
    st.write(f"- Home: {ml_home - pct_home:+.1f} pp")
    st.write(f"- Draw: {ml_draw - pct_draw:+.1f} pp")
    st.write(f"- Away: {ml_away - pct_away:+.1f} pp")

    st.markdown("### 📈 Índice de Divergência Global")
    st.write(f"{status_icon} {status_text}")
    st.caption(f"Índice total de divergência: {divergence:.1f} pontos percentuais")

except Exception as e:
    st.warning(f"⚠️ Forecast Híbrido não pôde ser gerado: {e}")






