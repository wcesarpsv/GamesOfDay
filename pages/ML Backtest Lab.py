########################################
# BLOCO 1 – CONFIGURAÇÃO INICIAL
########################################
import streamlit as st

st.set_page_config(page_title="ML Backtest Lab", layout="wide")
st.title("⚡ ML Backtest Lab – Configuração e Análise")

st.markdown("""
Bem-vindo ao **laboratório de Machine Learning para apostas esportivas**.  
Aqui você pode:
- Selecionar diferentes modelos ML.
- Ajustar hiperparâmetros em tempo real.
- Comparar previsões com o modelo baseado em regras.
- Rodar backtests e visualizar métricas.

Use os checkboxes para **controlar o que será exibido** e otimizar a performance.
""")

st.divider()

########################################
# BLOCO 2 – DADOS
########################################
st.header("📂 Entrada de Dados")

# Opção de upload ou seleção de CSV do GamesDay
data_option = st.radio("Como deseja carregar os dados?",
                       ["Selecionar CSV do GamesDay", "Upload manual"])

if data_option == "Selecionar CSV do GamesDay":
    st.selectbox("Escolha o arquivo do dia", ["2025-09-28.csv", "2025-09-29.csv"])
else:
    st.file_uploader("Upload CSV", type=["csv"])

# Filtro extra
st.checkbox("Filtrar ligas (excluir Cup/UEFA)", value=True)

st.divider()

########################################
# BLOCO 3 – CONFIGURAÇÃO DO MODELO
########################################
st.header("⚙️ Configuração do Modelo ML")

# Escolher modelo
model_choice = st.selectbox(
    "Selecione o algoritmo de Machine Learning:",
    ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "Neural Network (MLP)"]
)

# Ajustes dinâmicos conforme modelo selecionado
st.subheader("🔧 Hiperparâmetros do Modelo")

if model_choice == "Random Forest":
    n_estimators = st.slider("n_estimators", 100, 1000, 500, step=50)
    max_depth = st.slider("max_depth", 2, 30, 12)
    max_features = st.selectbox("max_features", ["sqrt", "log2", "None"])
    min_samples_split = st.slider("min_samples_split", 2, 20, 10)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 4)

elif model_choice == "XGBoost":
    n_estimators = st.slider("n_estimators", 100, 1000, 300, step=50)
    max_depth = st.slider("max_depth", 2, 15, 6)
    learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)
    subsample = st.slider("subsample", 0.5, 1.0, 0.8)
    colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8)

elif model_choice == "LightGBM":
    num_leaves = st.slider("num_leaves", 10, 200, 31)
    learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)
    feature_fraction = st.slider("feature_fraction", 0.5, 1.0, 0.8)
    bagging_fraction = st.slider("bagging_fraction", 0.5, 1.0, 0.8)

elif model_choice == "Logistic Regression":
    c_value = st.slider("C (Regularização)", 0.01, 10.0, 1.0)
    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])

elif model_choice == "Neural Network (MLP)":
    hidden_layers = st.slider("Camadas Ocultas", 1, 5, 2)
    neurons_per_layer = st.slider("Neuronios por Camada", 4, 128, 32)
    learning_rate_init = st.slider("Learning Rate Inicial", 0.001, 0.1, 0.01)

# Checkboxes extras
st.subheader("⚡ Opções Extras")
st.checkbox("Aplicar calibração isotônica", value=False)
st.checkbox("Salvar previsões em CSV", value=False)
st.checkbox("Comparar com modelo baseado em regras", value=True)

st.divider()

########################################
# BLOCO 4 – CONFIGURAÇÃO DO BACKTEST
########################################
st.header("📈 Backtest")

st.date_input("Data inicial do histórico", value=None)
st.date_input("Data final do histórico", value=None)

st.selectbox("Métrica principal para avaliação", ["ROI", "AUC", "LogLoss", "Winrate"])

st.number_input("Tamanho mínimo do histórico (dias)", min_value=7, max_value=90, value=30)
st.number_input("Tamanho do período de teste (dias)", min_value=1, max_value=30, value=7)

st.divider()

########################################
# BLOCO 5 – VISUALIZAÇÃO DE RESULTADOS
########################################
st.header("📊 Resultados e Visualizações")

st.markdown("Selecione os gráficos e relatórios que deseja exibir:")

show_table = st.checkbox("Mostrar tabela final com jogos", value=True)
show_roi = st.checkbox("Mostrar gráfico de ROI acumulado", value=True)
show_histogram = st.checkbox("Mostrar histograma de probabilidades", value=False)
show_calibration = st.checkbox("Mostrar gráfico de calibração (linha perfeita vs modelo)", value=True)
show_feature_importance = st.checkbox("Mostrar importância das features", value=False)
show_compare_models = st.checkbox("Comparar múltiplos modelos ML", value=False)

# Placeholders para exibição futura
if show_table:
    st.info("📋 Tabela final será exibida aqui (dados fictícios por enquanto).")

if show_roi:
    st.info("📈 Gráfico de ROI acumulado será exibido aqui.")

if show_histogram:
    st.info("📊 Histograma de probabilidades será exibido aqui.")

if show_calibration:
    st.info("📉 Gráfico de calibração será exibido aqui.")

if show_feature_importance:
    st.info("🔥 Importância das features será exibida aqui.")

if show_compare_models:
    st.info("⚖️ Comparação de múltiplos modelos será exibida aqui.")

st.divider()

st.success("Layout pronto! Agora podemos integrar os cálculos e modelos passo a passo.")
