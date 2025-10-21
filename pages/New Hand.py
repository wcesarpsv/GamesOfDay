# =====================================================
# BLOCO 1 - CONFIGURAÇÕES E CONSTANTES
# =====================================================

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

# Configuração da página
st.set_page_config(page_title="Sistema Handicap - Bet Indicator", layout="wide")
st.title("🎯 Sistema de Handicap - Análise Preditiva")

# Constantes do sistema
GAMES_FOLDER = "GamesDay"
MODELS_FOLDER = "Models_Handicap"

# Criar pasta de modelos se não existir
os.makedirs(MODELS_FOLDER, exist_ok=True)

# =====================================================
# BLOCO 2 - CARREGAMENTO E PREPROCESSAMENTO
# =====================================================

def verificar_estrutura_pastas():
    """Verifica se as pastas necessárias existem"""
    if not os.path.exists(GAMES_FOLDER):
        st.error(f"❌ Pasta '{GAMES_FOLDER}' não encontrada!")
        st.info("📁 Crie uma pasta chamada 'GamesDay' com arquivos CSV dos jogos")
        return False
    return True

def carregar_dados_simples():
    """Carrega dados com fallbacks se arquivos não existirem"""
    
    # Verificar se existe a pasta
    if not verificar_estrutura_pastas():
        return None, None, "2024-01-01"
    
    # Listar arquivos CSV
    try:
        files = [f for f in os.listdir(GAMES_FOLDER) if f.endswith(".csv")]
        if not files:
            st.warning("📂 Nenhum arquivo CSV encontrado na pasta GamesDay")
            
            # Criar dados de exemplo para teste
            st.info("🔄 Criando dados de exemplo para demonstração...")
            return criar_dados_exemplo(), criar_dados_exemplo(), "2024-01-01"
        
        # Usar o arquivo mais recente
        arquivo_recente = sorted(files)[-1]
        st.success(f"📁 Arquivo carregado: {arquivo_recente}")
        
        # Carregar dados reais
        games_today = pd.read_csv(os.path.join(GAMES_FOLDER, arquivo_recente))
        historico = games_today.copy()  # Para demo, usamos os mesmos dados
        
        return games_today, historico, arquivo_recente
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return criar_dados_exemplo(), criar_dados_exemplo(), "2024-01-01"

def criar_dados_exemplo():
    """Cria dados de exemplo para demonstração"""
    np.random.seed(42)
    
    times = [
        "Flamengo", "Palmeiras", "São Paulo", "Corinthians", "Grêmio",
        "Internacional", "Atlético-MG", "Santos", "Fluminense", "Botafogo",
        "Fortaleza", "Bahia", "Cruzeiro", "Vasco", "Bragantino"
    ]
    
    ligas = ["Brasileirão A", "Brasileirão B", "Copa do Brasil", "Libertadores"]
    
    dados = []
    for i in range(50):
        home = np.random.choice(times)
        away = np.random.choice([t for t in times if t != home])
        
        dados.append({
            'Home': home,
            'Away': away,
            'League': np.random.choice(ligas),
            'Goals_H_FT': np.random.randint(0, 4),
            'Goals_A_FT': np.random.randint(0, 4),
            'Asian_Line': np.random.choice(["-0.5", "0.0", "+0.5", "-1.0", "+1.0"]),
            'Date': f"2024-01-{np.random.randint(1, 30):02d}"
        })
    
    return pd.DataFrame(dados)

# =====================================================
# BLOCO 3 - SISTEMA DE HANDICAP
# =====================================================

def convert_asian_line(line_str):
    """Converte string de linha asiática em média numérica"""
    try:
        if pd.isna(line_str) or line_str == "":
            return 0.0
        line_str = str(line_str).strip().replace('+', '').replace(',', '.')
        
        if "/" in line_str:
            parts = [float(x) for x in line_str.split("/")]
            return sum(parts) / len(parts)
        else:
            return float(line_str)
    except:
        return 0.0

def criar_target_handicap(df):
    """
    Cria target para vencedor do handicap asiático
    1 = Home cobre o handicap | 0 = Away cobre o handicap
    """
    df = df.copy()
    
    # Converter linha asiática
    df['Asian_Line_Numeric'] = df['Asian_Line'].apply(convert_asian_line)
    
    # Calcular resultado com handicap
    df['Margin_With_Handicap'] = df['Goals_H_FT'] - df['Goals_A_FT'] - df['Asian_Line_Numeric']
    
    # Definir vencedor (1 = Home cobre, 0 = Away cobre)
    df['Target_Handicap'] = (df['Margin_With_Handicap'] > 0).astype(int)
    
    return df

# =====================================================
# BLOCO 4 - SISTEMA DE QUADRANTES SIMPLIFICADO
# =====================================================

def criar_features_avancadas(df):
    """Cria features avançadas para o modelo"""
    df = df.copy()
    
    # 1. Features básicas de performance
    df['Total_Goals'] = df['Goals_H_FT'] + df['Goals_A_FT']
    df['Goal_Difference'] = df['Goals_H_FT'] - df['Goals_A_FT']
    df['Is_Home_Win'] = (df['Goals_H_FT'] > df['Goals_A_FT']).astype(int)
    df['Is_Draw'] = (df['Goals_H_FT'] == df['Goals_A_FT']).astype(int)
    
    # 2. Features de "força" dos times (simuladas para demo)
    np.random.seed(hash(str(df.iloc[0]['Home']) if len(df) > 0 else 42) % 1000)
    
    # Criar IDs consistentes para os times
    all_teams = list(set(df['Home'].unique()) | set(df['Away'].unique()))
    team_strength = {team: np.random.normal(0.5, 0.2) for team in all_teams}
    team_attack = {team: np.random.normal(1.5, 0.5) for team in all_teams}
    team_defense = {team: np.random.normal(1.5, 0.5) for team in all_teams}
    
    # Aplicar features aos times
    df['Home_Strength'] = df['Home'].map(team_strength)
    df['Away_Strength'] = df['Away'].map(team_strength)
    df['Home_Attack'] = df['Home'].map(team_attack)
    df['Away_Attack'] = df['Away'].map(team_attack)
    df['Home_Defense'] = df['Home'].map(team_defense)
    df['Away_Defense'] = df['Away'].map(team_defense)
    
    # 3. Features derivadas
    df['Strength_Diff'] = df['Home_Strength'] - df['Away_Strength']
    df['Attack_Defense_Ratio_Home'] = df['Home_Attack'] / (df['Away_Defense'] + 0.1)
    df['Attack_Defense_Ratio_Away'] = df['Away_Attack'] / (df['Home_Defense'] + 0.1)
    
    # 4. Quadrantes simplificados (0-3)
    df['Quadrante_Home'] = np.random.randint(0, 4, len(df))
    df['Quadrante_Away'] = np.random.randint(0, 4, len(df))
    
    return df

def preparar_features_ml(history, games_today):
    """Prepara features para o modelo ML"""
    
    # Aplicar target de handicap ao histórico
    history = criar_target_handicap(history)
    
    # Criar features avançadas
    history = criar_features_avancadas(history)
    games_today = criar_features_avancadas(games_today)
    
    # Selecionar features para o modelo
    feature_cols = [
        'Home_Strength', 'Away_Strength', 'Strength_Diff',
        'Home_Attack', 'Away_Attack', 'Home_Defense', 'Away_Defense',
        'Attack_Defense_Ratio_Home', 'Attack_Defense_Ratio_Away',
        'Quadrante_Home', 'Quadrante_Away'
    ]
    
    # One-hot encoding para ligas
    ligas_history = pd.get_dummies(history['League'], prefix='League')
    ligas_today = pd.get_dummies(games_today['League'], prefix='League')
    
    # Combinar todas as features
    X_history = pd.concat([history[feature_cols], ligas_history], axis=1)
    X_today = pd.concat([games_today[feature_cols], ligas_today], axis=1)
    
    # Garantir mesmas colunas
    all_columns = X_history.columns
    X_today = X_today.reindex(columns=all_columns, fill_value=0)
    
    y_history = history['Target_Handicap']
    
    return X_history, y_history, X_today, history, games_today

def treinar_modelo_handicap(X, y):
    """Treina modelo Random Forest para handicap"""
    try:
        model = RandomForestClassifier(
            n_estimators=50,  # Reduzido para performance
            random_state=42,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=7
        )
        
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"❌ Erro ao treinar modelo: {e}")
        return None

# =====================================================
# BLOCO 5 - VISUALIZAÇÃO E ANÁLISE
# =====================================================

def plot_resultados_handicap(games_today):
    """Plot da distribuição das probabilidades"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'Prob_Home_Cobre' in games_today.columns:
        # Histograma das probabilidades
        ax[0].hist(games_today['Prob_Home_Cobre'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax[0].axvline(x=0.5, color='red', linestyle='--', label='Linha Neutra (50%)')
        ax[0].set_xlabel('Probabilidade Home Cobrir Handicap')
        ax[0].set_ylabel('Número de Jogos')
        ax[0].set_title('Distribuição das Probabilidades')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Scatter plot: Força vs Probabilidade
        if 'Home_Strength' in games_today.columns:
            ax[1].scatter(games_today['Home_Strength'], games_today['Prob_Home_Cobre'], 
                         alpha=0.6, c=games_today['Prob_Home_Cobre'], cmap='RdYlGn')
            ax[1].set_xlabel('Força do Time da Casa')
            ax[1].set_ylabel('Probabilidade Home Cobre')
            ax[1].set_title('Relação: Força vs Probabilidade')
            ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =====================================================
# BLOCO 6 - SISTEMA DE RECOMENDAÇÕES
# =====================================================

def gerar_recomendacoes_inteligentes(df):
    """Gera recomendações baseadas nas probabilidades"""
    if df.empty or 'Prob_Home_Cobre' not in df.columns:
        return df
    
    df = df.copy()
    
    # Classificar por confiança
    conditions_confianca = [
        (df['Prob_Home_Cobre'] >= 0.70) | (df['Prob_Home_Cobre'] <= 0.30),
        (df['Prob_Home_Cobre'] >= 0.65) | (df['Prob_Home_Cobre'] <= 0.35),
        (df['Prob_Home_Cobre'] >= 0.60) | (df['Prob_Home_Cobre'] <= 0.40),
        (df['Prob_Home_Cobre'] >= 0.55) | (df['Prob_Home_Cobre'] <= 0.45)
    ]
    
    choices_confianca = ['🎯 MUITO ALTA', '✅ ALTA', '📊 MÉDIA', '⚖️ BAIXA']
    df['Confianca'] = np.select(conditions_confianca, choices_confianca, default='⚖️ BAIXA')
    
    # Gerar recomendações específicas
    def criar_recomendacao(row):
        prob_home = row['Prob_Home_Cobre']
        linha = row['Asian_Line']
        
        if prob_home >= 0.65:
            return f"🎯 HOME +{linha} ({(prob_home*100):.0f}% conf)"
        elif prob_home <= 0.35:
            return f"🎯 AWAY {linha} ({((1-prob_home)*100):.0f}% conf)"
        elif prob_home >= 0.55:
            return f"✅ HOME +{linha} ({(prob_home*100):.0f}% conf)" 
        elif prob_home <= 0.45:
            return f"✅ AWAY {linha} ({((1-prob_home)*100):.0f}% conf)"
        else:
            return "⚖️ ANALISAR (mercado equilibrado)"
    
    df['Recomendacao'] = df.apply(criar_recomendacao, axis=1)
    
    # Calcular valor da recomendação (distância da linha neutra)
    df['Valor_Recomendacao'] = abs(df['Prob_Home_Cobre'] - 0.5)
    df['Ranking'] = df['Valor_Recomendacao'].rank(ascending=False, method='dense').astype(int)
    
    return df

# =====================================================
# BLOCO 7 - RELATÓRIO EXECUTIVO
# =====================================================

def criar_relatorio_executivo(df):
    """Cria relatório executivo completo"""
    
    st.markdown("## 📊 RELATÓRIO EXECUTIVO - HANDICAP ASIÁTICO")
    
    if df.empty:
        st.info("📝 Nenhum dado disponível para análise")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(df)
        st.metric("Total de Jogos", total)
    
    with col2:
        recomendados = len(df[df['Confianca'].isin(['🎯 MUITO ALTA', '✅ ALTA'])])
        st.metric("Jogos Recomendados", recomendados)
    
    with col3:
        home_recom = len(df[df['Recomendacao'].str.contains('HOME')])
        st.metric("HOME Recomendado", home_recom)
    
    with col4:
        away_recom = len(df[df['Recomendacao'].str.contains('AWAY')])
        st.metric("AWAY Recomendado", away_recom)
    
    # Melhores oportunidades
    st.markdown("### 🎯 MELHORES OPORTUNIDADES")
    
    melhores = df[df['Confianca'].isin(['🎯 MUITO ALTA', '✅ ALTA'])].copy()
    melhores = melhores.sort_values('Valor_Recomendacao', ascending=False)
    
    if not melhores.empty:
        cols_display = ['Ranking', 'Home', 'Away', 'League', 'Asian_Line', 
                       'Prob_Home_Cobre', 'Recomendacao', 'Confianca']
        
        # Formatar a exibição
        display_df = melhores[cols_display].head(10).copy()
        display_df['Prob_Home_Cobre'] = (display_df['Prob_Home_Cobre'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("🤔 Nenhuma oportunidade de alta confiança encontrada hoje")
    
    # Distribuição de confiança
    st.markdown("### 📈 DISTRIBUIÇÃO DE CONFIÂNCIA")
    dist_confianca = df['Confianca'].value_counts()
    st.bar_chart(dist_confianca)

# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================

def main():
    """Função principal do sistema"""
    
    st.sidebar.markdown("## ⚙️ CONFIGURAÇÕES")
    
    # Bloco 2 - Carregar dados
    with st.spinner("🔄 Carregando dados..."):
        games_today, history, arquivo = carregar_dados_simples()
    
    if games_today is None:
        return
    
    st.success(f"✅ Dados carregados: {len(games_today)} jogos encontrados")
    
    # Bloco 4 - Preparar features e modelo
    with st.spinner("🤖 Treinando modelo de machine learning..."):
        X, y, X_today, history, games_today = preparar_features_ml(history, games_today)
        modelo = treinar_modelo_handicap(X, y)
    
    if modelo is None:
        st.error("❌ Não foi possível treinar o modelo. Verifique os dados.")
        return
    
    st.success("✅ Modelo treinado com sucesso!")
    
    # Fazer previsões
    try:
        probabilidades = modelo.predict_proba(X_today)[:, 1]  # Prob Home cobre
        games_today['Prob_Home_Cobre'] = probabilidades
        games_today['Prob_Away_Cobre'] = 1 - probabilidades
    except Exception as e:
        st.error(f"❌ Erro nas previsões: {e}")
        return
    
    # Bloco 6 - Gerar recomendações
    games_com_recomendacoes = gerar_recomendacoes_inteligentes(games_today)
    
    # Bloco 5 - Visualizações
    st.markdown("## 📊 ANÁLISE VISUAL")
    st.pyplot(plot_resultados_handicap(games_com_recomendacoes))
    
    # Bloco 7 - Relatório
    criar_relatorio_executivo(games_com_recomendacoes)
    
    # Tabela completa
    st.markdown("### 📋 TODOS OS JOGOS ANALISADOS")
    
    cols_finais = ['Ranking', 'Home', 'Away', 'League', 'Asian_Line', 
                   'Prob_Home_Cobre', 'Prob_Away_Cobre', 'Recomendacao', 'Confianca']
    cols_finais = [c for c in cols_finais if c in games_com_recomendacoes.columns]
    
    display_final = games_com_recomendacoes[cols_finais].sort_values('Ranking').copy()
    display_final['Prob_Home_Cobre'] = (display_final['Prob_Home_Cobre'] * 100).round(1).astype(str) + '%'
    display_final['Prob_Away_Cobre'] = (display_final['Prob_Away_Cobre'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(display_final, use_container_width=True)
    
    # Informações técnicas
    with st.expander("🔍 INFORMAÇÕES TÉCNICAS"):
        st.write(f"📊 **Total de jogos analisados:** {len(games_com_recomendacoes)}")
        st.write(f"🤖 **Modelo utilizado:** Random Forest Classifier")
        st.write(f"🎯 **Features utilizadas:** {X.shape[1]} variáveis")
        st.write(f"📈 **Acurácia do modelo (treino):** {modelo.score(X, y):.1%}")

# Executar a aplicação
if __name__ == "__main__":
    main()
