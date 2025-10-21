# ---------------- ML PARA RANKEAR CONFRONTOS (HOME E AWAY) ----------------
def treinar_modelo_quadrantes_dual(history, games_today):
    """Treina modelo ML para Home e Away baseado em quadrantes + ligas"""
    
    # Preparar features: one-hot encoding de quadrantes e ligas
    quadrantes_home = pd.get_dummies(history['Quadrante_Home'], prefix='QH')
    quadrantes_away = pd.get_dummies(history['Quadrante_Away'], prefix='QA')
    ligas_dummies = pd.get_dummies(history['League'], prefix='League')
    
    # Combinar features
    X = pd.concat([quadrantes_home, quadrantes_away, ligas_dummies], axis=1)
    
    # Target para Home
    y_home = history['Target_AH_Home']
    
    # Target para Away (inverso do Home)
    y_away = 1 - y_home  # Quando Home perde, Away ganha
    
    # Garantir que todas as colunas existem
    quadrante_cols = list(quadrantes_home.columns) + list(quadrantes_away.columns)
    liga_cols = list(ligas_dummies.columns)
    
    # Treinar modelo para Home
    model_home = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model_home.fit(X, y_home)
    
    # Treinar modelo para Away
    model_away = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model_away.fit(X, y_away)
    
    # Preparar dados de hoje
    qh_today = pd.get_dummies(games_today['Quadrante_Home'], prefix='QH').reindex(columns=quadrantes_home.columns, fill_value=0)
    qa_today = pd.get_dummies(games_today['Quadrante_Away'], prefix='QA').reindex(columns=quadrantes_away.columns, fill_value=0)
    ligas_today = pd.get_dummies(games_today['League'], prefix='League').reindex(columns=liga_cols, fill_value=0)
    
    X_today = pd.concat([qh_today, qa_today, ligas_today], axis=1)
    
    # Fazer previsões para ambos os lados
    probas_home = model_home.predict_proba(X_today)[:, 1]
    probas_away = model_away.predict_proba(X_today)[:, 1]
    
    games_today['Quadrante_ML_Score_Home'] = probas_home
    games_today['Quadrante_ML_Score_Away'] = probas_away
    
    # Score principal (maior probabilidade entre Home e Away)
    games_today['Quadrante_ML_Score_Main'] = np.maximum(probas_home, probas_away)
    games_today['ML_Side'] = np.where(probas_home > probas_away, 'HOME', 'AWAY')
    
    return model_home, model_away, games_today

# ---------------- SISTEMA DE INDICAÇÕES EXPLÍCITAS DUAL ----------------
def adicionar_indicadores_explicativos_dual(df):
    """Adiciona classificações e recomendações explícitas para Home e Away"""
    df = df.copy()
    
    # 1. CLASSIFICAÇÃO DE VALOR PARA HOME
    conditions_home = [
        df['Quadrante_ML_Score_Home'] >= 0.60,
        df['Quadrante_ML_Score_Home'] >= 0.55,
        df['Quadrante_ML_Score_Home'] >= 0.50,
        df['Quadrante_ML_Score_Home'] >= 0.45,
        df['Quadrante_ML_Score_Home'] < 0.45
    ]
    choices_home = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Home'] = np.select(conditions_home, choices_home, default='⚖️ NEUTRO')
    
    # 2. CLASSIFICAÇÃO DE VALOR PARA AWAY
    conditions_away = [
        df['Quadrante_ML_Score_Away'] >= 0.60,
        df['Quadrante_ML_Score_Away'] >= 0.55,
        df['Quadrante_ML_Score_Away'] >= 0.50,
        df['Quadrante_ML_Score_Away'] >= 0.45,
        df['Quadrante_ML_Score_Away'] < 0.45
    ]
    choices_away = ['🏆 ALTO VALOR', '✅ BOM VALOR', '⚖️ NEUTRO', '⚠️ CAUTELA', '🔴 ALTO RISCO']
    df['Classificacao_Valor_Away'] = np.select(conditions_away, choices_away, default='⚖️ NEUTRO')
    
    # 3. RECOMENDAÇÃO DE APOSTA DUAL
    def gerar_recomendacao_dual(row):
        home_q = row['Quadrante_Home_Label']
        away_q = row['Quadrante_Away_Label']
        score_home = row['Quadrante_ML_Score_Home']
        score_away = row['Quadrante_ML_Score_Away']
        ml_side = row['ML_Side']
        
        # Combinações específicas com perspectiva dual
        if home_q == 'Underdog Value' and away_q == 'Market Overrates':
            return f'🎯 VALUE NO HOME ({score_home:.1%})'
        elif home_q == 'Market Overrates' and away_q == 'Underdog Value':
            return f'🎯 VALUE NO AWAY ({score_away:.1%})'
        elif home_q == 'Favorite Reliable' and away_q == 'Weak Underdog':
            return f'💪 FAVORITO HOME ({score_home:.1%})'
        elif home_q == 'Weak Underdog' and away_q == 'Favorite Reliable':
            return f'💪 FAVORITO AWAY ({score_away:.1%})'
        elif ml_side == 'HOME' and score_home >= 0.55:
            return f'📈 MODELO CONFIA HOME ({score_home:.1%})'
        elif ml_side == 'AWAY' and score_away >= 0.55:
            return f'📈 MODELO CONFIA AWAY ({score_away:.1%})'
        elif 'Market Overrates' in home_q and score_away >= 0.55:
            return f'🔴 HOME SUPERAVALIADO → AWAY ({score_away:.1%})'
        elif 'Market Overrates' in away_q and score_home >= 0.55:
            return f'🔴 AWAY SUPERAVALIADO → HOME ({score_home:.1%})'
        else:
            return f'⚖️ ANALISAR (H:{score_home:.1%} A:{score_away:.1%})'
    
    df['Recomendacao'] = df.apply(gerar_recomendacao_dual, axis=1)
    
    # 4. RANKING POR MELHOR PROBABILIDADE
    df['Ranking'] = df['Quadrante_ML_Score_Main'].rank(ascending=False, method='dense').astype(int)
    
    return df

def estilo_tabela_quadrantes_dual(df):
    """Aplica estilo colorido à tabela dual"""
    def cor_classificacao(valor):
        if '🏆 ALTO VALOR' in str(valor): return 'background-color: #90EE90; font-weight: bold'
        elif '✅ BOM VALOR' in str(valor): return 'background-color: #FFFFE0; font-weight: bold' 
        elif '🔴 ALTO RISCO' in str(valor): return 'background-color: #FFB6C1; font-weight: bold'
        elif 'VALUE' in str(valor): return 'background-color: #98FB98'
        elif 'EVITAR' in str(valor): return 'background-color: #FFCCCB'
        elif 'SUPERAVALIADO' in str(valor): return 'background-color: #FFA07A'
        else: return ''
    
    return df.style.applymap(cor_classificacao, subset=['Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao'])\
                  .background_gradient(subset=['Quadrante_ML_Score_Home'], cmap='RdYlGn')\
                  .background_gradient(subset=['Quadrante_ML_Score_Away'], cmap='RdYlGn')\
                  .background_gradient(subset=['Quadrante_ML_Score_Main'], cmap='RdYlGn')

# ---------------- ANÁLISE DE PADRÕES DUAL ----------------
def analisar_padroes_quadrantes_dual(df):
    """Analisa padrões recorrentes nas combinações de quadrantes com perspectiva dual"""
    st.markdown("### 🔍 Análise de Padrões por Combinação (Dual)")
    
    padroes = {
        'Underdog Value vs Market Overrates': {
            'descricao': '🎯 **MELHOR PADRÃO HOME** - Zebra com valor vs Favorito supervalorizado',
            'lado_recomendado': 'HOME',
            'prioridade': 1
        },
        'Market Overrates vs Underdog Value': {
            'descricao': '🎯 **MELHOR PADRÃO AWAY** - Favorito supervalorizado vs Zebra com valor', 
            'lado_recomendado': 'AWAY',
            'prioridade': 1
        },
        'Favorite Reliable vs Weak Underdog': {
            'descricao': '💪 **PADRÃO FORTE HOME** - Favorito confiável contra time fraco',
            'lado_recomendado': 'HOME',
            'prioridade': 2
        },
        'Weak Underdog vs Favorite Reliable': {
            'descricao': '💪 **PADRÃO FORTE AWAY** - Time fraco contra favorito confiável',
            'lado_recomendado': 'AWAY', 
            'prioridade': 2
        }
    }
    
    # Ordenar padrões por prioridade
    padroes_ordenados = sorted(padroes.items(), key=lambda x: x[1]['prioridade'])
    
    for padrao, info in padroes_ordenados:
        home_q, away_q = padrao.split(' vs ')
        jogos = df[
            (df['Quadrante_Home_Label'] == home_q) & 
            (df['Quadrante_Away_Label'] == away_q)
        ]
        
        if not jogos.empty:
            st.write(f"**{padrao}**")
            st.write(f"{info['descricao']}")
            
            # Selecionar colunas baseadas no lado recomendado
            if info['lado_recomendado'] == 'HOME':
                score_col = 'Quadrante_ML_Score_Home'
            else:
                score_col = 'Quadrante_ML_Score_Away'
                
            cols_padrao = ['Ranking', 'Home', 'Away', 'League', score_col, 'Recomendacao']
            cols_padrao = [c for c in cols_padrao if c in jogos.columns]
            
            st.dataframe(
                jogos[cols_padrao]
                .sort_values(score_col, ascending=False)
                .style.format({score_col: '{:.1%}'})
                .background_gradient(subset=[score_col], cmap='RdYlGn'),
                use_container_width=True
            )
            st.write("---")

# ---------------- ATUALIZAR EXECUÇÃO PRINCIPAL ----------------

# No treinamento do modelo, substituir a chamada anterior:
if not history.empty:
    modelo_home, modelo_away, games_today = treinar_modelo_quadrantes_dual(history, games_today)
    st.success("✅ Modelo dual (Home/Away) treinado com sucesso!")
else:
    st.warning("⚠️ Histórico vazio - não foi possível treinar o modelo")

# ---------------- EXIBIÇÃO DOS RESULTADOS DUAL ----------------
st.markdown("## 🏆 Melhores Confrontos por Quadrantes ML (Home & Away)")

if not games_today.empty and 'Quadrante_ML_Score_Home' in games_today.columns:
    # Preparar dados para exibição
    ranking_quadrantes = games_today.copy()
    ranking_quadrantes['Quadrante_Home_Label'] = ranking_quadrantes['Quadrante_Home'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    ranking_quadrantes['Quadrante_Away_Label'] = ranking_quadrantes['Quadrante_Away'].map(
        lambda x: QUADRANTES_8.get(x, {}).get('nome', 'Neutro') if x != 0 else 'Neutro'
    )
    
    # Aplicar indicadores explicativos dual
    ranking_quadrantes = adicionar_indicadores_explicativos_dual(ranking_quadrantes)
    
    # Ordenar por score principal
    ranking_quadrantes = ranking_quadrantes.sort_values('Quadrante_ML_Score_Main', ascending=False)
    
    # Colunas para exibir
    cols_finais = [
        'Ranking', 'Home', 'Away', 'League', 'ML_Side',
        'Quadrante_Home_Label', 'Quadrante_Away_Label',
        'Quadrante_ML_Score_Home', 'Quadrante_ML_Score_Away', 
        'Classificacao_Valor_Home', 'Classificacao_Valor_Away', 'Recomendacao'
    ]
    
    # Filtrar colunas existentes
    cols_finais = [c for c in cols_finais if c in ranking_quadrantes.columns]
    
    st.dataframe(
        estilo_tabela_quadrantes_dual(ranking_quadrantes[cols_finais].head(20)),
        use_container_width=True
    )
    
    # Análise de padrões dual
    analisar_padroes_quadrantes_dual(ranking_quadrantes)
    
    # Resumo de distribuição
    st.markdown("#### 📊 Distribuição de Lados Recomendados")
    dist_lados = ranking_quadrantes['ML_Side'].value_counts()
    st.dataframe(dist_lados, use_container_width=True)
    
else:
    st.info("⚠️ Aguardando dados para gerar ranking dual")
