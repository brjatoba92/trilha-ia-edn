# Importação das bibliotecas
import numpy as np #arrays
import pandas as pd #manipulação de arquivos
import matplotlib.pyplot as plt #graficos
import os #criação do diretorio, checando se há ou não
from sklearn.preprocessing import StandardScaler #Escalonamento dos dados
from sklearn.model_selection import train_test_split #Conjuntos de dados de treino
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor #Modelo de Rede Neural
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report #Erro medio quadratico, acuracia e relatorio
from sklearn.cluster import KMeans 
import datetime

#Criação do diretorio
def criar_diretorio(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

# Configuração da semente para reprodutibilidade
np.random.seed(42)

# Função para gerar dados simulados de mercado financeiro
def gerar_dados_mercado_financeiro(num_amostras=1000):
    # Simulando retornos de ações com distribuição normal, criando dataframe
    dados = pd.DataFrame({
        'data': [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(num_amostras)],
        'preco_abertura': np.random.normal(100, 15, num_amostras),
        'preco_fechamento': np.random.normal(100, 15, num_amostras),
        'volume_negociacao': np.random.normal(1000000, 250000, num_amostras),
        'volatilidade': np.abs(np.random.normal(0, 0.02, num_amostras)),
        'retorno': np.random.normal(0.001, 0.02, num_amostras)
    })
    
    dados['retorno_acumulado'] = dados['retorno'].cumsum()
    
    return dados

# 1. Modelo de Regressão de Previsão de Preços com Random Forest
"""
- Informar as colunas de amostras e alvo
- Separação dos dados em conjuntos de treino e teste
- Escalonamento dos dados
- Modelo de Regressão Forest
- Previsão
- Metrica de Erro Medio Quadratico
- Salva os resultados em arquivos txt
- Gera graficos
"""
def modelo_previsao_precos(dados):
    X = dados[['volume_negociacao', 'volatilidade', 'retorno_acumulado']].values
    y = dados['preco_fechamento'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_scaled, y_train)
    
    predicoes = rf_regressor.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predicoes)
    
    # Salvando resultados
    criar_diretorio('resultados_modelos_financas')
    with open('resultados_modelos_financas/resultado_random_forest.txt', 'w') as f:
        f.write(f"Erro Quadrático Médio: {mse}\n")
        f.write("Previsões vs Valores Reais:\n")
        for real, prev in zip(y_test[:10], predicoes[:10]):
            f.write(f"Real: {real:.2f}, Previsto: {prev:.2f}\n")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predicoes, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Previsão de Preços - Random Forest')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.savefig('resultados_modelos_financas/previsao_precos_random_forest.png')
    plt.close()

# Modelo 2 - Classificação da Tendência
def modelo_classificacao_tendencia(dados):
    """
    - Cria a coluna tendencia
    - Informar as colunas de amostras e alvo
    - Separação dos dados em conjuntos de treino e teste
    - Cria a metrica de GradienteBoosting
    - Treina a metrica
    - Previsão
    - Metrica de Acuracia
    - Cria diretorio
    - Salva os dados
    """
    dados['tendencia'] = np.where(dados['retorno'] > 0, 1, 0)

    X = dados[['volume_negociacao', 'volatilidade', 'retorno_acumulado']].values
    y = dados['tendencia'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)

    predicoes = gb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicoes)

    criar_diretorio('resultados_modelos_financas')
    with open('resultados_modelos_financas/resultado_classificacao_tendencia.txt', 'w') as f:
        f.write(f"Acurácia do Modelo: {accuracy}\n")
        f.write(classification_report(y_test, predicoes))

# Modelo 3 - Regressão Suport Vector
def modelo_regressao_suport_vetorial(dados):
    #Amostras e alvo
    X = dados[['volume_negociacao', 'volatilidade']].values
    y = dados['retorno'].values

    #Conjuntos de dados de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #Escalonamento dos dados de treino e de teste
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Criação do modelo e treino
    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train)

    #Previsão e metrica de erro quadratico medio
    predicoes = svr.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predicoes)
    
    #Criação do diretorio e salva os dados
    criar_diretorio('resultados_modelos_financas')
    with open('resultados_modelos_financas/resultado_svr.txt', 'w') as f:
        f.write(f"Erro Quadrático Médio (SVR): {mse}\n")

# Modelo 4 - Redes Neurais
def modelo_rede_neural(dados):
    
    #Amostras e alvo
    X = dados[['preco_abertura', 'preco_fechamento', 'volume_negociacao']].values
    y = dados['volatilidade'].values
    
    #Conjuntos de dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #Escalonamento dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Modelo de Regressão com Rede Neural e treino
    mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    #Previsão e metrica de erro quaratico medio
    predicoes = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predicoes)

    #Criação do diretorio e salva os dados
    criar_diretorio('resultados_modelos_financas')
    with open('resultados_modelos_financas/resultado_rede_neural.txt', 'w') as f:
        f.write(f'Erro Quadrático Médio (Rede Neural): {mse}\n')

# Modelo 5 - Agrupamento de Ativos
def modelo_agrupamento_ativos(dados):
    
    #Array dos dados
    X = dados[['preco_fechamento', 'volume_negociacao', 'retorno']].values
    
    #Escalonamento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Clusterização
    kmeans = KMeans(n_clusters=3, random_state=42)
    dados['cluster'] = kmeans.fit_predict(X_scaled)

    #Cria diretorio e salva os arquivos
    criar_diretorio('resultados_modelos_financas')
    dados.to_csv('resultados_modelos_financas/dados_agrupados.csv', index=False)
    with open('resultados_modelos_financas/resultados_agrupamento.txt', 'w') as f:
        f.write(f'Centroides de Clusters:\n')
        for i, centroide in enumerate(kmeans.cluster_centers_):
            f.write(f'Cluster {i}: {centroide}\n')
    plt.figure(figsize=(10, 6))
    plt.scatter(dados['preco_fechamento'], dados['volume_negociacao'], c=dados['cluster'], cmap='viridis')
    plt.title('Agrupamento de ativos')
    plt.xlabel('Preço de Fechamento')
    plt.ylabel('Volume de negociação')
    plt.colorbar(label='Cluster')
    plt.savefig('resultados_modelos_financas/agrupamento_ativos.png')
    plt.close()


# Executando os modelos
dados = gerar_dados_mercado_financeiro()
modelo_previsao_precos(dados)
modelo_classificacao_tendencia(dados)
modelo_regressao_suport_vetorial(dados)
modelo_rede_neural(dados)
modelo_agrupamento_ativos(dados)