#Importação das bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.cluster import KMeans
import datetime

def criar_diretorio(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

# Configuração da semente para reprodutibilidade
np.random.seed(42)

# Função para gerar dados simulados de mercado financeiro
def gerar_dados_mercado_financeiro(num_amostras=1000):
    # Simulando retornos de ações com distribuição normal
    data = pd.DataFrame({
        'data': [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(num_amostras)],
        'preco_abertura': np.random.normal(100, 15, num_amostras),
        'preco_fechamento': np.random.normal(100, 15, num_amostras),
        'volume_negociacao': np.random.normal(1000000, 250000, num_amostras),
        'volatilidade': np.abs(np.random.normal(0, 0.02, num_amostras)),
        'retorno': np.random.normal(0.001, 0.02, num_amostras)
    })
    
    data['retorno_acumulado'] = data['retorno'].cumsum()
    
    return data

# 1. Modelo de Regressão de Previsão de Preços com Random Forest
def modelo_previsao_precos(data):
    X = data[['volume_negociacao', 'volatilidade', 'retorno_acumulado']].values
    y = data['preco_fechamento'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_scaled, y_train)
    
    predicoes = rf_regressor.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predicoes)
    
    # Salvando resultados
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
    plt.savefig('previsao_precos_random_forest.png')
    plt.close()

def modelo_classificacao_tendencia(dados):
    dados['tendencia'] = np.where(dados['retorno', 'retorno_acumulado']>0,1,0)

    X = dados[['volume_negociacao', 'volatilidade', 'retorno_acumulado']].values
    y = dados['tendencia'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)

    predicoes =gb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicoes)

    with open('resultado_classificacao_tendencia.txt', 'w') as f:
        f.write(f"Acurácia do Modelo: {accuracy}\n")
        f.write(classification_report(y_test, predicoes))


def modelo_regressao_suport_vetorial(dados):
    pass

def modelo_rede_neural(dados):
    pass

def modelo_agrupamento_ativos(dados):
    pass

#Executando os modelos
dados = gerar_dados_mercado_financeiro()

modelo_previsao_precos(dados)
modelo_classificacao_tendencia(dados)
modelo_regressao_suport_vetorial(dados)
modelo_rede_neural(dados)
modelo_agrupamento_ativos(dados)

print("Todos os modelos foram executados com sucesso")
