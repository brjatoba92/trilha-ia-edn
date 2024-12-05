#Importação de bibliotecas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, r2_score
import seaborn as sns

#Criando diretorio
def criar_diretorio(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)

#Salvar resultados
def salvar_resultados(modelo, nome_modelo, x_test, y_test, predicoes, metricas, tipo='classificação'):
    #Criar diteorio de resultados
    criar_diretorio('resultados')
    caminho_resultados = os.path.join('resultados', nome_modelo)
    criar_diretorio(caminho_resultados)

    #Salvar metricas em txt
    with open(os.path.join(caminho_resultados, 'metricas.txt'), 'w') as f:
        f.write(f"Modelo: {nome_modelo}\n")
        f.write("metricas de Desempemho:\n")
        for metrica, valor in metricas.items():
            f.write(f"{metrica}: {valor}\n")
    
    #Gerar visualizações
    plt.figure(figsize=(10,6))

    if tipo == 'classificacao':
        # matriz de confusão
        cm = confusion_matrix(y_test, predicoes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão = {nome_modelo}')
        plt.xlabel('Real')
        plt.ylabel('Predito')
        plt.tight_layout()
        plt.savefig(os.path.join(caminho_resultados, 'matriz_confusão.png'))
        plt.close()
    else:
        #Grafico de dispersão para regressão
        plt.scatter(y_test, predicoes)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'Valores Reais vs Preditos - {nome_modelo}')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Preditos')
        plt.tight_layout()
        plt.savefig(os.path.join(caminho_resultados, 'dispersão_predicoes.png'))
        plt.close()

#Modelo 1 - Classificação - Prever Resultado do Jogo
"""
Classificação usando Random Forest
Prevê o resultado do jogo (derrota, empate, vitória)
Usa características como posse de bola e desempenho
"""
def modelo_resultado_jogo():
    #Dados simualdos
    np.random.seed(42)
    n_amostras = 1000
    
    #Caracteristicas: Posse de bola, chutes no gol, cartões, etc
    X = np.random.rand(n_amostras, 5)
    #0 -Derrota, 1: Empate, 2: Vitoria
    y = np.random.choice([0,1,2], size=n_amostras)

    #Separação em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X_train_scaled, y_train)

    predicoes = modelo.predict(X_test_scaled)

    metricas = {
        'Acuracia': accuracy_score(y_test, predicoes),
        'Relatorio de Classificação': classification_report(y_test, predicoes)
    }

    salvar_resultados(modelo, 'resultado_jogo', X_test, y_test, predicoes, metricas)
    return metricas

#Modelo 2 - Regressão - Prever Numeros de Gols
def modelo_numero_gols():
    np.random.seed(42)
    n_amostras = 1000
    #Caracteristicas: Historico de Gols, Força do ataque, defesa
    X = np.random.rand(n_amostras, 4)
    y = 2 * X[:,0] + 1.5 * X[:, 1] + np.random.normal(0,0.5, n_amostras)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelo = RandomForestRegressor(n_estimators=100)
    modelo.fit(X_train_scaled, y_train)

    predicoes = modelo.predict(X_test_scaled)

    metricas = {
        'MSE': mean_squared_error(y_test, predicoes),
        'R2': r2_score(y_test, predicoes)
    }

    salvar_resultados(modelo, 'numero_gols', X_test, y_test, predicoes, metricas, tipo='regressao')
    return metricas

#Modelo 3 - Classificação - Prever Cartão Vermelho

#Modelo 4 - Regressão -Prever valor de mercado do jogador

#Modelo 5 - Classificação - Prever Time Campeão

#Executar todos os modelos