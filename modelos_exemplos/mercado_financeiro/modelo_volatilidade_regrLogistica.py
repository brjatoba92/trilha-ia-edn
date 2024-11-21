"""
Previsão de Volatilidade do Mercado com Regressão Logística
Descrição: Usar regressão logística para prever se o mercado será volátil baseado em indicadores econômicos
"""

#Importação de bibliotecas
import pandas as pd #manipulação e análise de dados.
from sklearn.model_selection import train_test_split #dividir os dados em conjuntos de treinamento e teste.
from sklearn.linear_model import LogisticRegression #criar o modelo de regressão logística.
from sklearn.metrics import accuracy_score #calcular a precisão das previsões do modelo.
import numpy as np #operações com arrays.

## PARTE 1 - TRANSFORMANDO UM DICIONARIO PARA SER SALVO EM UM ARQUIVO, USANDO O PANDAS

# Dados de exemplo
dados = {
    'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
    'GDP': [21000, 21100, 21300, 21500, 21600],
    'unemployment_rate': [5.0, 4.8, 4.6, 4.7, 4.5],
    'interest_rate': [2.5, 2.4, 2.3, 2.6, 2.4],
    'volatility': [1, 0, 1, 0, 1]
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Salvar DataFrame como CSV
df.to_csv('market_volatility.csv', index=False)

## PARTE 2 - PROBLEMA PROPOSTO

#Carregando os dados
dados = pd.read_csv('market_volatility.csv')

#selecionar as caracteristicas e o alvo
x = dados[['GDP', 'unemployment_rate', 'interest_rate']] #Seleciona as colunas GDP, unemployment_rate e interest_rate como características (variáveis independentes) e as armazena na variável x.
y = dados[['volatility']] #Seleciona a coluna volatility como o alvo (variável dependente) e a armazena na variável y

#dividir dados em conjuntos de treinamento e teste
"""
Divide os dados em conjuntos de treinamento e teste. 
test_size=0.2 indica que 20% dos dados serão usados para teste e 80% para treinamento. 
random_state=42 garante que a divisão dos dados seja reprodutível.
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Ajuste necessário para y
y_train = np.ravel(y_train) #ransforma y_train em um array unidimensional, conforme esperado pelo modelo de regressão logística.

#criar e treinar o modelo
modelo = LogisticRegression() #Cria uma instância do modelo de regressão logística.
modelo.fit(x_train, y_train) #Treina o modelo de regressão logística usando os dados de treinamento (x_train e y_train).

#Realizar previsões
previsoes = modelo.predict(x_test) #Usa o modelo treinado para fazer previsões nos dados de teste (x_test) e armazena as previsões na variável previsoes

#Modelo avaliado
accuracy = accuracy_score(y_test, previsoes) #Calcula a precisão do modelo comparando os valores reais (y_test) com as previsões (previsoes).
print(f'Accuracy: {accuracy}') #Imprime o valor da precisão.
