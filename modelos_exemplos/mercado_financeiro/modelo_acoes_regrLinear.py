"""
Previsão de Preços de Ações com Regressão Linear
Descrição: Usar regressão linear para prever o preço de fechamento de ações com base em características históricas
"""
#Importando as bibliotecas necessarias
import numpy as np #trabalha com arrays
import pandas as pd #manipulação e e analise de dados
from sklearn.model_selection import train_test_split #dividir os dados em conjuntos de treinamento e teste
from sklearn.linear_model import LinearRegression #cria o modelo de regressão linear
from sklearn.metrics import mean_squared_error #calculo do erro medio quadratico das previsões

## PARTE 1

# Base de dados na forma de dicionario com os dados de exemplo 
dicionario = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Open': [100.0, 102.0, 104.0, 108.0, 110.0],
    'High': [105.0, 107.0, 110.0, 112.0, 115.0],
    'Low': [95.0, 100.0, 103.0, 105.0, 109.0],
    'Close': [102.0, 104.0, 108.0, 110.0, 114.0],
    'Volume': [300000, 350000, 400000, 450000, 500000]
}

# Criando DataFrame
df = pd.DataFrame(dicionario)

# Salvando o DataFrame como CSV
df.to_csv('stock_prices.csv', index=False)


## PARTE 2 - PROBLEMA PROPOSTO

#carregando os dados contidos no arquivo csv e armazenando na variavel dados
dados = pd.read_csv('stock_prices.csv')

#selecionar as caracteristicas e alvo
x = dados[['Open', 'High', 'Low', 'Volume']] #colunas selecionadas das variaveis independentes
y = dados[['Close']] #coluna da variavel dependente

#Testes e treinamento
"""
- Dividir os dados em conjuntos de treinamento e teste
- 20% dos dados serão usados para teste e 80% para treinamento
- Divisão dos dados seja reprodutivel 
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#criar e treinar o modelo
modelo = LinearRegression() #Cria o modelo de regressão linear
modelo.fit(x_train, y_train) #Treina o modelo de regresão linear com os dados de treinamento

#fazer previsoes
previsoes = modelo.predict(x_test) 

#avaliar o modelo
mse = mean_squared_error(y_test, previsoes) #RME entre os valores reais(y_test) e das previsões (previsoes)
print(f'MSE: {mse}') #mostra na tela