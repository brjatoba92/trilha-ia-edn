"""
Análise de Séries Temporais de Temperatura
Enunciado do Exercício:
Objetivo: Analisar as tendências e padrões de temperatura ao longo do tempo usando dados históricos.
"""

#Importação de Bibliotecas
import numpy as np #array
import pandas as pd #manipulação de dados
from sklearn.model_selection import train_test_split #dividir os dados em blocos de treino e teste
from sklearn.preprocessing import StandardScaler #normalização dos dados
from statsmodels.tsa.arima.model import ARIMA #analise de series temporais
from sklearn.metrics import mean_squared_error #calculo do erro quadratico medio
import matplotlib.pyplot as plt #visualização de graficos

#Geração de dados simulados
np.random.seed(42) #gerar números aleatórios de forma repetível.
dates = pd.date_range('20230101', periods=100) #Cria uma sequência de datas, começando em 1º de janeiro de 2023, com um total de 100 períodos
data = {
    'temperature': np.random.uniform(15,30,100) #Gera 100 valores de temperatura distribuídos uniformemente entre 15 e 30(minimo, maximo, qtd_de_dados)
}

#Transformar em DataFrame
df = pd.DataFrame(data, index=dates) #Cria um DataFrame com os dados simulados, usando as datas como índice.

#Salvar em arquivo CSV
df.to_csv('temperature_series.csv', index=True) #Salva o DataFrame em um arquivo CSV chamado 'temperature_series.csv', incluindo o índice

#Carregando dados do csv
data = pd.read_csv('temperature_series.csv', index_col=0, parse_dates=True) #Carrega os dados do arquivo CSV, definindo a primeira coluna como o índice e interpretando as colunas de data corretamente.

#Treinar modelo ARIMA
"""
Cria um modelo ARIMA (Autoregressive Integrated Moving Average) com ordem (5, 1, 0),
5: Número de lags autoregressivos (AR).
1: Número de diferenciações (I) para tornar a série estacionária.
0: Número de lags da média móvel (MA).
"""
model = ARIMA(data['temperature'], order=(5, 1, 0))
model_fit = model.fit() #Ajusta o modelo ARIMA aos dados de temperatura.

#Prever dados
predictions = model_fit.forecast(steps=10) #Faz previsões para os próximos 10 períodos

#Avaliar modelo
mse = mean_squared_error(data['temperature'][-10:], predictions) #Calcula o Erro Quadrático Médio entre os valores reais e previstos
print(f'Erro Quadratico Médio (MSE): {mse:.2f}') #Exibe o MSE no console

#Visualizar resultados
data['temperature'].plot(label='Real') #Plota a série temporal de temperatura real.
predictions.plot(label='Previsto') #Plota as previsões feitas pelo modelo
plt.xlabel('Data') #eixo x
plt.ylabel('Temperatura') #eixo y
plt.title('Analise de Series Temporais de Temperatura') #titulo do grafico
plt.legend() #legenda
plt.show() #exibe o grafico

plt.savefig('analise_temporal_temperatura.png') #salva a figura do grafico

