import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Carregar os dados
dados = pd.read_csv('historical_stock_prices2.csv')

# Converter a coluna 'Date' para o formato datetime com dia-mês-ano
dados['Date'] = pd.to_datetime(dados['Date'], format='%d-%m-%Y')

# Definir a coluna 'Date' como índice
dados.set_index('Date', inplace=True)

# Calcular os retornos diários
dados['Return'] = dados['Close'].pct_change().dropna()

# Verificar se a coluna 'Return' não está vazia
if dados['Return'].isnull().all():
    raise ValueError("A coluna 'Return' está vazia. Verifique os dados de entrada.")

# Definir o modelo ARIMA
model = ARIMA(dados['Return'], order=(5,1,0))
model_fit = model.fit()

# Fazer previsões
predictions = model_fit.forecast(steps=30)

# Criar um índice de datas para as previsões
last_date = dados.index[-1]
prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Visualizar os resultados com melhorias na visualização
plt.figure(figsize=(12, 6))
plt.plot(dados['Return'], label='Retornos Históricos', linewidth=1.5)
plt.plot(prediction_dates, predictions, label='Previsão de Volatilidade', color='red', linewidth=2.5)
plt.xlabel('Data')
plt.ylabel('Retorno')
plt.title('Previsão de Volatilidade com ARIMA')
plt.legend()
plt.grid(True)
plt.show()

plt.gca().xaxis.set_major_locator(mdates.MonthLocator()) #Define o localizador para os principais ticks do eixo x para intervalos mensais.
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) #Formata os ticks do eixo x para mostrar o nome do mês abreviado e o ano (por exemplo, "Jan 2024").

plt.gcf().autofmt_xdate() #Rotaciona os rótulos das datas no eixo x para melhorar a legibilidade.

# Salvar o gráfico
plt.savefig('previsao_volatilidade2.png', dpi=300, bbox_inches='tight')
