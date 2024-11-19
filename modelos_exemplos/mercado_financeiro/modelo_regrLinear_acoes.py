"""
Utilize o modelo de regressão linear do sklearn para prever os preços das ações com base 
em dados históricos.

1. Descrição:
Dados: Utilize um conjunto de dados de preços históricos de uma ação.
Objetivo: Prever o preço da ação para uma data futura.

2. Passos:
Carregue os dados históricos dos preços das ações.
Divida os dados em conjuntos de treinamento e teste.
Utilize numpy para manipulação dos dados.
Crie e treine um modelo de regressão linear com sklearn.
Avalie a precisão do modelo e faça previsões.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#carregando os dados
dados = pd.read_csv('historical_stock_prices2.csv')


x = np.array(dados.index).reshape(-1, 1)
y = dados['Close'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Índice')
plt.ylabel('Preço de Fechamento') 
plt.title('Previsão de Preços de Ações')
plt.show()

# Salvar o gráfico
plt.savefig('previsao_acoes.png', dpi=300, bbox_inches='tight')

