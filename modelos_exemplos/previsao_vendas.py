"""
Crie um modelo de regressão para prever vendas de produtos considerando um efeito sazonal 
(ex: aumento de vendas em feriados). 
Utilize transformações de recursos para incorporar esta informação.

Treina um modelo de regressão e prevê vendas para um novo dia, incorporando o efeito dos feriados.

"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# dados de exemplo (dia, feriado[0, 1], vendas)
dados = np.array([[1, 0, 100],
                  [2, 0, 110],
                  [3, 1, 150], #feriado
                  [4, 0, 130],
                  [5, 0, 120],
                  [6, 1, 160], #feriado
                  [7, 0, 140]
])

x = dados[:, :-1] #dia e feriado
y = dados[:, -1] #vendas

#criação e treinamento do modelo
modelo_vendas = LinearRegression()
modelo_vendas.fit(x,y)

#Previsão dos novos dados
novos_dados = np.array([8, 0]).reshape(1, -1) #transforma o array em coluna para um dia que não é feriado (0)
previsao_vendas = modelo_vendas.predict(novos_dados)

#resultado
print(f'Previsão de vendas: {previsao_vendas[0]:.2f} unidades' )
