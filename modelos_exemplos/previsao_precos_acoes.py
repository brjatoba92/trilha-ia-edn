"""
Utilize um conjunto de dados histórico de preços de ações (dados de fechamento diário) para prever 
o preço de fechamento do próximo dia. 
Você precisará criar um modelo de regressão que utilize como entrada os preços de fechamento dos dias anteriores.
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

#array do fechamento das ações no ultimos 5 dias
precos_ibovespa = np.array([130515, 130661, 130341, 129682, 127830]) #semana passada valores de fechamento ibovespa

#criação da matriz de caracteristicas
x = np.array([precos_ibovespa[i:i+4] for i in range(len(precos_ibovespa)-4)]) #4 dias consecutivos 
y = precos_ibovespa[4:] #preco de fechamento do dia seguinte

# criação e treinamento modelo
modelo_acoes = LinearRegression() #criando
modelo_acoes.fit(x, y) #treinando

#previsão para o proximo dia
novo_dia = np.array([130661, 130341, 129682, 127845]).reshape(1, -1)
previsao_preco = modelo_acoes.predict(novo_dia)

print(f'O preço do fechamento para o dia seguinte é igual a {previsao_preco[0]:.2f}')



