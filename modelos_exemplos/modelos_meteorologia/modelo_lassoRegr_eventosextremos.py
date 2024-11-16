"""
Utilize um conjunto de dados com múltiplas variáveis meteorológicas (temperatura, umidade, pressão atmosférica, 
velocidade do vento, precipitação) para prever eventos extremos de temperatura. 
Aplique regressão de Lasso para selecionar as características mais importantes.
"""

from sklearn.linear_model import LassoCV
import numpy as np

#dados [temperatura, umidade, pressão atm, velocidade do vento, precipitação, eventos extremos]
dados = np.array([
    [22, 65, 1015, 12, 5, 1],
    [23, 70, 1012, 14, 7, 0],
    [24, 75, 1009, 16, 10, 1],
    [25, 80, 1007, 18, 12, 0], 
    [26, 85, 1005, 20, 15, 1],
    [27, 90, 1003, 22, 17, 0],
    [28, 95, 1001, 24, 20, 1]
])

x = dados[:, :-1] #temp, umd, pres, vel.vento, precip
y = dados[:, -1] #eventos extremos

modelo = LassoCV(cv=5)
modelo.fit(x, y)

novos_dados = np.array([[25, 82, 1006, 19, 8]])
previsao_evento_extremo = modelo.predict(novos_dados)

print(f'Probablidade de evento extremos prevista: {previsao_evento_extremo[0]:.2f}')