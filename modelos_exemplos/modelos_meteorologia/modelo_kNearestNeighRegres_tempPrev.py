"""
Utilize um conjunto de dados com variáveis meteorológicas (umidade, pressão atmosférica, velocidade do vento, 
índice de radiação solar e temperatura)  para prever a temperatura média diária utilizando o algoritmo K-Nearest Neighbors (KNN).
"""

from sklearn.neighbors import KNeighborsRegressor
import numpy as np

#dados
dados = np.array([
    [65, 1015, 12, 300, 22],
    [70, 1012, 14, 320, 23],
    [75, 1009, 16, 340, 24],
    [80, 1007, 18, 360, 25],
    [85, 1005, 20, 380, 26],
    [90, 1003, 22, 400, 27],
    [95, 1001, 24, 420, 28]
])

x = dados[:, :-1]
y = dados[:, -1]

modelo = KNeighborsRegressor(n_neighbors=3)
modelo.fit(x, y)

novos_dados = np.array([[85, 1004, 19, 370]])
previsao_temperatura = modelo.predict(novos_dados)

print(f'A temperatura média diaria prevista: {previsao_temperatura[0]:.2f} °C')