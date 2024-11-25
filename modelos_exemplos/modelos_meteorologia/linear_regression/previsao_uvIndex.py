from sklearn.linear_model import LinearRegression
import numpy as np

#dados [temperatura, umidade, pressao, velocidade, indice uv]

dados = np.array([
    [22, 70, 1015, 12, 3], 
    [23, 72, 1013, 14, 4],
    [24, 75, 1011, 16, 5],
    [25, 78, 1009, 18, 6], 
    [26, 80, 1007, 20, 7]
])

x = dados[:, :-1]
y = dados[:, -1]

modelo = LinearRegression()
modelo.fit(x, y)

novos_dados = np.array([[25, 77, 1010, 19]])
previsao_uv = modelo.predict(novos_dados)

print(f'Indice UV previsto: {previsao_uv[0]:.2f}')