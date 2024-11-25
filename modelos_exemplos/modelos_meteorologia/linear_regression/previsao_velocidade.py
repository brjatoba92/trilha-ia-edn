"""
Crie um modelo de regressão múltipla para prever a velocidade do vento diária utilizando variáveis como 
temperatura, umidade, pressão atmosférica e índice de radiação solar.
"""

from sklearn.linear_model import LinearRegression
import numpy as np

dados = np.array([
    [22, 70, 1012, 300, 12],
    [23, 75, 1010, 320, 14],
    [24, 80, 1008, 340, 16],
    [25, 85, 1005, 360, 18],
    [26, 90, 1002, 380, 20]
])

#selecao dos dados
x = dados[:, :-1] #dados de temperatura, umidade, pressao atmosferica e radiação solar 
y = dados[:, -1] #velocidade do vento

#criando e treinando o modelo
modelo = LinearRegression()
modelo.fit(x, y)

#previsao
novos_dados = np.array([[25, 87, 1006, 370]])
previsao_vento = modelo.predict(novos_dados)

#resultado
print(f'Velocidade do vento prevista: {previsao_vento[0]:.2f} m/s ')