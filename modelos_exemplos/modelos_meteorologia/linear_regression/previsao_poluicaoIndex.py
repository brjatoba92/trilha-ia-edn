"""
Utilize variáveis como temperatura, umidade, pressão atmosférica, velocidade do vento e precipitação
para prever o índice de qualidade do ar. 
Crie um modelo de regressão múltipla que utilize esses dados.
"""

from sklearn.linear_model import LinearRegression
import numpy as np

dados = np.array([
    [22, 65, 1016, 10, 0, 50],
    [23, 68, 1014, 12, 0, 52],
    [24, 70, 1012, 14, 2, 55],
    [25, 73, 1010, 16, 3, 58],
    [26, 75, 1008, 18, 4, 60]
])

x = dados[:, : -1]
y = dados[:, -1]

modelo = LinearRegression()
modelo.fit(x, y)

novos_dados = np.array([[25, 70, 1013, 15, 1]])
previsao_iqa = modelo.predict(novos_dados)

print(f'Indice de Qualidade do Ar Previsto: {previsao_iqa[0]:.2f}')
