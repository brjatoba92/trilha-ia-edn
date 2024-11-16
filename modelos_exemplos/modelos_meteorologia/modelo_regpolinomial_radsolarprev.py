"""
Crie um modelo de regressão polinomial para prever o índice de radiação solar diária utilizando variáveis 
como temperatura, umidade, pressão atmosférica e velocidade do vento.
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

#dados
dados = np.array([
    [22, 65, 1015, 12, 300], 
    [23, 70, 1012, 14, 320],
    [24, 75, 1009, 16, 340], 
    [25, 80, 1007, 18, 360],
    [26, 85, 1005, 20, 380],
    [27, 90, 1003, 22, 400],
    [28, 95, 1001, 24, 420]
])

x = dados[:, :-1]
y = dados[:, -1]

modelo = make_pipeline (PolynomialFeatures(degree=2), LinearRegression())
modelo.fit(x, y)

novos_dados = np.array([[25, 82, 1006, 19]])
previsao_radiacao = modelo.predict(novos_dados)

print(f'Radiação solar prevista: {previsao_radiacao[0]:.2f} Wm2')