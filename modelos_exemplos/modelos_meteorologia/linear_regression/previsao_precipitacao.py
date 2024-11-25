"""
Utilize variáveis como temperatura, umidade, pressão atmosférica e dia da semana para prever a precipitação diária. 
Crie um modelo de regressão múltipla que utiliza esses dados.
"""

from sklearn.linear_model import LinearRegression
import numpy as np

#base de dados
estacao_metorologica = np.array([
    [22, 80, 1010, 1, 5], 
    [23, 82, 1008, 2, 7],
    [24, 85, 1006, 3, 10],
    [25, 87, 1004, 4, 12],
    [26, 90, 1002, 5, 15]
])

x = estacao_metorologica[:, :-1]
y = estacao_metorologica[:, -1]

#criação e treinamento

modelo_estacao = LinearRegression()
modelo_estacao.fit(x, y)

#previsao
novos_dados = np.array([[28, 71, 1006, 9]])
previsao_precipitacao = modelo_estacao.predict(novos_dados)

print(f'Precipitação prevista: {previsao_precipitacao[0]:.2f} °C')

