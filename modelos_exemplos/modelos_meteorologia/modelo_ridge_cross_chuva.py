"""
Utilize um conjunto de dados com variáveis meteorológicas 
(temperatura, umidade, pressão atmosférica, velocidade do vento) para prever a quantidade de chuva em milímetros. 
Aplique regressão de Ridge com validação cruzada para selecionar o melhor parâmetro de regularização.

"""

from sklearn.linear_model import RidgeCV #Utilizando a regressão Ridge com validação cruzada para prever a quantidade de chuva com base em variáveis meteorológicas
import numpy as np

#dados [temperatura, umidade, pressão, velocidade do vento, precipitação]
dados = np.array([[22, 65, 1015, 12, 5], 
                   [23, 70, 1012, 14, 7],
                   [24, 75, 1009, 16, 10],
                   [25, 80, 1007, 18, 12],
                   [26, 85, 1005, 20, 15],
                   [27, 90, 1003, 22, 17],
                   [28, 95, 1001, 24, 20]
])

#seleção dos dados
x = dados[:, :-1] #temp, umd, pressão, vel.vento
y = dados[:, -1] #precipitação

#criação e treinamento
alphas = np.logspace(-6, 6, 13) #Cria uma sequência de valores de alpha (parâmetro de regularização) em uma escala logarítmica 10^-6 a 10^6
modelo = RidgeCV(alphas=alphas, cv=5) #Cria um modelo de regressão Ridge com validação cruzada (5-fold cross-validation) para selecionar o melhor valor de alpha.
modelo.fit(x, y) #treinamento com os dados x e y

#inclusão de novos dados e previsão
novos_dados = np.array([[25, 82, 1006, 19]]) #inclusão de novos dados
previsao_chuva = modelo.predict(novos_dados) #previsão

#resultado
print(f'Chuva prevista: {previsao_chuva[0]:.2f} mm') #precipitação com dois pontos flutuates (2 casas decimais)