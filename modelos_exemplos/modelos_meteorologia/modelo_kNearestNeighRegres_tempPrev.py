"""
Utilize um conjunto de dados com variáveis meteorológicas (umidade, pressão atmosférica, velocidade do vento, 
índice de radiação solar e temperatura)  para prever a temperatura média diária utilizando o algoritmo K-Nearest Neighbors (KNN).
"""
#bibliotecas utilizadas
from sklearn.neighbors import KNeighborsRegressor #Biblioteca para criar modelo de regressão KNN
import numpy as np #utilizar arrays

#dados [umidade, pressão atmosferica, velocidade do vento, indice de radiação solar, temperatura]
dados = np.array([
    [65, 1015, 12, 300, 22],
    [70, 1012, 14, 320, 23],
    [75, 1009, 16, 340, 24],
    [80, 1007, 18, 360, 25],
    [85, 1005, 20, 380, 26],
    [90, 1003, 22, 400, 27],
    [95, 1001, 24, 420, 28]
])

#selecionando os dados 
x = dados[:, :-1] #umidade, pressão, vel_vento, rad.solar
y = dados[:, -1] #temperatura (alvo)

#criando e treinando o modelo
modelo = KNeighborsRegressor(n_neighbors=3) # Criado modelo de regressão K-Nearest Neighbors com 3 vizinhos.
modelo.fit(x, y) #treinando o modelo com as caracteristicas x e y

#inserindo novos dados e a previsão
novos_dados = np.array([[85, 1004, 19, 370]]) #novo array de dados
previsao_temperatura = modelo.predict(novos_dados) #previsão da temperatura com base no novo array

#resultado
print(f'A temperatura média diaria prevista: {previsao_temperatura[0]:.2f} °C')