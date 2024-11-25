"""
Utilize um conjunto de dados com múltiplas variáveis meteorológicas (temperatura, umidade, pressão atmosférica, 
velocidade do vento, precipitação) para prever eventos extremos de temperatura. 
Aplique regressão de Lasso para selecionar as características mais importantes.
"""

#Bibliotecas utilizadas
from sklearn.linear_model import LassoCV #Regressão de Lasso para validação cruzada
import numpy as np #Arrays

#Dados [temperatura, umidade, pressão atm, velocidade do vento, precipitação, eventos extremos]
dados = np.array([
    [22, 65, 1015, 12, 5, 1],
    [23, 70, 1012, 14, 7, 0],
    [24, 75, 1009, 16, 10, 1],
    [25, 80, 1007, 18, 12, 0], 
    [26, 85, 1005, 20, 15, 1],
    [27, 90, 1003, 22, 17, 0],
    [28, 95, 1001, 24, 20, 1]
])

#Seleção de dados
x = dados[:, :-1] #temp, umd, pres, vel.vento, precip (seleção de todas as colunas, exceto a ultima)
y = dados[:, -1] #eventos extremos (alvo) (seleção da ultima coluna)

#Criação e treinamento do modelo
modelo = LassoCV(cv=5) #Modelo criado (Cria um modelo de regressão Lasso com validação cruzada (5-fold cross-validation)
modelo.fit(x, y) #Modelo treinando com as caracteristicas x e y

#Inserindo novos dados e previsão
novos_dados = np.array([[25, 82, 1006, 19, 8]]) #Array novo (temp, umd, pres, vel.vento e precip)
previsao_evento_extremo = modelo.predict(novos_dados) #Previsão da probabilidade de eventos extremos com base nos novos dados

#Resultado
print(f'Probablidade de evento extremos prevista: {previsao_evento_extremo[0]:.2f}')