"""
Utilize um conjunto de dados que contém os gastos com publicidade (em milhares de reais) 
e as vendas (em milhares de unidades) para criar um modelo de regressão linear que preveja as vendas 
com base nos gastos com publicidade.
"""

print("----------PROGRAMA PREVISÃO DOS GASTOS COM PUBLICIDADE------------------------")

from sklearn.linear_model import LinearRegression
import numpy as np

#carregando os dados
gastos_publicidade = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
vendas = np.array([15,25,35,45,55])

#treinando o modelo
modelo_agencia = LinearRegression()
modelo_agencia.fit(gastos_publicidade, vendas)

#usuario informa a venda total
quanto_gastei = float(input("Quanto foi a venda total, em milhares de reais? "))

#previsão dos gastos com publicidade
novos_gastos_com_publicidade = np.array([[quanto_gastei]])
previsão_gastos_com_publicidade = modelo_agencia.predict(novos_gastos_com_publicidade)

print(f'Para um gasto de R$ {quanto_gastei} mil será necessario investir em publidade de R$ {previsão_gastos_com_publicidade[0]:.0f} mil')