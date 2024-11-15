"""
Crie um modelo para prever o consumo de energia (em kWh) com base na temperatura média diária (em graus Celsius). 
Utilize dados históricos de temperatura e consumo.
"""

print("--------PROGRAMA PREVISÃO GASTO DE ENERGIA-----------------")

from sklearn.linear_model import LinearRegression
import numpy as np

#carregando os dados
temperaturas = np.array([22,24,26,28,30,29,23]).reshape(-1,1)
consumo_energia = np.array([15, 16, 18, 20, 25, 22, 17])

#treinando o modelo
modelo_energia = LinearRegression()
modelo_energia.fit(temperaturas, consumo_energia)

#Input do usuario que informa uma temperatura
usuario = float(input("Informe uma temperatura: "))

#previsão de uma nova temperatura
nova_temperatura = np.array([[usuario]])
previsao_consumo = modelo_energia.predict(nova_temperatura)

print(f'Para uma temperatura diaria de {usuario} °C haverá um consumo de {previsao_consumo[0]:.2f} kwh')