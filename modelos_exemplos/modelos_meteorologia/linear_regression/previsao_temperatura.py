"""
Crie um modelo de regressão múltipla para prever a temperatura média diária considerando várias variáveis 
como umidade, pressão atmosférica e velocidade do vento.
"""

from sklearn.linear_model import LinearRegression
import numpy as np

#base de dados [umidade, pressao atmosferica, velocidade do vento, temperatura]
base_meteorologica_mcz = np.array([[65, 1015, 12, 22], # Dia 1 
                        [70, 1012, 14, 23], # Dia 2
                        [75, 1009, 16, 24], # Dia 3
                        [80, 1007, 18, 25], # Dia 4
                        [85, 1005, 20, 26], # Dia 5
                        [90, 1003, 22, 27], # Dia 6 
                        [95, 1001, 24, 28] # Dia 7]
])

#seleção dos dados
x = base_meteorologica_mcz[:, :-1] #dados de umidade, pressão e vel.vento
y = base_meteorologica_mcz[:, -1] #dados de temperatura

#criação e treinamento do modelo
modelo_meteorologia = LinearRegression() #criando
modelo_meteorologia.fit(x, y) #treino

#novos dados
novos_dados = np.array([[78, 1010, 15]])
previsao_temperatura = modelo_meteorologia.predict(novos_dados)

#resultado
print(f'Temperatura media prevista é igual a {previsao_temperatura[0]:.2f} °C')



