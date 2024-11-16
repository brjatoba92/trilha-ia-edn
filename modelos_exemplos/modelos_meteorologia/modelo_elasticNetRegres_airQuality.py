"""
Utilize variáveis meteorológicas (temperatura, umidade, pressão atmosférica, velocidade do vento, precipitação) 
para prever o índice de qualidade do ar. 
Aplique regressão de Elastic Net para equilibrar a seleção de características e a regularização.
"""

#bibliotecas utilizadas
from sklearn.linear_model import ElasticNetCV #Regressão Elastic Net
import numpy as np #array

#dados [temperatura, umidade, pressão atmosferica, velocidade do vento, precipitação, qualidade do ar]
dados = np.array([
    [22, 65, 1015, 12, 5, 50],
    [23, 70, 1012, 14, 7, 52],
    [24, 75, 1009, 16, 10, 55],
    [25, 80, 1007, 18, 12, 58],
    [26, 85, 1005, 20, 15, 60],
    [27, 90, 1003, 22, 17, 62],
    [28, 95, 1001, 24, 20, 65]
])

#seleção dos dados
x = dados[:, :-1] #temp, umd, pressao atm, vel_vento, precip
y = dados[:, -1] #qual_ar

#Criação e treino do modelo
modelo = ElasticNetCV(cv=5) #modelo criado - modelo de regressão elastic net com validação cruzada
modelo.fit(x, y) #treino do modelo com as caracteristicas x e y

#novos dados e previsão
novos_dados = np.array([[25, 82, 1006, 19, 8]]) #array de novos dados
previsao_qualidade_do_ar = modelo.predict(novos_dados) #previsão da qualidade do ar com base no novo array de dados

#resultado
print(f'Qualidade do ar prevista: {previsao_qualidade_do_ar[0]:.2f}')