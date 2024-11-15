"""
Crie um modelo de regressão linear para prever o preço de casas com base na área da casa (em metros quadrados).
Utilize um conjunto de dados simples e faça a previsão para uma nova casa.
"""

print("----------PREVISÃO VALOR DE UM IMOVEL----------------------------")

from sklearn.linear_model import LinearRegression
import numpy as np

# carregando os dados
areas = np.array([110, 120, 130, 140, 150]).reshape(-1,1)
precos = np.array([1870000, 2160000, 1950000, 2240000, 3000000])

#treinando o modelo

modelo_imobiliaria = LinearRegression()
modelo_imobiliaria.fit(areas, precos)

#input do usuario
comprador = float(input("Informe a metragem do seu imovel: "))

# previsão do valor do imovel mediante a area informada pelo usuario
nova_area = np.array([[comprador]])
previsao_imovel = modelo_imobiliaria.predict(nova_area)

print(f'Para um imovel de {comprador} m2, o preço estimado esta avaliado em R$ {previsao_imovel[0]:.2f}')