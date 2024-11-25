"""
Crie um modelo de regressão polinomial para prever o índice de radiação solar diária utilizando variáveis 
como temperatura, umidade, pressão atmosférica e velocidade do vento.
"""
#Importando as bibliotecas necessarias
from sklearn.preprocessing import PolynomialFeatures #Gerando caracteristicas polinomiais a partir das caracteristicas originais
from sklearn.pipeline import make_pipeline #Criação de pipeline do modelo
from sklearn.linear_model import LinearRegression #Criação de modelo de regressão linear
import numpy as np #Arrays

#dados [temperatura, umidade, pressão atmosferica, velocidade do vento e indice de radiação solar]
dados = np.array([
    [22, 65, 1015, 12, 300], 
    [23, 70, 1012, 14, 320],
    [24, 75, 1009, 16, 340], 
    [25, 80, 1007, 18, 360],
    [26, 85, 1005, 20, 380],
    [27, 90, 1003, 22, 400],
    [28, 95, 1001, 24, 420]
])

#selecionando os dados
x = dados[:, :-1] #temp, umd, pres_atm, vel_vento 
y = dados[:, -1] #rad_solar

#criação e treino do modelo
modelo = make_pipeline (PolynomialFeatures(degree=2), LinearRegression()) #transformando as caracteristicas polinomiais de grau 2 e depois aplica a regressão linear
modelo.fit(x, y) #treino do modelo com base nas caracteristicas x e y

#novos dados e previsão da radiação
novos_dados = np.array([[25, 82, 1006, 19]]) #novo array de dados
previsao_radiacao = modelo.predict(novos_dados) #previsão

#resultado
print(f'Radiação solar prevista: {previsao_radiacao[0]:.2f} Wm2')