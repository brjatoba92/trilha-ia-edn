from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

dados = pd.read_csv('maceio.csv')

#selecao dos dados [prec, pres, temp, umd, vel]
x = dados.iloc[:, :-1] #todos os dado, exceto a vel.vento
y = dados.iloc[:, -1] #velocidade

modelo = LinearRegression()
modelo.fit(x, y)

novos_dados = np.array([[25, 1010, 29.8, 72]])
previsao_vento = modelo.predict(novos_dados)

print(f'Velocidade do vento prevista para Maceió: {previsao_vento[0]:.2f} m/s')

