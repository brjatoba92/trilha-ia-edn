from sklearn.linear_model import LinearRegression

import numpy as np

#preparação dos dados
horas_estudo = np.array([1,2,3,4,5]).reshape(-1, 1) #valor de referencia
notas = np.array([40,50,60,70,80])

#modelo treina
modelo = LinearRegression()
modelo.fit(horas_estudo, notas)

#pergunto ao usuario final
horas = float(input("Digite o numero de horas estudadas: "))

#nota prevista
nota_prevista = modelo.predict(np.array([[horas]]))

print(f'Com {horas} de estudo, a nota prevsta é {nota_prevista[0]:.2f}')