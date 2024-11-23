"""
Exemplo: Previsão da Velocidade do Vento
Enunciado do Exercício:
Objetivo: Desenvolver um modelo de machine learning para prever a velocidade do vento futura com base em dados históricos de temperatura, umidade e pressão atmosférica.

Passos:

Carregar os dados do conjunto meteorológico.

Preparar os dados e normalizar as features.

Dividir os dados em conjuntos de treinamento e teste.

Criar e treinar um modelo de Regressão Linear usando sklearn.

Avaliar o desempenho do modelo usando métricas de regressão.

Visualizar os resultados com gráficos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Gerar dados simuiados
np.random.seed(42)
data = {
    'temperature': np.random.uniform(10, 35, 100),
    'humidity': np.random.uniform(30, 100, 100),
    'pressure': np.random.uniform(980, 1050, 100),
    'wind_speed': np.random.uniform(0, 20, 100)
}

#Transformar em DataFrame
df = pd.DataFrame(data)

#Salvar em csv
df.to_csv('wind_speed_data.csv', index=False)

#Carrewgar dados do csv
data = pd.read_csv('wind_speed_data.csv')

#Features e Target
X = data[['temperature', 'humidity', 'pressure']]
y = data['wind_speed']

#Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

#Prever dados de teste
y_pred = model.predict(X_test)

#Avaliação do desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erro Quadratico Médio (MSE): {mse:.2f}')
print(f'Coeficiente de Detereminação {r2:.2f}')

"""
plt.figure(figsize=(10, 6))
for real, pred in zip(y_test, y_pred): 
    if pred > real:
        plt.scatter(real, pred, color='red')
    else:
        plt.scatter(real, pred, color='blue')
"""

#Visualização dos resultados
#plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Velocidade do Vento Real (m/s)')
plt.ylabel('Velocidad do Vneto Prevista (m/s)')
plt.title('Velocidade do Vneto Real vs Prevista')
plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
#plt.legend(['Previsão Maior que Real', 'Previsão Menor que Real'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.savefig('wind_speed_prediction.png')
plt.show()