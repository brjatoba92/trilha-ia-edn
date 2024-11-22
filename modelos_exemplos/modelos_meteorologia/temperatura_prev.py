"""
1. Previsão de Temperatura
Enunciado do Exercício:
Objetivo: Prever a temperatura futura com base em dados históricos de temperatura, umidade e pressão atmosférica.
"""

#Bibliotecas importadas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Gerar dados simulados
np.random.seed(42)
data = {
    'temperatura': np.random.uniform(15, 30, 100),
    'umidade': np.random.uniform(40, 70, 100),
    'pressao': np.random.uniform(1000, 1020, 100)
}

# Transformar em dataframe
df = pd.DataFrame(data)
df['temperature_next_day'] = df['temperatura'].shift(-1)

#remover a ultima linha com NaN
df.dropna(inplace=True)

#Salvar em arquivo csv
df.to_csv('weather_data.csv', index=False)

#Carregar dados do CSV
data = pd.read_csv('weather_data.csv')

#Dividir dados em features e target
X = data[['temperatura', 'umidade', 'pressao']]
y = data['temperature_next_day']

#Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Treinar modelo de regressão linear

model = LinearRegression()
model.fit(X_train, y_train)

#Previsao
y_pred = model.predict(X_test)

#Avaliar o desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Resultado
print(f'Erro Medio Quadratico (MSE): {mse:.2f}')
print(f'Coeficiente de determinação (r2): {r2:.2f}')

#Visualização
plt.scatter(y_test, y_pred)
plt.xlabel('Temperatura Real')
plt.ylabel('Temperatura Prevista')
plt.title('Temperatura Real vs Prevista')
plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('temperatura.png')
