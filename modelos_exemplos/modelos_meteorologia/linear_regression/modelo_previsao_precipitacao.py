"""
Previsão de Precipitação
Enunciado do Exercício:
Objetivo: Prever a quantidade de precipitação futura com base em dados históricos de temperatura, umidade e pressão atmosférica.
"""

#importacao das bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Geração de dados simulados
np.random.seed(42)
data = {
    'temperature': np.random.uniform(10, 35, 100),
    'humidity': np.random.uniform(30, 100, 100),
    'pressure': np.random.uniform(980, 1050, 100),
    'precipitation': np.random.uniform(0, 50, 100)
}

#Transformação em DataFrame
df=pd.DataFrame(data)

#Salva em arquivo CSV
df.to_csv('precipitacao_data.csv')

#Carregando os dados
data = pd.read_csv('precipitacao_data.csv')

#Dividir dados em features e target
X = data[['temperature', 'humidity', 'pressure']]
y = data['precipitation']

#Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Treinar modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

#Previsão
y_pred = model.predict(X_test)

#Avaliação do desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erro Quadratico Médio (MSE): {mse:.2f}')
print(f'Coeficiente de Determinação (R^2): {r2:.2f}')

"""
plt.figure(figsize=(10, 6))
for real, pred in zip(y_test, y_pred): 
    if pred > real:
        plt.scatter(real, pred, color='red')
    else:
        plt.scatter(real, pred, color='blue')
"""
#Visualização dos resultados
plt.scatter(y_test, y_pred)
plt.xlabel('Precipitação Real (mm)')
plt.ylabel('Precipitação Prevista (mm)')
plt.title('Precipitação Real vs Prevista')
plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
#plt.legend(['Previsão Maior que Real', 'Previsão Menor que Real'], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

plt.savefig('previsao_precipitacao.png')