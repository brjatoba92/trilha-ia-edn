#Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#dados = pd.read_csv('2024_brasileirao.csv')

#dados simulados
np.random.seed(42)
data={
    'possession': np.random.uniform(40, 60, 100),
    'shots_on_target': np.random.uniform(5,15,100),
    'pass_accuracy': np.random.uniform(70,90,100),
    'goals': np.random.uniform(0,3,100)
}

df = pd.DataFrame(data)
df.to_csv('football_goals.csv', index=False)

#Carregar os dados
data = pd.read_csv('football_goals.csv')

#Dados Features e Target
X = data[['possession', 'shots_on_target', 'pass_accuracy']]
y = data['goals']

#Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Dividir dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

#Previsão
y_pred = model.predict(X_test)

#Avaliar o desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erro Quadraticoi Médio (MSE): {mse:.2f}')
print(f'Coeficiente de Determinação(R2): {r2:.2f}')

#Visualizando o retultado
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Gols Reais')
plt.ylabel('Gols Previstos')
plt.title('Gols Reais vs Previstos')
plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('gols_previstos.png')
plt.show()

