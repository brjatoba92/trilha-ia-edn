"""
Modelagem de Temperatura com Random Forest
Enunciado: Treine um modelo de Random Forest para prever temperaturas futuras com base em dados históricos de temperatura, umidade e pressão atmosférica.
"""
#Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Dados fictícios como dicionário
np.random.seed(42)
data = {
    'Date': pd.date_range('20230101', periods=365),
    'Temperature': np.random.uniform(10, 35, size=(365)),
    'Humidity': np.random.uniform(30, 100, size=(365)),
    'Pressure': np.random.uniform(950, 1050, size=(365))
}

# Criação do DataFrame a partir do dicionário
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Salvar banco de dados em CSV
df.to_csv('model_temperature_randomForest.csv', index=True)

# Preparação dos dados
X = df[['Humidity', 'Pressure']]
y = df['Temperature']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Plotagem
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Temperature'], label='Dados Reais')
plt.plot(df.index[-len(y_test):], y_pred, label='Previsões Random Forest')
plt.xlabel('Data')
plt.ylabel('Temperatura (°C)')
plt.title('Modelagem de Temperatura com Random Forest')
plt.legend()
plt.savefig('model_temperature_randomForest.png')
plt.show()
