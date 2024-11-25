"""
2. Classificação de Fenômenos Climáticos
Enunciado do Exercício:
Objetivo: Classificar fenômenos climáticos (chuva, neve, sol, etc.) com base em dados de temperatura, umidade e velocidade do vento.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Gerar dados simulados
np.random.seed(42)
data = {
    'temperature': np.random.uniform(-10, 35, 100),
    'humidity': np.random.uniform(20, 100, 100),
    'wind_speed': np.random.uniform(0, 20, 100),
    'phenomenon': np.random.choice(['sunny', 'rain', 'snow'], 100)
}
# Transformar em DataFrame
df = pd.DataFrame(data)

# Salvar em arquivo CSV
df.to_csv('climate_data.csv', index=False)

# Carregar dados do CSV
data = pd.read_csv('climate_data.csv')

# Dividir dados em features e target
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['phenomenon']

# Transformação de variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prever dados de teste
y_pred = model.predict(X_test)

# Avaliar desempenho
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:\n", report)

# Visualizar árvore de decisão
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
plt.title('Árvore de Decisão para Classificação de Fenômenos Climáticos')
plt.show()

plt.savefig('arvore_clima.png')