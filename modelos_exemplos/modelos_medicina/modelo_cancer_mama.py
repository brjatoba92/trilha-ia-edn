"""
2. Classificação de Câncer de Mama
Enunciado do Exercício:
Objetivo: Desenvolver um modelo de machine learning para classificar tumores de mama como malignos ou benignos com base em características observadas.

Passos:

Carregue os dados do conjunto de dados de Câncer de Mama.

Prepare os dados e normalize as features.

Divida os dados em conjuntos de treinamento e teste.

Crie e treine um modelo de floresta aleatória (Random Forest) usando sklearn.

Avalie a precisão do modelo e gere um relatório de classificação.
"""

# Importe as bibliotecas
import numpy as np #array
import pandas as pd #manipulação de dados
from sklearn.model_selection import train_test_split #separação dos dados em conjuntos de treinamento e teste
from sklearn.preprocessing import StandardScaler #normalização dos dados
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar dados
url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
data = pd.read_csv(url)

# Preparação dos dados
X = data.drop('Class', axis=1)
y = data['Class'].apply(lambda x: 1 if x == 'malignant' else 0)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criação e treinamento do modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:\n", report)
