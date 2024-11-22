"""
Classificação de Doenças Cardíacas
Enunciado do Exercício:
Objetivo: Desenvolver um modelo de machine learning para classificar se um paciente tem uma doença cardíaca com base em diversas medidas clínicas.

Passos:
- Carregar os dados do conjunto Heart Disease.
- Preparar os dados e normalizar as features.
- Dividir os dados em conjuntos de treinamento e teste.
- Criar e treinar um modelo de K-Nearest Neighbors (KNN) usando sklearn.
- Avaliar a precisão do modelo e gerar um relatório de classificação.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, header=None, names=column_names)



#Substituição de ? por NaN e remover linhas com valores faltantes
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

#Preparação dos dados
x = data.drop('target', axis=1)
y = data['target'].apply(lambda x: 1 if x>0 else 0)

#Normalização dos dados
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Divisão em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#Criação e treinamento do modelo
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

#Previsão
y_pred = model.predict(x_test)

#Avaliação do modelo
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test, y_pred)

#Resultado
print(f'Acuracia: {accuracy:.2f}')
print("Relatorio de Classificação: \n", report)
