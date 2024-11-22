"""
Previsão de Sobrevivência em Pacientes com Câncer de Mama
Enunciado do Exercício:
Objetivo: Desenvolver um modelo de machine learning para prever a sobrevivência de pacientes com câncer de mama com base em características do tumor.

Passos:

Carregar os dados do conjunto Breast Cancer Wisconsin.

Preparar os dados e normalizar as features.

Dividir os dados em conjuntos de treinamento e teste.

Criar e treinar um modelo de SVM (Support Vector Machine) usando sklearn.

Avaliar a precisão do modelo e gerar um relatório de classificação.
"""

#Importação das bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Carregando os dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv(url, header=None, names=column_names)

#Preparação dos dados
x = data.drop('diagnosis', axis=1)
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

#Normalização dos dados
scaller = StandardScaler()
x_scalled = scaller.fit_transform(x)

#Divisião em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_scalled, y, test_size=0.2, random_state=42)

# Criação e treinamento do modelo
model = SVC(kernel='linear')
model.fit(x_train, y_train)

#Previsão
y_pred = model.predict(x_test)

#Avaliação do modelo
accuracy= accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#Resultado
print(f'Acuracia: {accuracy:.2f}')
print("Relatorio: ", report)
