"""
Detecção de Diabetes com Análise de Componentes Principais

Enunciado do Exercício:
Objetivo: Desenvolver um modelo de machine learning para detectar diabetes usando Análise de Componentes Principais (PCA) para redução de dimensionalidade.

Passos:

Carregar os dados do conjunto Pima Indians Diabetes.

Preparar os dados e aplicar PCA para redução de dimensionalidade.

Dividir os dados em conjuntos de treinamento e teste.

Criar e treinar um modelo de Regressão Logística usando sklearn.

Avaliar a precisão do modelo e gerar um relatório de classificação.
"""

#Importando as bibliotecas
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Carregando os dados
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'] 
data = pd.read_csv(url, header=None, names=column_names)

#Preparação dos dados
x = data.drop('Outcome', axis=1)
y = data['Outcome']

#Normalização dos dados
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Aplicação do PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

#Divisão em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

#Criação e treinamento do modelo
model = LogisticRegression()
model.fit(x_train, y_train)

#Previsão
y_pred = model.predict(x_test)

#Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#Resultado
print(f'Acuracia: {accuracy:.2f}')
print("Relatorio: ", report)
