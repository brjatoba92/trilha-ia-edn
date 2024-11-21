"""
Modelo 1 - Diagnostico de Diabetes
Desenvolva um modelo de ML para realizar a previsão se um paciente tem diabetes com base nas medidas de saude fornecidas

- Carregue os dados do conjunto Pima Indians Diabetes.
- Prepare os dados e normalize as features.
- Divida os dados em conjuntos de treinamento e teste.
-Crie e treine um modelo de regressão logística usando sklearn.
- Avalie a precisão do modelo e gere um relatório de classificação.
"""

"""
Pregnancies: Número de vezes que a paciente ficou grávida.

Glucose: Nível de glicose no sangue após 2 horas em um teste de tolerância à glicose oral.

BloodPressure: Pressão arterial diastólica medida em mm Hg.

SkinThickness: Espessura da dobra cutânea medida em milímetros.

Insulin: Nível de insulina no sangue medido em U/mL após 2 horas.

BMI: Índice de massa corporal (BMI), que é calculado como peso em kg dividido pela altura em metros ao quadrado.

DiabetesPedigreeFunction: Função de pedigree da diabetes, que indica a probabilidade de diabetes com base no histórico familiar.

Age: Idade do paciente medida em anos.

Outcome: Resultado binário onde 1 indica que a paciente tem diabetes e 0 indica que não tem diabetes.
"""

#Importação das bibliotecas
import numpy as np #array
import pandas as pd #manipulação de dados
from sklearn.model_selection import train_test_split #divide os dados em conjuntos de treinamento e teste
from sklearn.preprocessing import StandardScaler #
from sklearn.linear_model import LogisticRegression #criar modelo de regressão logistica
from sklearn.metrics import accuracy_score, classification_report #

#Load dos dados
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" #importação do arquivo csv a partir de uma url

column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'] #seleção das colunas
data = pd.read_csv(url, header=None, names=column_names)

#preparação dos dados
x = data.drop('Outcome', axis=1)
y = data['Outcome']

#normalização dos dados
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Divisão dos dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#Criação e treino do modelo
model = LogisticRegression()
model.fit(x_train, y_train)

#previsão
y_pred = model.predict(x_test)

#Avaliação da precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#Gere o relatorio
print(f'Acuracia: {accuracy:.2f}')
print('Relatorio de Classificação \n', report)
