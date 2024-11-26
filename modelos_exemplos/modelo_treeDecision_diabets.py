"""
Diagnóstico de Diabetes
Enunciado: Utilizar dados de pacientes para prever se eles têm diabetes ou não.

Falta plotar um grafico
"""

#Importando
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Geração dos dados simulados
np.random.seed(42)
tamanho_dos_dados = 1000
data = pd.DataFrame({
    'idade': np.random.randint(18, 80, tamanho_dos_dados),
    'imc': np.random.uniform(18, 40, tamanho_dos_dados),
    'glicose': np.random.uniform(70, 200, tamanho_dos_dados),
    'pressao': np.random.uniform(60, 120, tamanho_dos_dados),
    'diabetes': np.random.randint(0, 2, tamanho_dos_dados)
})

#Salvar em csv
data.to_csv('modelo_treeDecision_diabets.csv', index=False)

X = data[['idade', 'imc', 'glicose', 'pressao']]
y = data['diabetes']

#Dados divididos em conjuntos d etreino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Criar modelo de arvore de decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Previsão
y_pred = model.predict(X_test)

#Avaliação do desempenho
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracia: {accuracy:.2f}')

# Calcular a matriz de confusão
#cm = confusion_matrix(y_test,y_pred)

#