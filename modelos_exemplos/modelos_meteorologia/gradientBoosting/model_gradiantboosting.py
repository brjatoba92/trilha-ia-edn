"""
Use algoritmos de Gradient Boosting, como XGBoost, para prever a ocorrência de tempestades. 
Esses modelos são bons para lidar com dados desbalanceados e podem capturar relações complexas.
"""
#Importação de bibliotecas
import pandas as pd #manipulação de dados
import numpy as np #array
from sklearn.model_selection import train_test_split #conjunto de dados de treino e de teste
from xgboost import XGBClassifier #Classificador de Gradiant Boosting
from sklearn.metrics import classification_report, roc_auc_score #Avaliação do modelo
import matplotlib.pyplot as plt #Gerar graficos
from sklearn.metrics import roc_curve #

#Dados simuulados
np.random.seed(42) #peditividade dos dados aleatorios
data = pd.DataFrame({ #geração de 100 dados de variaveis meteorologicas
    'Umidade': np.random.uniform(30,100,1000),
    'Pressão': np.random.uniform(350, 1050, 1000),
    'Velocidade_Do_Vento': np.random.uniform(0,40,1000),
    'Tempestade': np.random.choice([0,1], 1000, p=[0.8, 0.2]) #20% de tempestade
})

#Salvar em csv
df = pd.DataFrame(data)
df.to_csv('model_gradiantboosting.csv', index=False)

#peraparção dos dados
X = data[['Umidade', 'Pressão', 'Velocidade_Do_Vento']] #Features(Caracteristicas)
y = data[['Tempestade']] #Target (Alvo)

#Conjunto dos dados
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42) #Conjuntos de 80% de dados de treino e 20% de teste com garantia de preprodutibilidade (random_state)

#Criação e treino do modelo
model = XGBClassifier(random_state=42) #Criação do modelo
model.fit(X_train, y_train) #Treinamento do modelo

#Previsao
y_pred = model.predict(X_test) #Previsão com os dados de teste
y_pred_proba = model.predict_proba(X_test)[:, 1] #Calculo das probabilidades preditas, com a obtenção da probabilidade de classe positiva (:, -1)

#AVALIAÇÃO
print(classification_report(y_test, y_pred)) #Relatorio das metricas
roc_auc = roc_auc_score(y_test, y_pred_proba) #Capacidade de discriminação do modelo
print(f'AUC-ROC: {roc_auc:.2f}')

#Plotagem da curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba) #Taxas de falsos positivos e verdadeiros positivos
plt.plot(fpr, tpr, color='blue', label=f'XGBoost (AUC = {roc_auc:.2f})') #Plotagem da curva ROC
plt.plot([0, 1], [0, 1], color='red', linestyle='--') #Classificação aleatoria
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title('Curva ROC - Previsão de Tempestades')
plt.legend(loc='lower right')
plt.savefig('model_gradiantboosting.png') #Gera a figura
plt.show()
