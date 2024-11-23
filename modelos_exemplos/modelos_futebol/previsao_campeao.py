#Importação das bibliotecas
import numpy as np #arrays
import pandas as pd #manipulação de arquivos
from sklearn.model_selection import train_test_split #divisão dos dados em conjuntos de treino e de teste
from sklearn.preprocessing import StandardScaler #normalização dos dados
from sklearn.ensemble import RandomForestClassifier #Algoritmo de aprendizado de maquina
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve #avaliar a performance do modelo
import matplotlib.pyplot as plt #plotar graficos

#Geração de dados simulados
np.random.seed(42) #resultados sejam reprodutiveis
data = {
    'points':np.random.uniform(30,80,100), #valores aleatórios uniformemente distribuídos para cada variável.
    'wins':np.random.uniform(8,25,100),
    'draws': np.random.uniform(5,15,100),
    'losses':np.random.uniform(5,15,100),
    'goals_for':np.random.uniform(20,70,100),
    'goals_against': np.random.uniform(10,50,100),
    'champion': np.random.choice([0,1],100, p=[0.85, 0.15]) #gera valores aleatórios binários para a coluna champion, com 15% de chances de serem campeões.
}

#Transformando em dataframe
df = pd.DataFrame(data) #transforma o dicionário de dados em um DataFrame

#Salvar
df.to_csv('previsao_campeao.csv', index=False) #salva o DataFrame como um arquivo CSV.
#Ler
data = pd.read_csv('previsao_campeao.csv') #lê o arquivo CSV em um DataFrame

#Features e Target
X = data[['points',  'wins', 'draws', 'losses', 'goals_for', 'goals_against']] #caracteristicas do modelo
y = data['champion'] #variavel alvo

#Normalização dos dados
scaler = StandardScaler() #normalizar os dados, ajustando-os para ter média 0 e variância 1
X_scaled = scaler.fit_transform(X)

#Treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #divide os dados em conjuntos de treino (80%) e teste (20%), garantindo reprodutibilidade com random_state=42

#Treinar o modelo de RandomForest
model = RandomForestClassifier(random_state=42) #Criação do modelo
model.fit(X_train, y_train) #treina o modelo com os dados de treino(80% dos dados)

#Prever dados de teste
y_pred = model.predict(X_test) #previsões sobre o conjunto de teste.
y_pred_proba = model.predict_proba(X_test)[:, 1] #obtém as probabilidades preditas para a classe positiva

#Avaliação do desempenho
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Acuracia: {accuracy:.2f}')
print(f'AUC-ROC: {roc_auc:.2f}')

#Visualizar o resultado
fpr, tpr, _ = roc_curve(y_test, y_pred_proba) #calcula as taxas de falsos positivos (FPR) e verdadeiros positivos (TPR) para diferentes limiares.
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color='blue', label=f'Random Forest (AUC = {roc_auc:.2f})') #plota a Curva ROC
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right') #legenda ao grafico
plt.savefig('previsao_campeao.png') #salva a figura
plt.show()