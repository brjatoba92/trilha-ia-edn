"""
Previsão de Probabilidade de Gol
Enunciado: Prever a probabilidade de um chute resultar em gol com base na posição do chute e no ângulo em relação ao gol.
"""
#Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Dados não reais
data = pd.DataFrame({
    'distancia': np.random.rand(100) * 30, #distancia do chute ao gol
    'angulo': np.random.rand(100)*90, #angulo de chute em relação ao gol
    'gol': np.random.choice([0,1], 100) #1-gol e 0-fora 
})

data.to_csv('dados_prob_gols.csv', index=False)

X = data[['distancia', 'angulo']]
y = data['gol']

#Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modelo
model = LogisticRegression()
model.fit(X_train, y_train)

#Previsão
y_pred = model.predict(X_test)

#Acuracia
print(f'Acuracia: {accuracy_score(y_test, y_pred)}')

#Plotagem
plt.scatter(data['distancia'], data['angulo'], c=data['gol'], cmap='bwr', alpha=0.5)
plt.xlabel('Distância do Chute')
plt.ylabel('Ângulo do Chute')
plt.title('Probabilidade de Gol')
plt.colorbar(label='Gol')
plt.savefig('prob_gols.png')
plt.show()

