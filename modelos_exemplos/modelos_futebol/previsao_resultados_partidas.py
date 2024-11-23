"""
Previsão de Resultados das Partidas
Enunciado: Prever o resultado de uma partida de futebol (vitória, empate ou derrota) com base em características históricas dos times, como posse de bola, chutes a gol e passes certos.
"""

#Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Dados simulados
data = pd.DataFrame({
    'posse_bola': np.random.randint(30, 70, 100),
    'chutes_a_gol': np.random.randint(1, 20, 100),
    'passes_certos': np.random.randint(300, 700, 100),
    'resultado': np.random.choice(['Vitoria', 'Empate', 'Derrota'], 100) 
})

# Salvar dados em CSV
data.to_csv('dados_futebol.csv', index=False)

#Pre processamento
X = data[['posse_bola', 'chutes_a_gol', 'passes_certos']]
y = data['resultado']

#Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Previsão
y_pred = model.predict(X_test)

#Relatorio
print(classification_report(y_test, y_pred))

#Plotagem
importances = model.feature_importances_
features = X.columns
plt.barh(features, importances)
plt.xlabel('Importancia')
plt.title('Importancia das Variaveis')
#plt.savefig('resultados_partidas.png')
plt.show()