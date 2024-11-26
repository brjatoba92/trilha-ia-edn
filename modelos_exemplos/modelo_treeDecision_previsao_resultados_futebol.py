"""
Previsão de Resultado de Partidas de Futebol (Vitória/Empate/Derrota)
Previsão do resultado de um time em partidas de futebol com base em estatísticas pré-jogo.
"""
#Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Somulando os dados

data = pd.DataFrame({
    'posse_bola': np.random.randint(30, 70, 500),
    'chutes_gols': np.random.randint(1, 20, 500),
    'faltas': np.random.randint(5, 20, 500),
    'escanteios': np.random.randint(0, 10, 500),
    'resultado': np.random.choice(['Vitoria', 'Empate', 'Derrota'], 500)
})

#Salvar em csv
data.to_csv('modelo_treeDecision_previsao_resultados_futebol.csv', index=False)

#Ler o csv
data = pd.read_csv('modelo_treeDecision_previsao_resultados_futebol.csv')

#Variaveis independentes e alvo
X = data[['posse_bola', 'chutes_gols', 'faltas', 'escanteios']]
y = data['resultado']

#Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modelo
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

#Previsões
y_pred = model.predict(X_test)

#Relatorio
report = classification_report(y_test, y_pred)
print(f'Relatorio de Classificação: \n {report}')

#Resultado
plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.savefig('modelo_treeDecision_previsao_resultados_futebol.png')
plt.show()