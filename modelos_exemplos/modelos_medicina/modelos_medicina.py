import numpy as np #arrays
import pandas as pd #manipulação de dados
import matplotlib.pyplot as plt #graficos
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression #Modelo de Regressão Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #Classificadores
from sklearn.svm import SVC #uport Vector Machine
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

#Função para gerar dados simulados de saude
def gerar_dados_saude():
    """
    programadores e cientistas de dados adotaram o número 42 como uma escolha divertida e simbólica ao definir seeds para geradores de números aleatórios.
    """
    np.random.seed(42) #Certifica que será possivel gerar a mesmo array de numeros aleatorios (42,51,1202,...)
    
    ##Simulando dados de diagnostico de doenças
    n_samples = 1000 #Numeros de amostras
    
    #Features utilçizando a distribuição normal dos dados
    """
    Distribuição Normal (50,15,n_samples)
    50 - Media da Distribuição: idades serão em média 50 anos
    15 - Desvio Padrão: Idades geradas terão uma variação tipica de 15 amos em torno da média
    n_samples: numeros de amostras (1000)
    """
    idade = np.random.normal(50,15,n_samples)
    pressao_sistemtica = np.random.normal(120,20,n_samples)
    colesterol = np.random.normal(200,40,n_samples)
    glicemia = np.random.normal(100,25, n_samples)
    imc = np.random.normal(25,5,n_samples)
    historico_familiar = np.random.choice([0,1], n_samples, p=[.7,.3]) #70% de ser 0 (negativo) e 30% de 1(positivo) de haver uma doença com historico familiar

    #Targets - risco de doença cardiovascular 0 - baixo e 1 - alto
    risco = (
        (idade>55)+
        (pressao_sistemtica>140)+
        (colesterol>240)+
        (glicemia>125)+
        (imc > 30)+
        historico_familiar
    )>=3

    #Dataframe dos dados na forma de um dicionario
    dados = pd.DataFrame({
        'idade': idade,
        'pressao_sistematica': pressao_sistemtica,
        'colesterol': colesterol,
        'glicemia': glicemia,
        'imc': imc,
        'historico_familiar': historico_familiar,
        'risco_cardiovascular': risco.astype(int)
    })
    return dados #returna o dataframe

#Chamando a função
dados = gerar_dados_saude()
#Retirando da features a coluna risco cardiovascular
X = dados.drop('risco_cardiovascular', axis=1) #axis=1(operação para remover a coluna), 0 (linha)
#Target é a coluna risco_cardiovascular
y = dados['risco_cardiovascular']

#Divisao em conjuntos de dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% dos dados para teste e assegurar que essas condições vão se reeplicar(random_state=42)

#Escalonamento dos dados
"""
processo de ajustar os valores de dados para que eles tenham uma média de 0 e um desvio padrão de 1.
"""
scaler = StandardScaler() #chamando a função
X_train_scaled = scaler.fit_transform(X_train) #escalonar os dados de treino da feature
X_test_scaled = scaler.fit_transform(X_test) #escalonar os dados de teste da feature

## Modelos

# Modelo 1 - Regressão Logistica
def modelo_regressao_logistica():
    print("\n---Regressão Logistica ---")
    clf = LogisticRegression(max_iter=1000) #Modelo de Regressão Logistica
    clf.fit(X_train_scaled, y_train) #treinando o modelo
    #Previsão
    y_pred = clf.predict(X_test_scaled)
    print("Acuracia: ", accuracy_score(y_test, y_pred))
    print("\nRelatorio de Classificação:\n", classification_report(y_test, y_pred))
    
    return clf

#MNodelo 2 - Random Forest
def modelo_random_forest():
    print("\n---Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42) #Modelo de Regressão Logistica
    clf.fit(X_train_scaled, y_train) #treinando o modelo
    
    #Previsão
    y_pred = clf.predict(X_test_scaled)
    print("Acuracia: ", accuracy_score(y_test, y_pred))
    print("\nRelatorio de Classificação:\n", classification_report(y_test, y_pred))
    
    #Feature importance
    importancia = pd.DataFrame({
        'feature': X.columns,
        'importancia': clf.feature_importances_
    }).sort_values('importancia', ascending=False)

    #Plotar grafico
    plt.figure(figsize=(10,6))
    plt.bar(importancia['feature'], importancia['importancia'])
    plt.title('Importancia das Features - Random Forest')
    plt.xlabel('Features')
    plt.ylabel('Importancia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return clf


#Modelo 3 - Gradiant Boosting
def modelo_gradiant_boosting():
    print("\n - Gradient Boosting ---")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print("Acuracia: ", accuracy_score(y_test, y_pred))
    print("\nRelatorio de Classificação:\n", classification_report(y_test, y_pred))


#4-Suport Vector Machine
def modelo_svm():
    pass
#5 Rede Naural Multilayer Perceptron -
def modelo_mlp():
    pass

#Executando todos os modelos
def main():
    modelos = [
        modelo_regressao_logistica,
        modelo_random_forest,
        modelo_gradiant_boosting,
        modelo_svm,
        modelo_mlp
    ]
    for modelo in modelos:
        modelo()

if __name__ == "__main__":
    main()