#Importação das bibliotecas necessarias
import numpy as np #arrays
import pandas as pd #manipulação de dados
import matplotlib.pyplot as plt #graficos
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression #Modelo de Regressão Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #Classificadores
from sklearn.svm import SVC #uport Vector Machine
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def criar_diretorio(diretorio):
    if not os.path.exists(diretorio): #Checa se não existe o diretorio
        os.makedirs(diretorio) #caso positivo, cria

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
    pressao_sistemica = np.random.normal(120,20,n_samples)
    colesterol = np.random.normal(200,40,n_samples)
    glicemia = np.random.normal(100,25, n_samples)
    imc = np.random.normal(25,5,n_samples)
    historico_familiar = np.random.choice([0,1], n_samples, p=[0.7,0.3]) #70% de ser 0 (negativo) e 30% de 1(positivo) de haver uma doença com historico familiar

    #Targets - risco de doença cardiovascular 0 - baixo e 1 - alto
    risco = (
        (idade>55).astype(int)+
        (pressao_sistemica>140).astype(int)+
        (colesterol>240).astype(int)+
        (glicemia>125).astype(int)+
        (imc > 30).astype(int)+
        historico_familiar
    )>=2

    # Garantir que risco seja binário
    risco = (risco > 0).astype(int)

    #Dataframe dos dados na forma de um dicionario
    dados = pd.DataFrame({
        'idade': idade,
        'pressao_sistemica': pressao_sistemica,
        'colesterol': colesterol,
        'glicemia': glicemia,
        'imc': imc,
        'historico_familiar': historico_familiar,
        'risco_cardiovascular': risco.astype(int)
    })
    return dados #returna o dataframe

#Preparação dos dados
dados = gerar_dados_saude() #chamando a função
X = dados.drop('risco_cardiovascular', axis=1) #Retirando da features a coluna risco cardiovascular, axis=1(operação para remover a coluna), 0 (linha)
y = dados['risco_cardiovascular'] ##Target é a coluna-alvo risco_cardiovascular

# Verificação de tipos
print("Tipo de X:", type(X))
print("Tipo de y:", type(y))
print("Valores únicos em y:", np.unique(y))

#Divisao em conjuntos de dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #20% dos dados para teste e assegurar que essas condições vão se reeplicar(random_state=42)

#Escalonamento dos dados
"""
processo de ajustar os valores de dados para que eles tenham uma média de 0 e um desvio padrão de 1.
"""
scaler = StandardScaler() #chamando a função
X_train_scaled = scaler.fit_transform(X_train) #escalonar os dados de treino da feature
X_test_scaled = scaler.transform(X_test) #escalonar os dados de teste da feature

#Criar diretorio para resultados
criar_diretorio('resultados_modelos')

## Modelos

# Modelo 1 - Regressão Logistica
"""
Modelo linear para classificação binária
Útil para entender o risco cardiovascular
Interpreta probabilidades de risco
"""
def modelo_regressao_logistica():
    print("\n---Regressão Logistica ---")
    clf = LogisticRegression(max_iter=1000) #Modelo de Regressão Logistica
    clf.fit(X_train_scaled, y_train) #treinando o modelo
    #Previsão
    y_pred = clf.predict(X_test_scaled)
    #Criando variaveis para a acuracia e relatorio e imprimindo
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)
    print("Acuracia: ", acuracia)
    print("\nRelatorio de Classificação:\n", relatorio)
    
    #Salvar em arquivos
    with open('resultados_modelos/regressao_logistica.txt','w') as f:
        f.write(f'Modelo: Regressão Logistica\n')
        f.write(f'Acuracia: {acuracia}\n\n')
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)
    
    return clf

#MNodelo 2 - Random Forest
"""
Algoritmo de ensemble que cria múltiplas árvores de decisão
Identifica quais features são mais importantes para o diagnóstico
Robusto contra overfitting
"""
def modelo_random_forest():
    print("\n---Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42) #Modelo de Regressão Logistica
    clf.fit(X_train_scaled, y_train) #treinando o modelo
    
    #Previsão
    y_pred = clf.predict(X_test_scaled) #Realizando a previsão com os dados escalonados de features 
    
    #Criando variaveis para a acuracia e relatorio e imprimindo
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)
    print("Acuracia: ", acuracia)
    print("\nRelatorio de Classificação:\n", relatorio)
    
    #Salvar em arquivos
    with open('resultados_modelos/random_forest.txt','w') as f:
        f.write(f'Modelo: Random Forest\n')
        f.write(f'Acuracia: {acuracia}\n\n')
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)
    
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
"""
Construção sequencial de modelos para melhorar precisão
Eficiente para problemas complexos de saúde
Reduz erros gradualmente
"""
def modelo_gradiant_boosting():
    print("\n - Gradient Boosting ---")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    #Previsão
    y_pred = clf.predict(X_test_scaled)
    
    #Criando variaveis para a acuracia e relatorio e imprimindo
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)
    print("Acuracia: ", acuracia)
    print("\nRelatorio de Classificação:\n", relatorio)
    
    #Salvar em arquivos
    with open('resultados_modelos/gradient_boosting.txt','w') as f:
        f.write(f'Modelo: Gradient Boosting\n')
        f.write(f'Acuracia: {acuracia}\n\n')
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)


#4-Suport Vector Machine
"""
Encontra o melhor hiperplano para separação
Usa kernel RBF para capturar relações não-lineares
Inclui análise da Curva ROC para avaliação de desempenho
"""
def modelo_svm():
    #Criação e treino do modelo
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
   #Previsão
    y_pred = clf.predict(X_test_scaled)
    
    #Criando variaveis para a acuracia e relatorio e imprimindo
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)
    print("Acuracia: ", acuracia)
    print("\nRelatorio de Classificação:\n", relatorio)
    
    #Salvar em arquivos
    with open('resultados_modelos/svm.txt','w') as f:
        f.write(f'Modelo: Suport Vector Machine\n')
        f.write(f'Acuracia: {acuracia}\n\n')
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)

    #Curva ROC
    y_pred_proba = clf.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC - {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    return clf

#5 Rede Naural Multilayer Perceptron -
"""
Rede neural com camadas ocultas
Capaz de aprender padrões complexos
Flexível para modelar relacionamentos não-lineares
"""
def modelo_mlp():
    print("\n--- Multilayer Perceptron ---")
    #Criando e treinando o modelo
    clf = MLPClassifier(
        hidden_layer_sizes=(50,25),
        max_iter=500,
        random_state=42
    ) #modelo
    clf.fit(X_train_scaled, y_train) #treino

    #Previsão
    y_pred = clf.predict(X_test_scaled) #realizando previsão com dados escalonados de features
    
    #Criando variaveis para a acuracia e relatorio e imprimindo
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)
    print("Acuracia: ", acuracia)
    print("\nRelatorio de Classificação:\n", relatorio)
    
    #Salvar em arquivos
    with open('resultados_modelos/neural_network.txt','w') as f:
        f.write(f'Modelo: Rede Neural Multilayer Perceptron\n')
        f.write(f'Acuracia: {acuracia}\n\n')
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)

    return clf



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