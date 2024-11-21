"""
3. Previsão de Readmissão Hospitalar
Enunciado do Exercício:
Objetivo: Desenvolver um modelo de machine learning para prever se um paciente será readmitido no hospital dentro de 30 dias após a alta.

Passos:

Carregue os dados de readmissão hospitalar.

Prepare os dados, realizando a normalização e transformação de categorias.

Divida os dados em conjuntos de treinamento e teste.

Crie e treine um modelo de Gradiente Boosting usando sklearn.

Avalie a precisão do modelo e gere um relatório de classificação.
"""
#Importando as bibliotecas necessarias
import numpy as np #arrays
import pandas as pd #carrega os dados
from sklearn.model_selection import train_test_split #separa os dados em conjuntos de treinamento e de teste
from sklearn.preprocessing import StandardScaler, OneHotEncoder #normalização dos dados numericos e codificação dos dados categoricos
from sklearn.compose import ColumnTransformer #transformações especificas a colunas numericas e categoricas
from sklearn.pipeline import Pipeline #encapsulamento dos passos de preprocessamento e modelo em um pipeline
from sklearn.ensemble import GradientBoostingClassifier #modelo de classificação com base em Gradienet Boosting
from sklearn.metrics import accuracy_score, classification_report #avaliação e de desempenho do modelo

# Carregar os dados
data = pd.read_csv('diabetic_data.csv')

# Seleção de recursos e pré-processamento
X = data.drop(['readmitted', 'encounter_id', 'patient_nbr'], axis=1) #X: Contém todos os atributos, exceto readmitted, encounter_id e patient_nbr
y = data['readmitted'].apply(lambda x: 1 if x == '<30' else 0) #y: Coluna alvo que indica se o paciente foi readmitido em menos de 30 dias (<30).

# Transformação de dados categóricos
categorical_features = X.select_dtypes(include=['object']).columns #identificação das colunas categoricas 
numeric_features = X.select_dtypes(exclude=['object']).columns #identificação das colunas numericas

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features), #normaliza os dados numericos
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) #Codifica os dados categóricos em valores binários, ignorando valores desconhecidos.
    ]) 

# Criação do pipeline de pré-processamento e modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
]) #Cria um pipeline que primeiro aplica o pré-processamento e depois treina o modelo de Gradient Boosting.

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Divide os dados em conjuntos de treino (80%) e teste (20%).

# Treinamento do modelo
pipeline.fit(X_train, y_train) #Treina o pipeline (pré-processamento e modelo) nos dados de treino.

# Previsão
y_pred = pipeline.predict(X_test) #Usa o modelo treinado para fazer previsões no conjunto de teste

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred) #Calcula a precisão do modelo.
report = classification_report(y_test, y_pred) #Gera um relatório detalhado das métricas de classificação, incluindo precisão, revocação e F1-Score.

# Resultado
print(f'Acurácia: {accuracy:.2f}')
print("Relatório de Classificação:\n", report)