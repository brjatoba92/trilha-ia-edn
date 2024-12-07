#Importação de bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix, r2_score

#Geração de dados simulados realistas
class CarajasHomeCenterDataGenerator:
    def __init__(self, num_samples=5000):
        np.random.seed(42)
        self.num_samples = num_samples
    def generate_data(self):
        data = pd.DataFrame({
            'loja_id': np.random.randint(1,50,self.num_samples),
            'regiao': np.random.choice(['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste'], self.num_samples),
            'tamanho_loja': np.random.choice(['Pequena', 'Media', 'Grande'], self.num_samples),
            'dia_semana': np.random.randint(1,8, self.num_samples),
            'mes': np.random.randint(1,13, self.num_samples),
            'ano': np.random.randint(2018,2024, self.num_samples),
            'temperatura': np.random.normal(25,5,self.num_samples),
            'precipitação': np.random.normal(50,20, self.num_samples),
            'dia_pagamento': np.random.randint(1,31, self.num_samples),
            #Variaveis target - alvo
            'vendas_total': np.random.normal(50000,15000, self.num_samples),
            'probabilidade_inadimplencia': np.ramdom.normal(0.1, 0.05, self.num_samples),
            'categoria_produto_mais_vendido': np.random.choice([
                'Materiais basicos', 'Ferramentas', 'Acabamento', 'Hidraulica', 'Eletrica', 'Construção', 'Eletronicos', 'Eletroportateis'
            ], self.num_samples)
        })
        #Adicionando correlação sintetica
        for i in range(self.num_samples):
            if data['regiao'][i] == 'Nordeste':
                data['vendas_totais'][i] *= 0.8 #Menor numero de vendas
            if data['tamanho_loja'][i] == 'Grande':
                data['vendas_totais'][i] *= 1.5 #Maior volume para lojas grandes
        return data

class CarajasHomeCenterMLProject:
    """
    Geração de Dados Sintéticos
    Criei um gerador de dados realista simulando uma rede de lojas
    Variáveis incluem: 
    - ID da loja, 
    - região
    - tamanho da loja
    - características temporais
    - dados climáticos
    Targets criados:
    - Vendas totais (regressão)
    - Probabilidade de inadimplência (classificação binária)
    - Categoria de produto mais vendido
    """
    
    def __init__(self):
        #Pasta para resultados
        self.output_dir = f'resultado_ml_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.output_dir, exist_ok=True)
        #Gerar dados
        data_generator = CarajasHomeCenterDataGenerator()
        self.df = data_generator.generate_data()
        self.df.to_csv(f'{self.output_dir}/dados_originais.csv', index=False)
    
    def preparar_dados(self):
        #Separação de features e targets
        X = self.df.drop(['vendas_total', 'probabilidade_inadimplencia', 'categoria_produto_mais_vendido'], axis=1) #todos menos as colunas deste array
        y_regressao = self.df['vendas_total']
        y_classificacao_inadimplencia = (self.df['probabilidade_inadimplencia']>0.15).astype(int)
        y_categoria_produto_mais_vendido = self.df['categoria_produto_mais_vendido']

        #Separando em dados categoricos e numericos
        colunas_categoricas = ['regiao', 'tamanho_loja']
        colunas_numericas = [
            'loja_id', 'mes', 'ano', 
            'temperatura', 'precipitação', 'dia_pagamento'
        ]

        #Preprocessamento
        preprocessador = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), colunas_numericas)
                ('cat', OneHotEncoder(), colunas_categoricas)
            ]
        )
        return (
            preprocessador,
            X,
            y_regressao,
            y_classificacao_inadimplencia,
            y_categoria_produto_mais_vendido
        )
    
    #Modelos desenvolvidos
    def modelo_regressao_vendas(self, preprocessador, X, y):
        """
        Modelos de Regressão:
        Random Forest Regressor
        Support Vector Regression (SVR)
        Linear Regression
        """
        #Modelos
        modelos_regressao = {
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'SVR': SVR(kernel='rbf'),
            'Linear Regression': LinearRegression()
        }
        resultados_regressao = {}

        for nome, modelo in modelos_regressao.items():
            
            #Pipeline
            pipeline = Pipeline([
                ('preprocessador', preprocessador)
                ('regressor', modelo)
            ])
            
            #Divisão dos dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            #Treinamento
            pipeline.fit(X_train, y_train)
            
            #Predicoes
            y_pred = pipeline.predict(X_test)
            
            #Metricas
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            #Resultados
            resultados_regressao[nome] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }

        #Salvar os dados
        with open (f'{self.output_dir}/resultados_regressao.txt', 'w') as f:
            for modelo, metricas in resultados_regressao.items():
                f.write(f"Modelo: {modelo}\n")
                f.write(f"MSE: {metricas['MSE']}\n")
                f.write(f"MAE: {metricas['MAE']}\n")
                f.write(f"R2: {metricas['R2']}\n")
        return resultados_regressao
    
    def modelo_classificacao_inadimplencia(self, preprocessador, X, y):
        """
        Modelos de Classificação:   
        Logistic Regression
        Gradient Boosting Classifier
        """
        #Modelo de classificação binaria
        modelos_classificacao_binaria = {
            'Logistic Regression': LogisticRegression(),
            'Gradiente Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        resultados_classificacao = {}

        for nome, modelo in modelos_classificacao_binaria.items():
            #Pipeline
            pipeline = Pipeline([
                ('preprocessador', preprocessador)
                ('classificador', modelo)
            ])
            
            #Divisão dos dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            #Treinamento
            pipeline.fit(X_train, y_train)
            
            #Predições
            y_pred = pipeline.predict(X_test)
            
            #Metricas
            accuracy = accuracy_score(y_test, y_pred)
            rel_classificacao = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            #Resultados
            resultados_classificacao[nome] = {
                'Accuracy': accuracy,
                'Classification Report': rel_classificacao,
                'Confusion Matrix': conf_matrix
            }
        with open (f'{self.output_dir}/ resultados_classificacao.txt', 'w') as f:
            for modelo, metricas in  resultados_classificacao.items():
                f.write(f"Modelo: {modelo}\n")
                f.write(f"Accuracy: {metricas['Accuracy']}\n")
                f.write(f"Classification Report: {metricas['Classification Report']}\n")
                f.write(metricas['Classification Report'], "\n\n")
        return resultados_classificacao
