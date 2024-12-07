import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
import datetime
import os

# Criação de dados sintéticos realistas para uma rede de materiais de construção
class CarajasBaseDeDados:
    def __init__(self, num_samples=5000):
        np.random.seed(42)
        self.num_samples = num_samples
        
    def generate_data(self):
        # Colunas sintéticas representando diferentes aspectos do negócio
        data = {
            'loja_id': np.random.randint(1, 50, self.num_samples),
            'regiao': np.random.choice(['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'], self.num_samples),
            'tamanho_loja': np.random.choice(['Pequena', 'Média', 'Grande'], self.num_samples),
            'dia_semana': np.random.randint(1, 8, self.num_samples),
            'mes': np.random.randint(1, 13, self.num_samples),
            'ano': np.random.randint(2018, 2024, self.num_samples),
            'temperatura': np.random.normal(25, 5, self.num_samples),
            'precipitacao': np.random.normal(50, 20, self.num_samples),
            'dia_pagamento': np.random.randint(1, 31, self.num_samples),
            
            # Variáveis target
            'vendas_total': np.random.normal(50000, 15000, self.num_samples),
            'probabilidade_inadimplencia': np.random.normal(0.1, 0.05, self.num_samples),
            'categoria_produto_mais_vendido': np.random.choice([
                'Materiais Básicos', 'Ferramentas', 'Acabamento', 
                'Hidráulica', 'Elétrica', 'Construção'
            ], self.num_samples)
        }
        
        # Adiciona alguma correlação sintética
        for i in range(self.num_samples):
            if data['regiao'][i] == 'Nordeste':
                data['vendas_total'][i] *= 0.8  # Menor volume de vendas
            if data['tamanho_loja'][i] == 'Grande':
                data['vendas_total'][i] *= 1.5  # Maior volume para lojas grandes
        
        return pd.DataFrame(data)

# Classe para gerenciar o projeto de ML
class CarajasMLProject:
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
        # Pasta para resultados
        self.output_dir = f'resultados_ml_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Gerar dados
        data_generator = CarajasBaseDeDados()
        self.df = data_generator.generate_data()
        self.df.to_csv(f'{self.output_dir}/dados_originais.csv', index=False)
        
    def preparar_dados(self):
        # Separação de features e targets
        X = self.df.drop(['vendas_total', 'probabilidade_inadimplencia', 'categoria_produto_mais_vendido'], axis=1)
        y_regressao = self.df['vendas_total']
        y_classificacao_inadimplencia = (self.df['probabilidade_inadimplencia'] > 0.15).astype(int)
        y_classificacao_categoria = self.df['categoria_produto_mais_vendido']
        
        # Separação de dados categóricos e numéricos
        colunas_categoricas = ['regiao', 'tamanho_loja']
        colunas_numericas = [
            'loja_id', 'dia_semana', 'mes', 'ano', 
            'temperatura', 'precipitacao', 'dia_pagamento'
        ]
        
        # Preprocessamento
        preprocessador = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), colunas_numericas),
                ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)
            ])
        
        return (
            preprocessador, 
            X, 
            y_regressao, 
            y_classificacao_inadimplencia, 
            y_classificacao_categoria
        )
    
    def modelo_regressao_vendas(self, preprocessador, X, y):
        """
        Modelos de Regressão:
        - Random Forest Regressor
        - Support Vector Regression (SVR)
        - Linear Regression
        """
        modelos_regressao = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Linear Regression': LinearRegression()
        }
        
        resultados_regressao = {}
        
        for nome, modelo in modelos_regressao.items():
            # Pipeline
            pipeline = Pipeline([
                ('preprocessador', preprocessador),
                ('regressor', modelo)
            ])
            
            # Divisão dos dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Treinamento
            pipeline.fit(X_train, y_train)
            
            # Predições
            y_pred = pipeline.predict(X_test)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Resultados
            resultados_regressao[nome] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }
        
        # Salvar resultados
        with open(f'{self.output_dir}/resultados_regressao.txt', 'w') as f:
            for modelo, metricas in resultados_regressao.items():
                f.write(f"Modelo: {modelo}\n")
                f.write(f"MSE: {metricas['MSE']}\n")
                f.write(f"MAE: {metricas['MAE']}\n")
                f.write(f"R2: {metricas['R2']}\n\n")
        
        return resultados_regressao
    
    def modelo_classificacao_inadimplencia(self, preprocessador, X, y):
        """
        Modelos de Classificação:
        - Logistic Regression
        - Gradient Boosting Classifier
        """
        #Modelos
        modelos_classificacao_binaria = {
            'Logistic Regression': LogisticRegression(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        resultados_classificacao = {}
        
        for nome, modelo in modelos_classificacao_binaria.items():
            # Pipeline
            pipeline = Pipeline([
                ('preprocessador', preprocessador),
                ('classificador', modelo)
            ])
            
            # Divisão dos dados - conjuntos de treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Treinamento
            pipeline.fit(X_train, y_train)
            
            # Predições
            y_pred = pipeline.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            rel_classificacao = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Resultados
            resultados_classificacao[nome] = {
                'Accuracy': accuracy,
                'Classification Report': rel_classificacao,
                'Confusion Matrix': conf_matrix
            }
        
        # Salvar resultados
        with open(f'{self.output_dir}/resultados_classificacao_inadimplencia.txt', 'w') as f:
            for modelo, metricas in resultados_classificacao.items():
                f.write(f"Modelo: {modelo}\n")
                f.write(f"Accuracy: {metricas['Accuracy']}\n")
                f.write("Classification Report:\n")
                f.write(metricas['Classification Report'] + "\n\n")
        
        return resultados_classificacao
    
    def visualizar_resultados(self, resultados_regressao, resultados_classificacao):
        # Gráfico de barras para métricas de regressão
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        modelos_regressao = list(resultados_regressao.keys())
        r2_scores = [res['R2'] for res in resultados_regressao.values()]
        plt.bar(modelos_regressao, r2_scores)
        plt.title('Comparação R2 - Modelos de Regressão')
        plt.ylabel('R2 Score')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        modelos_classificacao = list(resultados_classificacao.keys())
        accuracies = [res['Accuracy'] for res in resultados_classificacao.values()]
        plt.bar(modelos_classificacao, accuracies)
        plt.title('Comparação Accuracy - Modelos de Classificação')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparacao_modelos.png')
        plt.close()
    
    def executar_projeto(self):
        # Preparar dados
        preprocessador, X, y_regressao, y_classificacao_inadimplencia, y_classificacao_categoria = self.preparar_dados()
        
        # Executar modelos
        resultados_regressao = self.modelo_regressao_vendas(preprocessador, X, y_regressao)
        resultados_classificacao = self.modelo_classificacao_inadimplencia(preprocessador, X, y_classificacao_inadimplencia)
        
        # Visualizar resultados
        self.visualizar_resultados(resultados_regressao, resultados_classificacao)

# Executar projeto
if __name__ == "__main__":
    projeto = CarajasMLProject()
    projeto.executar_projeto()