#Importando bibliotecas
import numpy as np #array
import pandas as pd #manipulação de dados
import matplotlib.pyplot as plt #geração dos graficos

from sklearn.model_selection import train_test_split #conjunto de dados de treino e de teste
from sklearn.preprocessing import StandardScaler #normalização dos dados
from sklearn.metrics import mean_squared_error, r2_score, classification_report #Erro Medio Quadratico, Coeficiente de Determinação, Relatorio das Metricas

from sklearn.linear_model import LinearRegression,LogisticRegression #Modelo de Regressão Logistica
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

class MeteorologicalModels:

    def __init__(self, data_path):
        """
        Inicializa os modelos de Machine Learning para previsões meteorológicas
        
        :param data_path: Caminho para o arquivo de dados meteorológicos
        """
        self.data = pd.read_csv(data_path)
        self._analisar_dados_faltantes()
    
    def _analisar_dados_faltantes(self):
        """
        Analise detalhada de dados faltantes
        """
        #Porcentagem de dados faltantes por coluna
        print("Analise detalhada de dados faltantes por coluna")
        missing_percentual = self.data.isnull().mean()*100
        missing_percentual = missing_percentual[missing_percentual > 0]

        if not missing_percentual.empty:
            print("\nColunas de Dados Faltantes:")
            for coluna, percentual in missing_percentual.items():
                print(f'{coluna}: {percentual:.2f}')

            #Visualização dos dados faltantes
            plt.figure(figsize=(10,6))
            missing_percentual.plot(kind='bar')
            plt.title('Percentual de Dados Faltantes por Coluna')
            plt.xlabel('Colunas')
            plt.ylabel('Percentual')
            plt.show()
        else:
            print("Nenhum dado faltante encontrado")


    def _criar_pipeline_tratamento(self, features, estrategia_imputer='mean'):
        """
        CONTINUAR AQUI
        """
        pass

        
    def temperatura_previsao_regressao(self):
        """
        Modelo de Regressão Linear para Previsão de Temperatura
        """
        # Preparação dos dados
        self.X = self.data[['umidade', 'pressao_atmosferica', 'velocidade_vento']]
        self.y = self.data['temperatura']
        
        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Padronização dos dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos de Regressão
        modelos = {
            'Regressão Linear': LinearRegression(),
            'Regressão SVR': SVR(kernel='rbf'),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100)
        }
        
        resultados = {}
        
        for nome, modelo in modelos.items():
            # Treinamento
            modelo.fit(X_train_scaled, y_train)
            
            # Previsão
            y_pred = modelo.predict(X_test_scaled)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            resultados[nome] = {
                'MSE': mse,
                'R2': r2
            }
        
        # Visualização dos resultados
        plt.figure(figsize=(10, 6))
        plt.title('Comparação de Modelos de Regressão para Temperatura')
        plt.bar(resultados.keys(), [r['R2'] for r in resultados.values()])
        plt.ylabel('Coeficiente R²')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return resultados

    def classificacao_clima_extremo(self):
        """
        Modelos de Classificaçãp para Identificar Condições Cimaticas Extremas
        """
        self.X = self.data[['temperatura','umidade', 'pressao_atmosferica', 'velocidade_vento']]
        self.y = self.data['clima_extremo']

        #Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        #Padronização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Modelos de Classificacao

        modelos = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest Classififier': RandomForestClassifier(n_estimators=100)
        }

        resultados = {}

        for nome, modelo in modelos.items():
            #Treinamento
            modelo.fit(X_train_scaled, y_train)

            #Previsao
            y_pred = modelo.predict(X_test_scaled)

            #metricas de Classificação
            relatorio = classification_report(y_test, y_pred, output_dict=True)

            resultados[nome] = relatorio

        return resultados
        
    def previsao_neural_network(self):
        """
        Modelo de Rede Neural para Prevsião Avançada
        """

        #Preparação dos dados
        features = ['temperatura', 'umidade', 'pressao_atmosferica', 'velocidade_vento']
        self.X = self.data[features]
        self.y = self.data['precipitacao'] #Alvo

        #Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        #Padronização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Rede Naural Multi-Layer Perceptron
        mlp = MLPRegressor(
            hidden_layer_sizes=(50,25), #duas camadas ocultas
            max_iter=500,
            activation='relu',
            solver='adam'
        )

        #Treinamento
        mlp.fit(X_train_scaled, y_train)

        #Previsão
        y_pred = mlp.predict(X_test_scaled)

        #Metricas
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        #Visualização
        plt.figure(figsize=(10,6))
        plt.scatter(y_test, y_pred, alpha=.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',lw=2)
        plt.title('Previsão de Precipitação - Rede Neural')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.tight_layout()
        plt.show()

        return {
            'MSE': mse,
            'R2': r2
        }


def main():
    #Caminho do arquivo
    meteorologia = MeteorologicalModels('dados_meteorologicos.csv')
    print("1. Previsão de Temperatura")
    resultado_temp = meteorologia.temperatura_previsao_regressao()
    print(resultado_temp)
    print("2. Classificação de Clima Extremo")
    resultado_clima = meteorologia.classificacao_clima_extremo()
    print(resultado_clima)
    print("3. Previsão de Precipitação com Redes Neurais")
    resultado_temp = meteorologia.previsao_neural_network()
    print(resultado_temp)

if __name__ == "__main__":
    main()