# Importando bibliotecas
import numpy as np  # array
import pandas as pd  # manipulação de dados
import matplotlib.pyplot as plt  # geração dos gráficos

from sklearn.model_selection import train_test_split  # conjunto de dados de treino e de teste
from sklearn.preprocessing import StandardScaler  # normalização dos dados

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, r2_score  # Erro Médio Quadrático, Coeficiente de Determinação

from sklearn.linear_model import LinearRegression  # Modelos de Regressão
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

class MeteorologicalModels:

    def __init__(self, data_path):
        """
        Inicializa os modelos de Machine Learning para previsões meteorológicas

        :param data_path: Caminho para o arquivo de dados meteorológicos
        """
        # Carregando os dados com verificação de missing values
        self.data = pd.read_csv(data_path)
        #sUBSTITUE NA POR QUALQUER OUTRO VALOR
        self.data.fillna(0,inplace=True)
        #remover as linhas com NA
        #self.data.dropna(inplace=True) 
        self._analisar_dados_faltantes()

    def _analisar_dados_faltantes(self):
        """
        Analise detalhada de dados faltantes
        """
        # Porcentagem de dados faltantes por coluna
        print("Análise detalhada de dados faltantes: ")
        missing_percentual = self.data.isnull().mean() * 100
        missing_percentual = missing_percentual[missing_percentual > 0]

        if not missing_percentual.empty:
            print("\nColunas de Dados Faltantes:")
            for coluna, percentual in missing_percentual.items():
                print(f'{coluna}: {percentual:.2f}%')

            # Visualização dos dados faltantes
            #plt.figure(figsize=(12, 10))
            missing_percentual.plot(kind='bar')
            plt.title('Percentual de Dados Faltantes por Coluna')
            plt.xlabel('Colunas')
            plt.ylabel('Percentual')
            plt.savefig('dados_faltantes.png')
            #plt.show()
        else:
            print("Nenhum dado faltante encontrado")

    def _criar_pipeline_tratamento(self, features, estrategia_imputer='mean'):
        """
        Criando pipeline de pré-processamento com tratamento de dados faltantes

        :param features: Lista de features numéricas
        :param estrategia_imputer: Estratégia de imputação (mean, median, most_frequent, knn)
        :return: pipeline de preprocessamento
        """
        # Escolha da estratégia de imputação
        if estrategia_imputer == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif estrategia_imputer == 'median':
            imputer = SimpleImputer(strategy='median')
        elif estrategia_imputer == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
        elif estrategia_imputer == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError("Estratégia inválida")

        # Criação de pipeline de pré-processamento
        preprocessor = ColumnTransformer(
            transformers=[
                ('imputer', imputer, features),
                ('scaler', StandardScaler(), features)
            ]
        )

        return preprocessor

    def temperatura_previsao_regressao(self, estrategia_imputer='mean'):
        """
        Modelo de Regressão Linear para Previsão de Temperatura
        com tratamento de dados faltantes
        """
        # Preparação dos dados
        features = ['umid', 'p_atm', 'vel_vento']
        target = 'temp'

        # Verificação de dados faltantes no target
        if self.data[target].isnull().sum() > 0:
            print(f'Atenção! Existem dados faltantes na variável target {target}')
            # Remoção das linhas com target faltante
            self.data = self.data.dropna(subset=[target])

        X = self.data[features]
        y = self.data[target]

        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pipeline de preprocessamento
        preprocessador = self._criar_pipeline_tratamento(features, estrategia_imputer)

        # Modelos de Regressão com Pipeline
        modelos = {
            'Regressao Linear': Pipeline([
                ('preprocessador', preprocessador),
                ('regressor', LinearRegression())
            ]),
            'Regressao SVR': Pipeline([
                ('preprocessador', preprocessador),
                ('regressor', SVR(kernel='rbf'))
            ]),
            'Random Forest Regressor': Pipeline([
                ('preprocessador', preprocessador),
                ('regressor', RandomForestRegressor(n_estimators=100))
            ])
        }

        resultados = {}

        for nome, modelo in modelos.items():
            # Treinamento
            modelo.fit(X_train, y_train)

            # Previsão
            y_pred = modelo.predict(X_test)

            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            resultados[nome] = {
                'MSE': mse,
                'R2': r2
            }

        # Visualização dos resultados
        #plt.figure(figsize=(10, 6))
        plt.title(f'Comparação de Modelos de Regressão (Imputação {estrategia_imputer})')
        plt.bar(resultados.keys(), [r['R2'] for r in resultados.values()])
        plt.ylabel('Coeficiente R²')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'resultados_{estrategia_imputer}.png')
        #plt.show()

        return resultados

    def demonstrar_tecnicas_imputacao(self):
        """
        Demonstra diferentes técnicas de imputação de dados
        """

        # Técnicas de imputação
        tecnicas = [
            'mean',
            'median',
            'knn'
        ]

        resultados_imputacao = {}

        for tecnica in tecnicas:
            print(f'\nTestando técnica de imputação: {tecnica}')
            resultado = self.temperatura_previsao_regressao(estrategia_imputer=tecnica)
            resultados_imputacao[tecnica] = resultado
        return resultados_imputacao


def main():
    # Caminho do arquivo
    meteorologia = MeteorologicalModels('fortaleza.csv')
    # Demonstrar técnicas de imputação
    resultados_imputacao = meteorologia.demonstrar_tecnicas_imputacao()
    # Imprimir comparação das técnicas de imputação
    print("\nComparação das Técnicas de Imputação:")
    for tecnica, resultado in resultados_imputacao.items():
        print(f'\n{tecnica.upper()}:')
        for modelo, metricas in resultado.items():
            print(f'{modelo}: R2 = {metricas["R2"]:.4f}, MSE = {metricas["MSE"]:.4f}')


if __name__ == "__main__":
    main()
