#Importação das bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
import datetime

#Criação de diretorio
def criar_diretorio(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

def criando_dados_financeiros():
    np.random.seed(42) #gera os mesmos dados randomicos a cada execução
    n_samples = 1000 #quantidade de amostras

    #Criando um dicionario com os dados
    data = pd.DataFrame({
        'data': [datetime.date(2020,1,1) + datetime.timedelta(days=i) for i in range(n_samples)],
        'preco_abertura': np.random.normal(100,15,n_samples),
        'preco_fechamento': np.random.normal(100,15,n_samples),
        'valor_negociacao': np.random.normal(1000000,250000,n_samples),
        'volatilidade': np.abs(np.random.normal(0,0.02, n_samples)),
        'retorno': np.random.normal(0.001, 0.02, n_samples)
    })

