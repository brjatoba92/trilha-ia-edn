"""
Utilize o modelo de regressão linear do sklearn para prever os preços das ações com base 
em dados históricos.

1. Descrição:
Dados: Utilize um conjunto de dados de preços históricos de uma ação.
Objetivo: Prever o preço da ação para uma data futura.

2. Passos:
Carregue os dados históricos dos preços das ações.
Divida os dados em conjuntos de treinamento e teste.
Utilize numpy para manipulação dos dados.
Crie e treine um modelo de regressão linear com sklearn.
Avalie a precisão do modelo e faça previsões.
"""
#Carregando as bibliotecas
import numpy as np #arrays
import pandas as pd #carregar e manipular de dados
from sklearn.model_selection import train_test_split #divisão dos dados em conjuntos de treino e teste
from sklearn.linear_model import LinearRegression #criação do modelo de regerssão linear
import matplotlib.pyplot as plt #criação e manipulação de dados

#carregando os dados
dados = pd.read_csv('Dados Históricos_Ibovespa_2004_2024.csv') #carregando o arquivo csv

#Preparação dos dados
x = np.array(dados.index).reshape(-1, 1) #conversão dos dados para um array e reformatado para um array bidimensional
y = dados['Close'].values #seleção da coluna dos valores de fechamento das ações

#separa os dados em conjuntos de treinamento (80%) e teste (20%) com esta divisão sendo replicavel
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

#Criação e treinamento do modelo
modelo = LinearRegression() #criação do modelo
modelo.fit(x_train, y_train) #treinamento com o conjunto de dados 

#Previsão
y_pred = modelo.predict(x_test) #Usa o modelo treinado para prever os valores das ações no conjunto de teste 

#Visualização dos resultados
plt.scatter(x_test, y_test, color='black') #Plota os dados reais na forma de grafico de dispersão
plt.plot(x_test, y_pred, color='blue', linewidth=3) #Plota a previsão do modelo (linha azul)

"""
índices dos dados do conjunto de teste. Essencialmente, cada ponto ao longo do eixo x corresponde a um índice no DataFrame original 
que foi utilizado para dividir os dados em treinamento e teste. 
"""
plt.xlabel('Índice') #eixo x com o parametro inidice

plt.ylabel('Preço de Fechamento') #eixo y
plt.title('Previsão de Preços de Ações') #Titulo
plt.show() #mostra a figura

# Salvar o gráfico em uma imagem
plt.savefig('previsao_acoes.png', dpi=300, bbox_inches='tight')

