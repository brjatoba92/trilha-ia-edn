"""
Segmentação de Jogadores com Base em Estatísticas
Enunciado: Agrupar jogadores em diferentes categorias com base em suas estatísticas de jogos, como posse de bola, chutes a gol e passes certos.
"""

import pandas as pd #manipulação dos dados
import numpy as np #array
from sklearn.cluster import KMeans #clusterização
import matplotlib.pyplot as plt #criação de graficos

#Dados simulados
data = pd.DataFrame({
    'posse_bola': np.random.randint(30,70,100), #100 valores aleatorios entre 30% e 70%
    'chutes_a_gol': np.random.randint(1,20,100), #100 valores entre 1 e 20
    'passes_certos': np.random.randint(300,700,100) #100 valores entre 300 e 700
})

# Salvar dados em CSV
data.to_csv('segmentacao_jogadores.csv', index=False) #Salvando o dataframe em um csv sem um index

#Cluster
kmeans = KMeans(n_clusters=3) #Define o algoritmo KMeans para criar 3 clusters.
data['cluster'] = kmeans.fit_predict(data) #Ajusta o modelo KMeans aos dados e atribui um cluster a cada jogador. O resultado é adicionado ao DataFrame data na nova coluna cluster.

#Plotagem
plt.scatter(data['posse_bola'], data['chutes_a_gol'], c=data['cluster'], cmap='viridis') #Cria um gráfico de dispersão das estatísticas de posse_bola e chutes_a_gol, colorindo os pontos de acordo com o cluster ao qual pertencem.
plt.xlabel('Posse de Bola') #rotulo eixo x
plt.ylabel('Chutes a Gol') #rotulo eixo y
plt.title('Cluster de Jogadores') #titulo
plt.colorbar() #barra de cores 
plt.savefig('segmentacao_jogadores.png') #salvar em imagem png
plt.show()