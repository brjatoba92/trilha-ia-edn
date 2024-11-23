"""
Segmentação de Jogadores com Base em Estatísticas
Enunciado: 
- Agrupar jogadores em diferentes categorias com base em suas estatísticas de jogos, 
como posse de bola, chutes a gol e passes certos.
"""

# Bibliotecas
import pandas as pd  # Manipulação de dados
import numpy as np  # Arrays
import matplotlib.pyplot as plt  # Geração do gráfico 
import seaborn as sns  # Visualização de dados
from sklearn.cluster import KMeans  # Clusterização KMeans
from sklearn.decomposition import PCA  # Redução de dimensionalidade
from sklearn.metrics import silhouette_score  # Métrica de avaliação

# Dados simulados com 6 variáveis
data = pd.DataFrame({
    'posse_bola': np.random.randint(30, 70, 100),
    'chutes_a_gol': np.random.randint(1, 20, 100),
    'passes_certos': np.random.randint(300, 700, 100),
    'distancia_percorrida': np.random.randint(5000, 12000, 100),
    'cartoes_recebidos': np.random.randint(0, 5, 100),
    'intervencoes_defensivas': np.random.randint(0, 10, 100)
})

# Salvar dados em CSV
data.to_csv('segmentacao_jogadores_pca.csv', index=False)

# Número de clusters a ser testado
num_clusters = [2, 3, 4, 5]

# Variável para armazenar o melhor número de clusters
best_n_clusters = 0
best_silhouette_score = -1

# Loop para encontrar o melhor número de clusters
for n_clusters in num_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    data['cluster'] = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data[['posse_bola', 'chutes_a_gol', 'passes_certos', 'distancia_percorrida', 'cartoes_recebidos', 'intervencoes_defensivas']], data['cluster'])
    print(f"Para {n_clusters} clusters, o Silhouette Score é {silhouette_avg}")
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_n_clusters = n_clusters

# Melhor modelo baseado no Silhouette Score
kmeans = KMeans(n_clusters=best_n_clusters)
data['cluster'] = kmeans.fit_predict(data)

# Aplicação do PCA para visualização
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[['posse_bola', 'chutes_a_gol', 'passes_certos', 'distancia_percorrida', 'cartoes_recebidos', 'intervencoes_defensivas']])

# Plotagem dos clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(x=principal_components[:, 0], y=principal_components[:, 1], c=data['cluster'], cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title(f'Cluster de Jogadores (n_clusters={best_n_clusters})')
plt.legend(*scatter.legend_elements(), title='Cluster')
plt.colorbar(scatter)  # Associando a barra de cores ao gráfico de dispersão
plt.savefig('segmentacao_jogadores_pca.png') 
plt.show()

# Salvar os resultados finais em um CSV
data.to_csv('segmentacao_jogadores_pca_resultados.csv', index=False)

print(f"Melhor número de clusters: {best_n_clusters} com Silhouette Score de {best_silhouette_score}")
