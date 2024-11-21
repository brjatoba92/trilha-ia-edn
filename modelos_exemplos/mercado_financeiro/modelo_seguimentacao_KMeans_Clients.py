"""
Segmentação de Clientes com K-Means

Descrição: Segmentar clientes de um banco com base em suas transações usando K-Means
"""

#Importando as bibliotecas necessarias:
import pandas as pd #manipulação e análise de dados.
from sklearn.cluster import KMeans #criar e treinar o modelo de clustering K-Means
import matplotlib.pyplot as plt #mporta a biblioteca Matplotlib, usada para criar visualizações de dados.

## PARTE 1
# Dados de exemplo
dados = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7],
    'annual_income': [35000, 40000, 50000, 60000, 45000, 80000, 55000],
    'spending_score': [40, 70, 30, 90, 60, 20, 50]
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Salvar DataFrame como CSV
df.to_csv('customer_transactions.csv', index=False)

##PARTE 2

#Carregando os dados
dados = pd.read_csv('customer_transactions.csv')

#Selecionar as caracteristicas para o clustering
x = dados[['annual_income', 'spending_score']] #Seleciona as colunas annual_income e spending_score como características (variáveis independentes) e as armazena na variável x.

#Criar e treinar o modelo
kmeans = KMeans(n_clusters=3) #Cria uma instância do modelo K-Means, especificando que queremos dividir os dados em 3 clusters (n_clusters=3)
kmeans.fit(x) #Treina o modelo K-Means usando os dados de entrada x

#Adicionar os rotulos ao DataFrame
dados['cluster'] = kmeans.labels_ #Adiciona uma nova coluna ao DataFrame dados chamada cluster, que contém os rótulos dos clusters atribuídos a cada cliente pelo modelo K-Means.

#Visualizando os clusters
plt.figure(figsize=(10, 6)) #Cria uma nova figura com tamanho 10x6 polegadas para a visualização.
plt.scatter(dados['annual_income'], dados['spending_score'], c=dados['cluster']) #Cria um gráfico de dispersão onde o eixo x representa a annual_income e o eixo y representa o spending_score. Os pontos são coloridos de acordo com o cluster a que pertencem (c=dados['cluster']).
plt.xlabel('Annual Income') #Define o rótulo do eixo x
plt.ylabel('Spending Score') #Define o rótulo do eixo y
plt.title('Customer Segments') #Define o titulo

# Salvar a figura
plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight') #Salva a figura criada no arquivo customer_segments.png com uma resolução de 300 DPI e ajusta a caixa delimitadora para incluir todos os elementos.

plt.show() # Exibe a figura na tela.
