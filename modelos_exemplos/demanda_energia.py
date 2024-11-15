"""
REGRESSÃO MULTIPLA

Utilize um conjunto de dados com múltiplos recursos (como temperatura, umidade, e dia da semana) 
para prever a demanda de energia elétrica. 
Crie um modelo de regressão múltipla que use todos esses recursos.
"""
print("-----PREVISÃO DE ENERGIA PARA NOVOS DADOS---------")
from sklearn.linear_model import LinearRegression
import numpy as np

#arrays de dados (temperatura, umidade, dia da semana, consumo de energia)
dados = np.array([[30, 70, 1, 500],
                  [28, 65, 2, 450], 
                  [25, 60, 3, 400], 
                  [27, 68, 4, 420], 
                  [26, 64, 5, 410], 
                  [29, 75, 6, 480], 
                  [31, 80, 7, 510]
])

x = dados[:, :-1] #temperatura, umidade e dia da semana - pega todas as linhas e exclue a ultima coluna
y = dados[:, -1] #consumo de energia - pega todas as linhas e inclui a ultima coluna

# criação e treinamento do modelo
modelo_energia = LinearRegression() #criando
modelo_energia.fit(x, y) #treinando

#input do usuario para inserir novos dados
dados = [] #array vazio
print("Informe a temperatura, umidade e dia da semana: ")
for i in range(3):
    dado = float(input("Digite: "))
    dados.append(dado) #acrescenta a cada interação do laço for um novo dado no array dados 

#previsão
novos_dados = np.array([dados[0], dados[1], dados[2]]).reshape(1, -1) #deixa o array no formato de coluna
previsao_consumo = modelo_energia.predict(novos_dados) #realiza a previsão

print(f'Demanda energia prevista: {previsao_consumo[0]:.2f} kwh') #resultado
