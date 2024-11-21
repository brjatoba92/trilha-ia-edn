"""
Classificação de Sentimento em Notícias Financeiras
Descrição: Classificar notícias financeiras em positivas ou negativas usando Naive Bayes
"""

#Importando as bibliotecas
import pandas as pd  #manipulação e análise de dados.
from sklearn.feature_extraction.text import CountVectorizer #transformar o texto em uma matriz de contagem de tokens.
from sklearn.model_selection import train_test_split # dividir os dados em conjuntos de treinamento e teste.
from sklearn.naive_bayes import MultinomialNB #criar o modelo Naive Bayes
from sklearn.metrics import accuracy_score #calcular a precisão das previsões do modelo.

## PARTE 1
# Dados de exemplo
dados = {
    'id': [1, 2, 3, 4, 5, 6, 7],
    'text': [
        'Stocks rallied today as the market rebounded.',
        'The company reported a significant loss.',
        'Economic outlook shows signs of recovery.',
        'Investor confidence is at an all-time low.',
        'New technologies are driving growth.',
        'Concerns over trade wars impact markets.',
        'Strong quarterly earnings boost stocks.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive']
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Salvar DataFrame como CSV
df.to_csv('financial_news.csv', index=False)

# PARTE 2
#carregando os dados
dados = pd.read_csv('financial_news.csv') # #Carrega os dados do arquivo CSV chamado financial_news.csv para um DataFrame do pandas chamado dados

#vetorização de texto
vectorizer = CountVectorizer() #Cria uma instância do CountVectorizer, que será usado para converter o texto em uma matriz de contagem de tokens.
x = vectorizer.fit_transform(dados['text']) #Ajusta e transforma a coluna de texto text em uma matriz esparsa de contagem de tokens e a armazena na variável x

#alvo
y = dados['sentiment'] #Seleciona a coluna sentiment como o alvo (variável dependente) e a armazena na variável y

#dividir dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size=0.2, random_state=42) #Divide os dados em conjuntos de treinamento e teste. test_size=0.2 indica que 20% dos dados serão usados para teste e 80% para treinamento. random_state=42 garante que a divisão dos dados seja reprodutível.

#criar e treinar o modelo
modelo = MultinomialNB() #Cria uma instância do modelo Naive Bayes multinomial
modelo.fit(x_train, y_train) #Treina o modelo Naive Bayes usando os dados de treinamento (x_train e y_train).

#fazer previsões
previsoes = modelo.predict(x_test) #Treina o modelo Naive Bayes usando os dados de treinamento (x_train e y_train).

#avaliação do modelo
accuracy = accuracy_score(y_test, previsoes) #Calcula a precisão do modelo comparando os valores reais (y_test) com as previsões (previsoes).
print(f'Accuracy: {accuracy}') #Imprime o valor da precisão
