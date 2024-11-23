import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Dados simulados
data = pd.DataFrame({
    'comentarios': np.random.choice([
        'Excelente jogo, muito emocionante',
        'Péssima atuação do goleiro',
        'O árbitro foi injusto',
        'Jogo incrível', 
        'o time está de parabéns',
        'Horrível, não jogaram nada',
        'Muito bom, grande vitória',
        'Partida entediante, podia ser melhor',
        'Ótima estratégia, jogaram muito bem',
        'Decepcionante, esperava mais',
        'Fantástico, que jogo sensacional'], 100),
    'sentimentos': np.random.choice([
        'Positivo',
        'Negativo'], 100) 
})

# Salvar dados em CSV
data.to_csv('comentarios_futebol.csv', index=False)

# Pré-processamento
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['comentarios'])
y = data['sentimentos']

# Transformação TF-IDF
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Acurácia e Relatório de Classificação
print(f'Acurácia: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negativo', 'positivo'], yticklabels=['negativo', 'positivo'])
plt.xlabel('Valor Previsto')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão')
plt.savefig('comentarios_futebol.png')
plt.show()
