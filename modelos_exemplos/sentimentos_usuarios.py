import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import re
import os

#Definir diretorio atual como caminho para o download
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
#Baixar stepwords para portugues no diretorio especifico
nltk.download('stopwords', download_dir=nltk_data_path)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.stop_words = set(stopwords.words('portuguese'))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)

    def train(self, texts, labels):
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        self.classifier.fit(X, labels)

    def predict(self, text):
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(vectorized_text)
        probabilities = self.classifier.predict_proba(vectorized_text)
        
        sentimento_map = {
            0: 'Negativo', 
            1: 'Neutro', 
            2: 'Positivo'
        }
        
        return {
            'sentimento': sentimento_map[prediction[0]],
            'confianca': round(max(probabilities[0]) * 100, 2)
        }

# Exemplo de uso
def main():
    textos = [
        'Estou muito feliz hoje!', 
        'Esse dia está terrível', 
        'Nada de especial aconteceu', 
        'Que dia incrível!',
        'Estou frustrado com tudo'
    ]
    
    labels = [2, 0, 1, 2, 0]  # 0: Negativo, 1: Neutro, 2: Positivo

    modelo = SentimentAnalyzer()
    modelo.train(textos, labels)

    teste = 'Estou muito feliz hoje!'
    resultado = modelo.predict(teste)
    
    print(f"Texto: {teste}")
    print(f"Sentimento: {resultado['sentimento']}")
    print(f"Confiança: {resultado['confianca']}%")

if __name__ == "__main__":
    main()