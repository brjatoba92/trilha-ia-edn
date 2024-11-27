import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#Download dos dados
def baixar_dados_volatilidade(ticker='^VIX', periodo='5y'):
    dados = yf.download(ticker, period=periodo)

    #Features
    dados['Return'] = dados['Close'].pct_change()
    dados['MA10'] = dados['Close'].rolling(window=10).mean()
    dados['MA30'] = dados['Close'].rolling(window=30).mean()

    return dados.dropna()

def preparar_dados_svr(dados):
    features = ['Open', 'High', 'Low', 'Volume', 'Return', 'MA10', 'MA30']
    X = dados[features]
    y = dados['Close']

    return X, y

def treinar_svr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    modelo = SVR(kernel='rbf', C=100, epsilon=0.1)
    modelo.fit(X_train_scaled, y_train)
    
    return modelo, X_test_scaled, y_test, scaler

#Visualizar resultados da volatilidade
def visualizar_volatilidade(modelo, X_test, y_test, scaler):
    previsoes = modelo.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, previsoes, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Previsão de Volatilidade - Support Vector Regression')
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.savefig('svr.png')
    plt.show()

# Executar todo o processo
def main():
    dados = baixar_dados_volatilidade()
    X, y = preparar_dados_svr(dados)
    modelo, X_test, y_test, scaler = treinar_svr(X, y)
    visualizar_volatilidade(modelo, X_test, y_test, scaler)

if __name__ == "__main__":
    main()