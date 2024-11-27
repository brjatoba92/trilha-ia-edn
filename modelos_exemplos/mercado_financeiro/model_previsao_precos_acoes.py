import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#Baixando os dados da B3
def baixar_dados_acoes(ticker='PETR4.SA', periodo='5y'):
    dados = yf.download(ticker, period=periodo)

    dados['Return'] = dados['Close'].pct_change()
    dados['MA20'] = dados['Close'].rolling(window=20).mean()
    dados['MA50'] = dados['Close'].rolling(window=50).mean()
    dados['Volatility'] = dados['Return'].rolling(window=20).std()

    return dados.dropna()

#Preparar os dados para o modelo
def preparar_dados_modelos(dados):
    features = ['Open', 'High', 'Low', 'Volume', 'Return', 'MA20', 'MA50', 'Volatility']
    X = dados[features]
    y = dados['Close'].shift(-1) #Prever proximo fechamento

    return X.iloc[:-1], y.iloc[:-1]

def treinar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    return modelo, X_test_scaled, y_test, scaler

def visualizar_resultados(modelo, X_test, y_test, scaler):
    previsoes = modelo.predict(X_test)

    plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label='Valores Reais', color='blue')
    plt.plot(previsoes, label='Previsoes', color='red')
    plt.title('Previsões de Preços das Ações - Random Forest')
    plt.xlabel('Amostras')
    plt.ylabel('Preço')
    plt.legend()
    plt.savefig('model_previsao_precos_acoes.png')
    plt.show()

def main():
    dados = baixar_dados_acoes()
    X, y = preparar_dados_modelos(dados)
    modelo, X_test, y_test, scaler = treinar_modelo(X, y)
    visualizar_resultados(modelo, X_test, y_test, scaler)

if __name__ == "__main__":
    main()