"""
Detecção de Anomalias em Transações com Isolation Forest
Descrição: Detectar transações anômalas usando Isolation Forest
"""
#Importando bibliotecas necessarias
from sklearn.ensemble import IsolationForest #criar e treinar o modelo de detecção de anomalias.
import pandas as pd #manipulação e análise de dados.

# Carregar dados
dados = pd.read_csv('transactions.csv')

# Converter a coluna 'transaction_type' para valores numéricos usando dummies
dados = pd.get_dummies(dados, columns=['transaction_type']) #Converte a coluna categórica transaction_type em colunas dummy binárias, criando colunas como transaction_type_credit e transaction_type_debit

# Selecionar características
"""
Seleciona as colunas amount, account_age, transaction_type_credit e transaction_type_debit 
como características (variáveis independentes) e as armazena na variável x
"""
x = dados[['amount', 'account_age', 'transaction_type_credit', 'transaction_type_debit']]

# Criar e treinar o modelo
"""
Cria uma instância do modelo Isolation Forest. 
O parâmetro contamination=0.01 especifica a proporção esperada de anomalias nos dados (1% neste caso). 
random_state=42 garante que a randomização seja reprodutível.
"""
model = IsolationForest(contamination=0.01, random_state=42)
"""
Treina o modelo Isolation Forest usando os dados de entrada x 
e armazena os rótulos de anomalias na coluna anomaly do DataFrame dados.
O valor 1 indica uma anomalia e -1 indica uma transação normal.
"""
dados['anomaly'] = model.fit_predict(x)

# Exibir transações anômalas
anomalies = dados[dados['anomaly'] == -1] #Filtra o DataFrame dados para incluir apenas as transações que foram classificadas como anômalas (anomaly == -1) e armazena o resultado na variável anomalies.
print(anomalies) #Exibe as transações anômalas na saída

