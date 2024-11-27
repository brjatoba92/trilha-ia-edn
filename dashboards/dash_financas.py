import pandas as pd

# Carregar os dados
df = pd.read_csv('financeiro.csv')

# Converter a coluna 'Data' para datetime
df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')

# Processar os dados
df['Mês'] = df['Data'].dt.to_period('M')
df_agg = df.groupby(['Mês', 'Tipo'])['Valor'].sum().reset_index()
df['Saldo Acumulado'] = df['Valor'].cumsum()

# Salvar os dados processados em arquivos CSV
df.to_csv('financeiro_completo.csv', index=False)
df_agg.to_csv('resumo_mensal.csv', index=False)
df[['Data', 'Saldo Acumulado']].to_csv('saldo_acumulado.csv', index=False)

print("Dados salvos em 'financeiro_completo.csv', 'resumo_mensal.csv' e 'saldo_acumulado.csv'")
