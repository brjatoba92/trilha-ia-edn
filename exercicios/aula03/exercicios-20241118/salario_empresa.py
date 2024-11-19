"""
0 - 400 = 15%
400.01 - 800  = 12%
800.01 - 1200 = 10%
1200.01 - 2000 = 7%
> 2000 = 4%
"""

salario = float(input("Digite o salario: "))
if salario <= 400:
    percentual = 15
elif salario <= 800:
    percentual = 12
elif salario <= 1200:
    percentual = 10
elif salario <= 2000:
    percentual = 7
else:
    percentual = 4

reajuste = salario * (percentual/100)
novo_salario = salario + reajuste

print(f'Novo salario: R$ {novo_salario:.2f}')
print(f'Reajuste: R$ {reajuste:.2f}')
print(f'Em percentual: {percentual} %')
