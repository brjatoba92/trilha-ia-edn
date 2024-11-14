# Faça um Programa que pergunte quanto você ganha por hora e o número de horas trabalhadas no mês. Calcule e mostre o total do seu salário no referido mês.

salario_hora = float(input("Informe seu salario hora: "))
qnt_horas_mes = float(input("Quantidade de horas trabalhadas o mes: "))

salario_mensal = salario_hora * qnt_horas_mes

print(f'Seu salario é igual a R$ {salario_mensal}')