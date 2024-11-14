"""
Faça um Programa que pergunte quanto você ganha por hora e o número de horas trabalhadas no mês. 
Calcule e mostre o total do seu salário no referido mês, sabendo-se que são descontados 

a. 11% para o Imposto de Renda, 
b. 8% para o INSS e 
c. 5% para o sindicato, 

faça um programa que nos dê:
1. salário bruto.
2. quanto pagou ao INSS.
3. quanto pagou ao sindicato.
4. o salário líquido.

calcule os descontos e o salário líquido, conforme a tabela abaixo:
+ Salário Bruto : R$
- IR (11%) : R$
- INSS (8%) : R$
- Sindicato ( 5%) : R$
= Salário Liquido : R$

Obs.: Salário Bruto - Descontos = Salário Líquido.
"""
salario_hora = float(input("Quanto você ganha por hora? "))
horas_no_mes = int(input("Horas trabalhadas no mes: "))

salario_bruto = salario_hora * horas_no_mes

ir = salario_bruto * 0.11
inss = salario_bruto * 0.08
sindicato = salario_bruto * 0.05

soma_deducoes = ir + inss + sindicato

salario_liquido = salario_bruto - soma_deducoes

print(f'Salario Bruto ==> + R$ {salario_bruto} // Deduções ==> IR (11%): - R$ {ir:.2f}, INSS: - R$ {inss:.2f}, Sindicato: - R$ {sindicato:.2f} // Salario Liquido ==>  + R$ {salario_liquido:.2f}')