"""
Faça um Programa que peça 2 números inteiros e um número real. Calcule e mostre:
o produto do dobro do primeiro com metade do segundo .
a soma do triplo do primeiro com o terceiro.
o terceiro elevado ao cubo.
"""

numero_inteiro1 = int(input("Informe um numero inteiro: "))
numero_inteiro2 = int(input("Informe um segundo numero inteiro: "))
numero_real = float(input("Informe um numero real: "))

a = (2*numero_inteiro1) * (numero_inteiro2/2)
b = 3*numero_inteiro1 + numero_real
c = numero_real ** 3

print(f'Os resultados são: {a}, {b} e {c}')