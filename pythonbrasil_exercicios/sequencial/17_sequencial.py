"""
Faça um Programa para uma loja de tintas. 
O programa deverá pedir o tamanho em metros quadrados da área a ser pintada. 
Considere que a cobertura da tinta é de 1 litro para cada 6 metros quadrados
e que a tinta é vendida em latas de 18 litros, que custam R$ 80,00 ou em galões de 3,6 litros, 
que custam R$ 25,00.
Informe ao usuário as quantidades de tinta a serem compradas e os respectivos preços em 3 situações:
comprar apenas latas de 18 litros;
comprar apenas galões de 3,6 litros;
misturar latas e galões, de forma que o desperdício de tinta seja menor. Acrescente 10% de folga e sempre arredonde os valores para cima, isto é, considere latas cheias.
"""

"""
The math.floor() function in Python returns the largest integer less than or equal to a given number. This is useful for rounding down floating-point numbers to the nearest whole number
Example

import math

print(math.floor(0.6)) # Output: 0
print(math.floor(1.4)) # Output: 1
print(math.floor(5.3)) # Output: 5
print(math.floor(-5.3)) # Output: -6
print(math.floor(22.6)) # Output: 22
print(math.floor(10.0)) # Output: 10
"""

"""
The math.ceil() function in Python is used to round a number up to the nearest integer. This function is part of the math module and returns the smallest integer greater than or equal to the given number

Example
import math

# Round a number upward to its nearest integer
print(math.ceil(1.4)) # Output: 2
print(math.ceil(5.3)) # Output: 6
print(math.ceil(-5.3)) # Output: -5
print(math.ceil(22.6)) # Output: 23
print(math.ceil(10.0)) # Output: 10
"""

import math
area_pintada = float(input("Informe a area a ser pintada, em metros"))

area_com_folga = area_pintada * 1.1

litros_consumidos = area_com_folga/6

quantidade_latas = math.ceil(litros_consumidos /18)
quantidade_galoes = math.ceil(litros_consumidos / 3.6)

#misturando latas e galões
latas_mistura = math.floor(litros_consumidos /18)
restante = litros_consumidos - (latas_mistura*18)
galoes_mistura = math.ceil(restante / 3.6)

#calculando os custos
custo_latas = quantidade_latas * 80
custo_galoes = quantidade_galoes * 25
custo_mistura = (latas_mistura*80) + (galoes_mistura*3.6)

print(f'Compra de somente {quantidade_latas} latas tem o custo de R$ {custo_latas:.2f} e somente {quantidade_galoes} galões tem o custo de R$ {custo_galoes:.2f}')
print(f'O custo misturando latas e galões foi de R$ {custo_mistura:.2f}')
