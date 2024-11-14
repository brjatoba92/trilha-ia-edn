# Faça um programa para uma loja de tintas. 
# O programa deverá pedir o tamanho em metros quadrados da área a ser pintada. 
# Considere que a cobertura da tinta é de 1 litro para cada 3 metros quadrados !!!!
# e que a tinta é vendida em latas de 18 litros, que custam R$ 80,00. 
# Informe ao usuário a quantidades de latas de tinta a serem compradas e o preço total.

import math
area_pintada = float(input("Informe a area a ser pintada, em metros quadrados: "))

litros_consumidos = area_pintada / 3

quantidade_latas18litros = round((litros_consumidos / 18))

valor_gasto = quantidade_latas18litros * 80

print(f'Foram compradas {quantidade_latas18litros} latas de 18 litros a um custo de R$ {valor_gasto}')