# Faça um Programa que peça o raio de um círculo, calcule e mostre sua área.

import math

raio = float(input("Informe o raio em metros: "))

area = math.pi * (raio ** 2)

print(f'Um circulo com raio igual a {raio} m é igual a {area:.2f} m')