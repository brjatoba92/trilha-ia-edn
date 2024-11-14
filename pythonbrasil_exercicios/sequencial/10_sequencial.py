# Faça um Programa que peça a temperatura em graus Celsius, transforme e mostre em graus Fahrenheit.

celsius = float(input('Informe a tempetaura em Celsius neste exato momento: '))

farenheit = celsius * (9/5) + 32

print(f'A conversão de {celsius}°C para Farenheit é igual a: {farenheit}°F')