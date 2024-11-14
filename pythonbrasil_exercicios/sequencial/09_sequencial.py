"""
Faça um Programa que peça a temperatura em graus Fahrenheit, transforme e mostre a temperatura em graus Celsius.
C = 5 * ((F-32) / 9).
"""

temperatura_farenheit = float(input("Informe a temperatura em farenheit: "))

temperatura_celsius = 5 * ((temperatura_farenheit-32)/9)

print(f'A conversão de {temperatura_farenheit} °F para Celsius é igual a {temperatura_celsius:.2f} °C')