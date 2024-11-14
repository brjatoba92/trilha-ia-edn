"""
Este codigo faz a condicional de um numero inteiro 

Tres possibilidades:

1. positivo
2. negativo
3. zero

"""

numero = int(input("Informe um numero inteiro: "))

# Condicional
if numero > 0:
    print("O numero informado é positivo")
elif numero < 0:
    print("O numero informado é negativo")
else:
    print("O numero informado é nulo")