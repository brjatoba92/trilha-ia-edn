"""
4 numeros
B > C
D > A

C + D > A + B
C e D  ==> %2 == 0 POSITIVOS

A % 2 == 0

Valores aceitos

caso contrario
Valores não aceitos
"""

a, b, c, d = map(int, input("Informe um numero: ").split()) #valores inteiros, entrada de um valor, delmitador(por padrão é o espaço)

if b > c and d > a and (c+d) > (a+b) and a % 2 == 0: #colocando todas as condições do problema
    print("Valores aceitos")
else:
    print("Valores não aceitos")

