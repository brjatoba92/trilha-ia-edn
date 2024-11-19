"""
1 - R$ 4
2 - R$ 4.5
3 - R$ 5
4 - R$ 2
5 - R$ 1.5
"""

codigo, quantidade = map(int, input("Digite o codigo do produto e a entrada (separada por espaço)").split())

precos = {
    1: 4.00,
    2: 4.50,
    3: 5,
    4: 2,
    5: 1.5
}

if codigo in precos:
    total = precos[codigo] * quantidade
    print(f'Total: R$ {total:.2f}')
else:
    print("Codigo invalido")