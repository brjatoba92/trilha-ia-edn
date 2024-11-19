#criar validação para que não seja sequencial

def sao_sequenciais(a, b, c, d):
    lista = [a, b, c, d]
    lista_ordenada = sorted(lista)
    for i in range(len(lista_ordenada) - 1):
        if lista_ordenada[i] + 1 != lista_ordenada[i + 1]:
            return False
    return True

a, b, c, d = map(int, input("Informe quatro números separados por espaço: ").split())

if b > c and d > a and (c + d) > (a + b) and a % 2 == 0 and not sao_sequenciais(a, b, c, d):
    print("Valores aceitos")
else:
    print("Valores não aceitos")
