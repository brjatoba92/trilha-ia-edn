import random
#def main():
valor = [random.randint(1,10000) for _ in range(100)]
maior = (max(valor))
posicao = valor.index(maior)+1
resultado = {
    "numeros": valor,
    "maior valor": maior,
    "posicao": posicao
}
print("\n Numeros gerados: ", valor)
print("\n Maior valor: ", maior)
print("\n Posição do maior valor: ", posicao)

#if __name__ == "main":
#    main()