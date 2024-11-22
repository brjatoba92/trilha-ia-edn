#Sequencia

def sequencia_logica(n):
    """
    1a função - imprime a sequencia para o numero inteiro
    """
    for i in range(1, n+1):
        print(i, i**2, i**3)
        print(i, i**2 + 1, i**3 + 1)

def main():
    try:
        entrada = input('Digite a quantidade de testes: ')

        if not entrada.isdigit():
            raise ValueError("Insira o numero positivo")
        
        N = int(entrada)

        if N < 1 or N>=1000:
            raise ValueError("O numero deve estar entre 1 e 1000")
        sequencia_logica(N)
    except ValueError as ve:
        print(f'Error: {ve}')

if __name__ == "__main__":
    main()