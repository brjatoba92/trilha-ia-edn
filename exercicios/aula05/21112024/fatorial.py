# Criar fatorial de N (N é a multiplicação de N pelos seus antecessores)

def calcula_fatorial(n):
    """
    Esta função calcula o fatorial de um numero inteiro N
    """
    fatorial = 1
    # Cria as regras do numero fatorial
    for i in range(2, n+1):
        fatorial *= i
    return fatorial

# Cria a função principal
def main():
    try:
        N = int(input('Digite um número entre 1 e 12: '))
        if N <= 0 or N >= 13:
            raise ValueError("O numero precisa ser entre 1 e 12")
        resultado = calcula_fatorial(N)
        print(f'Fatorial de {N} é {resultado}')
    except ValueError as error:
        print(f'Erro: {error}')

if __name__ == "__main__":
    main()
