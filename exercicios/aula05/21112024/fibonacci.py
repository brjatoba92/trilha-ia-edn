"""
Gerar a serie de Fibonacci(lógioca) até o N-esimo termo
"""
def fibonacci(n):
    #Iniciar os primeiros valores da serie
    a, b = 0, 1
    resultado = [a]
    #Geração do termo (é a logica por tras da resolução do problema)
    for _ in range(1, n):
        resultado.append(b)
        a, b = b, a+b
    #Função devolve o resultado
    return resultado

def main():
    try:
        N = int(input('Informe a quantidade de numeros da sequencia: '))
        if N<=0 or N>=46:
            raise ValueError('O numero tem que ser entre 0 e 45')
        fibonacci_sequence = fibonacci(N) #pega o resultado sem tratar
        print(f'Sem tratar: {fibonacci_sequence}')
        print(" ".join(map(str, fibonacci_sequence))) #Pega o resultado tratado

    except ValueError as ve:
        print(f'Erro {ve}')

if __name__ == "__main__":
    main()