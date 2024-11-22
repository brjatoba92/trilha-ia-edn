"""
CIDADE A (SETE LAGOAS) e CIDADE B (Oswaldo Cruz)
INDICE DE CRESCIMENTO (61 e 62)
"""

def calculo_anos(PA, PB, G1, G2):
    """
    Função que calcula quantos anos a população de A ultrapasse B
    se ULTRAPASSA 100 ANOS, MAIS DE UM SECULO
    """
    anos = 0

    while PA <= PB:
        #Incremento dos anos de forma proporcional, multiplicando a porcentagem de G(Indice)
        PA += int(PA * (G1/100))
        PB += int(PB * (G2/100))
        anos += 1

        if anos > 100:
            return 'Mais de um 1 seculo' #dentro do if
        return f'{anos} anos'
    
def main():
    try:
        T = int(input()) #numeros de teste
        #[100 400 500 100]
        for i in range(T):
            PA, PB, G1, G2 = map(float, input().split())
            PA, PB = int(PA), int(PB)
            resultado = calculo_anos(PA, PB, G1, G2)
            print(resultado)
    except ValueError as ve:
        print(f'Erro: {ve}')

if __name__ == "__main__":
    main()
        