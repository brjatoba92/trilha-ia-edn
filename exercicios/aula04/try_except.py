try:
    numero1 = int(input('Digite um numero: '))
    numero2 = int(input('Digite outro numero: '))
    resultado = numero1/numero2
    print(f'Resultado: {resultado:.2f}')
except ZeroDivisionError:
    print("Não pode ser dividido por zero")
except ValueError:
    print('Operação não permitida')