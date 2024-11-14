codigo_colaborador = int(input("Digite o codgo do coolaborador: "))
horas_trabalhadas = int(input("Qunatidade de horas trabalhadas: "))
valor_hora = float(input("Seu salario por hora: "))

salario = horas_trabalhadas * valor_hora

print(f'O salario do colaborador {codigo_colaborador} é de R$ {salario:.2f}')
