## Faça um Programa que peça as 4 notas bimestrais e mostre a média.

nota1 = float(input("Primeira nota: "))
nota2 = float(input("Segunda nota: "))
nota3 = float(input("Terceira nota: "))
nota4 = float(input("Quarta nota: "))

soma = nota1 + nota2 + nota3 + nota4

media = soma/4

print(f'A media do aluno foi:  {media}')