# Programa para calcular a faixa etaria de uma pessoa
idade = int(input("Digite a sua idade: "))

if idade >= 18:
    print("Você é maior de idade")
elif 12<=idade<18:
    print("Você é adolescente")
else:
    print("Você é uma criança")