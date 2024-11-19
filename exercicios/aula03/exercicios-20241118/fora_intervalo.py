"""
Qualquer valor entre [0,25] [25,50], [50,75] ou [75,100]]
Preciso incluir os numeros do intervalo
"""

numero = round(float(input("Informe o numero com no maximo duas casas decimais: ")), 2)

if 0 <= numero <= 25:
    print("Numero entre 0 e 25")
elif 25 < numero <= 50:
    print("Numero entre 25 e 50")
elif 50 < numero <= 75:
    print("Numero entre 50 e 75")
else:
    print("Numero entre 75 e 100")
