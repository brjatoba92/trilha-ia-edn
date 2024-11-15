import math

raio = float(input("Digite o raio do circulo, em metros: "))
area_do_circulo = math.pi * (raio ** 2)

print(f'A area do circulo com raio igual a {raio} m é igual a: {area_do_circulo:.2f} m') # .2f 2 casas decimais