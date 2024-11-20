import math
def area_circulo(pi, raio):
    area = pi*(raio**2)
    return area

resultado = area_circulo(math.pi, int(input('Informe o raio do circulo, em metros: ')))
print(f'Area do circulo: {resultado:.2f} m2')