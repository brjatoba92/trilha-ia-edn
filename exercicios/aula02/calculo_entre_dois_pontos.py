# Calculo distancia entre dois pontos
# distancia = raiz de (x2-x1)**2 + (y2-y1)**2

import math

x1, y1 = map(float, input("Digite a coordenada x1 e y1: ").split())
x2, y2 = map(float, input("Digite a coordenada x2 e y2: ").split())

distancia = math.sqrt((x2 - x1)**2 + (y2-y1)**2)

print(f'{distancia:.4f}')
