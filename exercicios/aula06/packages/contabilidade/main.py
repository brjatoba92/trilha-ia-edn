from packages.escola_matematica import operacoes
from packages.contabilidade import juros
import math as m
from datetime import datetime

hora_atual = datetime.now()
hora_ontem = datetime.weekday()

import random

print(operacoes.multiplicar(2,3))
print(juros.calcular_juros(1000, 0.02, 3))
print(m.sqrt(16))
print(f'Hora atual: {hora_atual}')
print(random.randint(10,100))
